"""评分维度与 judge 输出合同。

设计决策：LLM 只评"软"维度，硬指标代码算。

- 需求覆盖 / 文案质量 / 整店一致性 / 转化设计：涉及语义理解，交给
  LLM-as-judge，但每维都写死 1/3/5 分锚点（anchor），把"评分标准
  显式化"——没有锚点的 rubric 等于让 judge 自由发挥，人工一致率
  必然低。
- 占位数 / 悬空链接 / 页面数：确定性代码算（run_eval.deterministic_metrics），
  一个字符都不会错的东西交给 LLM 是浪费钱还引入噪声。

维度只有 4 个、分值只有 5 档：最小可行评测要控制标注成本——
人工标注 20 任务 × 4 维 = 80 个分数，半小时能标完；维度再多，
人工先疲劳弃疗。
"""

from pydantic import BaseModel, Field

# 维度键 -> 中文名（报告与 CSV 用，排序即展示序）
DIMENSIONS: dict[str, str] = {
    "coverage": "需求覆盖",
    "copy_quality": "文案质量",
    "consistency": "整店一致性",
    "conversion": "转化设计",
}

JUDGED_DIMENSIONS = list(DIMENSIONS)  # judge 打分的维度（结构指标不在内）


class JudgeVerdict(BaseModel):
    """单任务的 judge 裁决 —— structured_call 的校验合同。

    分值约束 ge=1 le=5 由 Pydantic 强制：judge 输出 0 分或 6 分会
    触发校验失败 -> 错误回填重试，而不是静默流进结果集。
    """

    coverage: int = Field(ge=1, le=5, description="需求覆盖：品类/风格/客群/特殊要求是否体现")
    copy_quality: int = Field(ge=1, le=5, description="文案质量：具体、有卖点、无空话")
    consistency: int = Field(ge=1, le=5, description="整店一致性：风格口吻统一、互不冲突")
    conversion: int = Field(ge=1, le=5, description="转化设计：CTA 与跳转明确、动线合理")
    rationale: str = Field(description="总体评价两三句：先优点后问题")
    weak_points: list[str] = Field(default_factory=list, description="最突出的问题清单（可为空）")
