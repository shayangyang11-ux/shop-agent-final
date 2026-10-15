"""领域模型层（合同层）。

这一层定义"店铺装修"业务的静态结构，是整个系统的单一事实源（SSOT）：
LangGraph 的运行时状态（state.py）持有这里的模型；所有节点、渲染器、
测试用例只依赖这里的类型，不各自发明结构。

三个关键设计决策（对应 codemind 叙事）：

1. 树结构用 dict[id, node] 扁平存储，而非嵌套 children
   —— 让"按 id 归并"成为 O(1) 操作。这是 Send fan-out 归并 reducer
   （state.py 的 merge_tree）和占位回填（字典覆写）的前提。
   层级关系由 page.zone_id 表达，遍历走 pages_by_zone()。

2. 占位语义显式建模（NodeStatus.PLACEHOLDER）
   —— "页面存在但内容未生成"是一等公民状态，不是"content 为 None"
   的隐式约定。隐式约定意味着每个读代码的人都要记得特判，忘一处就出
   bug；显式状态机 + model_validator 在类型边界强制不变量。

3. 跳转关系（LinkSpec）单独建模
   —— "列表页跳转指向一个尚未生成的详情页"是占位回填机制的载体；
   dangling_links() 就是"回填是否完成"的验收判据。
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# 枚举：业务的受控词汇表
# ---------------------------------------------------------------------------


class PageType(str, Enum):
    """页面类型 —— 决定生成 prompt 与渲染模板。"""

    BANNER = "banner"  # 店铺首屏 / 门头
    LIST = "list"  # 商品列表页
    DETAIL = "detail"  # 商品详情页
    ACTIVITY = "activity"  # 活动 / 营销页
    ABOUT = "about"  # 品牌 / 关于我们
    CUSTOM = "custom"  # 自定义页


class NodeStatus(str, Enum):
    """节点状态机：placeholder -> draft -> ready。"""

    PLACEHOLDER = "placeholder"  # 仅结构占位，无内容（框架模式的产物）
    DRAFT = "draft"  # 已有生成内容，未定稿
    READY = "ready"  # 已定稿（用户确认）


class EntryChoice(str, Enum):
    """三入口 —— 对应 codemind 的三种建树方式。"""

    MANUAL = "manual"  # 手动创建：Agent 只响应指令，不主动生成
    FRAMEWORK = "framework"  # 生成框架：树结构 + 占位页面，不生成内容
    ONE_CLICK = "one_click"  # 一键补齐：结构 + 全部内容（Send 并行）


# ---------------------------------------------------------------------------
# 页面内容与跳转
# ---------------------------------------------------------------------------


class Component(BaseModel):
    """页面内组件。type 约定组件形态，props 存差异参数。

    组件类型膨胀后可演进为 Pydantic 判别联合（discriminated union），
    每种组件获得独立的强类型 props —— 面试可讲的演进路线。
    """

    type: str = Field(description="组件类型：hero / grid / carousel / text / form / gallery ...")
    props: dict[str, Any] = Field(default_factory=dict, description="组件参数，如商品数、文案")


class PageContent(BaseModel):
    """页面的生成内容 —— 相当于 codemind 中 PRD 对页面的角色。"""

    headline: str | None = Field(default=None, description="主标题")
    subheadline: str | None = Field(default=None, description="副标题")
    components: list[Component] = Field(default_factory=list)
    notes: str | None = Field(default=None, description="生成备注：依据的需求片段")


class LinkSpec(BaseModel):
    """页面间跳转。占位回填的核心载体。

    from_page 的某个交互触发后跳到 to_page；to_page 可以指向一个
    尚未生成、以 PLACEHOLDER 状态存在的页面 id —— 回填完成后，
    这条跳转就不再是 dangling 状态。
    """

    trigger: str = Field(description="触发交互，如：点击商品卡片")
    to_page: str = Field(description="目标页面 id（可以是占位页）")
    label: str | None = Field(default=None, description="链接文案")


# ---------------------------------------------------------------------------
# 树节点
# ---------------------------------------------------------------------------


class ZoneNode(BaseModel):
    """分区层节点 —— 对应 codemind 的"模块层"。"""

    id: str
    name: str
    brief: str | None = Field(default=None, description="本分区的装修需求摘要")


class PageNode(BaseModel):
    """页面层节点 —— 对应 codemind 的"页面层"。

    不变量（由 model_validator 在构造时强制）：
    - PLACEHOLDER 节点不允许携带内容
    - 非 PLACEHOLDER 节点必须有内容
    """

    id: str
    zone_id: str
    name: str
    page_type: PageType
    status: NodeStatus = NodeStatus.PLACEHOLDER
    content: PageContent | None = None
    links: list[LinkSpec] = Field(default_factory=list)

    @model_validator(mode="after")
    def _status_content_consistent(self) -> PageNode:
        if self.status == NodeStatus.PLACEHOLDER and self.content is not None:
            raise ValueError("占位节点不允许携带内容（status=placeholder 时 content 必须为 None）")
        if self.status != NodeStatus.PLACEHOLDER and self.content is None:
            raise ValueError("非占位节点必须有内容（status!=placeholder 时 content 不能为 None）")
        return self

    def is_placeholder(self) -> bool:
        return self.status == NodeStatus.PLACEHOLDER


# ---------------------------------------------------------------------------
# 店铺树：SSOT
# ---------------------------------------------------------------------------


class ShopTree(BaseModel):
    """店铺结构树：整个应用的单一事实源。

    zones / pages 是以 id 为键的扁平字典，而非嵌套 children：
    - 归并（merge_tree reducer）按 id 做 O(1) 定位
    - 回填（placeholder -> real）就是一次字典覆写
    - 层级关系由 page.zone_id 表达，遍历用 pages_by_zone()
    """

    shop_name: str = "我的店铺"
    zones: dict[str, ZoneNode] = Field(default_factory=dict)
    pages: dict[str, PageNode] = Field(default_factory=dict)

    # ---- 查询辅助：SSOT 自带查询能力，节点逻辑不直接摸字典细节 ----

    def pages_by_zone(self, zone_id: str) -> list[PageNode]:
        """某分区下的全部页面（按插入序）。"""
        return [p for p in self.pages.values() if p.zone_id == zone_id]

    def placeholder_page_ids(self) -> list[str]:
        """仍处于占位状态的页面 —— 框架模式的产出、待回填的目标。"""
        return [pid for pid, p in self.pages.items() if p.is_placeholder()]

    def dangling_links(self) -> list[tuple[str, LinkSpec]]:
        """悬空跳转：目标不存在、或目标仍是占位页的链接。

        一键模式的验收判据之一：回填完成后 dangling_links() == []。
        """
        result: list[tuple[str, LinkSpec]] = []
        for pid, page in self.pages.items():
            for link in page.links:
                target = self.pages.get(link.to_page)
                if target is None or target.is_placeholder():
                    result.append((pid, link))
        return result


# ---------------------------------------------------------------------------
# 节点间的"合同"模型：上下游解耦的数据载体
# ---------------------------------------------------------------------------


class InputAnalysis(BaseModel):
    """parse_input 节点的 LLM 输出合同（批次3 用到）。

    结构化输出的意义：LLM 的自由文本不可依赖，可校验的合同才可依赖。
    """

    needs_summary: str = Field(description="需求一句话摘要")
    product_categories: list[str] = Field(default_factory=list, description="识别到的商品品类")
    recommended_entry: EntryChoice = Field(description="推荐的三入口之一")
    reason: str = Field(description="推荐理由（会展示给用户）")
    missing_info: list[str] = Field(default_factory=list, description="信息缺口，供手动/框架模式提示")


class PageDraft(BaseModel):
    """分解节点 -> 生成节点的任务合同（批次4 的 Send fan-out 载体）。

    Send 出去的是数据不是引用：每个并行分支拿到一份 PageDraft
    （外加树快照），互不干扰，靠 state.py 的 reducer 归并。
    """

    page_id: str
    zone_id: str
    name: str
    page_type: PageType
    requirement_excerpt: str = Field(description="该页面对应的需求片段")
    placeholder: bool = Field(
        default=False,
        description="True = 不生成内容，仅在树中落一个占位节点（信息不足的页面）",
    )


# ---------------------------------------------------------------------------
# 整店规划合同（批次4：分解节点的 LLM 输出）
# ---------------------------------------------------------------------------


class PlannedZone(BaseModel):
    """规划阶段的分区：LLM 只需给出人类可读的名字。"""

    name: str = Field(description="分区名，如：首页、商品专区、活动区")


class PlannedPage(BaseModel):
    """规划阶段的页面。

    注意 links 里 LinkSpec.to_page 填的是目标页面的 **name**（LLM 的
    视角只有名字，让它编机器 ID 必然不可靠）。name -> page_id 的翻译
    由 nodes/planning.py 的 compile_plan 做确定性收口 —— LLM 负责
    意图，确定性代码负责机器标识，两者边界清晰。
    """

    name: str = Field(description="页面名，全店唯一，如：夏季新品列表")
    zone: str = Field(description="所属分区名（PlannedZone.name 之一）")
    page_type: PageType
    requirement_excerpt: str = Field(description="该页面对应的需求片段")
    placeholder: bool = Field(
        default=False,
        description="True = 信息不足，只落占位不生成内容（一键模式的韧性设计）"
    )
    links: list[LinkSpec] = Field(default_factory=list, description="出链，to_page 填目标页面名")


class TreePlan(BaseModel):
    """一键 / 框架模式共用的整店规划合同。

    同一份合同服务两种模式：framework 只消费 compile_plan 产出的
    占位树；one_click 在此之上继续对非占位页做 Send 并行回填。
    合同复用 = 两种模式的树完全同构，后续从框架升级到一键零迁移。
    """

    shop_name: str = Field(description="店铺名（商家输入中提取或概括）")
    zones: list[PlannedZone] = Field(default_factory=list)
    pages: list[PlannedPage] = Field(min_length=1, description="至少一个页面")
