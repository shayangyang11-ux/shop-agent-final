"""评测子包：LLM-as-judge 最小评测闭环（任务3）。

流水线：20 个开店需求 -> 走生产图（one_click 路径）生成整店 ->
确定性指标（占位/悬空链接，代码算）+ LLM 评审（四维 rubric）->
盲标 CSV 供人工打分 -> 一致率分析（Pearson / Spearman / QWK）。

用法：
    python -m shop_agent.eval run [--limit N] [--out DIR] [--dry-run]
    python -m shop_agent.eval agreement 标注.csv [--results results.json]
"""
