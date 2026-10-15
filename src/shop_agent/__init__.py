"""shop-agent：AI 电商店铺装修 Agent。

LangGraph 编排范式项目，覆盖五个核心机制：
- interrupt：三入口人机协同（用户选择装修模式）
- Send：一键模式下按页面并行 fan-out
- 条件边：手动 / 框架 / 一键 三模式路由
- checkpointer：SQLite 持久化 + 崩溃恢复
- 占位回填：结构先行、内容后补的依赖回填机制
"""

__version__ = "0.1.0"
