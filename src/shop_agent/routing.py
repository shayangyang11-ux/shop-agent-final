"""路由层：条件边的纯函数集合。

为什么路由单独成文件：路由是业务规则（"什么状态去什么分支"），
不是编排细节（"把边挂到哪个节点上"）。规则集中在此，可以穷举
测试、可以单独评审；graph.py（批次5）只负责把规则装进图里。
"""

from __future__ import annotations

from shop_agent.state import ShopState

# 分支节点名（批次4 实现节点实体，名称在此统一定义避免魔法字符串）
BRANCH_MANUAL = "manual_mode"
BRANCH_FRAMEWORK = "framework_mode"
BRANCH_ONE_CLICK = "one_click_mode"


def route_after_parse(state: ShopState) -> str:
    """parse_input 之后：解析失败走异常终止，成功进入人机协同。

    注意 abort 是"图层面的终止"，但终止前 error 已写入 state —— 
    checkpointer 落盘后调用方仍能读到失败原因，不是静默吞掉。
    """
    return "abort" if state.get("error") else "choose_entry"


def route_after_choice(state: ShopState) -> str:
    """choose_entry 之后：按用户选择进三分支。

    未知值兜底到 manual —— 与 resolve_choice 的 fail-safe 哲学一致：
    路由层不信任上游，宁可不生成，不可乱生成。
    """
    return {
        "manual": BRANCH_MANUAL,
        "framework": BRANCH_FRAMEWORK,
        "one_click": BRANCH_ONE_CLICK,
    }.get(state.get("entry_choice") or "", BRANCH_MANUAL)
