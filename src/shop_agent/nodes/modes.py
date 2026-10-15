"""三入口分支节点 + Send fan-out 分发函数。

图拓扑（one_click 路径，批次5 在 graph.py 全量组装）：

    choose_entry --route--> one_click_mode --fan_out_pages--> generate_page ×N
                                                              |
                                                        (普通边，barrier)
                                                              v
                                                          assemble

三个编排要点（对着图理解）：

1. fan_out_pages 是**条件边函数**，返回 list[Send | str]（1.2.x 源码
   _branch.py::_finish 明确支持混合列表：Send 直达目标节点，str 查
   路由表）。非空计划 -> N 个 Send；空计划 -> ["assemble"] 兜底，
   保证收尾节点总能执行。

2. Send 携带**私有 state**：generate_page 收到的不是全局累积 state，
   而是 Send 的 arg。这是"数据不是引用"的具体形态 —— 分支之间互不
   可见，全局 state 在它们运行期间的变化与它们无关，归并只走
   merge_tree reducer。payload 用原生 dict（model_dump）而非
   Pydantic 对象：Send arg 要过 checkpointer 序列化边界，原生
   数据类型最稳。

3. barrier 免费获得：generate_page -> assemble 是普通边。LangGraph
   的 superstep 语义保证：N 个并行分支全部完成后，assemble 才在
   下一个 superstep 执行一次。不需要手写任何"等待/计数"逻辑 ——
   汇聚是图拓扑的固有语义。
"""

from __future__ import annotations

from shop_agent.llm import LLMServiceError, LLMStructureError, structured_call
from shop_agent.nodes.planning import compile_plan
from shop_agent.prompts import build_plan_messages
from shop_agent.schema import TreePlan
from shop_agent.state import ShopState


# ---------------------------------------------------------------------------
# 手动模式：Agent 待命
# ---------------------------------------------------------------------------


def manual_mode(state: ShopState) -> dict:
    """手动创建：本次运行不生成任何内容。

    语义上这很重要：不是"失败"，是"用户选择了掌控权"。树不建立
    （空树没有意义），提示语引导用户下一步（批次5 CLI 的对话循环
    里，用户逐条下指令、Agent 逐条响应 —— 那是另一张子图的事）。
    """
    analysis = state.get("analysis")
    hints = ""
    if analysis and analysis.missing_info:
        hints = f"补充这些信息能加快装修：{'、'.join(analysis.missing_info)}。"
    return {"summary": f"已进入手动模式，Agent 待命。{hints}下一条消息开始逐条指令。"}


# ---------------------------------------------------------------------------
# 框架模式：结构先行，全部占位
# ---------------------------------------------------------------------------


def framework_mode(state: ShopState, client: object = None) -> dict:
    """生成框架：LLM 规划整店结构，全部页面落 PLACEHOLDER，不生成内容。

    TreePlan 合同与一键模式共用 —— 两模式的树完全同构，用户之后
    说"补齐"即可从框架无缝升级到一键（回填路径同一套）。
    """
    result = _plan_and_compile(state, client)
    if "error" in result:
        return result
    tree, drafts, dropped = result["_tree"], result["_drafts"], result["_dropped"]

    # 框架模式不需要 drafts（不生成内容），显式清空避免误触发下游
    return {
        "shop_tree": tree,
        "page_drafts": [],
        "summary": _structure_report("框架", tree, dropped),
    }


# ---------------------------------------------------------------------------
# 一键模式：结构 + 并行回填
# ---------------------------------------------------------------------------


def one_click_mode(state: ShopState, client: object = None) -> dict:
    """一键补齐：落全占位树 + 输出 PageDraft 列表（Send 的数据源）。

    注意本节点**不生成任何页面内容** —— 它只做规划与落树。内容生成
    是 fan_out_pages 发出的 N 个 generate_page 分支的事。节点职责
    单一，是并行正确性的前提。
    """
    result = _plan_and_compile(state, client)
    if "error" in result:
        return result
    tree, drafts, dropped = result["_tree"], result["_drafts"], result["_dropped"]

    return {
        "shop_tree": tree,  # 全占位树先进 state（分支结果在此基础上回填）
        "page_drafts": drafts,
        "summary": None,  # 一键模式的中途不写报告，收尾由 assemble 负责
    }


def fan_out_pages(state: ShopState) -> list:
    """条件边：把 PageDraft 列表展开为 N 个 Send（或空计划兜底）。

    为什么 payload 里要带树快照：generate_page 是并行分支，收到的
    是 Send 的私有 state 而非全局 state —— 它需要树才能构造"只含
    自己那页"的部分树（取该页的占位节点、zone 归属、出链），需要
    店名与其他页面名来保证文案的整店一致性。每个分支各带一份快照，
    互不共享 —— 这是"数据不是引用"的成本面（内存 xN），树很小，
    可接受；换来的是分支间零竞争。
    """
    from langgraph.types import Send  # 延迟导入：框架边界

    drafts = state.get("page_drafts") or []
    if not drafts:
        return ["assemble"]  # 空计划：无 Send，直接字符串路由到收尾

    tree = state.get("shop_tree")
    tree_payload = tree.model_dump() if tree else {}

    return [
        Send("generate_page", {"draft": draft.model_dump(), "tree": tree_payload})
        for draft in drafts
    ]


# ---------------------------------------------------------------------------
# 共用：规划 + 编译
# ---------------------------------------------------------------------------


def _plan_and_compile(state: ShopState, client: object) -> dict:
    """调 LLM 出 TreePlan 并编译。失败时返回 {"error": ...}（异常是数据）。"""
    user_input = state.get("user_input") or ""
    analysis = state.get("analysis")

    messages = build_plan_messages(
        user_input,
        needs_summary=analysis.needs_summary if analysis else user_input[:50],
        categories=analysis.product_categories if analysis else [],
    )

    try:
        plan = structured_call(messages, TreePlan, client=client)
    except LLMServiceError as err:
        return {"error": f"{err}（规划请求未完成，稍后重试即可，已选模式不受影响）"}
    except LLMStructureError as err:
        return {"error": f"整店规划失败：{err}。可重试或改用手动模式。"}

    tree, drafts, dropped = compile_plan(plan)
    return {"_tree": tree, "_drafts": drafts, "_dropped": dropped}


def _structure_report(mode: str, tree, dropped: int) -> str:
    """框架/失败路径共用的结构报告。"""
    parts = [
        f"{mode}模式完成：{len(tree.zones)} 个分区、{len(tree.pages)} 个页面",
        f"{len(tree.placeholder_page_ids())} 个占位页待补内容",
    ]
    if dropped:
        parts.append(f"忽略了 {dropped} 条指向不存在页面的跳转（模型幻觉，已防御）")
    return "；".join(parts) + "。"
