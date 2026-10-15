"""端到端预览：完整跑通三入口闭环（批次4 版）。

覆盖路径：
  输入 -> parse_input(LLM) -> interrupt 暂停 -> 用户选择
    -> manual:   待命提示 -> END
    -> framework: LLM 规划 -> 全占位树 + 报告 -> END
    -> one_click: LLM 规划 -> 全占位树 -> Send 并行生成 ×N
                  -> merge_tree 归并 -> assemble 验收 -> END

运行（conda 环境，已 pip install -e .）：
    python -m shop_agent.preview

它是"最小可运行图"的活文档：确认 interrupt / Send / 归并 / 收尾
闭环行为，跑它就够。正式 CLI（多轮对话 + 崩溃恢复）是批次5。
"""

from __future__ import annotations

from uuid import uuid4

from shop_agent.config import get_settings
from shop_agent.nodes import (
    assemble,
    choose_entry,
    fan_out_pages,
    framework_mode,
    generate_page,
    manual_mode,
    one_click_mode,
    parse_input,
)
from shop_agent.routing import (
    BRANCH_FRAMEWORK,
    BRANCH_MANUAL,
    BRANCH_ONE_CLICK,
    route_after_choice,
    route_after_parse,
)
from shop_agent.state import ShopState

# 终端菜单序号 -> EntryChoice 值。这是"展示层协议"：Agent 侧合同是
# 字符串值，序号只在终端里存在，进图之前就翻译掉。
_MENU_ALIASES = {"1": "manual", "2": "framework", "3": "one_click"}


def build_preview_graph():
    """三入口完整图（与批次5 正式图同构，仅 checkpointer 不同）。"""
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import END, START, StateGraph

    builder = StateGraph(ShopState)
    builder.add_node("parse_input", parse_input)
    builder.add_node("choose_entry", choose_entry)
    builder.add_node(BRANCH_MANUAL, manual_mode)
    builder.add_node(BRANCH_FRAMEWORK, framework_mode)
    builder.add_node(BRANCH_ONE_CLICK, one_click_mode)
    builder.add_node("generate_page", generate_page)  # Send 的并行目标
    builder.add_node("assemble", assemble)

    builder.add_edge(START, "parse_input")
    builder.add_conditional_edges(
        "parse_input",
        route_after_parse,
        {"choose_entry": "choose_entry", "abort": END},
    )
    builder.add_conditional_edges(
        "choose_entry",
        route_after_choice,
        {name: name for name in (BRANCH_MANUAL, BRANCH_FRAMEWORK, BRANCH_ONE_CLICK)},
    )

    # 手动 / 框架：单节点即终
    builder.add_edge(BRANCH_MANUAL, END)
    builder.add_edge(BRANCH_FRAMEWORK, END)

    # 一键：fan-out（条件边返回 Send 列表或 ["assemble"] 兜底）
    builder.add_conditional_edges(BRANCH_ONE_CLICK, fan_out_pages)
    # barrier：N 个并行分支全部归并后，assemble 执行一次
    builder.add_edge("generate_page", "assemble")
    builder.add_edge("assemble", END)

    # 预览用内存检查点即可：本批次验证 interrupt + Send 闭环，
    # 不验证跨进程恢复（SqliteSaver 是批次5 的主题）
    return builder.compile(checkpointer=InMemorySaver())


def _render_final(final: dict) -> None:
    """收尾渲染：错误优先，其次 summary，最后树概览。"""
    if final.get("error"):
        print(f"\n[异常路径] {final['error']}")
        return

    if final.get("summary"):
        print(f"\n{final['summary']}")

    tree = final.get("shop_tree")
    if tree and tree.pages:
        print("\n店铺结构：")
        for zone_id, zone in tree.zones.items():
            print(f"  [{zone.name}]")
            for page in tree.pages_by_zone(zone_id):
                mark = "✓" if not page.is_placeholder() else "○"
                print(f"    {mark} {page.name}（{page.page_type.value}）")


def main() -> None:
    settings = get_settings()
    if not settings.is_configured:
        print("✗ 未配置 API Key：请复制 .env.example 为 .env 并填写 SHOP_AGENT_LLM_API_KEY")
        return

    graph = build_preview_graph()
    # 每次运行新 thread_id：预览不续接历史会话（跨进程恢复是批次5 的主题）
    thread = {"configurable": {"thread_id": f"preview-{uuid4().hex[:8]}"}}

    user_input = input("描述你的店铺（品类 / 风格 / 期望）：").strip()
    print("\n正在解析需求……")
    result = graph.invoke({"user_input": user_input}, config=thread)

    # 路径一：异常输入（空/过短/解析耗尽）-> abort -> END，error 在 state 里
    if result.get("error"):
        print(f"\n[异常路径] {result['error']}")
        return

    # 路径二：正常在 choose_entry 处暂停
    interrupts = result.get("__interrupt__") or []
    if not interrupts:
        print("\n[意外] 图未在选择入口处暂停 —— 请检查 langgraph 版本 >= 1.0")
        return

    # 渲染分析结果（invoke 的返回值就是当前落盘的 state）
    analysis = result.get("analysis")
    if analysis:
        print(f"\n需求摘要：{analysis.needs_summary}")
        if analysis.product_categories:
            print(f"识别品类：{'、'.join(analysis.product_categories)}")

    # 渲染 interrupt payload —— 注意这段代码不知道图的内部结构，
    # 只消费 payload 合同：换 Web UI 时逻辑原样搬走
    payload = interrupts[0].value
    print(f"\n{payload['title']}")
    for i, opt in enumerate(payload["options"], start=1):
        print(f"  {i}. {opt['label']} —— {opt['desc']}")
    print(f"  推荐：{payload['recommended']}（{payload['reason']}）")
    if payload.get("missing_info"):
        print(f"  信息缺口：{'、'.join(payload['missing_info'])}")

    raw = input("\n你的选择（1/2/3 或 manual/framework/one_click）：").strip()
    resume_value = _MENU_ALIASES.get(raw, raw)

    # 恢复：同一个 thread_id + Command(resume=...)。此刻 choose_entry
    # 节点从头重跑，interrupt() 第二次执行拿到 resume_value。
    # 选了一键模式的话，本次 invoke 会跑完 fan-out -> 并行生成 -> 归并
    # -> assemble 全流程 —— 单次 invoke，内部并行。
    from langgraph.types import Command

    if resume_value == "one_click":
        print("\n一键模式：规划结构后按页面并行生成，请稍候……")
    elif resume_value == "framework":
        print("\n框架模式：正在规划整店结构……")

    final = graph.invoke(Command(resume=resume_value), config=thread)
    _render_final(final)
    print("\n✓ 本次运行结束。")


if __name__ == "__main__":
    main()
