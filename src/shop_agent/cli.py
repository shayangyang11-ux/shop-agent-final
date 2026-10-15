"""正式 CLI：多轮会话 + interrupt 菜单 + 崩溃恢复 + 占位补齐。

分层声明：CLI 是框架的"终端适配层"——像 llm.py 之于 openai 一样，
这一层被允许直接依赖 langgraph（Command / get_state），因为它的
职责就是把图的运行时行为翻译成终端交互。业务节点依然零框架依赖。

三个入口：
    python -m shop_agent.cli                # 新会话
    python -m shop_agent.cli --resume <id>  # 崩溃恢复（批次4 思考题的答案）
    shop-agent                              # pip install -e . 后的命令行入口

崩溃恢复的两种形态（对号入座第 3 周实验）：
1. interrupt 处"死机"（如 Ctrl+C 在菜单上）：
   检查点停在 choose_entry 完成之前。--resume 检测到挂起的中断，
   重新渲染菜单，Command(resume=...) 恢复 —— parse_input 不重跑。
2. 节点执行中途"死机"（如 one_click_mode 调 LLM 时断电）：
   检查点停在已完成节点之后。--resume 检测到 next 非空但无挂起中断，
   invoke(None) 从检查点继续 —— 未完成节点重跑（at-least-once），
   已完成节点不重跑；重放写入靠 merge_tree 幂等吸收。
"""

from __future__ import annotations

import argparse
from uuid import uuid4

# CLI 是终端适配层：直接依赖框架（langgraph）是这层的合法权利（业务层没有）
from langgraph.types import Command
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.tree import Tree as RichTree

from shop_agent.config import get_settings
from shop_agent.graph import build_backfill_graph, build_graph, open_sqlite_checkpointer
from shop_agent.schema import ShopTree

console = Console()

# 终端菜单序号 -> EntryChoice 值（展示层协议，与 preview.py 一致）
_MENU_ALIASES = {"1": "manual", "2": "framework", "3": "one_click"}

_STATUS_BADGE = {
    "placeholder": "[yellow]◻ 占位[/yellow]",
    "draft": "[green]✓ 已生成[/green]",
    "ready": "[green]✓ 已定稿[/green]",
}


# ---------------------------------------------------------------------------
# 渲染（展示层：只消费 payload / ShopTree 合同，不碰图内部）
# ---------------------------------------------------------------------------


def _render_menu(payload: dict) -> None:
    console.print(Panel(payload["title"], style="bold"))
    for i, opt in enumerate(payload["options"], start=1):
        console.print(f"  [bold]{i}.[/bold] {opt['label']} —— {opt['desc']}")
    # reason / missing_info 来自 LLM 分析，可能带 rich 标记字符（如 [），必须转义
    console.print(f"  [dim]AI 推荐：{payload['recommended']}（{escape(payload['reason'])}）[/dim]")
    if payload.get("missing_info"):
        missing = "、".join(escape(m) for m in payload["missing_info"])
        console.print(f"  [dim]信息缺口：{missing}[/dim]")


def _render_tree(tree: ShopTree) -> None:
    rich_tree = RichTree(f"🏬 [bold]{escape(tree.shop_name)}[/bold]")
    for zone_id, zone in tree.zones.items():
        zone_branch = rich_tree.add(f"[bold]{escape(zone.name)}[/bold]")
        for page in tree.pages_by_zone(zone_id):
            badge = _STATUS_BADGE[page.status.value]
            label = f"{escape(page.name)} [dim]· {page.page_type.value}[/dim] {badge}"
            if page.links:
                targets = "、".join(
                    tree.pages[lnk.to_page].name if lnk.to_page in tree.pages else lnk.to_page
                    for lnk in page.links
                )
                label += f" [dim]→ {escape(targets)}[/dim]"
            zone_branch.add(label)
    console.print(rich_tree)


def _render_result(result: dict, thread_id: str) -> ShopTree | None:
    """渲染一次运行的最终状态，返回当前树（供会话循环判断可否补齐）。"""
    if result.get("error"):
        console.print(Panel(escape(str(result["error"])), title="未完成", border_style="red"))
        return result.get("shop_tree")

    tree = result.get("shop_tree")
    if tree is not None:
        _render_tree(tree)
        _export_tree(tree, thread_id)
    if result.get("summary"):
        # summary 内嵌 LLM 生成的页面名，同样需要转义
        console.print(Panel(escape(result["summary"]), title="执行报告", border_style="green"))
    return tree


def _export_tree(tree: ShopTree, thread_id: str) -> None:
    """把最终树导出为 JSON —— 用户看得见、拿得走的交付物。"""
    settings = get_settings()
    out_dir = settings.db_path.parent / "trees"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{thread_id}.json"
    path.write_text(tree.model_dump_json(indent=2), encoding="utf-8")
    console.print(f"[dim]店铺树已导出：{path}[/dim]")


# ---------------------------------------------------------------------------
# 会话流
# ---------------------------------------------------------------------------


def _read_choice() -> str:
    raw = input("你的选择（1/2/3 或 manual/framework/one_click）> ").strip()
    return _MENU_ALIASES.get(raw, raw)


def _run_with_menu(graph, payload_first: dict | None, config: dict, *, first_result: dict | None):
    """公共循环：处理（可能多层的）interrupt 菜单直到图跑完。

    first_result / payload_first 二选一传入：新会话从 invoke 结果里
    拿中断；--resume 从检查点快照里拿。循环本身对来源无感。
    """
    result = first_result
    payload = payload_first
    while True:
        if payload is None and result is not None and "__interrupt__" in result:
            payload = result["__interrupt__"][0].value
        if payload is None:
            return result
        _render_menu(payload)
        result = graph.invoke(Command(resume=_read_choice()), config)
        payload = None


def _new_requirement_turn(graph, thread_id: str, preset: str | None = None) -> ShopTree | None:
    """一轮新需求：输入 -> 解析 -> 菜单 -> 分支执行。

    preset：由会话循环"下一步"直接透传的新需求（用户在提示符里
    输入的那句话就是需求本身，不再追问一遍）。
    """
    if preset:
        text = preset
        console.print(f"[dim]新需求：{escape(text)}[/dim]")
    else:
        text = input("\n请描述你的店铺需求（exit 退出）> ").strip()
    if not text:
        return None
    config = {"configurable": {"thread_id": thread_id}}
    console.print("[dim]正在解析需求……[/dim]")
    result = graph.invoke({"user_input": text}, config)
    result = _run_with_menu(graph, None, config, first_result=result)
    return _render_result(result, thread_id)


def _resume_pending(graph, thread_id: str) -> ShopTree | None:
    """--resume：从检查点续接未完成会话（两种崩溃形态都处理）。"""
    config = {"configurable": {"thread_id": thread_id}}
    snap = graph.get_state(config)

    if not snap.next:
        console.print("[yellow]该会话没有未完成的步骤。[/yellow]")
        return snap.values.get("shop_tree")

    # 形态 1：interrupt 处中断（菜单挂起）
    payload = None
    for task in snap.tasks:
        if getattr(task, "interrupts", None):
            payload = task.interrupts[0].value
            break

    if payload is not None:
        console.print("[green]检测到挂起的模式选择（解析结果未丢失，无需重跑）。[/green]")
        result = _run_with_menu(graph, payload, config, first_result=None)
        return _render_result(result, thread_id)

    # 形态 2：节点执行中途崩溃 -> 从检查点重放（at-least-once）
    console.print("[yellow]检测到未完成的节点，从检查点继续（该节点将重跑）……[/yellow]")
    result = graph.invoke(None, config)
    result = _run_with_menu(graph, None, config, first_result=result)
    return _render_result(result, thread_id)


def _backfill_turn(backfill_graph, tree: ShopTree) -> ShopTree:
    """补齐轮：对占位页 Send 并行回填（新 thread，不污染主图检查点）。"""
    bf_thread = f"backfill-{uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": bf_thread}}
    n = len(tree.placeholder_page_ids())
    console.print(f"[dim]补齐 {n} 个占位页（Send 并行，复用主图生成节点）……[/dim]")

    result = backfill_graph.invoke({"shop_tree": tree}, config)
    new_tree = result.get("shop_tree") or tree
    _render_result(result, bf_thread)
    return new_tree


def run_session(graph, backfill_graph, resume_thread: str | None = None) -> None:
    """CLI 主循环：新需求 / 补齐 / 退出。会话内树保存在内存。"""
    if resume_thread:
        current_tree = _resume_pending(graph, resume_thread)
    else:
        thread_id = uuid4().hex[:8]
        console.print(
            f"[dim]会话 {thread_id} 已开启 "
            f"（中断后可用 python -m shop_agent.cli --resume {thread_id} 续接）[/dim]"
        )
        current_tree = _new_requirement_turn(graph, thread_id)

    while True:
        placeholders = current_tree.placeholder_page_ids() if current_tree else []
        hint = "输入新需求开一家新店 / exit 退出"
        if placeholders:
            hint = "输入 [bold]补齐[/bold] 为占位页生成内容 / " + hint
        raw = input(f"\n下一步（{hint}）> ").strip()

        if raw.lower() in {"exit", "quit", "q", "退出"}:
            console.print("[dim]再见。[/dim]")
            return
        if raw == "补齐" and placeholders:
            current_tree = _backfill_turn(backfill_graph, current_tree)
        elif raw:
            thread_id = uuid4().hex[:8]
            console.print(f"[dim]会话 {thread_id}（--resume {thread_id}）[/dim]")
            # "下一步"里输入的那句话直接作为新店需求（不再重复追问）
            current_tree = _new_requirement_turn(graph, thread_id, preset=raw)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="shop-agent", description="AI 电商店铺装修 Agent（LangGraph 编排）"
    )
    parser.add_argument(
        "--resume", metavar="THREAD_ID", help="续接未完成的会话（崩溃恢复入口）"
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    if not settings.is_configured:
        console.print(
            "[red]未配置 API Key：请复制 .env.example 为 .env 并填写 SHOP_AGENT_LLM_API_KEY[/red]"
        )
        raise SystemExit(1)
    settings.ensure_runtime_dirs()

    # SqliteSaver 以上下文管理器持有：整个 CLI 会话期间连接常开，
    # 退出时自动关闭。跨进程崩溃恢复依赖这个文件落盘。
    with open_sqlite_checkpointer(settings.db_path) as saver:
        graph = build_graph(checkpointer=saver)
        backfill_graph = build_backfill_graph(checkpointer=saver)
        try:
            run_session(graph, backfill_graph, args.resume)
        except (KeyboardInterrupt, EOFError):
            console.print(
                "\n[yellow]会话被中断。已完成节点的检查点已落盘，"
                "可用 --resume <THREAD_ID> 续接（不会重跑解析）。[/yellow]"
            )


if __name__ == "__main__":
    main()
