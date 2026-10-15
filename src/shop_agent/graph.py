"""图组装层：把节点、路由、检查点装配成可运行的完整图。

本文件是框架耦合的"汇聚点"—— schema/state/nodes/prompts 全部
不依赖 langgraph，只有这里（和 CLI 适配层）import 框架。换框架
（或升级大版本）时，业务层不动，改这一个文件。

两张图：

主图（三入口全流程）：
    START -> parse_input ->(error? END : choose_entry)
    choose_entry -> interrupt（人机协同）
    -> manual_mode   -> END
    -> framework_mode -> END
    -> one_click_mode -> fan_out_pages -> [Send xN] generate_page -> assemble -> END
                       -> ["assemble"]（空计划兜底）———————————————^

补齐图（对既有树的占位页回填，会话内多轮）：
    START -> fan_out_placeholders -> [Send xN] generate_page -> assemble -> END

补齐图是对"占位是任何时刻都合法的状态"这句话的兑现：主图留下的
占位页，在补齐图里走与主图**同一个 generate_page 节点、同一套
归并律**完成回填 —— 不是另一套生成逻辑，是同一机制的再入。
"""

from __future__ import annotations

from pathlib import Path

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
from shop_agent.schema import PageDraft, ShopTree
from shop_agent.state import ShopState


def build_graph(checkpointer=None):
    """主图。checkpointer 由调用方注入：CLI/崩溃恢复用 SqliteSaver，
    普通测试用 InMemorySaver，不传则无持久化（interrupt 将不可用）。
    """
    from langgraph.graph import END, START, StateGraph

    builder = StateGraph(ShopState)

    builder.add_node("parse_input", parse_input)
    builder.add_node("choose_entry", choose_entry)
    builder.add_node(BRANCH_MANUAL, manual_mode)
    builder.add_node(BRANCH_FRAMEWORK, framework_mode)
    builder.add_node(BRANCH_ONE_CLICK, one_click_mode)
    builder.add_node("generate_page", generate_page)
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
    builder.add_edge(BRANCH_MANUAL, END)
    builder.add_edge(BRANCH_FRAMEWORK, END)

    # Send fan-out：条件边返回 [Send xN] 或 ["assemble"]（空计划兜底）。
    # path_map 用 list 形式（框架归一为 {name: name} 恒等映射），
    # Send 对象直通目标节点（1.2.x _branch.py::_finish 语义）。
    builder.add_conditional_edges(
        BRANCH_ONE_CLICK,
        fan_out_pages,
        ["generate_page", "assemble"],
    )
    # 普通边 = 免费 barrier：N 个并行分支全部归并后，assemble 才执行
    builder.add_edge("generate_page", "assemble")
    builder.add_edge("assemble", END)

    if checkpointer is not None:
        return builder.compile(checkpointer=checkpointer)
    return builder.compile()


def fan_out_placeholders(state: ShopState):
    """补齐图的条件边：对树上全部占位页发 Send。

    与主图的 fan_out_pages 同构：构造 PageDraft（占位页没有需求
    片段，用"商家要求补齐"作为 excerpt），携带树快照，逐页回填。
    """
    from langgraph.types import Send

    tree: ShopTree | None = state.get("shop_tree")
    if tree is None:
        return ["assemble"]

    tree_payload = tree.model_dump()
    sends = []
    for page_id in tree.placeholder_page_ids():
        page = tree.pages[page_id]
        draft = PageDraft(
            page_id=page_id,
            zone_id=page.zone_id,
            name=page.name,
            page_type=page.page_type,
            requirement_excerpt=(
                f"商家要求补齐「{page.name}」页面；"
                f"结合全店「{tree.shop_name}」的既有风格与页面清单生成内容"
            ),
        )
        sends.append(Send("generate_page", {"draft": draft.model_dump(), "tree": tree_payload}))

    return sends or ["assemble"]


def build_backfill_graph(checkpointer=None):
    """补齐图：复用主图的 generate_page 与 assemble。

    注意：与主图**不共享 thread**（不同拓扑不能挂在同一检查点序列
    上）。CLI 在会话内从主图 get_state 拿到当前树，作为补齐图的
    初始状态注入 —— get_state 是读取检查点的公开 API，也是 CLI
    实现"会话续接"的基础。
    """
    from langgraph.graph import END, START, StateGraph

    builder = StateGraph(ShopState)
    builder.add_node("generate_page", generate_page)
    builder.add_node("assemble", assemble)

    builder.add_conditional_edges(START, fan_out_placeholders, ["generate_page", "assemble"])
    builder.add_edge("generate_page", "assemble")
    builder.add_edge("assemble", END)

    if checkpointer is not None:
        return builder.compile(checkpointer=checkpointer)
    return builder.compile()


def _schema_allowlist_serde():
    """构造带显式白名单的序列化器（msgpack 反序列化授权）。

    langgraph-checkpoint 4.x 起，从检查点反序列化"安全类型清单之外"
    的自定义类会打警告（未来版本默认拒绝）。我们的检查点里出现的
    自定义类型（InputAnalysis / EntryChoice / ShopTree 及其子模型）
    全部定义在 shop_agent.schema 一个模块里 —— 这里运行时反射枚举
    该模块全部类注册进白名单，新增模型零维护。

    旧版 langgraph-checkpoint 没有 allowed_msgpack_modules 参数 /
    jsonplus 路径变动时，退回 None（saver 用默认序列化器），行为不变。
    """
    import inspect

    try:
        from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
    except ImportError:  # pragma: no cover - 路径变动的未来版本
        return None

    from shop_agent import schema

    classes = tuple(
        obj for _, obj in inspect.getmembers(schema, inspect.isclass)
        if obj.__module__ == schema.__name__
    )
    try:
        return JsonPlusSerializer(allowed_msgpack_modules=classes)
    except TypeError:  # pragma: no cover - 旧版无此参数
        return None


def open_sqlite_checkpointer(db_path: Path):
    """打开 SqliteSaver（跨进程崩溃恢复的载体）。

    双路径导入（对着 langgraph-checkpoint-sqlite 3.1.1 的 RECORD
    核实过）：该包安装为命名空间包 langgraph/checkpoint/sqlite/，
    与 langgraph 主包在同一 site-packages 里目录共存；顶层包名
    langgraph_checkpoint_sqlite 是未来版本的迁移方向，作为次选
    兜底 —— 两条都试，兼容新旧环境。

    serde 传入 schema 白名单序列化器：消除 4.x 的
    "Deserializing unregistered type" 警告（见 _schema_allowlist_serde）。
    官方 from_conn_string 在 3.1.1 不透传 serde，这里按其源码等价展开
    （closing + check_same_thread=False），改走构造函数注入。

    返回上下文管理器：with open_sqlite_checkpointer(p) as saver: ...
    （等价 SqliteSaver.from_conn_string 官方用法，退出时自动关闭连接）
    """
    import sqlite3
    from contextlib import closing, contextmanager

    try:
        from langgraph.checkpoint.sqlite import SqliteSaver  # 3.1.1 实际路径
    except ImportError:  # pragma: no cover - 未来版本顶层包名
        from langgraph_checkpoint_sqlite import SqliteSaver  # type: ignore[no-redef]

    serde = _schema_allowlist_serde()

    @contextmanager
    def _saver_cm():
        with closing(sqlite3.connect(str(db_path), check_same_thread=False)) as conn:
            yield SqliteSaver(conn, serde=serde)

    return _saver_cm()
