"""批次5 验收测试：五个端到端用例 + 补齐闭环（真实 langgraph 图级运行）。

需要真实安装 langgraph（importorskip 门控，未装则整文件跳过）。
LLM 一律用假客户端注入：monkeypatch 掉 shop_agent.llm.get_client，
节点内 structured_call 的 client=None 分支自动落到替身上 —— 图级
测试走的完全是生产代码路径，零网络零费用。

五个用例（对应 README「五个验收用例」，也是大作业验收清单）：
  1. 完整需求  -> one_click -> Send 并行 -> 全部回填
  2. 模糊需求  -> framework -> 全占位树
  3. 未开始    -> manual    -> 待命（不建树）
  4. 异常输入  -> abort（零 LLM 调用）
  5. 崩溃恢复  -> interrupt 处"换进程"，同 thread 恢复，解析不重跑
加映：6. 占位补齐闭环（一键留下的占位页，补齐图回填成功）
"""

import importlib.util
import json

import pytest

pytest.importorskip("langgraph")

from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402

from shop_agent.graph import (  # noqa: E402
    build_backfill_graph,
    build_graph,
    fan_out_placeholders,
    open_sqlite_checkpointer,
)

# SqliteSaver 双路径：3.1.1 实际安装为命名空间包（RECORD 核实过），
# 顶层包名为未来迁移方向；两条都没有则第 5/7 用例跳过。
# 只探测可用性不导入（open_sqlite_checkpointer 内部有自己的双路径导入）。
_SQLITE_SAVER = (
    importlib.util.find_spec("langgraph.checkpoint.sqlite") is not None
    or importlib.util.find_spec("langgraph_checkpoint_sqlite") is not None
)

# ---------------------------------------------------------------------------
# 假 LLM 的响应脚本（按节点调用顺序消费）
# ---------------------------------------------------------------------------

FULL_INPUT = "我想开一家卖手工皮具的店，主打定制钱包和卡包，风格复古，客群是年轻男性"

PARSE_ONE_CLICK = json.dumps(
    {
        "needs_summary": "复古手工皮具店，定制钱包卡包",
        "product_categories": ["钱包", "卡包"],
        "recommended_entry": "one_click",
        "reason": "品类、风格、客群都齐全",
        "missing_info": [],
    }
)
PARSE_FRAMEWORK = json.dumps(
    {
        "needs_summary": "想开店，方向模糊",
        "product_categories": [],
        "recommended_entry": "framework",
        "reason": "只给了大致方向，先看结构",
        "missing_info": ["具体品类", "风格偏好"],
    }
)
PARSE_MANUAL = json.dumps(
    {
        "needs_summary": "咨询开店",
        "product_categories": [],
        "recommended_entry": "manual",
        "reason": "需求尚未开始",
        "missing_info": ["店铺定位"],
    }
)

PLAN_WITH_PLACEHOLDER = json.dumps(
    {
        "shop_name": "山野皮具",
        "zones": [{"name": "首页"}, {"name": "商品专区"}],
        "pages": [
            {
                "name": "品牌首屏",
                "zone": "首页",
                "page_type": "banner",
                "requirement_excerpt": "复古调性首屏",
                "links": [],
            },
            {
                "name": "商品列表",
                "zone": "商品专区",
                "page_type": "list",
                "requirement_excerpt": "钱包与卡包列表",
                "links": [{"trigger": "点击商品", "to_page": "定制详情"}],
            },
            {
                "name": "定制详情",
                "zone": "商品专区",
                "page_type": "detail",
                "requirement_excerpt": "缺定制流程与材质信息",
                "placeholder": True,  # 信息不足 -> 占位，一键模式不生成
                "links": [],
            },
            {
                "name": "幻觉目标页",  # 无人声明此名 -> 链接应被编译层丢弃
                "zone": "不存在分区",
                "page_type": "about",
                "requirement_excerpt": "x",
                "links": [{"trigger": "点", "to_page": "品牌首屏"}],
            },
        ],
    }
)
PLAN_ALL_GENERATABLE = json.dumps(
    {
        **json.loads(PLAN_WITH_PLACEHOLDER),
        "pages": [
            page | {"placeholder": False}
            for page in json.loads(PLAN_WITH_PLACEHOLDER)["pages"]
        ],
    }
)

PAGE_CONTENT = json.dumps(
    {
        "headline": "复古手作，一针一线",
        "subheadline": "植鞣革 · 手工缝线 · 可刻字",
        "components": [
            {"type": "hero", "props": {"title": "复古皮具", "style": "复古"}},
            {"type": "grid", "props": {"items": 6}},
        ],
        "notes": "依据复古风格与品类描述生成",
    }
)


@pytest.fixture
def llm_stub(monkeypatch, fake_llm_client):
    """图级注入缝：替换 get_client，让整张图用假客户端跑生产代码。"""

    def install(responses: list[str]):
        client, calls = fake_llm_client(responses)
        monkeypatch.setattr("shop_agent.llm.get_client", lambda: client)
        return calls

    return install


def _role_counts(calls) -> dict[str, int]:
    """按 system prompt 标记统计各节点真实调用次数。"""
    markers = {"parse": "需求分析模块", "plan": "整店规划模块", "page": "页面文案设计师"}
    counts = {"parse": 0, "plan": 0, "page": 0}
    for call in calls:
        system = call["messages"][0]["content"]
        for role, marker in markers.items():
            if marker in system:
                counts[role] += 1
                break
    return counts


def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


# ---------------------------------------------------------------------------
# 用例 1：完整需求 -> one_click -> 并行 -> 全部回填
# ---------------------------------------------------------------------------


def test_case1_full_requirement_one_click_fully_backfilled(llm_stub):
    calls = llm_stub([PARSE_ONE_CLICK, PLAN_ALL_GENERATABLE] + [PAGE_CONTENT] * 4)

    graph = build_graph(checkpointer=InMemorySaver())
    result = graph.invoke({"user_input": FULL_INPUT}, _config("case1"))

    assert "__interrupt__" in result  # 在三入口菜单处暂停
    from langgraph.types import Command

    result = graph.invoke(Command(resume="one_click"), _config("case1"))

    tree = result["shop_tree"]
    assert result["entry_choice"] == "one_click"
    assert len(tree.pages) == 4
    assert tree.placeholder_page_ids() == []  # 全部回填
    assert tree.dangling_links() == []
    assert "全部回填" in result["summary"]

    # 并行真的发生了：3 页生成（第 4 页在"不存在分区"但未标占位，也生成）
    roles = _role_counts(calls)
    assert roles == {"parse": 1, "plan": 1, "page": 4}


# ---------------------------------------------------------------------------
# 用例 2：模糊需求 -> framework -> 全占位树
# ---------------------------------------------------------------------------


def test_case2_vague_requirement_framework_tree(llm_stub):
    calls = llm_stub([PARSE_FRAMEWORK, PLAN_WITH_PLACEHOLDER])

    graph = build_graph(checkpointer=InMemorySaver())
    result = graph.invoke({"user_input": "我想开个店，卖啥还没想好"}, _config("case2"))

    from langgraph.types import Command

    result = graph.invoke(Command(resume="framework"), _config("case2"))

    tree = result["shop_tree"]
    assert all(p.is_placeholder() for p in tree.pages.values())  # 全占位
    assert _role_counts(calls) == {"parse": 1, "plan": 1, "page": 0}  # 零内容生成
    assert "待补" in result["summary"]
    # 幻觉链接被编译层丢弃：所有链接目标都真实存在
    assert all(lnk.to_page in tree.pages for p in tree.pages.values() for lnk in p.links)


# ---------------------------------------------------------------------------
# 用例 3：未开始 -> manual -> 待命
# ---------------------------------------------------------------------------


def test_case3_manual_mode_stands_by(llm_stub):
    calls = llm_stub([PARSE_MANUAL])

    graph = build_graph(checkpointer=InMemorySaver())
    result = graph.invoke({"user_input": "开店都需要准备什么？"}, _config("case3"))

    from langgraph.types import Command

    result = graph.invoke(Command(resume="manual"), _config("case3"))

    assert result["entry_choice"] == "manual"
    assert "shop_tree" not in result  # 不建树
    assert "待命" in result["summary"]
    assert _role_counts(calls) == {"parse": 1, "plan": 0, "page": 0}


# ---------------------------------------------------------------------------
# 用例 4：异常输入 -> abort，零 LLM 调用
# ---------------------------------------------------------------------------


def test_case4_invalid_input_aborts_without_llm(llm_stub):
    calls = llm_stub([])  # 队列为空：任何 LLM 调用都会 IndexError -> 测试失败

    graph = build_graph(checkpointer=InMemorySaver())
    result = graph.invoke({"user_input": "开店"}, _config("case4"))

    assert "过短" in result["error"]
    assert "shop_tree" not in result
    assert calls == []  # 边界拦截，一次 LLM 都没花


# ---------------------------------------------------------------------------
# 用例 5：崩溃恢复 —— interrupt 处"换进程"，解析不重跑
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _SQLITE_SAVER, reason="未安装 langgraph-checkpoint-sqlite")
def test_case5_crash_recovery_same_thread_no_reparse(llm_stub, tmp_path):
    db = tmp_path / "crash.sqlite"
    calls = llm_stub([PARSE_ONE_CLICK, PLAN_ALL_GENERATABLE] + [PAGE_CONTENT] * 4)

    from langgraph.types import Command

    # 阶段一：跑到 interrupt 处"死机"（进程退出、连接关闭）
    # 经 open_sqlite_checkpointer（CLI 同款入口，含 serde 白名单）
    with open_sqlite_checkpointer(db) as saver1:
        graph1 = build_graph(checkpointer=saver1)
        result = graph1.invoke({"user_input": FULL_INPUT}, _config("case5"))
        assert "__interrupt__" in result
    # with 块退出 = 连接关闭，模拟进程死亡

    # 阶段二：新进程、新图实例、同一个 sqlite 文件、同一个 thread_id
    with open_sqlite_checkpointer(db) as saver2:
        graph2 = build_graph(checkpointer=saver2)
        result = graph2.invoke(Command(resume="one_click"), _config("case5"))

    tree = result["shop_tree"]
    assert tree.placeholder_page_ids() == []
    assert "全部回填" in result["summary"]

    # 核心断言：需求解析全程只发生一次（恢复不重跑 parse_input）
    roles = _role_counts(calls)
    assert roles["parse"] == 1
    assert roles == {"parse": 1, "plan": 1, "page": 4}


# ---------------------------------------------------------------------------
# 加映：占位补齐闭环（一键留下的占位页 -> 补齐图回填）
# ---------------------------------------------------------------------------


def test_bonus_backfill_closes_the_loop(llm_stub):
    calls_main = llm_stub(
        [PARSE_ONE_CLICK, PLAN_WITH_PLACEHOLDER, PAGE_CONTENT, PAGE_CONTENT, PAGE_CONTENT]
    )

    graph = build_graph(checkpointer=InMemorySaver())
    from langgraph.types import Command

    result = graph.invoke({"user_input": FULL_INPUT}, _config("bf-main"))
    result = graph.invoke(Command(resume="one_click"), _config("bf-main"))

    tree = result["shop_tree"]
    assert tree.placeholder_page_ids() == ["p-003"]  # 定制详情留占位
    assert len(tree.dangling_links()) == 1  # 列表页那条跳转悬空

    # 用户说「补齐」：换新 stub（等价于新一轮 LLM 会话），补齐图对
    # 占位页走同一套 generate_page 回填
    calls_backfill = llm_stub([PAGE_CONTENT])
    backfill = build_backfill_graph(checkpointer=InMemorySaver())
    result = backfill.invoke({"shop_tree": tree}, _config("bf-backfill"))

    new_tree = result["shop_tree"]
    assert new_tree.placeholder_page_ids() == []
    assert new_tree.dangling_links() == []  # 悬空解除：回填闭环完成
    assert "全部回填" in result["summary"]
    # 主图 3 个生成页 + 补齐图 1 页；解析/规划不因补齐重跑
    assert _role_counts(calls_main) == {"parse": 1, "plan": 1, "page": 3}
    assert _role_counts(calls_backfill) == {"parse": 0, "plan": 0, "page": 1}


def test_fan_out_placeholders_targets_only_placeholders():
    """补齐 fan-out 的单元级确认：只对占位页发 Send，空树兜底 assemble。

    前置状态必须是**生成后的树**（DRAFT + PLACEHOLDER 混合）——刚
    compile_plan 出来的树全页占位（结构先行），那是主图 fan_out_pages
    的前置，不是补齐图的。
    """
    from shop_agent.nodes.planning import compile_plan
    from shop_agent.schema import (
        NodeStatus,
        PageContent,
        PlannedPage,
        PlannedZone,
        ShopTree,
        TreePlan,
    )

    plan = TreePlan(
        shop_name="t",
        zones=[PlannedZone(name="z")],
        pages=[
            PlannedPage(name="a", zone="z", page_type="list", requirement_excerpt="x"),
            PlannedPage(
                name="b", zone="z", page_type="detail", requirement_excerpt="x", placeholder=True
            ),
        ],
    )
    tree, drafts, _ = compile_plan(plan)

    # 模拟主图生成完成后的状态：有 draft 的页回填为 DRAFT
    pages = dict(tree.pages)
    for d in drafts:
        pages[d.page_id] = pages[d.page_id].model_copy(
            update={"status": NodeStatus.DRAFT, "content": PageContent(headline=d.name)}
        )
    tree = ShopTree(shop_name=tree.shop_name, zones=tree.zones, pages=pages)

    sends = fan_out_placeholders({"shop_tree": tree})

    assert len(sends) == 1
    assert sends[0].node == "generate_page"
    assert sends[0].arg["draft"]["page_id"] == "p-002"
    assert fan_out_placeholders({}) == ["assemble"]  # 无树兜底


# ---------------------------------------------------------------------------
# 用例 7（回归）：跨进程恢复检查点零"unregistered type"警告
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _SQLITE_SAVER, reason="未安装 langgraph-checkpoint-sqlite")
def test_case7_checkpoint_serde_allowlist_no_warning(llm_stub, tmp_path, caplog):
    """langgraph-checkpoint 4.x 对未注册自定义类型打警告（未来默认拒绝）。

    open_sqlite_checkpointer 内置 schema 白名单序列化器，因此：
    1. 换 saver 恢复检查点，不再出现 "Deserializing unregistered type"；
    2. 白名单不误伤 —— 自定义类型照常还原为 pydantic 实例（拦截会
       直接报错，还原为 dict 同样过不了 isinstance）。
    """
    import logging

    from shop_agent.schema import InputAnalysis

    # 警告是"每类型每进程只发一次"，清掉计数避免被前面的用例消耗掉
    try:
        from langgraph.checkpoint.serde import jsonplus as _jsonplus

        _jsonplus._warned_unregistered_types.clear()
    except (ImportError, AttributeError):  # pragma: no cover - 内部实现变动
        pass

    db = tmp_path / "serde.sqlite"
    calls = llm_stub([PARSE_ONE_CLICK, PLAN_ALL_GENERATABLE] + [PAGE_CONTENT] * 4)

    with open_sqlite_checkpointer(db) as saver1:
        g1 = build_graph(checkpointer=saver1)
        result = g1.invoke({"user_input": FULL_INPUT}, _config("case7"))
        assert "__interrupt__" in result

    # 换 saver（等价换进程）：从 sqlite 反序列化通道值与挂起的 interrupt
    with open_sqlite_checkpointer(db) as saver2:
        g2 = build_graph(checkpointer=saver2)
        with caplog.at_level(logging.DEBUG, logger="langgraph.checkpoint.serde.jsonplus"):
            snap = g2.get_state(_config("case7"))
            payloads = [
                i.value for t in snap.tasks for i in (getattr(t, "interrupts", None) or [])
            ]

    assert payloads, "应停在菜单挂起处"  # 菜单 interrupt 的 payload 已反序列化
    assert isinstance(snap.values.get("analysis"), InputAnalysis)  # 类型还原成功
    assert "unregistered" not in caplog.text  # 且零警告
    assert calls  # llm_stub 安装成功（确保走的真是生产路径）
