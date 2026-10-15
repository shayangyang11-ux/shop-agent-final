"""批次4 测试：规划编译 / 三模式节点 / generate_page / fan-out / assemble。

设计说明：
- 需要真实 Send 类的两条 fan-out 测试用 importorskip 门控 —— 装了
  langgraph 就跑，没装则跳过（CI 分层：纯逻辑测试不依赖框架）。
- 其余全部用假客户端注入，零网络零费用。
- 本文件与 test_state.py 里的 reducer 穷举互为表里：那里证明
  "归并律成立"，这里证明"分支真的按归并律生产部分树"。
"""

import json

import pytest

from shop_agent.nodes.assemble import assemble, build_summary
from shop_agent.nodes.generate_page import generate_page
from shop_agent.nodes.modes import (
    fan_out_pages,
    framework_mode,
    manual_mode,
    one_click_mode,
)
from shop_agent.nodes.planning import compile_plan
from shop_agent.schema import (
    NodeStatus,
    PageDraft,
    PageType,
    PlannedPage,
    PlannedZone,
    ShopTree,
    TreePlan,
)

# ---------------------------------------------------------------------------
# fixtures：一份"带占位详情页 + 一条幻觉链接 + 重名页面"的典型规划
# ---------------------------------------------------------------------------

VALID_TREE_PLAN_JSON = json.dumps(
    {
        "shop_name": "山野茶铺",
        "zones": [{"name": "首页"}, {"name": "商品区"}],
        "pages": [
            {
                "name": "品牌首屏",
                "zone": "首页",
                "page_type": "banner",
                "requirement_excerpt": "山野茶调性，主推春茶",
                "placeholder": False,
                "links": [],
            },
            {
                "name": "茶叶列表",
                "zone": "商品区",
                "page_type": "list",
                "requirement_excerpt": "全部在售茶叶",
                "placeholder": False,
                "links": [
                    {"trigger": "点击商品卡片", "to_page": "茶叶详情"},
                    {"trigger": "查看活动", "to_page": "不存在的页面"},  # 幻觉链接
                ],
            },
            {
                "name": "茶叶详情",
                "zone": "商品区",
                "page_type": "detail",
                "requirement_excerpt": "缺具体单品信息",
                "placeholder": True,
                "links": [],
            },
        ],
    },
    ensure_ascii=False,
)

VALID_PAGE_CONTENT_JSON = json.dumps(
    {
        "headline": "山野之间，一盏春茶",
        "subheadline": "高山生态茶园直供",
        "components": [{"type": "hero", "props": {"title": "春茶上新"}}],
        "notes": "依据：山野茶调性，主推春茶",
    },
    ensure_ascii=False,
)


def _plan() -> TreePlan:
    return TreePlan.model_validate(json.loads(VALID_TREE_PLAN_JSON))


def _analysis_state(**extra):
    """带最小 analysis 的 state（modes 节点读取它做上下文）。"""
    from shop_agent.schema import InputAnalysis

    base = {
        "user_input": "开一家山野风格的茶叶店，主推春茶",
        "analysis": InputAnalysis(
            needs_summary="山野风茶叶店",
            recommended_entry="one_click",
            reason="描述完整",
            product_categories=["茶叶"],
        ),
    }
    base.update(extra)
    return base


# ---------------------------------------------------------------------------
# compile_plan：确定性编译层
# ---------------------------------------------------------------------------


def test_compile_plan_translates_names_to_stable_ids():
    """页面名/分区名 -> p-xxx/z-xxx 稳定翻译；links 的 to_page 同步翻译。"""
    tree, drafts, _ = compile_plan(_plan())

    assert set(tree.zones) == {"z-001", "z-002"}
    assert set(tree.pages) == {"p-001", "p-002", "p-003"}
    list_page = tree.pages["p-002"]
    assert list_page.links[0].to_page == "p-003"  # 名字"茶叶详情"已翻译成 id


def test_compile_plan_lands_all_placeholder_tree():
    """结构先行：无论 placeholder 标记如何，树里全部先落 PLACEHOLDER。"""
    tree, drafts, _ = compile_plan(_plan())

    assert all(p.status is NodeStatus.PLACEHOLDER for p in tree.pages.values())
    assert all(p.content is None for p in tree.pages.values())
    # 只有非占位页进入生成计划
    assert [d.page_id for d in drafts] == ["p-001", "p-002"]
    assert all(d.page_type in (PageType.BANNER, PageType.LIST) for d in drafts)


def test_compile_plan_drops_hallucinated_links():
    """幻觉链接防御：指向不存在页面的 to_page 被丢弃并计数。"""
    tree, _, dropped = compile_plan(_plan())

    assert dropped == 1
    assert len(tree.pages["p-002"].links) == 1  # 只有合法那条保留


def test_compile_plan_dedupes_duplicate_page_names():
    """LLM 给了重名页面：确定性去重（后缀 #2），不炸、不静默合并。"""
    plan = TreePlan(
        shop_name="重名店",
        zones=[PlannedZone(name="主区")],
        pages=[
            PlannedPage(name="列表", zone="主区", page_type="list", requirement_excerpt="r"),
            PlannedPage(name="列表", zone="主区", page_type="list", requirement_excerpt="r2"),
        ],
    )
    tree, drafts, _ = compile_plan(plan)

    assert len(tree.pages) == 2  # 两页都在，不互相吞
    assert [p.name for p in tree.pages.values()] == ["列表", "列表#2"]


def test_compile_plan_is_functional():
    """纯函数纪律：编译不改入参（重名改名在副本上发生）。"""
    plan = _plan()
    raw_names = [p.name for p in plan.pages]

    compile_plan(plan)

    assert [p.name for p in plan.pages] == raw_names


def test_tree_plan_rejects_empty_pages():
    """合同边界：零页面的规划在 Pydantic 层就被拒绝（min_length=1）。"""
    with pytest.raises(ValueError):
        TreePlan(shop_name="空店", zones=[], pages=[])


# ---------------------------------------------------------------------------
# modes：三入口分支节点（假客户端）
# ---------------------------------------------------------------------------


def test_manual_mode_writes_summary_only(fake_llm_client):
    """手动模式：不建树。它没有 client 参数 —— 签名层面就没有 LLM 依赖。"""
    result = manual_mode(_analysis_state())

    assert "shop_tree" not in result
    assert "待命" in result["summary"]


def test_framework_mode_produces_all_placeholder_tree(fake_llm_client):
    """框架模式：结构 + 全占位，不产出任何生成任务。"""
    client, _ = fake_llm_client([VALID_TREE_PLAN_JSON])

    result = framework_mode(_analysis_state(), client=client)

    assert result["page_drafts"] == []
    tree = result["shop_tree"]
    assert isinstance(tree, ShopTree)
    assert len(tree.placeholder_page_ids()) == 3
    assert "占位页" in result["summary"]


def test_one_click_mode_emits_tree_and_drafts(fake_llm_client):
    """一键模式：全占位树 + 生成计划（drafts 是 Send 的数据源）。"""
    client, _ = fake_llm_client([VALID_TREE_PLAN_JSON])

    result = one_click_mode(_analysis_state(), client=client)

    assert len(result["shop_tree"].pages) == 3
    assert [d.page_id for d in result["page_drafts"]] == ["p-001", "p-002"]


def test_plan_failure_returns_error_not_crash(fake_llm_client):
    """规划 LLM 耗尽重试 -> error 写入 state（异常是数据，图不裸崩）。"""
    client, _ = fake_llm_client(["不是 JSON"] * 10)

    result = one_click_mode(_analysis_state(), client=client)

    assert "error" in result and "整店规划失败" in result["error"]


# ---------------------------------------------------------------------------
# generate_page：并行分支本体（假客户端）
# ---------------------------------------------------------------------------


def _fanout_payload(drafts, tree):
    return [{"draft": d.model_dump(), "tree": tree.model_dump()} for d in drafts]


def test_generate_page_success_returns_partial_tree(fake_llm_client):
    """成功路径：部分树 = 该页(DRAFT+内容) + 该页分区，其余一概不带。"""
    client, calls = fake_llm_client([VALID_PAGE_CONTENT_JSON])
    tree, drafts, _ = compile_plan(_plan())
    payload = _fanout_payload(drafts, tree)[0]

    result = generate_page(payload, client=client)

    partial = result["shop_tree"]
    assert set(partial.pages) == {"p-001"}
    assert set(partial.zones) == {"z-001"}
    assert partial.shop_name == ""  # 不覆盖全局店名（merge 规则前提）
    assert partial.pages["p-001"].status is NodeStatus.DRAFT
    assert partial.pages["p-001"].content.headline == "山野之间，一盏春茶"
    # prompt 合同：整店一致性靠"看见兄弟页面清单"
    user_msg = calls[0]["messages"][1]["content"]
    assert "茶叶列表" in user_msg and "茶叶详情" in user_msg


def test_generate_page_preserves_planned_links(fake_llm_client):
    """结构写权限分离：links 来自规划快照，生成节点原样携带（列表页）。"""
    client, _ = fake_llm_client([VALID_PAGE_CONTENT_JSON])
    tree, drafts, _ = compile_plan(_plan())
    payload = _fanout_payload(drafts, tree)[1]  # p-002 茶叶列表

    result = generate_page(payload, client=client)

    links = result["shop_tree"].pages["p-002"].links
    assert len(links) == 1 and links[0].to_page == "p-003"


def test_generate_page_llm_failure_leaves_placeholder(fake_llm_client):
    """韧性核心：单页生成失败 -> 返回 {}，该页留占位，不炸整图。"""
    client, _ = fake_llm_client(["垃圾输出"] * 10)
    tree, drafts, _ = compile_plan(_plan())
    payload = _fanout_payload(drafts, tree)[0]

    result = generate_page(payload, client=client)

    assert result == {}


def test_generate_page_malformed_payload_returns_empty():
    """畸形 Send payload：防御性返回 {}，不打断其他 N-1 个分支。"""
    assert generate_page({"draft": "不是dict"}) == {}
    assert generate_page({}) == {}


# ---------------------------------------------------------------------------
# fan_out_pages：Send 展开（需 langgraph，importorskip 门控）
# ---------------------------------------------------------------------------


def test_fan_out_empty_plan_routes_to_assemble():
    """空计划兜底：返回 ["assemble"]（1.2.x 混合列表语义）。"""
    pytest.importorskip("langgraph.types")
    result = fan_out_pages({"page_drafts": [], "shop_tree": ShopTree(shop_name="x")})
    assert result == ["assemble"]


def test_fan_out_one_send_per_draft():
    """非空计划：每个 draft 恰好一个 Send，payload 含快照。"""
    pytest.importorskip("langgraph.types")
    tree, drafts, _ = compile_plan(_plan())

    sends = fan_out_pages({"page_drafts": drafts, "shop_tree": tree})

    assert len(sends) == 2
    assert all(s.node == "generate_page" for s in sends)
    assert sends[0].arg["draft"]["page_id"] == "p-001"
    assert sends[1].arg["draft"]["page_id"] == "p-002"
    assert sends[0].arg["tree"]["shop_name"] == "山野茶铺"  # 原生 dict 快照


def test_fan_out_payload_is_plain_dict():
    """序列化边界纪律：Send arg 是原生 dict，不是 Pydantic 对象。"""
    pytest.importorskip("langgraph.types")
    tree, drafts, _ = compile_plan(_plan())

    sends = fan_out_pages({"page_drafts": drafts, "shop_tree": tree})

    arg = sends[0].arg
    assert isinstance(arg, dict)
    assert isinstance(arg["draft"], dict)
    json.dumps(arg, ensure_ascii=False)  # 能过 JSON 序列化 = checkpointer 友好


# ---------------------------------------------------------------------------
# assemble：收尾验收
# ---------------------------------------------------------------------------


def _filled_tree() -> ShopTree:
    """全部回填完成的树（占位 -> draft，含规划的占位详情页被后补）。

    p-003（规划期信息不足的详情页）也被填上 —— 占位回填闭环的
    终点形态：列表页跳转的目标页最终存在，dangling 清零。
    """
    from shop_agent.schema import PageContent

    tree, _, _ = compile_plan(_plan())
    pages = {
        pid: page.model_copy(
            update={"status": NodeStatus.DRAFT, "content": PageContent(headline=pid)}
        )
        for pid, page in tree.pages.items()
    }
    return ShopTree(shop_name=tree.shop_name, zones=tree.zones, pages=pages)


def test_assemble_full_success():
    result = assemble({"shop_tree": _filled_tree()})
    assert "全部回填" in result["summary"]


def test_assemble_reports_placeholders():
    """部分完成：列出占位页 + 悬空跳转提示 + 重试指引。"""
    tree, _, _ = compile_plan(_plan())  # 全占位树

    result = assemble({"shop_tree": tree})

    assert "3 页留待回填" in result["summary"]
    assert "茶叶详情" in result["summary"]
    assert "悬空" in result["summary"]


def test_assemble_no_tree():
    assert "没有可汇总" in assemble({})["summary"]


def test_build_summary_dangling_counted():
    tree, _, _ = compile_plan(_plan())
    assert build_summary(tree).count("悬空") == 1
