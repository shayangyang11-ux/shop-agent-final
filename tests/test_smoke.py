"""冒烟测试：不调 LLM、不花钱，验证合同层可导入、不变量生效。

运行：uv run pytest

批次说明：这 6 个测试预演了整个项目的核心语义 ——
特别是 test_backfill_semantics_placeholder_replaced，它就是批次4
占位回填机制的最小形态（字典覆写 + dangling 清零）。
"""

import pytest
from pydantic import ValidationError

from shop_agent.schema import (
    LinkSpec,
    NodeStatus,
    PageContent,
    PageNode,
    PageType,
    ShopTree,
    ZoneNode,
)


def _demo_tree() -> ShopTree:
    """一个带占位详情页的最小店铺树：列表页已生成，详情页待回填。"""
    return ShopTree(
        shop_name="Demo 店",
        zones={"z-home": ZoneNode(id="z-home", name="首页分区")},
        pages={
            "p-list": PageNode(
                id="p-list",
                zone_id="z-home",
                name="商品列表",
                page_type=PageType.LIST,
                status=NodeStatus.DRAFT,
                content=PageContent(headline="全部商品"),
                links=[LinkSpec(trigger="点击商品卡片", to_page="p-detail-1")],
            ),
            "p-detail-1": PageNode(
                id="p-detail-1",
                zone_id="z-home",
                name="商品详情",
                page_type=PageType.DETAIL,
                status=NodeStatus.PLACEHOLDER,
            ),
        },
    )


def test_placeholder_has_no_content():
    tree = _demo_tree()
    assert tree.pages["p-detail-1"].is_placeholder()
    assert tree.pages["p-detail-1"].content is None


def test_invariant_placeholder_with_content_rejected():
    """占位节点带内容 -> 构造时即抛错（类型边界挡住非法状态）。"""
    with pytest.raises(ValidationError):
        PageNode(
            id="p-bad",
            zone_id="z-home",
            name="坏节点",
            page_type=PageType.DETAIL,
            status=NodeStatus.PLACEHOLDER,
            content=PageContent(headline="占位却带内容"),
        )


def test_invariant_real_page_requires_content():
    """非占位节点没内容 -> 构造时即抛错。"""
    with pytest.raises(ValidationError):
        PageNode(
            id="p-bad2",
            zone_id="z-home",
            name="坏节点",
            page_type=PageType.LIST,
            status=NodeStatus.DRAFT,
        )


def test_dangling_links_targets_placeholder():
    tree = _demo_tree()
    dangling = tree.dangling_links()
    assert len(dangling) == 1
    assert dangling[0][0] == "p-list"
    assert dangling[0][1].to_page == "p-detail-1"


def test_pages_by_zone():
    tree = _demo_tree()
    assert [p.id for p in tree.pages_by_zone("z-home")] == ["p-list", "p-detail-1"]


def test_backfill_semantics_placeholder_replaced():
    """占位回填的最小语义：字典覆写 + dangling 清零。

    批次4 的 merge_tree reducer 会把这个动作泛化为
    "分支树 ⊕ 当前树" 的归并规则；此处先钉死语义基线。
    """
    tree = _demo_tree()
    assert len(tree.dangling_links()) == 1

    filled = tree.pages["p-detail-1"].model_copy(
        update={"status": NodeStatus.DRAFT, "content": PageContent(headline="商品详情页")}
    )
    tree.pages["p-detail-1"] = filled

    assert not tree.pages["p-detail-1"].is_placeholder()
    assert tree.dangling_links() == []
    assert tree.placeholder_page_ids() == []
