"""merge_tree reducer 的穷举测试：并行归并 / 占位回填 / 拒绝降级 / 幂等。

这批测试在没有安装 langgraph 的环境也能跑——reducer 是纯函数，
不依赖框架。这是"框架放边缘"设计直接兑现的红利。
"""

from typing import get_type_hints

from shop_agent.schema import (
    LinkSpec,
    NodeStatus,
    PageContent,
    PageNode,
    PageType,
    ShopTree,
    ZoneNode,
)
from shop_agent.state import ShopState, merge_tree


# ---- 构造助手 ----


def _page(page_id: str, status: NodeStatus, links: list[LinkSpec] | None = None) -> PageNode:
    return PageNode(
        id=page_id,
        zone_id="z",
        name=page_id,
        page_type=PageType.LIST,
        status=status,
        content=None if status is NodeStatus.PLACEHOLDER else PageContent(headline=page_id),
        links=links or [],
    )


def _tree(pages: list[PageNode], name: str = "测试店") -> ShopTree:
    return ShopTree(
        shop_name=name,
        zones={"z": ZoneNode(id="z", name="主分区")},
        pages={p.id: p for p in pages},
    )


# ---- 基础语义 ----


def test_first_write_and_none_passthrough():
    """首写采用 update；update 为 None 时原样返回（节点未写该键的旁路）。"""
    t = _tree([_page("p1", NodeStatus.DRAFT)])
    assert merge_tree(None, t) is t
    assert merge_tree(t, None) is t
    assert merge_tree(None, None) is None


def test_parallel_disjoint_branches_merge():
    """Send fan-out：两个分支各持一页，归并后两页俱在，且与完成顺序无关。"""
    branch_a = _tree([_page("p1", NodeStatus.DRAFT)])
    branch_b = _tree([_page("p2", NodeStatus.DRAFT)])

    ab = merge_tree(merge_tree(None, branch_a), branch_b)
    ba = merge_tree(merge_tree(None, branch_b), branch_a)

    assert set(ab.pages) == {"p1", "p2"}
    assert ab == ba  # 交换律：分支完成顺序不影响最终状态


def test_placeholder_backfill():
    """核心机制：占位页被已填充页覆盖（批次4 generate_page 的归并形态）。"""
    current = _tree([_page("p-detail", NodeStatus.PLACEHOLDER)])
    update = _tree([_page("p-detail", NodeStatus.DRAFT)])

    merged = merge_tree(current, update)

    assert not merged.pages["p-detail"].is_placeholder()
    assert merged.pages["p-detail"].content is not None


def test_reject_placeholder_downgrade():
    """陈旧分支晚到：已填充的页不许被打回占位（乱序到达的脏情况）。"""
    current = _tree([_page("p1", NodeStatus.DRAFT)])
    update = _tree([_page("p1", NodeStatus.PLACEHOLDER)])

    merged = merge_tree(current, update)

    assert not merged.pages["p1"].is_placeholder()
    assert merged.pages["p1"].content is not None


def test_merge_is_idempotent():
    """幂等：merge(t, t) == t。

    对应 checkpointer 重放——崩溃恢复后同一份节点输出被二次应用，
    结果必须不变。不满足这条，at-least-once 语义下恢复即数据损坏。
    """
    t = _tree(
        [
            _page("p-list", NodeStatus.DRAFT, links=[LinkSpec(trigger="点击", to_page="p-detail")]),
            _page("p-detail", NodeStatus.PLACEHOLDER),
        ]
    )
    assert merge_tree(t, t) == t

    filled = _tree([_page("p-detail", NodeStatus.DRAFT)])
    merged_once = merge_tree(t, filled)
    merged_twice = merge_tree(merged_once, filled)
    assert merged_once == merged_twice


def test_merge_preserves_pages_absent_from_update():
    """归并是加法不是替换：框架重建树时，用户手建的页面不被吞掉。"""
    current = _tree([_page("p-user", NodeStatus.DRAFT), _page("p-old", NodeStatus.PLACEHOLDER)])
    regen = _tree([_page("p-new", NodeStatus.DRAFT)], name="")

    merged = merge_tree(current, regen)

    assert set(merged.pages) == {"p-user", "p-old", "p-new"}
    assert merged.shop_name == "测试店"  # update 的空店名不覆盖


def test_merge_is_functional():
    """纯函数性：归并不修改入参（可测试性与幂等性的共同前提）。"""
    current = _tree([_page("p1", NodeStatus.PLACEHOLDER)])
    update = _tree([_page("p1", NodeStatus.DRAFT)])

    merge_tree(current, update)

    assert current.pages["p1"].is_placeholder()  # 入参未被改动


def test_state_annotation_wiring():
    """合同验证：shop_tree 键确实挂上了 merge_tree（无 langgraph 也能断言）。"""
    hints = get_type_hints(ShopState, include_extras=True)
    metadata = hints["shop_tree"].__metadata__
    assert merge_tree in metadata
