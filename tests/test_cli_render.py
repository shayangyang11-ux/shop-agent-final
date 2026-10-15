"""CLI 渲染层回归测试：不调 LLM、不花钱，只验证展示函数消费 ShopTree 的方式。

为什么存在：`_render_tree` 曾在真机运行时崩溃 —— 调 pages_by_zone()
漏传 zone_id 参数。62 个测试全绿也没拦住，因为渲染层此前零覆盖
（测试测节点和数据，没人测"打印"）。本文件把这条路径钉进测试网：
树渲染、含 rich 标记字符的 LLM 文本、菜单与报告面板，都不允许再抛异常。
"""

from shop_agent.cli import _render_menu, _render_result, _render_tree
from shop_agent.schema import (
    LinkSpec,
    NodeStatus,
    PageContent,
    PageNode,
    PageType,
    ShopTree,
    ZoneNode,
)


def _framework_tree(shop_name: str = "手作咖啡豆铺") -> ShopTree:
    """模拟框架模式产物：双分区、占位页、指向占位页的悬空链接。"""
    return ShopTree(
        shop_name=shop_name,
        zones={
            "z-home": ZoneNode(id="z-home", name="首页分区"),
            "z-shop": ZoneNode(id="z-shop", name="商品分区"),
        },
        pages={
            "p-list": PageNode(
                id="p-list",
                zone_id="z-home",
                name="咖啡豆列表",
                page_type=PageType.LIST,
                status=NodeStatus.DRAFT,
                content=PageContent(headline="全部豆单"),
                links=[LinkSpec(trigger="点击商品卡片", to_page="p-detail-1")],
            ),
            "p-detail-1": PageNode(
                id="p-detail-1",
                zone_id="z-shop",
                name="耶加雪菲详情",
                page_type=PageType.DETAIL,
                status=NodeStatus.PLACEHOLDER,
            ),
        },
    )


def test_render_tree_framework_result():
    """回归：双分区 + 占位页 + 悬空链接的树，渲染不允许抛异常。"""
    _render_tree(_framework_tree())


def test_render_tree_escapes_rich_markup():
    """回归：LLM 生成的名字带 [ ] 时，必须原样显示而不是被 rich 当标记解析。"""
    tree = _framework_tree(shop_name="咖啡[豆]屋")
    tree.zones["z-home"].name = "首[页]"
    tree.pages["p-list"].name = "豆单[列表]"
    _render_tree(tree)


def test_render_menu_escapes_llm_text():
    """回归：推荐理由 / 信息缺口来自 LLM 分析，带标记字符时不能崩。"""
    payload = {
        "title": "请选择店铺装修模式",
        "options": [{"label": "手动创建", "desc": "Agent 只响应指令"}],
        "recommended": "framework",
        "reason": "需求提到了[日系]风格",
        "missing_info": ["品牌[故事]", "色彩偏好"],
    }
    _render_menu(payload)


def test_render_result_renders_summary(monkeypatch):
    """回归：完整结果（树 + 报告）渲染不崩；JSON 导出被隔离，不落盘。"""
    monkeypatch.setattr("shop_agent.cli._export_tree", lambda tree, tid: None)
    tree = _framework_tree()
    result = {
        "shop_tree": tree,
        "summary": "部分完成：2 页中 1 页已生成，1 页留待回填：\n  - 耶加雪菲[详情]（p-detail-1）",
    }
    assert _render_result(result, "t-test") is tree
