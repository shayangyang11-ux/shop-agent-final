"""评测子包测试：judge 渲染、确定性指标、统计公式、端到端管线。

原则与既有测试一致：图级用例走生产代码路径 + 假客户端注入，
零网络零费用；统计函数用手算已知向量钉死正确性。
"""

import json

import pytest

pytest.importorskip("langgraph")

from shop_agent.eval._fakes import make_eval_fake_client  # noqa: E402
from shop_agent.eval.agreement import (  # noqa: E402
    pearson,
    quadratic_weighted_kappa,
    spearman,
)
from shop_agent.eval.judge import build_judge_messages, render_tree_for_judge  # noqa: E402
from shop_agent.eval.run_eval import deterministic_metrics, run  # noqa: E402
from shop_agent.nodes.planning import compile_plan  # noqa: E402
from shop_agent.schema import TreePlan  # noqa: E402


# ------------------------------------------------------------ 固定小树 --

PLAN = TreePlan.model_validate(
    {
        "shop_name": "测试小店",
        "zones": [{"name": "首页", "brief": "门面"}, {"name": "商品专区", "brief": "陈列"}],
        "pages": [
            {"name": "首页banner", "zone": "首页", "page_type": "banner",
             "requirement_excerpt": "门面", "placeholder": False, "links": []},
            {"name": "商品列表页", "zone": "商品专区", "page_type": "list",
             "requirement_excerpt": "陈列", "placeholder": False,
             "links": [{"trigger": "点击商品", "to_page": "商品详情页", "label": "详情"}]},
            {"name": "商品详情页", "zone": "商品专区", "page_type": "detail",
             "requirement_excerpt": "转化", "placeholder": True, "links": []},
        ],
    }
)


@pytest.fixture
def small_tree():
    tree, _, _ = compile_plan(PLAN)
    return tree


@pytest.fixture
def backfilled_tree(small_tree):
    """模拟 Send 回填后的树：首页 banner 有了内容（深拷贝，别污染原树）。"""
    from shop_agent.schema import NodeStatus, PageContent

    tree = small_tree.model_copy(deep=True)
    page = tree.pages["p-001"]
    tree.pages["p-001"] = page.model_copy(
        update={
            "status": NodeStatus.DRAFT,
            "content": PageContent(
                headline="首页主标题",
                subheadline="副标题",
                components=[
                    {"type": "hero", "props": {"title": "主视觉"}},
                    {"type": "grid", "props": {"items": 6}},
                ],
            ),
        }
    )
    return tree


# ----------------------------------------------------------- 结构指标 --

def test_deterministic_metrics_counts(small_tree, backfilled_tree):
    # 全占位树（compile_plan 的产物）
    m = deterministic_metrics(small_tree)
    assert m["pages"] == 3
    assert m["zones"] == 2
    assert m["placeholders"] == 3          # 结构先行：全部占位
    assert m["dangling_links"] == 1        # 列表页链到占位详情页
    assert m["components"] == 0

    # 回填一页后：占位-1、组件+2，悬空链接不变（详情页仍是占位）
    m2 = deterministic_metrics(backfilled_tree)
    assert m2["placeholders"] == 2
    assert m2["components"] == 2
    assert m2["dangling_links"] == 1


# ------------------------------------------------------------ 渲染 --

def test_render_tree_for_judge(small_tree):
    text = render_tree_for_judge(small_tree)
    assert "测试小店" in text
    assert "商品列表页" in text
    assert "占位页：内容未生成" in text                    # 占位显式标注
    assert "商品详情页" in text                             # 跳转目标渲染为页面名
    assert "状态 placeholder" in text or "状态 PLACEHOLDER" in text

    msgs = build_judge_messages("卖咖啡", small_tree)
    assert msgs[0]["role"] == "system"
    assert "评分纪律" in msgs[0]["content"]
    assert "卖咖啡" in msgs[1]["content"]


# ------------------------------------------------------- 统计公式（手算向量）--

def test_pearson_known_vector():
    # 手算：x=[1,2,3,4,5], y=[1,2,3,4,4] -> r≈0.9701
    r = pearson([1, 2, 3, 4, 5], [1, 2, 3, 4, 4])
    assert round(r, 4) == 0.9701


def test_pearson_perfect_and_constant():
    assert pearson([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)
    assert pearson([3, 3, 3], [3, 3, 3]) == 1.0   # 双方零方差且全同
    assert pearson([3, 3, 3], [1, 2, 3]) == 0.0   # 一方零方差


def test_spearman_with_ties():
    # y 有并列（4,4）：平均秩 [1,2,3,4.5,4.5]，手算 rho≈0.9747
    rho = spearman([1, 2, 3, 4, 5], [1, 2, 3, 4, 4])
    assert round(rho, 4) == 0.9747


def test_qwk_known_vectors():
    # 完全一致 -> 1
    assert quadratic_weighted_kappa([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)
    # 手算复核：[1,2,3] vs [3,2,1]，num=2.0，den=1.0（含 4 个 1/12 的
    # 次对角项）-> kappa = 1 - 2 = -1。教训：手工推导别漏项。
    assert quadratic_weighted_kappa([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)


# ------------------------------------------------------------- 端到端 --

def test_run_eval_end_to_end_with_fakes(tmp_path):
    client, calls = make_eval_fake_client()
    payload = run(limit=2, out_dir=tmp_path, client=client)

    assert payload["n_tasks"] == 2
    ok = [r for r in payload["results"] if r["status"] == "ok"]
    assert len(ok) == 2
    for r in ok:
        assert set(r["judge"]) >= {"coverage", "copy_quality", "consistency", "conversion"}
        assert all(1 <= r["judge"][d] <= 5 for d in
                   ("coverage", "copy_quality", "consistency", "conversion"))
        assert r["metrics"]["placeholders"] == 0     # 假计划全部可生成
        assert r["metrics"]["dangling_links"] == 0
        assert r["tree"]["pages"]                     # 树快照可序列化

    # 产物三件套
    assert (tmp_path / "results.json").exists()
    annotation = (tmp_path / "annotation.csv").read_text(encoding="utf-8-sig")
    assert "human_score" in annotation
    assert "judge_score" not in annotation            # 盲标：不含 judge 分
    assert annotation.count("T0") == 2 * 4            # 2 任务 × 4 维度
    assert (tmp_path / "judge_scores.csv").exists()
    assert (tmp_path / "report.md").exists()

    # 真的走了生产路径：parse + plan + 4 页生成 + judge 都有调用
    assert len(calls) >= 2 * (1 + 1 + 4 + 1)

    # results.json 可被 json 反序列化（Enum 已转值）
    json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))


def test_run_eval_generation_failure_recorded(tmp_path, monkeypatch):
    """单任务异常不拖垮整场：记录后继续。"""
    import shop_agent.eval.run_eval as re_mod

    def _boom(requirement, thread_id):
        return None, "注入的生成失败"

    monkeypatch.setattr(re_mod, "_run_one_click", _boom)
    payload = run(limit=2, out_dir=tmp_path, client=None)
    assert all(r["status"] == "generation_error" for r in payload["results"])
    assert len(payload["results"]) == 2
