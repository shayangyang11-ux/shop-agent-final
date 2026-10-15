"""评测编排：跑生产图生成整店 -> 确定性指标 + judge 裁决 -> 落盘产物。

生产路径复用（不是旁路复刻）：评测直接驱动 build_graph() 真图，
parse_input -> interrupt 菜单 -> Command(resume="one_click") ->
fan_out_pages Send 并行 -> assemble。评测看到的生成质量和真实用户
看到的完全一致 —— 旁路复刻一条"评测专用简化链路"是评测体系的
经典反模式（评了个寂寞）。

产物（写入 out_dir，默认 eval_results/）：
  results.json        全量结果（含每任务店铺树快照，供复查与 debug）
  annotation.csv      人工盲标模板（不含 judge 分数）
  judge_scores.csv    judge 分数（标注完成前别看，保持盲评）
  report.md           汇总报告
"""

from __future__ import annotations

import csv
import json
import logging
from contextlib import nullcontext
from pathlib import Path
from typing import Optional
from unittest.mock import patch

from shop_agent.eval.rubric import DIMENSIONS, JUDGED_DIMENSIONS, JudgeVerdict
from shop_agent.eval.tasks import EVAL_TASKS

logger = logging.getLogger(__name__)


def deterministic_metrics(tree) -> dict:
    """代码可判定的结构指标 —— 不进 judge，直接进结果集。"""
    components = sum(
        len(p.content.components) for p in tree.pages.values() if p.content is not None
    )
    return {
        "pages": len(tree.pages),
        "zones": len(tree.zones),
        "placeholders": len(tree.placeholder_page_ids()),
        "dangling_links": len(tree.dangling_links()),
        "components": components,
    }


def _run_one_click(requirement: str, thread_id: str):
    """驱动生产图走 one_click 全流程，返回 (tree, error)。"""
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.types import Command

    from shop_agent.graph import build_graph

    graph = build_graph(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": thread_id}}
    first = graph.invoke({"user_input": requirement}, config)
    if first.get("error"):
        return None, first["error"]
    final = graph.invoke(Command(resume="one_click"), config)
    tree = final.get("shop_tree")
    return tree, final.get("error")


def run(
    *,
    limit: Optional[int] = None,
    out_dir: Path | str = Path("eval_results"),
    client: Optional[object] = None,
    model_info: str = "",
) -> dict:
    """跑评测。client 传入假客户端 = 零成本管线自检（--dry-run）。

    真跑（client=None）时 judge 与生产节点都走 get_client()；
    传 client 时统一替换 —— 图节点经 patch(get_client)，judge 经
    参数直传，两条注入缝殊途同归。
    """
    from shop_agent.eval.judge import judge_tree
    from shop_agent.eval.report import write_report

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    tasks = EVAL_TASKS[: limit] if limit else EVAL_TASKS
    results: list[dict] = []

    patch_ctx = (
        patch("shop_agent.llm.get_client", lambda: client) if client is not None else nullcontext()
    )
    with patch_ctx:
        for i, task in enumerate(tasks, 1):
            tid = task["id"]
            record: dict = {"task_id": tid, "requirement": task["requirement"], "note": task["note"]}
            try:
                tree, error = _run_one_click(task["requirement"], f"eval-{tid}")
                if error or tree is None:
                    record["status"] = "generation_error"
                    record["error"] = error or "no tree"
                else:
                    record["status"] = "ok"
                    record["metrics"] = deterministic_metrics(tree)
                    record["tree"] = tree.model_dump(mode="json")
                    verdict: JudgeVerdict = judge_tree(task["requirement"], tree, client=client)
                    record["judge"] = verdict.model_dump()
            except Exception as err:  # noqa: BLE001 - 单任务失败不能拖垮整场评测
                logger.warning("task %s failed: %r", tid, err)
                record["status"] = "exception"
                record["error"] = f"{type(err).__name__}: {err}"
            results.append(record)
            done = f"[{record['status']}]"
            scores = (
                " ".join(f"{d}={record['judge'][d]}" for d in JUDGED_DIMENSIONS)
                if "judge" in record
                else ""
            )
            print(f"({i}/{len(tasks)}) {tid} {done} {scores}", flush=True)

    payload = {
        "n_tasks": len(results),
        "model_info": model_info,
        "results": results,
    }
    _write_annotation_csv(out, results)
    _write_judge_scores_csv(out, results)
    (out / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(payload, out)
    return payload


def _write_annotation_csv(out: Path, results: list[dict]) -> Path:
    """人工盲标模板：judge 分数绝不出现，防锚定。"""
    path = out / "annotation.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["task_id", "dimension", "dimension_cn", "human_score", "note"])
        for r in results:
            if r["status"] != "ok":
                continue
            for dim in JUDGED_DIMENSIONS:
                writer.writerow([r["task_id"], dim, DIMENSIONS[dim], "", r["note"]])
    return path


def _write_judge_scores_csv(out: Path, results: list[dict]) -> Path:
    path = out / "judge_scores.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["task_id", "dimension", "dimension_cn", "judge_score"])
        for r in results:
            if "judge" not in r:
                continue
            for dim in JUDGED_DIMENSIONS:
                writer.writerow([r["task_id"], dim, DIMENSIONS[dim], r["judge"][dim]])
    return path
