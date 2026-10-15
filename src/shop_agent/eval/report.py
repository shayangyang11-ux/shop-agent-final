"""评测汇总报告生成（report.md）。"""

from __future__ import annotations

import statistics
from pathlib import Path

from shop_agent.eval.rubric import JUDGED_DIMENSIONS


def _mean_sd(values: list[float]) -> tuple[float, str]:
    if not values:
        return 0.0, "-"
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return statistics.mean(values), f"{sd:.2f}"


def write_report(payload: dict, out_dir: Path) -> Path:
    results = payload["results"]
    ok = [r for r in results if "judge" in r]
    gen_fail = [r for r in results if r.get("status") != "ok"]

    lines = ["# shop-agent 生成质量评测报告", ""]
    lines += [
        "## 一、运行信息",
        "",
        f"- 任务数：{len(results)}（正常生成 {len(ok)}，异常 {len(gen_fail)}）",
        f"- 模型：{payload.get('model_info') or '见 .env 配置（SHOP_AGENT_LLM_MODEL）'}",
        "- 生成路径：生产图 one_click（Send 并行回填后由 assemble 验收）",
        "",
    ]

    if not ok:
        lines.append("没有可统计的任务，请先排查生成失败原因。")
        (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
        return out_dir / "report.md"

    lines += ["## 二、judge 评分总览", "", "| 维度 | 均值 | 标准差 |", "|---|---|---|"]
    for dim in JUDGED_DIMENSIONS:
        m, sd = _mean_sd([float(r["judge"][dim]) for r in ok])
        lines.append(f"| {dim} | {m:.2f} | {sd} |")
    total_placeholders = sum(r["metrics"]["placeholders"] for r in ok if "metrics" in r)
    total_dangling = sum(r["metrics"]["dangling_links"] for r in ok if "metrics" in r)
    lines += [
        "",
        "## 三、确定性结构指标",
        "",
        f"- 占位页总数：{total_placeholders}（一键模式验收要求为 0）",
        f"- 悬空链接总数：{total_dangling}（一键模式验收要求为 0）",
        f"- 平均页面数：{statistics.mean([r['metrics']['pages'] for r in ok if 'metrics' in r]):.1f}",
        f"- 平均组件数：{statistics.mean([r['metrics']['components'] for r in ok if 'metrics' in r]):.1f}",
        "",
        "## 四、任务明细",
        "",
        "| 任务 | 需求摘要 | coverage | copy | consist | convert | 占位 | 悬空 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in ok:
        brief = r["requirement"][:18] + ("…" if len(r["requirement"]) > 18 else "")
        j = r["judge"]
        m = r.get("metrics", {})
        lines.append(
            f"| {r['task_id']} | {brief} | {j['coverage']} | {j['copy_quality']} "
            f"| {j['consistency']} | {j['conversion']} | {m.get('placeholders', '-')} "
            f"| {m.get('dangling_links', '-')} |"
        )

    if gen_fail:
        lines += ["", "## 五、异常任务", ""]
        for r in gen_fail:
            lines.append(f"- {r['task_id']}（{r.get('status')}）：{r.get('error', '')}")

    lines += [
        "",
        "## 五、下一步：人工标注与一致率",
        "",
        "1. 打开 `annotation.csv`，对照 `results.json` 里各任务的店铺树，",
        "   按 README 的锚点描述给 human_score 填 1-5 分（盲标：别看 judge_scores.csv）。",
        "2. 执行 `python -m shop_agent.eval agreement annotation.csv`，",
        "   得到 Pearson / Spearman / QWK 一致率与人工复核清单。",
    ]
    path = out_dir / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
