"""judge-人工一致率分析（纯 Python 实现，零科学计算栈依赖）。

三个指标各答一个问题：

- Pearson r：分数的线性一致性 —— judge 和人是否在同一个标尺上打分
- Spearman rho：秩一致性 —— 就算标尺有系统偏移（judge 整体偏严），
  排序对不对？排序对 = 可以用 judge 筛"最差的 20%"做回归门禁
- 二次加权 kappa（QWK）：有序评分的标准一致率 —— 错 1 档的惩罚
  远小于错 4 档（线性加权），且对偶然一致做了校正（随机打分也会
  撞对一些，kappa 把这部分扣掉）

判读基准（人工评测的通行惯例，报告里会引用）：
  QWK >= 0.8 强一致 / 0.61-0.80 中等偏上 / 0.41-0.60 中等 / < 0.41 弱
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from shop_agent.eval.rubric import DIMENSIONS, JUDGED_DIMENSIONS


# --------------------------------------------------------------- 基础统计 --
def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def _variance(xs: list[float]) -> float:
    m = _mean(xs)
    return sum((x - m) ** 2 for x in xs) / len(xs)


def pearson(xs: list[float], ys: list[float]) -> float:
    """线性相关系数。任一方零方差：全同则 1.0，否则 0.0（约定）。"""
    if len(xs) != len(ys) or not xs:
        raise ValueError("pearson 需要等长非空序列")
    vx, vy = _variance(xs), _variance(ys)
    if vx == 0 or vy == 0:
        return 1.0 if xs == ys else 0.0
    mx, my = _mean(xs), _mean(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / len(xs)
    return cov / math.sqrt(vx * vy)


def _ranks(xs: list[float]) -> list[float]:
    """平均秩（ties 各取均值）—— Spearman 对并列分数的正确处理。"""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1  # 秩从 1 开始
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(xs: list[float], ys: list[float]) -> float:
    return pearson(_ranks(xs), _ranks(ys))


def quadratic_weighted_kappa(
    xs: list[int], ys: list[int], low: int = 1, high: int = 5
) -> float:
    """QWK：1 = 完全一致；0 = 偶然水平；可到负值（系统性反着打）。"""
    if len(xs) != len(ys) or not xs:
        raise ValueError("kappa 需要等长非空序列")
    k = high - low + 1
    n = len(xs)
    obs = [[0] * k for _ in range(k)]
    for x, y in zip(xs, ys):
        obs[x - low][y - low] += 1
    row = [sum(r) for r in obs]
    col = [sum(obs[i][j] for i in range(k)) for j in range(k)]
    num = den = 0.0
    for i in range(k):
        for j in range(k):
            w = (i - j) ** 2 / (k - 1) ** 2
            num += w * obs[i][j]
            den += w * row[i] * col[j] / n
    if den == 0:
        return 1.0
    return 1.0 - num / den


def _round1(x: float) -> float:
    return round(x, 3)


# ------------------------------------------------------------- 对齐与汇总 --
def load_annotation(csv_path: Path | str) -> dict[tuple[str, str], int]:
    """读人工标注 CSV -> {(task_id, dimension): human_score}。"""
    scores: dict[tuple[str, str], int] = {}
    with Path(csv_path).open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            raw = (row.get("human_score") or "").strip()
            if not raw:
                continue
            scores[(row["task_id"].strip(), row["dimension"].strip())] = int(float(raw))
    return scores


def collect_pairs(
    results: dict, annotation: dict[tuple[str, str], int]
) -> dict[str, list[tuple[int, int, str]]]:
    """按维度对齐 (judge, human, task_id) 三元组。"""
    pairs: dict[str, list[tuple[int, int, str]]] = {d: [] for d in JUDGED_DIMENSIONS}
    for r in results.get("results", []):
        if "judge" not in r:
            continue
        for dim in JUDGED_DIMENSIONS:
            human = annotation.get((r["task_id"], dim))
            if human is not None:
                pairs[dim].append((r["judge"][dim], human, r["task_id"]))
    return pairs


def agreement_stats(judge: list[int], human: list[int]) -> dict:
    exact = sum(1 for j, h in zip(judge, human) if j == h)
    within1 = sum(1 for j, h in zip(judge, human) if abs(j - h) <= 1)
    n = len(judge)
    return {
        "n": n,
        "pearson": _round1(pearson([float(x) for x in judge], [float(y) for y in human])),
        "spearman": _round1(spearman([float(x) for x in judge], [float(y) for y in human])),
        "qwk": _round1(quadratic_weighted_kappa(judge, human)),
        "exact_rate": _round1(exact / n),
        "within1_rate": _round1(within1 / n),
    }


def analyze(results_path: Path | str, annotation_csv: Path | str) -> dict:
    """完整一致率分析：逐维度 + 总体 + 最大分歧对（人工复核清单）。"""
    payload = json.loads(Path(results_path).read_text(encoding="utf-8"))
    annotation = load_annotation(annotation_csv)
    pairs = collect_pairs(payload, annotation)

    report: dict = {"per_dimension": {}, "overall": None, "worst_gaps": []}
    all_j, all_h = [], []
    for dim, triple in pairs.items():
        if not triple:
            continue
        js = [t[0] for t in triple]
        hs = [t[1] for t in triple]
        report["per_dimension"][dim] = {
            "dimension_cn": DIMENSIONS[dim],
            **agreement_stats(js, hs),
        }
        all_j.extend(js)
        all_h.extend(hs)

    if all_j:
        report["overall"] = agreement_stats(all_j, all_h)

    gaps = []
    for dim, triple in pairs.items():
        for j, h, tid in triple:
            gaps.append((abs(j - h), dim, tid, j, h))
    gaps.sort(reverse=True)
    report["worst_gaps"] = [
        {"task_id": tid, "dimension": dim, "judge": j, "human": h, "gap": g}
        for g, dim, tid, j, h in gaps[:10]
        if g >= 2  # 只列 2 分以上的分歧，1 分以内在噪声范围
    ]
    return report


def format_agreement_markdown(analysis: dict) -> str:
    lines = ["# judge-人工一致率报告", ""]
    if analysis["overall"] is None:
        lines.append("没有可对齐的标注对。请检查 annotation.csv 是否已填写 human_score。")
        return "\n".join(lines)

    o = analysis["overall"]
    lines += [
        "## 总体",
        "",
        f"- 样本对数：{o['n']}",
        f"- Pearson r = {o['pearson']} / Spearman rho = {o['spearman']} / QWK = {o['qwk']}",
        f"- 完全一致率 {o['exact_rate']:.0%}，±1 档内 {o['within1_rate']:.0%}",
        "",
        "判读：QWK ≥ 0.8 强一致；0.61-0.80 中等偏上；0.41-0.60 中等；< 0.41 弱。",
        "",
        "## 分维度",
        "",
        "| 维度 | n | Pearson | Spearman | QWK | 完全一致 | ±1 档内 |",
        "|---|---|---|---|---|---|---|",
    ]
    for dim, s in analysis["per_dimension"].items():
        lines.append(
            f"| {s['dimension_cn']} ({dim}) | {s['n']} | {s['pearson']} | {s['spearman']} "
            f"| {s['qwk']} | {s['exact_rate']:.0%} | {s['within1_rate']:.0%} |"
        )
    if analysis["worst_gaps"]:
        lines += [
            "",
            "## 人工复核清单（分歧 ≥ 2 分）",
            "",
            "| 任务 | 维度 | judge | 人工 | 差 |",
            "|---|---|---|---|---|",
        ]
        for g in analysis["worst_gaps"]:
            lines.append(
                f"| {g['task_id']} | {g['dimension']} | {g['judge']} | {g['human']} "
                f"| {g['judge'] - g['human']:+d} |"
            )
        lines += [
            "",
            "复核这些任务可定位分歧来源：是锚点写得模糊、judge 偏严/偏松，",
            "还是生成本身处于『可上可下』的边界区间。",
        ]
    return "\n".join(lines)
