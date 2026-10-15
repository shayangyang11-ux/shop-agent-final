"""评测 CLI 入口：python -m shop_agent.eval ...

命令：
  run        跑完整评测（生成 -> 指标 -> judge -> 落盘）
  agreement  人工标注完成后，算 judge-人工一致率
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console
from rich.table import Table

from shop_agent.eval.agreement import analyze, format_agreement_markdown

console = Console()


def _cmd_run(args: argparse.Namespace) -> int:
    from shop_agent.eval.run_eval import run

    client = None
    if args.dry_run:
        from shop_agent.eval._fakes import make_eval_fake_client

        client, _ = make_eval_fake_client()
        console.print("[yellow]DRY-RUN：使用角色感知假客户端，零 API 费用[/yellow]")

    payload = run(limit=args.limit, out_dir=Path(args.out), client=client)

    table = Table(title="评测结果总览")
    table.add_column("任务", style="cyan")
    for col in ("coverage", "copy", "consist", "convert", "占位", "悬空"):
        table.add_column(col, justify="center")
    for r in payload["results"]:
        if "judge" not in r:
            table.add_row(r["task_id"], *[r.get("status", "?")] + [""] * 5)
            continue
        j, m = r["judge"], r.get("metrics", {})
        table.add_row(
            r["task_id"], str(j["coverage"]), str(j["copy_quality"]),
            str(j["consistency"]), str(j["conversion"]),
            str(m.get("placeholders", "-")), str(m.get("dangling_links", "-")),
        )
    console.print(table)
    console.print(f"产物目录：{Path(args.out).resolve()}")
    console.print("下一步：人工标注 annotation.csv，再运行 agreement 子命令。")
    return 0


def _cmd_agreement(args: argparse.Namespace) -> int:
    analysis = analyze(args.results, args.annotation)
    console.print(format_agreement_markdown(analysis))
    out = Path(args.out or Path(args.annotation).parent / "agreement.md")
    out.write_text(format_agreement_markdown(analysis), encoding="utf-8")
    console.print(f"已写入：{out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m shop_agent.eval")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="跑完整评测")
    p_run.add_argument("--limit", type=int, default=None, help="只跑前 N 个任务")
    p_run.add_argument("--out", default="eval_results", help="产物目录")
    p_run.add_argument("--dry-run", action="store_true", help="假客户端零费用验证管线")
    p_run.set_defaults(func=_cmd_run)

    p_agr = sub.add_parser("agreement", help="judge-人工一致率分析")
    p_agr.add_argument("annotation", help="已填写 human_score 的 annotation.csv")
    p_agr.add_argument("--results", default="eval_results/results.json")
    p_agr.add_argument("--out", default=None, help="报告输出路径")
    p_agr.set_defaults(func=_cmd_agreement)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
