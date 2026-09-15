"""Summarize Skill-evolution JSONL results with paired statistics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from deerflow.skill_evolution.experiment import (
    build_experiment_report,
    read_results_jsonl,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=("Build aggregate and per-family Skill-evolution metrics with paired bootstrap intervals and McNemar."))
    parser.add_argument("results", type=Path)
    parser.add_argument(
        "--reference",
        default="baseline-no-evolution",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=10_000,
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".deer-flow/benchmarks/skill-evolution/report.json"),
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        rows = read_results_jsonl(args.results)
        report = build_experiment_report(
            rows,
            reference_condition_id=args.reference,
            bootstrap_samples=args.bootstrap_samples,
            random_seed=args.random_seed,
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            report.model_dump(mode="json"),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "experiment_id": report.experiment_id,
                "conditions": len(report.aggregate),
                "families": len(report.by_family),
                "comparisons": len(report.comparisons),
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
