"""Run the frozen Skill-evolution experiment with an external executor."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
from pathlib import Path

from deerflow.skill_evolution.experiment import (
    CohortExperimentExecutor,
    ExperimentCondition,
    ExperimentExecutor,
    build_default_conditions,
    build_default_manifest,
    read_results_jsonl,
    run_experiment,
    write_results_jsonl,
)


def _resolve_executor(
    path: str,
) -> ExperimentExecutor | CohortExperimentExecutor:
    module_name, separator, attribute_name = path.partition(":")
    if not separator or not module_name or not attribute_name:
        raise ValueError("executor must use module.path:factory format")
    module = importlib.import_module(module_name)
    factory = getattr(module, attribute_name)
    executor = factory()
    execute = getattr(executor, "execute", None)
    execute_cohort = getattr(executor, "execute_cohort", None)
    if not callable(execute) and not callable(execute_cohort):
        raise TypeError("experiment executor must define async execute() or execute_cohort()")
    return executor


def _parse_seeds(raw: str) -> list[int]:
    try:
        seeds = [int(item.strip()) for item in raw.split(",")]
    except ValueError as exc:
        raise ValueError("seeds must be comma-separated integers") from exc
    if not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError("seeds must be unique non-negative integers")
    return seeds


def _select_conditions(raw: str) -> list[ExperimentCondition]:
    available = {condition.condition_id: condition for condition in build_default_conditions()}
    if raw == "all":
        return list(available.values())
    requested = [item.strip() for item in raw.split(",") if item.strip()]
    missing = sorted(set(requested) - set(available))
    if missing:
        raise ValueError(f"unknown conditions: {', '.join(missing)}")
    if not requested:
        raise ValueError("at least one condition is required")
    return [available[item] for item in requested]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=("Run the frozen 60-task Skill-evolution benchmark. A production Replay executor is required unless --validate-only is used."))
    parser.add_argument(
        "--experiment-id",
        default="skill-evolution-phase11",
    )
    parser.add_argument(
        "--executor",
        help="External executor factory as module.path:factory",
    )
    parser.add_argument(
        "--conditions",
        default="all",
        help="Comma-separated condition IDs or 'all'",
    )
    parser.add_argument(
        "--seeds",
        default="1,2,3",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".deer-flow/benchmarks/skill-evolution/results.jsonl"),
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        help=("Run metadata sidecar. Defaults to <output>.metadata.json."),
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=("Resume complete family cohorts from an existing output JSONL and checkpoint each newly completed cohort."),
    )
    return parser


def _write_metadata(
    path: Path,
    payload: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_name(
        f".{path.name}.tmp",
    )
    staged.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    staged.replace(path)


async def _run(args: argparse.Namespace) -> int:
    manifest = build_default_manifest()
    conditions = _select_conditions(args.conditions)
    seeds = _parse_seeds(args.seeds)
    if args.manifest_output is not None:
        args.manifest_output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        args.manifest_output.write_text(
            json.dumps(
                manifest.model_dump(mode="json"),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    planned = len(manifest.tasks) * len(conditions) * len(seeds)
    if args.validate_only:
        print(
            json.dumps(
                {
                    "manifest_id": manifest.manifest_id,
                    "tasks": len(manifest.tasks),
                    "conditions": len(conditions),
                    "seeds": seeds,
                    "planned_results": planned,
                    "executed": False,
                },
                sort_keys=True,
            )
        )
        return 0
    if args.executor is None:
        raise ValueError("--executor is required for a real experiment run")
    executor = _resolve_executor(args.executor)
    metadata_output = args.metadata_output if args.metadata_output is not None else args.output.with_suffix(args.output.suffix + ".metadata.json")
    existing = read_results_jsonl(args.output) if args.resume and args.output.is_file() else []
    result_order = {
        (
            condition.condition_id,
            task.task_id,
            seed,
        ): index
        for index, (
            condition,
            task,
            seed,
        ) in enumerate(
            (
                condition,
                task,
                seed,
            )
            for condition in conditions
            for task in manifest.tasks
            for seed in seeds
        )
    }
    checkpointed = {
        (
            row.condition_id,
            row.task_id,
            row.seed,
        ): row
        for row in existing
    }

    def ordered_rows():
        return sorted(
            checkpointed.values(),
            key=lambda row: result_order.get(
                (
                    row.condition_id,
                    row.task_id,
                    row.seed,
                ),
                planned,
            ),
        )

    def metadata_payload(
        *,
        run_status: str,
    ) -> dict:
        return {
            "schema_version": ("deerflow.skill-evolution.experiment-run-metadata.v1"),
            "experiment_id": args.experiment_id,
            "manifest": manifest.model_dump(mode="json"),
            "conditions": [condition.model_dump(mode="json") for condition in conditions],
            "seeds": seeds,
            "executor": args.executor,
            "max_concurrency": args.max_concurrency,
            "result_count": len(checkpointed),
            "planned_result_count": planned,
            "run_status": run_status,
            "results_path": str(args.output),
        }

    if not args.resume:
        checkpointed.clear()
        write_results_jsonl(
            args.output,
            [],
        )
    _write_metadata(
        metadata_output,
        metadata_payload(run_status="in_progress"),
    )

    async def checkpoint(
        completed,
    ) -> None:
        for row in completed:
            checkpointed[
                (
                    row.condition_id,
                    row.task_id,
                    row.seed,
                )
            ] = row
        write_results_jsonl(
            args.output,
            ordered_rows(),
        )
        _write_metadata(
            metadata_output,
            metadata_payload(run_status="in_progress"),
        )

    rows = await run_experiment(
        experiment_id=args.experiment_id,
        manifest=manifest,
        conditions=conditions,
        seeds=seeds,
        executor=executor,
        max_concurrency=args.max_concurrency,
        existing_results=existing,
        on_results_completed=checkpoint,
    )
    write_results_jsonl(args.output, rows)
    checkpointed.clear()
    checkpointed.update(
        {
            (
                row.condition_id,
                row.task_id,
                row.seed,
            ): row
            for row in rows
        }
    )
    _write_metadata(
        metadata_output,
        metadata_payload(run_status="completed"),
    )
    print(
        json.dumps(
            {
                "experiment_id": args.experiment_id,
                "metadata": str(metadata_output),
                "results": len(rows),
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


def main() -> int:
    args = _parser().parse_args()
    try:
        return asyncio.run(_run(args))
    except (ImportError, AttributeError, TypeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    raise SystemExit(main())
