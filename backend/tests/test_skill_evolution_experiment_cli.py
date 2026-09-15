from __future__ import annotations

import sys
from types import ModuleType

from scripts.benchmark.skill_evolution.run import (
    _parser,
    _resolve_executor,
)


def test_cli_accepts_cohort_only_executor(
    monkeypatch,
) -> None:
    module = ModuleType("test_cohort_executor")

    class CohortExecutor:
        async def execute_cohort(
            self,
            tasks,
            condition,
            seed,
        ):
            raise NotImplementedError

    executor = CohortExecutor()
    module.create_executor = lambda: executor
    monkeypatch.setitem(
        sys.modules,
        module.__name__,
        module,
    )

    assert (
        _resolve_executor(
            "test_cohort_executor:create_executor",
        )
        is executor
    )


def test_cli_exposes_resume_flag() -> None:
    args = _parser().parse_args(
        [
            "--resume",
        ]
    )

    assert args.resume is True
