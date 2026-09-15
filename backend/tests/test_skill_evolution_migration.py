from __future__ import annotations

import pytest
from sqlalchemy import inspect, text
from sqlalchemy.dialects import postgresql
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.schema import CreateIndex, CreateTable

from deerflow.persistence.base import Base
from deerflow.persistence.engine import (
    close_engine,
    get_engine,
    init_engine,
)
from deerflow.persistence.skill_evolution.model import (
    SkillEvolutionClusterRow,
    SkillEvolutionCreditRow,
    SkillEvolutionEvaluationRow,
    SkillEvolutionEventRow,
    SkillEvolutionJobRow,
    SkillEvolutionProposalRow,
    SkillEvolutionPublicationRow,
)

_EXPECTED_TABLES = {
    "skill_evolution_events",
    "skill_evolution_clusters",
    "skill_evolution_proposals",
    "skill_evolution_evaluations",
    "skill_evolution_publications",
    "skill_evolution_jobs",
    "skill_evolution_credits",
}

_EXPECTED_INDEXES = {
    "skill_evolution_events": {
        "ix_skill_evolution_events_user_created",
        "ix_skill_evolution_events_user_kind",
        "ix_skill_evolution_events_user_task_signature",
        "ix_skill_evolution_events_user_target_skill",
    },
    "skill_evolution_clusters": {
        "ix_skill_evolution_clusters_user_status",
        "ix_skill_evolution_clusters_user_target_skill",
        "ix_skill_evolution_clusters_user_updated",
    },
    "skill_evolution_proposals": {
        "ix_skill_evolution_proposals_user_status",
        "ix_skill_evolution_proposals_user_skill",
        "ix_skill_evolution_proposals_user_cluster",
        "ix_skill_evolution_proposals_user_created",
    },
    "skill_evolution_evaluations": {
        "ix_skill_evolution_evaluations_user_proposal",
        "ix_skill_evolution_evaluations_user_decision",
        "ix_skill_evolution_evaluations_user_created",
    },
    "skill_evolution_publications": {
        "ix_skill_evolution_publications_user_status",
        "ix_skill_evolution_publications_user_skill",
        "ix_skill_evolution_publications_user_updated",
    },
    "skill_evolution_jobs": {
        "ix_skill_evolution_jobs_status_due",
        "ix_skill_evolution_jobs_status_lease",
        "ix_skill_evolution_jobs_user_run",
    },
    "skill_evolution_credits": {
        "ix_skill_evolution_credits_user_kind_created",
        "ix_skill_evolution_credits_user_skill_created",
        "ix_skill_evolution_credits_user_publication",
    },
}

_ORM_TABLES = [
    SkillEvolutionEventRow.__table__,
    SkillEvolutionClusterRow.__table__,
    SkillEvolutionProposalRow.__table__,
    SkillEvolutionEvaluationRow.__table__,
    SkillEvolutionPublicationRow.__table__,
    SkillEvolutionJobRow.__table__,
    SkillEvolutionCreditRow.__table__,
]


@pytest.mark.parametrize("table", _ORM_TABLES, ids=lambda table: table.name)
def test_skill_evolution_schema_compiles_for_postgres(table) -> None:
    dialect = postgresql.dialect()

    table_ddl = str(CreateTable(table).compile(dialect=dialect))
    index_ddl = [str(CreateIndex(index).compile(dialect=dialect)) for index in table.indexes]

    assert f"CREATE TABLE {table.name}" in table_ddl
    assert "JSON" in table_ddl
    assert all("CREATE INDEX" in statement for statement in index_ddl)


async def _seed_revision(database_url: str, revision: str) -> None:
    engine = create_async_engine(database_url)
    try:
        async with engine.begin() as connection:
            await connection.execute(text("CREATE TABLE alembic_version (version_num VARCHAR(64) NOT NULL PRIMARY KEY)"))
            await connection.execute(
                text("INSERT INTO alembic_version(version_num) VALUES (:revision)"),
                {"revision": revision},
            )
    finally:
        await engine.dispose()


async def _seed_0012_schema(database_url: str) -> None:
    engine = create_async_engine(database_url)
    try:
        async with engine.begin() as connection:
            await connection.run_sync(
                Base.metadata.create_all,
                tables=[
                    SkillEvolutionEventRow.__table__,
                    SkillEvolutionClusterRow.__table__,
                    SkillEvolutionProposalRow.__table__,
                    SkillEvolutionEvaluationRow.__table__,
                ],
            )
            await connection.execute(text("CREATE TABLE alembic_version (version_num VARCHAR(64) NOT NULL PRIMARY KEY)"))
            await connection.execute(text("INSERT INTO alembic_version(version_num) VALUES ('0012_skill_evolution')"))
    finally:
        await engine.dispose()


async def _seed_0013_schema(database_url: str) -> None:
    engine = create_async_engine(database_url)
    try:
        async with engine.begin() as connection:
            await connection.run_sync(
                Base.metadata.create_all,
                tables=[
                    SkillEvolutionEventRow.__table__,
                    SkillEvolutionClusterRow.__table__,
                    SkillEvolutionProposalRow.__table__,
                    SkillEvolutionEvaluationRow.__table__,
                    SkillEvolutionPublicationRow.__table__,
                ],
            )
            await connection.execute(text("CREATE TABLE alembic_version (version_num VARCHAR(64) NOT NULL PRIMARY KEY)"))
            await connection.execute(text("INSERT INTO alembic_version(version_num) VALUES ('0013_skill_publications')"))
    finally:
        await engine.dispose()


async def _seed_0014_schema(database_url: str) -> None:
    engine = create_async_engine(database_url)
    try:
        async with engine.begin() as connection:
            await connection.run_sync(
                Base.metadata.create_all,
                tables=[
                    SkillEvolutionEventRow.__table__,
                    SkillEvolutionClusterRow.__table__,
                    SkillEvolutionProposalRow.__table__,
                    SkillEvolutionEvaluationRow.__table__,
                    SkillEvolutionPublicationRow.__table__,
                    SkillEvolutionJobRow.__table__,
                ],
            )
            await connection.execute(text("CREATE TABLE alembic_version (version_num VARCHAR(64) NOT NULL PRIMARY KEY)"))
            await connection.execute(text("INSERT INTO alembic_version(version_num) VALUES ('0014_skill_evolution_jobs')"))
    finally:
        await engine.dispose()


async def _inspect_schema(connection):
    def _inspect(sync_connection):
        inspector = inspect(sync_connection)
        tables = set(inspector.get_table_names())
        indexes = {table: {item["name"] for item in inspector.get_indexes(table) if item.get("name")} for table in _EXPECTED_TABLES if table in tables}
        return tables, indexes

    return await connection.run_sync(_inspect)


@pytest.mark.asyncio
async def test_upgrade_from_0011_creates_skill_evolution_and_publication_schema(
    tmp_path,
) -> None:
    database_path = tmp_path / "skill-evolution-migration.db"
    database_url = f"sqlite+aiosqlite:///{database_path}"
    await _seed_revision(database_url, "0011_mcp_tasks")

    try:
        await init_engine(
            "sqlite",
            url=database_url,
            sqlite_dir=str(tmp_path),
        )
        engine = get_engine()
        assert engine is not None
        async with engine.connect() as connection:
            tables, indexes = await _inspect_schema(connection)
            revision = await connection.scalar(text("SELECT version_num FROM alembic_version"))

        assert revision == "0016_direct_skill_publication"
        assert _EXPECTED_TABLES <= tables
        for table, expected in _EXPECTED_INDEXES.items():
            assert expected <= indexes[table]
        async with engine.connect() as connection:
            evaluation_column = await connection.run_sync(lambda sync_connection: next(column for column in inspect(sync_connection).get_columns("skill_evolution_publications") if column["name"] == "evaluation_id"))
        assert evaluation_column["nullable"] is True
    finally:
        await close_engine()


@pytest.mark.asyncio
async def test_upgrade_from_0012_adds_publication_schema(
    tmp_path,
) -> None:
    database_path = tmp_path / "skill-publication-migration.db"
    database_url = f"sqlite+aiosqlite:///{database_path}"
    await _seed_0012_schema(database_url)

    try:
        await init_engine(
            "sqlite",
            url=database_url,
            sqlite_dir=str(tmp_path),
        )
        engine = get_engine()
        assert engine is not None
        async with engine.connect() as connection:
            tables, indexes = await _inspect_schema(connection)
            revision = await connection.scalar(text("SELECT version_num FROM alembic_version"))

        assert revision == "0016_direct_skill_publication"
        assert "skill_evolution_publications" in tables
        assert "skill_evolution_jobs" in tables
        assert _EXPECTED_INDEXES["skill_evolution_publications"] <= indexes["skill_evolution_publications"]
        assert _EXPECTED_INDEXES["skill_evolution_jobs"] <= indexes["skill_evolution_jobs"]
    finally:
        await close_engine()


@pytest.mark.asyncio
async def test_upgrade_from_0013_adds_durable_job_schema(
    tmp_path,
) -> None:
    database_path = tmp_path / "skill-evolution-job-migration.db"
    database_url = f"sqlite+aiosqlite:///{database_path}"
    await _seed_0013_schema(database_url)

    try:
        await init_engine(
            "sqlite",
            url=database_url,
            sqlite_dir=str(tmp_path),
        )
        engine = get_engine()
        assert engine is not None
        async with engine.connect() as connection:
            tables, indexes = await _inspect_schema(connection)
            revision = await connection.scalar(text("SELECT version_num FROM alembic_version"))

        assert revision == "0016_direct_skill_publication"
        assert "skill_evolution_jobs" in tables
        assert _EXPECTED_INDEXES["skill_evolution_jobs"] <= indexes["skill_evolution_jobs"]
    finally:
        await close_engine()


@pytest.mark.asyncio
async def test_upgrade_from_0014_adds_credit_schema(
    tmp_path,
) -> None:
    database_path = tmp_path / "skill-credit-migration.db"
    database_url = f"sqlite+aiosqlite:///{database_path}"
    await _seed_0014_schema(database_url)

    try:
        await init_engine(
            "sqlite",
            url=database_url,
            sqlite_dir=str(tmp_path),
        )
        engine = get_engine()
        assert engine is not None
        async with engine.connect() as connection:
            tables, indexes = await _inspect_schema(connection)
            revision = await connection.scalar(text("SELECT version_num FROM alembic_version"))

        assert revision == "0016_direct_skill_publication"
        assert "skill_evolution_credits" in tables
        assert _EXPECTED_INDEXES["skill_evolution_credits"] <= indexes["skill_evolution_credits"]
    finally:
        await close_engine()
