"""Версионированные миграции схемы Postgres."""

from __future__ import annotations

from typing import Callable, List, Tuple

MigrationFn = Callable[[object], None]

MIGRATIONS: List[Tuple[int, MigrationFn]] = []


def _migration(version: int):
    def decorator(fn: MigrationFn) -> MigrationFn:
        MIGRATIONS.append((version, fn))
        MIGRATIONS.sort(key=lambda item: item[0])
        return fn

    return decorator


@_migration(1)
def _v1_benchmark_runs(cur) -> None:
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS benchmark_runs (
            id UUID PRIMARY KEY,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            status VARCHAR(32) NOT NULL DEFAULT 'queued',
            progress_done INTEGER NOT NULL DEFAULT 0,
            progress_total INTEGER NOT NULL DEFAULT 0,
            avg_accuracy DOUBLE PRECISION,
            config JSONB NOT NULL DEFAULT '{}'::jsonb,
            input_key TEXT NOT NULL,
            results_key TEXT,
            error TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_benchmark_runs_created
            ON benchmark_runs (created_at DESC);
        """
    )


@_migration(2)
def _v2_benchmark_run_metadata(cur) -> None:
    cur.execute(
        "ALTER TABLE benchmark_runs ADD COLUMN IF NOT EXISTS run_title TEXT NOT NULL DEFAULT ''"
    )
    cur.execute(
        "ALTER TABLE benchmark_runs ADD COLUMN IF NOT EXISTS run_description TEXT NOT NULL DEFAULT ''"
    )


@_migration(3)
def _v3_judge_presets(cur) -> None:
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS judge_presets (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            name TEXT NOT NULL,
            description TEXT NOT NULL DEFAULT '',
            config JSONB NOT NULL DEFAULT '{}'::jsonb,
            is_default BOOLEAN NOT NULL DEFAULT FALSE,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        CREATE UNIQUE INDEX IF NOT EXISTS idx_judge_presets_name ON judge_presets (name);
        """
    )


@_migration(4)
def _v4_judge_samples(cur) -> None:
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS judge_samples (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            name TEXT NOT NULL,
            description TEXT NOT NULL DEFAULT '',
            case_count INTEGER NOT NULL DEFAULT 0,
            annotated_count INTEGER NOT NULL DEFAULT 0,
            label_mode TEXT NOT NULL DEFAULT 'binary',
            minio_key TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        CREATE UNIQUE INDEX IF NOT EXISTS idx_judge_samples_name ON judge_samples (name);
        """
    )
    cur.execute(
        "ALTER TABLE judge_samples ADD COLUMN IF NOT EXISTS criteria_json JSONB NOT NULL DEFAULT '[]'::jsonb"
    )


def apply_migrations(conn) -> int:
    """Применяет все неприменённые миграции. Возвращает итоговую версию."""

    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version INTEGER PRIMARY KEY,
                applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
        cur.execute("SELECT COALESCE(MAX(version), 0) FROM schema_migrations")
        current = int(cur.fetchone()[0])
        for version, fn in MIGRATIONS:
            if version <= current:
                continue
            fn(cur)
            cur.execute(
                "INSERT INTO schema_migrations (version) VALUES (%s)",
                (version,),
            )
            current = version
    conn.commit()
    return current
