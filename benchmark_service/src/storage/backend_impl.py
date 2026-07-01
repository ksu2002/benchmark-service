"""
Postgres (метаданные запусков), MinIO (артефакты), RabbitMQ (очередь задач).
"""

from __future__ import annotations

import json
import os
import uuid
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
from minio import Minio
from minio.error import S3Error

# Репозиторий: src/storage/backend_impl.py → родитель src → корень проекта
_REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_REPO_ROOT / ".env", override=False)

SCHEMA_SQL = """
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
CREATE INDEX IF NOT EXISTS idx_benchmark_runs_created ON benchmark_runs (created_at DESC);
"""

JUDGE_PRESETS_SCHEMA_SQL = """
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

JUDGE_SAMPLES_SCHEMA_SQL = """
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


def get_postgres_dsn() -> Optional[str]:
    return os.getenv("BENCHMARK_POSTGRES_DSN") or os.getenv("DATABASE_URL")


def queue_backend_missing_vars() -> List[str]:
    missing: List[str] = []
    if not get_postgres_dsn():
        missing.append("BENCHMARK_POSTGRES_DSN или DATABASE_URL")
    if not os.getenv("MINIO_ENDPOINT"):
        missing.append("MINIO_ENDPOINT")
    if not os.getenv("RABBITMQ_URL"):
        missing.append("RABBITMQ_URL")
    return missing


def queue_backend_enabled() -> bool:
    return len(queue_backend_missing_vars()) == 0


@contextmanager
def db_conn():
    dsn = get_postgres_dsn()
    if not dsn:
        raise RuntimeError("BENCHMARK_POSTGRES_DSN или DATABASE_URL не задан")
    conn = psycopg2.connect(dsn)
    try:
        yield conn
    finally:
        conn.close()


def ensure_schema():
    from storage.schema_migrations import apply_migrations

    with db_conn() as conn:
        apply_migrations(conn)


def input_key_for_run(run_id: str) -> str:
    return f"runs/{run_id}/input.jsonl"


def results_key_for_run(run_id: str) -> str:
    return f"runs/{run_id}/results.jsonl"


def enrich_benchmark_config_for_storage(cfg: dict) -> dict:
    """Копия конфига с метаданными постановки в очередь (не ломает benchmark_config_from_dict)."""
    from datetime import datetime, timezone

    out = dict(cfg)
    out["litellm_api_base"] = (os.getenv("LITELLM_API_BASE") or "").strip()
    out["enqueue_meta"] = {
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "litellm_model_name_env": os.getenv("LITELLM_MODEL_NAME", ""),
        "litellm_api_base_env": os.getenv("LITELLM_API_BASE", ""),
    }
    return out


def create_run(
    config: dict,
    input_key: str,
    run_id: Optional[str] = None,
    title: str = "",
    description: str = "",
    progress_total_initial: int = 0,
) -> str:
    rid = run_id or str(uuid.uuid4())
    rk = results_key_for_run(rid)
    pt0 = max(0, int(progress_total_initial or 0))
    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO benchmark_runs (
                    id, status, config, input_key, progress_total,
                    run_title, run_description, results_key
                )
                VALUES (%s, 'queued', %s, %s, %s, %s, %s, %s)
                """,
                (
                    rid,
                    psycopg2.extras.Json(config),
                    input_key,
                    pt0,
                    title or "",
                    description or "",
                    rk,
                ),
            )
        conn.commit()
    return rid


def try_claim_run(run_id: str) -> bool:
    """Атомарно переводит queued → running. False, если уже взяли другой воркер."""
    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE benchmark_runs SET status = 'running'
                WHERE id = %s::uuid AND status = 'queued'
                RETURNING id
                """,
                (run_id,),
            )
            ok = cur.fetchone() is not None
        conn.commit()
    return ok


def update_run_progress(run_id: str, done: int, total: int, avg_accuracy: Optional[float]):
    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE benchmark_runs
                SET progress_done = %s, progress_total = %s, avg_accuracy = %s
                WHERE id = %s::uuid AND status = 'running'
                """,
                (done, total, avg_accuracy, run_id),
            )
        conn.commit()


def complete_run(run_id: str, results_key: str, avg_accuracy: float):
    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE benchmark_runs
                SET status = 'completed', results_key = %s, avg_accuracy = %s,
                    progress_done = progress_total
                WHERE id = %s::uuid
                """,
                (results_key, avg_accuracy, run_id),
            )
        conn.commit()


def fail_run(run_id: str, error: str):
    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE benchmark_runs
                SET status = 'failed', error = %s
                WHERE id = %s::uuid
                """,
                (error[:8000], run_id),
            )
        conn.commit()


def cancel_run(run_id: str) -> bool:
    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE benchmark_runs SET status = 'cancelled'
                WHERE id = %s::uuid AND status IN ('queued', 'running')
                RETURNING id
                """,
                (run_id,),
            )
            ok = cur.fetchone() is not None
        conn.commit()
    return ok


def attach_results_to_cancelled_run(
    run_id: str, results_key: str, avg_accuracy: float, progress_done: int, progress_total: int
):
    """После остановки: сохранить частичные результаты, статус остаётся cancelled."""
    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE benchmark_runs
                SET results_key = %s, avg_accuracy = %s,
                    progress_done = %s, progress_total = %s
                WHERE id = %s::uuid AND status = 'cancelled'
                """,
                (results_key, avg_accuracy, progress_done, progress_total, run_id),
            )
        conn.commit()


def get_run(run_id: str) -> Optional[Dict[str, Any]]:
    with db_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM benchmark_runs WHERE id = %s::uuid",
                (run_id,),
            )
            row = cur.fetchone()
    if not row:
        return None
    d = dict(row)
    d["id"] = str(d["id"])
    if isinstance(d.get("config"), str):
        d["config"] = json.loads(d["config"])
    return d


def list_recent_runs(limit: int = 50, search: Optional[str] = None) -> List[Dict[str, Any]]:
    needle = (search or "").strip()
    with db_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            base_select = """
                SELECT id, created_at, status, progress_done, progress_total,
                       avg_accuracy, error, results_key,
                       run_title, run_description,
                       config->>'scenario_id' AS scenario_id
                FROM benchmark_runs
            """
            if needle:
                pat = f"%{needle}%"
                cur.execute(
                    base_select
                    + """
                    WHERE COALESCE(run_title, '') ILIKE %s
                       OR COALESCE(run_description, '') ILIKE %s
                       OR CAST(id AS TEXT) ILIKE %s
                       OR COALESCE(config->>'scenario_id', '') ILIKE %s
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    (pat, pat, pat, pat, limit),
                )
            else:
                cur.execute(
                    base_select
                    + """
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    (limit,),
                )
            rows = cur.fetchall()
    out = []
    for row in rows:
        d = dict(row)
        d["id"] = str(d["id"])
        out.append(d)
    return out


def minio_endpoint_hint() -> str:
    """Подсказка по MINIO_ENDPOINT для UI (согласована с docker-compose.yml)."""
    return (
        "В Docker Compose для UI и воркера — **minio-dev:9000**; "
        "с хоста к compose — **localhost:9004**."
    )


def _minio_client() -> Minio:
    endpoint = os.getenv("MINIO_ENDPOINT", "localhost:9004")
    access = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
    secret = os.getenv("MINIO_SECRET_KEY", "minioadmin")
    secure = os.getenv("MINIO_USE_SSL", "false").lower() in ("1", "true", "yes")
    return Minio(endpoint, access_key=access, secret_key=secret, secure=secure)


def minio_bucket() -> str:
    return os.getenv("MINIO_BUCKET", "benchmarks")


def ensure_minio_bucket():
    client = _minio_client()
    b = minio_bucket()
    if not client.bucket_exists(b):
        client.make_bucket(b)


def minio_put_bytes(key: str, data: bytes, content_type: str = "application/octet-stream"):
    ensure_minio_bucket()
    client = _minio_client()
    client.put_object(
        minio_bucket(),
        key,
        BytesIO(data),
        length=len(data),
        content_type=content_type,
    )


def minio_get_bytes(key: str) -> bytes:
    client = _minio_client()
    resp = client.get_object(minio_bucket(), key)
    try:
        return resp.read()
    finally:
        resp.close()
        resp.release_conn()


def minio_try_get_bytes_with_error(key: Optional[str]) -> tuple[Optional[bytes], Optional[str]]:
    """(data, error). error=None, если ключа/бакета ещё нет (NoSuchKey / NoSuchBucket)."""
    if not key:
        return None, None
    try:
        return minio_get_bytes(key), None
    except S3Error as e:
        if e.code in ("NoSuchKey", "NoSuchBucket"):
            return None, None
        return None, f"{e.code}: {e.message}"
    except Exception as e:
        return None, str(e)


def minio_try_get_bytes(key: Optional[str]) -> Optional[bytes]:
    """Возвращает None, если ключа нет или объект не найден (удобно для частичных результатов)."""
    data, _ = minio_try_get_bytes_with_error(key)
    return data


def format_minio_results_miss(
    results_key: str,
    *,
    run_status: Optional[str] = None,
    minio_error: Optional[str] = None,
) -> tuple[str, str]:
    """Сообщение для UI: ('caption'|'warning', markdown)."""
    endpoint = os.getenv("MINIO_ENDPOINT", "(не задан)")
    if run_status in ("queued", "running") and not minio_error:
        return (
            "caption",
            "Ожидание **results.jsonl** в MinIO (воркер ещё не записал файл)…",
        )
    parts = [
        "Не удалось прочитать **results.jsonl** из MinIO.",
        f"Ключ: `{results_key}` · **MINIO_ENDPOINT:** `{endpoint}`.",
    ]
    if minio_error:
        parts.append(f"Ошибка: `{minio_error}`.")
    else:
        parts.append("Объект отсутствует.")
    parts.append(minio_endpoint_hint())
    return ("warning", " ".join(parts))


def publish_benchmark_job(run_id: str):
    import pika

    url = os.getenv("RABBITMQ_URL")
    if not url:
        raise RuntimeError("RABBITMQ_URL не задан")
    queue = os.getenv("RABBITMQ_QUEUE", "benchmark.run")
    body = json.dumps({"run_id": run_id})
    params = pika.URLParameters(url)
    connection = pika.BlockingConnection(params)
    try:
        channel = connection.channel()
        channel.queue_declare(queue=queue, durable=True)
        channel.basic_publish(
            exchange="",
            routing_key=queue,
            body=body.encode("utf-8"),
            properties=pika.BasicProperties(delivery_mode=2),
        )
    finally:
        connection.close()


def _normalize_judge_preset_row(row: Optional[dict]) -> Optional[Dict[str, Any]]:
    if not row:
        return None
    d = dict(row)
    d["id"] = str(d["id"])
    if isinstance(d.get("config"), str):
        d["config"] = json.loads(d["config"])
    return d


def list_judge_presets() -> List[Dict[str, Any]]:
    with db_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, name, description, config, is_default, created_at, updated_at
                FROM judge_presets
                ORDER BY is_default DESC, name ASC
                """
            )
            rows = cur.fetchall()
    return [_normalize_judge_preset_row(r) for r in rows]


def get_judge_preset(preset_id: str) -> Optional[Dict[str, Any]]:
    with db_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, name, description, config, is_default, created_at, updated_at
                FROM judge_presets
                WHERE id = %s::uuid
                """,
                (preset_id,),
            )
            row = cur.fetchone()
    return _normalize_judge_preset_row(row)


def get_default_judge_preset() -> Optional[Dict[str, Any]]:
    with db_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, name, description, config, is_default, created_at, updated_at
                FROM judge_presets
                WHERE is_default = TRUE
                ORDER BY updated_at DESC
                LIMIT 1
                """
            )
            row = cur.fetchone()
    return _normalize_judge_preset_row(row)


def _clear_default_judge_presets(cur) -> None:
    cur.execute("UPDATE judge_presets SET is_default = FALSE WHERE is_default = TRUE")


def save_judge_preset(
    name: str,
    config: dict,
    *,
    description: str = "",
    preset_id: Optional[str] = None,
    is_default: bool = False,
) -> str:
    title = (name or "").strip()
    if not title:
        raise ValueError("Имя пресета не может быть пустым")
    rid = preset_id or str(uuid.uuid4())
    with db_conn() as conn:
        with conn.cursor() as cur:
            if is_default:
                _clear_default_judge_presets(cur)
            if preset_id:
                cur.execute(
                    """
                    UPDATE judge_presets
                    SET name = %s,
                        description = %s,
                        config = %s::jsonb,
                        is_default = %s,
                        updated_at = NOW()
                    WHERE id = %s::uuid
                    """,
                    (
                        title,
                        (description or "").strip(),
                        json.dumps(config, ensure_ascii=False),
                        is_default,
                        rid,
                    ),
                )
                if cur.rowcount == 0:
                    raise ValueError(f"Пресет {rid} не найден")
            else:
                cur.execute(
                    """
                    INSERT INTO judge_presets (id, name, description, config, is_default)
                    VALUES (%s::uuid, %s, %s, %s::jsonb, %s)
                    """,
                    (
                        rid,
                        title,
                        (description or "").strip(),
                        json.dumps(config, ensure_ascii=False),
                        is_default,
                    ),
                )
        conn.commit()
    return rid


def delete_judge_preset(preset_id: str) -> bool:
    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM judge_presets WHERE id = %s::uuid", (preset_id,))
            deleted = cur.rowcount > 0
        conn.commit()
    return deleted


def judge_storage_missing_vars() -> List[str]:
    missing: List[str] = []
    if not get_postgres_dsn():
        missing.append("BENCHMARK_POSTGRES_DSN или DATABASE_URL")
    if not os.getenv("MINIO_ENDPOINT"):
        missing.append("MINIO_ENDPOINT")
    return missing


def judge_storage_enabled() -> bool:
    return len(judge_storage_missing_vars()) == 0


def judge_sample_minio_key(sample_id: str) -> str:
    return f"judge-samples/{sample_id}/data.jsonl"


def _normalize_judge_sample_row(row: Optional[dict]) -> Optional[Dict[str, Any]]:
    if not row:
        return None
    d = dict(row)
    d["id"] = str(d["id"])
    if isinstance(d.get("criteria_json"), str):
        d["criteria_json"] = json.loads(d["criteria_json"])
    return d


def list_judge_samples(limit: int = 100) -> List[Dict[str, Any]]:
    with db_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, name, description, case_count, annotated_count,
                       label_mode, criteria_json, minio_key, created_at, updated_at
                FROM judge_samples
                ORDER BY updated_at DESC
                LIMIT %s
                """,
                (limit,),
            )
            rows = cur.fetchall()
    return [_normalize_judge_sample_row(r) for r in rows]


def get_judge_sample(sample_id: str) -> Optional[Dict[str, Any]]:
    with db_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, name, description, case_count, annotated_count,
                       label_mode, criteria_json, minio_key, created_at, updated_at
                FROM judge_samples
                WHERE id = %s::uuid
                """,
                (sample_id,),
            )
            row = cur.fetchone()
    return _normalize_judge_sample_row(row)


def get_judge_sample_by_name(name: str) -> Optional[Dict[str, Any]]:
    title = (name or "").strip()
    if not title:
        return None
    with db_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, name, description, case_count, annotated_count,
                       label_mode, criteria_json, minio_key, created_at, updated_at
                FROM judge_samples
                WHERE name = %s
                """,
                (title,),
            )
            row = cur.fetchone()
    return _normalize_judge_sample_row(row)


def load_judge_sample_jsonl(sample_id: str) -> bytes:
    row = get_judge_sample(sample_id)
    if not row:
        raise ValueError(f"Выборка {sample_id} не найдена")
    return minio_get_bytes(row["minio_key"])


def save_judge_sample(
    name: str,
    jsonl_data: bytes,
    *,
    description: str = "",
    case_count: int = 0,
    annotated_count: int = 0,
    label_mode: str = "binary",
    criteria_json: Optional[List[str]] = None,
    sample_id: Optional[str] = None,
) -> str:
    title = (name or "").strip()
    if not title:
        raise ValueError("Название выборки не может быть пустым")
    if not jsonl_data:
        raise ValueError("Нет данных для сохранения")
    rid = sample_id or str(uuid.uuid4())
    key = judge_sample_minio_key(rid)
    minio_put_bytes(key, jsonl_data, content_type="application/x-ndjson")
    crit = json.dumps(criteria_json or [], ensure_ascii=False)
    with db_conn() as conn:
        with conn.cursor() as cur:
            if sample_id:
                cur.execute(
                    """
                    UPDATE judge_samples
                    SET name = %s,
                        description = %s,
                        case_count = %s,
                        annotated_count = %s,
                        label_mode = %s,
                        criteria_json = %s::jsonb,
                        minio_key = %s,
                        updated_at = NOW()
                    WHERE id = %s::uuid
                    """,
                    (
                        title,
                        (description or "").strip(),
                        int(case_count),
                        int(annotated_count),
                        (label_mode or "binary").strip(),
                        crit,
                        key,
                        rid,
                    ),
                )
                if cur.rowcount == 0:
                    raise ValueError(f"Выборка {rid} не найдена")
            else:
                cur.execute(
                    """
                    INSERT INTO judge_samples (
                        id, name, description, case_count, annotated_count,
                        label_mode, criteria_json, minio_key
                    )
                    VALUES (%s::uuid, %s, %s, %s, %s, %s, %s::jsonb, %s)
                    """,
                    (
                        rid,
                        title,
                        (description or "").strip(),
                        int(case_count),
                        int(annotated_count),
                        (label_mode or "binary").strip(),
                        crit,
                        key,
                    ),
                )
        conn.commit()
    return rid


def delete_judge_sample(sample_id: str) -> bool:
    row = get_judge_sample(sample_id)
    if not row:
        return False
    try:
        client = _minio_client()
        client.remove_object(minio_bucket(), row["minio_key"])
    except Exception:
        pass
    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM judge_samples WHERE id = %s::uuid", (sample_id,))
            deleted = cur.rowcount > 0
        conn.commit()
    return deleted
