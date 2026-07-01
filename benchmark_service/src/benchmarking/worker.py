"""
Воркер: читает задачи из RabbitMQ, выполняет бенчмарк, пишет результаты в MinIO и Postgres.

Запуск (из корня репозитория):
  PYTHONPATH=src python -m benchmarking.worker
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time

import pika
from dotenv import load_dotenv
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_REPO_ROOT / ".env", override=False)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("benchmark_worker")

# при запуске с PYTHONPATH=src
_SCR = os.path.dirname(os.path.abspath(__file__))
if _SCR not in sys.path:
    sys.path.insert(0, _SCR)

from storage.benchmark_backend import (  # noqa: E402
    attach_results_to_cancelled_run,
    complete_run,
    ensure_minio_bucket,
    ensure_schema,
    fail_run,
    get_run,
    minio_get_bytes,
    minio_put_bytes,
    results_key_for_run,
    try_claim_run,
    update_run_progress,
)
from benchmarking.runner import (  # noqa: E402,F401
    benchmark_config_from_dict,
    benchmark_mean_reply_times,
    parse_jsonl_cases,
    run_benchmark_cases,
)
from common.utils import (  # noqa: E402
    enrich_benchmark_results_timing_inplace,
    metric_float,
    row_dialog_duration_sec_effective,
)

QUEUE = os.getenv("RABBITMQ_QUEUE", "benchmark.run")


def process_run(run_id: str):
    ensure_schema()
    row = get_run(run_id)
    if not row:
        log.error("Запуск %s не найден в БД", run_id)
        return

    if not try_claim_run(run_id):
        log.info("Запуск %s уже обрабатывается или не в очереди", run_id)
        return

    try:
        ensure_minio_bucket()
        raw = minio_get_bytes(row["input_key"])
        text = raw.decode("utf-8")
        test_cases = parse_jsonl_cases(text)
        if not test_cases:
            fail_run(run_id, "Пустой JSONL")
            return

        cfg_dict = row["config"]
        if isinstance(cfg_dict, str):
            cfg_dict = json.loads(cfg_dict)
        cfg = benchmark_config_from_dict(cfg_dict)
        default_model = os.getenv("LITELLM_MODEL_NAME", "openai/gpt-4o-mini")

        n = len(test_cases)
        rk = row.get("results_key") or results_key_for_run(run_id)
        update_run_progress(run_id, 0, n, 0.0)
        run_wall0 = time.monotonic()
        # Пустой файл — UI может опрашивать MinIO до первого диалога
        minio_put_bytes(rk, b"", "application/x-ndjson")

        def flush_to_minio(partial: list):
            wall = round(time.monotonic() - run_wall0, 4)
            enrich_benchmark_results_timing_inplace(partial)
            for x in partial:
                if not isinstance(x, dict):
                    continue
                x["benchmark_run_duration_sec"] = wall
                if metric_float(x.get("dialog_duration_sec")) is None:
                    eff = row_dialog_duration_sec_effective(x)
                    if eff is not None:
                        x["dialog_duration_sec"] = round(float(eff), 4)
            enrich_benchmark_results_timing_inplace(partial)
            body = "\n".join(
                json.dumps(x, ensure_ascii=False, default=str) for x in partial
            )
            minio_put_bytes(rk, body.encode("utf-8"), "application/x-ndjson")

        def stop_check():
            r = get_run(run_id)
            return r is not None and r.get("status") == "cancelled"

        def on_progress(done: int, total: int, partial: list):
            avg = sum(r["accuracy"] for r in partial) / len(partial) if partial else 0.0
            update_run_progress(run_id, done, total, avg)
            flush_to_minio(partial)

        results = run_benchmark_cases(
            test_cases,
            cfg,
            default_model=default_model,
            stop_check=stop_check,
            on_progress=on_progress,
        )

        flush_to_minio(results)

        if results:
            tr = metric_float(results[0].get("benchmark_run_duration_sec"))
            if tr is not None:
                log.info(
                    "Запуск %s: длительность полного прогона %.4f с",
                    run_id,
                    tr,
                )

        avg = sum(r["accuracy"] for r in results) / len(results) if results else 0.0
        m_as, m_us = benchmark_mean_reply_times(results)
        if m_as is not None:
            log.info(
                "Запуск %s: среднее время ответа ассистента по диалогам с метрикой: %.4f с",
                run_id,
                m_as,
            )
        if m_us is not None:
            log.info(
                "Запуск %s: среднее время ответа пользователя (сим.) по диалогам с метрикой: %.4f с",
                run_id,
                m_us,
            )
        final = get_run(run_id)

        if final and final.get("status") == "cancelled":
            attach_results_to_cancelled_run(
                run_id, rk, avg, len(results), n
            )
            log.info("Запуск %s остановлен, сохранено %s/%s", run_id, len(results), n)
        else:
            complete_run(run_id, rk, avg)
            log.info("Запуск %s завершён, точность %.4f", run_id, avg)

    except Exception as e:
        log.exception("Ошибка запуска %s", run_id)
        fail_run(run_id, str(e))


def main():
    url = os.getenv("RABBITMQ_URL")
    if not url:
        log.error("Задайте RABBITMQ_URL")
        sys.exit(1)

    ensure_schema()
    ensure_minio_bucket()

    params = pika.URLParameters(url)
    connection = pika.BlockingConnection(params)
    channel = connection.channel()
    channel.queue_declare(queue=QUEUE, durable=True)
    channel.basic_qos(prefetch_count=1)

    def callback(ch, method, _properties, body):
        # Долгий process_run не должен выполняться в I/O-потоке pika — иначе не
        # обрабатываются heartbeat'ы и RabbitMQ рвёт соединение (~60 с).
        conn = ch.connection

        def worker():
            try:
                msg = json.loads(body.decode("utf-8"))
                run_id = msg.get("run_id")
                if not run_id:
                    log.error("Нет run_id в сообщении")
                else:
                    process_run(str(run_id))
            except Exception:
                log.exception("Сбой обработки сообщения")
            finally:
                def ack():
                    try:
                        ch.basic_ack(delivery_tag=method.delivery_tag)
                    except Exception:
                        log.exception("basic_ack после обработки сообщения")

                conn.add_callback_threadsafe(ack)

        threading.Thread(target=worker, name="benchmark_run", daemon=True).start()

    channel.basic_consume(queue=QUEUE, on_message_callback=callback)
    log.info("Ожидание задач в очереди %s…", QUEUE)
    channel.start_consuming()


if __name__ == "__main__":
    main()
