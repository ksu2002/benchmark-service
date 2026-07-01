"""Группировка и кластеризация ошибок бенчмарка по полю записи results.jsonl."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from clustering import (
    CUSTOM_RECORD_FIELD_SENTINEL,
    extract_cluster_text,
    is_history_cluster_field,
    is_valid_custom_record_field,
    record_field_label,
    validate_record_field,
)

SUGGESTED_ERROR_CLUSTER_FIELDS: List[Tuple[str, str]] = [
    ("reason", "reason — обоснование LLM-судьи"),
    ("goals_text", "goals_text — цель"),
    ("eval_field_value", "eval_field_value — значение поля оценки"),
    ("result", "result — вердикт судьи"),
    ("context.category", "context.category — категория из context"),
    ("benchmark_run_exception", "benchmark_run_exception — исключение прогона"),
    ("dialog", "dialog — весь текст диалога (history)"),
    ("assistant", "assistant — реплики ассистента"),
    ("user", "user — реплики пользователя"),
    (CUSTOM_RECORD_FIELD_SENTINEL, "Своё поле (ключ JSONL или context.field)"),
]


def _stringify_value(raw: object) -> str:
    if raw is None:
        return ""
    if isinstance(raw, bool):
        return "true" if raw else "false"
    if isinstance(raw, (int, float)):
        return str(raw)
    if isinstance(raw, str):
        return raw.strip()
    if isinstance(raw, (list, dict)):
        try:
            return json.dumps(raw, ensure_ascii=False, sort_keys=True)
        except (TypeError, ValueError):
            return str(raw).strip()
    return str(raw).strip()


def _get_nested_value(record: Mapping[str, Any], path: str) -> object:
    obj: object = record
    for part in path.split("."):
        if not isinstance(obj, dict):
            return None
        obj = obj.get(part)
    return obj


def resolve_error_cluster_field(select_value: str, custom_value: str) -> str:
    if select_value == CUSTOM_RECORD_FIELD_SENTINEL:
        field = (custom_value or "").strip()
    else:
        field = (select_value or "").strip()
    if not field:
        raise ValueError("Укажите поле для группировки ошибок.")
    root = field.split(".", 1)[0]
    if is_history_cluster_field(root):
        return validate_record_field(root)
    if "." in field:
        if not all(p.strip() for p in field.split(".")):
            raise ValueError(f"Некорректный путь к полю: {field!r}")
        return field
    if is_valid_custom_record_field(field):
        return field
    raise ValueError(
        f"Недопустимое поле {field!r}. Пример: reason, goals_text, context.category."
    )


def extract_error_cluster_text(
    record: Mapping[str, Any],
    field: str,
    *,
    turn_index: int = 0,
) -> str:
    root = field.split(".", 1)[0]
    if is_history_cluster_field(root):
        return extract_cluster_text(record, root, turn_index=turn_index)
    if "." in field:
        return _stringify_value(_get_nested_value(record, field))
    return _stringify_value(record.get(field))


def field_label_for_ui(field: str) -> str:
    for key, label in SUGGESTED_ERROR_CLUSTER_FIELDS:
        if key == field:
            return label.split(" — ", 1)[0]
    if field.startswith("context."):
        return field
    return record_field_label(field) if is_history_cluster_field(field) else field


@dataclass(frozen=True)
class FailureFieldGroup:
    value: str
    records: Tuple[dict, ...]

    @property
    def count(self) -> int:
        return len(self.records)


def group_failures_by_field(
    failures: Sequence[dict],
    field: str,
    *,
    turn_index: int = 0,
    empty_label: str = "(пусто)",
) -> List[FailureFieldGroup]:
    buckets: Dict[str, List[dict]] = defaultdict(list)
    for row in failures:
        if not isinstance(row, dict):
            continue
        text = extract_error_cluster_text(row, field, turn_index=turn_index).strip()
        key = text if text else empty_label
        buckets[key].append(row)
    groups: List[FailureFieldGroup] = []
    for value, recs in sorted(buckets.items(), key=lambda item: (-len(item[1]), item[0].casefold())):
        groups.append(FailureFieldGroup(value=value, records=tuple(recs)))
    return groups


def failure_groups_summary_rows(
    groups: Sequence[FailureFieldGroup],
    *,
    field: str,
    total_failures: int,
    max_value_len: int = 120,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for i, grp in enumerate(groups, start=1):
        display = grp.value
        if len(display) > max_value_len:
            display = display[: max_value_len - 1] + "…"
        share = 100.0 * grp.count / total_failures if total_failures else 0.0
        dialog_ids = [str(r.get("dialog_id", ""))[:12] for r in grp.records[:5] if r.get("dialog_id") is not None]
        rows.append({"№": i, field_label_for_ui(field): display, "Кейсов": grp.count, "Доля, %": round(share, 1), "dialog_id (до 5)": ", ".join(dialog_ids) if dialog_ids else "—"})
    return rows


def failures_fingerprint(failures: Sequence[dict], field: str, turn_index: int) -> str:
    h = hashlib.sha256()
    h.update(f"{field}|{turn_index}|{len(failures)}".encode())
    for row in failures:
        if not isinstance(row, dict):
            continue
        h.update(str(row.get("dialog_id", "")).encode())
        h.update(extract_error_cluster_text(row, field, turn_index=turn_index).encode())
    return h.hexdigest()[:16]
