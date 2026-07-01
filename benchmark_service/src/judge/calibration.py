"""Калибровка LLM-судьи: разметка, прогон, метрики согласия с человеком."""

from __future__ import annotations

import copy
import json
import math
import queue
import random
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from itertools import combinations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

try:
    import krippendorff
except ImportError:  # pragma: no cover
    krippendorff = None

try:
    from scipy.stats import pearsonr, spearmanr
except ImportError:  # pragma: no cover
    pearsonr = None
    spearmanr = None

from ui.judge_settings_ui import build_llm_context_from_judge_config, run_judge_test_on_case

LabelMode = str  # "binary" | "binary_multi" | "ordinal" | "categorical"

CONFIDENCE_Z_SCORES: Dict[float, float] = {
    0.90: 1.645,
    0.95: 1.96,
    0.99: 2.576,
}

HUMAN_LABEL_KEYS = (
    "human_label",
    "gold_label",
    "reference_label",
    "expected_result",
    "human_verdict",
)

HUMAN_LABELS_DICT_KEY = "human_labels"
HUMAN_NOTES_DICT_KEY = "human_notes"
LEGACY_ANNOTATOR_KEY = "(без имени)"


def normalize_annotator_name(name: str) -> str:
    return (name or "").strip()


def _labels_dict_from_mapping(raw: Any) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    return {str(k): v for k, v in raw.items() if str(k).strip()}


def get_human_labels_dict(case: dict) -> Dict[str, Any]:
    """Все метки разметчиков: ``human_labels`` и устаревшее поле ``human_label``."""
    labels = _labels_dict_from_mapping(case.get(HUMAN_LABELS_DICT_KEY))
    for key in HUMAN_LABEL_KEYS:
        if key == HUMAN_LABELS_DICT_KEY:
            continue
        if key in case and case[key] is not None:
            labels.setdefault(LEGACY_ANNOTATOR_KEY, case[key])
            break
    ctx = case.get("context")
    if isinstance(ctx, dict):
        nested = _labels_dict_from_mapping(ctx.get(HUMAN_LABELS_DICT_KEY))
        for k, v in nested.items():
            labels.setdefault(k, v)
        if not nested:
            for key in HUMAN_LABEL_KEYS:
                if key in ctx and ctx[key] is not None:
                    labels.setdefault(LEGACY_ANNOTATOR_KEY, ctx[key])
                    break
    return labels


def get_human_notes_dict(case: dict) -> Dict[str, str]:
    notes = {
        str(k): str(v)
        for k, v in _labels_dict_from_mapping(case.get(HUMAN_NOTES_DICT_KEY)).items()
        if str(k).strip()
    }
    legacy_note = case.get("human_note")
    if legacy_note and LEGACY_ANNOTATOR_KEY not in notes:
        notes[LEGACY_ANNOTATOR_KEY] = str(legacy_note)
    return notes


def list_annotators_from_cases(cases: Sequence[dict]) -> List[str]:
    names: set[str] = set()
    for case in cases:
        names.update(get_human_labels_dict(case).keys())
    return sorted(names, key=lambda x: (x == LEGACY_ANNOTATOR_KEY, x.lower()))


def get_annotator_label_from_case(case: dict, annotator: str, schema: "LabelSchema") -> Any:
    name = normalize_annotator_name(annotator)
    if not name:
        return None
    raw = get_human_labels_dict(case).get(name)
    if schema.mode == "binary_multi":
        return normalize_multi_binary_label(raw, schema.binary_criteria)
    return normalize_label_for_mode(raw, schema) if raw is not None else None


def get_annotator_note_from_case(case: dict, annotator: str) -> str:
    name = normalize_annotator_name(annotator)
    if not name:
        return ""
    return get_human_notes_dict(case).get(name, "")


def set_case_annotator_label(
    case: dict,
    annotator: str,
    label: Any,
    note: str = "",
) -> None:
    name = normalize_annotator_name(annotator)
    if not name:
        return
    labels = get_human_labels_dict(case)
    notes = get_human_notes_dict(case)
    labels[name] = label
    note_s = (note or "").strip()
    if note_s:
        notes[name] = note_s
    else:
        notes.pop(name, None)
    case[HUMAN_LABELS_DICT_KEY] = labels
    if notes:
        case[HUMAN_NOTES_DICT_KEY] = notes
    else:
        case.pop(HUMAN_NOTES_DICT_KEY, None)
    for key in HUMAN_LABEL_KEYS:
        case.pop(key, None)
    case.pop("human_note", None)


@dataclass
class CalibrationItem:
    idx: int
    case: dict
    human_label: Any = None
    human_note: str = ""
    llm_label: Any = None
    llm_reason: str = ""
    llm_raw: str = ""
    llm_error: str = ""
    llm_duration_sec: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "idx": self.idx,
            "case": self.case,
            "human_label": self.human_label,
            "human_note": self.human_note,
            "llm_label": self.llm_label,
            "llm_reason": self.llm_reason,
            "llm_raw": self.llm_raw,
            "llm_error": self.llm_error,
            "llm_duration_sec": self.llm_duration_sec,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CalibrationItem":
        raw_dur = d.get("llm_duration_sec")
        llm_duration_sec: Optional[float]
        if raw_dur is None or raw_dur == "":
            llm_duration_sec = None
        else:
            try:
                llm_duration_sec = float(raw_dur)
            except (TypeError, ValueError):
                llm_duration_sec = None
        return cls(
            idx=int(d.get("idx", 0)),
            case=d.get("case") or {},
            human_label=d.get("human_label"),
            human_note=d.get("human_note") or "",
            llm_label=d.get("llm_label"),
            llm_reason=d.get("llm_reason") or "",
            llm_raw=d.get("llm_raw") or "",
            llm_error=d.get("llm_error") or "",
            llm_duration_sec=llm_duration_sec,
        )


@dataclass
class LabelSchema:
    mode: LabelMode = "binary"
    ordinal_min: int = 1
    ordinal_max: int = 5
    categories: List[str] = field(default_factory=lambda: ["да", "нет"])
    llm_field: str = "result"
    binary_criteria: List[str] = field(default_factory=lambda: ["result"])

    def binary_options(self) -> List[Tuple[Any, str]]:
        return [(True, "✅ Да (цель достигнута)"), (False, "❌ Нет (цель не достигнута)")]

    def binary_multi_options(self) -> List[Tuple[int, str]]:
        return [(1, "1 — да"), (0, "0 — нет")]

    def ordinal_options(self) -> List[int]:
        return list(range(self.ordinal_min, self.ordinal_max + 1))

    def categorical_options(self) -> List[str]:
        return list(self.categories)

    def llm_eval_fields_list(self) -> List[str]:
        if self.mode == "binary_multi":
            fields = list(self.binary_criteria)
            if "reason" not in fields:
                fields.append("reason")
            return fields
        primary = (self.llm_field or "result").strip() or "result"
        return [primary, "reason"]


def infer_label_schema_from_cases(cases: Sequence[dict]) -> Optional[LabelSchema]:
    """Определить схему меток по ключам в human_labels (для загрузки JSONL)."""
    key_sets: List[frozenset] = []
    for case in cases:
        for raw in get_human_labels_dict(case).values():
            if isinstance(raw, dict) and raw:
                key_sets.append(frozenset(str(k) for k in raw))
    if not key_sets:
        return None
    common = Counter(key_sets).most_common(1)[0][0]
    criteria = sorted(common)
    if len(criteria) >= 2:
        return LabelSchema(mode="binary_multi", binary_criteria=criteria)
    if len(criteria) == 1:
        return LabelSchema(mode="binary", llm_field=criteria[0])
    return None


def _truthy_label(val: Any) -> Optional[bool]:
    if val is None:
        return None
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    s = str(val).strip().lower()
    if s in ("true", "1", "yes", "да", "pass", "успех", "ok", "success"):
        return True
    if s in ("false", "0", "no", "нет", "fail", "ошибка", "bad"):
        return False
    return None


def parse_binary_criteria_csv(raw: str) -> List[str]:
    return [c.strip() for c in (raw or "").split(",") if c.strip()]


def extract_human_label_from_case(case: dict, annotator: Optional[str] = None) -> Any:
    if annotator:
        name = normalize_annotator_name(annotator)
        if name:
            return get_human_labels_dict(case).get(name)
    labels = get_human_labels_dict(case)
    if len(labels) == 1:
        return next(iter(labels.values()))
    if LEGACY_ANNOTATOR_KEY in labels:
        return labels[LEGACY_ANNOTATOR_KEY]
    return None


def normalize_multi_binary_label(
    raw: Any,
    criteria: List[str],
) -> Optional[Dict[str, int]]:
    if not criteria or raw is None:
        return None
    if isinstance(raw, dict):
        out: Dict[str, int] = {}
        for c in criteria:
            if c not in raw:
                return None
            b = _truthy_label(raw[c])
            if b is None:
                return None
            out[c] = int(b)
        return out
    if isinstance(raw, list) and len(raw) == len(criteria):
        out = {}
        for c, v in zip(criteria, raw):
            b = _truthy_label(v)
            if b is None:
                return None
            out[c] = int(b)
        return out
    return None


def _human_label_valid_for_schema(raw: Any, schema: LabelSchema) -> bool:
    if raw is None:
        return False
    if schema.mode == "binary_multi":
        return normalize_multi_binary_label(raw, schema.binary_criteria) is not None
    return normalize_label_for_mode(raw, schema) is not None


def is_item_fully_annotated(item: CalibrationItem, schema: LabelSchema) -> bool:
    if _human_label_valid_for_schema(item.human_label, schema):
        return True
    for raw in get_human_labels_dict(item.case).values():
        if _human_label_valid_for_schema(raw, schema):
            return True
    return False


def is_item_llm_scored(item: CalibrationItem, schema: LabelSchema) -> bool:
    if schema.mode == "binary_multi":
        return normalize_multi_binary_label(item.llm_label, schema.binary_criteria) is not None
    return normalize_label_for_mode(item.llm_label, schema) is not None


def apply_annotator_to_item(
    item: CalibrationItem,
    annotator: str,
    schema: LabelSchema,
) -> None:
    name = normalize_annotator_name(annotator)
    if not name:
        return
    label = get_annotator_label_from_case(item.case, name, schema)
    item.human_label = label
    item.human_note = get_annotator_note_from_case(item.case, name)


def items_with_annotator_labels(
    items: Sequence[CalibrationItem],
    annotator: str,
    schema: LabelSchema,
) -> List[CalibrationItem]:
    out: List[CalibrationItem] = []
    for item in items:
        cloned = CalibrationItem.from_dict(item.to_dict())
        apply_annotator_to_item(cloned, annotator, schema)
        out.append(cloned)
    return out


def is_item_annotated_by(
    item: CalibrationItem,
    annotator: str,
    schema: LabelSchema,
) -> bool:
    name = normalize_annotator_name(annotator)
    if not name:
        return is_item_fully_annotated(item, schema)
    label = get_annotator_label_from_case(item.case, name, schema)
    if schema.mode == "binary_multi":
        return label is not None
    return normalize_label_for_mode(label, schema) is not None


def count_annotated_by(
    items: Sequence[CalibrationItem],
    annotator: str,
    schema: LabelSchema,
) -> int:
    return sum(1 for it in items if is_item_annotated_by(it, annotator, schema))


def _annotator_normalized_labels_for_consensus(
    item: CalibrationItem,
    schema: LabelSchema,
    annotators: Sequence[str],
) -> Optional[List[Any]]:
    """Нормализованные метки всех разметчиков или None, если кто-то не разметил."""
    names = [normalize_annotator_name(a) for a in annotators if normalize_annotator_name(a)]
    if len(names) < 2:
        return None
    labels: List[Any] = []
    for ann in names:
        raw = get_annotator_label_from_case(item.case, ann, schema)
        if schema.mode == "binary_multi":
            parsed = normalize_multi_binary_label(raw, schema.binary_criteria)
            if parsed is None:
                return None
            labels.append(tuple(parsed[c] for c in schema.binary_criteria))
        else:
            norm = normalize_label_for_mode(raw, schema) if raw is not None else None
            if norm is None:
                return None
            labels.append(norm)
    return labels


def is_item_annotator_consensus(
    item: CalibrationItem,
    schema: LabelSchema,
    annotators: Sequence[str],
) -> bool:
    """Возвращает ``True``, если все выбранные разметчики разметили кейс и их метки совпали."""
    labels = _annotator_normalized_labels_for_consensus(item, schema, annotators)
    if not labels:
        return False
    return len(set(labels)) == 1


def count_annotator_consensus(
    items: Sequence[CalibrationItem],
    schema: LabelSchema,
    annotators: Sequence[str],
) -> int:
    return sum(
        1 for it in items if is_item_annotator_consensus(it, schema, annotators)
    )


def is_item_annotated_for_compare(
    item: CalibrationItem,
    schema: LabelSchema,
    annotators: Optional[Sequence[str]] = None,
) -> bool:
    if not annotators:
        return is_item_fully_annotated(item, schema)
    names = [normalize_annotator_name(a) for a in annotators if normalize_annotator_name(a)]
    if not names:
        return is_item_fully_annotated(item, schema)
    return any(is_item_annotated_by(item, a, schema) for a in names)


def compute_human_label_statistics(
    items: Sequence[CalibrationItem],
    schema: LabelSchema,
) -> dict:
    """Распределение человеческих меток по выборке."""
    total = len(items)
    annotated = 0
    unannotated = 0

    if schema.mode == "binary_multi":
        per_criterion: Dict[str, Counter] = {c: Counter() for c in schema.binary_criteria}
        for item in items:
            label_map = normalize_multi_binary_label(item.human_label, schema.binary_criteria)
            if label_map is None:
                unannotated += 1
                continue
            annotated += 1
            for criterion, value in label_map.items():
                per_criterion[criterion][value] += 1
        return {
            "mode": "binary_multi",
            "total": total,
            "annotated": annotated,
            "unannotated": unannotated,
            "per_criterion": {c: dict(cnt) for c, cnt in per_criterion.items()},
        }

    counts: Counter = Counter()
    for item in items:
        label = normalize_label_for_mode(item.human_label, schema)
        if label is None:
            unannotated += 1
            continue
        annotated += 1
        counts[label] += 1

    return {
        "mode": schema.mode,
        "total": total,
        "annotated": annotated,
        "unannotated": unannotated,
        "counts": dict(counts),
    }


def cases_to_calibration_items(cases: List[dict]) -> List[CalibrationItem]:
    items: List[CalibrationItem] = []
    for i, case in enumerate(cases):
        case_copy = copy.deepcopy(case)
        hl = extract_human_label_from_case(case_copy)
        note = ""
        if hl is not None and len(get_human_labels_dict(case_copy)) == 1:
            only_name = next(iter(get_human_labels_dict(case_copy)))
            note = get_annotator_note_from_case(case_copy, only_name)
        items.append(
            CalibrationItem(idx=i, case=case_copy, human_label=hl, human_note=note)
        )
    return items


def list_annotators_from_items(items: Sequence[CalibrationItem]) -> List[str]:
    return list_annotators_from_cases([it.case for it in items])


def calibration_items_to_session(items: List[CalibrationItem]) -> List[dict]:
    return [it.to_dict() for it in items]


def calibration_items_from_session(raw: List[dict]) -> List[CalibrationItem]:
    return [CalibrationItem.from_dict(d) for d in (raw or [])]


JUDGE_SPLIT_KEY = "judge_split"
JUDGE_SPLIT_TRAIN = "train"
JUDGE_SPLIT_TEST = "test"


def get_case_split(case: dict) -> Optional[str]:
    raw = case.get(JUDGE_SPLIT_KEY)
    if raw in (JUDGE_SPLIT_TRAIN, JUDGE_SPLIT_TEST):
        return raw
    return None


def set_case_split(case: dict, split: str) -> None:
    if split in (JUDGE_SPLIT_TRAIN, JUDGE_SPLIT_TEST):
        case[JUDGE_SPLIT_KEY] = split
    else:
        case.pop(JUDGE_SPLIT_KEY, None)


def clear_pool_splits(items: Sequence[CalibrationItem]) -> None:
    for item in items:
        item.case.pop(JUDGE_SPLIT_KEY, None)


def pool_has_train_test_split(items: Sequence[CalibrationItem]) -> bool:
    return any(get_case_split(it.case) in (JUDGE_SPLIT_TRAIN, JUDGE_SPLIT_TEST) for it in items)


def filter_items_by_split(
    items: Sequence[CalibrationItem],
    split: str,
) -> List[CalibrationItem]:
    return [it for it in items if get_case_split(it.case) == split]


def count_items_by_split(items: Sequence[CalibrationItem]) -> Dict[str, int]:
    counts = {JUDGE_SPLIT_TRAIN: 0, JUDGE_SPLIT_TEST: 0, "unset": 0}
    for item in items:
        s = get_case_split(item.case)
        if s in counts:
            counts[s] += 1
        else:
            counts["unset"] += 1
    return counts


def _raw_human_label_from_case(
    case: dict,
    schema: LabelSchema,
    *,
    annotator: Optional[str] = None,
) -> Any:
    if annotator:
        return get_annotator_label_from_case(case, annotator, schema)
    raw = extract_human_label_from_case(case)
    if raw is not None:
        return raw
    labels = get_human_labels_dict(case)
    for name in sorted(labels.keys()):
        raw = get_annotator_label_from_case(case, name, schema)
        if raw is not None:
            return raw
    return None


def case_stratification_group_key(
    case: dict,
    schema: LabelSchema,
    *,
    annotator: Optional[str] = None,
) -> Optional[str]:
    """
    Ключ страты для train/test: комбинация значений по критериям текущей схемы.
    None — human-метки нет или она не парсится.
    """
    raw = _raw_human_label_from_case(case, schema, annotator=annotator)
    if raw is None:
        return None
    if schema.mode == "binary_multi":
        parsed = normalize_multi_binary_label(raw, schema.binary_criteria)
        if parsed is None:
            return None
        return ", ".join(f"{c}={parsed[c]}" for c in schema.binary_criteria)
    norm = normalize_label_for_mode(raw, schema)
    if norm is None:
        return None
    field = (schema.llm_field or "result").strip() or "result"
    if schema.mode == "binary":
        return f"{field}={int(bool(norm))}"
    return f"{field}={norm}"


def split_per_criterion_marginals(
    items: Sequence[CalibrationItem],
    schema: LabelSchema,
    *,
    annotator: Optional[str] = None,
) -> Dict[str, Dict[str, Dict[str, int]]]:
    """Обучение и тест по каждому критерию (маржинально), если split уже проставлен."""
    out: Dict[str, Dict[str, Dict[str, int]]] = {}
    if schema.mode != "binary_multi":
        return out
    for criterion in schema.binary_criteria:
        out[criterion] = {
            JUDGE_SPLIT_TRAIN: {0: 0, 1: 0},
            JUDGE_SPLIT_TEST: {0: 0, 1: 0},
        }
    for item in items:
        split = get_case_split(item.case)
        if split not in (JUDGE_SPLIT_TRAIN, JUDGE_SPLIT_TEST):
            continue
        raw = _raw_human_label_from_case(item.case, schema, annotator=annotator)
        parsed = normalize_multi_binary_label(raw, schema.binary_criteria)
        if parsed is None:
            continue
        for criterion in schema.binary_criteria:
            val = int(parsed[criterion])
            out[criterion][split][val] += 1
    return out


def _split_index_group(
    indices: Sequence[int],
    test_ratio: float,
    rng: random.Random,
) -> Tuple[List[int], List[int]]:
    shuffled = list(indices)
    rng.shuffle(shuffled)
    n = len(shuffled)
    if n == 0:
        return [], []
    n_test = int(round(n * test_ratio))
    if n >= 2 and n_test < 1:
        n_test = 1
    if n >= 2 and n_test >= n:
        n_test = n - 1
    test_idx = shuffled[:n_test]
    train_idx = shuffled[n_test:]
    return train_idx, test_idx


def stratified_train_test_split(
    items: Sequence[CalibrationItem],
    schema: LabelSchema,
    *,
    test_ratio: float = 0.2,
    seed: int = 42,
    annotator: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Стратифицированный split: внутри каждой группы меток (по критериям схемы)
    сохраняется доля test. Метка judge_split записывается в case.
    """
    test_ratio = max(0.05, min(0.5, float(test_ratio)))
    rng = random.Random(int(seed))
    groups: Dict[str, List[int]] = {}
    for i, item in enumerate(items):
        key = case_stratification_group_key(
            item.case, schema, annotator=annotator
        )
        bucket = key if key is not None else "__unlabeled__"
        groups.setdefault(bucket, []).append(i)

    train_pool: List[int] = []
    test_pool: List[int] = []
    per_group: Dict[str, dict] = {}
    for name, idxs in sorted(groups.items(), key=lambda x: (-len(x[1]), x[0])):
        tr, te = _split_index_group(idxs, test_ratio, rng)
        train_pool.extend(tr)
        test_pool.extend(te)
        per_group[name] = {"total": len(idxs), "train": len(tr), "test": len(te)}

    for i in train_pool:
        set_case_split(items[i].case, JUDGE_SPLIT_TRAIN)
    for i in test_pool:
        set_case_split(items[i].case, JUDGE_SPLIT_TEST)

    return {
        "train": len(train_pool),
        "test": len(test_pool),
        "test_ratio": test_ratio,
        "seed": seed,
        "per_group": per_group,
        "per_criterion": split_per_criterion_marginals(
            items, schema, annotator=annotator
        ),
        "annotator": normalize_annotator_name(annotator or "") or None,
        "schema_mode": schema.mode,
    }


def normalize_label_for_mode(raw: Any, schema: LabelSchema) -> Any:
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return None
    if schema.mode == "binary_multi":
        return normalize_multi_binary_label(raw, schema.binary_criteria)
    if schema.mode == "binary":
        b = _truthy_label(raw)
        return b
    if schema.mode == "ordinal":
        try:
            v = int(float(raw))
        except (TypeError, ValueError):
            return None
        if schema.ordinal_min <= v <= schema.ordinal_max:
            return v
        return None
    # categorical
    s = str(raw).strip()
    if s in schema.categories:
        return s
    # case-insensitive match
    lower_map = {c.lower(): c for c in schema.categories}
    return lower_map.get(s.lower())


def extract_llm_label(result: dict, schema: LabelSchema) -> Any:
    if not result or result.get("llm_error"):
        return None
    if schema.mode == "binary_multi":
        return normalize_multi_binary_label(
            {c: result.get(c) for c in schema.binary_criteria},
            schema.binary_criteria,
        )
    field_name = (schema.llm_field or "result").strip() or "result"
    raw = result.get(field_name)
    if raw is None and field_name == "result":
        raw = result.get("result")
    return normalize_label_for_mode(raw, schema)


def run_judge_on_item(
    item: CalibrationItem,
    config: Dict[str, Any],
    schema: LabelSchema,
) -> CalibrationItem:
    t0 = time.monotonic()
    try:
        result = run_judge_test_on_case(item.case, config)
        item.llm_duration_sec = round(time.monotonic() - t0, 4)
        item.llm_error = ""
        item.llm_raw = (result.get("raw_output") or "").strip()
        item.llm_reason = str(result.get("reason") or "")
        item.llm_label = extract_llm_label(result, schema)
        if item.llm_label is None and not item.llm_raw:
            item.llm_error = "Не удалось извлечь метку из ответа LLM"
    except Exception as e:
        item.llm_duration_sec = round(time.monotonic() - t0, 4)
        item.llm_error = str(e)
        item.llm_label = None
        item.llm_raw = ""
        item.llm_reason = ""
    return item


def clone_calibration_items(items: Sequence[CalibrationItem]) -> List[CalibrationItem]:
    return [CalibrationItem.from_dict(it.to_dict()) for it in items]


@dataclass
class CalibrationCompareJob:
    result_key: str
    config: Dict[str, Any]
    meta: Dict[str, Any] = field(default_factory=dict)


def job_model_name(config: Dict[str, Any]) -> str:
    return (config.get("evaluator") or {}).get("model") or "—"


def collect_batch_run_errors(
    batch: Sequence[CalibrationItem],
    schema: LabelSchema,
) -> List[dict]:
    errors: List[dict] = []
    for item in batch:
        if not is_item_fully_annotated(item, schema):
            continue
        err = (item.llm_error or "").strip()
        if not err and not is_item_llm_scored(item, schema):
            err = "Не удалось получить метку от LLM"
        if not err:
            continue
        goals = str(item.case.get("goals") or "").strip()
        if len(goals) > 160:
            goals = goals[:157] + "…"
        errors.append(
            {
                "idx": item.idx,
                "goals": goals or "—",
                "error": err,
            }
        )
    return errors


def run_judge_batch(
    items: List[CalibrationItem],
    config: Dict[str, Any],
    schema: LabelSchema,
    *,
    only_annotated: bool = True,
    annotators: Optional[Sequence[str]] = None,
    progress_callback: Optional[Callable[[int, int, CalibrationItem], None]] = None,
) -> List[CalibrationItem]:
    out: List[CalibrationItem] = []
    total = sum(
        1
        for it in items
        if not only_annotated
        or is_item_annotated_for_compare(it, schema, annotators)
    )
    run_count = 0
    for item in items:
        if only_annotated and not is_item_annotated_for_compare(
            item, schema, annotators
        ):
            out.append(item)
            continue
        run_count += 1
        scored = run_judge_on_item(item, config, schema)
        out.append(scored)
        if progress_callback is not None:
            progress_callback(run_count, total, scored)
    return out


def _run_single_compare_job(
    items: Sequence[CalibrationItem],
    schema: LabelSchema,
    job: CalibrationCompareJob,
    *,
    annotators: Optional[Sequence[str]] = None,
    progress_callback: Optional[Callable[[int, int, CalibrationItem], None]] = None,
    bootstrap: bool = False,
    max_disagreements: Optional[int] = None,
) -> Tuple[Dict[str, dict], Dict[str, List[CalibrationItem]]]:
    ann_names = [
        normalize_annotator_name(a)
        for a in (annotators or [])
        if normalize_annotator_name(a)
    ]
    batch_items = clone_calibration_items(items)
    annotated = sum(
        1
        for it in batch_items
        if is_item_annotated_for_compare(it, schema, ann_names or None)
    )
    batch = run_judge_batch(
        batch_items,
        job.config,
        schema,
        only_annotated=True,
        annotators=ann_names or None,
        progress_callback=progress_callback,
    )
    run_errors = collect_batch_run_errors(batch, schema)
    compare_targets = ann_names or [""]
    results: Dict[str, dict] = {}
    scored_batches: Dict[str, List[CalibrationItem]] = {}
    for ann in compare_targets:
        if ann:
            ann_batch = items_with_annotator_labels(batch, ann, schema)
            result_key = (
                f"{job.result_key} · {ann}"
                if len(compare_targets) > 1
                else job.result_key
            )
            ann_count = count_annotated_by(batch, ann, schema)
        else:
            ann_batch = batch
            result_key = job.result_key
            ann_count = annotated
        metrics = compute_agreement_metrics(
            ann_batch,
            schema,
            bootstrap=bootstrap,
            max_disagreements=max_disagreements,
        ).with_details()
        metrics.update(job.meta)
        metrics["model"] = metrics.get("model") or job_model_name(job.config)
        metrics["cases_scored"] = sum(
            1 for it in ann_batch if is_item_llm_scored(it, schema)
        )
        metrics["cases_annotated"] = ann_count
        metrics["run_errors"] = run_errors
        metrics["cases_errors"] = len(run_errors)
        if ann:
            metrics["annotator"] = ann
        results[result_key] = metrics
        scored_batches[result_key] = ann_batch
    return results, scored_batches


def _drain_compare_job_events(
    event_queue: queue.Queue,
    *,
    on_job_start: Optional[Callable[[CalibrationCompareJob, str], None]],
    on_job_done: Optional[Callable[[str, dict, float], None]],
    on_case_progress: Optional[Callable[[str, int, int, str], None]],
) -> None:
    while True:
        try:
            kind, payload = event_queue.get_nowait()
        except queue.Empty:
            break
        if kind == "start" and on_job_start:
            on_job_start(payload[0], payload[1])
        elif kind == "done" and on_job_done:
            on_job_done(payload[0], payload[1], payload[2])
        elif kind == "case" and on_case_progress:
            on_case_progress(payload[0], payload[1], payload[2], payload[3])


def run_calibration_compare_jobs(
    items: Sequence[CalibrationItem],
    schema: LabelSchema,
    jobs: List[CalibrationCompareJob],
    *,
    parallel_models: bool = True,
    annotators: Optional[Sequence[str]] = None,
    bootstrap: bool = False,
    max_disagreements: Optional[int] = None,
    on_job_start: Optional[Callable[[CalibrationCompareJob, str], None]] = None,
    on_job_done: Optional[Callable[[str, dict, float], None]] = None,
    on_case_progress: Optional[Callable[[str, int, int, str], None]] = None,
) -> Tuple[Dict[str, dict], Dict[str, List[CalibrationItem]]]:
    """Прогон вариантов сравнения. При нескольких jobs — параллельно (каждый job в своём потоке)."""
    if not jobs:
        return {}, {}

    use_parallel = parallel_models and len(jobs) > 1
    # В параллели не шлём per-case callbacks — иначе Streamlit захлёбывается обновлениями.
    case_progress_cb = on_case_progress if not use_parallel else None
    has_ui_callbacks = bool(on_job_start or on_job_done or on_case_progress)
    event_queue: Optional[queue.Queue] = (
        queue.Queue() if use_parallel and has_ui_callbacks else None
    )

    def emit_start(job: CalibrationCompareJob, model: str) -> None:
        if event_queue is not None:
            event_queue.put(("start", (job, model)))
        elif on_job_start:
            on_job_start(job, model)

    def emit_done(key: str, metrics: dict, elapsed: float) -> None:
        if event_queue is not None:
            event_queue.put(("done", (key, metrics, elapsed)))
        elif on_job_done:
            on_job_done(key, metrics, elapsed)

    def emit_case(key: str, cur: int, total: int, model: str) -> None:
        if event_queue is not None:
            event_queue.put(("case", (key, cur, total, model)))
        elif on_case_progress:
            on_case_progress(key, cur, total, model)

    def run_one(job: CalibrationCompareJob) -> Tuple[Dict[str, dict], Dict[str, List[CalibrationItem]]]:
        model = job_model_name(job.config)
        t0 = time.monotonic()
        emit_start(job, model)

        def case_cb(cur: int, total: int, _item: CalibrationItem) -> None:
            emit_case(job.result_key, cur, total, model)

        result_map, scored_map = _run_single_compare_job(
            items,
            schema,
            job,
            annotators=annotators,
            progress_callback=case_cb if case_progress_cb else None,
            bootstrap=bootstrap,
            max_disagreements=max_disagreements,
        )
        elapsed = time.monotonic() - t0
        per_elapsed = elapsed / max(1, len(result_map))
        for key, metrics in result_map.items():
            metrics["elapsed_sec"] = round(per_elapsed, 1)
            emit_done(key, metrics, per_elapsed)
        return result_map, scored_map

    results: Dict[str, dict] = {}
    scored_batches: Dict[str, List[CalibrationItem]] = {}
    if not use_parallel:
        for job in jobs:
            job_results, job_scored = run_one(job)
            results.update(job_results)
            scored_batches.update(job_scored)
        return results, scored_batches

    with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
        futures = {executor.submit(run_one, job): job for job in jobs}
        pending = set(futures.keys())
        while pending:
            if event_queue is not None:
                _drain_compare_job_events(
                    event_queue,
                    on_job_start=on_job_start,
                    on_job_done=on_job_done,
                    on_case_progress=on_case_progress,
                )
            done, pending = wait(pending, timeout=0.25, return_when=FIRST_COMPLETED)
            for fut in done:
                job_results, job_scored = fut.result()
                results.update(job_results)
                scored_batches.update(job_scored)
        if event_queue is not None:
            _drain_compare_job_events(
                event_queue,
                on_job_start=on_job_start,
                on_job_done=on_job_done,
                on_case_progress=on_case_progress,
            )
    return results, scored_batches


def paired_labels(
    items: List[CalibrationItem],
    schema: LabelSchema,
) -> Tuple[List[Any], List[Any]]:
    human: List[Any] = []
    llm: List[Any] = []
    for item in items:
        h = normalize_label_for_mode(item.human_label, schema)
        l = normalize_label_for_mode(item.llm_label, schema)
        if h is None or l is None:
            continue
        human.append(h)
        llm.append(l)
    return human, llm


def pabak_coefficient(y1: Sequence, y2: Sequence) -> Optional[float]:
    """
    PABAK (Prevalence-adjusted Bias-adjusted Kappa) для парных меток:
    PABAK = 2 * Po - 1, где Po — наблюдаемая доля совпадений (exact match).
    """
    po = exact_match_rate(y1, y2)
    if po is None:
        return None
    return 2.0 * po - 1.0


def cohens_kappa(y1: Sequence, y2: Sequence) -> Optional[float]:
    pairs = [(a, b) for a, b in zip(y1, y2)]
    n = len(pairs)
    if n < 2:
        return None
    labels = sorted(
        set(y1) | set(y2),
        key=lambda x: (str(type(x).__name__), str(x)),
    )
    idx = {lab: i for i, lab in enumerate(labels)}
    matrix = [[0] * len(labels) for _ in range(len(labels))]
    for a, b in pairs:
        matrix[idx[a]][idx[b]] += 1
    po = sum(matrix[i][i] for i in range(len(labels))) / n
    row_marginals = [sum(row) for row in matrix]
    col_marginals = [
        sum(matrix[r][c] for r in range(len(labels))) for c in range(len(labels))
    ]
    pe = sum(row_marginals[i] * col_marginals[i] for i in range(len(labels))) / (n * n)
    if pe >= 1.0:
        return 1.0 if po >= 1.0 else 0.0
    return (po - pe) / (1.0 - pe)


def _labels_for_krippendorff(values: Sequence) -> List[Any]:
    """Приводим bool к int — иначе krippendorff может упасть на смешанных типах."""
    out: List[Any] = []
    for v in values:
        if isinstance(v, bool):
            out.append(int(v))
        else:
            out.append(v)
    return out


def krippendorff_alpha_value(
    human: Sequence,
    llm: Sequence,
    *,
    mode: LabelMode,
) -> Optional[float]:
    if krippendorff is None:
        return None
    if len(human) < 2:
        return None
    level = "nominal"
    if mode == "ordinal":
        level = "ordinal"
    elif mode == "binary":
        level = "nominal"
    # Два кодировщика: строки — human и llm, столбцы — объекты
    data = [
        _labels_for_krippendorff(human),
        _labels_for_krippendorff(llm),
    ]
    try:
        return float(krippendorff.alpha(reliability_data=data, level_of_measurement=level))
    except Exception:
        return None


def _is_missing_rater_label(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return False


def _krippendorff_cell(value: Any) -> Any:
    if _is_missing_rater_label(value):
        return float("nan")
    if isinstance(value, bool):
        return int(value)
    return value


def krippendorff_alpha_multi_raters(
    rater_rows: Sequence[Sequence[Any]],
    *,
    mode: LabelMode,
) -> Optional[float]:
    """Krippendorff α для нескольких разметчиков (строки — кодировщики, столбцы — объекты)."""
    if krippendorff is None or len(rater_rows) < 2:
        return None
    n_items = len(rater_rows[0]) if rater_rows else 0
    if n_items < 1:
        return None
    overlap = 0
    for col in range(n_items):
        vals = [
            rater_rows[row][col]
            for row in range(len(rater_rows))
            if not _is_missing_rater_label(rater_rows[row][col])
        ]
        if len(vals) >= 2:
            overlap += 1
    if overlap < 1:
        return None
    level = "ordinal" if mode == "ordinal" else "nominal"
    data = [[_krippendorff_cell(v) for v in row] for row in rater_rows]
    try:
        return float(krippendorff.alpha(reliability_data=data, level_of_measurement=level))
    except Exception:
        return None


def exact_match_rate(y1: Sequence, y2: Sequence) -> Optional[float]:
    if not y1:
        return None
    matches = sum(1 for a, b in zip(y1, y2) if a == b)
    return matches / len(y1)


def plus_minus_one_agreement(y1: Sequence, y2: Sequence) -> Optional[float]:
    pairs = []
    for a, b in zip(y1, y2):
        try:
            pairs.append((int(a), int(b)))
        except (TypeError, ValueError):
            return None
    if not pairs:
        return None
    ok = sum(1 for a, b in pairs if abs(a - b) <= 1)
    return ok / len(pairs)


def pearson_correlation(y1: Sequence, y2: Sequence) -> Optional[float]:
    if pearsonr is None or len(y1) < 2:
        return None
    try:
        r, _ = pearsonr([float(x) for x in y1], [float(x) for x in y2])
        return float(r)
    except Exception:
        return None


def spearman_correlation(y1: Sequence, y2: Sequence) -> Optional[float]:
    if spearmanr is None or len(y1) < 2:
        return None
    try:
        r, _ = spearmanr([float(x) for x in y1], [float(x) for x in y2])
        return float(r)
    except Exception:
        return None


def binary_precision_recall(
    y_true: Sequence,
    y_pred: Sequence,
    *,
    positive: Any,
) -> Tuple[Optional[float], Optional[float]]:
    """Эталон — human, предсказание — LLM. Положительный класс задаётся явно."""
    tp = fp = fn = 0
    for t, p in zip(y_true, y_pred):
        t_pos = t == positive
        p_pos = p == positive
        if t_pos and p_pos:
            tp += 1
        elif not t_pos and p_pos:
            fp += 1
        elif t_pos and not p_pos:
            fn += 1
    has_true_pos = any(t == positive for t in y_true)
    has_pred_pos = any(p == positive for p in y_pred)
    if not has_true_pos and not has_pred_pos:
        return None, None
    precision: Optional[float]
    recall: Optional[float]
    if tp + fp:
        precision = tp / (tp + fp)
    elif has_pred_pos:
        precision = 0.0
    else:
        precision = None
    if tp + fn:
        recall = tp / (tp + fn)
    elif has_true_pos:
        recall = 0.0
    else:
        recall = None
    return precision, recall


def macro_precision_recall(
    y_true: Sequence,
    y_pred: Sequence,
) -> Tuple[Optional[float], Optional[float]]:
    """Macro-усреднённые precision/recall по всем классам в парах меток."""
    labels = sorted(
        set(y_true) | set(y_pred),
        key=lambda x: (str(type(x).__name__), str(x)),
    )
    if not labels:
        return None, None
    precs: List[float] = []
    recs: List[float] = []
    for c in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        precs.append(tp / (tp + fp) if tp + fp else 0.0)
        recs.append(tp / (tp + fn) if tp + fn else 0.0)
    return (
        (sum(precs) / len(precs)) if precs else None,
        (sum(recs) / len(recs)) if recs else None,
    )


def precision_recall_for_mode(
    y_true: Sequence,
    y_pred: Sequence,
    *,
    mode: LabelMode,
) -> Tuple[Optional[float], Optional[float]]:
    if not y_true:
        return None, None
    if mode == "binary":
        return binary_precision_recall(y_true, y_pred, positive=True)
    if mode == "binary_multi":
        return binary_precision_recall(y_true, y_pred, positive=1)
    return macro_precision_recall(y_true, y_pred)


def binary_confusion_counts(
    y_true: Sequence,
    y_pred: Sequence,
    *,
    positive: Any,
) -> Tuple[int, int, int, int]:
    tp = fp = fn = tn = 0
    for t, p in zip(y_true, y_pred):
        t_pos = t == positive
        p_pos = p == positive
        if t_pos and p_pos:
            tp += 1
        elif not t_pos and p_pos:
            fp += 1
        elif t_pos and not p_pos:
            fn += 1
        else:
            tn += 1
    return tp, fp, fn, tn


def binary_f1_score(
    y_true: Sequence,
    y_pred: Sequence,
    *,
    positive: Any = 1,
) -> Optional[float]:
    if not y_true:
        return None
    tp, fp, fn, _tn = binary_confusion_counts(y_true, y_pred, positive=positive)
    if tp == 0 and fp == 0 and fn == 0:
        return None
    if tp == 0:
        return 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    if prec + rec == 0:
        return 0.0
    return 2.0 * prec * rec / (prec + rec)


def matthews_correlation(
    y_true: Sequence,
    y_pred: Sequence,
    *,
    positive: Any = 1,
) -> Optional[float]:
    if not y_true:
        return None
    tp, fp, fn, tn = binary_confusion_counts(y_true, y_pred, positive=positive)
    denom = math.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denom == 0:
        return None
    return (tp * tn - fp * fn) / denom


def f1_for_mode(
    y_true: Sequence,
    y_pred: Sequence,
    *,
    mode: LabelMode,
) -> Optional[float]:
    if not y_true:
        return None
    if mode in ("binary", "binary_multi"):
        positive = True if mode == "binary" else 1
        return binary_f1_score(y_true, y_pred, positive=positive)
    precs, recs = [], []
    labels = sorted(
        set(y_true) | set(y_pred),
        key=lambda x: (str(type(x).__name__), str(x)),
    )
    for c in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        if tp + fp + fn == 0:
            continue
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        if prec + rec:
            precs.append(2.0 * prec * rec / (prec + rec))
        else:
            precs.append(0.0)
    return (sum(precs) / len(precs)) if precs else None


def mcc_for_mode(
    y_true: Sequence,
    y_pred: Sequence,
    *,
    mode: LabelMode,
) -> Optional[float]:
    if not y_true:
        return None
    if mode in ("binary", "binary_multi"):
        positive = True if mode == "binary" else 1
        return matthews_correlation(y_true, y_pred, positive=positive)
    labels = sorted(
        set(y_true) | set(y_pred),
        key=lambda x: (str(type(x).__name__), str(x)),
    )
    mccs: List[float] = []
    for c in labels:
        y_t = [1 if t == c else 0 for t in y_true]
        y_p = [1 if p == c else 0 for p in y_pred]
        v = matthews_correlation(y_t, y_p, positive=1)
        if v is not None:
            mccs.append(v)
    return (sum(mccs) / len(mccs)) if mccs else None


def bootstrap_metric_ci(
    y_true: Sequence,
    y_pred: Sequence,
    metric_fn: Callable[[Sequence, Sequence], Optional[float]],
    *,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: Optional[int] = 42,
    min_pairs: int = 5,
) -> Optional[Tuple[float, float]]:
    n = len(y_true)
    if n < min_pairs:
        return None
    rng = random.Random(seed)
    samples: List[float] = []
    for _ in range(n_bootstrap):
        idx = [rng.randrange(n) for _ in range(n)]
        h = [y_true[i] for i in idx]
        l = [y_pred[i] for i in idx]
        v = metric_fn(h, l)
        if v is not None:
            samples.append(float(v))
    if len(samples) < max(50, n_bootstrap // 10):
        return None
    samples.sort()
    alpha = (1.0 - confidence) / 2.0
    lo = samples[int(alpha * len(samples))]
    hi = samples[min(len(samples) - 1, int((1.0 - alpha) * len(samples)))]
    return lo, hi


def _min_optional(values: Sequence[Optional[float]]) -> Optional[float]:
    nums = [v for v in values if v is not None]
    return min(nums) if nums else None


COMPARE_RANK_WORST_KAPPA = "worst_kappa"
COMPARE_RANK_MEAN_KAPPA = "mean_kappa"
COMPARE_RANK_MEAN_PABAK = "mean_pabak"
COMPARE_RANK_WORST_PABAK = "worst_pabak"
COMPARE_RANK_WORST_F1 = "worst_f1"
COMPARE_RANK_WORST_MCC = "worst_mcc"
COMPARE_RANK_MEAN_F1 = "mean_f1"


def compare_variant_rank_key(
    metrics: dict,
    *,
    strategy: str = COMPARE_RANK_WORST_KAPPA,
) -> Tuple[float, float, float]:
    """
    Ключ для max() при выборе лучшего варианта сравнения.
    Возвращает tuple для лексикографической сортировки (больше — лучше).
    """
    per = metrics.get("per_criterion") or {}
    if per:
        kappas = [v.get("cohen_kappa") for v in per.values()]
        pabaks = [v.get("pabak") for v in per.values()]
        f1s = [v.get("f1") for v in per.values()]
        mccs = [v.get("mcc") for v in per.values()]
        if strategy == COMPARE_RANK_WORST_KAPPA:
            primary = _min_optional(kappas) or -2.0
        elif strategy == COMPARE_RANK_MEAN_KAPPA:
            nums = [v for v in kappas if v is not None]
            primary = (sum(nums) / len(nums)) if nums else -2.0
        elif strategy == COMPARE_RANK_WORST_PABAK:
            primary = _min_optional(pabaks) or -2.0
        elif strategy == COMPARE_RANK_WORST_F1:
            primary = _min_optional(f1s) or -2.0
        elif strategy == COMPARE_RANK_WORST_MCC:
            primary = _min_optional(mccs) or -2.0
        elif strategy == COMPARE_RANK_MEAN_F1:
            nums = [v for v in f1s if v is not None]
            primary = (sum(nums) / len(nums)) if nums else -2.0
        else:
            nums = [v for v in pabaks if v is not None]
            primary = (sum(nums) / len(nums)) if nums else -2.0
        secondary = metrics.get("pabak") or 0.0
        tertiary = _min_optional(f1s) or -2.0
        return float(primary), float(secondary), float(tertiary)
    if strategy in (COMPARE_RANK_WORST_KAPPA, COMPARE_RANK_MEAN_KAPPA):
        primary = metrics.get("cohen_kappa")
    elif strategy == COMPARE_RANK_WORST_F1 or strategy == COMPARE_RANK_MEAN_F1:
        primary = metrics.get("f1")
    elif strategy == COMPARE_RANK_WORST_MCC:
        primary = metrics.get("mcc")
    else:
        primary = metrics.get("pabak")
    return (
        float(primary if primary is not None else -2.0),
        float(metrics.get("pabak") or 0.0),
        float(metrics.get("f1") or -2.0),
    )


def compare_rank_strategy_label(strategy: str) -> str:
    labels = {
        COMPARE_RANK_WORST_KAPPA: "Худший Cohen κ (min по критериям)",
        COMPARE_RANK_MEAN_KAPPA: "Средний Cohen κ",
        COMPARE_RANK_MEAN_PABAK: "Средний PABAK",
        COMPARE_RANK_WORST_PABAK: "Худший PABAK (min по критериям)",
        COMPARE_RANK_WORST_F1: "Худший F1 (min по критериям)",
        COMPARE_RANK_WORST_MCC: "Худший MCC (min по критериям)",
        COMPARE_RANK_MEAN_F1: "Средний F1",
    }
    return labels.get(strategy, strategy)


def pabak_quality_label(pabak: Optional[float]) -> str:
    if pabak is None:
        return "—"
    if pabak >= 0.8:
        return "хорошо (≥ 0.8)"
    if pabak >= 0.6:
        return "приемлемо (≥ 0.6)"
    return "ниже порога (< 0.6)"


kappa_quality_label = pabak_quality_label


@dataclass
class AgreementMetrics:
    n_pairs: int = 0
    exact_match: Optional[float] = None
    plus_minus_one: Optional[float] = None
    pabak: Optional[float] = None
    cohen_kappa: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None
    mcc: Optional[float] = None
    min_pabak: Optional[float] = None
    min_f1: Optional[float] = None
    min_mcc: Optional[float] = None
    krippendorff_alpha: Optional[float] = None
    pearson_r: Optional[float] = None
    spearman_r: Optional[float] = None
    confusion: Dict[str, int] = field(default_factory=dict)
    disagreements: List[dict] = field(default_factory=list)
    per_criterion: Dict[str, dict] = field(default_factory=dict)
    bootstrap: Dict[str, Any] = field(default_factory=dict)
    mean_judge_reply_sec: Optional[float] = None
    n_judge_timed: int = 0

    def to_dict(self) -> dict:
        out = {
            "n_pairs": self.n_pairs,
            "exact_match": self.exact_match,
            "plus_minus_one": self.plus_minus_one,
            "pabak": self.pabak,
            "cohen_kappa": self.cohen_kappa,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "mcc": self.mcc,
            "min_pabak": self.min_pabak,
            "min_f1": self.min_f1,
            "min_mcc": self.min_mcc,
            "krippendorff_alpha": self.krippendorff_alpha,
            "pearson_r": self.pearson_r,
            "spearman_r": self.spearman_r,
            "pabak_quality": pabak_quality_label(self.pabak),
            "kappa_quality": kappa_quality_label(self.cohen_kappa),
            "mean_judge_reply_sec": self.mean_judge_reply_sec,
            "n_judge_timed": self.n_judge_timed,
        }
        if self.bootstrap:
            out["bootstrap"] = dict(self.bootstrap)
        if self.per_criterion:
            out["per_criterion"] = self.per_criterion
        return out

    def with_details(self) -> dict:
        """Метрики + матрица меток и расхождения для детального просмотра."""
        out = self.to_dict()
        out["confusion"] = dict(self.confusion)
        out["disagreements"] = list(self.disagreements)
        out["mismatches_count"] = len(self.disagreements)
        if self.per_criterion:
            per = {}
            for crit, md in self.per_criterion.items():
                row = dict(md)
                if "confusion" not in row:
                    row["confusion"] = md.get("confusion") or {}
                if "disagreements" not in row:
                    row["disagreements"] = md.get("disagreements") or []
                per[crit] = row
            out["per_criterion"] = per
        return out


@dataclass
class InterAnnotatorMetrics:
    n_annotators: int = 0
    n_items_compared: int = 0
    n_items_full_overlap: int = 0
    krippendorff_alpha: Optional[float] = None
    mean_pairwise_exact_match: Optional[float] = None
    mean_pairwise_cohen_kappa: Optional[float] = None
    mean_pairwise_pabak: Optional[float] = None
    pearson_r: Optional[float] = None
    spearman_r: Optional[float] = None
    plus_minus_one: Optional[float] = None
    pairwise: Dict[str, dict] = field(default_factory=dict)
    per_criterion: Dict[str, dict] = field(default_factory=dict)
    disagreements: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_annotators": self.n_annotators,
            "n_items_compared": self.n_items_compared,
            "n_items_full_overlap": self.n_items_full_overlap,
            "krippendorff_alpha": self.krippendorff_alpha,
            "mean_pairwise_exact_match": self.mean_pairwise_exact_match,
            "mean_pairwise_cohen_kappa": self.mean_pairwise_cohen_kappa,
            "mean_pairwise_pabak": self.mean_pairwise_pabak,
            "pearson_r": self.pearson_r,
            "spearman_r": self.spearman_r,
            "plus_minus_one": self.plus_minus_one,
            "kappa_quality": kappa_quality_label(self.mean_pairwise_cohen_kappa),
            "pairwise": dict(self.pairwise),
            "per_criterion": dict(self.per_criterion),
            "disagreements": list(self.disagreements),
            "disagreements_count": len(self.disagreements),
        }


def _mean_optional(values: Sequence[Optional[float]]) -> Optional[float]:
    nums = [v for v in values if v is not None]
    if not nums:
        return None
    return sum(nums) / len(nums)


def _pairwise_label_lists(
    rater_rows: Sequence[Sequence[Any]],
    idx_a: int,
    idx_b: int,
) -> Tuple[List[Any], List[Any]]:
    left: List[Any] = []
    right: List[Any] = []
    n_items = len(rater_rows[0]) if rater_rows else 0
    for col in range(n_items):
        a = rater_rows[idx_a][col]
        b = rater_rows[idx_b][col]
        if _is_missing_rater_label(a) or _is_missing_rater_label(b):
            continue
        left.append(a)
        right.append(b)
    return left, right


def _pairwise_annotator_metrics_dict(
    y1: Sequence[Any],
    y2: Sequence[Any],
    *,
    mode: LabelMode,
) -> dict:
    n = len(y1)
    out = {"n_pairs": n}
    if not n:
        return out
    out["exact_match"] = exact_match_rate(y1, y2)
    out["pabak"] = pabak_coefficient(y1, y2)
    out["cohen_kappa"] = cohens_kappa(y1, y2)
    if mode == "ordinal":
        out["plus_minus_one"] = plus_minus_one_agreement(y1, y2)
        out["pearson_r"] = pearson_correlation(y1, y2)
        out["spearman_r"] = spearman_correlation(y1, y2)
    return out


def _count_rater_overlap(rater_rows: Sequence[Sequence[Any]]) -> Tuple[int, int]:
    n_items = len(rater_rows[0]) if rater_rows else 0
    n_compared = 0
    n_full = 0
    n_raters = len(rater_rows)
    for col in range(n_items):
        present = sum(
            1 for row in range(n_raters) if not _is_missing_rater_label(rater_rows[row][col])
        )
        if present >= 2:
            n_compared += 1
        if present == n_raters and n_raters >= 2:
            n_full += 1
    return n_compared, n_full


def _collect_inter_annotator_disagreements(
    items: Sequence[CalibrationItem],
    schema: LabelSchema,
    annotators: Sequence[str],
    *,
    max_disagreements: Optional[int] = 50,
) -> List[dict]:
    names = [normalize_annotator_name(a) for a in annotators if normalize_annotator_name(a)]
    if len(names) < 2:
        return []

    disputes: List[dict] = []
    for item in items:
        if schema.mode == "binary_multi":
            label_maps: Dict[str, Dict[str, int]] = {}
            for ann in names:
                raw = get_annotator_label_from_case(item.case, ann, schema)
                parsed = normalize_multi_binary_label(raw, schema.binary_criteria)
                if parsed is not None:
                    label_maps[ann] = parsed
            if len(label_maps) < 2:
                continue
            disputed: List[str] = []
            labels_by_ann: Dict[str, dict] = {}
            for criterion in schema.binary_criteria:
                vals = {ann: label_maps[ann][criterion] for ann in label_maps}
                labels_by_ann[criterion] = vals
                if len(set(vals.values())) > 1:
                    disputed.append(criterion)
            if not disputed:
                continue
            row = {
                "idx": item.idx,
                "mode": "binary_multi",
                "disputed_criteria": disputed,
                "labels_by_criterion": labels_by_ann,
                "annotators": list(label_maps.keys()),
            }
        else:
            labels: Dict[str, Any] = {}
            for ann in names:
                raw = get_annotator_label_from_case(item.case, ann, schema)
                norm = normalize_label_for_mode(raw, schema) if raw is not None else None
                if norm is not None:
                    labels[ann] = norm
            if len(labels) < 2 or len(set(labels.values())) <= 1:
                continue
            row = {
                "idx": item.idx,
                "mode": schema.mode,
                "labels": labels,
                "annotators": list(labels.keys()),
            }
        row.update(_case_disagreement_context(item))
        disputes.append(row)
        if _disagreement_limit_reached(disputes, max_disagreements):
            break
    return disputes


def _compute_inter_annotator_from_rater_rows(
    rater_rows: Sequence[Sequence[Any]],
    annotators: Sequence[str],
    *,
    mode: LabelMode,
    items: Sequence[CalibrationItem],
    schema: LabelSchema,
    max_disagreements: Optional[int],
    criterion: str = "",
) -> InterAnnotatorMetrics:
    names = list(annotators)
    n_compared, n_full = _count_rater_overlap(rater_rows)
    alpha = krippendorff_alpha_multi_raters(rater_rows, mode=mode)

    pairwise: Dict[str, dict] = {}
    exacts: List[Optional[float]] = []
    kappas: List[Optional[float]] = []
    pabaks: List[Optional[float]] = []
    pearsons: List[Optional[float]] = []
    spearmans: List[Optional[float]] = []
    pm1s: List[Optional[float]] = []

    for a, b in combinations(range(len(names)), 2):
        y1, y2 = _pairwise_label_lists(rater_rows, a, b)
        pm = _pairwise_annotator_metrics_dict(y1, y2, mode=mode)
        key = f"{names[a]} × {names[b]}"
        if criterion:
            pm["criterion"] = criterion
        pairwise[key] = pm
        exacts.append(pm.get("exact_match"))
        kappas.append(pm.get("cohen_kappa"))
        pabaks.append(pm.get("pabak"))
        if mode == "ordinal":
            pearsons.append(pm.get("pearson_r"))
            spearmans.append(pm.get("spearman_r"))
            pm1s.append(pm.get("plus_minus_one"))

    disagreements = _collect_inter_annotator_disagreements(
        items, schema, names, max_disagreements=max_disagreements
    )
    if criterion:
        disagreements = [
            d for d in disagreements if criterion in (d.get("disputed_criteria") or [])
        ]

    return InterAnnotatorMetrics(
        n_annotators=len(names),
        n_items_compared=n_compared,
        n_items_full_overlap=n_full,
        krippendorff_alpha=alpha,
        mean_pairwise_exact_match=_mean_optional(exacts),
        mean_pairwise_cohen_kappa=_mean_optional(kappas),
        mean_pairwise_pabak=_mean_optional(pabaks),
        pearson_r=_mean_optional(pearsons),
        spearman_r=_mean_optional(spearmans),
        plus_minus_one=_mean_optional(pm1s),
        pairwise=pairwise,
        disagreements=disagreements,
    )


def compute_inter_annotator_agreement(
    items: Sequence[CalibrationItem],
    schema: LabelSchema,
    annotators: Sequence[str],
    *,
    max_disagreements: Optional[int] = 50,
) -> InterAnnotatorMetrics:
    """Согласие между разметчиками (inter-annotator agreement)."""
    names = [normalize_annotator_name(a) for a in annotators if normalize_annotator_name(a)]
    if len(names) < 2:
        return InterAnnotatorMetrics(n_annotators=len(names))

    if schema.mode == "binary_multi":
        per: Dict[str, dict] = {}
        alphas: List[Optional[float]] = []
        exacts: List[Optional[float]] = []
        kappas: List[Optional[float]] = []
        pabaks: List[Optional[float]] = []
        for criterion in schema.binary_criteria:
            rater_rows: List[List[Any]] = []
            for ann in names:
                row: List[Any] = []
                for item in items:
                    raw = get_annotator_label_from_case(item.case, ann, schema)
                    parsed = normalize_multi_binary_label(raw, schema.binary_criteria)
                    row.append(parsed[criterion] if parsed is not None else None)
                rater_rows.append(row)
            cm = _compute_inter_annotator_from_rater_rows(
                rater_rows,
                names,
                mode="binary_multi",
                items=items,
                schema=schema,
                max_disagreements=0,
                criterion=criterion,
            )
            crit_out = cm.to_dict()
            crit_out.pop("disagreements", None)
            per[criterion] = crit_out
            alphas.append(cm.krippendorff_alpha)
            exacts.append(cm.mean_pairwise_exact_match)
            kappas.append(cm.mean_pairwise_cohen_kappa)
            pabaks.append(cm.mean_pairwise_pabak)

        disagreements = _collect_inter_annotator_disagreements(
            items, schema, names, max_disagreements=max_disagreements
        )
        return InterAnnotatorMetrics(
            n_annotators=len(names),
            n_items_compared=max(
                (v.get("n_items_compared") or 0 for v in per.values()),
                default=0,
            ),
            n_items_full_overlap=min(
                (v.get("n_items_full_overlap") or 0 for v in per.values()),
                default=0,
            ),
            krippendorff_alpha=_mean_optional(alphas),
            mean_pairwise_exact_match=_mean_optional(exacts),
            mean_pairwise_cohen_kappa=_mean_optional(kappas),
            mean_pairwise_pabak=_mean_optional(pabaks),
            per_criterion=per,
            disagreements=disagreements,
        )

    rater_rows = []
    for ann in names:
        row = []
        for item in items:
            raw = get_annotator_label_from_case(item.case, ann, schema)
            norm = normalize_label_for_mode(raw, schema) if raw is not None else None
            row.append(norm)
        rater_rows.append(row)

    return _compute_inter_annotator_from_rater_rows(
        rater_rows,
        names,
        mode=schema.mode,
        items=items,
        schema=schema,
        max_disagreements=max_disagreements,
    )


def _case_disagreement_context(item: CalibrationItem) -> dict:
    return {
        "goals": str(item.case.get("goals") or ""),
        "history": item.case.get("history") or [],
    }


def confusion_to_table_rows(confusion: Dict[str, int]) -> List[dict]:
    rows: List[dict] = []
    for key, cnt in sorted(confusion.items(), key=lambda x: (-x[1], x[0])):
        if " → " not in key:
            continue
        human, llm = key.split(" → ", 1)
        rows.append(
            {
                "Эталон (human)": human,
                "Ответ LLM": llm,
                "Кол-во": cnt,
                "Расхождение": human != llm,
            }
        )
    return rows


_CONFUSION_LABEL_ORDER: Dict[str, int] = {
    "0": 0,
    "1": 1,
    "false": 0,
    "true": 1,
    "нет": 0,
    "да": 1,
    "no": 0,
    "yes": 1,
    "неверно": 0,
    "верно": 1,
    "fail": 0,
    "pass": 1,
}


def _order_confusion_labels(labels: Sequence[str]) -> List[str]:
    unique = sorted({str(label) for label in labels})

    def sort_key(label: str) -> tuple:
        lowered = label.strip().casefold()
        if lowered in _CONFUSION_LABEL_ORDER:
            return (0, _CONFUSION_LABEL_ORDER[lowered], lowered)
        try:
            return (1, float(label.replace(",", ".")), lowered)
        except ValueError:
            return (2, lowered, label)

    return sorted(unique, key=sort_key)


@dataclass
class ConfusionMatrixData:
    human_labels: List[str]
    llm_labels: List[str]
    counts: List[List[int]]
    total: int
    correct: int
    errors: int


def build_confusion_matrix(confusion: Dict[str, int]) -> Optional[ConfusionMatrixData]:
    """Строит матрицу human (строки) × LLM (столбцы) из словаря пар «h → l»."""
    if not confusion:
        return None

    pair_counts: Dict[Tuple[str, str], int] = {}
    for key, cnt in confusion.items():
        if " → " not in key:
            continue
        amount = int(cnt)
        if amount <= 0:
            continue
        human, llm = key.split(" → ", 1)
        pair_key = (human, llm)
        pair_counts[pair_key] = pair_counts.get(pair_key, 0) + amount

    if not pair_counts:
        return None

    human_set = {human for human, _ in pair_counts}
    llm_set = {llm for _, llm in pair_counts}
    label_order = _order_confusion_labels(human_set | llm_set)
    human_labels = [label for label in label_order if label in human_set]
    llm_labels = [label for label in label_order if label in llm_set]

    counts: List[List[int]] = []
    total = 0
    correct = 0
    for human in human_labels:
        row: List[int] = []
        for llm in llm_labels:
            amount = pair_counts.get((human, llm), 0)
            row.append(amount)
            total += amount
            if human == llm:
                correct += amount
        counts.append(row)

    return ConfusionMatrixData(
        human_labels=human_labels,
        llm_labels=llm_labels,
        counts=counts,
        total=total,
        correct=correct,
        errors=total - correct,
    )


def _disagreement_limit_reached(
    disagreements: Sequence[dict],
    max_disagreements: Optional[int],
) -> bool:
    return max_disagreements is not None and len(disagreements) >= max_disagreements


def _compute_single_criterion_metrics(
    human: Sequence,
    llm: Sequence,
    *,
    mode: LabelMode,
    max_disagreements: Optional[int],
    criterion: str = "",
) -> AgreementMetrics:
    metrics = AgreementMetrics(n_pairs=len(human))
    if not human:
        return metrics
    metrics.exact_match = exact_match_rate(human, llm)
    if mode == "ordinal":
        metrics.plus_minus_one = plus_minus_one_agreement(human, llm)
        metrics.pearson_r = pearson_correlation(human, llm)
        metrics.spearman_r = spearman_correlation(human, llm)
    if mode in ("binary", "binary_multi", "categorical", "ordinal"):
        metrics.pabak = pabak_coefficient(human, llm)
        metrics.cohen_kappa = cohens_kappa(human, llm)
        metrics.krippendorff_alpha = krippendorff_alpha_value(human, llm, mode=mode)
        prec, rec = precision_recall_for_mode(human, llm, mode=mode)
        metrics.precision = prec
        metrics.recall = rec
        metrics.f1 = f1_for_mode(human, llm, mode=mode)
        metrics.mcc = mcc_for_mode(human, llm, mode=mode)
    pair_counter = Counter(zip(human, llm))
    metrics.confusion = {f"{h} → {l}": c for (h, l), c in pair_counter.most_common()}
    for h, l in zip(human, llm):
        if h == l:
            continue
        metrics.disagreements.append({"criterion": criterion, "human": h, "llm": l})
        if _disagreement_limit_reached(metrics.disagreements, max_disagreements):
            break
    return metrics


def _attach_bootstrap_cis(
    metrics: AgreementMetrics,
    human: Sequence,
    llm: Sequence,
    *,
    mode: LabelMode,
    bootstrap: bool,
    bootstrap_n: int = 1000,
    bootstrap_seed: Optional[int] = 42,
) -> AgreementMetrics:
    if not bootstrap or len(human) < 5:
        return metrics
    pabak_ci = bootstrap_metric_ci(
        human,
        llm,
        lambda h, l: pabak_coefficient(h, l),
        n_bootstrap=bootstrap_n,
        seed=bootstrap_seed,
    )
    kappa_ci = bootstrap_metric_ci(
        human,
        llm,
        lambda h, l: cohens_kappa(h, l),
        n_bootstrap=bootstrap_n,
        seed=(bootstrap_seed or 42) + 1,
    )
    if pabak_ci:
        metrics.bootstrap["pabak_ci"] = list(pabak_ci)
    if kappa_ci:
        metrics.bootstrap["cohen_kappa_ci"] = list(kappa_ci)
    if mode in ("binary", "binary_multi"):
        f1_ci = bootstrap_metric_ci(
            human,
            llm,
            lambda h, l: f1_for_mode(h, l, mode=mode),
            n_bootstrap=bootstrap_n,
            seed=(bootstrap_seed or 42) + 2,
        )
        if f1_ci:
            metrics.bootstrap["f1_ci"] = list(f1_ci)
    return metrics


def mean_judge_reply_time_sec(
    items: Sequence[CalibrationItem],
) -> Tuple[Optional[float], int]:
    """Среднее wall-clock время одного вызова судьи (сек) по кейсам с замером."""
    vals = [
        float(it.llm_duration_sec)
        for it in items
        if it.llm_duration_sec is not None
    ]
    if not vals:
        return None, 0
    return sum(vals) / len(vals), len(vals)


def _attach_judge_timing_metrics(
    metrics: AgreementMetrics,
    items: Sequence[CalibrationItem],
) -> AgreementMetrics:
    mean_sec, n_timed = mean_judge_reply_time_sec(items)
    metrics.mean_judge_reply_sec = mean_sec
    metrics.n_judge_timed = n_timed
    return metrics


def compute_agreement_metrics(
    items: List[CalibrationItem],
    schema: LabelSchema,
    *,
    max_disagreements: Optional[int] = 20,
    bootstrap: bool = False,
    bootstrap_n: int = 1000,
    bootstrap_seed: Optional[int] = 42,
) -> AgreementMetrics:
    if schema.mode == "binary_multi":
        per: Dict[str, dict] = {}
        all_disagreements: List[dict] = []
        pabaks: List[float] = []
        kappas: List[float] = []
        exacts: List[float] = []
        precisions: List[float] = []
        recalls: List[float] = []
        f1s: List[float] = []
        mccs: List[float] = []
        total_pairs = 0
        bootstrap_rows: List[Tuple[List[int], List[int]]] = []
        for criterion in schema.binary_criteria:
            human: List[int] = []
            llm: List[int] = []
            for item in items:
                hmap = normalize_multi_binary_label(item.human_label, schema.binary_criteria)
                lmap = normalize_multi_binary_label(item.llm_label, schema.binary_criteria)
                if not hmap or not lmap:
                    continue
                human.append(hmap[criterion])
                llm.append(lmap[criterion])
            cm = _compute_single_criterion_metrics(
                human,
                llm,
                mode="binary_multi",
                max_disagreements=5,
                criterion=criterion,
            )
            crit_out = cm.to_dict()
            crit_out["confusion"] = dict(cm.confusion)
            crit_out["disagreements"] = list(cm.disagreements)
            per[criterion] = crit_out
            if cm.n_pairs:
                total_pairs += cm.n_pairs
                bootstrap_rows.append((list(human), list(llm)))
                if cm.pabak is not None:
                    pabaks.append(cm.pabak)
                if cm.cohen_kappa is not None:
                    kappas.append(cm.cohen_kappa)
                if cm.exact_match is not None:
                    exacts.append(cm.exact_match)
                if cm.precision is not None:
                    precisions.append(cm.precision)
                if cm.recall is not None:
                    recalls.append(cm.recall)
                if cm.f1 is not None:
                    f1s.append(cm.f1)
                if cm.mcc is not None:
                    mccs.append(cm.mcc)
            for item in items:
                hmap = normalize_multi_binary_label(item.human_label, schema.binary_criteria)
                lmap = normalize_multi_binary_label(item.llm_label, schema.binary_criteria)
                if not hmap or not lmap:
                    continue
                if hmap[criterion] == lmap[criterion]:
                    continue
                row = {
                    "idx": item.idx,
                    "criterion": criterion,
                    "human": hmap[criterion],
                    "llm": lmap[criterion],
                    "llm_reason": item.llm_reason,
                }
                row.update(_case_disagreement_context(item))
                all_disagreements.append(row)
                if _disagreement_limit_reached(all_disagreements, max_disagreements):
                    break
            if _disagreement_limit_reached(all_disagreements, max_disagreements):
                break
        metrics = AgreementMetrics(
            n_pairs=total_pairs // max(1, len(schema.binary_criteria)),
            exact_match=(sum(exacts) / len(exacts)) if exacts else None,
            pabak=(sum(pabaks) / len(pabaks)) if pabaks else None,
            cohen_kappa=(sum(kappas) / len(kappas)) if kappas else None,
            precision=(sum(precisions) / len(precisions)) if precisions else None,
            recall=(sum(recalls) / len(recalls)) if recalls else None,
            f1=(sum(f1s) / len(f1s)) if f1s else None,
            mcc=(sum(mccs) / len(mccs)) if mccs else None,
            min_pabak=_min_optional(pabaks),
            min_f1=_min_optional(f1s),
            min_mcc=_min_optional(mccs),
            krippendorff_alpha=None,
            per_criterion=per,
            disagreements=all_disagreements[:max_disagreements],
        )
        if bootstrap and bootstrap_rows:
            n = min(len(row[0]) for row in bootstrap_rows)
            if n >= 5:
                rng = random.Random(bootstrap_seed)
                pabak_samples: List[float] = []
                kappa_samples: List[float] = []
                for _ in range(bootstrap_n):
                    idx = [rng.randrange(n) for _ in range(n)]
                    crit_pabaks: List[float] = []
                    crit_kappas: List[float] = []
                    for human_row, llm_row in bootstrap_rows:
                        h = [human_row[i] for i in idx]
                        l = [llm_row[i] for i in idx]
                        pb = pabak_coefficient(h, l)
                        k = cohens_kappa(h, l)
                        if pb is not None:
                            crit_pabaks.append(pb)
                        if k is not None:
                            crit_kappas.append(k)
                    if crit_pabaks:
                        pabak_samples.append(sum(crit_pabaks) / len(crit_pabaks))
                    if crit_kappas:
                        kappa_samples.append(sum(crit_kappas) / len(crit_kappas))
                if len(pabak_samples) >= bootstrap_n // 10:
                    pabak_samples.sort()
                    lo = pabak_samples[int(0.025 * len(pabak_samples))]
                    hi = pabak_samples[int(0.975 * len(pabak_samples)) - 1]
                    metrics.bootstrap["pabak_ci"] = [lo, hi]
                if len(kappa_samples) >= bootstrap_n // 10:
                    kappa_samples.sort()
                    lo = kappa_samples[int(0.025 * len(kappa_samples))]
                    hi = kappa_samples[int(0.975 * len(kappa_samples)) - 1]
                    metrics.bootstrap["cohen_kappa_ci"] = [lo, hi]
        return _attach_judge_timing_metrics(metrics, items)

    human, llm = paired_labels(items, schema)
    metrics = AgreementMetrics(n_pairs=len(human))
    if not human:
        return _attach_judge_timing_metrics(metrics, items)

    metrics.exact_match = exact_match_rate(human, llm)

    if schema.mode == "ordinal":
        metrics.plus_minus_one = plus_minus_one_agreement(human, llm)
        metrics.pearson_r = pearson_correlation(human, llm)
        metrics.spearman_r = spearman_correlation(human, llm)

    if schema.mode in ("binary", "categorical", "ordinal"):
        metrics.pabak = pabak_coefficient(human, llm)
        metrics.cohen_kappa = cohens_kappa(human, llm)
        metrics.krippendorff_alpha = krippendorff_alpha_value(
            human, llm, mode=schema.mode
        )
        prec, rec = precision_recall_for_mode(human, llm, mode=schema.mode)
        metrics.precision = prec
        metrics.recall = rec
        metrics.f1 = f1_for_mode(human, llm, mode=schema.mode)
        metrics.mcc = mcc_for_mode(human, llm, mode=schema.mode)
        metrics.min_pabak = metrics.pabak
        metrics.min_f1 = metrics.f1
        metrics.min_mcc = metrics.mcc

    pair_counter = Counter(zip(human, llm))
    metrics.confusion = {f"{h} → {l}": c for (h, l), c in pair_counter.most_common()}

    disagreements: List[dict] = []
    for item in items:
        h = normalize_label_for_mode(item.human_label, schema)
        l = normalize_label_for_mode(item.llm_label, schema)
        if h is None or l is None or h == l:
            continue
        row = {
            "idx": item.idx,
            "human": h,
            "llm": l,
            "llm_reason": item.llm_reason,
        }
        row.update(_case_disagreement_context(item))
        disagreements.append(row)
        if _disagreement_limit_reached(disagreements, max_disagreements):
            break
    metrics.disagreements = disagreements
    _attach_bootstrap_cis(
        metrics,
        human,
        llm,
        mode=schema.mode,
        bootstrap=bootstrap,
        bootstrap_n=bootstrap_n,
        bootstrap_seed=bootstrap_seed,
    )
    return _attach_judge_timing_metrics(metrics, items)


def format_turn_for_display(turn: dict) -> str:
    if not isinstance(turn, dict):
        return str(turn)
    role_raw = str(turn.get("role") or "").strip().lower()
    content = turn.get("content") or turn.get("message") or turn.get("text") or ""
    if role_raw in ("user", "human", "client", "пользователь"):
        prefix = "👤 Пользователь"
    elif role_raw in ("assistant", "operator", "bot", "ассистент"):
        prefix = "🤖 Ассистент"
    else:
        prefix = role_raw or "?"
    return f"{prefix}: {content}"


def export_calibration_jsonl(items: List[CalibrationItem]) -> str:
    lines: List[str] = []
    for item in items:
        row = {
            "idx": item.idx,
            "goals": item.case.get("goals"),
            "human_label": item.human_label,
            "human_note": item.human_note,
            "llm_label": item.llm_label,
            "llm_reason": item.llm_reason,
            "llm_error": item.llm_error,
        }
        lines.append(json.dumps(row, ensure_ascii=False))
    return "\n".join(lines) + ("\n" if lines else "")


def export_annotated_dataset_jsonl(items: List[CalibrationItem]) -> str:
    """JSONL в формате кейсов бенчмарка + human_labels / human_notes (можно снова загрузить)."""
    lines: List[str] = []
    for item in items:
        row = dict(item.case)
        labels = get_human_labels_dict(row)
        notes = get_human_notes_dict(row)
        if labels:
            row[HUMAN_LABELS_DICT_KEY] = labels
        if notes:
            row[HUMAN_NOTES_DICT_KEY] = notes
        for key in HUMAN_LABEL_KEYS:
            row.pop(key, None)
        row.pop("human_note", None)
        if isinstance(item.llm_label, dict):
            row["llm_labels"] = item.llm_label
        elif item.llm_label is not None:
            row["llm_label"] = item.llm_label
        if item.llm_reason:
            row["llm_reason"] = item.llm_reason
        lines.append(json.dumps(row, ensure_ascii=False))
    return "\n".join(lines) + ("\n" if lines else "")


def sample_size_infinite(z: float, p: float, margin_error: float) -> float:
    """
    Размер выборки для бесконечной совокупности:
    n = Z² × p × (1 − p) / E²
    """
    if margin_error <= 0 or not 0 < p < 1:
        return 0.0
    return (z**2) * p * (1.0 - p) / (margin_error**2)


def recommended_sample_size(
    population_size: int,
    *,
    confidence: float = 0.95,
    p: float = 0.5,
    margin_error: float = 0.05,
) -> int:
    """
    Рекомендуемый размер выборки при известной генеральной совокупности N.
    Сначала n∞ по формуле выше, затем поправка на конечную совокупность:
    n = n∞ / (1 + (n∞ − 1) / N)
    """
    z = CONFIDENCE_Z_SCORES.get(confidence, 1.96)
    n_inf = sample_size_infinite(z, p, margin_error)
    if population_size <= 0:
        return max(1, math.ceil(n_inf))
    if n_inf <= 0:
        return 1
    n_fin = n_inf / (1.0 + (n_inf - 1.0) / population_size)
    return max(1, min(population_size, math.ceil(n_fin)))


def draw_random_sample_indices(
    population_size: int,
    sample_size: int,
    *,
    seed: Optional[int] = None,
) -> List[int]:
    n = max(0, min(int(sample_size), int(population_size)))
    if n <= 0:
        return []
    if n >= population_size:
        return list(range(population_size))
    rng = random.Random(seed)
    return sorted(rng.sample(range(population_size), n))


def judge_config_for_calibration(
    schema: LabelSchema,
    base_config: Dict[str, Any],
) -> Dict[str, Any]:
    """Конфиг судьи для калибровки: поля JSON-ответа берутся из схемы меток."""
    cfg = json.loads(json.dumps(base_config))
    cfg["llm_eval_fields"] = ",".join(schema.llm_eval_fields_list())
    return cfg


def judge_config_with_model(base_config: Dict[str, Any], model: str) -> Dict[str, Any]:
    cfg = json.loads(json.dumps(base_config))
    ev = cfg.setdefault("evaluator", {})
    ev["model"] = model
    return cfg
