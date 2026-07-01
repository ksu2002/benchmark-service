import json
import os
from pathlib import Path
from typing import Optional, Sequence

from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_REPO_ROOT / ".env")

import streamlit as st

st.set_page_config(page_title="Калибровка LLM-судьи", layout="wide")

from judge.calibration import (
    AgreementMetrics,
    CalibrationCompareJob,
    CalibrationItem,
    LabelSchema,
    COMPARE_RANK_MEAN_F1,
    COMPARE_RANK_MEAN_KAPPA,
    COMPARE_RANK_MEAN_PABAK,
    COMPARE_RANK_WORST_F1,
    COMPARE_RANK_WORST_KAPPA,
    COMPARE_RANK_WORST_MCC,
    COMPARE_RANK_WORST_PABAK,
    apply_annotator_to_item,
    calibration_items_from_session,
    calibration_items_to_session,
    cases_to_calibration_items,
    compare_rank_strategy_label,
    compare_variant_rank_key,
    compute_agreement_metrics,
    compute_human_label_statistics,
    compute_inter_annotator_agreement,
    count_annotator_consensus,
    count_items_by_split,
    build_confusion_matrix,
    confusion_to_table_rows,
    count_annotated_by,
    clear_pool_splits,
    draw_random_sample_indices,
    export_annotated_dataset_jsonl,
    export_calibration_jsonl,
    filter_items_by_split,
    format_turn_for_display,
    get_case_split,
    get_human_labels_dict,
    is_item_annotated_by,
    is_item_annotator_consensus,
    items_with_annotator_labels,
    job_model_name,
    judge_config_for_calibration,
    judge_config_with_model,
    JUDGE_SPLIT_TEST,
    JUDGE_SPLIT_TRAIN,
    list_annotators_from_items,
    normalize_annotator_name,
    pool_has_train_test_split,
    run_calibration_compare_jobs,
    set_case_annotator_label,
    stratified_train_test_split,
    kappa_quality_label,
    pabak_quality_label,
    infer_label_schema_from_cases,
    is_item_fully_annotated,
    is_item_llm_scored,
    normalize_label_for_mode,
    normalize_multi_binary_label,
    parse_binary_criteria_csv,
    run_judge_batch,
)
from ui.judge_settings_ui import (
    PRESET_KIND_JUDGE,
    PRESET_KIND_PROMPT,
    apply_judge_config_dict,
    apply_judge_prompt_dict,
    current_judge_config_dict,
    current_prompt_config_dict,
    flush_pending_judge_state,
    init_judge_session_state,
    is_prompt_preset_config,
    load_cases_from_text,
    render_judge_context_prompt_help,
    schedule_apply_judge_saved,
    schedule_judge_evaluator_model,
)
from integrations.litellm import get_model_names
from ui.dialog_results_ui import render_dialog_length_distribution_panel
from dialog_clustering import (
    run_dialog_bertopic_clustering,
    serialize_cluster_tfidf_words,
)
from ui.clustering_ui import (
    CLUSTERING_ALGORITHM_LABEL,
    render_clustering_pipeline_ui,
    render_clustering_results_panel,
)
from ui.sample_storage_ui import render_load_saved_samples_ui


try:
    SUPPORTED_MODELS = get_model_names()
except Exception:
    SUPPORTED_MODELS = [os.getenv("LITELLM_MODEL_NAME", "openai/gpt-4o-mini")]

try:
    from storage.benchmark_backend import (
        delete_judge_sample,
        ensure_schema,
        get_default_judge_preset,
        get_judge_preset,
        get_judge_sample,
        judge_storage_enabled,
        judge_storage_missing_vars,
        list_judge_presets,
        list_judge_samples,
        load_judge_sample_jsonl,
        queue_backend_enabled,
        queue_backend_missing_vars,
        save_judge_preset,
        save_judge_sample,
    )

    _QUEUE_MISSING = queue_backend_missing_vars()
    _STORAGE_MISSING = judge_storage_missing_vars()
    PRESETS_AVAILABLE = queue_backend_enabled()
    SAMPLES_STORAGE_AVAILABLE = judge_storage_enabled()
except ImportError:
    PRESETS_AVAILABLE = False
    SAMPLES_STORAGE_AVAILABLE = False
    _QUEUE_MISSING = ["установите зависимости: psycopg2-binary, pika, minio"]
    _STORAGE_MISSING = _QUEUE_MISSING

init_judge_session_state(
    SUPPORTED_MODELS,
    default_api_key=os.getenv("LITELLM_API_KEY", ""),
    default_model=os.getenv("LITELLM_MODEL_NAME", SUPPORTED_MODELS[0]),
)

if SAMPLES_STORAGE_AVAILABLE or PRESETS_AVAILABLE:
    try:
        ensure_schema()
    except Exception as e:
        st.sidebar.warning(f"Хранилище недоступно: {e}")

if PRESETS_AVAILABLE:
    try:
        if not st.session_state.get("_judge_default_loaded"):
            default_preset = get_default_judge_preset()
            if default_preset:
                dcfg = default_preset.get("config") or {}
                if is_prompt_preset_config(dcfg):
                    apply_judge_prompt_dict(
                        dcfg,
                        preset_name=default_preset.get("name") or "",
                    )
                else:
                    apply_judge_config_dict(
                        dcfg,
                        preset_name=default_preset.get("name") or "",
                    )
            st.session_state["_judge_default_loaded"] = True
    except Exception as e:
        st.sidebar.warning(f"Пресеты недоступны: {e}")

flush_pending_judge_state()


def _get_sample_meta_name() -> str:
    return (st.session_state.get("_judge_sample_meta_name") or "").strip()


def _schedule_sample_meta(
    *,
    name: str = "",
    sample_id: str = "",
    msg: str = "",
    clear: bool = False,
) -> None:
    if clear:
        st.session_state["_pending_judge_sample_meta_clear"] = True
        return
    st.session_state["_pending_judge_sample_meta"] = {
        "name": name,
        "id": sample_id,
        "msg": msg,
    }


def flush_pending_sample_meta() -> None:
    if st.session_state.pop("judge_cal_sample_name", None) is not None:
        pass
    if st.session_state.pop("_pending_judge_sample_meta_clear", False):
        st.session_state["_judge_sample_meta_name"] = ""
        st.session_state["judge_active_sample_id"] = ""
    pending = st.session_state.pop("_pending_judge_sample_meta", None)
    if pending:
        if pending.get("name") is not None:
            st.session_state["_judge_sample_meta_name"] = pending.get("name") or ""
        if pending.get("id") is not None:
            st.session_state["judge_active_sample_id"] = pending.get("id") or ""
        msg = pending.get("msg")
        if msg:
            st.session_state["_judge_dataset_loaded_msg"] = msg


def _schedule_label_schema(
    *,
    label_mode: Optional[str] = None,
    binary_criteria: Optional[str] = None,
    llm_field: Optional[str] = None,
) -> None:
    """Отложенное обновление схемы меток — до отрисовки виджетов на следующем rerun."""
    pending = dict(st.session_state.get("_pending_judge_label_schema") or {})
    if label_mode is not None:
        pending["label_mode"] = label_mode
    if binary_criteria is not None:
        pending["binary_criteria"] = binary_criteria
    if llm_field is not None:
        pending["llm_field"] = llm_field
    st.session_state["_pending_judge_label_schema"] = pending


def flush_pending_label_schema() -> None:
    pending = st.session_state.pop("_pending_judge_label_schema", None)
    if not pending:
        return
    label_mode = pending.get("label_mode")
    if label_mode:
        st.session_state["judge_cal_label_mode"] = label_mode
    if "binary_criteria" in pending:
        st.session_state["judge_cal_binary_criteria"] = pending["binary_criteria"] or ""
    if pending.get("llm_field"):
        st.session_state["judge_cal_llm_field"] = pending["llm_field"]


flush_pending_sample_meta()
flush_pending_label_schema()

CASES_PER_PAGE = 10

# Пояснения к метрикам (help при наведении на st.metric).
_METRIC_HELP = {
    "agreement": (
        "Доля кейсов, где метки полностью совпали. "
        "Для ordinal — совпадение цифр; для binary_multi — по каждому критерию отдельно."
    ),
    "pabak": (
        "Prevalence-adjusted Bias-adjusted Kappa: 2 × Совпадение − 1. "
        "Устойчивее обычного κ при дисбалансе классов."
    ),
    "cohen_kappa": (
        "Cohen's κ: согласие с поправкой на случайное совпадение меток. "
        "> 0.6 — приемлемо, > 0.8 — хорошо."
    ),
    "krippendorff_alpha": (
        "Krippendorff's α: согласие нескольких кодировщиков; устойчив к пропускам. "
        "Для LLM↔human — два кодировщика."
    ),
    "precision": (
        "Precision для положительного класса (binary/binary_multi: 1/да; "
        "categorical/ordinal: macro по классам). "
        "Не определена (—), если в эталоне нет положительных меток."
    ),
    "recall": (
        "Recall для положительного класса. "
        "Не определена (—), если в эталоне нет положительных меток."
    ),
    "pearson": (
        "Pearson r (только ordinal): линейная корреляция оценок. "
        "Чувствителен к систематическому сдвигу шкалы."
    ),
    "spearman": (
        "Spearman ρ (только ordinal): ранговая корреляция — важен порядок, не абсолютные значения."
    ),
    "plus_minus_one": (
        "±1 (только ordinal): доля пар, где оценки отличаются не более чем на 1 балл."
    ),
    "n_pairs": "Число пар «эталон + ответ», по которым посчитаны метрики.",
    "inter_krippendorff": "Согласие всех выбранных разметчиков одновременно.",
    "inter_pairwise_kappa": "Среднее попарное κ между разметчиками.",
    "inter_pairwise_agreement": "Средняя доля полных совпадений в попарных сравнениях разметчиков.",
    "inter_pairwise_pabak": "Средний PABAK по всем парам разметчиков.",
    "judge_reply_sec": "Среднее wall-clock время одного вызова LLM-судьи.",
    "judge_calls": "Число успешных вызовов судьи с замером времени.",
    "f1": (
        "F1 для положительного класса (binary/binary_multi: 1/да; "
        "categorical/ordinal: macro по классам). Устойчивее accuracy при дисбалансе."
    ),
    "mcc": (
        "Matthews Correlation Coefficient (−1…1): балансированная метрика для "
        "бинарной классификации; полезна при редких «1»."
    ),
    "bootstrap_ci": (
        "95% bootstrap-доверительный интервал: пересэмплирование кейсов с "
        "возвращением (1000 итераций). Широкий интервал — мало данных или нестабильность."
    ),
}


def _fmt_bootstrap_ci(raw: Optional[list]) -> str:
    if not raw or len(raw) != 2:
        return "—"
    return f"[{_fmt_num(raw[0])}, {_fmt_num(raw[1])}]"


def _best_compare_variant(
    results: dict,
    *,
    strategy: str,
) -> tuple[str, dict]:
    return max(
        results.items(),
        key=lambda x: compare_variant_rank_key(x[1], strategy=strategy),
    )


def _llm_compare_metrics_for_schema(schema: LabelSchema) -> dict[str, bool]:
    """Какие метрики показываются при сравнении LLM с эталоном разметчика."""
    return {
        "cohen_kappa": True,
        "pabak": True,
        "precision": schema.mode in ("binary", "binary_multi", "categorical", "ordinal"),
        "recall": schema.mode in ("binary", "binary_multi", "categorical", "ordinal"),
        "f1": schema.mode in ("binary", "binary_multi", "categorical", "ordinal"),
    }


def _core_metrics_row(md: dict) -> dict:
    """Строка таблицы: Cohen κ, PABAK, Precision, Recall, F1."""
    kappa = md.get("cohen_kappa")
    return {
        "Cohen κ": _fmt_num(kappa),
        "PABAK": _fmt_num(md.get("pabak")),
        "Precision": _fmt_pct(md.get("precision")),
        "Recall": _fmt_pct(md.get("recall")),
        "F1": _fmt_pct(md.get("f1")),
        "κ качество": md.get("kappa_quality") or kappa_quality_label(kappa),
    }


def _bootstrap_ci_cell(boot: dict, key: str) -> str:
    ci = boot.get(key)
    return _fmt_bootstrap_ci(ci) if ci else "—"


def _inter_rater_metrics_for_schema(schema: LabelSchema) -> dict[str, bool]:
    """Какие метрики считаются при согласии между разметчиками."""
    return {
        "krippendorff_alpha": True,
        "cohen_kappa": True,
        "agreement": True,
        "pabak": True,
        "pearson": schema.mode == "ordinal",
        "spearman": schema.mode == "ordinal",
        "plus_minus_one": schema.mode == "ordinal",
        "per_criterion": schema.mode == "binary_multi",
    }


def _metrics_mode_caption(schema: LabelSchema, *, context: str) -> None:
    """Краткая справка: какие метрики считаются в текущем режиме."""
    if context == "llm":
        mode_labels = {
            "binary": "бинарный (да/нет)",
            "binary_multi": "несколько бинарных критериев",
            "ordinal": "ordinal (шкала)",
            "categorical": "категориальный",
        }
        st.caption(
            f"**Режим:** {mode_labels.get(schema.mode, schema.mode)}. "
            "**Cohen κ** — главный критерий (учитывает случайное совпадение). "
            "**PABAK** — диагностика смещения и дисбаланса классов; сравнивайте разрыв с κ. "
            "**Precision / Recall / F1** — характер ошибок и баланс. "
            "Пороги κ: **< 0.6** — ниже порога, **≥ 0.6** — приемлемо, **> 0.8** — сильный судья."
        )
    elif context == "inter":
        flags = _inter_rater_metrics_for_schema(schema)
        parts = ["**Krippendorff α**", "**Cohen κ**", "**Совпадение**", "**PABAK**"]
        if flags["pearson"]:
            parts.extend(["**Pearson**", "**Spearman**", "**±1**"])
        if flags["per_criterion"]:
            parts.append("таблица **по критериям**")
        st.caption(
            f"Inter-annotator agreement · режим **{schema.mode}**. "
            f"Считаются: {', '.join(parts)}."
        )


def _apply_consensus_sample_filter(
    schema: LabelSchema,
    selected_annotators: list[str],
) -> int:
    """Сузить выборку до кейсов, где все разметчики согласны."""
    pool = _get_pool()
    if not pool:
        return 0
    current = _get_sample_indices(len(pool))
    if not st.session_state.get("judge_cal_sample_indices_before_consensus"):
        st.session_state["judge_cal_sample_indices_before_consensus"] = list(current)
    kept = [
        pool_idx
        for pool_idx in current
        if is_item_annotator_consensus(pool[pool_idx], schema, selected_annotators)
    ]
    st.session_state["judge_cal_sample_indices"] = kept
    st.session_state["judge_cal_consensus_filter_applied"] = True
    _sync_cal_items_from_indices()
    return len(kept)


def _restore_sample_before_consensus() -> None:
    prev = st.session_state.pop("judge_cal_sample_indices_before_consensus", None)
    if prev is not None:
        st.session_state["judge_cal_sample_indices"] = prev
    st.session_state.pop("judge_cal_consensus_filter_applied", None)
    _sync_cal_items_from_indices()


def _current_label_schema() -> LabelSchema:
    cats = [
        c.strip()
        for c in (st.session_state.get("judge_cal_categories") or "").split(",")
        if c.strip()
    ]
    if not cats:
        cats = ["да", "нет"]
    criteria = parse_binary_criteria_csv(
        st.session_state.get("judge_cal_binary_criteria", "")
    )
    if not criteria:
        criteria = ["result"]
    return LabelSchema(
        mode=st.session_state.get("judge_cal_label_mode", "binary"),
        ordinal_min=int(st.session_state.get("judge_cal_ordinal_min", 1)),
        ordinal_max=int(st.session_state.get("judge_cal_ordinal_max", 5)),
        categories=cats,
        llm_field=st.session_state.get("judge_cal_llm_field", "result"),
        binary_criteria=criteria,
    )


def _get_pool() -> list[CalibrationItem]:
    pool = calibration_items_from_session(st.session_state.get("judge_cal_pool") or [])
    if not pool:
        legacy = calibration_items_from_session(st.session_state.get("judge_cal_items") or [])
        if legacy:
            st.session_state["judge_cal_pool"] = calibration_items_to_session(legacy)
            pool = legacy
    return pool


def _sync_cal_items_from_indices() -> None:
    pool = calibration_items_from_session(st.session_state.get("judge_cal_pool") or [])
    if not pool:
        st.session_state["judge_cal_items"] = []
        return
    indices = _get_sample_indices(len(pool))
    st.session_state["judge_cal_items"] = calibration_items_to_session(
        [pool[i] for i in indices]
    )


def _save_pool(pool: list[CalibrationItem]) -> None:
    st.session_state["judge_cal_pool"] = calibration_items_to_session(pool)
    _sync_cal_items_from_indices()


def _get_sample_indices(pool_len: int) -> list[int]:
    raw = st.session_state.get("judge_cal_sample_indices") or []
    valid = sorted({int(i) for i in raw if 0 <= int(i) < pool_len})
    if valid:
        return valid
    if pool_len:
        return list(range(pool_len))
    return []


def _get_cal_items(*, split: Optional[str] = None) -> list[CalibrationItem]:
    pool = _get_pool()
    if not pool:
        return []
    indices = _get_sample_indices(len(pool))
    if split in (JUDGE_SPLIT_TRAIN, JUDGE_SPLIT_TEST):
        indices = [i for i in indices if get_case_split(pool[i].case) == split]
    return [pool[i] for i in indices]


def _split_counts_caption(items: list[CalibrationItem]) -> str:
    c = count_items_by_split(items)
    if c[JUDGE_SPLIT_TRAIN] + c[JUDGE_SPLIT_TEST] == 0:
        return ""
    return (
        f" · train **{c[JUDGE_SPLIT_TRAIN]}** · test **{c[JUDGE_SPLIT_TEST]}**"
        + (f" · без split: **{c['unset']}**" if c["unset"] else "")
    )


def _apply_train_test_split_to_pool(
    *,
    test_ratio: float,
    seed: int,
    annotator: str,
) -> dict:
    pool = _get_pool()
    schema = _current_label_schema()
    stats = stratified_train_test_split(
        pool,
        schema,
        test_ratio=test_ratio,
        seed=seed,
        annotator=annotator or None,
    )
    _save_pool(pool)
    st.session_state["judge_cal_split_stats"] = stats
    return stats


def _clear_train_test_split() -> None:
    pool = _get_pool()
    clear_pool_splits(pool)
    _save_pool(pool)
    st.session_state.pop("judge_cal_split_stats", None)


def _store_best_compare_config(
    schema: LabelSchema,
    label: str,
    md: dict,
) -> None:
    try:
        cfg, _, _ = _judge_cfg_from_compare_variant(schema, label, md)
        st.session_state["judge_cal_best_test_cfg"] = cfg
    except Exception:
        st.session_state.pop("judge_cal_best_test_cfg", None)


def _resolve_test_judge_config(
    schema: LabelSchema,
    compare_results: dict,
) -> tuple[Optional[dict], str]:
    cached = st.session_state.get("judge_cal_best_test_cfg")
    if isinstance(cached, dict) and cached.get("llm_eval_prompt"):
        return cached, "сохранённый лучший вариант сравнения"
    if compare_results:
        rank_strategy = st.session_state.get(
            "judge_cal_rank_strategy", COMPARE_RANK_WORST_KAPPA
        )
        best_name, best_md = _best_compare_variant(
            compare_results, strategy=rank_strategy
        )
        try:
            cfg, _, _ = _judge_cfg_from_compare_variant(schema, best_name, best_md)
            return cfg, best_name
        except Exception:
            pass
    cfg = judge_config_for_calibration(schema, current_judge_config_dict())
    model = (cfg.get("evaluator") or {}).get("model") or "—"
    return cfg, f"текущие настройки судьи · {model}"


def _save_cal_items(subset: list[CalibrationItem]) -> None:
    pool = _get_pool()
    by_idx = {it.idx: it for it in pool}
    for it in subset:
        by_idx[it.idx] = it
    _save_pool([by_idx[k] for k in sorted(by_idx)])


def _apply_sample_selection(n: int, *, seed: Optional[int] = None) -> None:
    pool = _get_pool()
    if not pool:
        return
    n = max(1, min(int(n), len(pool)))
    indices = draw_random_sample_indices(len(pool), n, seed=seed)
    st.session_state["judge_cal_sample_indices"] = indices
    st.session_state["judge_cal_num_to_annotate"] = n
    st.session_state["judge_cal_page"] = 0
    _save_pool(pool)


def _on_dataset_loaded(
    items: list[CalibrationItem],
    *,
    sample_name: str = "",
    sample_id: str = "",
) -> None:
    st.session_state["judge_cal_pool"] = calibration_items_to_session(items)
    st.session_state["judge_cal_sample_indices"] = list(range(len(items)))
    st.session_state["judge_cal_num_to_annotate"] = len(items)
    st.session_state["judge_cal_page"] = 0
    st.session_state.pop("judge_cal_consensus_filter_applied", None)
    st.session_state.pop("judge_cal_sample_indices_before_consensus", None)
    st.session_state.pop("judge_cal_annotation_widgets_annotator", None)
    annotators = list_annotators_from_items(items)
    if len(annotators) == 1:
        only = annotators[0]
        st.session_state["judge_annotator_name"] = only
        st.session_state["judge_cal_annotator_name_input"] = only
        st.session_state["judge_cal_annotator_pick"] = only
    if sample_name or sample_id:
        _schedule_sample_meta(name=sample_name, sample_id=sample_id)
    inferred = infer_label_schema_from_cases([it.case for it in items])
    if inferred:
        _schedule_label_schema(
            label_mode=inferred.mode,
            binary_criteria=",".join(inferred.binary_criteria),
            llm_field=inferred.llm_field if inferred.mode == "binary" else None,
        )
    _sync_cal_items_from_indices()


def _load_judge_sample_from_db(sample_id: str) -> None:
    row = get_judge_sample(sample_id)
    if not row:
        raise ValueError("Выборка не найдена")
    raw = load_judge_sample_jsonl(sample_id)
    parsed, err = load_cases_from_text(raw.decode("utf-8", errors="replace"))
    if err:
        raise ValueError(err)
    items = cases_to_calibration_items(parsed)
    _on_dataset_loaded(
        items,
        sample_name=row.get("name") or "",
        sample_id=str(row["id"]),
    )
    if row.get("label_mode"):
        _schedule_label_schema(label_mode=row["label_mode"])
    crit = row.get("criteria_json") or []
    if crit:
        _schedule_label_schema(binary_criteria=",".join(crit))


def _annotated_count(
    items: list[CalibrationItem],
    annotator: Optional[str] = None,
) -> int:
    schema = _current_label_schema()
    if annotator and normalize_annotator_name(annotator):
        return count_annotated_by(items, annotator, schema)
    return sum(1 for it in items if is_item_fully_annotated(it, schema))


def _get_annotator_name(ui_prefix: str = "judge_cal") -> str:
    key = "judge_annotator_name" if ui_prefix == "judge_cal" else f"{ui_prefix}_annotator_name"
    return normalize_annotator_name(st.session_state.get(key) or "")


def _sync_annotator_pick_to_name(ui_prefix: str) -> None:
    pick_key = _ui_key(ui_prefix, "annotator_pick")
    widget_key = _ui_key(ui_prefix, "annotator_name_input")
    pick = normalize_annotator_name(st.session_state.get(pick_key) or "")
    if pick:
        st.session_state[widget_key] = pick
    st.session_state.pop(f"{ui_prefix}_annotation_widgets_annotator", None)


def _annotation_widgets_sync_key(ui_prefix: str) -> str:
    return f"{ui_prefix}_annotation_widgets_annotator"


def _sync_annotation_widgets_from_cases(
    items: Sequence[CalibrationItem],
    annotator: str,
    schema: LabelSchema,
    *,
    ui_prefix: str = "judge_cal",
) -> int:
    """
    Подставить в session_state виджетов формы метки выбранного разметчика из case.
    Вызывать до отрисовки формы, когда сменился разметчик или загружен датасет.
    """
    ann = normalize_annotator_name(annotator)
    if not ann:
        return 0
    human_key = f"{ui_prefix}_human"
    note_key = f"{ui_prefix}_note"
    loaded = 0
    for item in items:
        apply_annotator_to_item(item, ann, schema)
        has_label = is_item_annotated_by(item, ann, schema)
        if has_label:
            loaded += 1
        if schema.mode == "binary_multi":
            cur_map = (
                normalize_multi_binary_label(item.human_label, schema.binary_criteria)
                or {}
            )
            for crit in schema.binary_criteria:
                st.session_state[f"{human_key}_{item.idx}_{crit}"] = int(
                    cur_map.get(crit, 0)
                )
        elif schema.mode == "binary":
            options = schema.binary_options()
            wkey = f"{human_key}_{item.idx}"
            if item.human_label is not None:
                for val, lbl in options:
                    if item.human_label == val:
                        st.session_state[wkey] = lbl
                        break
            elif wkey in st.session_state:
                del st.session_state[wkey]
        elif schema.mode == "ordinal":
            opts = schema.ordinal_options()
            wkey = f"{human_key}_{item.idx}"
            if item.human_label in opts:
                st.session_state[wkey] = item.human_label
            elif wkey in st.session_state:
                del st.session_state[wkey]
        else:
            opts = schema.categorical_options()
            wkey = f"{human_key}_{item.idx}"
            if item.human_label in opts:
                st.session_state[wkey] = item.human_label
            elif wkey in st.session_state:
                del st.session_state[wkey]
        st.session_state[f"{note_key}_{item.idx}"] = item.human_note or ""
    st.session_state[_annotation_widgets_sync_key(ui_prefix)] = ann
    return loaded


def _ensure_annotation_widgets_for_annotator(
    items: list[CalibrationItem],
    annotator: str,
    schema: LabelSchema,
    *,
    ui_prefix: str = "judge_cal",
    known_annotators: Optional[Sequence[str]] = None,
) -> None:
    ann = normalize_annotator_name(annotator)
    if not ann or not items:
        return
    sync_key = _annotation_widgets_sync_key(ui_prefix)
    if st.session_state.get(sync_key) == ann:
        return
    known = {
        normalize_annotator_name(a)
        for a in (known_annotators or [])
        if normalize_annotator_name(a)
    }
    first_sync = sync_key not in st.session_state
    if ann not in known and not first_sync:
        st.session_state[sync_key] = ann
        return
    loaded = _sync_annotation_widgets_from_cases(
        items, ann, schema, ui_prefix=ui_prefix
    )
    if loaded:
        st.session_state["_judge_annotator_labels_loaded_msg"] = (
            f"Подгружена разметка **{ann}**: **{loaded}** из **{len(items)}** кейсов."
        )


def _render_annotator_name_input(
    *,
    ui_prefix: str = "judge_cal",
    known_annotators: Optional[list[str]] = None,
) -> str:
    state_key = "judge_annotator_name" if ui_prefix == "judge_cal" else f"{ui_prefix}_annotator_name"
    widget_key = _ui_key(ui_prefix, "annotator_name_input")
    known = [a for a in (known_annotators or []) if normalize_annotator_name(a)]
    col_pick, col_name = st.columns([1, 2])
    with col_pick:
        if known:
            st.selectbox(
                "Из списка",
                options=[""] + known,
                format_func=lambda x: "— выбрать —" if not x else x,
                key=_ui_key(ui_prefix, "annotator_pick"),
                on_change=_sync_annotator_pick_to_name,
                args=(ui_prefix,),
            )
    with col_name:
        name = st.text_input(
            "Имя разметчика",
            key=widget_key,
            placeholder="Например: Иванова А.",
            help="Метки сохраняются отдельно для каждого разметчика в поле human_labels.",
        )
    name = normalize_annotator_name(name)
    st.session_state[state_key] = name
    return name


def _render_cal_annotator_selector(items: list[CalibrationItem]) -> list[str]:
    pool = _get_pool()
    annotators = list_annotators_from_items(pool or items)
    st.markdown("**Разметчики для сравнения**")
    if not annotators:
        st.warning(
            "В выборке нет размеченных имён. Укажите имя на вкладке «Датасет» "
            "и сохраните разметку — метки попадут в human_labels."
        )
        return []
    schema = _current_label_schema()
    default = [
        a
        for a in (st.session_state.get("judge_cal_selected_annotators") or [])
        if a in annotators
    ] or annotators[:1]
    selected = st.multiselect(
        "Разметчики",
        options=annotators,
        default=default,
        key="judge_cal_selected_annotators",
        help="Выберите одного или нескольких разметчиков для сравнения с LLM-судьёй.",
    )
    if selected:
        parts = [
            f"**{a}**: {_annotated_count(items, a)} кейсов"
            for a in selected
        ]
        st.caption(" · ".join(parts))
    return selected


def _llm_done_count(items: list[CalibrationItem]) -> int:
    schema = _current_label_schema()
    return sum(1 for it in items if is_item_llm_scored(it, schema))


def _render_judge_timing_metrics(metrics: AgreementMetrics) -> None:
    if not metrics.n_judge_timed and metrics.mean_judge_reply_sec is None:
        return
    rt_cols = st.columns(2)
    with rt_cols[0]:
        rt = metrics.mean_judge_reply_sec
        st.metric(
            "Ср. время ответа судьи",
            f"{rt:.3f} с" if rt is not None else "—",
            help="Среднее wall-clock время одного вызова LLM-судьи на кейс.",
        )
    with rt_cols[1]:
        st.metric(
            "Вызовов судьи",
            metrics.n_judge_timed if metrics.n_judge_timed else "—",
            help=_METRIC_HELP["judge_calls"],
        )


def _render_confusion_matrix_viz(
    confusion: dict,
    *,
    chart_key: str,
) -> bool:
    """Интерактивная heatmap human × LLM. Возвращает ``True``, если матрица отрисована."""
    matrix = build_confusion_matrix(confusion)
    if not matrix or matrix.total <= 0:
        return False

    accuracy = matrix.correct / matrix.total if matrix.total else 0.0
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Всего пар", matrix.total)
    with m2:
        st.metric("Совпадений", matrix.correct)
    with m3:
        st.metric(
            "Совпадение",
            f"{accuracy * 100:.1f}%",
            help=_METRIC_HELP["agreement"],
        )

    row_totals = [sum(row) for row in matrix.counts]
    max_count = max((cell for row in matrix.counts for cell in row), default=1)
    text_cells: list[list[str]] = []
    hover_cells: list[list[str]] = []
    color_cells: list[list[float]] = []

    for row_idx, row in enumerate(matrix.counts):
        human = matrix.human_labels[row_idx]
        row_total = row_totals[row_idx]
        text_row: list[str] = []
        hover_row: list[str] = []
        color_row: list[float] = []
        for col_idx, count in enumerate(row):
            llm = matrix.llm_labels[col_idx]
            share = (100.0 * count / row_total) if row_total else 0.0
            if count:
                text_row.append(f"{count}\n{share:.0f}%")
            else:
                text_row.append("")
            match = human == llm
            hover_row.append(
                f"Эталон: {human}<br>LLM: {llm}<br>"
                f"Кол-во: {count}<br>Доля строки: {share:.1f}%<br>"
                f"{'Совпадение' if match else 'Расхождение'}"
            )
            if count <= 0:
                color_row.append(0.0)
            elif match:
                color_row.append(0.55 + 0.45 * (count / max_count))
            else:
                color_row.append(0.45 * (count / max_count))
        text_cells.append(text_row)
        hover_cells.append(hover_row)
        color_cells.append(color_row)

    try:
        import plotly.graph_objects as go

        fig = go.Figure(
            data=go.Heatmap(
                z=color_cells,
                x=matrix.llm_labels,
                y=matrix.human_labels,
                text=text_cells,
                hovertext=hover_cells,
                hoverinfo="text",
                texttemplate="%{text}",
                textfont={"size": 13},
                colorscale=[
                    [0.0, "#f3f4f6"],
                    [0.01, "#fee2e2"],
                    [0.25, "#fca5a5"],
                    [0.44, "#fecaca"],
                    [0.46, "#ffffff"],
                    [0.54, "#ffffff"],
                    [0.75, "#bbf7d0"],
                    [1.0, "#16a34a"],
                ],
                showscale=False,
                xgap=2,
                ygap=2,
            )
        )
        fig.update_layout(
            title="Матрица ошибок (human → LLM)",
            xaxis_title="Ответ LLM",
            yaxis_title="Эталон (human)",
            height=max(320, 90 + 46 * len(matrix.human_labels)),
            margin={"l": 20, "r": 20, "t": 50, "b": 20},
            yaxis={"autorange": "reversed"},
        )
        st.plotly_chart(fig, use_container_width=True, key=chart_key)
        st.caption(
            "Строки — эталон разметчика, столбцы — ответ LLM. "
            "Зелёные ячейки — совпадения, красные — расхождения; "
            "число — количество кейсов, процент — доля от строки."
        )
    except ImportError:
        import pandas as pd

        st.warning("Установите plotly для heatmap: `pip install plotly`")
        table_rows = []
        for row_idx, row in enumerate(matrix.counts):
            human = matrix.human_labels[row_idx]
            row_total = row_totals[row_idx]
            for col_idx, count in enumerate(row):
                if count <= 0:
                    continue
                llm = matrix.llm_labels[col_idx]
                share = (100.0 * count / row_total) if row_total else 0.0
                table_rows.append(
                    {
                        "Эталон (human)": human,
                        "Ответ LLM": llm,
                        "Кол-во": count,
                        "Доля строки, %": round(share, 1),
                        "Расхождение": human != llm,
                    }
                )
        if table_rows:
            st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

    return True


def _render_bootstrap_metrics(metrics: AgreementMetrics) -> None:
    boot = metrics.bootstrap or {}
    if not boot:
        return
    parts = []
    if boot.get("cohen_kappa_ci"):
        parts.append(f"Cohen κ {_fmt_bootstrap_ci(boot['cohen_kappa_ci'])}")
    if boot.get("pabak_ci"):
        parts.append(f"PABAK {_fmt_bootstrap_ci(boot['pabak_ci'])}")
    if boot.get("f1_ci"):
        parts.append(f"F1 {_fmt_bootstrap_ci(boot['f1_ci'])}")
    if parts:
        st.caption(
            "**95% bootstrap CI:** " + " · ".join(parts) + "  \n"
            + _METRIC_HELP["bootstrap_ci"]
        )


def _render_metrics_panel(metrics: AgreementMetrics, schema: LabelSchema) -> None:
    _metrics_mode_caption(schema, context="llm")

    if schema.mode == "binary_multi" and metrics.per_criterion:
        st.markdown("**Метрики по критериям**")
        rows = []
        for crit, md in metrics.per_criterion.items():
            row = {"Критерий": crit, "N": md.get("n_pairs")}
            row.update(_core_metrics_row(md))
            rows.append(row)
        st.dataframe(rows, use_container_width=True, hide_index=True)
        avg_parts = [
            f"Cohen κ **{_fmt_num(metrics.cohen_kappa)}**",
            f"PABAK **{_fmt_num(metrics.pabak)}**",
            f"Precision **{_fmt_pct(metrics.precision)}**",
            f"Recall **{_fmt_pct(metrics.recall)}**",
            f"F1 **{_fmt_pct(metrics.f1)}**",
        ]
        if metrics.cohen_kappa is not None and metrics.pabak is not None:
            gap = abs(metrics.cohen_kappa - metrics.pabak)
            avg_parts.append(f"разрыв κ−PABAK **{_fmt_num(gap)}**")
        st.caption("Среднее по критериям: " + ", ".join(avg_parts) + ".")
        _render_bootstrap_metrics(metrics)
        _render_judge_timing_metrics(metrics)

    if metrics.n_pairs == 0 and not metrics.per_criterion:
        _render_judge_timing_metrics(metrics)
        st.warning("Нет пар «человек + LLM» для расчёта. Сначала разметьте кейсы и прогоните судью.")
        return

    if schema.mode == "binary_multi":
        if metrics.disagreements:
            with st.expander("Расхождения по критериям", expanded=False):
                for d in metrics.disagreements:
                    st.markdown(
                        f"**#{d.get('idx')}** · `{d.get('criterion')}` · "
                        f"human=`{d.get('human')}` · llm=`{d.get('llm')}`  \n"
                        f"{d.get('goals', '')}"
                    )
        if not metrics.per_criterion:
            _render_judge_timing_metrics(metrics)
        return

    st.markdown(f"**Сравнено пар:** {metrics.n_pairs}")

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric(
            "Cohen's κ",
            _fmt_num(metrics.cohen_kappa),
            help=_METRIC_HELP["cohen_kappa"],
        )
        if metrics.cohen_kappa is not None:
            st.caption(kappa_quality_label(metrics.cohen_kappa))
    with c2:
        st.metric("PABAK", _fmt_num(metrics.pabak), help=_METRIC_HELP["pabak"])
        if metrics.pabak is not None:
            st.caption(kappa_quality_label(metrics.pabak))
    with c3:
        st.metric("Precision", _fmt_pct(metrics.precision), help=_METRIC_HELP["precision"])
    with c4:
        st.metric("Recall", _fmt_pct(metrics.recall), help=_METRIC_HELP["recall"])
    with c5:
        st.metric("F1", _fmt_pct(metrics.f1), help=_METRIC_HELP["f1"])

    if (
        metrics.cohen_kappa is not None
        and metrics.pabak is not None
        and metrics.cohen_kappa != metrics.pabak
    ):
        st.caption(
            f"Разрыв Cohen κ − PABAK: **{_fmt_num(abs(metrics.cohen_kappa - metrics.pabak))}** — "
            "может указывать на смещение судьи (слишком часто ставит «1» или «0»)."
        )

    _render_judge_timing_metrics(metrics)
    _render_bootstrap_metrics(metrics)

    if metrics.disagreements:
        with st.expander("Расхождения (первые 20)", expanded=False):
            for d in metrics.disagreements:
                st.markdown(
                    f"**#{d.get('idx')}** · human=`{d.get('human')}` · llm=`{d.get('llm')}`  \n"
                    f"Цель: {d.get('goals') or '—'}  \n"
                    f"Причина LLM: {d.get('llm_reason') or '—'}"
                )


def _render_judge_settings_block() -> None:
    col_model, col_key = st.columns([2, 2])
    with col_model:
        model_idx = 0
        cur_model = st.session_state.get("judge_evaluator_model", SUPPORTED_MODELS[0])
        if cur_model in SUPPORTED_MODELS:
            model_idx = SUPPORTED_MODELS.index(cur_model)
        st.selectbox(
            "Модель оценщика",
            options=SUPPORTED_MODELS,
            index=min(model_idx, len(SUPPORTED_MODELS) - 1),
            key="judge_evaluator_model",
        )
    with col_key:
        st.text_input("API Key", type="password", key="judge_evaluator_api_key")

    st.text_area("Параметры JSON", height=70, key="judge_evaluator_params_json")
    st.text_area("Промпт LLM-судьи", height=200, key="judge_llm_eval_prompt")
    render_judge_context_prompt_help([item.case for item in _get_pool()])
    st.text_input(
        "Поля JSON-ответа (через запятую)",
        key="judge_llm_eval_fields",
        help="Для бинарной оценки: result,reason. Для шкалы добавьте score.",
    )

def _saved_preset_label(preset: dict) -> str:
    cfg = preset.get("config") or {}
    kind = "Промпт" if is_prompt_preset_config(cfg) else "Судья"
    return f"{kind}: {preset.get('name') or '—'}"


def _render_judge_presets_save_apply() -> None:
    st.subheader("Сохранение и применение")
    st.caption(
        "Сохраните **судью** (модель, промпт, поля ответа) или только **промпт** "
        "(текст промпта). Датасет и разметка не входят."
    )

    active_preset = st.session_state.get("judge_active_preset_name", "").strip()
    if active_preset:
        st.caption(f"Последнее применённое: **{active_preset}**")

    if not PRESETS_AVAILABLE:
        st.info(f"Пресеты в БД недоступны: {', '.join(_QUEUE_MISSING)}")
    else:
        presets = list_judge_presets()
        preset_by_id = {str(p["id"]): p for p in presets}
        preset_ids = list(preset_by_id.keys())
        pc = st.columns([3, 1])
        with pc[0]:
            selected_preset_id = st.selectbox(
                "Сохранённое",
                options=[""] + preset_ids,
                format_func=lambda pid: (
                    "— не выбрано —"
                    if not pid
                    else _saved_preset_label(preset_by_id[pid])
                ),
                key="judge_preset_select",
            )
        with pc[1]:
            if st.button("Применить", disabled=not selected_preset_id, key="judge_load_preset"):
                loaded = get_judge_preset(selected_preset_id)
                if loaded:
                    schedule_apply_judge_saved(
                        loaded.get("config") or {},
                        preset_name=loaded.get("name") or "",
                    )
                    st.rerun()

        with st.form("judge_save_preset_form"):
            save_kind = st.radio(
                "Что сохранить",
                [PRESET_KIND_JUDGE, PRESET_KIND_PROMPT],
                format_func=lambda k: "Судья (полная конфигурация)" if k == PRESET_KIND_JUDGE else "Промпт",
                horizontal=True,
            )
            save_name = st.text_input("Имя")
            save_desc = st.text_input("Описание")
            if st.form_submit_button("Сохранить", type="primary"):
                if not save_name.strip():
                    st.error("Укажите имя")
                else:
                    cfg = (
                        current_prompt_config_dict()
                        if save_kind == PRESET_KIND_PROMPT
                        else current_judge_config_dict()
                    )
                    save_judge_preset(
                        save_name.strip(),
                        cfg,
                        description=save_desc,
                    )
                    st.session_state["judge_active_preset_name"] = save_name.strip()
                    st.rerun()


def _render_load_saved_samples_source(
    *,
    select_key: str = "judge_saved_sample_select",
    load_key: str = "judge_load_saved_sample",
    delete_key: str = "judge_delete_saved_sample",
    msg_key: str = "_judge_dataset_loaded_msg",
    show_delete: bool = True,
    caption: Optional[str] = None,
) -> None:
    if caption is None:
        caption = "Именованные выборки с разметкой, сохранённые ранее на этой странице."

    active_name = _get_sample_meta_name()
    active_id = (st.session_state.get("judge_active_sample_id") or "").strip()
    if active_name:
        st.caption(
            f"Текущая выборка: **{active_name}**"
            + (f" (id: {active_id[:8]}…)" if active_id else "")
        )

    def _on_load(_raw: bytes, row: dict) -> None:
        _load_judge_sample_from_db(str(row["id"]))
        st.session_state[msg_key] = f"Загружена выборка «{row.get('name')}»."
        st.rerun()

    def _on_after_delete(sample_id: str) -> None:
        if st.session_state.get("judge_active_sample_id") == sample_id:
            _schedule_sample_meta(clear=True)

    render_load_saved_samples_ui(
        key_prefix="judge_saved",
        on_load=_on_load,
        show_delete=show_delete,
        caption=caption,
        select_key=select_key,
        load_key=load_key,
        delete_key=delete_key,
        on_after_delete=_on_after_delete if show_delete else None,
    )


def _load_uploaded_jsonl_cases(
    uploaded,
    *,
    msg_key: str,
    success_prefix: str = "Загружено",
) -> None:
    if uploaded is None:
        return
    st.caption(f"Файл: **{uploaded.name}** ({uploaded.size} байт)")
    if st.button("Загрузить", type="primary", key=f"{msg_key}_btn"):
        text = uploaded.getvalue().decode("utf-8", errors="replace")
        if uploaded.name.endswith(".json"):
            try:
                text = json.dumps(json.loads(text), ensure_ascii=False)
            except json.JSONDecodeError as e:
                st.error(f"Невалидный JSON: {e}")
                text = ""
        if text:
            parsed, err = load_cases_from_text(text)
            if err:
                st.error(err)
            elif parsed:
                items = cases_to_calibration_items(parsed)
                _on_dataset_loaded(items)
                st.session_state[msg_key] = f"{success_prefix} **{len(items)}** кейсов."
                st.rerun()


def _render_cal_sample_step() -> bool:
    """Шаг 1 калибровки: загрузка размеченной выборки."""
    st.subheader("1. Размеченная выборка")

    loaded_msg = st.session_state.pop("_judge_cal_loaded_msg", None)
    if loaded_msg:
        st.success(loaded_msg)

    src = st.radio(
        "Источник",
        ["Из системы", "Из файла"],
        horizontal=True,
        key="judge_cal_sample_src",
    )
    if src == "Из системы":
        _render_load_saved_samples_source(
            select_key="judge_cal_saved_sample_select",
            load_key="judge_cal_load_saved_sample",
            msg_key="_judge_cal_loaded_msg",
            show_delete=False,
            caption="Сохранённые выборки с разметкой.",
        )
    else:
        st.caption(
            "JSONL / JSON с полями `human_labels` (метки по разметчикам) — "
            "размеченный датасет с вкладки «Разметка» или экспорт."
        )
        uploaded = st.file_uploader(
            "Размеченный датасет (.jsonl / .json)",
            type=["jsonl", "json", "txt"],
            key="judge_cal_sample_upload",
        )
        _load_uploaded_jsonl_cases(
            uploaded,
            msg_key="_judge_cal_loaded_msg",
            success_prefix="Загружено",
        )

    pool = _get_pool()
    items = _get_cal_items()
    if not pool or not items:
        st.info("Загрузите размеченную выборку из системы или файла.")
        return False

    ann_n = _annotated_count(items)
    annotators = list_annotators_from_items(pool)
    sample_name = _get_sample_meta_name()
    name_part = f" · **{sample_name}**" if sample_name else ""
    ann_part = (
        f" · разметчики: **{', '.join(annotators)}**"
        if annotators
        else ""
    )
    st.caption(
        f"В работе: **{len(items)}** кейсов{name_part}{ann_part}{_split_counts_caption(items)} · "
        f"размечено (любой): **{ann_n}** · оценено LLM: **{_llm_done_count(items)}**"
    )
    if pool_has_train_test_split(_get_pool()):
        train_n = len(filter_items_by_split(items, JUDGE_SPLIT_TRAIN))
        test_n = len(filter_items_by_split(items, JUDGE_SPLIT_TEST))
        st.caption(
            f"Train/test: **train {train_n}** (сравнение промптов) · "
            f"**test {test_n}** (финальная оценка + bootstrap)"
        )
    if ann_n == 0:
        st.warning(
            "В выборке нет разметки. Загрузите файл с human_labels или "
            "разметьте на вкладке «Разметка» с указанием имени разметчика."
        )
        return False
    if ann_n < len(items):
        st.info(
            f"Частичная разметка: **{ann_n}** из **{len(items)}**. "
            "Прогон возможен только по размеченным кейсам."
        )
    return True


CAL_COMPARE_PROMPTS = "prompts_models"
CAL_COMPARE_JUDGES = "judges"


def _prompt_presets(presets: list[dict]) -> list[dict]:
    return [p for p in presets if is_prompt_preset_config(p.get("config") or {})]


def _cfg_with_prompt_overlay(
    schema: LabelSchema,
    prompt_config: dict,
    model: Optional[str] = None,
) -> dict:
    merged = current_judge_config_dict()
    merged["llm_eval_prompt"] = prompt_config.get("llm_eval_prompt") or merged.get(
        "llm_eval_prompt"
    )
    merged["use_tools"] = bool(
        prompt_config.get("use_tools", merged.get("use_tools", False))
    )
    merged["assistant_prompt"] = prompt_config.get("assistant_prompt") or merged.get(
        "assistant_prompt", ""
    )
    merged["user_prompt"] = prompt_config.get("user_prompt") or merged.get("user_prompt", "")
    merged["assistant_tools"] = prompt_config.get("assistant_tools") or merged.get(
        "assistant_tools", "[]"
    )
    cfg = judge_config_for_calibration(schema, merged)
    if model:
        cfg = judge_config_with_model(cfg, model)
    return cfg


def _preset_option_label(preset: dict) -> str:
    cfg = preset.get("config") or {}
    if is_prompt_preset_config(cfg):
        return f"{preset.get('name') or '—'} · промпт"
    model = (cfg.get("evaluator") or {}).get("model") or "—"
    return f"{preset.get('name') or '—'} · {model}"


def _full_judge_presets(presets: list[dict]) -> list[dict]:
    return [p for p in presets if not is_prompt_preset_config(p.get("config") or {})]


def _build_prompt_model_jobs(
    schema: LabelSchema,
    prompt_by_id: dict[str, dict],
    prompt_ids: list[str],
    models: list[str],
) -> list[CalibrationCompareJob]:
    jobs: list[CalibrationCompareJob] = []
    for prompt_id in prompt_ids:
        row = prompt_by_id.get(prompt_id) or get_judge_preset(prompt_id)
        if not row:
            continue
        pcfg = row.get("config") or {}
        pname = row.get("name") or prompt_id
        for model in models:
            jobs.append(
                CalibrationCompareJob(
                    result_key=f"{pname} · {model}",
                    config=_cfg_with_prompt_overlay(schema, pcfg, model),
                    meta={
                        "prompt_id": prompt_id,
                        "prompt_name": pname,
                        "model": model,
                        "compare_kind": CAL_COMPARE_PROMPTS,
                    },
                )
            )
    return jobs


def _build_judge_jobs(
    schema: LabelSchema,
    judge_by_id: dict[str, dict],
    judge_ids: list[str],
) -> list[CalibrationCompareJob]:
    jobs: list[CalibrationCompareJob] = []
    for preset_id in judge_ids:
        row = judge_by_id.get(preset_id) or get_judge_preset(preset_id)
        if not row:
            continue
        cfg = judge_config_for_calibration(schema, row.get("config") or {})
        jobs.append(
            CalibrationCompareJob(
                result_key=row.get("name") or preset_id,
                config=cfg,
                meta={
                    "preset_id": preset_id,
                    "model": job_model_name(cfg),
                    "compare_kind": CAL_COMPARE_JUDGES,
                },
            )
        )
    return jobs


def _rebuild_compare_jobs_from_results(
    schema: LabelSchema,
    compare_results: dict,
) -> list[CalibrationCompareJob]:
    """Восстановить уникальные jobs из результатов train-сравнения."""
    if not compare_results or not PRESETS_AVAILABLE:
        return []
    jobs: list[CalibrationCompareJob] = []
    seen: set[str] = set()
    for md in compare_results.values():
        kind = md.get("compare_kind")
        if kind == CAL_COMPARE_JUDGES:
            pid = str(md.get("preset_id") or "")
            if not pid or pid in seen:
                continue
            seen.add(pid)
            row = get_judge_preset(pid)
            if not row:
                continue
            cfg = judge_config_for_calibration(schema, row.get("config") or {})
            jobs.append(
                CalibrationCompareJob(
                    result_key=row.get("name") or pid,
                    config=cfg,
                    meta={
                        "preset_id": pid,
                        "model": job_model_name(cfg),
                        "compare_kind": CAL_COMPARE_JUDGES,
                    },
                )
            )
        else:
            prompt_id = str(md.get("prompt_id") or "")
            model = str(md.get("model") or "")
            dedupe = f"p:{prompt_id}:{model}"
            if not prompt_id or not model or dedupe in seen:
                continue
            seen.add(dedupe)
            row = get_judge_preset(prompt_id)
            if not row:
                continue
            pcfg = row.get("config") or {}
            pname = md.get("prompt_name") or row.get("name") or prompt_id
            jobs.append(
                CalibrationCompareJob(
                    result_key=f"{pname} · {model}",
                    config=_cfg_with_prompt_overlay(schema, pcfg, model),
                    meta={
                        "prompt_id": prompt_id,
                        "prompt_name": pname,
                        "model": model,
                        "compare_kind": CAL_COMPARE_PROMPTS,
                    },
                )
            )
    return jobs


def _execute_calibration_compare(
    items: list[CalibrationItem],
    schema: LabelSchema,
    jobs: list[CalibrationCompareJob],
    *,
    compare_mode: str,
    selected_annotators: list[str],
    bootstrap: bool = False,
    max_disagreements: Optional[int] = None,
    status_title: str = "Сравнение",
) -> tuple[dict, dict[str, list[CalibrationItem]]]:
    annotated = max(
        (_annotated_count(items, a) for a in selected_annotators),
        default=0,
    )
    models = sorted({job_model_name(j.config) for j in jobs})
    parallel = len(jobs) > 1
    total_result_slots = len(jobs) * len(selected_annotators)
    lines: list[str] = []
    done = 0

    with st.status(
        f"{status_title}: **{len(jobs)}** вариант(ов) · **{len(selected_annotators)}** разметчик(ов) · "
        f"**{annotated}** размеченных кейсов"
        + (f" · параллельно (**{len(jobs)}** потоков)" if parallel else "")
        + (" · **bootstrap 95% CI**" if bootstrap else ""),
        expanded=True,
    ) as status:
        st.markdown(
            f"**Модели:** {', '.join(f'`{m}`' for m in models)}  \n"
            f"**Разметчики:** {', '.join(f'`{a}`' for a in selected_annotators)}  \n"
            f"**Режим:** {'параллельный' if parallel else 'последовательный'} прогон"
            + (f" ({len(jobs)} jobs)" if parallel else "")
        )
        log_box = st.empty()
        prog = st.progress(0.0, text="Старт…")

        def on_start(job: CalibrationCompareJob, model: str) -> None:
            lines.append(f"▶ **{job.result_key}** · модель `{model}`")
            log_box.markdown("\n\n".join(lines[-10:]))

        def on_done(key: str, metrics: dict, elapsed: float) -> None:
            nonlocal done
            done += 1
            scored_n = metrics.get("cases_scored", 0)
            ann = metrics.get("cases_annotated", annotated)
            err_n = int(metrics.get("cases_errors") or 0)
            err_part = f" · ⚠️ **{err_n}** ошибок" if err_n else ""
            if lines and lines[-1].startswith("▶") and key in lines[-1]:
                lines[-1] = (
                    f"✓ **{key}** · `{metrics.get('model', '—')}` · "
                    f"оценено **{scored_n}/{ann}**{err_part} · "
                    f"κ **{_fmt_num(metrics.get('cohen_kappa'))}** · "
                    f"PABAK **{_fmt_num(metrics.get('pabak'))}** · "
                    f"P **{_fmt_pct(metrics.get('precision'))}** · "
                    f"R **{_fmt_pct(metrics.get('recall'))}** · "
                    f"F1 **{_fmt_pct(metrics.get('f1'))}** · "
                    f"**{elapsed:.1f}** с"
                )
            else:
                lines.append(
                    f"✓ **{key}** · оценено {err_n}/{ann} · {elapsed:.1f} с"
                )
            log_box.markdown("\n\n".join(lines[-10:]))
            prog.progress(
                done / max(1, total_result_slots),
                text=f"Готово {done}/{total_result_slots}",
            )

        def on_case(key: str, cur: int, total: int, model: str) -> None:
            prog.progress(
                (done + cur / max(total, 1)) / max(1, total_result_slots),
                text=f"**{key}** · кейс {cur}/{total} · `{model}`",
            )

        results, scored_batches = run_calibration_compare_jobs(
            items,
            schema,
            jobs,
            parallel_models=True,
            annotators=selected_annotators,
            bootstrap=bootstrap,
            max_disagreements=max_disagreements,
            on_job_start=on_start,
            on_job_done=on_done,
            on_case_progress=on_case,
        )
        total_sec = sum(float(m.get("elapsed_sec") or 0) for m in results.values())
        status.update(label="Сохранение меток лучшего варианта…", state="running")
        _sync_items_after_compare(
            items,
            schema,
            results,
            compare_mode=compare_mode,
            scored_batches=scored_batches,
        )
        status.update(
            label=(
                f"Готово: {len(results)} вариант(ов) · "
                f"суммарно {total_sec:.1f} с"
                + (f" (параллельно, {len(jobs)} jobs)" if parallel else "")
            ),
            state="complete",
        )

    return results, scored_batches


def _render_disagreement_dialogue(d: dict) -> None:
    st.markdown(f"**Цель:** {d.get('goals') or '—'}")
    st.markdown(
        f"**Эталон (human):** `{d.get('human')}` · "
        f"**LLM:** `{d.get('llm')}`"
        + (f" · **критерий:** `{d.get('criterion')}`" if d.get("criterion") else "")
    )
    reason = (d.get("llm_reason") or "").strip()
    if reason:
        st.markdown(f"**Обоснование LLM:** {reason}")
    history = d.get("history") or []
    if history:
        st.markdown("**Диалог**")
        for turn in history:
            st.text(format_turn_for_display(turn))
    else:
        st.caption("История диалога пуста")


def _render_compare_label_disagreements(
    results: dict,
    schema: LabelSchema,
    *,
    name_col: str,
) -> None:
    if not results:
        return

    st.markdown("---")
    st.subheader("Расхождения с разметкой")
    st.caption(
        "Для каждого варианта сравнения — матрица ошибок (human → LLM) "
        "и кейсы, где ответ судьи не совпал с человеческой разметкой."
    )

    options = list(results.keys())
    selected = st.selectbox(
        name_col,
        options=options,
        key="judge_cal_compare_disagree_variant",
    )
    md = results.get(selected) or {}
    model = md.get("model", "—")
    has_details = bool(
        md.get("confusion") or md.get("disagreements") or md.get("per_criterion")
    )
    if not has_details:
        st.info(
            "Для этого варианта нет сохранённых расхождений. "
            "Запустите сравнение заново, чтобы собрать матрицу и диалоги."
        )
        return
    st.markdown(
        f"**Модель:** `{model}` · **расхождений:** "
        f"{md.get('mismatches_count', len(md.get('disagreements') or []))}"
    )

    if schema.mode == "binary_multi":
        per = md.get("per_criterion") or {}
        criteria = list(per.keys()) or schema.binary_criteria
        crit = st.selectbox(
            "Критерий для матрицы",
            options=criteria,
            key="judge_cal_compare_disagree_crit",
        )
        confusion = (per.get(crit) or {}).get("confusion") or {}
        disagreements = [
            d for d in (md.get("disagreements") or []) if d.get("criterion") == crit
        ]
    else:
        confusion = md.get("confusion") or {}
        disagreements = md.get("disagreements") or []

    matrix_rows = confusion_to_table_rows(confusion)
    if matrix_rows:
        st.markdown("**Матрица ошибок**")
        chart_key = "judge_cal_compare_confusion"
        if schema.mode == "binary_multi":
            chart_key = f"judge_cal_compare_confusion_{crit}"
        _render_confusion_matrix_viz(
            confusion,
            chart_key=chart_key,
        )
        mismatch_rows = [r for r in matrix_rows if r.get("Расхождение")]
        if mismatch_rows:
            st.caption(
                f"Несовпадающих типов пар: **{len(mismatch_rows)}** · "
                f"всего ошибочных предсказаний: "
                f"**{sum(r['Кол-во'] for r in mismatch_rows)}**"
            )
    else:
        st.info("Нет пар меток для матрицы.")

    if disagreements:
        st.markdown(f"**Кейсы с расхождениями ({len(disagreements)})**")
        for d in disagreements:
            crit_part = f" · `{d.get('criterion')}`" if d.get("criterion") else ""
            with st.expander(
                f"#{d.get('idx')} · human=`{d.get('human')}` · llm=`{d.get('llm')}`{crit_part}",
                expanded=False,
            ):
                _render_disagreement_dialogue(d)
    elif matrix_rows and any(r.get("Расхождение") for r in matrix_rows):
        st.success("Все расхождения отражены в матрице; отдельных кейсов нет.")
    else:
        st.success("Полное совпадение с человеческой разметкой.")


def _render_compare_run_errors(results: dict) -> None:
    variants_with_errors = [
        (name, md) for name, md in results.items() if (md.get("run_errors") or [])
    ]
    if not variants_with_errors:
        return

    st.markdown("**Ошибки прогона**")
    for name, md in variants_with_errors:
        errs = md.get("run_errors") or []
        model = md.get("model", "—")
        with st.expander(
            f"{name} · `{model}` — {len(errs)} ошибок",
            expanded=False,
        ):
            for err in errs:
                st.markdown(
                    f"**Кейс #{err.get('idx')}**  \n"
                    f"Цель: {err.get('goals') or '—'}"
                )
                st.code(err.get("error") or "—", language=None)


def _render_comparison_results_table(
    results: dict,
    *,
    name_col: str,
    schema: LabelSchema,
    per_entity_label: str,
    show_bootstrap: bool = False,
) -> Optional[tuple[str, dict]]:
    if not results:
        return None
    _metrics_mode_caption(schema, context="llm")
    rows = []
    for name, md in results.items():
        row: dict = {
            name_col: name,
            "Разметчик": md.get("annotator") or "—",
            "Модель": md.get("model", "—"),
            "N": md.get("n_pairs"),
        }
        row.update(_core_metrics_row(md))
        if show_bootstrap:
            boot = md.get("bootstrap") or {}
            row["κ 95% CI"] = _bootstrap_ci_cell(boot, "cohen_kappa_ci")
            row["PABAK 95% CI"] = _bootstrap_ci_cell(boot, "pabak_ci")
            row["F1 95% CI"] = _bootstrap_ci_cell(boot, "f1_ci")
        kappa = md.get("cohen_kappa")
        pabak = md.get("pabak")
        if kappa is not None and pabak is not None:
            row["κ−PABAK"] = _fmt_num(abs(kappa - pabak))
        row["Ошибки"] = md.get("cases_errors", 0)
        rows.append(row)
    st.dataframe(rows, use_container_width=True, hide_index=True)
    _render_compare_run_errors(results)
    _render_compare_label_disagreements(results, schema, name_col=name_col)
    if schema.mode == "binary_multi":
        with st.expander(f"По критериям для каждого варианта ({per_entity_label})"):
            for name, md in results.items():
                per = md.get("per_criterion") or {}
                if per:
                    st.markdown(f"**{name}**")
                    per_rows = []
                    for c, v in per.items():
                        pr: dict = {"Критерий": c, "N": v.get("n_pairs")}
                        pr.update(_core_metrics_row(v))
                        per_rows.append(pr)
                    st.dataframe(
                        per_rows,
                        use_container_width=True,
                        hide_index=True,
                    )
    rank_strategy = st.session_state.get(
        "judge_cal_rank_strategy", COMPARE_RANK_WORST_KAPPA
    )
    return _best_compare_variant(results, strategy=rank_strategy)


def _judge_cfg_from_compare_variant(
    schema: LabelSchema,
    label: str,
    md: dict,
) -> tuple[dict, str, Optional[str]]:
    """Собрать полный конфиг судьи из лучшего варианта сравнения."""
    if md.get("compare_kind") == CAL_COMPARE_JUDGES:
        loaded = get_judge_preset(str(md.get("preset_id") or ""))
        if not loaded:
            raise ValueError("Судья не найден")
        cfg = dict(loaded.get("config") or {})
        cfg["preset_kind"] = PRESET_KIND_JUDGE
        return cfg, (loaded.get("name") or label).strip(), str(md.get("preset_id") or "") or None

    prompt_id = md.get("prompt_id")
    loaded = get_judge_preset(str(prompt_id or ""))
    if not loaded:
        raise ValueError("Промпт не найден")
    model = str(md.get("model") or "").strip()
    if not model:
        raise ValueError("Не указана модель")
    cfg = _cfg_with_prompt_overlay(schema, loaded.get("config") or {}, model)
    save_name = (md.get("prompt_name") or label).strip()
    if model and model not in save_name:
        save_name = f"{save_name} · {model}"
    return cfg, save_name, None


def _sync_items_after_compare(
    items: list[CalibrationItem],
    schema: LabelSchema,
    results: dict,
    *,
    compare_mode: str,
    scored_batches: Optional[dict[str, list[CalibrationItem]]] = None,
) -> None:
    if not results:
        return
    rank_strategy = st.session_state.get(
        "judge_cal_rank_strategy", COMPARE_RANK_WORST_KAPPA
    )
    best_name, best_md = _best_compare_variant(results, strategy=rank_strategy)
    if scored_batches and best_name in scored_batches:
        _save_cal_items(scored_batches[best_name])
        return
    if compare_mode == CAL_COMPARE_JUDGES:
        preset_id = best_md.get("preset_id")
        loaded = get_judge_preset(str(preset_id)) if preset_id else None
        if loaded:
            sync_cfg = judge_config_for_calibration(schema, loaded.get("config") or {})
            _save_cal_items(run_judge_batch(items, sync_cfg, schema, only_annotated=True))
    else:
        prompt_id = best_md.get("prompt_id")
        model = best_md.get("model")
        loaded = get_judge_preset(str(prompt_id)) if prompt_id else None
        if loaded and model:
            sync_cfg = _cfg_with_prompt_overlay(
                schema, loaded.get("config") or {}, str(model)
            )
            _save_cal_items(run_judge_batch(items, sync_cfg, schema, only_annotated=True))


def _render_inter_annotator_dispute(d: dict, annotators: list[str]) -> None:
    st.markdown(f"**Цель:** {d.get('goals') or '—'}")
    if d.get("mode") == "binary_multi":
        disputed = d.get("disputed_criteria") or []
        st.markdown(f"**Спорные критерии:** {', '.join(f'`{c}`' for c in disputed)}")
        by_crit = d.get("labels_by_criterion") or {}
        rows = []
        for crit in disputed:
            vals = by_crit.get(crit) or {}
            row: dict = {"Критерий": crit}
            for ann in annotators:
                if ann in vals:
                    row[ann] = vals[ann]
            rows.append(row)
        if rows:
            st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        labels = d.get("labels") or {}
        parts = [f"**{ann}:** `{labels[ann]}`" for ann in annotators if ann in labels]
        st.markdown(" · ".join(parts))
    history = d.get("history") or []
    if history:
        st.markdown("**Диалог**")
        for turn in history:
            st.text(format_turn_for_display(turn))
    else:
        st.caption("История диалога пуста")


def _render_inter_annotator_agreement(
    items: list[CalibrationItem],
    schema: LabelSchema,
    selected_annotators: list[str],
) -> None:
    st.markdown("**Согласие между разметчиками**")
    if len(selected_annotators) < 2:
        st.info(
            "Выберите **два и более** разметчиков, чтобы оценить согласие между людьми "
            "до сравнения с LLM-судьёй."
        )
        return

    metrics = compute_inter_annotator_agreement(
        items,
        schema,
        selected_annotators,
        max_disagreements=50,
    )
    if metrics.n_items_compared == 0:
        st.warning(
            "Нет кейсов, где хотя бы два выбранных разметчика поставили метку. "
            "Разметьте общие кейсы или выберите других разметчиков."
        )
        return

    _metrics_mode_caption(schema, context="inter")
    n_consensus = count_annotator_consensus(items, schema, selected_annotators)

    st.caption(
        f"Разметчиков: **{metrics.n_annotators}** · "
        f"кейсов с ≥2 метками: **{metrics.n_items_compared}** · "
        f"полное пересечение (все разметили): **{metrics.n_items_full_overlap}** · "
        f"согласованных (все совпали): **{n_consensus}** · "
        f"спорных: **{len(metrics.disagreements)}**"
    )

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric(
            "Krippendorff α",
            _fmt_num(metrics.krippendorff_alpha),
            help=_METRIC_HELP["inter_krippendorff"],
        )
    with m2:
        st.metric(
            "Cohen κ (сред.)",
            _fmt_num(metrics.mean_pairwise_cohen_kappa),
            help=_METRIC_HELP["inter_pairwise_kappa"],
        )
    with m3:
        st.metric(
            "Совпадение (сред.)",
            _fmt_pct(metrics.mean_pairwise_exact_match),
            help=_METRIC_HELP["inter_pairwise_agreement"],
        )
    with m4:
        st.metric(
            "PABAK (сред.)",
            _fmt_num(metrics.mean_pairwise_pabak),
            help=_METRIC_HELP["inter_pairwise_pabak"],
        )

    if schema.mode == "ordinal":
        o1, o2, o3 = st.columns(3)
        with o1:
            st.metric(
                "Pearson r (сред.)",
                _fmt_num(metrics.pearson_r),
                help=_METRIC_HELP["pearson"],
            )
        with o2:
            st.metric(
                "Spearman ρ (сред.)",
                _fmt_num(metrics.spearman_r),
                help=_METRIC_HELP["spearman"],
            )
        with o3:
            st.metric(
                "±1 (сред.)",
                _fmt_pct(metrics.plus_minus_one),
                help=_METRIC_HELP["plus_minus_one"],
            )

        with st.expander("Что означают Pearson / Spearman / ±1 (ordinal)", expanded=False):
            st.markdown(
                """
**Ordinal-режим** — метки упорядочены (например, 1–5). Метрики ниже считаются **попарно**
между разметчиками и усредняются по всем парам.

| Метрика | Смысл |
|---------|--------|
| **Pearson r** | Линейная корреляция оценок. Близко к 1 — разметчики двигаются «в одну сторону»: где один ставит выше, другой тоже. Не видит нелинейные сдвиги (например, один всегда на +1 выше). |
| **Spearman ρ** | Ранговая корреляция. Смотрит на порядок, а не на абсолютные числа: если один ставит 2–3–4, а другой 4–5–5 на тех же кейсах — ρ может быть высокой. |
| **±1** | Доля пар, где оценки отличаются **не более чем на 1** балл. Мягче Совпадения: 3 vs 4 считается согласием, 2 vs 5 — нет. |

**Совпадение** и **Cohen κ** при ordinal тоже доступны: Совпадение требует полного совпадения цифр,
κ учитывает случайное совпадение с учётом распределения классов.
"""
            )

    quality = kappa_quality_label(metrics.mean_pairwise_cohen_kappa)
    if quality and quality != "—":
        st.caption(f"Интерпретация среднего κ: **{quality}**")

    if metrics.pairwise:
        st.markdown("**Попарное согласие**")
        pair_rows = []
        for pair_name, pm in metrics.pairwise.items():
            row = {
                "Пара": pair_name,
                "N": pm.get("n_pairs"),
                "Совпадение": _fmt_pct(pm.get("exact_match")),
                "PABAK": _fmt_num(pm.get("pabak")),
                "Cohen κ": _fmt_num(pm.get("cohen_kappa")),
            }
            if schema.mode == "ordinal":
                row["Pearson"] = _fmt_num(pm.get("pearson_r"))
                row["Spearman"] = _fmt_num(pm.get("spearman_r"))
                row["±1"] = _fmt_pct(pm.get("plus_minus_one"))
            pair_rows.append(row)
        st.dataframe(pair_rows, use_container_width=True, hide_index=True)

    if schema.mode == "binary_multi" and metrics.per_criterion:
        st.markdown("**По критериям**")
        crit_rows = []
        for crit, md in metrics.per_criterion.items():
            crit_rows.append(
                {
                    "Критерий": crit,
                    "N": md.get("n_items_compared"),
                    "Krippendorff α": _fmt_num(md.get("krippendorff_alpha")),
                    "Cohen κ": _fmt_num(md.get("mean_pairwise_cohen_kappa")),
                    "Совпадение": _fmt_pct(md.get("mean_pairwise_exact_match")),
                }
            )
        st.dataframe(crit_rows, use_container_width=True, hide_index=True)

    fc1, fc2 = st.columns(2)
    with fc1:
        if st.button(
            "Оставить только согласованные кейсы",
            key="judge_cal_apply_consensus_filter",
            help=(
                "Сузить рабочую выборку: останутся кейсы, где **все** выбранные разметчики "
                "разметили и их метки **совпали**. Спорные и частично размеченные исключаются."
            ),
        ):
            kept = _apply_consensus_sample_filter(schema, selected_annotators)
            if kept:
                st.toast(f"Выборка сужена до {kept} согласованных кейсов")
            else:
                st.toast("Нет согласованных кейсов — выборка не изменена")
            st.rerun()
    with fc2:
        if st.session_state.get("judge_cal_consensus_filter_applied"):
            if st.button(
                "Вернуть полную выборку",
                key="judge_cal_restore_consensus_filter",
            ):
                _restore_sample_before_consensus()
                st.toast("Восстановлена выборка до фильтра")
                st.rerun()

    if st.session_state.get("judge_cal_consensus_filter_applied"):
        st.info(
            f"Активен фильтр **только согласованные**: **{len(_get_cal_items())}** кейсов "
            f"из {n_consensus} доступных с полным совпадением разметчиков."
        )

    if metrics.disagreements:
        st.markdown(f"**Спорные кейсы ({len(metrics.disagreements)})**")
        st.caption(
            "Кейсы, где выбранные разметчики поставили **разные** метки "
            "(хотя бы по одному критерию в режиме binary_multi)."
        )
        for d in metrics.disagreements:
            if d.get("mode") == "binary_multi":
                crits = ", ".join(d.get("disputed_criteria") or [])
                title = f"#{d.get('idx')} · спор: {crits}"
            else:
                labels = d.get("labels") or {}
                vals = " · ".join(f"{a}=`{labels[a]}`" for a in selected_annotators if a in labels)
                title = f"#{d.get('idx')} · {vals}"
            with st.expander(title, expanded=False):
                _render_inter_annotator_dispute(d, selected_annotators)
    else:
        st.success("Спорных кейсов нет — все разметчики полностью согласны на пересекающихся кейсах.")

    st.divider()


def _render_cal_compare_step(items: list[CalibrationItem]) -> None:
    schema = _current_label_schema()
    pool = _get_pool()
    split_active = pool_has_train_test_split(pool)
    train_items = (
        filter_items_by_split(items, JUDGE_SPLIT_TRAIN) if split_active else items
    )
    test_items = (
        filter_items_by_split(items, JUDGE_SPLIT_TEST) if split_active else []
    )
    compare_items = train_items if split_active else items

    st.subheader("2. Сравнение (train)")
    if split_active:
        st.caption(
            f"Сравнение промптов только на **train** ({len(compare_items)} кейсов). "
            f"Test ({len(test_items)} кейсов) — в блоке ниже, с bootstrap."
        )
    saved_judge_msg = st.session_state.pop("_judge_cal_saved_judge_msg", None)
    if saved_judge_msg:
        st.success(saved_judge_msg)

    selected_annotators = _render_cal_annotator_selector(compare_items)
    if not selected_annotators:
        st.stop()

    _render_inter_annotator_agreement(compare_items, schema, selected_annotators)

    if len(selected_annotators) == 1:
        _render_label_statistics(
            compare_items,
            annotator=selected_annotators[0],
            title="Распределение по критериям (train)",
        )
    else:
        st.markdown("**Распределение по критериям (train)**")
        for ann in selected_annotators:
            with st.expander(ann, expanded=False):
                _render_label_statistics(compare_items, annotator=ann, title="")
    st.divider()

    rank_options = [
        COMPARE_RANK_WORST_KAPPA,
        COMPARE_RANK_MEAN_KAPPA,
        COMPARE_RANK_WORST_PABAK,
        COMPARE_RANK_MEAN_PABAK,
        COMPARE_RANK_WORST_F1,
        COMPARE_RANK_MEAN_F1,
    ]
    st.selectbox(
        "Критерий выбора лучшего варианта",
        options=rank_options,
        format_func=compare_rank_strategy_label,
        key="judge_cal_rank_strategy",
        help=(
            "По умолчанию — **худший Cohen κ** по критериям: судья не может "
            "«прятать» провал одного критерия за хорошим средним."
        ),
    )

    compare_mode = st.radio(
        "Режим",
        [CAL_COMPARE_PROMPTS, CAL_COMPARE_JUDGES],
        format_func=lambda m: (
            "Промпты + модели"
            if m == CAL_COMPARE_PROMPTS
            else "Готовые судьи"
        ),
        horizontal=True,
        key="judge_cal_compare_mode",
    )

    if not PRESETS_AVAILABLE:
        st.info(f"Сохранённые пресеты недоступны: {', '.join(_QUEUE_MISSING)}")
    elif compare_mode == CAL_COMPARE_PROMPTS:
        st.caption(
            "Выберите сохранённые **промпты** и **модели** — каждая пара прогоняется "
            "на размеченной выборке. API key и параметры — с вкладки «Настройки судьи»."
        )
        all_presets = list_judge_presets()
        prompts = _prompt_presets(all_presets)
        if not prompts:
            st.info(
                "Нет сохранённых промптов — сохраните на вкладке «Настройки судьи» "
                "(тип «Промпт»)."
            )
        else:
            prompt_by_id = {str(p["id"]): p for p in prompts}
            prompt_ids = list(prompt_by_id.keys())
            sel_prompts = st.multiselect(
                "Промпты",
                options=prompt_ids,
                format_func=lambda pid: prompt_by_id[pid].get("name") or pid,
                key="judge_cal_compare_prompts",
            )
            default_models = [
                m
                for m in [
                    st.session_state.get("judge_evaluator_model", SUPPORTED_MODELS[0])
                ]
                if m in SUPPORTED_MODELS
            ] or [SUPPORTED_MODELS[0]]
            sel_models = st.multiselect(
                "Модели для прогона",
                options=SUPPORTED_MODELS,
                default=default_models,
                key="judge_cal_compare_models",
            )
            if st.button("Сравнить", type="primary", key="judge_cal_compare_run"):
                if not selected_annotators:
                    st.error("Выберите хотя бы одного разметчика.")
                elif not any(
                    _annotated_count(compare_items, a) > 0 for a in selected_annotators
                ):
                    st.error("У выбранных разметчиков нет разметки в train-выборке.")
                elif not sel_prompts:
                    st.error("Выберите хотя бы один промпт.")
                elif not sel_models:
                    st.error("Выберите хотя бы одну модель.")
                else:
                    jobs = _build_prompt_model_jobs(
                        schema, prompt_by_id, sel_prompts, sel_models
                    )
                    results_table, _scored = _execute_calibration_compare(
                        compare_items,
                        schema,
                        jobs,
                        compare_mode=CAL_COMPARE_PROMPTS,
                        selected_annotators=selected_annotators,
                    )
                    st.session_state["judge_cal_compare_results"] = results_table
                    st.session_state["judge_cal_last_compare_mode"] = CAL_COMPARE_PROMPTS
                    st.session_state.pop("judge_cal_test_compare_results", None)
                    best = _best_compare_variant(
                        results_table,
                        strategy=st.session_state.get(
                            "judge_cal_rank_strategy", COMPARE_RANK_WORST_KAPPA
                        ),
                    )
                    _store_best_compare_config(schema, best[0], best[1])
                    st.rerun()
    else:
        st.caption(
            "Выберите сохранённых **судей** (модель + промпт + поля ответа) "
            "и сравните метрики согласия."
        )
        judges = _full_judge_presets(list_judge_presets())
        if not judges:
            st.info("Нет сохранённых судей — создайте на вкладке «Настройки судьи».")
        else:
            judge_by_id = {str(p["id"]): p for p in judges}
            judge_ids = list(judge_by_id.keys())
            sel_judges = st.multiselect(
                "Судьи",
                options=judge_ids,
                format_func=lambda pid: _preset_option_label(judge_by_id[pid]),
                key="judge_cal_compare_judges",
            )
            if st.button("Сравнить", type="primary", key="judge_cal_compare_judges_run"):
                if not selected_annotators:
                    st.error("Выберите хотя бы одного разметчика.")
                elif not any(
                    _annotated_count(compare_items, a) > 0 for a in selected_annotators
                ):
                    st.error("У выбранных разметчиков нет разметки в train-выборке.")
                elif not sel_judges:
                    st.error("Выберите хотя бы одного судью.")
                else:
                    jobs = _build_judge_jobs(schema, judge_by_id, sel_judges)
                    results_table, _scored = _execute_calibration_compare(
                        compare_items,
                        schema,
                        jobs,
                        compare_mode=CAL_COMPARE_JUDGES,
                        selected_annotators=selected_annotators,
                    )
                    st.session_state["judge_cal_compare_results"] = results_table
                    st.session_state["judge_cal_last_compare_mode"] = CAL_COMPARE_JUDGES
                    st.session_state.pop("judge_cal_test_compare_results", None)
                    best = _best_compare_variant(
                        results_table,
                        strategy=st.session_state.get(
                            "judge_cal_rank_strategy", COMPARE_RANK_WORST_KAPPA
                        ),
                    )
                    _store_best_compare_config(schema, best[0], best[1])
                    st.rerun()

    compare_results = st.session_state.get("judge_cal_compare_results") or {}
    last_mode = st.session_state.get("judge_cal_last_compare_mode", compare_mode)
    name_col = "Вариант" if last_mode == CAL_COMPARE_PROMPTS else "Судья"
    best = _render_comparison_results_table(
        compare_results,
        name_col=name_col,
        schema=schema,
        per_entity_label="вариант",
    )
    if best:
        rank_label = compare_rank_strategy_label(
            st.session_state.get("judge_cal_rank_strategy", COMPARE_RANK_WORST_KAPPA)
        )
        st.success(
            f"Лучший вариант ({rank_label}): **{best[0]}** "
            f"(Cohen κ={_fmt_num(best[1].get('cohen_kappa'))}, "
            f"PABAK={_fmt_num(best[1].get('pabak'))}, "
            f"F1={_fmt_pct(best[1].get('f1'))})"
        )
        if st.button(f"Сохранить как судью «{best[0]}»", key="judge_cal_pick_best"):
            if not PRESETS_AVAILABLE:
                st.error(f"Сохранение недоступно: {', '.join(_QUEUE_MISSING)}")
            else:
                try:
                    cfg, save_name, update_id = _judge_cfg_from_compare_variant(
                        schema, best[0], best[1]
                    )
                    save_judge_preset(
                        save_name,
                        cfg,
                        description="Лучший по калибровке",
                        preset_id=update_id,
                    )
                    schedule_apply_judge_saved(cfg, preset_name=save_name)
                    st.session_state["_judge_cal_saved_judge_msg"] = (
                        f"Судья **«{save_name}»** сохранён в системе и применён."
                    )
                    st.rerun()
                except Exception as e:
                    st.error(str(e))

    st.subheader("3. Метрики согласия (train)")
    metrics_annotator = selected_annotators[0] if selected_annotators else ""
    if metrics_annotator:
        metrics_items = items_with_annotator_labels(
            compare_items, metrics_annotator, schema
        )
        if len(selected_annotators) > 1:
            st.caption(
                f"Метрики train для разметчика **{metrics_annotator}** "
                f"(без bootstrap). "
                f"Финальная оценка — на test ниже."
            )
    else:
        metrics_items = compare_items
        st.warning("Выберите разметчика выше — без него эталонные метки могут быть пустыми.")
    if split_active:
        st.caption("Bootstrap на train не считается — только point estimate.")
    train_metrics = compute_agreement_metrics(
        metrics_items, schema, bootstrap=False
    )
    _render_metrics_panel(train_metrics, schema)

    _render_test_evaluation_step(
        test_items,
        schema,
        selected_annotators,
        split_active=split_active,
    )

    st.download_button(
        "Скачать разметку (JSONL)",
        data=export_calibration_jsonl(items),
        file_name="judge_calibration.jsonl",
        mime="application/json",
    )


def _render_test_evaluation_step(
    test_items: list[CalibrationItem],
    schema: LabelSchema,
    selected_annotators: list[str],
    *,
    split_active: bool,
) -> None:
    st.subheader("4. Оценка test")
    if not split_active:
        st.info(
            "Разделите выборку на **train / test** на вкладке «Разметка» — "
            "здесь появится финальная оценка с **bootstrap 95% CI**."
        )
        return
    if not test_items:
        st.warning("В test-части выборки нет кейсов.")
        return

    metrics_annotator = selected_annotators[0] if selected_annotators else ""
    ann_n = (
        _annotated_count(test_items, metrics_annotator)
        if metrics_annotator
        else _annotated_count(test_items)
    )
    st.caption(
        f"Test: **{len(test_items)}** кейсов · размечено: **{ann_n}** · "
        f"эталон: **{metrics_annotator or '—'}**"
    )

    compare_results = st.session_state.get("judge_cal_compare_results") or {}
    test_jobs = _rebuild_compare_jobs_from_results(schema, compare_results)
    if test_jobs:
        st.caption(
            f"Будут прогнаны **{len(test_jobs)}** судей/вариантов из train-сравнения "
            f"(метрики с bootstrap 95% CI, без повторного выбора на test)."
        )
    else:
        cfg, cfg_label = _resolve_test_judge_config(schema, compare_results)
        if cfg:
            st.caption(
                f"Train-сравнение не выполнялось — будет прогнан один судья: **{cfg_label}**"
            )

    if st.button(
        "Прогон и оценка на test (все судьи)",
        type="primary",
        key="judge_cal_run_test_eval",
    ):
        if not metrics_annotator:
            st.error("Выберите разметчика для эталона.")
        elif ann_n == 0:
            st.error("В test нет разметки выбранного разметчика.")
        else:
            jobs = list(test_jobs)
            if not jobs:
                cfg, cfg_label = _resolve_test_judge_config(schema, compare_results)
                if not cfg:
                    st.error(
                        "Не удалось собрать конфиг судьи. "
                        "Сначала выполните сравнение на train."
                    )
                else:
                    jobs = [
                        CalibrationCompareJob(
                            result_key=cfg_label,
                            config=cfg,
                            meta={"compare_kind": "fallback"},
                        )
                    ]
            if jobs:
                last_mode = st.session_state.get(
                    "judge_cal_last_compare_mode", CAL_COMPARE_JUDGES
                )
                test_results, _scored = _execute_calibration_compare(
                    test_items,
                    schema,
                    jobs,
                    compare_mode=last_mode,
                    selected_annotators=[metrics_annotator],
                    bootstrap=True,
                    max_disagreements=50,
                    status_title="Test",
                )
                st.session_state["judge_cal_test_compare_results"] = test_results
                best = _best_compare_variant(
                    test_results,
                    strategy=st.session_state.get(
                        "judge_cal_rank_strategy", COMPARE_RANK_WORST_KAPPA
                    ),
                )
                _store_best_compare_config(schema, best[0], best[1])
                st.session_state["_judge_cal_test_eval_done"] = True
                st.rerun()

    test_results = st.session_state.get("judge_cal_test_compare_results") or {}
    if not test_results:
        st.info(
            "Нажмите «Прогон и оценка на test (все судьи)», "
            "чтобы получить метрики с bootstrap."
        )
        return

    st.markdown("**Финальные метрики test (bootstrap 95% CI)**")
    last_mode = st.session_state.get("judge_cal_last_compare_mode", CAL_COMPARE_JUDGES)
    name_col = "Вариант" if last_mode == CAL_COMPARE_PROMPTS else "Судья"
    best = _render_comparison_results_table(
        test_results,
        name_col=name_col,
        schema=schema,
        per_entity_label="судья",
        show_bootstrap=True,
    )
    if best:
        rank_label = compare_rank_strategy_label(
            st.session_state.get("judge_cal_rank_strategy", COMPARE_RANK_WORST_KAPPA)
        )
        st.success(
            f"Лучший на test ({rank_label}): **{best[0]}** "
            f"(Cohen κ={_fmt_num(best[1].get('cohen_kappa'))}, "
            f"PABAK={_fmt_num(best[1].get('pabak'))}, "
            f"F1={_fmt_pct(best[1].get('f1'))})"
        )


_SCHEMA_STATE_FIELDS = (
    "label_mode",
    "binary_criteria",
    "ordinal_min",
    "ordinal_max",
    "categories",
    "llm_field",
)

_UI_KEY_LEGACY: dict[str, str] = {
    "load_src": "judge_cal_load_src",
    "upload": "judge_cal_upload",
    "upload_btn": "judge_cal_upload_btn",
    "run_select": "judge_cal_run_select",
    "load_run": "judge_cal_load_run",
    "num_to_annotate_input": "judge_cal_num_to_annotate_input",
    "sample_seed_input": "judge_cal_sample_seed_input",
    "apply_random": "judge_cal_apply_random",
    "apply_all": "judge_cal_apply_all",
    "label_mode": "judge_cal_label_mode",
    "binary_criteria": "judge_cal_binary_criteria",
    "ordinal_min": "judge_cal_ordinal_min",
    "ordinal_max": "judge_cal_ordinal_max",
    "categories": "judge_cal_categories",
    "llm_field": "judge_cal_llm_field",
    "cal_prev": "judge_ds_cal_prev",
    "cal_next": "judge_ds_cal_next",
    "annotation_form": "judge_ds_annotation_form",
    "save_sample_form": "judge_save_sample_form",
    "save_sample_name_input": "judge_save_sample_name_input",
    "save_sample_desc": "judge_save_sample_desc",
    "save_sample_overwrite": "judge_save_sample_overwrite",
    "download_annotated": "judge_ds_download_annotated",
    "saved_sample_select": "judge_saved_sample_select",
    "load_saved_sample": "judge_load_saved_sample",
    "delete_saved_sample": "judge_delete_saved_sample",
}


def _ui_key(ui_prefix: str, name: str) -> str:
    if ui_prefix == "judge_cal" and name in _UI_KEY_LEGACY:
        return _UI_KEY_LEGACY[name]
    return f"{ui_prefix}_{name}"


def _schema_state_key(field: str) -> str:
    return f"judge_cal_{field}"


def _init_schema_widgets(ui_prefix: str) -> None:
    if ui_prefix == "judge_cal":
        return
    for field in _SCHEMA_STATE_FIELDS:
        widget_key = _ui_key(ui_prefix, field)
        state_key = _schema_state_key(field)
        if state_key in st.session_state and widget_key not in st.session_state:
            st.session_state[widget_key] = st.session_state[state_key]


def _sync_schema_widgets_to_state(ui_prefix: str) -> None:
    if ui_prefix == "judge_cal":
        return
    for field in _SCHEMA_STATE_FIELDS:
        widget_key = _ui_key(ui_prefix, field)
        if widget_key in st.session_state:
            st.session_state[_schema_state_key(field)] = st.session_state[widget_key]


def _render_save_sample_block(
    cal_items: list[CalibrationItem],
    *,
    ui_prefix: str = "judge_cal",
) -> None:
    if not cal_items:
        return

    st.markdown("#### Сохранить выборку в системе")
    if not SAMPLES_STORAGE_AVAILABLE:
        st.info(
            "Сохранение в систему недоступно — нужны: "
            f"**{', '.join(_STORAGE_MISSING)}**."
        )
        return

    active_name = _get_sample_meta_name()
    if active_name:
        st.caption(f"Загружена выборка: **{active_name}** — имя подставится при сохранении.")
    name_key = _ui_key(ui_prefix, "save_sample_name_input")
    if active_name and name_key not in st.session_state:
        st.session_state[name_key] = active_name

    with st.form(_ui_key(ui_prefix, "save_sample_form")):
        save_name = st.text_input(
            "Название выборки",
            key=name_key,
            placeholder="Например: pilot-50-dialogs-v1",
        )
        save_desc = st.text_input(
            "Описание (необязательно)",
            key=_ui_key(ui_prefix, "save_sample_desc"),
        )
        overwrite = st.checkbox(
            "Обновить уже сохранённую (если загружена из системы)",
            value=bool(st.session_state.get("judge_active_sample_id")),
            key=_ui_key(ui_prefix, "save_sample_overwrite"),
        )
        if st.form_submit_button("Сохранить выборку в системе", type="primary"):
            title = (save_name or "").strip()
            if not title:
                st.error("Укажите название выборки")
            else:
                try:
                    schema = _current_label_schema()
                    jsonl = export_annotated_dataset_jsonl(cal_items).encode("utf-8")
                    sid = (
                        st.session_state.get("judge_active_sample_id")
                        if overwrite
                        else None
                    )
                    new_id = save_judge_sample(
                        title,
                        jsonl,
                        description=save_desc,
                        case_count=len(cal_items),
                        annotated_count=_annotated_count(cal_items),
                        label_mode=schema.mode,
                        criteria_json=(
                            schema.binary_criteria
                            if schema.mode == "binary_multi"
                            else []
                        ),
                        sample_id=sid or None,
                    )
                    _schedule_sample_meta(
                        name=title,
                        sample_id=str(new_id),
                        msg=f"Выборка «{title}» сохранена в системе.",
                    )
                    st.rerun()
                except Exception as e:
                    st.error(str(e))


def _render_sample_selection_block(
    pool_len: int,
    *,
    ui_prefix: str = "judge_cal",
) -> None:
    st.markdown("#### Выборка для разметки")
    st.caption(f"Всего в датасете: **{pool_len}** кейсов.")

    default_n = int(st.session_state.get("judge_cal_num_to_annotate") or pool_len)
    default_n = max(1, min(default_n, pool_len))

    sel_cols = st.columns([2, 1, 1, 1])
    with sel_cols[0]:
        num_to_annotate = st.number_input(
            "Сколько кейсов размечать",
            min_value=1,
            max_value=pool_len,
            value=default_n,
            key=_ui_key(ui_prefix, "num_to_annotate_input"),
        )
        st.session_state["judge_cal_num_to_annotate"] = int(num_to_annotate)
    with sel_cols[1]:
        seed = st.number_input(
            "Seed",
            min_value=0,
            max_value=999999,
            value=int(st.session_state.get("judge_cal_sample_seed", 42)),
            key=_ui_key(ui_prefix, "sample_seed_input"),
        )
        st.session_state["judge_cal_sample_seed"] = int(seed)
    with sel_cols[2]:
        if st.button(
            "Случайная выборка",
            type="primary",
            key=_ui_key(ui_prefix, "apply_random"),
        ):
            _apply_sample_selection(int(num_to_annotate), seed=int(seed))
            st.rerun()
    with sel_cols[3]:
        if st.button("Все кейсы", key=_ui_key(ui_prefix, "apply_all")):
            _apply_sample_selection(pool_len, seed=int(seed))
            st.session_state["judge_cal_num_to_annotate"] = pool_len
            st.rerun()

    sample_indices = _get_sample_indices(pool_len)
    st.caption(f"В выборке: **{len(sample_indices)}** из **{pool_len}**")


def _render_train_test_split_block(
    pool: list[CalibrationItem],
    *,
    annotator: str = "",
) -> None:
    st.divider()
    st.markdown("#### Train / Test")
    schema = _current_label_schema()
    split_active = pool_has_train_test_split(pool)
    stats = st.session_state.get("judge_cal_split_stats") or {}
    if split_active:
        c = count_items_by_split(pool)
        st.info(
            f"Split активен: **train {c[JUDGE_SPLIT_TRAIN]}** · "
            f"**test {c[JUDGE_SPLIT_TEST]}**"
            + (f" · без split: {c['unset']}" if c["unset"] else "")
        )
        if stats.get("per_group"):
            rows = []
            for key, g in stats["per_group"].items():
                if not g.get("total"):
                    continue
                label = "Без метки" if key == "__unlabeled__" else key
                rows.append(
                    {
                        "Группа меток": label,
                        "Всего": g.get("total"),
                        "Train": g.get("train"),
                        "Test": g.get("test"),
                    }
                )
            if rows:
                st.dataframe(rows, use_container_width=True, hide_index=True)
        per_crit = stats.get("per_criterion") or {}
        if per_crit:
            st.markdown("**По критериям (train / test)**")
            crit_rows = []
            for crit, splits in per_crit.items():
                crit_rows.append(
                    {
                        "Критерий": crit,
                        "Train 0": splits.get(JUDGE_SPLIT_TRAIN, {}).get(0, 0),
                        "Train 1": splits.get(JUDGE_SPLIT_TRAIN, {}).get(1, 0),
                        "Test 0": splits.get(JUDGE_SPLIT_TEST, {}).get(0, 0),
                        "Test 1": splits.get(JUDGE_SPLIT_TEST, {}).get(1, 0),
                    }
                )
            st.dataframe(crit_rows, use_container_width=True, hide_index=True)

    split_cols = st.columns([1, 1, 1, 1])
    with split_cols[0]:
        test_ratio = st.number_input(
            "Доля test",
            min_value=0.05,
            max_value=0.5,
            value=float(st.session_state.get("judge_cal_split_test_ratio", 0.2)),
            step=0.05,
            key="judge_cal_split_test_ratio_input",
            help="Доля test внутри каждой группы меток текущей схемы.",
        )
        st.session_state["judge_cal_split_test_ratio"] = float(test_ratio)
    with split_cols[1]:
        split_seed = st.number_input(
            "Seed split",
            min_value=0,
            max_value=999999,
            value=int(st.session_state.get("judge_cal_split_seed", 42)),
            key="judge_cal_split_seed_input",
        )
        st.session_state["judge_cal_split_seed"] = int(split_seed)
    with split_cols[2]:
        if st.button(
            "Разделить train / test",
            type="primary",
            key="judge_cal_apply_train_test_split",
        ):
            new_stats = _apply_train_test_split_to_pool(
                test_ratio=float(test_ratio),
                seed=int(split_seed),
                annotator=annotator,
            )
            st.session_state["_judge_split_msg"] = (
                f"Split: train **{new_stats['train']}** · test **{new_stats['test']}** "
                f"(seed={new_stats['seed']})"
            )
            st.rerun()
    with split_cols[3]:
        if st.button(
            "Сбросить split",
            key="judge_cal_clear_train_test_split",
            disabled=not split_active,
        ):
            _clear_train_test_split()
            st.session_state["_judge_split_msg"] = "Split сброшен."
            st.rerun()

    split_msg = st.session_state.pop("_judge_split_msg", None)
    if split_msg:
        st.success(split_msg)


def _render_label_schema_controls(*, ui_prefix: str = "judge_cal") -> LabelSchema:
    _init_schema_widgets(ui_prefix)
    mode_key = _ui_key(ui_prefix, "label_mode")
    schema_cols = st.columns(4)
    with schema_cols[0]:
        st.selectbox(
            "Тип меток",
            options=["binary", "binary_multi", "ordinal", "categorical"],
            format_func=lambda x: {
                "binary": "Бинарная (одна оценка)",
                "binary_multi": "Бинарная (несколько критериев 0/1)",
                "ordinal": "Ординальная (шкала)",
                "categorical": "Категориальная",
            }[x],
            key=mode_key,
        )
    with schema_cols[1]:
        if st.session_state.get(mode_key) == "binary_multi":
            st.text_input(
                "Критерии (через запятую)",
                key=_ui_key(ui_prefix, "binary_criteria"),
                help="Имена полей в JSON ответа LLM и ключи в human_labels.",
            )
        elif st.session_state.get(mode_key) == "ordinal":
            st.number_input(
                "Мин. шкалы",
                min_value=0,
                max_value=10,
                key=_ui_key(ui_prefix, "ordinal_min"),
            )
            st.number_input(
                "Макс. шкалы",
                min_value=1,
                max_value=10,
                key=_ui_key(ui_prefix, "ordinal_max"),
            )
    with schema_cols[2]:
        if st.session_state.get(mode_key) == "categorical":
            st.text_input(
                "Категории (через запятую)",
                key=_ui_key(ui_prefix, "categories"),
            )
    with schema_cols[3]:
        if st.session_state.get(mode_key) != "binary_multi":
            st.text_input(
                "Поле ответа LLM",
                key=_ui_key(ui_prefix, "llm_field"),
                help="result — бинарно, score — шкала.",
            )
        else:
            st.caption("Поля LLM задаются списком критериев + reason.")
    _sync_schema_widgets_to_state(ui_prefix)
    return _current_label_schema()


def _render_annotation_ui(
    items: list[CalibrationItem],
    *,
    ui_prefix: str = "judge_cal",
    annotator: str = "",
) -> None:
    if not items:
        st.warning(
            "Выборка пуста. Укажите число кейсов и нажмите «Случайная выборка» или «Все кейсы»."
        )
        return

    annotator = normalize_annotator_name(annotator)
    if not annotator:
        st.warning("Укажите **имя разметчика** выше, чтобы сохранять метки.")
        return

    _sync_schema_widgets_to_state(ui_prefix)
    schema = _current_label_schema()
    _ensure_annotation_widgets_for_annotator(
        items,
        annotator,
        schema,
        ui_prefix=ui_prefix,
        known_annotators=list_annotators_from_items(items),
    )
    per_page = min(CASES_PER_PAGE, len(items))
    page_key = "judge_cal_page" if ui_prefix == "judge_cal" else f"{ui_prefix}_cal_page"
    st.caption(
        f"Разметчик: **{annotator}** · к разметке: **{len(items)}** кейсов · "
        f"размечено: **{_annotated_count(items, annotator)}** · "
        f"по **{per_page}** на страницу"
    )

    total_pages = max(1, (len(items) + per_page - 1) // per_page)
    page = min(int(st.session_state.get(page_key, 0)), total_pages - 1)
    st.session_state[page_key] = page

    nav1, nav2, nav3 = st.columns([1, 2, 1])
    with nav1:
        if st.button("← Назад", disabled=page <= 0, key=_ui_key(ui_prefix, "cal_prev")):
            st.session_state[page_key] = page - 1
            st.rerun()
    with nav2:
        st.markdown(f"**Страница {page + 1} / {total_pages}**")
    with nav3:
        if st.button(
            "Вперёд →",
            disabled=page >= total_pages - 1,
            key=_ui_key(ui_prefix, "cal_next"),
        ):
            st.session_state[page_key] = page + 1
            st.rerun()

    start = page * per_page
    page_items = items[start : start + per_page]
    for item in page_items:
        apply_annotator_to_item(item, annotator, schema)

    human_key = f"{ui_prefix}_human"
    note_key = f"{ui_prefix}_note"

    with st.form(_ui_key(ui_prefix, "annotation_form")):
        for item in page_items:
            st.markdown(f"#### Кейс #{item.idx}")
            st.write(f"**Цель:** {item.case.get('goals') or '—'}")
            history = item.case.get("history") or []
            if history:
                with st.expander("История диалога", expanded=True):
                    for turn in history:
                        st.text(format_turn_for_display(turn))
            else:
                st.caption("История пуста")

            if schema.mode == "binary_multi":
                cur_map = normalize_multi_binary_label(
                    item.human_label, schema.binary_criteria
                ) or {}
                for crit in schema.binary_criteria:
                    cur_val = cur_map.get(crit, 0)
                    st.radio(
                        f"{crit}",
                        options=[1, 0],
                        index=0 if cur_val == 1 else 1,
                        format_func=lambda x, c=crit: f"{c}: {x}",
                        key=f"{human_key}_{item.idx}_{crit}",
                        horizontal=True,
                    )
            elif schema.mode == "binary":
                options = schema.binary_options()
                default_idx = 0
                for i, (val, _) in enumerate(options):
                    if item.human_label == val:
                        default_idx = i
                        break
                st.radio(
                    "Ваша оценка",
                    options=[lbl for _, lbl in options],
                    index=default_idx,
                    key=f"{human_key}_{item.idx}",
                    horizontal=True,
                )
            elif schema.mode == "ordinal":
                opts = schema.ordinal_options()
                cur = item.human_label
                idx_o = opts.index(cur) if cur in opts else 0
                st.select_slider(
                    "Ваша оценка (шкала)",
                    options=opts,
                    value=opts[idx_o],
                    key=f"{human_key}_{item.idx}",
                )
            else:
                opts = schema.categorical_options()
                cur = item.human_label if item.human_label in opts else opts[0]
                st.radio(
                    "Ваша оценка",
                    options=opts,
                    index=opts.index(cur),
                    key=f"{human_key}_{item.idx}",
                    horizontal=True,
                )

            st.text_input(
                "Комментарий",
                value=item.human_note or "",
                key=f"{note_key}_{item.idx}",
            )
            st.divider()

        if st.form_submit_button("Сохранить разметку", type="primary"):
            idx_map = {it.idx: it for it in items}
            for it in page_items:
                if schema.mode == "binary_multi":
                    labels = {}
                    for crit in schema.binary_criteria:
                        val = st.session_state.get(f"{human_key}_{it.idx}_{crit}")
                        labels[crit] = int(val) if val is not None else 0
                    it.human_label = labels
                elif schema.mode == "binary":
                    options = schema.binary_options()
                    label_map = {lbl: val for val, lbl in options}
                    choice_lbl = st.session_state.get(f"{human_key}_{it.idx}")
                    it.human_label = label_map.get(choice_lbl, it.human_label)
                else:
                    it.human_label = st.session_state.get(
                        f"{human_key}_{it.idx}", it.human_label
                    )
                it.human_note = st.session_state.get(f"{note_key}_{it.idx}", "") or ""
                set_case_annotator_label(
                    it.case,
                    annotator,
                    it.human_label,
                    it.human_note,
                )
                idx_map[it.idx] = it
            _save_cal_items(list(idx_map.values()))
            st.success(f"Разметка сохранена для **{annotator}**")


def _render_label_statistics(
    items: list[CalibrationItem],
    *,
    annotator: str = "",
    title: str = "Статистика разметки",
) -> None:
    if not items:
        return

    schema = _current_label_schema()
    ann_name = normalize_annotator_name(annotator)
    if ann_name:
        stats_items = []
        for it in items:
            cloned = CalibrationItem.from_dict(it.to_dict())
            apply_annotator_to_item(cloned, ann_name, schema)
            stats_items.append(cloned)
    else:
        stats_items = items
    stats = compute_human_label_statistics(stats_items, schema)
    annotated = stats.get("annotated", 0)
    if annotated == 0:
        st.info("Статистика появится после сохранения хотя бы одной разметки.")
        return

    if title:
        st.subheader(title)
    unannotated = stats.get("unannotated", 0)
    st.caption(
        f"Размечено **{annotated}** из **{stats.get('total', len(items))}** кейсов"
        + (f" · без метки: **{unannotated}**" if unannotated else "")
    )

    if schema.mode == "binary_multi":
        rows = []
        for criterion in schema.binary_criteria:
            cnt = (stats.get("per_criterion") or {}).get(criterion) or {}
            n0 = int(cnt.get(0, 0))
            n1 = int(cnt.get(1, 0))
            n = n0 + n1
            rows.append(
                {
                    "Критерий": criterion,
                    "1 (да)": n1,
                    "0 (нет)": n0,
                    "Доля «1»": _fmt_pct(n1 / n) if n else "—",
                    "N": n,
                }
            )
        st.dataframe(rows, use_container_width=True, hide_index=True)
        return

    counts = stats.get("counts") or {}
    if schema.mode == "binary":
        label_names = {val: lbl for val, lbl in schema.binary_options()}
        ordered_keys = [True, False]
    elif schema.mode == "ordinal":
        label_names = {v: str(v) for v in schema.ordinal_options()}
        ordered_keys = schema.ordinal_options()
    else:
        label_names = {c: c for c in schema.categorical_options()}
        ordered_keys = schema.categorical_options()

    rows = []
    for key in ordered_keys:
        n = int(counts.get(key, 0))
        rows.append(
            {
                "Метка": label_names.get(key, str(key)),
                "Количество": n,
                "Доля": _fmt_pct(n / annotated) if annotated else "—",
            }
        )
    extra_keys = [k for k in counts if k not in ordered_keys]
    for key in sorted(extra_keys, key=lambda x: str(x)):
        n = int(counts[key])
        rows.append(
            {
                "Метка": label_names.get(key, str(key)),
                "Количество": n,
                "Доля": _fmt_pct(n / annotated) if annotated else "—",
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)


def _render_data_load_block(
    *,
    ui_prefix: str,
    msg_key: str,
    show_delete_saved: bool = True,
) -> None:
    load_src = st.radio(
        "Источник",
        ["Файл JSONL / JSON", "Сохранённые выборки"],
        horizontal=True,
        key=_ui_key(ui_prefix, "load_src"),
    )

    if load_src == "Файл JSONL / JSON":
        uploaded = st.file_uploader(
            "Датасет (.jsonl / .json)",
            type=["jsonl", "json", "txt"],
            key=_ui_key(ui_prefix, "upload"),
        )
        if uploaded is not None:
            st.caption(f"Файл: **{uploaded.name}** ({uploaded.size} байт)")
            if st.button(
                "Загрузить файл",
                type="primary",
                key=_ui_key(ui_prefix, "upload_btn"),
            ):
                text = uploaded.getvalue().decode("utf-8", errors="replace")
                if uploaded.name.endswith(".json"):
                    try:
                        text = json.dumps(json.loads(text), ensure_ascii=False)
                    except json.JSONDecodeError as e:
                        st.error(f"Невалидный JSON: {e}")
                        text = ""
                if text:
                    parsed, err = load_cases_from_text(text)
                    if err:
                        st.error(err)
                    elif parsed:
                        items = cases_to_calibration_items(parsed)
                        _on_dataset_loaded(items)
                        st.session_state[msg_key] = (
                            f"Загружено **{len(items)}** кейсов."
                        )
                        st.session_state.pop("judge_bertopic_result", None)
                        st.rerun()
    else:
        _render_load_saved_samples_source(
            select_key=_ui_key(ui_prefix, "saved_sample_select"),
            load_key=_ui_key(ui_prefix, "load_saved_sample"),
            delete_key=_ui_key(ui_prefix, "delete_saved_sample"),
            msg_key=msg_key,
            show_delete=show_delete_saved,
        )


def _render_analysis_data_load_block() -> None:
    _render_data_load_block(
        ui_prefix="judge_cal",
        msg_key="_judge_dataset_loaded_msg",
    )


def _render_sample_selection_only() -> None:
    pool = _get_pool()
    if not pool:
        st.info("Загрузите датасет, чтобы выбрать выборку и перейти к разметке.")
        return

    with_gold = sum(1 for it in pool if get_human_labels_dict(it.case))
    known_annotators = list_annotators_from_items(pool)
    st.caption(
        f"Всего в датасете: **{len(pool)}** · с разметкой: **{with_gold}**"
        + (
            f" · разметчики: **{', '.join(known_annotators)}**"
            if known_annotators
            else ""
        )
    )

    _render_sample_selection_block(len(pool), ui_prefix="judge_cal")


def _render_annotation_tab() -> None:
    st.subheader("Разметка")

    loaded_msg = st.session_state.pop("_judge_annotation_loaded_msg", None)
    if loaded_msg:
        st.success(loaded_msg)

    _render_data_load_block(
        ui_prefix="judge_annot",
        msg_key="_judge_annotation_loaded_msg",
    )

    pool = _get_pool()
    if not pool:
        st.info("Загрузите датасет или выборку для разметки.")
        return

    cal_items = _get_cal_items()
    if not cal_items:
        st.warning(
            "Выборка пуста. Загрузите файл или выборку, либо сформируйте выборку "
            "на вкладке «Анализ и подготовка данных»."
        )
        return

    st.divider()

    loaded_ann_msg = st.session_state.pop("_judge_annotator_labels_loaded_msg", None)
    if loaded_ann_msg:
        st.info(loaded_ann_msg)

    known_annotators = list_annotators_from_items(pool)
    annotator_name = _render_annotator_name_input(
        ui_prefix="judge_cal",
        known_annotators=known_annotators,
    )
    _render_label_schema_controls(ui_prefix="judge_cal")

    split_view = st.radio(
        "Показывать в разметке",
        options=["all", JUDGE_SPLIT_TRAIN, JUDGE_SPLIT_TEST],
        format_func=lambda x: {
            "all": "Все кейсы выборки",
            JUDGE_SPLIT_TRAIN: "Только train",
            JUDGE_SPLIT_TEST: "Только test",
        }[x],
        horizontal=True,
        key="judge_cal_annot_split_view",
        disabled=not pool_has_train_test_split(pool),
    )
    if split_view != "all":
        cal_items = _get_cal_items(split=split_view)
        if not cal_items:
            st.warning(f"В части «{split_view}» нет кейсов текущей выборки.")
    else:
        cal_items = _get_cal_items()

    _render_annotation_ui(
        cal_items,
        ui_prefix="judge_cal",
        annotator=annotator_name,
    )
    _render_label_statistics(cal_items, annotator=annotator_name)

    _render_save_sample_block(cal_items, ui_prefix="judge_cal")
    ann_n = _annotated_count(cal_items)
    st.download_button(
        "Скачать размеченный датасет (JSONL)",
        data=export_annotated_dataset_jsonl(cal_items),
        file_name="judge_annotated_dataset.jsonl",
        mime="application/json",
        type="primary",
        key=_ui_key("judge_cal", "download_annotated"),
        help="Кейсы в формате бенчмарка с полями human_labels и human_notes.",
    )
    st.caption(
        f"В файле **{len(cal_items)}** кейсов из выборки, "
        f"с разметкой: **{ann_n}**. Файл можно снова загрузить на этой странице."
    )

    _render_train_test_split_block(pool, annotator=annotator_name)


def _render_clustering_block() -> None:
    clust_records = _records_from_pool()
    if not clust_records:
        st.info(
            "Загрузите JSONL с диалогами (поле **history** с репликами `role` / `content`)."
        )
        return

    st.caption(f"В датасете: **{len(clust_records)}** диалогов")

    with st.expander("📊 Длина диалогов", expanded=True):
        clust_records_for_run, _clust_min_turns, _clust_max_turns = (
            render_dialog_length_distribution_panel(
                clust_records,
                key_prefix="judge_bertopic_len",
                title="",
                filter_caption=(
                    "Учитывается число реплик в `history`. "
                    "Фильтр применяется к шагам «удалить дубликаты» и «кластеризация». "
                    "0 слева — без нижней границы, 0 справа — без верхней."
                ),
            )
        )
    if not clust_records_for_run:
        st.warning("После фильтра по длине не осталось диалогов. Ослабьте фильтр.")
        return

    if len(clust_records_for_run) != len(clust_records):
        st.caption(
            f"К кластеризации будет использовано **{len(clust_records_for_run)}** "
            f"из **{len(clust_records)}** диалогов."
        )

    st.markdown("---")
    cfg = render_clustering_pipeline_ui(
        "judge_bertopic", bertopic=True, supported_models=SUPPORTED_MODELS
    )

    if st.button("Запустить кластеризацию", type="primary", key="judge_bertopic_run"):
        progress = st.progress(0.0, text="Старт…")
        status = st.empty()

        def on_progress(msg: str, frac: float) -> None:
            progress.progress(min(1.0, max(0.0, frac)), text=msg)
            status.caption(msg)

        try:
            result = run_dialog_bertopic_clustering(
                clust_records_for_run,
                cluster_field=cfg["cluster_field"],
                output_field=cfg["output_field"],
                umap_settings=cfg["umap_settings"],
                bertopic_settings=cfg["bertopic_settings"],
                with_llm=cfg["with_llm"],
                llm_model=cfg["llm_model"] if cfg["with_llm"] else "",
                llm_api_key=st.session_state.get("judge_evaluator_api_key", "")
                or os.getenv("LITELLM_API_KEY", ""),
                llm_api_base=os.getenv("LITELLM_API_BASE", ""),
                llm_sample_dialogs=int(cfg["llm_sample"]),
                llm_prompt_template=cfg["llm_prompt_template"],
                dedup_enabled=cfg["dedup_enabled"],
                dedup_field=cfg["dedup_field"],
                dedup_turn_index=cfg["dedup_turn_index"],
                cluster_turn_index=cfg["cluster_turn_index"],
                dedup_similarity_threshold=float(cfg["dedup_threshold"]),
                tfidf_top_n=cfg["tfidf_top_n"],
                on_progress=on_progress,
            )
            st.session_state["judge_bertopic_result"] = {
                "records": result.records,
                "stats_report": result.stats_report,
                "n_clusters": result.n_clusters,
                "cluster_field": result.cluster_field,
                "cluster_turn_index": result.cluster_turn_index,
                "output_field": result.output_field,
                "theme_counts": dict(result.theme_counts),
                "viz_points": result.viz_points,
                "n_input_before_dedup": result.n_input_before_dedup,
                "dedup_exact_removed": result.dedup_exact_removed,
                "dedup_semantic_removed": result.dedup_semantic_removed,
                "dedup_removed_details": result.dedup_removed_details,
                "dedup_field": result.dedup_field,
                "dedup_turn_index": result.dedup_turn_index,
                "algorithm": CLUSTERING_ALGORITHM_LABEL,
                "cluster_tfidf_words": serialize_cluster_tfidf_words(
                    result.cluster_tfidf_words
                ),
                "record_embeddings": result.record_embeddings,
                "cluster_quality": result.cluster_quality,
                "tfidf_top_n": cfg["tfidf_top_n"],
            }
            progress.progress(1.0, text="Готово")
            dedup_msg = ""
            if cfg["dedup_enabled"] and (
                result.dedup_exact_removed or result.dedup_semantic_removed
            ):
                dedup_msg = (
                    f" · дубликаты: −{result.dedup_exact_removed} точных, "
                    f"−{result.dedup_semantic_removed} семантических"
                )
            status.success(
                f"Кластеров: **{result.n_clusters}** · "
                f"диалогов после удаления дубликатов: **{len(result.records)}**"
                f"{dedup_msg}"
            )
            st.rerun()
        except Exception as e:
            progress.empty()
            status.empty()
            st.error(f"Ошибка кластеризации: {e}")

    stored = st.session_state.get("judge_bertopic_result")
    if not stored:
        return

    render_clustering_results_panel(
        stored,
        key_prefix="judge_bertopic",
        algorithm_name=CLUSTERING_ALGORITHM_LABEL,
        download_prefix="clustered",
    )


def _render_analysis_tab() -> None:
    st.subheader("Анализ и подготовка данных")

    loaded_msg = st.session_state.pop("_judge_dataset_loaded_msg", None)
    if loaded_msg:
        st.success(loaded_msg)

    _render_analysis_data_load_block()

    formation_mode = st.radio(
        "Метод формирования выборки",
        ["Случайная выборка", "Кластеризация"],
        horizontal=True,
        key="judge_analysis_formation_mode",
    )

    if formation_mode == "Случайная выборка":
        _render_sample_selection_only()
    else:
        _render_clustering_block()


def _fmt_pct(v) -> str:
    if v is None:
        return "—"
    return f"{float(v) * 100:.1f}%"


def _fmt_num(v) -> str:
    if v is None:
        return "—"
    return f"{float(v):.3f}"


def _records_from_pool() -> list[dict]:
    pool = _get_pool()
    return [dict(it.case) for it in pool if it.case]


st.title("🧑‍⚖️ Калибровка LLM-судьи")

st.markdown(
    """
**Порядок работы:** загрузите датасет и сформируйте выборку на вкладке «Анализ и подготовка данных» → разметьте выборку на вкладке «Разметка» → настройте LLM-судью → на вкладке «Калибровка» выберите разметчиков и сравните метрики.
"""
)

tab_analysis, tab_annotation, tab_settings, tab_cal = st.tabs(
    [
        "📊 Анализ и подготовка данных",
        "✏️ Разметка",
        "⚙️ Настройки судьи",
        "📊 Калибровка",
    ]
)

# ===================== TAB: АНАЛИЗ И ПОДГОТОВКА ДАННЫХ =====================
with tab_analysis:
    _render_analysis_tab()

# ===================== TAB: РАЗМЕТКА =====================
with tab_annotation:
    _render_annotation_tab()

# ===================== TAB: НАСТРОЙКИ СУДЬИ =====================
with tab_settings:
    st.subheader("Настройки LLM-судьи")
    _render_judge_settings_block()
    st.divider()
    _render_judge_presets_save_apply()

# ===================== TAB: КАЛИБРОВКА =====================
with tab_cal:
    st.caption(
        "Загрузите размеченную выборку → сравните промпты с моделями "
        "или готовых судей → посмотрите метрики согласия."
    )
    sample_ready = _render_cal_sample_step()
    if sample_ready:
        _render_cal_compare_step(_get_cal_items())
