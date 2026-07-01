import copy
import hashlib
import json
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from ui.dialog_results_ui import (
    format_goals_for_result_row,
    render_benchmark_bootstrap_metrics,
    render_dialogs_paginated,
    render_timing_summary_metrics,
)
from benchmarking.runner import is_llm_judge_eval_mode
from tools.run_compare import RunCompareResult, compare_runs
from ui.error_clustering_ui import render_failure_clustering_section

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_REPO_ROOT / ".env")

from common.security import redact_secrets_for_display
from common.time import format_datetime_utc_plus_5
from common.utils import (
    enrich_benchmark_results_timing_inplace,
)

try:
    from storage.benchmark_backend import (
        cancel_run,
        ensure_schema,
        get_run,
        list_recent_runs,
        minio_try_get_bytes,
        minio_try_get_bytes_with_error,
        minio_endpoint_hint,
        queue_backend_enabled,
        results_key_for_run,
    )

    _QUEUE_OK = queue_backend_enabled()
except ImportError:
    _QUEUE_OK = False


def _clear_results_derived_state() -> None:
    """Сброс bootstrap и «Анализ ошибок» при смене набора results."""
    for key in list(st.session_state.keys()):
        if key.startswith(("results_err_cluster", "results_bootstrap_bootstrap")):
            st.session_state.pop(key, None)


def _results_data_fingerprint(results: list, *, extra: str = "") -> str:
    h = hashlib.sha256()
    h.update((extra or "").encode())
    h.update(str(len(results)).encode())
    for row in results:
        if isinstance(row, dict):
            h.update(str(row.get("dialog_id", "")).encode())
            h.update(str(row.get("accuracy", "")).encode())
    return h.hexdigest()[:20]


def load_jsonl_as_list(file) -> list:
    text = file.getvalue().decode("utf-8").strip()
    if text.startswith("["):
        parsed = json.loads(text)
        if not isinstance(parsed, list):
            raise ValueError("JSON-корень должен быть массивом объектов.")
        data = [row for row in parsed if isinstance(row, dict)]
    else:
        data = []
        for line in text.split("\n"):
            if line.strip():
                row = json.loads(line)
                if isinstance(row, dict):
                    data.append(row)
    enrich_benchmark_results_timing_inplace(data)
    return data


def load_jsonl_from_bytes(raw: bytes) -> list:
    text = raw.decode("utf-8")
    data = []
    for line in text.strip().split("\n"):
        if line.strip():
            data.append(json.loads(line))
    enrich_benchmark_results_timing_inplace(data)
    return data


def _parse_row_config(row_info: dict) -> dict:
    cfg = row_info.get("config") or {}
    if isinstance(cfg, str):
        cfg = json.loads(cfg)
    return cfg if isinstance(cfg, dict) else {}


def _config_json_redacted(cfg: dict) -> str:
    c = copy.deepcopy(cfg) if cfg else {}
    return json.dumps(redact_secrets_for_display(c), ensure_ascii=False, indent=2)


def render_run_settings_from_config(cfg: dict):
    """Снимок настроек из БД (модели, промпты, режим) — как при фоновом запуске с бенчмарка."""
    if not cfg:
        st.info("В записи запуска нет сохранённого поля **config**.")
        return

    em = cfg.get("enqueue_meta") or {}
    if em:
        st.caption(
            f"Запись в БД (UTC): **{em.get('saved_at_utc', '—')}** · "
            f"`LITELLM_MODEL_NAME`: `{em.get('litellm_model_name_env') or '—'}` · "
            f"`LITELLM_API_BASE`: `{em.get('litellm_api_base_env') or '—'}`"
        )
    elif not any(
        cfg.get(k)
        for k in (
            "assistant_prompt",
            "user_prompt",
            "llm_eval_prompt",
            "roles",
        )
    ):
        st.caption(
            "Запуск создан до расширенного сохранения config — проверьте блок JSON внизу."
        )

    scid = (cfg.get("scenario_id") or "").strip()
    if scid:
        st.markdown(f"**ID сценария (запуск):** `{scid}`")

    st.markdown(f"**Режим ассистента:** `{cfg.get('mode', '—')}`")
    if cfg.get("assistant_url"):
        st.markdown(f"**URL ассистента:** `{cfg['assistant_url']}`")
    ecfm = (cfg.get("external_context_field_map_json") or "").strip()
    if cfg.get("mode") == "Внешний URL":
        if ecfm:
            st.caption(
                f"Маппинг context → тело API (JSON): `{ecfm[:200]}{'…' if len(ecfm) > 200 else ''}`"
            )
        else:
            st.caption("Внешний URL: в POST добавляется **весь** `context` кейса (маппинг не задан).")

    st.markdown(
        f"**Макс. ходов:** {cfg.get('max_turns', '—')} · "
        f"**Задержка LLM (сек):** {cfg.get('llm_delay', 0)} · "
        f"**Парсить ответ как JSON:** {cfg.get('parse_json_response')} · "
        f"**Инструменты (tools):** {cfg.get('use_tools')}"
    )
    umk = cfg.get("user_message_key")
    esf = (cfg.get("external_session_id_field") or "id").strip() or "id"
    rfp = cfg.get("response_field_path")
    if cfg.get("mode") == "Внешний URL":
        st.caption(f"Ключ id диалога в теле API: `{esf}`")
        if cfg.get("external_unique_session_id"):
            st.caption(
                "Id в теле API: **всегда новый UUID на прогон** (не из `dialog_id` кейса)."
            )
        ecif = (cfg.get("external_coerce_int_fields_csv") or "").strip()
        if ecif:
            st.caption(f"Приведение к int для полей: `{ecif}`")
    if umk:
        st.caption(f"Ключ реплики пользователя (внешний URL): `{umk}`")
    if rfp and cfg.get("mode") == "Внешний URL":
        st.caption(f"Путь к полю ответа: `{rfp}`")

    st.markdown(f"**Режим оценки:** {cfg.get('eval_mode', '—')}")
    efp = cfg.get("eval_field_path")
    if efp:
        st.caption(f"Поле / путь для оценки: `{efp}`")

    if is_llm_judge_eval_mode(str(cfg.get("eval_mode") or "")):
        st.caption(
            f"Только существующий диалог (без новых реплик): **{cfg.get('evaluate_existing_only')}**"
        )
        lef = (cfg.get("llm_eval_fields") or "").strip()
        if lef:
            st.caption(f"Ожидаемые поля JSON оценки: `{lef}`")

    if cfg.get("exit_when_condition_met"):
        st.caption(
            "**Ранний выход:** диалог останавливается, как только критерий оценки (сравнение / Python / тулзы) даёт успех."
        )

    st.markdown("**Модели по ролям**")
    roles = cfg.get("roles") or {}
    for key, label in (
        ("assistant", "Ассистент"),
        ("user", "Симулятор пользователя"),
        ("evaluator", "Оценщик (LLM)"),
    ):
        r = roles.get(key) or {}
        model = r.get("model") or "—"
        has_key = bool((r.get("api_key") or "").strip())
        st.markdown(
            f"- **{label}:** `{model}` · API key: **{'задан' if has_key else 'нет'}**"
        )
        pj = (r.get("params_json") or "").strip()
        if pj and pj != "{}":
            with st.expander(f"Параметры JSON — {label}", expanded=False):
                st.code(pj, language="json")

    ap = (cfg.get("assistant_prompt") or "").strip()
    if ap:
        with st.expander("Промпт ассистента (LLM)", expanded=False):
            st.text(ap)

    up = (cfg.get("user_prompt") or "").strip()
    if up:
        with st.expander("Промпт симулятора пользователя", expanded=False):
            st.text(up)

    lep = (cfg.get("llm_eval_prompt") or "").strip()
    if lep:
        with st.expander("Промпт LLM-оценщика", expanded=False):
            st.text(lep)

    tools = (cfg.get("assistant_tools") or "").strip()
    if tools and cfg.get("use_tools"):
        with st.expander("Инструменты ассистента (JSON)", expanded=False):
            st.code(tools, language="json")

    cec = (cfg.get("custom_eval_code") or "").strip()
    if cec:
        with st.expander("Код кастомного критерия (Python)", expanded=False):
            st.code(cec, language="python")

    with st.expander("Полный config (JSON, API keys скрыты)", expanded=False):
        st.code(_config_json_redacted(cfg), language="json")


def _run_pick_label(r: dict) -> str:
    """✓ — в MinIO есть results.jsonl; ○ — ещё нет (queued / running / failed и т.д.)."""
    title = (r.get("run_title") or "").strip()
    rid_short = str(r["id"])[:8]
    mark = "✓" if r.get("results_key") else "○"
    base = title if title else "(без названия)"
    when = format_datetime_utc_plus_5(r.get("created_at"))
    sc = (r.get("scenario_id") or "").strip()
    sc_part = f" | сценарий: `{sc}`" if sc else ""
    return f"{mark} 📌 {base} | {r['status']} | {when} | {rid_short}…{sc_part}"


def _fmt_pct(val: float | None) -> str:
    if val is None:
        return "—"
    return f"{val:.2%}"


def _fmt_ci(lo: float | None, hi: float | None) -> str:
    if lo is None or hi is None:
        return "—"
    return f"{lo:.1%} – {hi:.1%}"


def _load_run_results_from_minio(run_id: str) -> tuple[list | None, str | None]:
    """Загружает results.jsonl из MinIO. Возвращает (rows, error_message)."""
    if not _QUEUE_OK:
        return None, "Фоновые запуски недоступны (Postgres / MinIO / RabbitMQ)."
    row_info = get_run(run_id)
    if not row_info:
        return None, f"Запуск `{run_id}` не найден в БД."
    rk = row_info.get("results_key") or results_key_for_run(run_id)
    raw, minio_err = minio_try_get_bytes_with_error(rk)
    if raw is None:
        if minio_err:
            return None, f"MinIO: {minio_err}. {minio_endpoint_hint()}"
        return None, "Файл results.jsonl в MinIO отсутствует или недоступен."
    try:
        return load_jsonl_from_bytes(raw), None
    except Exception as ex:
        return None, str(ex)


def _render_compare_summary(cmp: RunCompareResult, label_a: str, label_b: str) -> None:
    st.markdown(f"**Прогон A (базовый):** {label_a}")
    st.markdown(f"**Прогон B (сравниваемый):** {label_b}")

    if cmp.n_paired == 0:
        st.error(
            "Нет общих `dialog_id` — сравнение невозможно. "
            "Убедитесь, что оба прогона используют один и тот же набор кейсов."
        )
        if cmp.n_only_a or cmp.n_only_b:
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Только в A", cmp.n_only_a)
            with c2:
                st.metric("Только в B", cmp.n_only_b)
        return

    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("Парных кейсов", cmp.n_paired)
    with m2:
        st.metric("Accuracy A", _fmt_pct(cmp.acc_a), help="По всем кейсам прогона A")
    with m3:
        st.metric("Accuracy B", _fmt_pct(cmp.acc_b), help="По всем кейсам прогона B")
    with m4:
        delta = cmp.acc_delta_paired
        st.metric(
            "Δ accuracy (paired)",
            f"{delta:+.2%}" if delta is not None else "—",
            help="Доля успехов B минус A только по общим dialog_id",
        )
    with m5:
        p = cmp.mcnemar_p
        st.metric(
            "McNemar p",
            f"{p:.4f}" if p is not None else "—",
            help="Статистика на discordant pairs: улучшилось vs ухудшилось",
        )

    st.caption(
        f"95% CI accuracy A: {_fmt_ci(*cmp.acc_a_ci)} · "
        f"B: {_fmt_ci(*cmp.acc_b_ci)} · "
        f"Только в A: {cmp.n_only_a} · только в B: {cmp.n_only_b}"
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Оба ✓", cmp.both_pass)
    with c2:
        st.metric("Оба ✗", cmp.both_fail)
    with c3:
        st.metric("Улучшилось (A✗→B✓)", cmp.improved)
    with c4:
        st.metric("Ухудшилось (A✓→B✗)", cmp.regressed)

    if cmp.mcnemar_p is not None:
        if cmp.mcnemar_p < 0.05:
            st.info(
                "Различие между прогонами **статистически значимо** (McNemar p < 0.05) "
                "на парных кейсах с discordant outcome."
            )
        else:
            st.caption(
                "McNemar p ≥ 0.05 — на парных кейсах нет статистически значимого "
                "перекоса между улучшениями и ухудшениями."
            )

    cat_rows = cmp.category_breakdown()
    if cat_rows and any(r["category"] != "(без category)" for r in cat_rows):
        st.markdown("##### По category (context)")
        df_cat = pd.DataFrame(cat_rows)
        df_cat["rate_a"] = df_cat["rate_a"].map(lambda x: f"{x:.1%}" if x is not None else "—")
        df_cat["rate_b"] = df_cat["rate_b"].map(lambda x: f"{x:.1%}" if x is not None else "—")
        df_cat["delta"] = df_cat["delta"].map(lambda x: f"{x:+.1%}" if x is not None else "—")
        st.dataframe(
            df_cat.rename(
                columns={
                    "category": "Категория",
                    "n": "N",
                    "acc_a": "✓ A",
                    "acc_b": "✓ B",
                    "rate_a": "Acc A",
                    "rate_b": "Acc B",
                    "delta": "Δ",
                    "improved": "↑",
                    "regressed": "↓",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )


def _render_compare_case_lists(cmp: RunCompareResult) -> None:
    change_labels = {
        "improved": "Улучшилось (A ✗ → B ✓)",
        "regressed": "Ухудшилось (A ✓ → B ✗)",
        "same_fail": "Оба не прошли",
        "same_pass": "Оба прошли",
    }
    filter_change = st.multiselect(
        "Фильтр по типу изменения",
        options=list(change_labels.keys()),
        default=["improved", "regressed"],
        format_func=lambda k: change_labels[k],
        key="compare_change_filter",
    )
    rows = [p.to_dict() for p in cmp.paired if p.change in filter_change]
    if not rows:
        st.info("Нет кейсов для выбранного фильтра.")
        return

    df = pd.DataFrame(rows)
    df["A"] = df["pass_a"].map(lambda x: "✓" if x else "✗")
    df["B"] = df["pass_b"].map(lambda x: "✓" if x else "✗")
    st.dataframe(
        df[
            ["dialog_id", "change", "A", "B", "category", "goals", "reason_a", "reason_b"]
        ].rename(
            columns={
                "dialog_id": "dialog_id",
                "change": "Изменение",
                "category": "category",
                "goals": "Цели",
                "reason_a": "Reason A",
                "reason_b": "Reason B",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "📥 Скачать таблицу сравнения (JSONL)",
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows),
        "run_compare_paired.jsonl",
        "application/json",
        key="compare_download_jsonl",
    )


def _render_compare_two_runs_page() -> None:
    st.subheader("Сравнение двух прогонов")
    st.caption(
        "Парное сравнение по **dialog_id**: accuracy, McNemar, списки улучшений и регрессий. "
        "**Прогон A** — базовый (старый), **прогон B** — новый."
    )

    cmp_source = st.radio(
        "Источник",
        ["Фоновые запуски", "JSONL файлы"],
        horizontal=True,
        key="compare_data_source",
    )

    results_a: list = []
    results_b: list = []
    label_a = "A"
    label_b = "B"

    if cmp_source == "Фоновые запуски":
        if not _QUEUE_OK:
            st.warning(
                "Фоновые запуски недоступны: задайте `BENCHMARK_POSTGRES_DSN`, "
                "`MINIO_ENDPOINT`, `RABBITMQ_URL` в `.env`."
            )
            st.stop()
        try:
            ensure_schema()
        except Exception as e:
            st.error(f"База данных недоступна: {e}")
            st.stop()

        cmp_search = st.text_input(
            "Фильтр запусков",
            placeholder="название, UUID…",
            key="compare_run_search",
        )
        runs_all = list_recent_runs(200, search=cmp_search or None)
        if not runs_all:
            st.info("Нет запусков для выбора.")
            st.stop()

        id_to_row = {str(r["id"]): r for r in runs_all}
        opts = list(id_to_row.keys())
        c1, c2 = st.columns(2)
        with c1:
            run_a = st.selectbox(
                "Прогон A (базовый)",
                options=opts,
                format_func=lambda rid: _run_pick_label(id_to_row[rid]),
                key="compare_run_a_id",
            )
        with c2:
            run_b = st.selectbox(
                "Прогон B (новый)",
                options=opts,
                index=min(1, len(opts) - 1),
                format_func=lambda rid: _run_pick_label(id_to_row[rid]),
                key="compare_run_b_id",
            )

        if run_a == run_b:
            st.warning("Выберите два разных прогона.")
            st.stop()

        label_a = _run_pick_label(id_to_row[run_a])
        label_b = _run_pick_label(id_to_row[run_b])

        if st.button("Загрузить и сравнить", type="primary", key="compare_load_runs"):
            err_msgs = []
            ra, err_a = _load_run_results_from_minio(run_a)
            rb, err_b = _load_run_results_from_minio(run_b)
            if err_a:
                err_msgs.append(f"**A:** {err_a}")
            if err_b:
                err_msgs.append(f"**B:** {err_b}")
            if err_msgs:
                for m in err_msgs:
                    st.error(m)
            else:
                st.session_state["compare_results_a"] = ra
                st.session_state["compare_results_b"] = rb
                st.session_state["compare_label_a"] = label_a
                st.session_state["compare_label_b"] = label_b
                st.rerun()

        results_a = st.session_state.get("compare_results_a") or []
        results_b = st.session_state.get("compare_results_b") or []
        label_a = st.session_state.get("compare_label_a") or label_a
        label_b = st.session_state.get("compare_label_b") or label_b

    else:
        c1, c2 = st.columns(2)
        with c1:
            file_a = st.file_uploader("JSONL прогона A", type=["jsonl"], key="compare_file_a")
        with c2:
            file_b = st.file_uploader("JSONL прогона B", type=["jsonl"], key="compare_file_b")
        if file_a:
            results_a = load_jsonl_as_list(file_a)
            label_a = file_a.name
        if file_b:
            results_b = load_jsonl_as_list(file_b)
            label_b = file_b.name

    if not results_a or not results_b:
        st.info("Загрузите оба прогона для сравнения.")
        st.stop()

    cmp = compare_runs(results_a, results_b)
    _render_compare_summary(cmp, label_a, label_b)
    if cmp.n_paired > 0:
        st.markdown("##### Парные кейсы")
        _render_compare_case_lists(cmp)


st.set_page_config(page_title="Анализ результатов", layout="wide")
st.title("🔍 Анализ результатов и ошибок")

page_mode = st.radio(
    "Режим страницы",
    ["Один прогон", "Сравнение двух прогонов"],
    horizontal=True,
    key="results_page_mode",
)

if page_mode == "Сравнение двух прогонов":
    _render_compare_two_runs_page()
    st.stop()

st.subheader("1. Загрузите результаты")

data_source = st.radio(
    "Источник данных",
    ["Фоновый запуск (история бенчмарка)", "JSONL файлы"],
    horizontal=True,
    key="results_data_source",
)

if "_results_src_prev" not in st.session_state:
    st.session_state["_results_src_prev"] = data_source
elif st.session_state["_results_src_prev"] != data_source:
    st.session_state["_results_src_prev"] = data_source
    st.session_state.pop("results_analysis_data", None)
    st.session_state.pop("results_analysis_meta", None)
    st.session_state.pop("_results_jsonl_fp", None)
    _clear_results_derived_state()

all_results = []
results_file_bytes = None
failures_file_bytes = None

if data_source == "Фоновый запуск (история бенчмарка)":
    if not _QUEUE_OK:
        st.warning(
            "Фоновые запуски недоступны: задайте в `.env` переменные "
            "`BENCHMARK_POSTGRES_DSN` (или `DATABASE_URL`), `MINIO_ENDPOINT`, `RABBITMQ_URL`."
        )
        st.stop()
    try:
        ensure_schema()
    except Exception as e:
        st.error(f"База данных недоступна: {e}")
        st.stop()

    st.caption(
        "Поиск по **названию**, **описанию** или фрагменту **UUID**. "
        "В списке: **✓** — можно скачать/анализировать **results.jsonl**; **○** — запуск ещё без файла (смотрите сводку ниже)."
    )
    res_search = st.text_input(
        "Фильтр запусков",
        placeholder="например: baseline, промпт, 3f2a1b…",
        key="results_run_search",
    )
    runs_all = list_recent_runs(200, search=res_search or None)
    if not runs_all:
        if res_search.strip():
            st.info("По фильтру нет запусков — сбросьте поиск или измените запрос.")
        else:
            st.info(
                "В базе пока нет записей **benchmark_runs**. Запустите бенчмарк **в фоне** на странице «Бенчмарк»."
            )
        st.stop()

    id_to_row = {str(r["id"]): r for r in runs_all}
    run_id = st.selectbox(
        "Запуск",
        options=list(id_to_row.keys()),
        format_func=lambda rid: _run_pick_label(id_to_row[rid]),
        key="results_history_pick_id",
    )

    prev_id = st.session_state.get("_results_selected_run_id")
    run_just_changed = prev_id != run_id
    if run_just_changed:
        st.session_state["_results_selected_run_id"] = run_id
        st.session_state.pop("results_analysis_data", None)
        st.session_state.pop("results_analysis_meta", None)
        _clear_results_derived_state()

    row_info = get_run(run_id)

    if run_just_changed and row_info:
        _cfg_autoload = _parse_row_config(row_info)
        _rk_autoload = row_info.get("results_key") or results_key_for_run(run_id)
        try:
            _raw_autoload = minio_try_get_bytes(_rk_autoload)
            if _raw_autoload is not None:
                _parsed_autoload = load_jsonl_from_bytes(_raw_autoload)
                st.session_state["results_analysis_data"] = _parsed_autoload
                st.session_state["results_analysis_meta"] = {
                    "run_id": run_id,
                    "run_title": row_info.get("run_title"),
                    "raw_bytes": _raw_autoload,
                    "results_key": _rk_autoload,
                    "eval_mode": _cfg_autoload.get(
                        "eval_mode", "Сравнить цель с полем из ответа"
                    ),
                    "llm_eval_fields": _cfg_autoload.get(
                        "llm_eval_fields", "result,reason"
                    ),
                    "scenario_id": (
                        (_cfg_autoload.get("scenario_id") or "").strip() or None
                    ),
                }
        except Exception:
            pass
    st.subheader("Сводка по запуску")
    if row_info:
        rt = (row_info.get("run_title") or "").strip()
        rd = (row_info.get("run_description") or "").strip()
        _cfg0 = _parse_row_config(row_info)
        scid_run = (_cfg0.get("scenario_id") or "").strip()
        if rt:
            st.markdown(f"**Название:** {rt}")
        if rd:
            st.markdown(f"**Описание:** {rd}")
        if scid_run:
            st.markdown(f"**ID сценария (запуск):** `{scid_run}`")

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Статус", row_info.get("status", "—"))
        with m2:
            st.metric(
                "Прогресс",
                f"{row_info.get('progress_done', 0)}/{row_info.get('progress_total', 0)}",
            )
        with m3:
            aa = row_info.get("avg_accuracy")
            st.metric(
                "Средняя точность",
                f"{aa:.2%}" if aa is not None else "—",
            )
        with m4:
            if row_info.get("status") in ("queued", "running"):
                if st.button(
                    "Отменить этот запуск",
                    key="results_cancel_run",
                    help="Статус станет cancelled; воркер остановится после текущего диалога.",
                ):
                    if cancel_run(run_id):
                        st.success("Отмена запрошена.")
                        st.rerun()
                    else:
                        st.info("Нельзя отменить (запуск уже не в очереди или завершён).")

        _prev_results = st.session_state.get("results_analysis_data") or []
        if _prev_results:
            render_timing_summary_metrics(_prev_results, show_help=True)

        st.caption(
            f"**UUID:** `{row_info['id']}` · **Создан:** {format_datetime_utc_plus_5(row_info.get('created_at'))} "
        )
        if row_info.get("error"):
            st.error(row_info["error"][:4000])

        _cfg = _parse_row_config(row_info)
        with st.expander(
            "⚙️ Настройки запуска из БД (модели, промпты, режим)",
            expanded=False,
        ):
            render_run_settings_from_config(_cfg)

        rk = row_info.get("results_key") or results_key_for_run(run_id)
        if row_info.get("status") in ("queued", "running"):
            st.caption(
                "Пока запуск **queued** / **running**, **results.jsonl** в MinIO дополняется воркером. "
                "Автообновление страницы **отключено** — нажмите **«Загрузить results.jsonl из хранилища»** "
                "или **«Обновить из MinIO»** в блоке онлайн-диалогов, чтобы подтянуть актуальные данные."
            )
        else:
            st.caption(
                "Нажмите «Загрузить», чтобы подтянуть актуальный **results.jsonl** из MinIO."
            )
        if st.button("Загрузить results.jsonl из хранилища", key="results_load_from_minio"):
            try:
                raw = minio_try_get_bytes(rk)
                if raw is None:
                    st.warning(
                        "Объект в MinIO ещё не создан или недоступен "
                        "(задача в очереди, воркер не стартовал, сеть)."
                    )
                else:
                    parsed = load_jsonl_from_bytes(raw)
                    st.session_state["results_analysis_data"] = parsed
                    _cfg_btn = _parse_row_config(row_info)
                    st.session_state["results_analysis_meta"] = {
                        "run_id": run_id,
                        "run_title": row_info.get("run_title"),
                        "raw_bytes": raw,
                        "results_key": rk,
                        "eval_mode": _cfg_btn.get(
                            "eval_mode", "Сравнить цель с полем из ответа"
                        ),
                        "llm_eval_fields": _cfg_btn.get(
                            "llm_eval_fields", "result,reason"
                        ),
                        "scenario_id": (
                            (_cfg_btn.get("scenario_id") or "").strip() or None
                        ),
                    }
                    _clear_results_derived_state()
                    st.success(f"Загружено {len(parsed)} записей.")
                    st.rerun()
            except Exception as ex:
                st.error(str(ex))
    else:
        st.error("Запуск не найден в БД.")

    if (
        _QUEUE_OK
        and row_info
        and row_info.get("status") in ("queued", "running")
    ):

        @st.fragment
        def _results_live_benchmark_dialogs():
            rid = st.session_state.get("_results_selected_run_id")
            if not rid:
                return
            r = get_run(rid)
            if not r or r.get("status") not in ("queued", "running"):
                return
            cfg_r = _parse_row_config(r)
            em = cfg_r.get("eval_mode", "Сравнить цель с полем из ответа")
            lf = cfg_r.get("llm_eval_fields", "result,reason")
            rk = r.get("results_key") or results_key_for_run(rid)
            raw = minio_try_get_bytes(rk)
            if raw is None:
                st.caption("Ожидание первых строк **results.jsonl** в MinIO…")
                return
            lines = [ln for ln in raw.decode("utf-8").split("\n") if ln.strip()]
            if not lines:
                st.caption("Файл результатов в MinIO пока пуст — ждём первый диалог…")
                return
            partial = [json.loads(ln) for ln in lines]
            st.session_state["results_analysis_data"] = partial
            st.session_state["results_analysis_meta"] = {
                "run_id": rid,
                "run_title": r.get("run_title"),
                "raw_bytes": raw,
                "results_key": rk,
                "eval_mode": em,
                "llm_eval_fields": lf,
                "scenario_id": (
                    (_parse_row_config(r).get("scenario_id") or "").strip() or None
                ),
            }
            st.subheader(f"📋 Диалоги запуска (онлайн, {len(partial)} шт.)")
            render_timing_summary_metrics(partial, show_help=True)
            if st.button(
                "Обновить из MinIO",
                key=f"results_minio_pull_{rid}",
                help="Подтянуть актуальный results.jsonl из MinIO (обновление только вручную).",
            ):
                st.rerun()
            st.caption(
                "Данные из MinIO обновляются **только по кнопке** «Загрузить results.jsonl из хранилища» (сверху) "
                "или «Обновить из MinIO» (здесь), либо при перезагрузке страницы."
            )
            render_dialogs_paginated(
                partial,
                em,
                lf,
                session_page_key="results_live_page_idx",
                show_timing_summary=False,
            )

        _results_live_benchmark_dialogs()

    all_results = st.session_state.get("results_analysis_data") or []

else:
    st.caption(
        "Загрузите **results.jsonl** / **evaluation_results.jsonl** (экспорт со страницы «Бенчмарк») или только ошибки."
    )
    col1, col2 = st.columns(2)
    with col1:
        results_file = st.file_uploader(
            "Результаты выполнения (JSONL / JSON)",
            type=["jsonl", "json"],
            help="Полный вывод прогона: evaluation_results.jsonl, JSON-массив или скачанный экспорт.",
        )
    with col2:
        failures_file = st.file_uploader(
            "Только ошибки (failures.jsonl), опционально",
            type=["jsonl", "json"],
        )

    def _ingest_jsonl_upload(upload, *, failures_only: bool = False) -> None:
        raw = upload.getvalue()
        fp = hashlib.sha256(raw).hexdigest()[:20]
        if st.session_state.get("_results_jsonl_fp") == fp:
            return
        st.session_state["_results_jsonl_fp"] = fp
        st.session_state["results_analysis_data"] = load_jsonl_as_list(upload)
        st.session_state["results_analysis_meta"] = {
            "source": "jsonl",
            "file_name": upload.name,
            "failures_only": failures_only,
        }
        _clear_results_derived_state()

    if results_file:
        _ingest_jsonl_upload(results_file, failures_only=False)
        results_file_bytes = results_file.getvalue()
        n_loaded = len(st.session_state.get("results_analysis_data") or [])
        st.success(f"Загружено {n_loaded} результатов.")

    if failures_file and not results_file:
        _ingest_jsonl_upload(failures_file, failures_only=True)
        failures_file_bytes = failures_file.getvalue()
        st.warning("Загружены только ошибки. Полные результаты не доступны.")

    all_results = st.session_state.get("results_analysis_data") or []

if not all_results:
    if data_source == "Фоновый запуск (история бенчмарка)":
        st.info(
            "При выборе запуска при **первом открытии** подгружается снимок из MinIO, если файл уже есть. "
            "Для **running** / **queued** дальнейшее обновление **только вручную** — кнопки **«Загрузить results.jsonl»** "
            "и **«Обновить из MinIO»** в онлайн-блоке. Или переключитесь на **«JSONL файлы»**."
        )
    else:
        st.info(
            "Загрузите JSONL с результатами или переключитесь на **«Фоновый запуск»**, "
            "чтобы выбрать прогон из MinIO."
        )
    st.stop()

st.subheader("2. Фильтры")

view_mode = st.radio(
    "Режим просмотра",
    options=["Все результаты", "Только ошибки (accuracy = 0)"],
    horizontal=True,
)

if view_mode == "Только ошибки (accuracy = 0)":
    filtered = [r for r in all_results if r.get("accuracy", 1) == 0]
else:
    filtered = all_results

search_query = st.text_input("Поиск по цели, тексту диалога, ID диалога или сценария")

def _result_row_search_blob(r: dict) -> str:
    """Текст для полнотекстового поиска: цели, context и прочие скаляры из исходного кейса."""
    parts: list[str] = []
    parts.append(format_goals_for_result_row(r).lower())
    ctx = r.get("context")
    if isinstance(ctx, dict):
        try:
            parts.append(
                json.dumps(redact_secrets_for_display(ctx), ensure_ascii=False).lower()
            )
        except Exception:
            parts.append(str(ctx).lower())
    elif isinstance(ctx, str):
        parts.append(ctx.lower())
    skip = {
        "history",
        "full_response",
        "context",
        "goals",
        "goals_text",
        "predicted_intents",
        "eval_field_value",
        "raw_output",
    }
    for k, v in r.items():
        if k in skip or v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            parts.append(str(v).lower())
    return " ".join(parts)


if search_query:
    q = search_query.lower()
    filtered = [
        r
        for r in filtered
        if q in _result_row_search_blob(r)
        or any(q in msg.get("content", "").lower() for msg in r.get("history", []))
        or q in str(r.get("dialog_id", "")).lower()
        or q in str(r.get("scenario_id", "")).lower()
    ]

_max_turns = max((r.get("turns", 1) for r in filtered), default=1)
if _max_turns <= 1:
    turns_range = (1, 1)
else:
    turns_range = st.slider(
        "Число ходов",
        min_value=1,
        max_value=_max_turns,
        value=(1, _max_turns),
    )
filtered = [
    r for r in filtered if turns_range[0] <= r.get("turns", 0) <= turns_range[1]
]

st.write(f"Отображается: {len(filtered)} из {len(all_results)} записей")

st.subheader("3. Статистика")
total = len(all_results)
correct = sum(1 for r in all_results if r.get("accuracy", 0) == 1)
accuracy_rate = correct / total if total > 0 else 0

_meta_disp = st.session_state.get("results_analysis_meta") or {}
_disp_eval_mode = _meta_disp.get("eval_mode") or "Сравнить цель с полем из ответа"
_disp_llm_eval_fields = _meta_disp.get("llm_eval_fields") or "result,reason"

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Всего", total)
with col2:
    st.metric("Успешно", correct)
with col3:
    st.metric("Точность", f"{accuracy_rate:.2%}")

_results_fp = _results_data_fingerprint(
    all_results,
    extra=(
        st.session_state.get("_results_jsonl_fp")
        or str(_meta_disp.get("run_id") or "")
    ),
)

render_benchmark_bootstrap_metrics(
    all_results,
    llm_eval_fields=_disp_llm_eval_fields,
    eval_mode=_disp_eval_mode,
    key_prefix="results_bootstrap",
    data_fingerprint=_results_fp,
)

_failures_all = [r for r in all_results if r.get("accuracy", 1) == 0]
render_failure_clustering_section(
    _failures_all,
    key_prefix="results_err_cluster",
    data_fingerprint=_results_fp,
)

st.subheader("4. Результаты по диалогам")

_sid_meta = (_meta_disp.get("scenario_id") or "").strip()
if data_source == "Фоновый запуск (история бенчмарка)" and _sid_meta:
    st.caption(f"**ID сценария (из config прогона):** `{_sid_meta}`")

if not filtered:
    st.info("Нет данных, соответствующих фильтру.")
else:
    if data_source == "Фоновый запуск (история бенчмарка)":
        st.caption(
            "Отображение как на странице «Бенчмарк»: LLM-оценка, сырой вывод оценщика, "
            "tool_calls и ответы внешнего API по шагам — по режиму из **config** запуска."
        )
    render_dialogs_paginated(
        filtered,
        _disp_eval_mode,
        _disp_llm_eval_fields,
        session_page_key="results_filtered_page_idx",
    )

st.subheader("5. Экспорт данных")

if data_source == "JSONL файлы" and results_file_bytes:
    st.download_button(
        "📥 Скачать исходный файл результатов (JSONL)",
        results_file_bytes,
        "evaluation_results.jsonl",
        "application/json",
    )

if data_source == "JSONL файлы" and failures_file_bytes:
    st.download_button(
        "📥 Скачать исходный failures.jsonl",
        failures_file_bytes,
        "failures.jsonl",
        "application/json",
    )

meta = st.session_state.get("results_analysis_meta") or {}
if data_source != "JSONL файлы" and meta.get("raw_bytes"):
    fn = "results.jsonl"
    if meta.get("run_title"):
        safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in meta["run_title"][:40])
        fn = f"{safe}_{str(meta.get('run_id', ''))[:8]}.jsonl"
    try:
        _parsed_dl = load_jsonl_from_bytes(meta["raw_bytes"])
        _redacted_ndjson = "\n".join(
            json.dumps(redact_secrets_for_display(row), ensure_ascii=False)
            for row in _parsed_dl
        ).encode("utf-8")
    except Exception:
        _redacted_ndjson = meta["raw_bytes"]
    st.download_button(
        "Скачать загруженный results.jsonl (секреты скрыты)",
        _redacted_ndjson,
        fn,
        "application/json",
    )


if filtered:
    jsonl_filtered = "\n".join(
        json.dumps(redact_secrets_for_display(r), ensure_ascii=False) for r in filtered
    )
    st.download_button(
        "📥 Скачать отфильтрованные данные (JSONL, секреты скрыты)",
        jsonl_filtered,
        "filtered_results.jsonl",
        "application/json",
    )
