import json
import os
import uuid
from datetime import timedelta
from pathlib import Path
from typing import Dict, Optional

from dotenv import load_dotenv

# Streamlit часто стартует с cwd не из корня репозитория — подгружаем .env явно
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_REPO_ROOT / ".env")

import streamlit as st

from benchmarking.runner import (
    BenchmarkConfig,
    LLM_JUDGE_EVAL_MODE,
    SEMANTIC_SIMILARITY_EVAL_MODE,
    benchmark_config_to_dict,
    cases_to_assistant_gen_form_rows,
    is_llm_judge_eval_mode,
    is_semantic_similarity_eval_mode,
    normalize_parsed_case,
    parse_benchmark_cases_jsonl_text,
    summarize_context_fields,
)
from benchmarking.page_utils import (
    bench_history_label as _bench_history_label,
    build_benchmark_config,
    load_test_cases_from_jsonl_text as _load_test_cases_from_jsonl_text,
)
from ui.benchmark_eval_form import (
    eval_mode_supports_early_exit,
    render_custom_eval_block,
    render_exit_when_condition_met,
    render_llm_judge_eval_block,
    render_semantic_similarity_eval_block,
)
from ui.dialog_results_ui import (
    render_benchmark_bootstrap_metrics,
    render_dialogs_paginated,
    render_timing_summary_metrics,
)
from ui.judge_settings_ui import (
    ensure_bench_judge_form_defaults,
    flush_pending_bench_judge,
    render_benchmark_judge_loader,
)
from ui.sample_storage_ui import render_export_jsonl_actions, render_jsonl_dataset_source
from common.security import redact_secrets_for_display
from common.time import format_datetime_utc_plus_5
from integrations.litellm import get_model_names
from common.utils import (
    enrich_benchmark_results_timing_inplace,
)

try:
    SUPPORTED_MODELS = get_model_names()
except Exception:
    SUPPORTED_MODELS = [
        os.getenv("LITELLM_MODEL_NAME", "openai/gpt-4o-mini"),
    ]

try:
    from storage.benchmark_backend import (
        cancel_run,
        create_run,
        enrich_benchmark_config_for_storage,
        ensure_schema,
        get_run,
        input_key_for_run,
        list_recent_runs,
        minio_get_bytes,
        minio_put_bytes,
        format_minio_results_miss,
        minio_try_get_bytes,
        minio_try_get_bytes_with_error,
        publish_benchmark_job,
        queue_backend_enabled,
        queue_backend_missing_vars,
        results_key_for_run,
    )

    _QUEUE_MISSING = queue_backend_missing_vars()
    QUEUE_AVAILABLE = queue_backend_enabled()
except ImportError:
    QUEUE_AVAILABLE = False
    _QUEUE_MISSING = ["установите зависимости: psycopg2-binary, pika, minio"]

ROLES = ["assistant", "user", "evaluator"]
for role in ROLES:
    if f"{role}_model" not in st.session_state:
        st.session_state[f"{role}_model"] = SUPPORTED_MODELS[0]
    if f"{role}_api_key" not in st.session_state:
        st.session_state[f"{role}_api_key"] = os.getenv("LITELLM_API_KEY", "")
    if f"{role}_params_json" not in st.session_state:
        defaults = {
            "assistant": '{"temperature": 0.0, "max_tokens": 500}',
            "user": '{"temperature": 0.9, "max_tokens": 150}',
            "evaluator": '{"temperature": 0.0, "max_tokens": 300}',
        }
        st.session_state[f"{role}_params_json"] = defaults[role]

if "evaluation_results" not in st.session_state:
    st.session_state["evaluation_results"] = []
if "_display_eval_mode" not in st.session_state:
    st.session_state["_display_eval_mode"] = None
if "_display_llm_eval_fields" not in st.session_state:
    st.session_state["_display_llm_eval_fields"] = None

_ABENCH_PAGE_KEY = "abench"
flush_pending_bench_judge(_ABENCH_PAGE_KEY)
if "bench_llm_eval_no_gen" not in st.session_state:
    st.session_state["bench_llm_eval_no_gen"] = True


st.set_page_config(page_title="Оценка ассистента", layout="wide")
st.title("🧪 Оценка качества ассистента")

# if QUEUE_AVAILABLE:
#     st.success(
#         "История и живые результаты в блоке **«Фоновые запуски»** доступны "
#         "файл JSONL в браузере после обновления вкладки нужно загрузить снова."
#     )
# else:
#     st.warning(
#         "**В БД ничего не попадёт**, пока очередь выключена — не будет ни блока **«Фоновые запуски»**, ни фонового режима с названием запуска. "
#         "Чаще всего Streamlit не видел `.env` в корне проекта или выбран режим **«Локально»** (он не пишет в `benchmark_runs`).\n\n"
#         f"Не хватает переменных: **{', '.join(_QUEUE_MISSING)}** — см. `.env` рядом с `docker-compose.yml`."
#     )

st.sidebar.subheader("🔑 Настройки LLM")

for role_name, role_label in [
    ("assistant", "Ассистент"),
    ("user", "Симулятор пользователя"),
    ("evaluator", "Оценщик"),
]:
    st.sidebar.markdown(f"### 👤 {role_label}")

    api_key = st.sidebar.text_input(
        f"API Key ({role_label})",
        type="password",
        value=st.session_state[f"{role_name}_api_key"],
        key=f"{role_name}_api_key_input",
    )
    st.session_state[f"{role_name}_api_key"] = api_key

    selected_model = st.sidebar.selectbox(
        f"Модель ({role_label})",
        options=SUPPORTED_MODELS,
        index=min(
            SUPPORTED_MODELS.index(st.session_state[f"{role_name}_model"]),
            len(SUPPORTED_MODELS) - 1,
        )
        if st.session_state[f"{role_name}_model"] in SUPPORTED_MODELS
        else 0,
        key=f"{role_name}_model_input",
    )
    st.session_state[f"{role_name}_model"] = selected_model

    params_json = st.sidebar.text_area(
        f"Параметры JSON ({role_label})",
        value=st.session_state[f"{role_name}_params_json"],
        height=80,
        key=f"{role_name}_params_json_input",
    )
    st.session_state[f"{role_name}_params_json"] = params_json

    st.sidebar.markdown("---")

if not QUEUE_AVAILABLE:
    st.error(
        "Запуск в очередь недоступен: задайте в `.env` переменные "
        "`BENCHMARK_POSTGRES_DSN` (или `DATABASE_URL`), `MINIO_ENDPOINT`, "
        f"`RABBITMQ_URL`. Сейчас не хватает: **{', '.join(_QUEUE_MISSING)}** — страница открыта для просмотра настроек, "
        "кнопка «Запустить» не отправит задачу, пока очередь не настроена."
    )


st.subheader("1. Кейсы: генерация с нуля или JSONL")

_ABENCH_GEN_ROWS_KEY = "assistant_bench_gen_rows"
_ABENCH_SYNTH_CTX_EXAMPLE = """{
  "branch": "отделение на Ленина",
  "schedule": "Пн–Пт 10:00–18:00",
  "city": "Челябинск",
  "product": "домашний интернет"
}"""
_ABENCH_ROW_CTX_PLACEHOLDER = (
    '{\n  "caller_id": "+79001234567",\n  "тариф": "Базовый"\n}'
)


def _abench_ensure_gen_rows():
    if _ABENCH_GEN_ROWS_KEY not in st.session_state:
        st.session_state[_ABENCH_GEN_ROWS_KEY] = [
            {"text": "", "n": 5, "context": {}},
        ]


def _abench_row_context_widget_default(row: dict) -> str:
    c = row.get("context")
    if isinstance(c, dict) and c:
        return json.dumps(c, ensure_ascii=False, indent=2)
    if isinstance(c, str) and c.strip():
        return c.strip()
    return "{}"


def _abench_clear_row_widget_keys() -> None:
    for k in list(st.session_state.keys()):
        if not isinstance(k, str) or not k.startswith("abench_r"):
            continue
        if k.endswith("_text") or k.endswith("_n") or k.endswith("_context"):
            del st.session_state[k]


def _abench_apply_jsonl_to_form(cases: list) -> None:
    rows, common = cases_to_assistant_gen_form_rows(cases)
    _abench_clear_row_widget_keys()
    st.session_state[_ABENCH_GEN_ROWS_KEY] = rows
    st.session_state["abench_synth_ctx"] = json.dumps(
        common, ensure_ascii=False, indent=2
    )
    for ri, row in enumerate(rows):
        st.session_state[f"abench_r{ri}_text"] = row.get("text", "")
        st.session_state[f"abench_r{ri}_n"] = int(row.get("n", 1))
        st.session_state[f"abench_r{ri}_context"] = _abench_row_context_widget_default(
            row
        )


def _abench_reset_gen_form_to_default() -> None:
    _abench_clear_row_widget_keys()
    st.session_state[_ABENCH_GEN_ROWS_KEY] = [{"text": "", "n": 5, "context": {}}]
    st.session_state["abench_synth_ctx"] = "{}"
    st.session_state["abench_r0_text"] = ""
    st.session_state["abench_r0_n"] = 5
    st.session_state["abench_r0_context"] = "{}"


def _abench_build_cases_from_gen_session() -> tuple[list, Optional[str]]:
    """Кейсы как строки JSONL (после normalize_parsed_case), без dm_stop_at_block_id."""
    ctx_raw = st.session_state.get("abench_synth_ctx", "{}")
    try:
        ctx_obj = json.loads((ctx_raw or "").strip() or "{}")
        if not isinstance(ctx_obj, dict):
            return [], "Общий context должен быть JSON-объектом."
    except (json.JSONDecodeError, ValueError) as e:
        return [], f"Невалидный JSON в общем context: {e}"
    test_cases_local: list = []
    rows_struct = st.session_state.get(_ABENCH_GEN_ROWS_KEY) or []
    for ri, _row in enumerate(rows_struct):
        kt = f"abench_r{ri}_text"
        kn = f"abench_r{ri}_n"
        kc = f"abench_r{ri}_context"
        tx = (st.session_state.get(kt) or "").strip()
        try:
            raw_row_ctx = (st.session_state.get(kc) or "").strip() or "{}"
            row_ctx_parsed = json.loads(raw_row_ctx)
            if not isinstance(row_ctx_parsed, dict):
                raise ValueError("ожидается JSON-объект {...}")
        except (json.JSONDecodeError, ValueError) as e:
            return [], (
                f"Невалидный персональный `context` в строке **{ri + 1}**: {e}. "
                "Исправьте JSON или оставьте `{{}}`."
            )
        merged_ctx = {**ctx_obj, **row_ctx_parsed}
        n_dialogs = int(st.session_state.get(kn, 1) or 1)
        n_dialogs = max(1, min(500, n_dialogs))
        for _ in range(n_dialogs):
            case = {
                "dialog_id": str(uuid.uuid4()),
                "goals": [tx] if tx else [],
                "context": dict(merged_ctx),
            }
            normalize_parsed_case(case)
            test_cases_local.append(case)
    return test_cases_local, None


cases_source = st.radio(
    "Откуда брать список диалогов для прогона",
    options=["Генерация диалогов с нуля", "Файл JSONL", "Сохранённые выборки"],
    horizontal=True,
    help="Пустые кейсы без `history`: диалог строится через ассистента и симулятора по цели и `context`.",
)

uploaded_file = None
test_cases: list = []

if cases_source == "Генерация диалогов с нуля":
    _abench_ensure_gen_rows()
    gen_rows = st.session_state[_ABENCH_GEN_ROWS_KEY]

    st.caption(
        "Каждая **строка** — отдельная цель: в кейсе будет `goals` из одной этой строки. "
        "**Диалогов** — сколько таких кейсов сгенерировать (разные `dialog_id`). "
        "Персональный **context (JSON)** в строке объединяется с общим блоком: совпадающие ключи берутся из строки."
    )

    st.markdown("**Или загрузите JSONL-файл с кейсами**")
    st.caption(
        "После **Применить JSONL** заполняются **Цель**, **Блок цели**, "
        "**Диалогов**, персональный и **общий** `context`."
    )
    _abench_json_upl = st.file_uploader(
        "Кейсы для прогона (.jsonl)",
        type=["jsonl"],
        key="abench_upload_gen_cases_jsonl",
        help="Одна строка = один кейс: goals, context; опционально dialog_id.",
    )
    _jc1, _jc2 = st.columns(2)
    with _jc1:
        if st.button(
            "Применить JSONL",
            key="abench_apply_gen_cases_jsonl",
            disabled=_abench_json_upl is None,
        ):
            if _abench_json_upl is not None:
                try:
                    raw_jsonl = _abench_json_upl.getvalue().decode("utf-8")
                    _parsed, _perr = parse_benchmark_cases_jsonl_text(raw_jsonl)
                    if _perr:
                        st.error(_perr)
                    else:
                        _abench_apply_jsonl_to_form(_parsed)
                        st.success(
                            f"Загружено **{len(_parsed)}** кейс(ов): форма и общий `context` обновлены."
                        )
                        st.rerun()
                except UnicodeDecodeError as e:
                    st.error(f"Не удалось прочитать файл как UTF-8: {e}")
    with _jc2:
        if st.button("Сбросить форму генерации", key="abench_clear_gen_cases_jsonl"):
            _abench_reset_gen_form_to_default()
            st.rerun()

    h0, h1, h2 = st.columns([4, 1, 1])
    with h0:
        st.caption("Цель (`{goals}`)")
    with h1:
        st.caption("Диалогов")
    with h2:
        st.caption("")

    for ri, row in enumerate(gen_rows):
        kt = f"abench_r{ri}_text"
        kn = f"abench_r{ri}_n"
        if kt not in st.session_state:
            st.session_state[kt] = row.get("text", "")
        if kn not in st.session_state:
            st.session_state[kn] = int(row.get("n", 5))
        kc = f"abench_r{ri}_context"
        if kc not in st.session_state:
            st.session_state[kc] = _abench_row_context_widget_default(row)
        # text_area раньше кнопок ➕/➖: иначе st.rerun() по клику не даёт дойти до виджета — context в session_state теряется.
        st.text_area(
            f"Персональный `context` для кейса {ri + 1} (JSON-объект, опционально)",
            key=kc,
            height=72,
            placeholder=_ABENCH_ROW_CTX_PLACEHOLDER,
            help="Объединяется с общим context ниже: ключи строки перекрывают общие. Пусто или {} — только общий блок.",
        )
        rc1, rc2, rc3 = st.columns([4, 1, 1])
        with rc1:
            st.text_input(
                "Цель",
                key=kt,
                label_visibility="collapsed",
                placeholder="например: уточнить адрес отделения",
            )
        with rc2:
            st.number_input(
                "Диалогов",
                min_value=1,
                max_value=500,
                key=kn,
                label_visibility="collapsed",
                help="Сколько кейсов для этой цели",
            )
        with rc3:
            if ri == len(gen_rows) - 1:
                if st.button("➕", key="abench_add_row", help="Добавить строку цели"):
                    st.session_state[_ABENCH_GEN_ROWS_KEY].append(
                        {"text": "", "n": 5, "context": {}}
                    )
                    st.rerun()
                if len(gen_rows) > 1 and st.button(
                    "➖", key="abench_rm_row", help="Удалить последнюю строку"
                ):
                    last = len(gen_rows) - 1
                    for k in list(st.session_state.keys()):
                        if isinstance(k, str) and k.startswith(f"abench_r{last}_"):
                            del st.session_state[k]
                    st.session_state[_ABENCH_GEN_ROWS_KEY].pop()
                    st.rerun()

    if "abench_synth_ctx" not in st.session_state:
        st.session_state["abench_synth_ctx"] = "{}"
    synthetic_context_json = st.text_area(
        "Общий базовый `context` для всех кейсов (JSON-объект, опционально)",
        height=140,
        key="abench_synth_ctx",
        placeholder='{} или скопируйте пример из блока ниже',
        help="Плейсхолдеры в промптах: `{branch}`, `{city}` и т.д. в snake_case.",
    )
    with st.expander("Пример задания `context`", expanded=False):
        st.caption("Скопируйте в поле выше при необходимости.")
        st.code(_ABENCH_SYNTH_CTX_EXAMPLE.strip(), language="json")

    with st.expander("Скачивание сгенерированных кейсов (JSONL)", expanded=False):
        st.caption(
            "**JSONL** — один JSON-объект на строку, как в режиме «Файл JSONL»: "
            "`dialog_id`, `goals`, `context`."
        )
        cases_export, cases_export_err = _abench_build_cases_from_gen_session()
        jsonl_body = (
            "\n".join(json.dumps(c, ensure_ascii=False) for c in cases_export)
            + ("\n" if cases_export else "")
        ).encode("utf-8")
        st.download_button(
            "Скачать кейсы (JSONL)",
            data=jsonl_body if cases_export and not cases_export_err else b"",
            file_name="assistant_benchmark_cases.jsonl",
            mime="application/x-ndjson",
            disabled=bool(cases_export_err) or not cases_export,
            key="abench_download_gen_jsonl",
        )
        if cases_export and not cases_export_err:
            from ui.sample_storage_ui import render_save_sample_to_db

            render_save_sample_to_db(
                jsonl_body.decode("utf-8"),
                key_prefix="abench_gen_cases",
                case_count=len(cases_export),
                description="Сгенерированные кейсы бенчмарка ассистента",
                name_placeholder="abench-gen-cases-v1",
            )
        if cases_export_err:
            st.caption(f"⚠️ JSONL недоступен: {cases_export_err}")

    try:
        _ctx_check = json.loads((synthetic_context_json or "").strip() or "{}")
        if not isinstance(_ctx_check, dict):
            raise ValueError("context должен быть JSON-объектом")
    except (json.JSONDecodeError, ValueError) as e:
        st.error(f"Невалидный JSON в «Общий context»: {e}")
        test_cases = []
    else:
        test_cases, gen_build_err = _abench_build_cases_from_gen_session()
        if gen_build_err:
            st.error(gen_build_err)
            test_cases = []
        elif not test_cases:
            st.warning("Нет строк для генерации кейсов — добавьте строку в форме или загрузите JSONL.")
        else:
            st.success(
                f"Будет **{len(test_cases)}** диалог(ов) без загруженной истории — цели (в т.ч. пустые) и context."
            )

elif cases_source == "Файл JSONL":
    uploaded_file = st.file_uploader("Загрузите файл .jsonl", type=["jsonl"])
    if uploaded_file:
        try:
            text = uploaded_file.getvalue().decode("utf-8")
            for line in text.strip().split("\n"):
                if line.strip():
                    case = json.loads(line)
                    normalize_parsed_case(case)
                    if "dialog_id" not in case:
                        case["dialog_id"] = str(uuid.uuid4())
                    test_cases.append(case)
            st.success(f"Загружено {len(test_cases)} диалогов из файла.")
        except json.JSONDecodeError as e:
            st.error(
                f"Ошибка разбора JSON в строке JSONL: {e}\n\n"
                "**Частая причина:** поле `context` оформлено как строка с «вторым» JSON внутри, "
                "а кавычки у вложенных полей не экранированы — такая строка **невалидна** как JSON.\n\n"
                "**Как исправить:** задайте `context` как **вложенный объект**, без внешних кавычек вокруг объекта:\n"
                '`"context": {"schedule": "Пн–Пт 10–18", "branch": "центр"}`\n\n'
                "Либо одна строка с экранированием: "
                '`"context": "{\\"schedule\\": \\"...\\"}"`.'
            )
            test_cases = []
        except Exception as e:
            st.error(f"Ошибка загрузки: {e}")
            test_cases = []
    else:
        st.caption(
            "После обновления вкладки файл из браузера пропадает — **историю запусков и живые результаты** "
            "смотрите **внизу страницы** (раздел «Фоновые запуски»). "
            "Загрузите JSONL, выберите выборку из БД или переключитесь на «Генерация диалогов с нуля»."
        )

elif cases_source == "Сохранённые выборки":
    _db_msg = st.session_state.pop("_abench_db_loaded_msg", None)
    if _db_msg:
        st.success(_db_msg)

    def _on_abench_dataset_text(text: str) -> None:
        try:
            cases = _load_test_cases_from_jsonl_text(text)
            st.session_state["_abench_test_cases"] = cases
            st.session_state["_abench_db_loaded_msg"] = (
                f"Загружено **{len(cases)}** диалогов."
            )
            st.rerun()
        except Exception as e:
            st.error(str(e))

    render_jsonl_dataset_source(
        key_prefix="abench_saved",
        on_text_loaded=_on_abench_dataset_text,
        file_types=["jsonl"],
        upload_label="Датасет (.jsonl)",
    )
    test_cases = list(st.session_state.get("_abench_test_cases") or [])

if test_cases:
    ctx_summary = summarize_context_fields(test_cases)
    with_context = sum(
        1 for c in test_cases if isinstance(c.get("context"), dict) and c["context"]
    )
    if ctx_summary:
        keys_line = ", ".join(f"`{{{k}}}`" for k in ctx_summary)
        st.info(
            f"В датасете **{with_context}** из **{len(test_cases)}** диалогов с непустым `context`. "
            f"Поля для подстановки в промпты: {keys_line}. "
            "У диалогов без поля в `context` подставляется пустая строка."
        )
        with st.expander("Примеры значений `context` (до 3 на поле)", expanded=False):
            for field, examples in ctx_summary.items():
                st.markdown(f"**`{field}`**")
                for i, ex in enumerate(examples, 1):
                    st.caption(f"Пример {i}")
                    st.code(ex)
    else:
        st.caption(
            "Опционально: в каждой строке JSONL можно задать объект `context` "
            "(строки, числа, вложенные объекты). В промптах используйте плейсхолдеры "
            "`{имя_поля}` — как для `{goals}`. Если у диалога поля в `context` нет, "
            "в промпт подставляется пустая строка. Имена полей лучше в `snake_case` "
            "(в `{...}` недопустимы дефисы и пробелы)."
        )

st.subheader("2. Настройка ассистента и симулятора")

mode = st.radio(
    "Как отвечает оператор?", options=["LLM (через LiteLLM)", "Внешний URL"]
)
if mode == "Внешний URL":
    assistant_url = st.text_input(
        "URL ассистента", value="http://10.80.0.148:8000/chat-new"
    )
    st.caption(
        "В тело POST попадает `context` кейса (или поля по маппингу ниже), плюс **идентификатор диалога** и **текст реплики пользователя** "
        "— имена полей задаются отдельными полями (не фиксированные `id` / `message`)."
    )
    external_context_field_map_json = st.text_area(
        "Соответствие: поле в теле API → поле в context (JSON-объект)",
        value="",
        height=120,
        placeholder='{\n  "user_id": "user_id",\n  "building_id": "building_id",\n  "flat_id": "flat_id",\n  "phone": "phone"\n}',
        help="Ключ слева — имя поля в JSON тела запроса; значение справа — ключ из `context` кейса. "
        "Пустое поле — в запрос уходит весь `context`. `{}` — не добавлять поля из context (только id диалога и реплика).",
    )
    external_session_id_field = st.text_input(
        "Ключ id диалога в теле API",
        value="id",
        help="Например `dialog_id`, если так ожидает бэкенд. Значение — из поля `dialog_id` кейса или новый UUID (см. чекбокс ниже).",
    )
    external_unique_session_id = st.checkbox(
        "Новый уникальный id в теле API (не брать из кейса / разметки)",
        value=False,
        key="bench_external_unique_session_id",
        help="Если включено — в каждый POST уходит новый UUID в поле «Ключ id диалога», даже если в JSONL задан `dialog_id`. "
        "Иначе — как в кейсе, а при отсутствии — новый UUID.",
    )
    user_message_key = st.text_input(
        "Ключ реплики пользователя",
        value="message",
        help="Например `user_text`. Раньше в коде всегда уходило `message` — теперь используется это поле.",
    )
    external_coerce_int_fields_csv = st.text_input(
        "Поля целого типа в теле API (через запятую)",
        value="flat_id",
        help="Для перечисленных имён, если в JSON пришла строка из цифр, значение приводится к int (реже 422 у FastAPI). "
        "Пусто — без приведения.",
    )
    response_field_path = st.text_input("Путь к полю с ответом", value="response")
    use_tools = False
    assistant_prompt = ""
    parse_json_response = False
    assistant_tools = ""
else:
    assistant_url = ""
    external_context_field_map_json = ""
    external_session_id_field = "id"
    external_unique_session_id = False
    external_coerce_int_fields_csv = ""
    user_message_key = "message"
    response_field_path = "response"
    st.caption(
        "Плейсхолдеры: `{goals}`, `{history}`, `{user_prompt}`, `{available_tools}` и любые ключи из `context` кейса."
    )
    assistant_prompt = st.text_area(
        "Промпт для LLM-ассистента", value="Вы — оператор..."
    )
    parse_json_response = st.checkbox(
        "Парсить ответ как JSON (если возможно)", value=False
    )
    use_tools = st.checkbox("Использовать инструменты (tools)", value=False)

    assistant_tools = st.text_area(
        "Инструменты ассистента (в формате JSON)",
        value="""[
        [
            {
                "type": "function",
                "function": {
                "name": "get_ticket_num",
                "description": "Если ты определила по предмету залога номер залогового билета, то выполни эту функцию и обязательно верни этот номер. Если подходящих билета два, верни их одной строкой через пробел",
                "parameters": {
                    "type": "object",
                    "properties": {
                    "ticket_num": {
                        "type": "string",
                        "description": "Номер залогового билета"
                    }
                    },
                    "required": ["ticket_num"]
                }
            }
        }
            
    ]""",
        height=150,
        disabled=not use_tools,
    )

st.caption(
    "Симулятор: шаблон подставляет `{goals}` и поля из `context` (например `{schedule}`)."
)
user_prompt = st.text_area(
    "Промпт для симулятора пользователя",
    value="""Вы — обычный клиент, который звонит в ломбард "Фианит". Ваши цели: {goals}.

Говорите ТОЛЬКО как клиент: задавайте вопросы, просите помощи, описывайте проблему.
Можно быть: вежливым, грубым, растерянным.
Ответьте ровно одной репликой. Не объясняйте.""",
    height=150,
)

max_turns = st.slider("Макс. число ходов", 2, 10, 5)
bench_first_speaker = st.radio(
    "Кто первым говорит в новом диалоге",
    options=["Пользователь", "Ассистент"],
    index=0,
    horizontal=True,
    help="Используется, если в кейсе **пустая** `history` (в т.ч. генерация с нуля). "
    "Если в JSONL уже есть реплики — порядок задаётся ими.",
    key="bench_first_speaker",
)
assistant_speaks_first = bench_first_speaker == "Ассистент"
st.subheader("3. Дополнительные параметры")

llm_delay = st.number_input(
    "Задержка перед каждым вызовом LLM (секунды)",
    min_value=0.0,
    max_value=100.0,
    value=0.0,
    step=0.1,
    help="Используется как time.sleep() перед каждым вызовом LLM (ассистент или симулятор).",
)
rc1, rc2 = st.columns(2)
with rc1:
    repeats_per_case = st.number_input(
        "Повторов на кейс (repeats_per_case)",
        min_value=1,
        max_value=10,
        value=1,
        step=1,
        help="Сколько параллельных прогонов одного dialog_id. accuracy = mean@k.",
    )
with rc2:
    repeats_stagger_sec = st.number_input(
        "Задержка между стартами повторов (с)",
        min_value=0.0,
        max_value=10.0,
        value=1.0,
        step=0.5,
        disabled=int(repeats_per_case) <= 1,
        help="Повтор i стартует через i× секунд; запросы идут параллельно, не дожидаясь ответа первого.",
    )
st.subheader("4. Режим оценки точности")

eval_mode = st.radio(
    "Выберите способ оценки:",
    options=[
        "Сравнить цель с полем из ответа",
        "Кастомный критерий (Python)",
        LLM_JUDGE_EVAL_MODE,
        SEMANTIC_SIMILARITY_EVAL_MODE,
        "Проверка вызова тулзов",
    ],
    index=0,
)

custom_eval_code = ""
evaluate_existing_only = False
llm_eval_prompt = ""
llm_eval_fields = "result,reason"
eval_field_path = ""
semantic_pred_field_path = "theme"
semantic_ref_field_path = "subtopic"
semantic_similarity_threshold = 0.85

if eval_mode == "Сравнить цель с полем из ответа":
    if mode == "Внешний URL":
        eval_field_path = st.text_input("Путь к полю для сравнения", value="theme")
    else:
        eval_field_path = "response"
        st.info("Используется полный текст ответа для сравнения.")
    custom_eval_code = "goals.strip().lower() in str(eval_value).strip().lower()"

elif eval_mode == "Кастомный критерий (Python)":
    custom_eval_code = render_custom_eval_block(
        default_code='response.get("address", "").get("status") == "full"',
    )
    eval_field_path = ""


elif is_llm_judge_eval_mode(eval_mode):
    _llm_eval_existing_sources = ("Файл JSONL", "Сохранённые выборки")
    llm_eval_prompt, llm_eval_fields, evaluate_existing_only = render_llm_judge_eval_block(
        page_key=_ABENCH_PAGE_KEY,
        eval_mode=eval_mode,
        queue_available=QUEUE_AVAILABLE,
        queue_missing=_QUEUE_MISSING,
        use_tools=use_tools,
        allow_existing_only=cases_source in _llm_eval_existing_sources,
        existing_only_key="bench_llm_eval_no_gen",
        existing_only_help=(
            "Если включено — ассистент и симулятор не добавляют реплики; LLM-оценщик получает только поле "
            "`history` из кейса (например из JSONL). Если выключено — сначала прогоняется диалог до лимита ходов, "
            "затем оценка."
        ),
        existing_only_caption=(
            "Режим «только существующая история» доступен при источнике «Файл JSONL» или «Сохранённые выборки» "
            "(в кейсах должно быть поле `history`)."
        ),
    )

elif is_semantic_similarity_eval_mode(eval_mode):
    _sem_pred_default = "theme" if mode == "Внешний URL" else "response"
    semantic_pred_field_path, semantic_ref_field_path, semantic_similarity_threshold = (
        render_semantic_similarity_eval_block(
            pred_default=_sem_pred_default,
            ref_default="subtopic",
        )
    )
    evaluate_existing_only = False

else:
    custom_eval_code = ""
    llm_eval_prompt = ""
    llm_eval_fields = "result,reason"

exit_when_condition_met = False
if eval_mode_supports_early_exit(eval_mode):
    exit_when_condition_met = render_exit_when_condition_met()

st.subheader("5. Запуск")

st.caption(
    "Задача уходит в **очередь RabbitMQ**; воркер пишет прогресс в Postgres и **results.jsonl** в MinIO. "
    "Подпись запуска в БД помогает отличать прогоны в **истории внизу страницы**."
)
c_nm, c_ds = st.columns([1, 1])
with c_nm:
    st.text_input(
        "Название запуска",
        placeholder="например: baseline v2, сравнение промптов",
        key="bench_queue_title",
        help="Сохраняется в Postgres; в истории можно искать по этому тексту.",
    )
with c_ds:
    st.text_area(
        "Описание (опционально)",
        placeholder="Датасет, гипотеза, что меняли в настройках…",
        key="bench_queue_desc",
        height=88,
        help="Участвует в поиске в блоке «Фоновые запуски» внизу страницы.",
    )

run_button = st.button("🚀 Запустить оценку в очередь", key="run_eval")

bench_cfg = build_benchmark_config(
    session_state=st.session_state,
    supported_models=SUPPORTED_MODELS,
    mode=mode,
    assistant_url=assistant_url if mode == "Внешний URL" else "",
    user_message_key=user_message_key,
    response_field_path=response_field_path,
    assistant_prompt=assistant_prompt,
    parse_json_response=parse_json_response,
    use_tools=use_tools,
    assistant_tools=assistant_tools,
    user_prompt=user_prompt,
    max_turns=max_turns,
    llm_delay=llm_delay,
    repeats_per_case=int(repeats_per_case),
    repeats_stagger_sec=float(repeats_stagger_sec),
    eval_mode=eval_mode,
    eval_field_path=eval_field_path if eval_mode == "Сравнить цель с полем из ответа" else "",
    custom_eval_code=custom_eval_code,
    evaluate_existing_only=evaluate_existing_only,
    llm_eval_prompt=llm_eval_prompt if is_llm_judge_eval_mode(eval_mode) else "",
    llm_eval_fields=llm_eval_fields if is_llm_judge_eval_mode(eval_mode) else "result,reason",
    semantic_pred_field_path=semantic_pred_field_path
    if is_semantic_similarity_eval_mode(eval_mode)
    else "theme",
    semantic_ref_field_path=semantic_ref_field_path
    if is_semantic_similarity_eval_mode(eval_mode)
    else "subtopic",
    semantic_similarity_threshold=semantic_similarity_threshold
    if is_semantic_similarity_eval_mode(eval_mode)
    else 0.85,
    exit_when_condition_met=exit_when_condition_met,
    external_context_field_map_json=(
        external_context_field_map_json if mode == "Внешний URL" else ""
    ),
    external_session_id_field=(
        external_session_id_field if mode == "Внешний URL" else "id"
    ),
    external_unique_session_id=(
        external_unique_session_id if mode == "Внешний URL" else False
    ),
    external_coerce_int_fields_csv=(
        external_coerce_int_fields_csv if mode == "Внешний URL" else ""
    ),
    assistant_speaks_first=assistant_speaks_first,
)

if run_button:
    if not QUEUE_AVAILABLE:
        st.error(
            "Очередь не настроена — см. сообщение выше. Исправьте `.env` и перезапустите Streamlit."
        )
    elif not test_cases:
        st.error(
            "Нет кейсов для прогона: загрузите JSONL с кейсами или в режиме «Генерация с нуля» настройте строки "
            "(цель может быть пустой — в промптах подставится «Без цели»)."
        )
    else:
        try:
            ensure_schema()
            run_title_saved = (st.session_state.get("bench_queue_title") or "").strip()
            run_desc_saved = (st.session_state.get("bench_queue_desc") or "").strip()
            run_id = str(uuid.uuid4())
            ik = input_key_for_run(run_id)
            create_run(
                enrich_benchmark_config_for_storage(benchmark_config_to_dict(bench_cfg)),
                ik,
                run_id=run_id,
                title=run_title_saved,
                description=run_desc_saved,
            )
            if cases_source == "Файл JSONL" and uploaded_file is not None:
                input_ndjson_bytes = uploaded_file.getvalue()
            else:
                input_ndjson_bytes = (
                    "\n".join(
                        json.dumps(c, ensure_ascii=False, separators=(",", ":"))
                        for c in test_cases
                    )
                    + ("\n" if test_cases else "")
                ).encode("utf-8")
            minio_put_bytes(ik, input_ndjson_bytes, "application/x-ndjson")
            publish_benchmark_job(run_id)
            title_echo = run_title_saved or "(без названия)"
            st.session_state["_bench_queue_notice"] = (run_id, title_echo)
            st.rerun()
        except Exception as e:
            st.error(f"Ошибка постановки в очередь: {e}")

disp_mode = st.session_state.get("_display_eval_mode") or eval_mode
disp_llm_fields = st.session_state.get("_display_llm_eval_fields") or llm_eval_fields

if st.session_state["evaluation_results"]:
    results = st.session_state["evaluation_results"]
    enrich_benchmark_results_timing_inplace(results)
    total = len(results)
    avg_acc = sum(r["accuracy"] for r in results) / total if total > 0 else 0

    st.subheader("📊 Итоги")
    st.metric("Точность", f"{avg_acc:.2%}")
    st.metric("Диалогов", total)
    render_benchmark_bootstrap_metrics(
        results,
        llm_eval_fields=disp_llm_fields,
        eval_mode=disp_mode,
        key_prefix="abench_bootstrap",
    )
    render_timing_summary_metrics(results)

    st.subheader("📋 Все диалоги")
    render_dialogs_paginated(
        results,
        disp_mode,
        disp_llm_fields,
        session_page_key="bench_results_page_idx",
        show_timing_summary=False,
    )

    failures = [r for r in results if r["accuracy"] == 0]
    not_failures = [r for r in results if r["accuracy"] == 1]
    if failures:
        st.subheader(f"⛔ Ошибки ({len(failures)})")
        render_dialogs_paginated(
            failures,
            disp_mode,
            disp_llm_fields,
            session_page_key="bench_failures_page_idx",
            list_entity_caption="ошибки",
            expander_row_label="Ошибка",
            expander_title_show_accuracy_icon=False,
            show_dialog_level_api_response=True,
            show_timing_summary=True,
        )

    st.subheader("📥 Экспорт")
    jsonl_all = "\n".join(
        json.dumps(redact_secrets_for_display(r), ensure_ascii=False, default=str)
        for r in results
    )
    render_export_jsonl_actions(
        jsonl_all,
        key_prefix="abench_export_all",
        case_count=len(results),
        file_name="evaluation_results.jsonl",
        description="Результаты бенчмарка ассистента (секреты скрыты)",
        name_placeholder="abench-results-v1",
    )

    if failures:
        jsonl_fail = "\n".join(
            json.dumps(redact_secrets_for_display(r), ensure_ascii=False, default=str)
            for r in failures
        )
        render_export_jsonl_actions(
            jsonl_fail,
            key_prefix="abench_export_fail",
            case_count=len(failures),
            file_name="failures.jsonl",
            description="Ошибки бенчмарка ассистента",
            name_placeholder="abench-failures-v1",
        )
    if not_failures:
        jsonl_not_fail = "\n".join(
            json.dumps(redact_secrets_for_display(r), ensure_ascii=False, default=str)
            for r in not_failures
        )
        render_export_jsonl_actions(
            jsonl_not_fail,
            key_prefix="abench_export_ok",
            case_count=len(not_failures),
            file_name="success.jsonl",
            description="Успешные диалоги бенчмарка ассистента",
            name_placeholder="abench-success-v1",
        )

st.markdown("---")

if QUEUE_AVAILABLE:
    @st.fragment(run_every=timedelta(seconds=2))
    def _bench_live_poll():
        """Живое обновление статуса и диалогов из MinIO."""
        rid = st.session_state.get("_bench_live_run_id")
        if not rid:
            return
        holder = st.empty()
        with holder.container():
            r = get_run(rid)
            if not r:
                st.caption("Запуск не найден.")
                return
            rt = (r.get("run_title") or "").strip()
            rd = (r.get("run_description") or "").strip()
            if rt:
                st.markdown(f"**{rt}**")
            if rd:
                st.caption(rd)

            rk = r.get("results_key") or results_key_for_run(rid)
            raw, minio_err = minio_try_get_bytes_with_error(rk)
            partial: list = []
            if raw is not None:
                lines = [ln for ln in raw.decode("utf-8").split("\n") if ln.strip()]
                partial = [json.loads(ln) for ln in lines]
                enrich_benchmark_results_timing_inplace(partial)

            cfg = r.get("config") or {}
            if isinstance(cfg, str):
                cfg = json.loads(cfg)
            em = cfg.get("eval_mode", "Сравнить цель с полем из ответа")
            lf = cfg.get("llm_eval_fields", "result,reason")

            st.metric("Статус", r["status"])
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Прогресс", f"{r['progress_done']}/{r['progress_total']}")
            with c2:
                a = r.get("avg_accuracy")
                st.metric("Средняя точность", f"{a:.2%}" if a is not None else "—")
            with c3:
                if r.get("error"):
                    st.error(r["error"][:500])

            if partial:
                render_timing_summary_metrics(partial, show_help=True)

            if r["status"] in ("queued", "running"):
                st.caption(
                    "Диалоги подгружаются из MinIO каждые ~2 с по мере записи воркером."
                )

            if raw is None:
                level, msg = format_minio_results_miss(
                    rk, run_status=r["status"], minio_error=minio_err
                )
                if level == "caption":
                    st.caption(msg)
                else:
                    st.warning(msg)
            else:
                if partial:
                    st.subheader(f"📋 Результаты ({len(partial)} диалогов)")
                    render_dialogs_paginated(
                        partial,
                        em,
                        lf,
                        session_page_key="bench_live_page_idx",
                        show_timing_summary=False,
                    )
                elif r["status"] in ("queued", "running"):
                    st.caption("Ожидание первого диалога (пустой файл в MinIO)…")

if QUEUE_AVAILABLE:
    st.subheader("📡 Фоновые запуски")
    _queued = st.session_state.pop("_bench_queue_notice", None)
    if _queued:
        _qid, _qtitle = _queued
        st.success(
            f"Запуск **«{_qtitle}»** отправлен в очередь. UUID: `{_qid}`. "
            "История запусков ниже на этой странице — можно искать по названию в фильтре."
        )
    try:
        ensure_schema()
    except Exception as e:
        st.error(f"БД недоступна: {e}")
    else:
        st.markdown("**Поиск по названию, описанию или фрагменту UUID** (данные из БД).")
        history_search = st.text_input(
            "Фильтр",
            placeholder="например: baseline, промпт, 3f2a1b…",
            key="bench_history_search",
            label_visibility="collapsed",
        )
        runs = list_recent_runs(100, search=history_search or None)
        if history_search.strip() and not runs:
            st.caption("Ничего не найдено — очистите фильтр или проверьте написание.")
        run_options = {_bench_history_label(r): str(r["id"]) for r in runs}
        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            pick = st.selectbox(
                "История запусков",
                options=list(run_options.keys()) if run_options else ["(нет записей)"],
                key="bench_history_pick",
            )
        selected_run_id = run_options.get(pick) if run_options and pick != "(нет записей)" else None
        if selected_run_id:
            st.session_state["_bench_live_run_id"] = selected_run_id
        else:
            st.session_state.pop("_bench_live_run_id", None)

        with c2:
            if st.button("Обновить список", key="bench_refresh_runs"):
                st.rerun()
        with c3:
            load_btn = st.button("Загрузить результаты в UI", key="bench_load_results")

        if selected_run_id and load_btn:
            row = get_run(selected_run_id)
            if row:
                rk = row.get("results_key") or results_key_for_run(selected_run_id)
                try:
                    raw = minio_try_get_bytes(rk)
                    if raw is None:
                        st.warning(
                            "Файл результатов в MinIO ещё недоступен "
                            "(воркер не подхватил задачу или запись без ключа)."
                        )
                    else:
                        lines = [
                            ln for ln in raw.decode("utf-8").split("\n") if ln.strip()
                        ]
                        loaded = [json.loads(ln) for ln in lines]
                        enrich_benchmark_results_timing_inplace(loaded)
                        st.session_state["evaluation_results"] = loaded
                        cfg = row["config"]
                        if isinstance(cfg, str):
                            cfg = json.loads(cfg)
                        st.session_state["_display_eval_mode"] = cfg.get(
                            "eval_mode", "Сравнить цель с полем из ответа"
                        )
                        st.session_state["_display_llm_eval_fields"] = cfg.get(
                            "llm_eval_fields", "result,reason"
                        )
                        st.success(
                            f"Загружено {len(lines)} строк из запуска "
                            f"{selected_run_id[:8]}…"
                        )
                except Exception as ex:
                    st.error(str(ex))

        _bench_live_poll()

        if selected_run_id and st.button("Отменить выбранный запуск", key="bench_cancel"):
            if cancel_run(selected_run_id):
                st.success("Запрошена отмена (воркер остановится после текущего диалога).")
            else:
                st.info("Нельзя отменить (уже завершён или не найден).")
