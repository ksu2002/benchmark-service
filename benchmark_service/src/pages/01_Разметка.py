import ast
import hashlib
import io
import json
import os
import re
import uuid
import litellm
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from integrations.litellm import get_model_names
from common.utils import (
    TOOL_EVENT_SOURCE_HTTP_VARIABLES,
    TOOL_EVENT_SOURCE_TECH_LOG,
    extract_tool_calls_from_jsonl_history_message,
    format_tool_calls_json_pretty,
    is_tool_event_from_http_variables,
    markup_timeline_row_icon,
    restrict_variables_rows_to_allowlist,
    restrict_variables_snapshot_to_allowlist,
    skip_timeline_tech_log_tool_when_http_variables_ui,
    tech_log_openai_schema_as_markup_tool_calls,
    tool_calls_for_timeline_ui_display,
)
from tools.markup_jsonl_io import parse_jsonl_text_to_dialog_groups
from ui.sample_storage_ui import render_export_jsonl_actions, render_jsonl_dataset_source

load_dotenv()
SUPPORTED_MODELS = get_model_names()

# Примеры для подсказок context (JSON-объект, ключи в кавычках).
_EXPORT_CONTEXT_JSON_EXAMPLE_GLOBAL = """{
  "branch": "отделение на Ленина",
  "city": "Челябинск",
  "product": "домашний интернет"
}"""
_EXPORT_CONTEXT_JSON_EXAMPLE_DIALOG = """{
  "caller_id": "+777777777",
  "тариф": "Базовый"
}"""


def toggle_expander(dialog_id):
    """Переключает состояние раскрытия диалога"""
    current_state = st.session_state.get(f"expanded_{dialog_id}", False)
    st.session_state[f"expanded_{dialog_id}"] = not current_state

@st.cache_resource(ttl=3600)
def get_clickhouse_client():
    """NDA: параметры подключения к ClickHouse и клиент скрыты."""
    raise NotImplementedError(
        "NDA: подключение к ClickHouse недоступно в публичной версии репозитория"
    )


def clickhouse_select_to_df(client, query: str) -> pd.DataFrame:
    """NDA: выполнение SELECT к ClickHouse скрыто."""
    raise NotImplementedError(
        "NDA: SQL-запросы к ClickHouse недоступны в публичной версии репозитория"
    )


def get_litellm_kwargs():
    model = st.session_state.get("selected_model", SUPPORTED_MODELS[0])
    api_key = st.session_state.get("litellm_api_key", "").strip() or os.getenv(
        "LITELLM_API_KEY", None
    )
    api_base = os.getenv("LITELLM_API_BASE", None)
    kwargs = {
        "model": model,
        "api_key": api_key or None,
        "api_base": api_base or None,
        "extra_body": {"cache": {"no-cache": True}},
    }
    user_json_str = st.session_state.get("llm_params_json", "{}")
    if isinstance(user_json_str, str) and user_json_str.strip():
        try:
            model_params = json.loads(user_json_str.strip())
            kwargs.update(model_params)
        except json.JSONDecodeError:
            pass
    return kwargs

def normalize_intent(intent: str) -> str:
    if not isinstance(intent, str):
        return ""
    clean = intent.strip()
    clean = re.sub(r"[.,;:!?]+$", "", clean)
    return clean.lower()


def _parse_context_json_object(s: str):
    """Парсинг JSON-объекта для context. Успех: (dict, None), ошибка: (None, сообщение)."""
    raw = (s or "").strip()
    if not raw:
        return {}, None
    try:
        o = json.loads(raw)
    except json.JSONDecodeError as e:
        return None, str(e)
    if not isinstance(o, dict):
        return None, "ожидается JSON-объект {...}, не массив и не примитив"
    return o, None


def _merge_export_context(variables: dict, global_json: str, dialog_json: str) -> dict:
    """
    Итоговый context в JSONL: сначала переменные (если есть в данных диалога), затем общий context,
    затем context диалога — при совпадении ключей побеждает значение из диалога.
    """
    base = dict(variables) if isinstance(variables, dict) else {}
    g, _ = _parse_context_json_object(global_json)
    if g is None:
        g = {}
    d, _ = _parse_context_json_object(dialog_json)
    if d is None:
        d = {}
    return {**base, **g, **d}


def clean_json_string(s):
    if not isinstance(s, str):
        return s
    return re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", s)

def safe_json_loads(s):
    if not s or not isinstance(s, str):
        return []
    try:
        cleaned = clean_json_string(s)
        return json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        return []


def coerce_scenario_steps(cell) -> list:
    """
    Приводит значение колонки scenario_steps (ClickHouse / pandas) к list[dict].
    При выгрузке по дате часто приходят NaN, один JSON-объект вместо массива, bytes и т.д.
    """
    if cell is None:
        return []
    try:
        if pd.isna(cell):
            return []
    except (ValueError, TypeError):
        pass
    if isinstance(cell, bytes):
        try:
            cell = cell.decode("utf-8")
        except Exception:
            return []
    if isinstance(cell, str):
        s = cell.strip()
        if not s:
            return []
        try:
            parsed = json.loads(clean_json_string(s))
        except (json.JSONDecodeError, TypeError):
            return []
    elif isinstance(cell, (list, tuple)):
        parsed = list(cell)
    elif isinstance(cell, dict):
        parsed = cell
    else:
        try:
            import numpy as np

            if isinstance(cell, np.ndarray):
                return coerce_scenario_steps(cell.tolist())
        except ImportError:
            pass
        return []

    if isinstance(parsed, dict):
        if "block_id" in parsed or "message_id" in parsed:
            return [parsed]
        for k in ("steps", "scenario_steps", "data"):
            nested = parsed.get(k)
            if isinstance(nested, list):
                return [x for x in nested if isinstance(x, dict)]
        return []
    if isinstance(parsed, list):
        return [x for x in parsed if isinstance(x, dict)]
    return []


def scenario_steps_column_name(df: pd.DataFrame) -> str | None:
    """Имя колонки со шагами сценария без учёта регистра."""
    if df is None or df.empty:
        return None
    lower = {str(c).lower(): c for c in df.columns}
    return lower.get("scenario_steps")


def variables_column_name(df: pd.DataFrame) -> str | None:
    """Имя колонки variables (лог изменений переменных) без учёта регистра."""
    if df is None or df.empty:
        return None
    lower = {str(c).lower(): c for c in df.columns}
    return lower.get("variables")


def coerce_variables(cell) -> list:
    """Как scenario_steps: JSON-массив записей с message_id, variable_name, value, …"""
    return coerce_scenario_steps(cell)


def _parse_variable_name_list(text: str) -> list[str]:
    if text is None or not str(text).strip():
        return []
    out: list[str] = []
    for line in str(text).strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        for part in line.split(","):
            p = part.strip()
            if p:
                out.append(p)
    seen: set[str] = set()
    uniq: list[str] = []
    for x in out:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq


def _http_tool_tracked_variable_names() -> frozenset[str]:
    a = _parse_variable_name_list(
        st.session_state.get("ch_http_tool_name_variable_names") or ""
    )
    b = _parse_variable_name_list(
        st.session_state.get("ch_http_tool_args_variable_names") or ""
    )
    return frozenset(a) | frozenset(b)


def _http_tool_variable_config_enabled() -> bool:
    return len(_http_tool_tracked_variable_names()) > 0


def _markup_hide_tool_calls_ui() -> bool:
    """Чекбокс «Не сохранять вызовы инструментов»: без тулов в хронологии и в экспортном JSONL."""
    return bool(st.session_state.get("markup_hide_tool_calls"))


def _http_variables_allowlist_for_ui() -> frozenset[str] | None:
    if not _http_tool_variable_config_enabled():
        return None
    return _http_tool_tracked_variable_names()


def _render_http_blocks_llm_settings_expander():
    """Общий блок «Обращение к LLM через HTTP-блоки» для всех источников кроме загрузки готового JSONL."""
    with st.expander("Обращение к LLM через HTTP-блоки", expanded=False):
        st.caption(
            "Укажите переменые в которые заполняются результаты вызовов инструментов."
        )
        st.text_area(
            "tool_name → переменная заполняемая названием вызванной функции (можно несколько: объединяются через « / »)",
            height=72,
            placeholder="tool_name1\ntool_name2",
            key="ch_http_tool_name_variable_names",
        )
        st.text_area(
            "tool_arguments → переменная заполняемая названием агрументов вызванной функции(несколько имён станут ключами JSON аргументов)",
            height=72,
            placeholder="tool_arguments1\ntool_arguments2",
            key="ch_http_tool_args_variable_names",
        )


def is_noise(text):
    if not isinstance(text, str):
        return True
    stripped = text.strip()
    if not stripped:
        return True
    lower_text = stripped.lower()
    noise_patterns = {
        "*начало диалога*",
        "*конец диалога*",
        "*удерживаем абонента*",
        "*удержание абонента*",
        "*длительный запрос*",
        "*Молчание абонента*",
        "*молчание абонента*",
        "",
    }
    return lower_text in noise_patterns


def block_filter_bounds_from_session():
    """
    Читает границы фильтра по блокам из session_state.
    Пустое поле number_input часто даёт 0 — для нас это «не задано»
    (конец диалога = реальный терминал по next_block_id, начало — из start_block_id диалога).
    """
    def _one(key: str):
        v = st.session_state.get(key)
        if v is None:
            return None
        try:
            i = int(v)
        except (TypeError, ValueError):
            return None
        if i <= 0:
            return None
        return i

    return _one("filter_start_block"), _one("filter_end_block")


def parse_clickhouse_row(messages_str: str, tech_logs_str: str):
    """NDA."""
    return []

import time

def suggest_intent_with_llm(
    history_to_send: list,
    prompt_template: str,
    allowed_intents: str,
    llm_delay: int = 0,
) -> str:
    history_text = ""
    for msg in history_to_send:
        role = "Пользователь" if msg["role"] == "user" else "Ассистент"
        history_text += f"{role}: {msg['content']}\n"
    try:
        full_prompt = prompt_template.format(
            allowed_intents=allowed_intents, dialog=history_text.strip()
        )
        if llm_delay > 0:
            time.sleep(llm_delay)
        response = litellm.completion(
            messages=[{"role": "user", "content": full_prompt}], **get_litellm_kwargs()
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[Ошибка LLM: {str(e)}]"

def clean_line(line: str) -> str:
    line = str(line).strip()
    if line.startswith("- "):
        line = line[2:].strip()
    return line


def _scenario_id_from_ch_row(ci: dict, tup: tuple) -> int:
    """Колонка start_block_id в выборке опциональна — иначе scenario_id в UI/экспорте = 0."""
    idx = ci.get("start_block_id")
    if idx is None:
        return 0
    raw = tup[idx]
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return 0
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 0


def process_clickhouse_data(df_raw, remove_duplicate_dialogs: bool = False):
    """Обрабатывает DataFrame из ClickHouse и возвращает (dialog_groups, turns_df)"""
    seen_keys = set()
    duplicate_count = 0
    dialog_groups = []
    ci = {c: i for i, c in enumerate(df_raw.columns)}
    im, itl = ci.get("messages"), ci.get("tech_logs")
    with st.spinner("🔄 Выполнение SQL-запроса и обработка..."):
        for tup in df_raw.itertuples(index=False, name=None):
            dialog_id = int(tup[ci["id"]])
            scenario_id = _scenario_id_from_ch_row(ci, tup)
            messages_str = tup[im] if im is not None else ""
            tech_logs_str = tup[itl] if itl is not None else ""
            if messages_str is None or (
                isinstance(messages_str, float) and pd.isna(messages_str)
            ):
                messages_str = ""
            if tech_logs_str is None or (
                isinstance(tech_logs_str, float) and pd.isna(tech_logs_str)
            ):
                tech_logs_str = ""
            combined = parse_clickhouse_row(messages_str, tech_logs_str)
            timeline = []
            for event in combined:
                if event["type"] == "message":
                    if is_noise(event["text"]):
                        continue
                    role_orig = event["role"].upper()
                    role = "assistant" if role_orig == "BOT" else "user"
                    timeline.append(
                        {"role": role, "content": event["text"], "type": "message"}
                    )
                elif event["type"] == "tool_call":
                    _entry = {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": event["function_name"],
                                    "arguments": json.dumps(
                                        event["arguments"], ensure_ascii=False
                                    ),
                                },
                                "id": f"call_{str(uuid.uuid4()).replace('-', '')[:24]}",
                                "type": "function",
                            }
                        ],
                        "type": "tool_call",
                    }
                    if "http_tool_variable_names" in event:
                        _entry["http_tool_variable_names"] = event[
                            "http_tool_variable_names"
                        ]
                    if "tool_event_source" in event:
                        _entry["tool_event_source"] = event["tool_event_source"]
                    timeline.append(_entry)
                elif event["type"] == "tool_message":
                    _tm = {
                        "role": "tool",
                        "content": event["content"],
                        "type": "tool_message",
                        "name": event.get("name", "tool"),
                        "tool_variable": event.get("tool_variable"),
                        "message_id": event.get("message_id"),
                    }
                    if "tool_event_source" in event:
                        _tm["tool_event_source"] = event["tool_event_source"]
                    timeline.append(_tm)
                elif event["type"] == "tool_schema":
                    _tsm = {
                        "type": "tool_schema",
                        "name": event.get("name", "tool"),
                        "tool_variable": event.get("tool_variable"),
                        "tool_schema": event.get("tool_schema"),
                        "message_id": event.get("message_id"),
                    }
                    if "tool_event_source" in event:
                        _tsm["tool_event_source"] = event["tool_event_source"]
                    timeline.append(_tsm)
            normalized_timeline = []
            for msg in timeline:
                if msg["type"] in ("tool_call", "tool_message", "tool_schema"):
                    normalized_timeline.append(msg)
                else:
                    if (
                        normalized_timeline
                        and normalized_timeline[-1].get("type") == "message"
                        and normalized_timeline[-1].get("role") == msg.get("role")
                    ):
                        normalized_timeline[-1]["content"] += " " + msg["content"]
                    else:
                        normalized_timeline.append(msg)
            if not normalized_timeline:
                continue
            text_only = [m for m in normalized_timeline if m["type"] == "message"]
            if not text_only:
                continue
            key = dialog_to_key(text_only)
            is_dup = key in seen_keys
            if remove_duplicate_dialogs:
                if is_dup:
                    duplicate_count += 1
                    continue
                seen_keys.add(key)
            else:
                if is_dup:
                    duplicate_count += 1
                else:
                    seen_keys.add(key)
            dialog_groups.append(
                {
                    "dialog_id": dialog_id,
                    "scenario_id": scenario_id,
                    "timeline": normalized_timeline,
                }
            )
    if duplicate_count > 0:
        if remove_duplicate_dialogs:
            st.info(f"ℹ️ Удалено **{duplicate_count}** дубликатов.")
        else:
            st.info(
                f"ℹ️ Обнаружено **{duplicate_count}** дубликатов по тексту диалога "
                "(все строки показаны). Включите «Удалять дубликаты по тексту диалога», чтобы скрыть повторы."
            )
    records = []
    for item in dialog_groups:
        for msg in item["timeline"]:
            if msg["type"] == "message":
                records.append(
                    {
                        "dialog_id": item["dialog_id"],
                        "role": msg["role"],
                        "content": msg["content"],
                    }
                )
    turns_df = (
        pd.DataFrame(records)
        if records
        else pd.DataFrame(columns=["dialog_id", "role", "content"])
    )
    return dialog_groups, turns_df

def classify_role(line: str) -> str | None:
    clean = clean_line(line)
    if not clean or (clean.startswith("*") and clean.endswith("*")):
        return None
    for ch in clean:
        if ch.isalpha():
            return "assistant" if ch.isupper() else "user"
    return None

def normalize_dialog(messages: list) -> list:
    if not messages:
        return []
    normalized = []
    current_role = messages[0]["role"]
    current_text = messages[0]["content"]
    for msg in messages[1:]:
        if msg["role"] == current_role:
            current_text += " " + msg["content"]
        else:
            normalized.append({"role": current_role, "content": current_text.strip()})
            current_role = msg["role"]
            current_text = msg["content"]
    normalized.append({"role": current_role, "content": current_text.strip()})
    return normalized

def dialog_to_key(messages: list) -> str:
    return " || ".join(f"{msg['role']}:{msg['content']}" for msg in messages)

def clear_dialog_annotation_widget_state():
    """Сбрасывает session_state виджетов формы разметки диалогов, чтобы после LLM подтянуть значения из annotations."""
    prefixes = (
        "form_goal_",
        "form_save_",
        "form_custom_hist_",
        "form_custom_len_",
        "form_trunc_dir_",
        "form_dialog_ctx_",
    )
    for k in list(st.session_state.keys()):
        ks = str(k)
        if any(ks.startswith(p) for p in prefixes):
            st.session_state.pop(k, None)
        if ks.startswith("_ann_goal_sync_") or ks.startswith("_ann_ctx_sync_"):
            st.session_state.pop(k, None)
    st.session_state.pop("dialog_form_temp", None)


def sync_dialog_form_temp(dialogs: list, annotations: dict):
    """Синхронизирует временные значения форм с основным состоянием (единожды)"""
    if "dialog_form_temp" not in st.session_state:
        st.session_state["dialog_form_temp"] = {}
    
    for item in dialogs:
        dialog_id = item["dialog_id"]
        full_timeline = item["timeline"]
        max_len = len(full_timeline) if full_timeline else 1
        
        temp_keys = {
            "goal": f"temp_goal_{dialog_id}",
            "save": f"temp_save_{dialog_id}",
            "custom_hist": f"temp_custom_hist_{dialog_id}",
            "custom_len": f"temp_custom_len_{dialog_id}",
            "trunc_dir": f"temp_trunc_dir_{dialog_id}",
            "expanded": f"expanded_{dialog_id}",
        }
        
        ann = annotations.get(dialog_id, {})
        temp = st.session_state["dialog_form_temp"]
        
        # Синхронизация с защитой от выхода за границы
        temp[temp_keys["goal"]] = (ann.get("intent_mode") or "").strip()
        temp[temp_keys["save"]] = ann.get("save", True)
        temp[temp_keys["custom_hist"]] = ann.get("custom_history_enabled", False)
        temp[temp_keys["custom_len"]] = max(1, min(ann.get("custom_history_length", 4), max_len))
        temp[temp_keys["trunc_dir"]] = ann.get("truncation_direction", "с начала")
        
        if temp_keys["expanded"] not in st.session_state:
            st.session_state[temp_keys["expanded"]] = False
            

# === Инициализация кэша просмотренных диалогов ===
if "seen_dialog_ids" not in st.session_state:
    st.session_state["seen_dialog_ids"] = set()
# --- UI ---
st.set_page_config(page_title="Разметка диалогов с LLM", layout="wide")
# === ГЛОБАЛЬНЫЙ ФЛАГ ПЕРЕЗАГРУЗКИ (проверяется ДО любых форм!) ===
if "rerun_after_llm" not in st.session_state:
    st.session_state["rerun_after_llm"] = False

if st.session_state.get("rerun_after_llm", False):
    st.session_state["rerun_after_llm"] = False
    # ✅ Очищаем кэш виджетов, чтобы избежать дублирования
    keys_to_clear = [k for k in st.session_state.keys() if k.startswith("form_")]
    for key in keys_to_clear:
        st.session_state.pop(key, None)
    st.rerun()
st.title("📝 Разметка диалогов: ручная + LLM")
# === Настройки LLM (в боковой панели) ===
st.sidebar.subheader("🔑 Настройки LLM (для разметки)")
if "litellm_api_key" not in st.session_state:
    st.session_state["litellm_api_key"] = os.getenv("LITELLM_API_KEY", "")
if "selected_model" not in st.session_state:
    st.session_state["selected_model"] = SUPPORTED_MODELS[0]

litellm_api_key = st.sidebar.text_input(
    "LiteLLM API Key",
    type="password",
    value=st.session_state["litellm_api_key"],
    key="litellm_api_key_input",
)
st.session_state["litellm_api_key"] = litellm_api_key

selected_model = st.sidebar.selectbox(
    "Модель",
    options=SUPPORTED_MODELS,
    index=SUPPORTED_MODELS.index(st.session_state["selected_model"]),
    key="selected_model_input",
)
st.session_state["selected_model"] = selected_model

st.sidebar.subheader("⚙️ Параметры LLM (JSON)")
if "llm_params_json" not in st.session_state:
    st.session_state["llm_params_json"] = (
        '{"temperature": 0.7, "max_tokens": 150, "top_p": 1.0}'
    )
llm_params_json = st.sidebar.text_area(
    "Параметры в формате JSON",
    value=st.session_state["llm_params_json"],
    height=100,
    key="llm_params_json_input",
)
st.session_state["llm_params_json"] = llm_params_json


# === Очистка состояния при смене источника ===
source_option = st.radio(
    "Выберите источник:",
    options=[
        "XLSX-файл",
        "ClickHouse: по ID диалогов и сценария",
        "ClickHouse: по сценарии и дате",
        "ClickHouse: по SQL-запросу",
        "Загрузить готовый JSONL",
        "Сохранённая выборка (БД)",
    ],
    index=0,
    horizontal=True,
    key="source_option",
)

if (
    "last_source" not in st.session_state
    or st.session_state["last_source"] != source_option
):
    for key in [
        "df_raw",
        "transcript_col",
        "source_loaded",
        "_xlsx_load_key",
        "common_scenario_id",
        "annotations",
        "dialog_groups",
        "turns_df",
        "seen_dialog_ids",  # ← Добавили сброс кэша диалогов
    ]:
        st.session_state.pop(key, None)
    st.session_state["last_source"] = source_option
    st.rerun()  # ← Перезагрузка для применения сброса

# === Выбор источника данных ===
st.subheader("📥 Источник данных")
if source_option not in ("Загрузить готовый JSONL", "Сохранённая выборка (БД)"):
    st.markdown("##### Фильтр по диапазону блоков сценария (опционально)")
    _fcol1, _fcol2 = st.columns(2)
    with _fcol1:
        st.number_input(
            "Начальный block_id",
            min_value=0,
            value=None,
            step=1,
            key="filter_start_block",
            help="Пусто или 0 = с корня сценария диалога (start_block_id из БД). Нужен scenario_steps.",
        )
    with _fcol2:
        st.number_input(
            "Конечный block_id",
            min_value=0,
            value=None,
            step=1,
            key="filter_end_block",
            help="Пусто или 0 = до конца диалога по next_block_id (последний блок сценария).",
        )
    _fs, _fe = block_filter_bounds_from_session()
    if _fs is not None or _fe is not None:
        _s = _fs if _fs is not None else "корень сценария (start_block_id)"
        _e = _fe if _fe is not None else "конец диалога"
        st.info(f"🔍 Активен фильтр блоков сценария: **{_s} → {_e}**")

    st.checkbox(
        "Не сохранять вызовы инструментов",
        key="markup_hide_tool_calls",
        help=(
            "Если включено: в хронологии не отображаются вызовы инструментов (ни из логов, ни по HTTP variables); "
            "при скачивании JSONL они не попадают в поле history."
        ),
    )
    st.checkbox(
        "Удалять дубликаты по тексту диалога",
        value=False,
        key="ch_remove_duplicate_dialogs",
        help=(
            "Если включено: при одинаковой последовательности реплик пользователя и ассистента "
            "(после нормализации) в выборке остаётся только первый диалог. "
            "По умолчанию все строки из ClickHouse попадают в разметку, даже при совпадении текста."
        ),
    )

df_raw = None
transcript_col = None
dialog_groups = []
turns_df = pd.DataFrame(columns=["dialog_id", "role", "content"])

def build_block_chain_from_scenario(
    scenario_steps: list,
    start_block_id=None,
    end_block_id=None,
    default_start_block_id=None,
):
    """
    Участок фактического прохождения сценария в данных диалога: порядок шагов как в
    `ordered_block_ids_from_scenario_steps` (поле id в scenario_steps).

    От **первого** вхождения начального block_id в trace до **первого** вхождения
    конечного включительно. Если начало не задано — от шага с default_start_block_id
    из БД (или первого block_id >= него в порядке trace). Если конец не задан — до
    последнего шага trace.
    """
    trace = ordered_block_ids_from_scenario_steps(scenario_steps)
    if not trace:
        return set(), []

    start_block_id = (
        None if start_block_id is None else _normalize_block_id(start_block_id)
    )
    end_block_id = None if end_block_id is None else _normalize_block_id(end_block_id)

    if start_block_id is not None:
        i0 = None
        for k, b in enumerate(trace):
            if b == start_block_id:
                i0 = k
                break
        if i0 is None:
            return set(), []
    else:
        if default_start_block_id is not None:
            ds = _normalize_block_id(default_start_block_id)
            if ds is not None:
                i0 = None
                for k, b in enumerate(trace):
                    if b == ds:
                        i0 = k
                        break
                if i0 is None:
                    for k, b in enumerate(trace):
                        try:
                            if b >= ds:
                                i0 = k
                                break
                        except TypeError:
                            continue
                if i0 is None:
                    return set(), []
            else:
                i0 = 0
        else:
            i0 = 0

    if end_block_id is not None:
        i1 = None
        for k in range(i0, len(trace)):
            if trace[k] == end_block_id:
                i1 = k
                break
        if i1 is None:
            return set(), []
    else:
        i1 = len(trace) - 1

    segment = trace[i0 : i1 + 1]
    return set(segment), segment

def _normalize_block_id(bid):
    if bid is None:
        return None
    try:
        return int(bid)
    except (TypeError, ValueError):
        return bid


def _block_ids_from_msg_blocks(msg: dict):
    """Все block_id из msg.blocks — у реплики human часто несколько блоков (http, condition, …)."""
    if not isinstance(msg, dict):
        return frozenset()
    bl = msg.get("blocks")
    if not isinstance(bl, list):
        return frozenset()
    out = set()
    for b in bl:
        if isinstance(b, dict):
            x = _normalize_block_id(b.get("block_id"))
            if x is not None:
                out.add(x)
    return frozenset(out)


def _event_block_ids_for_chain(e: dict):
    """Идентификаторы блока для проверки попадания в цепочку фильтра."""
    primary = _normalize_block_id(e.get("block_id"))
    extra = e.get("block_ids_from_msg")
    if not extra:
        extra = frozenset()
    s = set(extra)
    if primary is not None:
        s.add(primary)
    sub = _normalize_block_id(e.get("subscenario_start_block_id"))
    if sub is not None:
        s.add(sub)
    return s


def _event_intersects_chain(e: dict, norm_chain: set) -> bool:
    return bool(_event_block_ids_for_chain(e) & norm_chain)


def _event_has_start_block(e: dict, want_start) -> bool:
    if want_start is None:
        return False
    return want_start in _event_block_ids_for_chain(e)


def _block_ids_from_scenario_steps(scenario_steps) -> set:
    """Все block_id из шагов сценария — фактический след прохождения диалога по блокам."""
    out = set()
    for step in scenario_steps or []:
        if not isinstance(step, dict):
            continue
        bid = _normalize_block_id(step.get("block_id"))
        if bid is not None:
            out.add(bid)
    return out


def _anchor_ts_slice_from_scenario_steps(scenario_steps, norm_chain: set, want_start):
    """
    Первый timestamp шага scenario_steps для среза хронологии, если якорь по сообщениям не найден:
    при заданном начале фильтра — первый шаг с этим block_id; иначе — первый с block_id ∈ norm_chain.
    """
    want = _normalize_block_id(want_start) if want_start is not None else None
    candidates = []
    for step in scenario_steps or []:
        if not isinstance(step, dict):
            continue
        bid = _normalize_block_id(step.get("block_id"))
        if bid is None:
            continue
        ts = _timestamp_from_created(step.get("created_at"))
        if ts is None:
            continue
        if want is not None:
            if bid == want:
                candidates.append(ts)
        elif bid in norm_chain:
            candidates.append(ts)
    return min(candidates) if candidates else None


def _timestamp_from_created(created_at):
    if not created_at:
        return None
    try:
        return pd.to_datetime(created_at).timestamp()
    except (ValueError, TypeError):
        return None


def _ts_from_created_at_str(created_at) -> float:
    if not created_at:
        return 0.0
    try:
        return float(pd.to_datetime(created_at).timestamp())
    except (ValueError, TypeError):
        return 0.0


def _max_message_id_from_timeline(timeline: list) -> int | None:
    """Максимальный message_id среди реплик/тулов в срезе хронологии."""
    best = None
    for msg in timeline or []:
        mid = msg.get("message_id")
        if mid is None:
            continue
        try:
            v = int(mid)
        except (TypeError, ValueError):
            continue
        if best is None or v > best:
            best = v
    return best


def snapshot_variables_latest_by_name(
    variables_raw: list, max_message_id: int | None = None
) -> dict:
    """
    Снимок переменных к концу диалога (или к max_message_id): для каждого variable_name —
    последняя запись по (message_id, created_at).
    """
    rows = []
    for r in variables_raw or []:
        if not isinstance(r, dict):
            continue
        name = r.get("variable_name")
        if not name:
            continue
        mid = r.get("message_id")
        if max_message_id is not None and mid is not None:
            try:
                if int(mid) > int(max_message_id):
                    continue
            except (TypeError, ValueError):
                continue
        rows.append(r)

    def _mid_sort_key(r):
        try:
            return int(r.get("message_id"))
        except (TypeError, ValueError):
            return -1

    rows.sort(
        key=lambda r: (_mid_sort_key(r), _ts_from_created_at_str(r.get("created_at")))
    )
    out = {}
    for r in rows:
        out[str(r["variable_name"])] = r.get("value")
    return out


def ordered_block_ids_from_scenario_steps(scenario_steps: list) -> list:
    """
    Все block_id из scenario_steps в порядке шага сценария: по полю id шага,
    иначе по индексу в массиве (как в JSON из ClickHouse). Одна строка — одно
    звено цепочки; подряд одинаковые block_id не схлопываются.
    """
    rows = []
    for i, step in enumerate(scenario_steps or []):
        if not isinstance(step, dict):
            continue
        bid = _normalize_block_id(step.get("block_id"))
        if bid is None:
            continue
        raw_id = step.get("id")
        try:
            sid = int(raw_id) if raw_id is not None else None
        except (TypeError, ValueError):
            sid = None
        rows.append((sid if sid is not None else i, i, bid))
    rows.sort(key=lambda r: (r[0], r[1]))
    return [r[2] for r in rows]


def _log_entry_message_id(entry: dict):
    mid = entry.get("message_id")
    if mid is None and isinstance(entry.get("data"), dict):
        mid = entry["data"].get("message_id")
    return mid


def _log_data_block_id(data):
    if isinstance(data, dict):
        return _normalize_block_id(data.get("block_id"))
    if isinstance(data, str):
        try:
            data_dict = json.loads(data)
            if isinstance(data_dict, dict):
                return _normalize_block_id(data_dict.get("block_id"))
        except json.JSONDecodeError:
            pass
    return None


def _collect_block_timeline_by_message_id(logs: list, scenario_steps: list | None):
    """(message_id, время) → список (timestamp, block_id) из tech_logs и scenario_steps."""
    from collections import defaultdict

    by_mid = defaultdict(list)
    for entry in logs or []:
        if not isinstance(entry, dict):
            continue
        mid = _log_entry_message_id(entry)
        bid = _log_data_block_id(entry.get("data"))
        if mid is None or bid is None:
            continue
        ts = _timestamp_from_created(entry.get("created_at"))
        if ts is None:
            continue
        by_mid[mid].append((ts, bid))
    for step in scenario_steps or []:
        if not isinstance(step, dict):
            continue
        mid = step.get("message_id")
        bid = _normalize_block_id(step.get("block_id"))
        if mid is None or bid is None:
            continue
        ts = _timestamp_from_created(step.get("created_at"))
        if ts is None:
            continue
        by_mid[mid].append((ts, bid))
    return by_mid


def _latest_block_id_for_message(by_mid: dict, msg_id):
    pairs = by_mid.get(msg_id, [])
    if not pairs:
        return None
    pairs = sorted(pairs, key=lambda x: x[0])
    return pairs[-1][1]


def _block_id_closest_to_anchor_time(by_mid: dict, msg_id, anchor_created_at: str):
    """
    Блок из tech_logs/scenario_steps для данного message_id, чей created_at
    ближе всего к опорному моменту (время самого сообщения или записи в логе).

    """
    pairs = by_mid.get(msg_id, [])
    if not pairs:
        return None
    anchor_ts = _timestamp_from_created(anchor_created_at)
    if anchor_ts is None:
        return _latest_block_id_for_message(by_mid, msg_id)
    _, bid = min(pairs, key=lambda p: (abs(p[0] - anchor_ts), p[0]))
    return bid


_TEXT_BLOCK_TYPES_FOR_BOT = frozenset({"text_blocks", "redirect_call_blocks"})


def _debug_flatten_by_mid(by_mid: dict) -> pd.DataFrame:
    rows = []
    for mid in sorted(by_mid.keys(), key=lambda x: (x is None, x)):
        for ts, bid in sorted(by_mid[mid], key=lambda x: x[0]):
            rows.append({"message_id": mid, "block_id": bid, "ts_unix": round(ts, 4)})
    if not rows:
        return pd.DataFrame(columns=["message_id", "block_id", "ts_unix"])
    return pd.DataFrame(rows)


def _debug_closest_breakdown(by_mid: dict, msg_id, anchor_created_at: str) -> pd.DataFrame:
    pairs = by_mid.get(msg_id, [])
    if not pairs:
        return pd.DataFrame(columns=["block_id", "step_ts", "delta_sec"])
    anchor_ts = _timestamp_from_created(anchor_created_at)
    rows = []
    for ts, bid in pairs:
        delta = None if anchor_ts is None else round(abs(ts - anchor_ts), 4)
        rows.append({"block_id": bid, "step_ts": round(ts, 4), "delta_sec": delta})
    df = pd.DataFrame(rows)
    if anchor_ts is not None and not df.empty:
        df = df.sort_values(["delta_sec", "step_ts"], na_position="last")
    return df


def _debug_bot_timing_breakdown(
    created_at: str, scenario_steps: list | None, window_sec: float = 5.0
) -> pd.DataFrame:
    cols = ["block_id", "block_type", "step_ts", "delta_sec", "match_lt_1s", "in_window"]
    if not scenario_steps or not created_at:
        return pd.DataFrame(columns=cols)
    msg_ts = _timestamp_from_created(created_at)
    if msg_ts is None:
        return pd.DataFrame(columns=cols)
    rows = []
    for step in scenario_steps:
        if not isinstance(step, dict):
            continue
        if step.get("block_type") not in _TEXT_BLOCK_TYPES_FOR_BOT:
            continue
        st_ts = _timestamp_from_created(step.get("created_at"))
        if st_ts is None:
            continue
        diff = abs(st_ts - msg_ts)
        rows.append(
            {
                "block_id": step.get("block_id"),
                "block_type": step.get("block_type"),
                "step_ts": round(st_ts, 4),
                "delta_sec": round(diff, 4),
                "match_lt_1s": diff < 1.0,
                "in_window": diff <= window_sec,
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("delta_sec")


def _bot_block_from_scenario_timing(created_at: str, scenario_steps: list | None):
    """Как в test.build_trace: ответ бота сопоставляется с text_blocks / redirect по времени."""
    if not scenario_steps or not created_at:
        return None
    msg_ts = _timestamp_from_created(created_at)
    if msg_ts is None:
        return None
    best_bid = None
    best_diff = float("inf")
    for step in scenario_steps:
        if not isinstance(step, dict):
            continue
        if step.get("block_type") not in _TEXT_BLOCK_TYPES_FOR_BOT:
            continue
        st_ts = _timestamp_from_created(step.get("created_at"))
        if st_ts is None:
            continue
        diff = abs(st_ts - msg_ts)
        if diff < 1.0 and diff < best_diff:
            best_diff = diff
            best_bid = _normalize_block_id(step.get("block_id"))
    return best_bid


def _bot_block_nearest_in_chain(
    created_at: str,
    scenario_steps: list | None,
    norm_chain: set,
    max_sec: float = 5.0,
):
    """
    При активном фильтре по цепочке: block_id бота — ближайший по времени шаг из
    scenario_steps, чей block_id входит в norm_chain (в т.ч. llm_dialog_blocks).
    """
    if not scenario_steps or not created_at or not norm_chain:
        return None
    msg_ts = _timestamp_from_created(created_at)
    if msg_ts is None:
        return None
    best_bid = None
    best_key = None
    for step in scenario_steps:
        if not isinstance(step, dict):
            continue
        bid = _normalize_block_id(step.get("block_id"))
        if bid is None or bid not in norm_chain:
            continue
        st_ts = _timestamp_from_created(step.get("created_at"))
        if st_ts is None:
            continue
        diff = abs(st_ts - msg_ts)
        if diff > max_sec:
            continue
        raw_sid = step.get("id")
        try:
            sid = int(raw_sid) if raw_sid is not None else 10**9
        except (TypeError, ValueError):
            sid = 10**9
        key = (diff, sid)
        if best_key is None or key < best_key:
            best_key = key
            best_bid = bid
    return best_bid


def _block_id_from_msg_blocks_field(msg: dict):
    bl = msg.get("blocks")
    if not isinstance(bl, list):
        return None
    for b in reversed(bl):
        if isinstance(b, dict):
            bid = _normalize_block_id(b.get("block_id"))
            if bid is not None:
                return bid
    return None


def _append_tool_calls_from_logs(logs: list, combined: list, by_mid: dict):
    """NDA: разбор tech_logs / variables платформы скрыт."""
    return None

def _append_tool_return_events_from_logs(logs: list, combined: list, by_mid: dict):
    """NDA: разбор tech_logs / variables платформы скрыт."""
    return None

def _variable_row_message_id(entity: dict) -> int | None:
    if not isinstance(entity, dict):
        return None
    v = entity.get("message_id")
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _variable_row_subscenario_id(entity: dict):
    if not isinstance(entity, dict):
        return None
    return _normalize_block_id(entity.get("subscenario_start_block_id"))


def _is_blank_http_tool_value(v) -> bool:
    """Возвращает ``True``, если значение переменной не нужно показывать в синтетическом tool_call."""
    if v is None:
        return True
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return True
        if s in ("[]", "[ ]", "{}", "{ }", "null", "None"):
            return True
        return False
    if isinstance(v, (list, dict, set)):
        return len(v) == 0
    return False


def _decode_http_tool_variable_value(v):
    """
    Если в ClickHouse пришла JSON-строка или текст с \\uXXXX — распарсить/раскодировать
    (чтобы в tool_call были нормальный Unicode и объекты, а не escape-последовательности).
    """
    if not isinstance(v, str):
        return v
    t = v.strip()
    if not t:
        return v
    if (t.startswith("{") and t.endswith("}")) or (t.startswith("[") and t.endswith("]")):
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            pass
    if "\\u" in v or "\\U" in v or "\\x" in v:
        try:
            return v.encode("utf-8", "surrogatepass").decode("unicode_escape")
        except (UnicodeDecodeError, UnicodeError):
            pass
    return v


def _http_tool_name_fragment_for_display(v) -> str:
    """Фрагмент имени функции (строка для OpenAI name)."""
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False)
    return str(v).strip() if v is not None else ""


def _append_http_blocks_tool_calls_from_variables(
    combined: list,
    scenario_steps: list | None,
    variables_raw: list | None,
    tool_name_variable_names: list[str],
    tool_args_variable_names: list[str],
) -> None:
    if not scenario_steps or not variables_raw:
        return
    name_vars = [
        str(x).strip()
        for x in (tool_name_variable_names or [])
        if x and str(x).strip()
    ]
    args_vars = [
        str(x).strip()
        for x in (tool_args_variable_names or [])
        if x and str(x).strip()
    ]
    tracked = set(name_vars) | set(args_vars)
    if not tracked:
        return

    running: dict = {}

    def _apply_rows_to_running(rows: list) -> None:
        for r in rows:
            if not isinstance(r, dict):
    """NDA: разбор tech_logs / variables платформы скрыт."""
    return None

def parse_clickhouse_row_scenario_id(
    messages_str: str,
    tech_logs_str: str,
    scenario_steps=None,
    chain_block_ids=None,
    filter_start_block=None,
    filter_end_block=None,
    debug=False,
    debug_bot_window: float = 5.0,
    variables_raw: list | None = None,
    http_tool_name_variable_names: list[str] | None = None,
    http_tool_args_variable_names: list[str] | None = None,
):
    """NDA: разбор messages/tech_logs/scenario_steps/variables из ClickHouse скрыт."""
    return []

def process_clickhouse_data_scenario_id(
    df_raw,
    filter_start_block=None,
    filter_end_block=None,
    remove_duplicate_dialogs: bool = False,
):
    """
    Обрабатывает DataFrame из ClickHouse.
    Корректно парсит scenario_steps даже если он приходит как JSON-строка.
    """
    seen_keys = set()
    duplicate_count = 0
    dialog_groups = []
    debug_map = bool(st.session_state.get("debug_block_id_mapping"))
    dbg_target = int(st.session_state.get("debug_block_id_which_dialog") or 0)
    dbg_window = float(st.session_state.get("debug_block_id_bot_window") or 5.0)
    http_name_vars = _parse_variable_name_list(
        st.session_state.get("ch_http_tool_name_variable_names") or ""
    )
    http_args_vars = _parse_variable_name_list(
        st.session_state.get("ch_http_tool_args_variable_names") or ""
    )
    _ssc = scenario_steps_column_name(df_raw)
    _vsc = variables_column_name(df_raw)

    ci = {c: i for i, c in enumerate(df_raw.columns)}
    im, itl = ci.get("messages"), ci.get("tech_logs")
    steps_i = ci.get(_ssc) if _ssc is not None else ci.get("scenario_steps")
    var_i = ci.get(_vsc) if _vsc is not None else ci.get("variables")
    if _http_tool_variable_config_enabled() and var_i is None:
        st.warning(
            "Включён режим **HTTP-блоки → инструмент**, но в выборке нет колонки **variables**. "
            "Добавьте её в SELECT ClickHouse."
        )
    with st.spinner("🔄 Выполнение SQL-запроса и обработка..."):
        for row_i, tup in enumerate(df_raw.itertuples(index=False, name=None)):
            dialog_id = int(tup[ci["id"]])
            scenario_id = _scenario_id_from_ch_row(ci, tup)
            _raw_steps = tup[steps_i] if steps_i is not None else None
            scenario_steps = coerce_scenario_steps(_raw_steps)
            messages_str = tup[im] if im is not None else ""
            tech_logs_str = tup[itl] if itl is not None else ""
            _raw_vars = tup[var_i] if var_i is not None else None
            variables_raw = coerce_variables(_raw_vars)
            if messages_str is None or (isinstance(messages_str, float) and pd.isna(messages_str)):
                messages_str = ""
            if tech_logs_str is None or (isinstance(tech_logs_str, float) and pd.isna(tech_logs_str)):
                tech_logs_str = ""

            show_dbg = debug_map and (
                (dbg_target == 0 and row_i == 0) or dialog_id == dbg_target
            )
            
            # Если задан фильтр по блокам — строим цепочку
            chain_block_ids = None
            if filter_start_block is not None or filter_end_block is not None:
                chain_block_ids, _ = build_block_chain_from_scenario(
                    scenario_steps,
                    start_block_id=filter_start_block,
                    end_block_id=filter_end_block,
                    default_start_block_id=scenario_id,
                )
                if not chain_block_ids:
                    if show_dbg:
                        with st.expander(
                            f"🐛 block_id — dialog_id={dialog_id} (пустая цепочка, диалог пропущен)",
                            expanded=True,
                        ):
                            st.write("**Фильтр блоков:**", filter_start_block, "→", filter_end_block)
                            st.write("**start_block_id строки:**", scenario_id)
                            st.write("**Число шагов scenario_steps:**", len(scenario_steps))
                    continue

            if show_dbg:
                with st.expander(
                    f"🐛 Отладка block_id — dialog_id={dialog_id}",
                    expanded=True,
                ):
                    st.write("**start_block_id (колонка БД):**", scenario_id)
                    st.write("**chain_block_ids (множество фильтра):**", chain_block_ids)
                    st.write("**Шагов scenario_steps:**", len(scenario_steps))
                    combined = parse_clickhouse_row_scenario_id(
                        messages_str,
                        tech_logs_str,
                        scenario_steps=scenario_steps,
                        chain_block_ids=chain_block_ids,
                        filter_start_block=filter_start_block,
                        filter_end_block=filter_end_block,
                        debug=True,
                        debug_bot_window=dbg_window,
                        variables_raw=variables_raw,
                        http_tool_name_variable_names=http_name_vars,
                        http_tool_args_variable_names=http_args_vars,
                    )
            else:
                combined = parse_clickhouse_row_scenario_id(
                    messages_str,
                    tech_logs_str,
                    scenario_steps=scenario_steps,
                    chain_block_ids=chain_block_ids,
                    filter_start_block=filter_start_block,
                    filter_end_block=filter_end_block,
                    debug=False,
                    variables_raw=variables_raw,
                    http_tool_name_variable_names=http_name_vars,
                    http_tool_args_variable_names=http_args_vars,
                )
            
            if not combined:
                continue
            
            timeline = []
            for event in combined:
                if event["type"] == "message":
                    if is_noise(event["text"]):
                        continue
                    role_orig = event["role"].upper()
                    role = "assistant" if role_orig == "BOT" else "user"
                    
                    timeline.append(
                        {
                            "role": role, 
                            "content": event["text"], 
                            "type": "message",
                            "block_id": event.get("block_id")
                        }
                    )
                    
                elif event["type"] == "tool_call":
                    _tcall = {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": event["function_name"],
                                    "arguments": json.dumps(
                                        event["arguments"], ensure_ascii=False
                                    ),
                                },
                                "id": f"call_{str(uuid.uuid4()).replace('-', '')[:24]}",
                                "type": "function",
                                "block_id": event.get("block_id"),
                            }
                        ],
                        "type": "tool_call",
                        "block_id": event.get("block_id"),
                    }
                    if "http_tool_variable_names" in event:
                        _tcall["http_tool_variable_names"] = event[
                            "http_tool_variable_names"
                        ]
                    if "tool_event_source" in event:
                        _tcall["tool_event_source"] = event["tool_event_source"]
                    timeline.append(_tcall)
                elif event["type"] == "tool_message":
                    _tm = {
                            "role": "tool",
                            "content": event["content"],
                            "type": "tool_message",
                            "name": event.get("name", "tool"),
                            "tool_variable": event.get("tool_variable"),
                            "block_id": event.get("block_id"),
                            "message_id": event.get("message_id"),
                    }
                    if "tool_event_source" in event:
                        _tm["tool_event_source"] = event["tool_event_source"]
                    timeline.append(_tm)
                elif event["type"] == "tool_schema":
                    _tsm = {
                            "type": "tool_schema",
                            "name": event.get("name", "tool"),
                            "tool_variable": event.get("tool_variable"),
                            "tool_schema": event.get("tool_schema"),
                            "block_id": event.get("block_id"),
                            "message_id": event.get("message_id"),
                    }
                    if "tool_event_source" in event:
                        _tsm["tool_event_source"] = event["tool_event_source"]
                    timeline.append(_tsm)
            
            # Нормализация
            normalized_timeline = []
            for msg in timeline:
                if msg["type"] in ("tool_call", "tool_message", "tool_schema"):
                    normalized_timeline.append(msg)
                else:
                    if (
                        normalized_timeline
                        and normalized_timeline[-1].get("type") == "message"
                        and normalized_timeline[-1].get("role") == msg.get("role")
                    ):
                        normalized_timeline[-1]["content"] += " " + msg["content"]
                        if not normalized_timeline[-1].get("block_id"):
                            normalized_timeline[-1]["block_id"] = msg.get("block_id")
                    else:
                        normalized_timeline.append(msg)
            
            if not normalized_timeline:
                continue
            
            text_only = [m for m in normalized_timeline if m["type"] == "message"]
            if not text_only:
                continue
            
            key = dialog_to_key(text_only)
            is_dup = key in seen_keys
            if remove_duplicate_dialogs:
                if is_dup:
                    duplicate_count += 1
                    continue
                seen_keys.add(key)
            else:
                if is_dup:
                    duplicate_count += 1
                else:
                    seen_keys.add(key)

            dialog_groups.append(
                {
                    "dialog_id": dialog_id,
                    "scenario_id": scenario_id,
                    "timeline": normalized_timeline,
                    "variables_raw": variables_raw,
                    "scenario_steps": scenario_steps,
                    "block_chain_order": ordered_block_ids_from_scenario_steps(
                        scenario_steps
                    )
                    or None,
                }
            )

    if duplicate_count > 0:
        if remove_duplicate_dialogs:
            st.info(f"ℹ️ Удалено **{duplicate_count}** дубликатов.")
        else:
            st.info(
                f"ℹ️ Обнаружено **{duplicate_count}** дубликатов по тексту диалога "
                "(все строки показаны). Включите «Удалять дубликаты по тексту диалога», чтобы скрыть повторы."
            )
    
    if not dialog_groups:
        st.warning("⚠️ После фильтрации по блокам не осталось диалогов. Проверьте диапазон block_id.")
    
    # Формируем turns_df
    records = []
    for item in dialog_groups:
        for msg in item["timeline"]:
            if msg["type"] == "message":
                records.append(
                    {
                        "dialog_id": item["dialog_id"],
                        "role": msg["role"],
                        "content": msg["content"],
                        "block_id": msg.get("block_id"),
                    }
                )
    
    turns_df = (
        pd.DataFrame(records)
        if records
        else pd.DataFrame(columns=["dialog_id", "role", "content", "block_id"])
    )
    
    return dialog_groups, turns_df


def run_clickhouse_processing_with_optional_block_filter(df_raw):
    """
    Без фильтра по блокам — как process_clickhouse_data (полный разбор).
    С фильтром или режимом http_blocks→tool — process_clickhouse_data_scenario_id (нужен scenario_steps).
    """
    remove_dup = bool(st.session_state.get("ch_remove_duplicate_dialogs", False))
    fs, fe = block_filter_bounds_from_session()
    need_scenario = (fs is not None or fe is not None) or _http_tool_variable_config_enabled()
    if need_scenario:
        ssc = scenario_steps_column_name(df_raw)
        if ssc is None:
            st.error(
                "Нужен столбец **scenario_steps**: при фильтре по блокам или при настройке "
                "«HTTP-блоки → вызов инструмента» добавьте `scenario_steps` в SELECT."
            )
            st.stop()
        if ssc != "scenario_steps":
            df_raw = df_raw.rename(columns={ssc: "scenario_steps"})
        return process_clickhouse_data_scenario_id(
            df_raw,
            filter_start_block=fs,
            filter_end_block=fe,
            remove_duplicate_dialogs=remove_dup,
        )
    return process_clickhouse_data(df_raw, remove_duplicate_dialogs=remove_dup)


def invalidate_dialog_groups_if_block_filter_changed():
    """Сбрасывает готовые группы диалогов, если пользователь изменил фильтр по блокам."""
    fk = (
        block_filter_bounds_from_session(),
        bool(st.session_state.get("debug_block_id_mapping")),
        int(st.session_state.get("debug_block_id_which_dialog") or 0),
        float(st.session_state.get("debug_block_id_bot_window") or 0),
        bool(st.session_state.get("ch_remove_duplicate_dialogs", False)),
        tuple(
            _parse_variable_name_list(
                st.session_state.get("ch_http_tool_name_variable_names") or ""
            )
        ),
        tuple(
            _parse_variable_name_list(
                st.session_state.get("ch_http_tool_args_variable_names") or ""
            )
        ),
    )
    prev = st.session_state.get("_block_filter_cache_key")
    if prev is not None and prev != fk:
        st.session_state.pop("dialog_groups", None)
        st.session_state.pop("turns_df", None)
    st.session_state["_block_filter_cache_key"] = fk

    # """Обрабатывает DataFrame из ClickHouse и возвращает (dialog_groups, turns_df)"""
    # seen_keys = set()
    # duplicate_count = 0
    # dialog_groups = []
    
    # with st.spinner("🔄 Выполнение SQL-запроса и обработка..."):
    #     for _, row in df_raw.iterrows():
    #         dialog_id = int(row["id"])
    #         scenario_id = int(row["start_block_id"])
    #         messages_str = row.get("messages", "")
    #         tech_logs_str = row.get("tech_logs", "")
            
    #         combined = parse_clickhouse_row_scenario_id(messages_str, tech_logs_str)
            
    #         timeline = []
    #         for event in combined:
    #             if event["type"] == "message":
    #                 if is_noise(event["text"]):
    #                     continue
    #                 role_orig = event["role"].upper()
    #                 role = "assistant" if role_orig == "BOT" else "user"
                    
    #                 # <-- ДОБАВЛЕНО: block_id в сообщение
    #                 timeline.append(
    #                     {
    #                         "role": role, 
    #                         "content": event["text"], 
    #                         "type": "message",
    #                         "block_id": event.get("block_id")  # Новый ключ
    #                     }
    #                 )
                    
    #             elif event["type"] == "tool_call":
    #                 # <-- ДОБАВЛЕНО: block_id в tool_call
    #                 timeline.append(
    #                     {
    #                         "role": "assistant",
    #                         "content": "",
    #                         "tool_calls": [
    #                             {
    #                                 "function": {
    #                                     "name": event["function_name"],
    #                                     "arguments": json.dumps(
    #                                         event["arguments"], ensure_ascii=False
    #                                     ),
    #                                 },
    #                                 "id": f"call_{str(uuid.uuid4()).replace('-', '')[:24]}",
    #                                 "type": "function",
    #                                 "block_id": event.get("block_id")  # Новый ключ
    #                             }
    #                         ],
    #                         "type": "tool_call",
    #                         "block_id": event.get("block_id")  # Дублируем для удобства
    #                     }
    #                 )
            
    #         # Нормализация: объединяем последовательные сообщения одной роли
    #         normalized_timeline = []
    #         for msg in timeline:
    #             if msg["type"] == "tool_call":
    #                 normalized_timeline.append(msg)
    #             else:
    #                 if (
    #                     normalized_timeline
    #                     and normalized_timeline[-1]["role"] == msg["role"]
    #                     and normalized_timeline[-1].get("type") == "message"
    #                 ):
    #                     normalized_timeline[-1]["content"] += " " + msg["content"]
    #                     # Если у предыдущего не было block_id, берём из текущего
    #                     if not normalized_timeline[-1].get("block_id"):
    #                         normalized_timeline[-1]["block_id"] = msg.get("block_id")
    #                 else:
    #                     normalized_timeline.append(msg)
            
    #         if not normalized_timeline:
    #             continue
            
    #         text_only = [m for m in normalized_timeline if m["type"] == "message"]
    #         if not text_only:
    #             continue
            
    #         key = dialog_to_key(text_only)
    #         if key in seen_keys:
    #             duplicate_count += 1
    #             continue
    #         seen_keys.add(key)
            
    #         dialog_groups.append(
    #             {
    #                 "dialog_id": dialog_id,
    #                 "scenario_id": scenario_id,
    #                 "timeline": normalized_timeline,  # Теперь с block_id
    #             }
    #         )
    
    # if duplicate_count > 0:
    #     st.info(f"ℹ️ Удалено **{duplicate_count}** дубликатов.")
    
    # # Формируем turns_df с block_id
    # records = []
    # for item in dialog_groups:
    #     for msg in item["timeline"]:
    #         if msg["type"] == "message":
    #             records.append(
    #                 {
    #                     "dialog_id": item["dialog_id"],
    #                     "role": msg["role"],
    #                     "content": msg["content"],
    #                     "block_id": msg.get("block_id"),  # <-- Новый столбец
    #                 }
    #             )
    
    # turns_df = (
    #     pd.DataFrame(records)
    #     if records
    #     else pd.DataFrame(columns=["dialog_id", "role", "content", "block_id"])
    # )
    
    # return dialog_groups, turns_df
# --- БЛОК ЗАГРУЗКИ ДАННЫХ ---
if source_option == "XLSX-файл":
    # Поле для указания таблицы (только для этого режима)
    table_name = st.text_input(
        "Таблица ClickHouse",
        value="[NDA_TABLE]",
        key="ch_table_xlsx",
        help="Имя таблицы в ClickHouse (без указания базы данных)"
    )
    full_table_name = f"[NDA_SCHEMA].{table_name}"
    
    uploaded_file = st.file_uploader("Загрузите XLSX с ID диалогов", type=["xlsx"])

    _render_http_blocks_llm_settings_expander()

    def _parse_xlsx_dialog_id(val):
        if pd.isna(val):
            return None
        try:
            f = float(val)
            if f != f:  # NaN
                return None
            iv = int(f)
            if abs(f - iv) > 1e-9:
                return None
            return iv
        except (TypeError, ValueError):
            s = str(val).strip()
            return int(s) if s.isdigit() else None

    # Пока файл «висит» в uploader, Streamlit rerun-ит страницу на каждое действие.
    # Раньше мы каждый раз заново дергали ClickHouse и делали pop(dialog_groups), из-за чего
    # сбрасывались виджеты формы и не сохранялись разметка/цель. Загружаем заново только если
    # сменились файл (байты) или имя таблицы — как в режимах с кнопкой «Загрузить».
    need_fresh_xlsx_load = False
    xlsx_load_key = None
    raw_bytes = b""
    if uploaded_file is not None:
        raw_bytes = uploaded_file.getvalue()
        fp = hashlib.sha256(raw_bytes).hexdigest()
        xlsx_load_key = (table_name.strip(), fp)
        need_fresh_xlsx_load = st.session_state.get("_xlsx_load_key") != xlsx_load_key

    if uploaded_file is not None and need_fresh_xlsx_load:
        try:
            df_ids = pd.read_excel(io.BytesIO(raw_bytes))
        except Exception as e:
            st.error(f"Ошибка чтения XLSX: {e}")
            st.stop()
        if "ID" not in df_ids.columns:
            st.error("❌ Не найден столбец 'ID' в файле. Добавьте колонку с названием 'ID'.")
            st.stop()
        raw_ids = df_ids["ID"].dropna().unique()
        dialog_ids = []
        for x in raw_ids:
            pid = _parse_xlsx_dialog_id(x)
            if pid is not None:
                dialog_ids.append(pid)
        dialog_ids = list(dict.fromkeys(dialog_ids))
        if not dialog_ids:
            st.error("❌ Не найдены валидные числовые ID в файле.")
            st.stop()
        st.info(f"🔍 Найдено **{len(dialog_ids)}** уникальных ID. Загружаю из ClickHouse...")
        try:
            with st.spinner("🔄 Выполнение SQL-запроса и обработка..."):
                client = get_clickhouse_client()
                ids_str = ", ".join(str(id_) for id_ in dialog_ids)
                query = (
                    "-- NDA: структура SQL-запроса к ClickHouse скрыт"
                )
                df_raw = clickhouse_select_to_df(client, query)
                st.session_state["df_raw"] = df_raw
                st.session_state["source_loaded"] = True
                st.session_state["_xlsx_load_key"] = xlsx_load_key
                st.session_state.pop("dialog_groups", None)
                st.session_state.pop("turns_df", None)
                st.success(f"✅ Загружено **{len(df_raw)}** диалогов из ClickHouse.")
        except Exception as e:
            st.error(f"❌ Ошибка ClickHouse: {e}")
            st.exception(e)
            st.stop()
    elif uploaded_file is not None and not need_fresh_xlsx_load:
        st.caption(
            "📎 Тот же XLSX и таблица — данные ClickHouse не перезагружаются, сохраняется разметка."
        )
    elif st.session_state.get("source_loaded") and "df_raw" in st.session_state:
        # После «Применить» и других rerun file_uploader часто пустой — продолжаем с кэшем,
        # иначе st.stop() выше не давал бы дойти до обработчика формы разметки.
        st.caption(
            "📎 Используются данные последней загрузки XLSX. При необходимости выберите файл снова."
        )
    else:
        st.info("Пожалуйста, загрузите XLSX-файл.")
        st.stop()
    if "df_raw" not in st.session_state or not st.session_state.get("source_loaded"):
        st.error("Не удалось загрузить данные.")
        st.stop()
    df_raw = st.session_state["df_raw"]
    invalidate_dialog_groups_if_block_filter_changed()
    if "dialog_groups" in st.session_state and "turns_df" in st.session_state:
        dialog_groups = st.session_state["dialog_groups"]
        turns_df = st.session_state["turns_df"]
    else:
        with st.spinner(
            "🔄 Разбор диалогов (messages / tech_logs) — может занять время при большой выборке..."
        ):
            dialog_groups, turns_df = run_clickhouse_processing_with_optional_block_filter(
                df_raw
            )
        st.session_state["dialog_groups"] = dialog_groups
        st.session_state["turns_df"] = turns_df
        # 🔥 Обновляем кэш просмотренных ID
        for item in dialog_groups:
            st.session_state["seen_dialog_ids"].add(item["dialog_id"])

# elif source_option == "ClickHouse: по ID диалогов и сценария":
#     # Поле для указания таблицы (только для этого режима)
#     table_name = st.text_input(
#         "Таблица ClickHouse",
#         value="[NDA_TABLE]",
#         key="ch_table_pairs",
#         help="Имя таблицы в ClickHouse (без указания базы данных)"
#     )
#     full_table_name = f"[NDA_SCHEMA].{table_name}"
    
#     st.markdown("#### Укажите пары `(id, start_block_id)`")
#     example_pairs = """[NDA]"""
#     pairs_input = st.text_area(
#         "Список пар (id, start_block_id), по одной на строку:",
#         value=example_pairs,
#         height=100,
#         key="pairs_input",
#     )
#     if st.button("Загрузить из ClickHouse", key="load_ch_pairs"):
#         if not pairs_input.strip():
#             st.warning("Введите хотя бы одну пару.")
#         else:
#             with st.spinner("🔄 Загрузка и обработка данных из ClickHouse..."):
#                 try:
#                     pairs = []
#                     for line in pairs_input.strip().split("\n"):
#                         line = line.strip()
#                         if not line:
#                             continue
#                         parts = line.split(",")
#                         if len(parts) != 2:
#                             raise ValueError(f"Неверный формат строки: {line}")
#                         dialog_id = int(parts[0].strip())
#                         scenario_id = int(parts[1].strip())
#                         pairs.append((dialog_id, scenario_id))
#                     values_str = ", ".join(f"({d}, {s})" for d, s in pairs)
#                     query = f"""
#                     SELECT id, start_block_id, messages, tech_logs
#                     FROM {full_table_name}
#                     WHERE (id, start_block_id) IN ({values_str})
#                     """
#                     client = get_clickhouse_client()
#                     result = client.query(query)
#                     df_raw = pd.DataFrame(result.named_results())
#                     st.session_state["df_raw"] = df_raw
#                     st.session_state["source_loaded"] = True
#                     st.session_state.pop("dialog_groups", None)
#                     st.session_state.pop("turns_df", None)
#                 except Exception as e:
#                     st.error(f"Ошибка ClickHouse (по ID): {e}")
#     if "df_raw" not in st.session_state or not st.session_state.get("source_loaded"):
#         st.info("Нажмите кнопку для загрузки данных из ClickHouse.")
#         st.stop()
#     df_raw = st.session_state["df_raw"]
#     if "dialog_groups" in st.session_state and "turns_df" in st.session_state:
#         dialog_groups = st.session_state["dialog_groups"]
#         turns_df = st.session_state["turns_df"]
#     else:
#         dialog_groups, turns_df = process_clickhouse_data(df_raw)
#         st.session_state["dialog_groups"] = dialog_groups
#         st.session_state["turns_df"] = turns_df
#         for item in dialog_groups:
#             st.session_state["seen_dialog_ids"].add(item["dialog_id"])

elif source_option == "ClickHouse: по ID диалогов и сценария":
    # Поле для указания таблицы
    table_name = st.text_input(
        "Таблица ClickHouse",
        value="[NDA_TABLE]",
        key="ch_table_pairs",
        help="Имя таблицы в ClickHouse (без указания базы данных)"
    )
    full_table_name = f"[NDA_SCHEMA].{table_name}"
    
    st.markdown("#### Укажите ID диалогов")
    example_dialog_ids = """[NDA_DIALOG_ID]"""
    dialog_ids_input = st.text_area(
        "По одному ID на строке (в таблице `id` уникален; `start_block_id` подставится из строки). "
        "Если скопирована старая строка «id, start_block_id», учитывается только первое число.",
        value=example_dialog_ids,
        height=100,
        key="ch_dialog_ids_input",
    )

    _render_http_blocks_llm_settings_expander()

    if st.button("Загрузить из ClickHouse", key="load_ch_pairs"):
        if not dialog_ids_input.strip():
            st.warning("Введите хотя бы один ID диалога.")
        else:
            with st.spinner("🔄 Выполнение SQL-запроса и обработка..."):
                try:
                    dialog_ids = []
                    for line in dialog_ids_input.strip().split("\n"):
                        line = line.strip()
                        if not line:
                            continue
                        first_token = line.split(",")[0].strip()
                        if not first_token:
                            raise ValueError(f"Пустой ID в строке: {line!r}")
                        dialog_ids.append(int(first_token))
                    dialog_ids = list(dict.fromkeys(dialog_ids))
                    ids_str = ", ".join(str(i) for i in dialog_ids)
                    # ДОБАВЛЕНО: scenario_steps в SELECT
                    query = (
                    "-- NDA: структура SQL-запроса к ClickHouse скрыт"
                )
                    client = get_clickhouse_client()
                    df_raw = clickhouse_select_to_df(client, query)
                    
                    st.session_state["df_raw"] = df_raw
                    st.session_state["source_loaded"] = True
                    st.session_state.pop("dialog_groups", None)
                    st.session_state.pop("turns_df", None)
                    
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Ошибка ClickHouse (по ID): {e}")
    
    if "df_raw" not in st.session_state or not st.session_state.get("source_loaded"):
        st.info("Нажмите кнопку для загрузки данных из ClickHouse.")
        st.stop()
    
    df_raw = st.session_state["df_raw"]
    invalidate_dialog_groups_if_block_filter_changed()

    if "dialog_groups" in st.session_state and "turns_df" in st.session_state:
        dialog_groups = st.session_state["dialog_groups"]
        turns_df = st.session_state["turns_df"]
    else:
        with st.spinner(
            "🔄 Разбор диалогов (messages / tech_logs) — может занять время при большой выборке..."
        ):
            dialog_groups, turns_df = run_clickhouse_processing_with_optional_block_filter(
                df_raw
            )
        st.session_state["dialog_groups"] = dialog_groups
        st.session_state["turns_df"] = turns_df
        if "seen_dialog_ids" not in st.session_state:
            st.session_state["seen_dialog_ids"] = set()
        for item in dialog_groups:
            st.session_state["seen_dialog_ids"].add(item["dialog_id"])

# elif source_option == "ClickHouse: по ID диалогов и сценария":
#     # Поле для указания таблицы (только для этого режима)
#     table_name = st.text_input(
#         "Таблица ClickHouse",
#         value="[NDA_TABLE]",
#         key="ch_table_pairs",
#         help="Имя таблицы в ClickHouse (без указания базы данных)"
#     )
#     full_table_name = f"[NDA_SCHEMA].{table_name}"
    
#     st.markdown("#### Укажите пары `(id, start_block_id)`")
#     example_pairs = """[NDA]"""
#     pairs_input = st.text_area(
#         "Список пар (id, start_block_id), по одной на строку:",
#         value=example_pairs,
#         height=100,
#         key="pairs_input",
#     )
#     if st.button("Загрузить из ClickHouse", key="load_ch_pairs"):
#         if not pairs_input.strip():
#             st.warning("Введите хотя бы одну пару.")
#         else:
#             with st.spinner("🔄 Загрузка и обработка данных из ClickHouse..."):
#                 try:
#                     pairs = []
#                     for line in pairs_input.strip().split("\n"):
#                         line = line.strip()
#                         if not line:
#                             continue
#                         parts = line.split(",")
#                         if len(parts) != 2:
#                             raise ValueError(f"Неверный формат строки: {line}")
#                         dialog_id = int(parts[0].strip())
#                         scenario_id = int(parts[1].strip())
#                         pairs.append((dialog_id, scenario_id))
#                     values_str = ", ".join(f"({d}, {s})" for d, s in pairs)
#                     query = f"""
#                     SELECT id, start_block_id, messages, tech_logs
#                     FROM {full_table_name}
#                     WHERE (id, start_block_id) IN ({values_str})
#                     """
#                     client = get_clickhouse_client()
#                     result = client.query(query)
#                     df_raw = pd.DataFrame(result.named_results())
#                     st.session_state["df_raw"] = df_raw
#                     st.session_state["source_loaded"] = True
#                     st.session_state.pop("dialog_groups", None)
#                     st.session_state.pop("turns_df", None)
#                 except Exception as e:
#                     st.error(f"Ошибка ClickHouse (по ID): {e}")
#     if "df_raw" not in st.session_state or not st.session_state.get("source_loaded"):
#         st.info("Нажмите кнопку для загрузки данных из ClickHouse.")
#         st.stop()
#     df_raw = st.session_state["df_raw"]
#     if "dialog_groups" in st.session_state and "turns_df" in st.session_state:
#         dialog_groups = st.session_state["dialog_groups"]
#         turns_df = st.session_state["turns_df"]
#     else:
#         dialog_groups, turns_df = process_clickhouse_data_scenario_id(df_raw)
#         st.session_state["dialog_groups"] = dialog_groups
#         st.session_state["turns_df"] = turns_df
#         for item in dialog_groups:
#             st.session_state["seen_dialog_ids"].add(item["dialog_id"])



elif source_option == "ClickHouse: по сценарии и дате":
    # Поле для указания таблицы (только для этого режима)
    table_name = st.text_input(
        "Таблица ClickHouse",
        value="[NDA_TABLE]",
        key="ch_table_date",
        help="Имя таблицы в ClickHouse (без указания базы данных)"
    )
    full_table_name = f"[NDA_SCHEMA].{table_name}"
    
    st.markdown("#### Укажите параметры фильтрации")
    scenario_id_input = st.number_input(
        "start_block_id", min_value=1, value=0, key="scenario_id_input"
    )
    start_date = st.date_input(
        "Начальная дата", value=pd.to_datetime("2026-01-01").date(), key="start_date"
    )
    end_date = st.date_input(
        "Конечная дата", value=pd.to_datetime("2026-01-02").date(), key="end_date"
    )
    max_rows_date = st.number_input(
        "Макс. число строк (0 = без лимита)",
        min_value=0,
        value=0,
        step=500,
        key="ch_date_max_rows",
        help="Лимит ускоряет загрузку и разбор: колонки messages/tech_logs очень тяжёлые. "
        "При лимите строки упорядочены по date_start, id.",
    )
    # st.caption(
    #     "Ускорение на стороне ClickHouse: в MergeTree желательно ключ сортировки, начинающийся с "
    #     "`start_block_id`, `date_start` (например `ORDER BY (start_block_id, date_start, id)`). "
    #     "Запрос использует **PREWHERE** по сценарию и дате, чтобы меньше читать тяжёлые колонки."
    # )
    _render_http_blocks_llm_settings_expander()

    if st.button("Загрузить из ClickHouse", key="load_ch_date"):
        with st.spinner("🔄 Выполнение SQL-запроса и обработка..."):
            try:
                start_dt = pd.to_datetime(start_date).strftime("%Y-%m-%d %H:%M:%S")
                end_dt = (pd.to_datetime(end_date) + pd.Timedelta(days=1)).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                limit_part = ""
                if max_rows_date and max_rows_date > 0:
                    limit_part = f"\n                ORDER BY date_start, id\n                LIMIT {int(max_rows_date)}"
                query = (
                    "-- NDA: структура SQL-запроса к ClickHouse скрыт"
                )
                client = get_clickhouse_client()
                df_raw = clickhouse_select_to_df(client, query)
                st.session_state["df_raw"] = df_raw
                st.session_state["source_loaded"] = True
                st.session_state.pop("dialog_groups", None)
                st.session_state.pop("turns_df", None)
                st.success(f"Загружено {len(df_raw)} диалогов из ClickHouse.")
                st.rerun()
            except Exception as e:
                st.error(f"Ошибка ClickHouse (по дате): {e}")
    if "df_raw" not in st.session_state or not st.session_state.get("source_loaded"):
        st.info("Нажмите кнопку для загрузки данных из ClickHouse.")
        st.stop()
    df_raw = st.session_state["df_raw"]
    invalidate_dialog_groups_if_block_filter_changed()
    if "dialog_groups" in st.session_state and "turns_df" in st.session_state:
        dialog_groups = st.session_state["dialog_groups"]
        turns_df = st.session_state["turns_df"]
    else:
        with st.spinner(
            "🔄 Разбор диалогов (messages / tech_logs) — может занять время при большой выборке..."
        ):
            dialog_groups, turns_df = run_clickhouse_processing_with_optional_block_filter(
                df_raw
            )
        st.session_state["dialog_groups"] = dialog_groups
        st.session_state["turns_df"] = turns_df
        # 🔥 Обновляем кэш просмотренных ID
        for item in dialog_groups:
            st.session_state["seen_dialog_ids"].add(item["dialog_id"])

elif source_option == "ClickHouse: по SQL-запросу":
    # Без поля таблицы — пользователь указывает её прямо в запросе
    st.markdown("#### Введите SQL-запрос к ClickHouse")
    st.info(
        "Обязательные столбцы: `id`, `messages`, `tech_logs`. "
        "Столбец **`start_block_id`** опционален (для `scenario_id` в интерфейсе и JSONL; без него — 0). "
        "Столбец **`scenario_steps`** нужен, если сверху включён фильтр по диапазону блоков сценария "
        "или разметка через HTTP-блоки в полях ниже. "
        "Столбец **`variables`** — для снимка переменных и для привязки HTTP-блоков к переменным "
        "(если ниже заданы имена переменных для tool_name / tool_arguments)."
    )
    default_sql = """SELECT
    id,
    messages,
    tech_logs,
    scenario_steps,
    variables
    FROM [NDA_SCHEMA].[NDA_TABLE]
    LIMIT 10"""
    user_sql = st.text_area(
        "SQL-запрос (только SELECT):",
        value=default_sql,
        height=150,
        key="custom_sql_query",
    )
    _render_http_blocks_llm_settings_expander()

    if st.button("Выполнить запрос", key="run_custom_sql"):
        if not user_sql.strip():
            st.warning("Введите SQL-запрос.")
        elif not user_sql.strip().lower().startswith("select"):
            st.error("Разрешены только SELECT-запросы.")
        else:
            with st.spinner("🔄 Выполнение SQL-запроса и обработка..."):
                try:
                    client = get_clickhouse_client()
                    df_raw = clickhouse_select_to_df(client, user_sql)
                    required_cols = {"id", "messages", "tech_logs"}
                    missing = required_cols - set(df_raw.columns)
                    if missing:
                        raise ValueError(f"Отсутствуют обязательные столбцы: {missing}")
                    st.session_state["df_raw"] = df_raw
                    st.session_state["source_loaded"] = True
                    st.session_state.pop("dialog_groups", None)
                    st.session_state.pop("turns_df", None)
                    st.success(f"Загружено {len(df_raw)} диалогов из ClickHouse.")
                except Exception as e:
                    st.error(f"Ошибка выполнения SQL-запроса: {e}")
                    st.exception(e)
    if "df_raw" not in st.session_state or not st.session_state.get("source_loaded"):
        st.info("Нажмите кнопку для выполнения SQL-запроса.")
        st.stop()
    df_raw = st.session_state["df_raw"]
    invalidate_dialog_groups_if_block_filter_changed()
    if "dialog_groups" in st.session_state and "turns_df" in st.session_state:
        dialog_groups = st.session_state["dialog_groups"]
        turns_df = st.session_state["turns_df"]
    else:
        with st.spinner(
            "🔄 Разбор диалогов (messages / tech_logs) — может занять время при большой выборке..."
        ):
            dialog_groups, turns_df = run_clickhouse_processing_with_optional_block_filter(
                df_raw
            )
        st.session_state["dialog_groups"] = dialog_groups
        st.session_state["turns_df"] = turns_df
        # 🔥 Обновляем кэш просмотренных ID
        for item in dialog_groups:
            st.session_state["seen_dialog_ids"].add(item["dialog_id"])

elif source_option in ("Загрузить готовый JSONL", "Сохранённая выборка (БД)"):
    jsonl_text = None
    if source_option == "Загрузить готовый JSONL":
        uploaded_jsonl = st.file_uploader(
            "Загрузите JSONL-файл с диалогами", type=["jsonl"]
        )
        if not uploaded_jsonl:
            st.info("Пожалуйста, загрузите JSONL-файл.")
            st.stop()
        jsonl_text = uploaded_jsonl.getvalue().decode("utf-8")
    else:
        _db_msg = st.session_state.pop("_markup01_db_loaded_msg", None)
        if _db_msg:
            st.success(_db_msg)

        def _on_markup01_text(text: str) -> None:
            st.session_state["_markup01_jsonl_text"] = text
            st.session_state["_markup01_db_loaded_msg"] = "Выборка загружена из БД."
            st.rerun()

        render_jsonl_dataset_source(
            key_prefix="markup01",
            on_text_loaded=_on_markup01_text,
            file_types=["jsonl"],
            upload_label="JSONL с диалогами",
        )
        jsonl_text = st.session_state.get("_markup01_jsonl_text")
        if not jsonl_text:
            st.info("Выберите сохранённую выборку или загрузите файл.")
            st.stop()

    try:
        dialog_groups, turns_df = parse_jsonl_text_to_dialog_groups(
            jsonl_text,
            is_noise=is_noise,
            seen_dialog_ids=st.session_state["seen_dialog_ids"],
        )
        st.session_state.pop("dialog_groups", None)
        st.session_state.pop("turns_df", None)
    except Exception as e:
        st.error(f"Ошибка при чтении JSONL: {e}")
        st.exception(e)
        st.stop()
else:
    st.error("Неизвестный источник")
    st.stop()
# --- КОНЕЦ БЛОКА ЗАГРУЗКИ ДАННЫХ ---


if turns_df.empty:
    st.warning("После обработки не осталось реплик.")
    st.stop()

num_dialogs = turns_df["dialog_id"].nunique()
st.success(f"✅ Обработано: {len(turns_df)} реплик из **{num_dialogs} диалогов**.")

if not dialog_groups:
    st.error("Не удалось создать диалоги.")
    st.stop()

total_dialogs = len(dialog_groups)


def get_dialog_history_length(dialog_id, timeline):
    ann = st.session_state["annotations"].get(dialog_id)
    max_len = len(timeline)
    if st.session_state.get("app_config", {}).get("save_full_by_default", False):
        if ann and ann.get("custom_history_enabled", False):
            return min(ann.get("custom_history_length", max_len), max_len)
        else:
            return max_len
    else:
        if ann and ann.get("custom_history_enabled", False):
            return min(
                ann.get("custom_history_length", st.session_state["app_config"]["default_history"]),
                max_len,
            )
        else:
            return min(st.session_state["app_config"]["default_history"], max_len)

# === Инициализация конфигурации (если нет) ===
if "app_config" not in st.session_state:
    st.session_state["app_config"] = {
        "num_to_annotate": min(20, total_dialogs),
        "save_full_by_default": False,  # ИСПРАВЛЕНО: по умолчанию False
        "default_history": min(20, max((len(item["timeline"]) for item in dialog_groups), default=4)),
        "allowed_intents": "Консультация, Уточнение условий залога, Погашение займа, Продление срока, Вопрос по залоговому билету",
        "export_context_json": "{}",
    }

# === ФОРМА НАСТРОЕК РАЗМЕТКИ (ИСПРАВЛЕНО 1: allowed_intents в форме) ===
st.subheader("🎛️ Настройки разметки")

# После смены выборки (фильтр, ClickHouse) total_dialogs может стать меньше сохранённого num_to_annotate — иначе StreamlitValueAboveMaxError.
_max_annotate = max(1, total_dialogs)
_cfg = st.session_state["app_config"]
_cfg.setdefault("export_context_json", "{}")
_cfg["num_to_annotate"] = max(1, min(_cfg["num_to_annotate"], _max_annotate))
_wn = "form_num_to_annotate"
if _wn in st.session_state and st.session_state[_wn] > _max_annotate:
    st.session_state[_wn] = _cfg["num_to_annotate"]

with st.form("annotation_settings_form"):
    config = st.session_state["app_config"]

    f_num_to_annotate = st.number_input(
        f"Выберите, сколько диалогов размечать (всего: {total_dialogs})",
        min_value=1,
        max_value=_max_annotate,
        value=config["num_to_annotate"],
        key="form_num_to_annotate",
    )
    
    st.markdown("⚙️ Глобальные настройки длины истории")
    f_save_full = st.checkbox(
        "☑️ Сохранять полную длину диалога (отключить обрезку по умолчанию)",
        value=config["save_full_by_default"],
        key="form_save_full"
    )
    
    f_default_history = st.number_input(
        "Длина истории по умолчанию (реплик)",
        min_value=1,
        max_value=20,
        value=config["default_history"],
        disabled=f_save_full,
        key="form_default_history"
    )
    
    f_apply_history_to_all = st.checkbox(
        "🔄 Применить эти настройки истории ко всем текущим диалогам",
        value=False,
        key="form_apply_history_all",
        help="Если отмечено, при нажатии 'Применить настройки' параметры истории обновятся для всех видимых диалогов"
    )

    st.markdown("🎯 Допустимые цели")
    f_allowed_intents = st.text_input(
        "Введите допустимые цели диалогов (через запятую)",
        value=config["allowed_intents"],
        key="form_allowed_intents"
    )

    _gctx = (config.get("export_context_json") or "").strip() or "{}"
    if (
        st.session_state.get("_export_global_ctx_sig") != _gctx
        or "form_cfg_global_context" not in st.session_state
    ):
        st.session_state["form_cfg_global_context"] = _gctx
        st.session_state["_export_global_ctx_sig"] = _gctx

    with st.expander("Пример задания context (JSON)", expanded=False):
        st.caption(
            "Один JSON-объект `{ ... }`. Ключи и строковые значения — в двойных кавычках. "
            "При экспорте объединяется с полями диалога (если есть) и с context из карточки диалога."
        )
        st.code(_EXPORT_CONTEXT_JSON_EXAMPLE_GLOBAL, language="json")
    st.text_area(
        "Общий context (JSON) для всех диалогов",
        height=120,
        key="form_cfg_global_context",
        placeholder='{\n  "branch": "центр",\n  "city": "Москва"\n}',
        help="Объект {...}. При экспорте объединяется с данными диалога; совпадающие ключи перезаписывает context из карточки диалога.",
    )
    
    submitted = st.form_submit_button("💾 Применить настройки", use_container_width=True)
    
    if submitted:
        _raw_g = (st.session_state.get("form_cfg_global_context", "") or "").strip() or "{}"
        _, _gerr = _parse_context_json_object(_raw_g)
        if _gerr:
            st.error(f"Общий context: невалидный JSON — {_gerr}")
        else:
            st.session_state["app_config"]["num_to_annotate"] = f_num_to_annotate
            st.session_state["app_config"]["save_full_by_default"] = f_save_full
            st.session_state["app_config"]["default_history"] = f_default_history
            st.session_state["app_config"]["allowed_intents"] = f_allowed_intents  # ИСПРАВЛЕНО 1: только по кнопке
            st.session_state["app_config"]["export_context_json"] = _raw_g
            st.session_state["_export_global_ctx_sig"] = _raw_g
            
            if f_apply_history_to_all:
                current_ids = {item["dialog_id"] for item in dialog_groups[:f_num_to_annotate]}
                for item in dialog_groups:
                    if item["dialog_id"] in current_ids:
                        dialog_id = item["dialog_id"]
                        if dialog_id not in st.session_state.get("annotations", {}):
                            st.session_state.setdefault("annotations", {})[dialog_id] = {}
                        
                        ann = st.session_state["annotations"][dialog_id]
                        ann["custom_history_enabled"] = False
                        if not f_save_full:
                            ann["custom_history_length"] = f_default_history
                
                st.success("✅ Настройки истории применены ко всем диалогам.")
            
            st.rerun()

# === Чтение активных настроек (после формы) ===
active_config = st.session_state["app_config"]
num_to_annotate = active_config["num_to_annotate"]
allowed_intents_str = active_config["allowed_intents"]

selected_dialogs = dialog_groups[:num_to_annotate]
current_ids = {item["dialog_id"] for item in selected_dialogs}

allowed_intents = [g.strip() for g in allowed_intents_str.split(",") if g.strip()]
normalized_to_original = {normalize_intent(g): g for g in allowed_intents}

if "annotations" not in st.session_state:
    st.session_state["annotations"] = {}

st.session_state["annotations"] = {
    k: v for k, v in st.session_state.get("annotations", {}).items() if k in current_ids
}

for item in selected_dialogs:
    dialog_id = item["dialog_id"]
    if dialog_id not in st.session_state["annotations"]:
        initial_goal = ""
        if "original_goals" in item and item["original_goals"]:
            og = item["original_goals"][0]
            if isinstance(og, str) and og.strip():
                cand = og.strip()
                nk = normalize_intent(cand)
                if nk in normalized_to_original:
                    initial_goal = normalized_to_original[nk]
        st.session_state["annotations"][dialog_id] = {
            "save": True,
            "intent_mode": initial_goal,
            "custom_history_enabled": False,
            "custom_history_length": 4,
            "truncation_direction": "с начала",
            "export_context_json": "",
        }

# Цель вне списка допустимых не храним — диалог считается неразмеченным
for _did, _ann in st.session_state["annotations"].items():
    if _did not in current_ids:
        continue
    _ann.setdefault("export_context_json", "")
    _raw = (_ann.get("intent_mode") or "").strip()
    if _raw and normalize_intent(_raw) not in normalized_to_original:
        _ann["intent_mode"] = ""

if "stop_annotation" not in st.session_state:
    st.session_state["stop_annotation"] = False

# === Инициализация настроек LLM (если нет) ===
if "llm_config" not in st.session_state:
    st.session_state["llm_config"] = {
        "llm_delay": 0.0,
        "llm_mode": "Только неразмеченные",
        "llm_prompt_template": """Проанализируй диалог и определи его основную цель.
Выбери из: {allowed_intents}.
Если не подходит — предложи кратко.
Диалог:
{dialog}
Ответь только названием цели.""",
    }

# === ФОРМА НАСТРОЕК LLM-РАЗМЕТКИ ===
st.subheader("🧠 LLM-разметка")

with st.form("llm_settings_form"):
    llm_config = st.session_state["llm_config"]
    
    st.markdown("⚙️ Параметры LLM-разметки")
    
    f_llm_delay = st.number_input(
        "Задержка перед каждым вызовом LLM (секунды)",
        min_value=0.0,
        max_value=100.0,
        value=llm_config["llm_delay"],
        step=0.1,
        help="Используется как time.sleep() перед каждым вызовом LLM.",
        key="form_llm_delay"
    )
    
    f_llm_mode = st.radio(
        "Режим разметки:",
        options=["Только неразмеченные", "Все диалоги"],
        index=0 if llm_config["llm_mode"] == "Только неразмеченные" else 1,
        horizontal=True,
        key="form_llm_mode"
    )
    
    f_llm_prompt_template = st.text_area(
        "Промпт для LLM",
        value=llm_config["llm_prompt_template"],
        height=120,
        key="form_llm_prompt_template"
    )
    
    col_btn1, col_btn2 = st.columns([3, 1])
    with col_btn1:
        f_submit_llm = st.form_submit_button("🤖 Разметить через LLM", use_container_width=True)
    with col_btn2:
        f_stop_llm = st.form_submit_button("⛔ Остановить", use_container_width=True)
    
    if f_submit_llm:
        st.session_state["llm_config"]["llm_delay"] = f_llm_delay
        st.session_state["llm_config"]["llm_mode"] = f_llm_mode
        st.session_state["llm_config"]["llm_prompt_template"] = f_llm_prompt_template
        st.session_state["stop_annotation"] = False
        
        if not st.session_state.get("litellm_api_key", "").strip():
            st.warning("⚠️ Укажите API-ключ в боковой панели.")
        else:
            with st.spinner("🔄 Выполнение разметки диалога..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                allowed_intents_str_for_llm = ", ".join(allowed_intents)
                updated_count = 0
                total_processed = 0
                total_to_process = 0
                
                for item in selected_dialogs:
                    dialog_id = item["dialog_id"]
                    ann = st.session_state["annotations"][dialog_id]
                    if not ann.get("save", False):
                        continue
                    if f_llm_mode == "Все диалоги":
                        total_to_process += 1
                    else:
                        current_goal = ann["intent_mode"]
                        if not (current_goal or "").strip():
                            total_to_process += 1
                
                current_progress = 0
                stopped_early = False
                for idx, item in enumerate(selected_dialogs):
                    if st.session_state.get("stop_annotation", False):
                        stopped_early = True
                        st.warning(f"⛔ Разметка остановлена пользователем. Обработано {total_processed} из {total_to_process} диалогов.")
                        break  
                    
                    dialog_id = item["dialog_id"]
                    ann = st.session_state["annotations"][dialog_id]
                    if not ann.get("save", False):
                        continue
                    
                    current_goal = ann["intent_mode"]
                    should_process = False
                    if f_llm_mode == "Все диалоги":
                        should_process = True
                    else:
                        if not (current_goal or "").strip():
                            should_process = True
                    
                    if should_process:
                        total_processed += 1
                        hist_len = len(item["timeline"])
                        history_to_send = item["timeline"][:hist_len]
                        
                        status_text.text(f"🔄 Обработка диалога {dialog_id} ({total_processed}/{total_to_process})...")
                        try:
                            suggested = suggest_intent_with_llm(
                                history_to_send, f_llm_prompt_template, allowed_intents_str_for_llm, f_llm_delay
                            )
                            norm_suggested = normalize_intent(suggested)
                            if norm_suggested in normalized_to_original:
                                final_goal = normalized_to_original[norm_suggested]
                                st.session_state["annotations"][dialog_id]["intent_mode"] = final_goal  # ИСПРАВЛЕНО 2: сохраняется сразу
                                st.session_state[f"goal_{dialog_id}"] = final_goal
                                updated_count += 1
                            else:
                                # Ответ не из списка целей — не сохраняем, диалог остаётся неразмеченным
                                st.session_state["annotations"][dialog_id]["intent_mode"] = ""
                                st.session_state.pop(f"goal_{dialog_id}", None)
                        except Exception as e:
                            st.error(f"Ошибка при разметке диалога {dialog_id}: {e}")
                            st.session_state["annotations"][dialog_id]["intent_mode"] = ""
                            st.session_state.pop(f"goal_{dialog_id}", None)
                        
                        current_progress += 1
                        if total_to_process > 0:
                            progress_bar.progress(current_progress / total_to_process)
                
                status_text.empty()
                progress_bar.empty()

                # После любой LLM-разметки annotations — истина; виджеты формы с key= держат старые значения
                # до сброса (особенно при ранней остановке: раньше selectbox оставался пустым).
                clear_dialog_annotation_widget_state()

                if stopped_early:
                    st.info(
                        f"⚠️ Прервано. Успешно размечено {updated_count} диалогов из {total_processed} обработанных."
                    )
                elif f_llm_mode == "Все диалоги":
                    st.success(f"✅ LLM переразметил {updated_count} из {total_processed} диалогов.")
                else:
                    st.success(f"✅ LLM разметил {updated_count} неразмеченных диалогов.")

                st.session_state["stop_annotation"] = False

    if f_stop_llm:
        st.session_state["stop_annotation"] = True
        st.warning("🛑 Флаг остановки установлен. Текущий диалог завершится, и процесс прекратится.")

if st.session_state.get("rerun_after_llm", False):
    st.session_state["rerun_after_llm"] = False  # сбрасываем флаг
    st.rerun()

# === ПАГИНАЦИЯ ===
dialogs_per_page = 10
total_dialogs_selected = len(selected_dialogs)
total_pages = (total_dialogs_selected + dialogs_per_page - 1) // dialogs_per_page

if "current_page" not in st.session_state:
    st.session_state.current_page = 0

current_page = st.session_state.current_page
col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    if st.button("← Назад", disabled=(current_page == 0)):
        st.session_state.current_page -= 1
        st.rerun()
with col2:
    st.markdown(f"Страница {current_page + 1} из {total_pages}")
with col3:
    if st.button("Вперёд →", disabled=(current_page >= total_pages - 1)):
        st.session_state.current_page += 1
        st.rerun()

start_idx = current_page * dialogs_per_page
end_idx = min(start_idx + dialogs_per_page, total_dialogs_selected)
page_dialogs = selected_dialogs[start_idx:end_idx]

# ✅ СИНХРОНИЗАЦИЯ — ЕДИНОЖДЫ, после определения page_dialogs и перед формой
sync_dialog_form_temp(page_dialogs, st.session_state["annotations"])

st.subheader(
    f"📝 Разметка: {num_to_annotate} диалогов (показано {len(page_dialogs)} из {total_dialogs_selected})"
)

# === Инициализация временных значений для формы ===
if "dialog_form_temp" not in st.session_state:
    st.session_state["dialog_form_temp"] = {}

for item in page_dialogs:
    dialog_id = item["dialog_id"]
    full_timeline = item["timeline"]
    max_len = len(full_timeline)
    
    temp_keys = {
        "goal": f"temp_goal_{dialog_id}",
        "save": f"temp_save_{dialog_id}",
        "custom_hist": f"temp_custom_hist_{dialog_id}",
        "custom_len": f"temp_custom_len_{dialog_id}",
        "trunc_dir": f"temp_trunc_dir_{dialog_id}",
        "expanded": f"expanded_{dialog_id}",
    }
    
    if temp_keys["goal"] not in st.session_state["dialog_form_temp"]:
        st.session_state["dialog_form_temp"][temp_keys["goal"]] = st.session_state["annotations"][dialog_id].get("intent_mode", "")
    if temp_keys["save"] not in st.session_state["dialog_form_temp"]:
        st.session_state["dialog_form_temp"][temp_keys["save"]] = st.session_state["annotations"][dialog_id].get("save", True)
    if temp_keys["custom_hist"] not in st.session_state["dialog_form_temp"]:
        st.session_state["dialog_form_temp"][temp_keys["custom_hist"]] = st.session_state["annotations"][dialog_id].get("custom_history_enabled", False)
    if temp_keys["custom_len"] not in st.session_state["dialog_form_temp"]:
        st.session_state["dialog_form_temp"][temp_keys["custom_len"]] = st.session_state["annotations"][dialog_id].get("custom_history_length", min(4, max_len))
    if temp_keys["trunc_dir"] not in st.session_state["dialog_form_temp"]:
        st.session_state["dialog_form_temp"][temp_keys["trunc_dir"]] = st.session_state["annotations"][dialog_id].get("truncation_direction", "с начала")
    if temp_keys["expanded"] not in st.session_state:
        st.session_state[temp_keys["expanded"]] = False
# === ПОДГОТОВКА ДАННЫХ (перед формой) ===
normalized_to_original = {normalize_intent(g): g for g in allowed_intents}
if "dialog_form_temp" not in st.session_state:
    st.session_state["dialog_form_temp"] = {}
# === ФОРМА РАЗМЕТКИ ДИАЛОГОВ ===
with st.form("dialog_annotation_form"):
    with st.expander("Пример дополнительного context в карточке диалога (JSON)", expanded=False):
        st.caption(
            "Формат тот же, что у общего context. Пустое поле или `{}` — без дополнений; "
            "иначе объект объединяется поверх данных диалога и общего context."
        )
        st.code(_EXPORT_CONTEXT_JSON_EXAMPLE_DIALOG, language="json")
    for item in page_dialogs:
        dialog_id = item["dialog_id"]
        full_timeline = item["timeline"]
        max_len = len(full_timeline)
        
        # 🔧 Защита от пустых диалогов
        if max_len == 0:
            max_len = 1
        
        temp_keys = {
            "goal": f"temp_goal_{dialog_id}",
            "save": f"temp_save_{dialog_id}",
            "custom_hist": f"temp_custom_hist_{dialog_id}",
            "custom_len": f"temp_custom_len_{dialog_id}",
            "trunc_dir": f"temp_trunc_dir_{dialog_id}",
            "expanded": f"expanded_{dialog_id}",
        }
        ann = st.session_state["annotations"][dialog_id]
        
        
        # Определяем эффективную длину истории для превью
        temp_save_full = st.session_state["dialog_form_temp"][temp_keys["custom_hist"]]
        temp_hist_len = st.session_state["dialog_form_temp"][temp_keys["custom_len"]]
        temp_trunc = st.session_state["dialog_form_temp"][temp_keys["trunc_dir"]]
        
        if st.session_state["app_config"]["save_full_by_default"]:
            if temp_save_full:
                effective_len = min(temp_hist_len, max_len)
            else:
                effective_len = max_len
        else:
            if temp_save_full:
                effective_len = min(temp_hist_len, max_len)
            else:
                effective_len = min(st.session_state["app_config"]["default_history"], max_len)
        
        if temp_trunc == "с начала":
            display_timeline = full_timeline[:effective_len]
        else:
            display_timeline = full_timeline[-effective_len:] if effective_len <= max_len else full_timeline
        # Заголовок: в annotations только пусто или цель из допустимого списка
        current_intent = st.session_state["annotations"][dialog_id].get("intent_mode", "").strip()
        if not current_intent:
            goal_display = ""
            status_label = " ⚠️ [НЕРАЗМЕЧЕН]"
        else:
            goal_display = f" → {current_intent}"
            status_label = ""
        scenario_info = f" (сценарий: {item['scenario_id']})" if item["scenario_id"] is not None else ""

        # 🔧 Ключи виджетов
        key_custom_hist = f"form_custom_hist_{dialog_id}"
        key_custom_len = f"form_custom_len_{dialog_id}"
        key_trunc_dir = f"form_trunc_dir_{dialog_id}"
        key_save = f"form_save_{dialog_id}"
        key_goal = f"form_goal_{dialog_id}"
        
        # 🔧 Инициализация ТОЛЬКО если ключей ещё нет в session_state
        if key_save not in st.session_state:
            st.session_state[key_save] = st.session_state["dialog_form_temp"][temp_keys["save"]]
        if key_custom_hist not in st.session_state:
            st.session_state[key_custom_hist] = st.session_state["dialog_form_temp"][temp_keys["custom_hist"]]
        if key_custom_len not in st.session_state:
            # 🔧 КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ: Ограничиваем значение в пределах min/max
            stored_len = st.session_state["dialog_form_temp"][temp_keys["custom_len"]]
            st.session_state[key_custom_len] = max(1, min(stored_len, max_len))
        if key_trunc_dir not in st.session_state:
            st.session_state[key_trunc_dir] = st.session_state["dialog_form_temp"][temp_keys["trunc_dir"]]
        
        current_goal = (st.session_state["annotations"][dialog_id].get("intent_mode") or "").strip()
        # Подтягиваем selectbox к annotations, когда цель в данных изменилась (LLM, импорт и т.д.),
        # но не затираем несохранённый выбор пользователя (пока intent_mode в annotations тот же).
        sync_meta = f"_ann_goal_sync_{dialog_id}"
        if st.session_state.get(sync_meta) != current_goal:
            st.session_state[key_goal] = current_goal
            st.session_state[sync_meta] = current_goal

        key_dialog_ctx = f"form_dialog_ctx_{dialog_id}"
        sync_ctx = f"_ann_ctx_sync_{dialog_id}"
        _stored_dctx = (ann.get("export_context_json") or "").strip()
        if st.session_state.get(sync_ctx) != _stored_dctx:
            st.session_state[key_dialog_ctx] = _stored_dctx or "{}"
            st.session_state[sync_ctx] = _stored_dctx

        options = [""] + allowed_intents
        if current_goal and current_goal not in options:
            options = [current_goal] + options
        with st.expander(
            f"Диалог {dialog_id}{scenario_info} (всего: {max_len}){goal_display}{status_label}",
            expanded=st.session_state.get(temp_keys["expanded"], False)
        ):
            _bchain = item.get("block_chain_order")
            if _bchain:
                st.caption(
                    f"Цепочка **block_id** по шагам `scenario_steps` (start_block_id в БД: "
                    f"{item.get('scenario_id')}): порядок поля **id** шага, иначе порядок строк в массиве; "
                    "каждая строка — отдельное звено (повторы block_id сохраняются)."
                )
                st.code(" → ".join(str(x) for x in _bchain), language=None)

            st.checkbox(
                "✅ Сохранить в итоговый файл",
                key=key_save,
            )
            
            st.selectbox(
                "🎯 Цель",
                options=options,
                index=options.index(current_goal) if current_goal in options else 0,
                key=key_goal,
            )

            st.text_area(
                "Дополнительный context диалога (JSON)",
                height=100,
                key=key_dialog_ctx,
                placeholder='{\n  "caller_id": "+79001234567"\n}',
                help="Объект {...}. При экспорте: данные диалога → общий context из настроек → этот JSON. См. пример в начале формы.",
            )
            
            st.markdown("**Хронологическая история (полная):**")
            _http_ui = _http_tool_variable_config_enabled()
            for msg in full_timeline:
                mt = msg.get("type")
                if _markup_hide_tool_calls_ui() and mt in (
                    "tool_call",
                    "tool_schema",
                    "tool_message",
                ):
                    continue
                # Synthetic по HTTP — только если введены имена переменных.
                if is_tool_event_from_http_variables(msg) and not _http_ui:
                    continue
                # Если имена заданы — не показываем инструменты из tech_logs (лишние get_connect_date / tool__GetDate).
                if skip_timeline_tech_log_tool_when_http_variables_ui(
                    msg, http_variables_ui=_http_ui
                ):
                    continue
                if mt in ("tool_call", "tool_schema"):
                    if mt == "tool_call":
                        _hv = msg.get("http_tool_variable_names") or []
                        _title = (
                            f"🔧 **Вызов инструмента:** ({', '.join(map(str, _hv))})"
                            if _hv
                            else "🔧 **Вызов инструмента:**"
                        )
                    else:
                        _tv = msg.get("tool_variable")
                        _title = (
                            f"🔧 **Вызов инструмента:** ({_tv})"
                            if _tv
                            else "🔧 **Вызов инструмента:**"
                        )
                    st.markdown(_title)
                    if mt == "tool_call":
                        _tc = tool_calls_for_timeline_ui_display(msg.get("tool_calls"))
                    else:
                        _tc = tool_calls_for_timeline_ui_display(
                            tech_log_openai_schema_as_markup_tool_calls(
                                msg.get("tool_schema"),
                                tool_variable=msg.get("tool_variable"),
                            )
                        )
                    st.code(
                        format_tool_calls_json_pretty(_tc),
                        language="json",
                    )
                elif mt == "tool_message":
                    _tn = msg.get("name") or "tool"
                    st.markdown(f"🔧 **Результат инструмента** (`{_tn}`):")
                    st.code(msg.get("content") or "", language=None)
                else:
                    _ic = markup_timeline_row_icon(msg)
                    st.text(f"{_ic} {msg.get('content', '')}")

            with st.expander("📊 Переменные (variables)", expanded=False):
                raw = item.get("variables_raw") or []
                _allow = _http_variables_allowlist_for_ui()
                if raw:
                    max_mid = _max_message_id_from_timeline(full_timeline)
                    snap = snapshot_variables_latest_by_name(raw, max_mid)
                    snap = restrict_variables_snapshot_to_allowlist(snap, _allow)
                    _cap = (
                        "Снимок к концу **полной** хронологии: последнее значение по каждому "
                        "`variable_name` с учётом `message_id` (как при экспорте без обрезки)."
                    )
                    if _allow is not None:
                        _cap += (
                            " **Фильтр имен:** только переменные из полей «Названия переменных "
                            "(tool_name)» и «(tool_arguments)» в настройках HTTP-блоков к ClickHouse."
                        )
                    st.caption(_cap)
                    if snap:
                        df_snap = pd.DataFrame(
                            sorted(snap.items(), key=lambda x: str(x[0]).lower()),
                            columns=["variable_name", "value"],
                        )
                        st.dataframe(df_snap, use_container_width=True, hide_index=True)
                    else:
                        st.caption(
                            "В логе нет записей с полем `variable_name` (возможны только параметры шагов)."
                            if _allow is None
                            else "Нет записей с выбранными именами переменных (проверьте настройки имён или лог)."
                        )
                    _raw_for_table = restrict_variables_rows_to_allowlist(raw, _allow)
                    _log_label = (
                        "Полный лог variables (все записи)"
                        if _allow is None
                        else "Лог variables (только выбранные имена)"
                    )
                    with st.expander(_log_label, expanded=False):
                        try:
                            st.dataframe(
                                pd.DataFrame(_raw_for_table), use_container_width=True
                            )
                        except Exception:
                            st.json(_raw_for_table)
                elif isinstance(item.get("variables"), dict) and item["variables"]:
                    st.caption("Данные из импортированного JSONL (`context` / `variables`), без сырого лога.")
                    df_imp = pd.DataFrame(
                        sorted(item["variables"].items(), key=lambda x: str(x[0]).lower()),
                        columns=["variable_name", "value"],
                    )
                    st.dataframe(df_imp, use_container_width=True, hide_index=True)
                else:
                    st.caption(
                        "Нет данных: добавьте колонку **variables** в запрос к ClickHouse или "
                        "импортируйте JSONL со снимком в `context`."
                    )

            st.markdown("**⚙️ Настройки длины истории для этого диалога:**")
            
            # ✅ Чекбокс кастомной длины
            st.checkbox(
                "✅ Использовать свою длину истории (иначе используется глобальная)",
                key=key_custom_hist,
            )
            
            # 🔢 Number input: ВСЕГДА видим, value из session_state (без параметра value=)
            st.number_input(
                "Сколько реплик сохранить?",
                min_value=1,
                max_value=max_len,
                key=key_custom_len,
                help="🔹 Активируйте чекбокс выше, чтобы это значение применялось при экспорте"
            )
            
            # 📍 Radio направления обрезки
            st.radio(
                "Направление обрезки:",
                options=["с начала", "с конца"],
                index=0 if st.session_state[key_trunc_dir] == "с начала" else 1,
                key=key_trunc_dir,
            )
    
    # 🔧 КНОПКА ОТПРАВКИ — ВНУТРИ БЛОКА with st.form()!
    submitted = st.form_submit_button("💾 Применить изменения на странице", use_container_width=True)

# === ОБРАБОТКА ОТПРАВКИ — ВНЕ БЛОКА form ===
if submitted:
    _ctx_bad = []
    for item in page_dialogs:
        dialog_id = item["dialog_id"]
        _d_raw = st.session_state.get(f"form_dialog_ctx_{dialog_id}", "")
        _, _d_err = _parse_context_json_object(_d_raw)
        if _d_err:
            _ctx_bad.append((dialog_id, _d_err))

    if _ctx_bad:
        for _did, _em in _ctx_bad:
            st.error(f"Диалог {_did}: невалидный context (JSON) — {_em}")
    else:
        for item in page_dialogs:
            dialog_id = item["dialog_id"]
            key_custom_hist = f"form_custom_hist_{dialog_id}"
            key_custom_len = f"form_custom_len_{dialog_id}"
            key_trunc_dir = f"form_trunc_dir_{dialog_id}"
            key_save = f"form_save_{dialog_id}"
            key_goal = f"form_goal_{dialog_id}"
            
            applied_goal = (st.session_state.get(key_goal, "") or "").strip()
            if applied_goal and normalize_intent(applied_goal) not in normalized_to_original:
                applied_goal = ""
            st.session_state["annotations"][dialog_id]["intent_mode"] = applied_goal
            st.session_state["annotations"][dialog_id]["save"] = st.session_state.get(key_save, True)
            st.session_state["annotations"][dialog_id]["custom_history_enabled"] = st.session_state.get(key_custom_hist, False)
            st.session_state["annotations"][dialog_id]["custom_history_length"] = st.session_state.get(key_custom_len, 4)
            st.session_state["annotations"][dialog_id]["truncation_direction"] = st.session_state.get(key_trunc_dir, "с начала")
            st.session_state["annotations"][dialog_id]["export_context_json"] = (
                st.session_state.get(f"form_dialog_ctx_{dialog_id}", "") or ""
            ).strip()
            if applied_goal:
                st.session_state[f"goal_{dialog_id}"] = applied_goal
            else:
                st.session_state.pop(f"goal_{dialog_id}", None)
        
        st.success("✅ Изменения применены!")
        st.session_state.pop("dialog_form_temp", None)
        st.rerun()

# === ЭКСПОРТ ===
st.subheader("📤 Настройки экспорта")
col_a, col_b = st.columns([2, 3])
with col_a:
    save_only_labeled = st.checkbox(
        "☑️ Сохранить только размеченные диалоги", value=False
    )
with col_b:
    all_possible_goals = set(allowed_intents)
    for ann in st.session_state.get("annotations", {}).values():
        g = ann.get("intent_mode", "").strip()
        all_possible_goals.add(g)
    sorted_goals = sorted(all_possible_goals, key=lambda x: (x != "", x))
    selected_export_goals = st.multiselect(
        "🎯 Выберите цели для сохранения (оставьте пустым — все)",
        options=sorted_goals,
        default=sorted_goals,
    )
filter_by_goals = len(selected_export_goals) > 0

st.subheader("📤 Экспорт")
export_data = []
normalized_to_original = {normalize_intent(g): g for g in allowed_intents}
_global_ctx_json = (st.session_state["app_config"].get("export_context_json") or "").strip() or "{}"
for item in selected_dialogs:
    dialog_id = item["dialog_id"]
    ann = st.session_state["annotations"][dialog_id]
    if not ann.get("save", False):
        continue
    current_goal = ann["intent_mode"].strip()
    is_labeled = (
        bool(current_goal) and normalize_intent(current_goal) in normalized_to_original
    )
    if save_only_labeled and not is_labeled:
        continue
    if filter_by_goals and current_goal not in selected_export_goals:
        continue
    
    if is_labeled:
        final_goal = normalized_to_original[normalize_intent(current_goal)]
    else:
        final_goal = current_goal
    
    full_timeline = item["timeline"]
    max_len = len(full_timeline)
    hist_len = get_dialog_history_length(dialog_id, full_timeline)
    trunc_dir = ann.get("truncation_direction", "с начала")
    
    if trunc_dir == "с начала":
        export_timeline = full_timeline[:hist_len]
    else:
        export_timeline = (
            full_timeline[-hist_len:] if hist_len <= max_len else full_timeline
        )
    
    history_for_export = []
    last_full_response = {"response": ""}
    _omit_tools = _markup_hide_tool_calls_ui()
    for msg in export_timeline:
        if _omit_tools and msg.get("type") in (
            "tool_call",
            "tool_message",
            "tool_schema",
        ):
            continue
        if msg["type"] == "tool_schema":
            continue
        exported_msg = {"role": msg["role"]}
        if msg["type"] == "message":
            content = msg["content"]
            exported_msg["content"] = content
            exported_msg["full_response"] = {"response": content}
            exported_msg["is_external"] = False
            last_full_response = {"response": content}
        elif msg["type"] == "tool_call":
            exported_msg["content"] = ""
            exported_msg["full_response"] = {
                "response": "",
                "tool_calls": msg["tool_calls"],
            }
            exported_msg["is_external"] = False
            last_full_response = exported_msg["full_response"]
        elif msg["type"] == "tool_message":
            exported_msg["content"] = msg.get("content", "")
            exported_msg["name"] = msg.get("name", "tool")
            exported_msg["full_response"] = {"response": msg.get("content", "")}
            exported_msg["is_external"] = False
            last_full_response = exported_msg["full_response"]
        history_for_export.append(exported_msg)

    max_mid_export = _max_message_id_from_timeline(export_timeline)
    raw_vars = item.get("variables_raw") or []
    if raw_vars:
        export_variables = snapshot_variables_latest_by_name(
            raw_vars, max_mid_export
        )
        export_variables = restrict_variables_snapshot_to_allowlist(
            export_variables, _http_variables_allowlist_for_ui()
        )
    elif isinstance(item.get("context"), dict):
        export_variables = item["context"]
    elif isinstance(item.get("variables"), dict):
        export_variables = item["variables"]
    else:
        export_variables = {}

    _dialog_ctx_json = (ann.get("export_context_json") or "").strip()
    _merged_ctx = _merge_export_context(
        export_variables, _global_ctx_json, _dialog_ctx_json
    )

    export_item = {
        "dialog_id": dialog_id,
        "scenario_id": item["scenario_id"],
        "goals": [final_goal] if final_goal != "" else [""],
        "history": history_for_export,
        "full_response": last_full_response,
        "context": _merged_ctx,
    }
    export_data.append(export_item)

if export_data:
    st.write(f"✅ Готово: {len(export_data)} диалогов")
    custom_count = sum(
        1
        for item in selected_dialogs
        if st.session_state["annotations"][item["dialog_id"]].get(
            "custom_history_enabled", False
        )
    )
    full_count = len(export_data) - custom_count
    st.info(
        f"📊 Статистика: {custom_count} с кастомной длиной, {full_count} с глобальной/полной"
    )
    jsonl_content = "\n".join(
        json.dumps(d, ensure_ascii=False, separators=(",", ":")) for d in export_data
    )
    render_export_jsonl_actions(
        jsonl_content,
        key_prefix="markup01_export",
        case_count=len(export_data),
        file_name="labeled_dialogs.jsonl",
        description="Размеченные диалоги (страница «Разметка»)",
        name_placeholder="labeled-dialogs-v1",
    )
else:
    st.info("Нет диалогов для экспорта. Проверьте фильтры и отметки.")
