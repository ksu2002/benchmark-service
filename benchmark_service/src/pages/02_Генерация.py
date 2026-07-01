import json
import os

import litellm
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from integrations.litellm import get_model_names
from ui.sample_storage_ui import render_export_jsonl_actions, render_jsonl_dataset_source
load_dotenv()

SUPPORTED_MODELS = get_model_names()


def get_litellm_kwargs():
    model = st.session_state.get("selected_model", SUPPORTED_MODELS[0])
    api_key = st.session_state.get("litellm_api_key", "").strip() or os.getenv(
        "LITELLM_API_KEY", None
    )
    return {
        "model": model,
        "temperature": 0.0,
        "timeout": 30,
        "api_key": api_key or None,
        "api_base": os.getenv("LITELLM_API_BASE", None) or None,
        "extra_body": {"cache": {"no-cache": True}},
    }


def normalize_role(role: str) -> str:
    """
    Преобразует любое обозначение роли к 'assistant' или 'user'.
    Поддерживает: 'assistant', 'user', 'Ассистент', 'Пользователь' и др.
    """
    role = str(role).strip().lower()
    if role in ["пользователь", "user", "клиент", "caller", "client"]:
        return "user"
    elif role in [
        "ассистент",
        "assistant",
        "оператор",
        "agent",
        "manager",
        "representative",
    ]:
        return "assistant"
    else:
        if "асс" in role or "agent" in role or role.startswith("а"):
            return "assistant"
        else:
            return "user"


import random  # добавьте в начало файла, если ещё не импортирован


def generate_dialog_for_goal(
    goal: str, example_dialogs: list = None, custom_prompt_template: str = ""
) -> dict:
    litellm_kwargs = get_litellm_kwargs()

    examples_text = ""
    if example_dialogs:
        # Выбираем до 20 случайных диалогов
        sample_size = min(20, len(example_dialogs))
        selected_examples = random.sample(example_dialogs, sample_size)

        examples_text = "\nПримеры реальных диалогов по этой цели:\n"
        for ex in selected_examples:
            for turn in ex["history"]:
                role = "Ассистент" if turn["role"] == "assistant" else "Пользователь"
                examples_text += f"- {role}: {turn['content']}\n"

        if custom_prompt_template.strip():
            prompt = custom_prompt_template.replace("{goal}", goal).replace(
                "{examples_text}", examples_text
            )
        else:
            prompt = custom_prompt_template.replace("{goal}", goal)

    try:
        response = litellm.completion(
            messages=[{"role": "user", "content": prompt}], **litellm_kwargs
        )
        raw_text = response.choices[0].message.content.strip()

        if raw_text.startswith("```json"):
            raw_text = raw_text.split("```json", 1)[1].split("```")[0]
        elif raw_text.startswith("```"):
            raw_text = raw_text.split("```", 1)[1].split("```")[0]
        print(raw_text)
        parsed = json.loads(raw_text)
        normalized_history = []
        print(parsed["history"])
        for turn in parsed["history"]:
            normalized_history.append(
                {"role": normalize_role(turn["role"]), "content": turn["content"]}
            )
        return {"goals": [goal], "history": normalized_history}
    except Exception as e:
        st.warning(f"Ошибка загрузите примеры диалогов '{goal}': {str(e)[:100]}")
        return {
            "goals": [goal],
            "history": [
                {"role": "assistant", "content": "Привет! Чем могу помочь?"},
                {"role": "user", "content": "[Ошибка генерации]"},
            ],
        }


def load_dialogs_from_text(text: str) -> list:
    dialogs = []
    for line in text.strip().split("\n"):
        if line.strip() and not line.strip().startswith("#"):
            dialogs.append(json.loads(line))
    return dialogs


def load_dialogs_from_file(uploaded_file) -> list:
    bytes_data = uploaded_file.getvalue()
    text = bytes_data.decode("utf-8")

    dialogs = []
    if uploaded_file.name.endswith(".jsonl"):
        for line in text.strip().split("\n"):
            if line.strip():
                dialogs.append(json.loads(line))
    elif uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        for _, row in df.iterrows():
            goals = json.loads(row["goals"])
            history = json.loads(row["history"])
            dialogs.append({"goals": goals, "history": history})
    else:
        raise ValueError("Поддерживаются только .jsonl и .csv")
    return dialogs


# --- UI ---
st.set_page_config(page_title="Генерация новых диалогов", layout="wide")
st.title("✨ Генерация новых диалогов через LiteLLM")
st.sidebar.subheader("🔑 Настройки LLM")
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

st.subheader("1. Укажите цели для генерации")
goals_input = st.text_input(
    "Цели (через запятую)",
    value="Консультация, Вопрос по залоговому билету, Перевод на оператора",
    key="gen_goals",
)
goals_list = [g.strip() for g in goals_input.split(",") if g.strip()]

if not goals_list:
    st.warning("Укажите хотя бы одну цель.")
    st.stop()

st.subheader("2. Загрузите существующие диалоги (для контекста)")
ctx_src = st.radio(
    "Источник",
    ["Файл", "Сохранённые выборки"],
    horizontal=True,
    key="gen_ctx_src",
)

existing_dialogs = list(st.session_state.get("gen_existing_dialogs") or [])

if ctx_src == "Файл":
    uploaded_file = st.file_uploader(
        "Загрузите JSONL или CSV с диалогами", type=["jsonl", "csv"]
    )
    if uploaded_file:
        try:
            existing_dialogs = load_dialogs_from_file(uploaded_file)
            st.session_state["gen_existing_dialogs"] = existing_dialogs
            st.success(f"Загружено {len(existing_dialogs)} диалогов.")
        except Exception as e:
            st.error(f"Ошибка загрузки: {e}")
else:
    _gen_db_msg = st.session_state.pop("_gen_ctx_db_loaded_msg", None)
    if _gen_db_msg:
        st.success(_gen_db_msg)

    def _on_gen_ctx_text(text: str) -> None:
        try:
            dialogs = load_dialogs_from_text(text)
            st.session_state["gen_existing_dialogs"] = dialogs
            st.session_state["_gen_ctx_db_loaded_msg"] = (
                f"Загружено **{len(dialogs)}** диалогов из выборки."
            )
            st.rerun()
        except Exception as e:
            st.error(str(e))

    render_jsonl_dataset_source(
        key_prefix="gen_ctx",
        on_text_loaded=_on_gen_ctx_text,
        file_types=["jsonl"],
        upload_label="JSONL с диалогами",
    )
    existing_dialogs = list(st.session_state.get("gen_existing_dialogs") or [])

from collections import defaultdict

existing_by_goal = defaultdict(list)
for d in existing_dialogs:
    goal = d["goals"][0] if d["goals"] else "Без цели"
    existing_by_goal[goal].append(d)

st.subheader("3. Настройки генерации")
num_per_goal = st.number_input(
    "Диалогов на каждую цель", min_value=1, max_value=20, value=3
)

custom_prompt = st.text_area(
    "Промпт (используйте {goal} и {examples_text})",
    value="""Ты — симулятор пользователя.
Сгенерируй **один короткий и реалистичный диалог** по цели: "{goal}".

{examples_text}

Правила:
- Диалог начинается с реплики ассистента (приветствие).
- Пользователь отвечает, уточняет, задаёт вопросы.
- Диалог содержит столько же реплик сколько и в примере.
- Формат вывода: только JSON в виде:
{"history": [{"role": "...", "content": "..."}], ...}

Не добавляй пояснений, только JSON.""",
    height=200,
    key="custom_prompt_input",
)
st.caption("💡 Переменные: `{goal}` — цель, `{examples_text}` — примеры диалогов")

if st.button("🚀 Сгенерировать диалоги"):
    with st.spinner("Генерация через LiteLLM..."):
        generated = []
        for goal in goals_list:
            examples = existing_by_goal.get(goal, [])
            for i in range(num_per_goal):
                dialog = generate_dialog_for_goal(goal, examples, custom_prompt)
                dialog["source"] = "generated"
                generated.append(dialog)

        st.session_state["generated_dialogs"] = generated
        st.session_state["all_dialogs_for_review"] = existing_dialogs + generated

reviewed = []
if "all_dialogs_for_review" in st.session_state:
    if "generation_goals_list" not in st.session_state:
        st.session_state["generation_goals_list"] = goals_list

    available_goals = st.session_state["generation_goals_list"]
    goal_options = available_goals
    all_dialogs = st.session_state["all_dialogs_for_review"]

    st.subheader("🔍 Проверка и переразметка")

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Выбрать все"):
            for i in range(len(all_dialogs)):
                st.session_state[f"include_{i}"] = True
    with col2:
        if st.button("Снять все"):
            for i in range(len(all_dialogs)):
                st.session_state[f"include_{i}"] = False
    with col3:
        if st.button("Инвертировать"):
            for i in range(len(all_dialogs)):
                st.session_state[f"include_{i}"] = not st.session_state.get(
                    f"include_{i}", True
                )

    st.markdown("---")

    reviewed = []
    included_count = 0

    for idx, dialog in enumerate(all_dialogs):
        include_key = f"include_{idx}"
        hist_len_key = f"hist_len_{idx}"

        if include_key not in st.session_state:
            st.session_state[include_key] = True
        if hist_len_key not in st.session_state:
            st.session_state[hist_len_key] = len(dialog["history"])

        current_goal = dialog["goals"][0] if dialog["goals"] else "Без цели"
        if current_goal not in goal_options:
            goal_options = goal_options + [current_goal]

        is_generated = dialog.get("source") == "generated"
        source_tag = "🟢 Сгенерированный" if is_generated else "🔵 Исходный"

        with st.container():
            col_check, col_goal, col_len, col_content = st.columns([0.5, 1.5, 0.8, 4])

            with col_check:
                include = st.checkbox(
                    "Включить", value=st.session_state[include_key], key=f"cb_{idx}"
                )
                st.session_state[include_key] = include

            with col_goal:
                selected_goal = st.selectbox(
                    "Цель",
                    options=goal_options,
                    index=(
                        goal_options.index(current_goal)
                        if current_goal in goal_options
                        else 0
                    ),
                    key=f"sel_goal_{idx}",
                    label_visibility="collapsed",
                )

            with col_len:
                max_len = len(dialog["history"])
                hist_len = st.number_input(
                    "Реплик",
                    min_value=1,
                    max_value=max_len,
                    value=st.session_state.get(hist_len_key, max_len),
                    key=f"num_hist_{idx}",
                    label_visibility="visible",
                )
                st.session_state[hist_len_key] = hist_len

            with col_content:
                dialog_lines = []
                for turn in dialog["history"]:
                    role_label = (
                        "👤 Пользователь" if turn["role"] == "user" else "💼 Ассистент"
                    )
                    dialog_lines.append(f"{role_label}: {turn['content']}")
                st.text("\n".join(dialog_lines))
                st.caption(f"_{source_tag}_")

        st.markdown("---")

        if include:
            final_history = dialog["history"][:hist_len]
            reviewed.append(
                {
                    "goals": [selected_goal] if selected_goal != "Без цели" else [],
                    "history": final_history,
                }
            )
            included_count += 1

    st.success(f"✅ Будет сохранено: {included_count} диалогов из {len(all_dialogs)}")


# === 6. Экспорт ===
st.subheader("📤 Сохранить отмеченные диалоги")
if reviewed:
    jsonl_lines = []
    for d in reviewed:
        jsonl_lines.append(json.dumps(d, ensure_ascii=False, separators=(",", ":")))
    jsonl_content = "\n".join(jsonl_lines)

    render_export_jsonl_actions(
        jsonl_content,
        key_prefix="gen_export",
        case_count=len(reviewed),
        file_name="filtered_dialogs.jsonl",
        description="Отмеченные диалоги со страницы «Генерация»",
        name_placeholder="generation-sample-v1",
    )
else:
    st.info("Нет диалогов, отмеченных для сохранения.")
