"""Загрузка и сохранение JSONL-выборок в Postgres + MinIO."""

from __future__ import annotations

import json
from typing import Callable, Optional

import streamlit as st

try:
    from storage.benchmark_backend import (
        delete_judge_sample,
        get_judge_sample,
        judge_storage_enabled,
        judge_storage_missing_vars,
        list_judge_samples,
        load_judge_sample_jsonl,
        save_judge_sample,
    )

    SAMPLES_STORAGE_AVAILABLE = judge_storage_enabled()
    _STORAGE_MISSING = judge_storage_missing_vars()
except ImportError:
    SAMPLES_STORAGE_AVAILABLE = False
    _STORAGE_MISSING = ["установите зависимости: psycopg2-binary, pika, minio"]
    delete_judge_sample = None  # type: ignore[assignment,misc]
    get_judge_sample = None  # type: ignore[assignment,misc]
    list_judge_samples = None  # type: ignore[assignment,misc]
    load_judge_sample_jsonl = None  # type: ignore[assignment,misc]
    save_judge_sample = None  # type: ignore[assignment,misc]


def load_sample_bytes_from_db(sample_id: str) -> tuple[bytes, dict]:
    if not SAMPLES_STORAGE_AVAILABLE or get_judge_sample is None or load_judge_sample_jsonl is None:
        raise RuntimeError("Загрузка из БД недоступна")
    row = get_judge_sample(sample_id)
    if not row:
        raise ValueError("Выборка не найдена")
    raw = load_judge_sample_jsonl(sample_id)
    return raw, dict(row)


def decode_uploaded_text(uploaded, *, support_json_array: bool = False) -> str:
    text = uploaded.getvalue().decode("utf-8", errors="replace")
    if support_json_array and uploaded.name.endswith(".json"):
        data = json.loads(text)
        if isinstance(data, list):
            return "\n".join(json.dumps(x, ensure_ascii=False) for x in data if isinstance(x, dict))
        raise ValueError("JSON должен быть массивом объектов")
    return text


def render_load_saved_samples_ui(
    *,
    key_prefix: str,
    on_load: Callable[[bytes, dict], None],
    show_delete: bool = True,
    caption: Optional[str] = None,
    select_key: Optional[str] = None,
    load_key: Optional[str] = None,
    delete_key: Optional[str] = None,
    on_after_delete: Optional[Callable[[str], None]] = None,
) -> None:
    if caption is None:
        caption = "Именованные выборки, сохранённые в Postgres + MinIO."
    st.caption(caption)

    if not SAMPLES_STORAGE_AVAILABLE:
        st.info(
            "Загрузка из БД недоступна — нужны: "
            f"**{', '.join(_STORAGE_MISSING)}**."
        )
        return

    samples = list_judge_samples() if list_judge_samples else []
    if not samples:
        st.info("В системе пока нет сохранённых выборок.")
        return

    sample_map = {s["id"]: s for s in samples}
    select_key = select_key or f"{key_prefix}_saved_sample_select"
    load_key = load_key or f"{key_prefix}_load_saved_sample"
    delete_key = delete_key or f"{key_prefix}_delete_saved_sample"

    col_widths = [3, 1, 1] if show_delete else [4, 1]
    sel_cols = st.columns(col_widths)
    with sel_cols[0]:
        picked = st.selectbox(
            "Выборка",
            options=[""] + list(sample_map.keys()),
            format_func=lambda sid: (
                "— выберите —"
                if not sid
                else (
                    f"{sample_map[sid]['name']} · {sample_map[sid]['case_count']} кейсов · "
                    f"размечено {sample_map[sid]['annotated_count']}"
                )
            ),
            key=select_key,
        )
    with sel_cols[1]:
        if st.button("Загрузить", disabled=not picked, key=load_key):
            try:
                raw, row = load_sample_bytes_from_db(picked)
                on_load(raw, row)
            except Exception as e:
                st.error(str(e))
    if show_delete and len(sel_cols) > 2 and delete_judge_sample is not None:
        with sel_cols[2]:
            if st.button("Удалить", disabled=not picked, key=delete_key):
                if delete_judge_sample(picked):
                    if on_after_delete:
                        on_after_delete(picked)
                    st.rerun()


def render_jsonl_dataset_source(
    *,
    key_prefix: str,
    on_text_loaded: Callable[[str], None],
    on_db_loaded: Optional[Callable[[bytes, dict], None]] = None,
    file_types: Optional[list[str]] = None,
    upload_label: str = "Датасет (.jsonl / .json)",
    support_json_array: bool = False,
    show_delete: bool = False,
    db_caption: Optional[str] = None,
    horizontal: bool = True,
) -> None:
    """Источник: файл или сохранённые выборки."""
    if on_db_loaded is None:
        on_db_loaded = lambda raw, _row: on_text_loaded(raw.decode("utf-8", errors="replace"))

    file_types = file_types or ["jsonl", "json", "txt"]
    src = st.radio(
        "Источник данных",
        ["Файл JSONL / JSON", "Сохранённые выборки"],
        horizontal=horizontal,
        key=f"{key_prefix}_src",
    )

    if src == "Файл JSONL / JSON":
        uploaded = st.file_uploader(
            upload_label,
            type=file_types,
            key=f"{key_prefix}_upload",
        )
        if uploaded is not None:
            st.caption(f"Файл: **{uploaded.name}** ({uploaded.size} байт)")
            if st.button("Загрузить файл", type="primary", key=f"{key_prefix}_upload_btn"):
                try:
                    text = decode_uploaded_text(
                        uploaded, support_json_array=support_json_array
                    )
                    on_text_loaded(text)
                except Exception as e:
                    st.error(str(e))
    else:
        render_load_saved_samples_ui(
            key_prefix=key_prefix,
            on_load=on_db_loaded,
            show_delete=show_delete,
            caption=db_caption,
        )


def render_save_sample_to_db(
    jsonl_content: str,
    *,
    key_prefix: str,
    case_count: int,
    description: str = "",
    disabled: bool = False,
    name_placeholder: str = "dataset-sample-v1",
    button_label: str = "Сохранить в БД",
) -> None:
    if not SAMPLES_STORAGE_AVAILABLE:
        st.caption(
            "Сохранение в БД недоступно — "
            f"нужны: **{', '.join(_STORAGE_MISSING)}**."
        )
        return

    save_name = st.text_input(
        "Название выборки",
        key=f"{key_prefix}_save_name",
        placeholder=name_placeholder,
        disabled=disabled,
    )
    if st.button(
        button_label,
        type="secondary",
        key=f"{key_prefix}_save_db",
        disabled=disabled or not jsonl_content.strip(),
    ):
        title = (save_name or "").strip()
        if not title:
            st.error("Укажите название выборки")
        elif save_judge_sample is None:
            st.error("Сохранение в БД недоступно")
        else:
            try:
                new_id = save_judge_sample(
                    title,
                    jsonl_content.encode("utf-8"),
                    description=description,
                    case_count=case_count,
                    annotated_count=0,
                    label_mode="binary_multi",
                    criteria_json=[],
                )
                st.success(f"Выборка «{title}» сохранена (id: {new_id}).")
            except Exception as e:
                st.error(str(e))


def render_export_jsonl_actions(
    jsonl_content: str,
    *,
    key_prefix: str,
    case_count: int,
    file_name: str,
    description: str = "",
    download_label: str = "Скачать JSONL",
    disabled: bool = False,
    name_placeholder: str = "dataset-sample-v1",
) -> None:
    """Сохранение в БД и скачивание JSONL в двух колонках."""
    if not jsonl_content.strip() and disabled:
        return
    cols = st.columns([2, 1])
    with cols[0]:
        render_save_sample_to_db(
            jsonl_content,
            key_prefix=key_prefix,
            case_count=case_count,
            description=description,
            disabled=disabled,
            name_placeholder=name_placeholder,
        )
    with cols[1]:
        st.markdown("<div style='height:1.75rem'></div>", unsafe_allow_html=True)
        st.download_button(
            download_label,
            jsonl_content,
            file_name,
            "application/json",
            type="primary",
            key=f"{key_prefix}_download",
            disabled=disabled or not jsonl_content.strip(),
            use_container_width=True,
        )
