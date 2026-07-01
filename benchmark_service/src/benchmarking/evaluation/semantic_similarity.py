"""Семантическое сходство текстов через API эмбеддингов."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from clustering.legacy import get_embeddings_batch


def embedding_cosine_similarity(text_a: str, text_b: str) -> Tuple[float, bool]:
    """Считает косинусное сходство эмбеддингов двух строк.

    Аргументы:
        text_a: Первый текст.
        text_b: Второй текст.

    Возвращает:
        Кортеж ``(similarity, ok)``. ``ok=False`` означает пустой текст,
        ошибку эмбеддинга или некорректное числовое значение.
    """

    a = (text_a or "").strip()
    b = (text_b or "").strip()
    if not a or not b:
        return 0.0, False

    embs = get_embeddings_batch([a, b])
    if embs[0] is None or embs[1] is None:
        return 0.0, False

    sim = float(cosine_similarity([embs[0]], [embs[1]])[0, 0])
    if np.isnan(sim):
        return 0.0, False
    return sim, True
