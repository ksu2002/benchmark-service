"""Совместимый фасад запуска бенчмарков.

Новый код должен импортировать API исполнения бенчмарков отсюда, пока
legacy-реализация постепенно выделяется из ``benchmark_runner``.
"""

from benchmarking.config_models import (  # noqa: F401
    LLM_JUDGE_EVAL_MODE,
    SEMANTIC_SIMILARITY_EVAL_MODE,
    is_llm_judge_eval_mode,
    is_semantic_similarity_eval_mode,
)
from benchmarking.parsing.cases import normalize_parsed_case  # noqa: F401
from benchmarking.core import *  # noqa: F401,F403
from benchmarking.core import (  # noqa: F401
    _dm_substitute_in_data_structure,
    _goals_from_case,
)
