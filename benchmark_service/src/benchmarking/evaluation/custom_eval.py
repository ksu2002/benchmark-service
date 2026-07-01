"""Безопасное выполнение пользовательских критериев оценки."""

from __future__ import annotations

import ast
from typing import Any, Callable, Dict, FrozenSet, Mapping, Optional, Set, Tuple, Union

_ALLOWED_BUILTINS: FrozenSet[str] = frozenset(
    {"len", "any", "all", "str", "bool", "isinstance", "int", "float"}
)

_ALLOWED_NODES: Tuple[type, ...] = (
    ast.Expression,
    ast.BoolOp,
    ast.BinOp,
    ast.UnaryOp,
    ast.Compare,
    ast.Call,
    ast.Constant,
    ast.Name,
    ast.Load,
    ast.Attribute,
    ast.Subscript,
    ast.List,
    ast.Tuple,
    ast.Dict,
    ast.Set,
    ast.Slice,
    ast.IfExp,
)


_ALLOWED_OPERATORS: Tuple[type, ...] = (
    ast.In,
    ast.NotIn,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.Is,
    ast.IsNot,
    ast.And,
    ast.Or,
    ast.Not,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Mod,
)


def _validate_ast(node: ast.AST, *, allowed_names: Set[str]) -> None:
    if isinstance(node, _ALLOWED_OPERATORS):
        return
    if not isinstance(node, _ALLOWED_NODES):
        raise ValueError(f"Недопустимая конструкция: {type(node).__name__}")
    if isinstance(node, ast.Name):
        if node.id not in allowed_names:
            raise ValueError(f"Недопустимое имя: {node.id!r}")
        return
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name):
            if node.func.id not in _ALLOWED_BUILTINS:
                raise ValueError("Разрешены только вызовы len/any/all/str/bool/isinstance/int/float")
        elif isinstance(node.func, ast.Attribute):
            _validate_ast(node.func.value, allowed_names=allowed_names)
        else:
            raise ValueError(f"Недопустимый вызов: {type(node.func).__name__}")
    for child in ast.iter_child_nodes(node):
        _validate_ast(child, allowed_names=allowed_names)


def evaluate_custom_eval_code(
    code: str,
    context: Mapping[str, Any],
    *,
    default: bool = False,
) -> bool:
    """Вычисляет bool-выражение критерия в ограниченном окружении.

    Аргументы:
        code: Python-выражение, возвращающее truthy/falsey.
        context: Переменные, доступные в выражении.
        default: Значение при пустом коде или ошибке.

    Возвращает:
        Результат выражения как bool.
    """

    expr = (code or "").strip()
    if not expr:
        return default
    allowed_names = set(context.keys()) | _ALLOWED_BUILTINS
    tree = ast.parse(expr, mode="eval")
    _validate_ast(tree.body, allowed_names=allowed_names)
    safe_builtins: Dict[str, Callable[..., Any]] = {
        "len": len,
        "any": any,
        "all": all,
        "str": str,
        "bool": bool,
        "isinstance": isinstance,
        "int": int,
        "float": float,
    }
    result = eval(compile(tree, "<custom_eval>", "eval"), {"__builtins__": safe_builtins}, dict(context))
    return bool(result)


def build_custom_eval_context(
    *,
    goals: str,
    response: Any,
    eval_value: Any,
    history: Any,
    context: Any,
) -> Dict[str, Any]:
    """Стандартный набор переменных для custom_eval_code."""

    return {
        "goals": goals,
        "response": response,
        "eval_value": eval_value,
        "history": history,
        "context": context,
        "len": len,
        "any": any,
        "all": all,
        "str": str,
        "bool": bool,
        "isinstance": isinstance,
    }
