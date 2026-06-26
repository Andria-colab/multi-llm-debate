"""Programmatic verifiers for the Dataset workstream (P2).

Public entry point ``verify(problem, answer)`` dispatches on ``problem.answer_type`` to a
per-type helper. Each helper normalizes BOTH the model's free-text ``answer`` and the
problem's ``ground_truth`` into a canonical form, then compares them.

Hard contract (consumed by Eval on raw model output): ``verify`` MUST NEVER raise. Every
helper body is wrapped in ``try/except Exception -> return False`` so unparseable garbage
scores as wrong rather than crashing the scorer.

Parsers accept the exact shapes that ``prompts.answer_format_hint`` asks the model for
(``{2, 3, 5}`` sets, ``(2, 3)`` tuples, ``3/8`` rationals, ``n*(n+1)/2`` expressions) but
are deliberately liberal: stray units, words, commas, ``$``/LaTeX ``\\frac{a}{b}`` and
``x=`` labels are stripped before comparison.
"""

from __future__ import annotations

import re
from typing import Any, Callable

from sympy import Rational, simplify, sympify
from sympy.parsing.sympy_parser import (
    implicit_multiplication_application,
    standard_transformations,
)

from ..interfaces import AnswerType, Problem

_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)


# --------------------------------------------------------------------------- #
# Shared normalization helpers
# --------------------------------------------------------------------------- #
def _strip_latex(s: str) -> str:
    """Turn common LaTeX/decoration into plain math: ``\\frac{a}{b}`` -> ``(a)/(b)``."""
    s = s.replace("$", "").replace("\\left", "").replace("\\right", "")
    s = re.sub(r"\\[dt]?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"((\1)/(\2))", s)
    s = s.replace("\\cdot", "*").replace("\\times", "*")
    return s


def _strip_thousands(s: str) -> str:
    """Drop thousands separators (commas between digits) without touching list commas."""
    return re.sub(r"(?<=\d),(?=\d)", "", s)


def _clean_numeric(s: str) -> str:
    """De-LaTeX, strip thousands separators, and trim surrounding whitespace."""
    return _strip_thousands(_strip_latex(s)).strip()


def _unwrap_brackets(s: str) -> str:
    """Remove one matching layer of surrounding ()/[]/{} if present."""
    s = s.strip()
    if len(s) >= 2 and s[0] in "([{" and s[-1] in ")]}":
        return s[1:-1].strip()
    return s


def _canon_elem(p: str) -> Any:
    """Canonicalize a set/tuple element: a sympy number if parseable, else lowered text."""
    p = p.strip()
    if "=" in p:  # accept 'x=2' labels -> keep only the value
        p = p.split("=", 1)[1].strip()
    try:
        return sympify(_strip_latex(p), rational=True)
    except Exception:
        return p.lower()


# --------------------------------------------------------------------------- #
# Per-AnswerType helpers (each: never raises, returns bool)
# --------------------------------------------------------------------------- #
def _parse_int(s: str) -> int:
    s = _clean_numeric(s)
    try:
        return int(s)
    except ValueError:
        m = re.search(r"[-+]?\d+", s)
        if m is None:
            raise ValueError(f"no integer in {s!r}")
        return int(m.group())


def _verify_integer(problem: Problem, answer: str) -> bool:
    try:
        return _parse_int(answer) == _parse_int(problem.ground_truth)
    except Exception:
        return False


def _parse_float(s: str) -> float:
    s = _clean_numeric(s)
    try:
        return float(s)
    except ValueError:
        m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
        if m is None:
            raise ValueError(f"no number in {s!r}")
        return float(m.group())


def _verify_real(problem: Problem, answer: str) -> bool:
    try:
        a = _parse_float(answer)
        b = _parse_float(problem.ground_truth)
        return abs(a - b) <= problem.rel_tol * max(1.0, abs(b))
    except Exception:
        return False


def _verify_rational(problem: Problem, answer: str) -> bool:
    try:
        a = Rational(sympify(_clean_numeric(answer), rational=True))
        b = Rational(sympify(_clean_numeric(problem.ground_truth), rational=True))
        return a == b
    except Exception:
        return False


def _verify_symbolic(problem: Problem, answer: str) -> bool:
    try:
        a = _parse_expr(answer)
        b = _parse_expr(problem.ground_truth)
        if simplify(a - b) == 0:
            return True
        return bool((a - b).equals(0))  # fallback for forms simplify leaves un-reduced
    except Exception:
        return False


def _parse_expr(s: str) -> Any:
    from sympy.parsing.sympy_parser import parse_expr

    return parse_expr(_strip_thousands(_strip_latex(s)).strip(), transformations=_TRANSFORMS)


def _verify_set(problem: Problem, answer: str) -> bool:
    try:
        return _parse_set(answer) == _parse_set(problem.ground_truth)
    except Exception:
        return False


def _parse_set(s: str) -> frozenset[Any]:
    body = _unwrap_brackets(_strip_latex(s).strip())
    return frozenset(_canon_elem(p) for p in body.split(",") if p.strip())


def _verify_tuple(problem: Problem, answer: str) -> bool:
    try:
        return _parse_tuple(answer) == _parse_tuple(problem.ground_truth)
    except Exception:
        return False


def _parse_tuple(s: str) -> tuple[Any, ...]:
    body = _unwrap_brackets(_strip_latex(s).strip())
    return tuple(_canon_elem(p) for p in body.split(",") if p.strip())


def _verify_string(problem: Problem, answer: str) -> bool:
    try:
        return answer.strip().casefold() == problem.ground_truth.strip().casefold()
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Dispatcher
# --------------------------------------------------------------------------- #
_DISPATCH: dict[AnswerType, Callable[[Problem, str], bool]] = {
    AnswerType.INTEGER: _verify_integer,
    AnswerType.RATIONAL: _verify_rational,
    AnswerType.REAL: _verify_real,
    AnswerType.SYMBOLIC: _verify_symbolic,
    AnswerType.SET: _verify_set,
    AnswerType.TUPLE: _verify_tuple,
    AnswerType.STRING: _verify_string,
}


def verify(problem: Problem, answer: str) -> bool:
    """Is ``answer`` correct for ``problem``? Dispatches on ``answer_type``; never raises."""
    try:
        if not isinstance(answer, str):
            return False
        helper = _DISPATCH.get(problem.answer_type)
        if helper is None:
            return False
        return helper(problem, answer)
    except Exception:
        return False
