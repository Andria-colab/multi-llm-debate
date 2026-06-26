"""Verifier tests — one synthetic Problem per AnswerType, no JSON loaded.

Each AnswerType is exercised four ways: canonical-correct, messy-but-valid (the noisy forms
the model actually emits per ``prompts.answer_format_hint``), wrong, and garbage. Plus
explicit regression tests for the documented parser trade-offs. ``verify`` must NEVER raise,
so the garbage set includes ``None`` passed where a ``str`` is expected.
"""

from __future__ import annotations

import pytest

from debate.dataset.verifiers import verify
from debate.interfaces import AnswerType, Category, Problem


def make_problem(answer_type: AnswerType, ground_truth: str, *, rel_tol: float = 1e-6) -> Problem:
    """A minimal valid Problem with the given answer_type + ground_truth."""
    return Problem(
        id=f"test_{answer_type.value}",
        text="synthetic test problem",
        category=Category.LOGIC,
        ground_truth=ground_truth,
        answer_type=answer_type,
        difficulty=1,
        rel_tol=rel_tol,
    )


# Garbage that must score False for EVERY type without raising. None is passed deliberately
# to confirm verify() tolerates a non-str answer (Eval feeds it raw model output).
COMMON_GARBAGE: list[object] = [
    "",
    "   ",
    "abc",
    "(1,2",  # unbalanced paren
    "{1,2",  # unbalanced brace
    "½",  # unicode fraction glyph (not parsed as 1/2)
    "x" * 5000,  # pathologically long string
    None,  # non-str: must return False, not raise
]


# (answer_type, ground_truth, canonical, [messy-valid...], wrong, rel_tol)
CASES = [
    (AnswerType.INTEGER, "42", "42", ["The answer is 42.", " 42 "], "43", 1e-6),
    (AnswerType.REAL, "9.81", "9.81", ["9.81 m/s^2", "9.810"], "9.0", 1e-3),
    (AnswerType.RATIONAL, "3/8", "3/8", ["\\frac{3}{8}", "0.375"], "3/7", 1e-6),
    (AnswerType.SYMBOLIC, "n*(n+1)/2", "n*(n+1)/2", ["n(n+1)/2", "n*(n + 1)/ 2"], "n**2", 1e-6),
    (AnswerType.SET, "{2, 3, 5}", "{2, 3, 5}", ["{2, 3, 5}", " { 5,3,2 } "], "{2, 3}", 1e-6),
    (AnswerType.TUPLE, "(2, 3)", "(2, 3)", ["(2, 3)", "x=2, y=3"], "(3, 2)", 1e-6),
    (AnswerType.STRING, "Yes", "Yes", ["  Yes  ", "YES"], "No", 1e-6),
]


@pytest.mark.parametrize(
    "answer_type, ground_truth, canonical, messy, wrong, rel_tol",
    CASES,
    ids=[c[0].value for c in CASES],
)
def test_canonical_correct(
    answer_type: AnswerType,
    ground_truth: str,
    canonical: str,
    messy: list[str],
    wrong: str,
    rel_tol: float,
) -> None:
    problem = make_problem(answer_type, ground_truth, rel_tol=rel_tol)
    assert verify(problem, canonical) is True


@pytest.mark.parametrize(
    "answer_type, ground_truth, canonical, messy, wrong, rel_tol",
    CASES,
    ids=[c[0].value for c in CASES],
)
def test_messy_valid_correct(
    answer_type: AnswerType,
    ground_truth: str,
    canonical: str,
    messy: list[str],
    wrong: str,
    rel_tol: float,
) -> None:
    problem = make_problem(answer_type, ground_truth, rel_tol=rel_tol)
    for form in messy:
        assert verify(problem, form) is True, f"messy form {form!r} should verify True"


@pytest.mark.parametrize(
    "answer_type, ground_truth, canonical, messy, wrong, rel_tol",
    CASES,
    ids=[c[0].value for c in CASES],
)
def test_wrong_answer(
    answer_type: AnswerType,
    ground_truth: str,
    canonical: str,
    messy: list[str],
    wrong: str,
    rel_tol: float,
) -> None:
    problem = make_problem(answer_type, ground_truth, rel_tol=rel_tol)
    assert verify(problem, wrong) is False


@pytest.mark.parametrize(
    "answer_type, ground_truth, canonical, messy, wrong, rel_tol",
    CASES,
    ids=[c[0].value for c in CASES],
)
def test_garbage_is_false_never_raises(
    answer_type: AnswerType,
    ground_truth: str,
    canonical: str,
    messy: list[str],
    wrong: str,
    rel_tol: float,
) -> None:
    problem = make_problem(answer_type, ground_truth, rel_tol=rel_tol)
    for junk in COMMON_GARBAGE:
        result = verify(problem, junk)  # type: ignore[arg-type]
        assert result is False, f"garbage {junk!r} should verify False, got {result!r}"


# "42 apples" is a WRONG numeric answer for a non-42 ground truth: confirm the unit-tolerant
# parser doesn't accidentally accept it where the number disagrees.
def test_number_with_units_still_compared_by_value() -> None:
    assert verify(make_problem(AnswerType.INTEGER, "5"), "42 apples") is False
    assert verify(make_problem(AnswerType.REAL, "5", rel_tol=1e-3), "42 apples") is False


# --------------------------------------------------------------------------- #
# Regression tests for the documented trade-offs
# --------------------------------------------------------------------------- #
def test_integer_rejects_empty_and_no_digits() -> None:
    problem = make_problem(AnswerType.INTEGER, "42")
    assert verify(problem, "") is False
    assert verify(problem, "no digits here") is False


def test_set_subset_is_not_equal() -> None:
    problem = make_problem(AnswerType.SET, "{1, 2, 3}")
    assert verify(problem, "{1, 2}") is False
    # symmetry: ground truth smaller than the (superset) answer is also unequal
    assert verify(make_problem(AnswerType.SET, "{1, 2}"), "{1, 2, 3}") is False
