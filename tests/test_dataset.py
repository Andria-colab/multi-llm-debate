"""Dataset load tests — exercises load_problems() against the real problems.json."""

from __future__ import annotations

import pytest

from debate.dataset.problems import load_problems
from debate.interfaces import AnswerType, Category, Problem


@pytest.fixture(scope="module")
def problems() -> list[Problem]:
    return load_problems()


def test_load_returns_list(problems: list[Problem]) -> None:
    assert isinstance(problems, list)
    assert len(problems) > 0


def test_count_is_26(problems: list[Problem]) -> None:
    assert len(problems) == 26


def test_all_are_problem_instances(problems: list[Problem]) -> None:
    for p in problems:
        assert isinstance(p, Problem), f"{p!r} is not a Problem"


def test_category_counts(problems: list[Problem]) -> None:
    from collections import Counter

    counts = Counter(p.category for p in problems)
    assert counts[Category.COMBINATORICS] == 7
    assert counts[Category.PHYSICS] == 6
    assert counts[Category.LOGIC] == 6
    assert counts[Category.GAME_THEORY] == 7


def test_all_ids_unique(problems: list[Problem]) -> None:
    ids = [p.id for p in problems]
    assert len(set(ids)) == len(ids), f"duplicate ids: {ids}"


def test_index_0_is_simple_numeric(problems: list[Problem]) -> None:
    """load_problems()[0] is used by test_engine.py; must be INTEGER or REAL."""
    p = problems[0]
    assert p.answer_type in (AnswerType.INTEGER, AnswerType.REAL), (
        f"index 0 ({p.id}) has answer_type {p.answer_type.value}; "
        "test_engine.py expects an INTEGER or REAL problem first"
    )


def test_all_difficulties_in_range(problems: list[Problem]) -> None:
    for p in problems:
        assert 1 <= p.difficulty <= 5, f"{p.id} has difficulty {p.difficulty}"
