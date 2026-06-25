"""Cache helper tests: key stability/sensitivity and round-trip hit/miss. Fully offline."""

from __future__ import annotations

from pathlib import Path

import pytest

from debate.cache import cache_get, cache_put, make_key
from debate.interfaces import Solution


def _key(**overrides) -> str:
    base = dict(
        problem_id="p1",
        stage="stage1",
        agent_id="formalist",
        model="gemini-2.5-flash",
        temperature=0.7,
        seed=42,
        thinking_budget=-1,
        system_prompt="you are a solver",
        user_prompt="solve this",
        schema_name="Solution",
    )
    base.update(overrides)
    return make_key(**base)


def test_key_is_deterministic() -> None:
    assert _key() == _key()


@pytest.mark.parametrize(
    "field,value",
    [
        ("seed", 43),
        ("temperature", 0.71),
        ("user_prompt", "solve that"),
        ("system_prompt", "you are a judge"),
        ("schema_name", "Review"),
        ("thinking_budget", 0),
        ("model", "gemini-2.5-pro"),
        ("agent_id", "lateral"),
    ],
)
def test_key_changes_when_any_input_changes(field: str, value) -> None:
    assert _key() != _key(**{field: value})


def test_put_then_get_round_trip(tmp_path: Path) -> None:
    key = _key()
    obj = Solution(solver_id="formalist", reasoning="r", answer="4", confidence=0.9)
    usage = {"prompt_tokens": 10, "output_tokens": 20, "thinking_tokens": 5, "total_tokens": 35}
    cache_put(key, obj, usage, problem_id="p1", stage="stage1", agent_id="formalist", root=tmp_path)

    hit = cache_get(key, Solution, problem_id="p1", stage="stage1", agent_id="formalist", root=tmp_path)
    assert hit is not None
    got, got_usage = hit
    assert got.model_dump() == obj.model_dump()
    assert got_usage == usage


def test_miss_returns_none(tmp_path: Path) -> None:
    assert cache_get(_key(), Solution, problem_id="p1", stage="stage1",
                     agent_id="formalist", root=tmp_path) is None


def test_wrong_key_is_a_miss(tmp_path: Path) -> None:
    """A stored entry whose full key differs (prefix collision) is treated as a miss."""
    obj = Solution(solver_id="a", reasoning="r", answer="4", confidence=0.9)
    cache_put(_key(), obj, {}, problem_id="p1", stage="stage1", agent_id="formalist", root=tmp_path)
    assert cache_get(_key(seed=999), Solution, problem_id="p1", stage="stage1",
                     agent_id="formalist", root=tmp_path) is None
