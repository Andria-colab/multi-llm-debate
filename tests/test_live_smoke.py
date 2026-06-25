"""Live smoke tests — these HIT THE REAL GEMINI API and spend quota.

Excluded from the default suite (the `live` marker). Run them explicitly:

    pytest -m live -s                 # both tests, -s shows the printed answer
    pytest tests/test_live_smoke.py -m live -k single -s   # just the cheap 1-call test

Requirements: a real GEMINI_API_KEY (from .env or the environment) and google-genai
installed. Do NOT set DEBATE_OFFLINE=1 when running these — offline mode refuses live calls.
Each test auto-skips if no real key is configured.
"""

from __future__ import annotations

import pytest

from debate.client import GeminiClient
from debate.config import SETTINGS, seed_for
from debate.dataset import load_problems, verify
from debate.engine import run_debate
from debate.interfaces import Solution
from debate.prompts import PERSONAS, stage1_solve

pytestmark = pytest.mark.live

_PLACEHOLDER = "your-google-ai-studio-api-key-here"
_HAS_KEY = bool(SETTINGS.api_key) and SETTINGS.api_key != _PLACEHOLDER
requires_key = pytest.mark.skipif(not _HAS_KEY, reason="no real GEMINI_API_KEY configured")


@requires_key
def test_single_structured_call() -> None:
    """Cheapest check: one structured call. Confirms key + SDK + structured output + usage."""
    problem = load_problems()[0]  # 2x4 domino tiling, ground truth "5"
    client = GeminiClient()
    res = client.call(
        stage="stage1_solve",
        agent_id="formalist",
        problem_id=problem.id,
        system_prompt=PERSONAS["formalist"],
        user_prompt=stage1_solve(problem),
        schema=Solution,
        temperature=0.15,
        seed=seed_for(problem.id, "formalist"),  # real derived seed (must fit INT32)
    )
    sol = res.parsed
    assert isinstance(sol, Solution)
    assert sol.reasoning.strip()
    assert sol.answer.strip()
    assert res.timing.total_tokens > 0  # usage was tracked
    print(f"\n[single-call] answer={sol.answer!r}  tokens={res.timing.total_tokens}")


@requires_key
def test_full_debate_end_to_end() -> None:
    """Exercises all of Stages 0-4 live (~17 calls) on one easy problem; checks the record."""
    problem = load_problems()[0]
    record = run_debate(problem)
    assert record.final_answer.strip()
    assert len(record.role_assessments) == 4
    assert len(record.solutions) == 3
    assert len(record.reviews) == 6
    assert len(record.refinements) == 3
    correct = verify(problem, record.final_answer)
    print(
        f"\n[full-debate] winner={record.judgment.winner_solver_id} "
        f"final={record.final_answer!r} correct={correct} "
        f"calls={record.cost.n_api_calls} cache_hits={record.cost.n_cache_hits} "
        f"tokens={record.cost.total_tokens}"
    )


if __name__ == "__main__":
    # Allow `python tests/test_live_smoke.py` (not just `pytest`): drive pytest on this file,
    # forcing the `live` marker and -s so the printed results show. Requires `pip install -e .`.
    raise SystemExit(pytest.main([__file__, "-m", "live", "-s"]))
