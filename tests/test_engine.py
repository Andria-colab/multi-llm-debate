"""Engine orchestration tests — the real run_debate driven offline via FakeClient.

Confirms Stages 0→4 wire together into a contract-valid DebateRecord without any network:
correct cardinality, engine-stamped identities, the winner-answer copy, cost aggregation,
and that GeminiClient refuses live calls in offline mode. No API calls.
"""

from __future__ import annotations

import pytest

from debate.client import FakeClient, GeminiClient, ModelCallError, OfflineError
from debate.config import Settings
from debate.dataset import load_problems, verify
from debate.engine import run_debate
from debate.interfaces import DebateRecord


def test_dataset_stub_loads() -> None:
    problems = load_problems()
    assert len(problems) >= 1
    assert len({p.id for p in problems}) == len(problems)  # unique ids


def test_run_debate_returns_valid_record() -> None:
    problem = load_problems()[0]
    record = run_debate(problem, client=FakeClient(answer=problem.ground_truth))
    assert isinstance(record, DebateRecord)
    # cardinality the frozen contract requires
    assert len(record.role_assessments) == 4
    assert len(record.solutions) == 3
    assert len(record.reviews) == 6
    assert len(record.refinements) == 3
    # final answer is copied from the winning refinement, not transcribed by the judge
    winner = record.judgment.winner_solver_id
    winning_refinement = next(r for r in record.refinements if r.solver_id == winner)
    assert record.final_answer == winning_refinement.refined_answer
    assert record.final_answer == record.judgment.final_answer


def test_engine_stamps_identities() -> None:
    """The model never sets *_id; the orchestrator does, consistently."""
    problem = load_problems()[0]
    record = run_debate(problem, client=FakeClient(answer="x"))

    assert {a.agent_id for a in record.role_assessments} == {
        "formalist",
        "lateral",
        "checker",
        "skeptic",
    }
    assert all(s.solver_id in record.assignment.solver_ids for s in record.solutions)
    # 6 reviews = every ordered (reviewer, target) solver pair, reviewer != target
    pairs = {(r.reviewer_id, r.target_solver_id) for r in record.reviews}
    assert len(pairs) == 6
    assert all(rid != tid for rid, tid in pairs)
    assert record.judgment.winner_solver_id in record.assignment.solver_ids


def test_truthful_answer_verifies() -> None:
    problem = load_problems()[0]
    record = run_debate(problem, client=FakeClient(answer=problem.ground_truth))
    assert verify(problem, record.final_answer) is True


def test_wrong_answer_fails_verify() -> None:
    problem = load_problems()[0]
    record = run_debate(problem, client=FakeClient(answer="__wrong__"))
    assert verify(problem, record.final_answer) is False


def test_cost_meta_counts_every_call() -> None:
    """Stage 0 ×4 + 1 ×3 + 2 ×6 + 3 ×3 + 4 ×1 = 17 calls, all recorded."""
    problem = load_problems()[0]
    record = run_debate(problem, client=FakeClient(answer="x"))
    assert len(record.cost.timings) == 17
    assert record.cost.n_api_calls + record.cost.n_cache_hits == 17


def test_seed_helpers_respect_injected_settings() -> None:
    """A custom base_seed must actually change the derived seeds (not read the global)."""
    from debate.config import per_problem_seed

    assert per_problem_seed("p1", Settings(base_seed=7)) != per_problem_seed(
        "p1", Settings(base_seed=42)
    )


def test_invalid_judge_label_fails_loud() -> None:
    """A judge label that isn't one of the presented candidates must raise, not silently
    credit whichever solver landed in the first shuffle slot."""
    problem = load_problems()[0]
    with pytest.raises(ModelCallError):
        run_debate(problem, client=FakeClient(answer="x", judge_pick_label="formalist"))


def test_offline_gemini_client_refuses_live_call() -> None:
    client = GeminiClient(Settings(offline=True))
    problem = load_problems()[0]
    with pytest.raises(OfflineError):
        run_debate(problem, client=client, settings=Settings(offline=True))
