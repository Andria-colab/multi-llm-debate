"""Unit tests for the metrics layer — pure functions over hand-built records, fully offline.

Covers accuracy (overall + by category/difficulty), cost/latency aggregation, the
verify-backed answer-equivalence helper, and the debate-dynamics stats.
"""

from __future__ import annotations

import pytest

import debate.eval.metrics as metrics_mod
from debate.client import FakeClient
from debate.dataset import load_problems
from debate.engine import run_debate
from debate.eval.metrics import (
    MethodResult,
    accuracy,
    accuracy_by_category,
    accuracy_by_difficulty,
    aggregate_cost,
    answers_equivalent,
    build_report,
    compute_dynamics,
    compute_method_metrics,
)
from debate.interfaces import AnswerType, Category, CostMeta, Problem, StageTiming


def _problem(
    pid: str,
    *,
    category: Category = Category.LOGIC,
    difficulty: int = 1,
    answer_type: AnswerType = AnswerType.INTEGER,
    ground_truth: str = "1",
    rel_tol: float = 1e-6,
) -> Problem:
    return Problem(
        id=pid,
        text="synthetic",
        category=category,
        ground_truth=ground_truth,
        answer_type=answer_type,
        difficulty=difficulty,
        rel_tol=rel_tol,
    )


def _result(
    method: str,
    *,
    correct: bool,
    category: Category = Category.LOGIC,
    difficulty: int = 1,
    cost: CostMeta | None = None,
) -> MethodResult:
    p = _problem(f"{method}_{category.value}_{difficulty}", category=category, difficulty=difficulty)
    return MethodResult(
        method=method,
        problem=p,
        answer="1" if correct else "0",
        is_correct=correct,
        cost=cost or CostMeta(),
    )


# --------------------------------------------------------------------------- #
# Accuracy
# --------------------------------------------------------------------------- #
def test_accuracy_overall() -> None:
    results = [_result("single", correct=True), _result("single", correct=False)]
    assert accuracy(results) == 0.5


def test_accuracy_empty_is_zero_not_error() -> None:
    assert accuracy([]) == 0.0


def test_none_counts_as_incorrect() -> None:
    r = _result("single", correct=True)
    r.is_correct = None
    assert accuracy([r]) == 0.0


def test_accuracy_by_category_and_difficulty() -> None:
    results = [
        _result("single", correct=True, category=Category.PHYSICS, difficulty=1),
        _result("single", correct=False, category=Category.PHYSICS, difficulty=5),
        _result("single", correct=True, category=Category.LOGIC, difficulty=1),
    ]
    by_cat = accuracy_by_category(results)
    assert by_cat["physics"] == 0.5
    assert by_cat["logic"] == 1.0
    by_diff = accuracy_by_difficulty(results)
    assert by_diff["1"] == 1.0  # both difficulty-1 problems correct
    assert by_diff["5"] == 0.0


# --------------------------------------------------------------------------- #
# Cost / latency aggregation
# --------------------------------------------------------------------------- #
def test_aggregate_cost_sums_and_counts() -> None:
    timings = [
        StageTiming(stage="s", agent_id="a", latency_s=1.0, prompt_tokens=10, output_tokens=5,
                    thinking_tokens=20, total_tokens=35, cache_hit=False),
        StageTiming(stage="s", agent_id="b", latency_s=0.0, prompt_tokens=10, output_tokens=5,
                    thinking_tokens=20, total_tokens=35, cache_hit=True),
    ]
    cost = aggregate_cost(timings)
    assert cost.total_tokens == 70
    assert cost.total_thinking_tokens == 40
    assert cost.total_latency_s == 1.0
    assert cost.n_api_calls == 1
    assert cost.n_cache_hits == 1


def test_method_metrics_means() -> None:
    cost = CostMeta(total_tokens=100, total_thinking_tokens=60, total_latency_s=2.0, n_api_calls=1)
    results = [
        _result("debate", correct=True, cost=cost),
        _result("debate", correct=False, cost=cost),
    ]
    m = compute_method_metrics("debate", results)
    assert m.n == 2
    assert m.n_correct == 1
    assert m.accuracy == 0.5
    assert m.total_tokens == 200
    assert m.mean_tokens == 100
    assert m.mean_thinking_tokens == 60
    assert m.mean_latency_s == 2.0
    assert m.total_api_calls == 2


def test_method_metrics_empty_is_zeroed() -> None:
    m = compute_method_metrics("single", [])
    assert m.n == 0
    assert m.accuracy == 0.0
    assert m.mean_tokens == 0.0


# --------------------------------------------------------------------------- #
# Answer equivalence (verify-backed)
# --------------------------------------------------------------------------- #
def test_answers_equivalent_normalizes() -> None:
    p = _problem("real", answer_type=AnswerType.REAL, ground_truth="14.7", rel_tol=1e-2)
    assert answers_equivalent(p, "14.7", "14.70") is True
    assert answers_equivalent(p, "14.7", "99.0") is False


def test_answers_equivalent_is_reflexive_even_if_verifier_isnt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Simulate the real verifier's non-reflexivity on unparseable answers: verify always False.
    monkeypatch.setattr(metrics_mod, "verify", lambda problem, answer: False)
    p = _problem("p")
    # byte-identical (even unparseable) -> True via the fast path, never consulting verify
    assert answers_equivalent(p, "infinitely many", "infinitely many") is True
    # genuinely different -> falls through to the (stubbed) verifier
    assert answers_equivalent(p, "a", "b") is False


# --------------------------------------------------------------------------- #
# Debate dynamics
# --------------------------------------------------------------------------- #
def test_dynamics_over_fake_debate() -> None:
    problem = load_problems()[0]
    record = run_debate(problem, client=FakeClient(answer="5"))
    dyn = compute_dynamics([record])
    assert dyn.n_records == 1
    # FakeClient never changes its answer and rebuts every critique
    assert dyn.answer_change_rate == 0.0
    assert dyn.critique_acceptance_rate == 0.0
    assert dyn.mean_critiques_per_review == 1.0  # one located error per review
    assert sum(dyn.severity_counts.values()) == 6  # 6 reviews x 1 error


def test_dynamics_unchanged_answer_not_counted_if_verifier_nonreflexive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Even with a non-reflexive verifier, a solver that kept its byte-identical Stage-1 answer
    # must NOT be counted as having changed it (the answers_equivalent fast path guarantees it).
    monkeypatch.setattr(metrics_mod, "verify", lambda problem, answer: False)
    problem = load_problems()[0]
    record = run_debate(problem, client=FakeClient(answer="5"))  # refined == stage1 == "5"
    assert compute_dynamics([record]).answer_change_rate == 0.0


def test_dynamics_detects_change_and_acceptance() -> None:
    problem = load_problems()[0]
    record = run_debate(problem, client=FakeClient(answer="5"))
    # Simulate one solver changing its answer and accepting its critique.
    record.refinements[0].refined_answer = "999"
    record.refinements[0].changes_made[0].accepted = True
    dyn = compute_dynamics([record])
    assert dyn.answer_change_rate == 1 / 3
    assert dyn.critique_acceptance_rate == 1 / 3


def test_build_report_orders_methods_and_skips_empty() -> None:
    results_by_method = {
        "debate": [_result("debate", correct=True)],
        "single": [_result("single", correct=True)],
        "voting": [],  # present but empty -> skipped
    }
    report = build_report(results_by_method, [])
    assert [m.method for m in report.methods] == ["single", "debate"]  # METHODS order, voting dropped
    assert report.dynamics is None  # no records supplied
