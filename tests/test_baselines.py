"""Baseline tests — single-call + majority-vote driven offline via FakeClient.

Confirms each baseline reuses the client (so it's cached/offline-safe), produces a scored
MethodResult with comparable cost, and that majority voting buckets answers semantically and
tie-breaks deterministically.
"""

from __future__ import annotations

import pytest

from debate.client import FakeClient
from debate.dataset import load_problems
from debate.eval.baselines import (
    DEFAULT_SINGLE_AGENT,
    majority_answer,
    run_single,
    run_voting,
)
from debate.interfaces import AnswerType, Category, Problem


def _problem(
    pid: str = "p",
    *,
    answer_type: AnswerType = AnswerType.INTEGER,
    ground_truth: str = "5",
    rel_tol: float = 1e-6,
) -> Problem:
    return Problem(
        id=pid,
        text="synthetic",
        category=Category.LOGIC,
        ground_truth=ground_truth,
        answer_type=answer_type,
        difficulty=1,
        rel_tol=rel_tol,
    )


# --------------------------------------------------------------------------- #
# Single-call baseline
# --------------------------------------------------------------------------- #
def test_single_correct_when_answer_matches() -> None:
    problem = _problem(ground_truth="5")
    client = FakeClient(answer="5")
    result = run_single(problem, client=client)
    assert result.method == "single"
    assert result.answer == "5"
    assert result.is_correct is True
    assert len(result.cost.timings) == 1  # exactly one solve call
    # reused the client under a distinct baseline stage, as the formalist persona
    assert client.calls == [("baseline_single", DEFAULT_SINGLE_AGENT)]


def test_single_wrong_when_answer_mismatches() -> None:
    result = run_single(_problem(ground_truth="5"), client=FakeClient(answer="6"))
    assert result.is_correct is False


# --------------------------------------------------------------------------- #
# Majority-vote baseline
# --------------------------------------------------------------------------- #
def test_voting_runs_n_voters_and_takes_mode() -> None:
    problem = _problem(ground_truth="5")
    client = FakeClient(answer="5")  # every voter returns the same answer
    result = run_voting(problem, client=client, n_voters=3)
    assert result.method == "voting"
    assert result.answer == "5"
    assert result.is_correct is True
    assert result.n_voters == 3
    assert result.votes == ["5", "5", "5"]
    assert len(result.cost.timings) == 3
    assert all(stage == "baseline_vote" for stage, _ in client.calls)
    assert len(client.calls) == 3


def test_voting_rejects_zero_voters() -> None:
    with pytest.raises(ValueError):
        run_voting(_problem(), client=FakeClient(answer="5"), n_voters=0)


def test_voting_more_voters_than_personas() -> None:
    # 6 voters cycles the 4 personas; FakeClient makes them unanimous, so it must not crash
    # and must produce 6 distinct cache cells (distinct agent_ids for the repeat cycle).
    client = FakeClient(answer="5")
    result = run_voting(_problem(ground_truth="5"), client=client, n_voters=6)
    assert result.n_voters == 6
    assert len(client.calls) == 6
    agent_ids = [aid for _, aid in client.calls]
    assert len(set(agent_ids)) == 6  # no two voters share an id -> no cache collision


# --------------------------------------------------------------------------- #
# majority_answer bucketing
# --------------------------------------------------------------------------- #
def test_majority_answer_empty() -> None:
    assert majority_answer(_problem(), []) == ""


def test_majority_answer_injected_equiv_buckets_numerically() -> None:
    # "5", "5.0", "5.00" are one bucket (3) vs "7" (1) under a numeric equiv -> mode "5".
    mode = majority_answer(
        _problem(), ["5", "5.0", "5.00", "7"], equiv=lambda a, b: float(a) == float(b)
    )
    assert mode == "5"


def test_majority_answer_verify_based_buckets_real() -> None:
    # REAL float comparison holds under both the stub and the real verifier.
    problem = _problem(answer_type=AnswerType.REAL, ground_truth="14.7", rel_tol=1e-3)
    mode = majority_answer(problem, ["14.7", "14.70", "99.0"])
    assert mode == "14.7"


def test_majority_answer_tiebreak_is_earliest_bucket() -> None:
    # Two singleton buckets tie at count 1 -> the earliest-seen answer wins, deterministically.
    mode = majority_answer(_problem(), ["a", "b"], equiv=lambda a, b: a == b)
    assert mode == "a"


def test_majority_answer_robust_to_nonreflexive_equiv() -> None:
    # The real verifier returns False for verify(gt=a, a) when `a` is unparseable, so the
    # default equiv is NOT reflexive on such answers. Byte-identical votes must still co-bucket:
    # with a pathologically non-reflexive equiv, three identical "NaN" votes (a real 3-vote
    # mode) must beat the lone "5", even though "5" arrives first.
    mode = majority_answer(_problem(), ["5", "NaN", "NaN", "NaN"], equiv=lambda a, b: False)
    assert mode == "NaN"


# --------------------------------------------------------------------------- #
# Realistic stub-dataset smoke
# --------------------------------------------------------------------------- #
def test_baselines_over_stub_dataset() -> None:
    for problem in load_problems():
        client = FakeClient(answer=problem.ground_truth)
        assert run_single(problem, client=client).is_correct is True
        assert run_voting(problem, client=FakeClient(answer=problem.ground_truth)).is_correct is True
