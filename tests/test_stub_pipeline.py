"""Phase-0 plumbing test: the stubs produce a valid pipeline end-to-end, offline.

Confirms the integration shape works (Engine -> DebateRecord -> verify) so Eval can build
on it before the real engine/dataset land. No API calls.
"""

from __future__ import annotations

import os

from debate.dataset import load_problems, verify
from debate.engine import run_debate
from debate.interfaces import DebateRecord


def test_stub_dataset_loads() -> None:
    problems = load_problems()
    assert len(problems) >= 1
    assert len({p.id for p in problems}) == len(problems)  # unique ids


def test_stub_run_debate_returns_valid_record() -> None:
    problem = load_problems()[0]
    record = run_debate(problem)
    assert isinstance(record, DebateRecord)
    # cardinality the contract requires
    assert len(record.role_assessments) == 4
    assert len(record.solutions) == 3
    assert len(record.reviews) == 6
    assert len(record.refinements) == 3
    # final answer is copied from the winning refinement
    winner = record.judgment.winner_solver_id
    winning_refinement = next(r for r in record.refinements if r.solver_id == winner)
    assert record.final_answer == winning_refinement.refined_answer


def test_stub_truthful_by_default_verifies() -> None:
    problem = load_problems()[0]
    record = run_debate(problem)
    assert verify(problem, record.final_answer) is True


def test_stub_wrong_rate_flips_answers(monkeypatch) -> None:
    """With DEBATE_STUB_WRONG_RATE=1.0 every answer is wrong, so the metric can move."""
    monkeypatch.setenv("DEBATE_STUB_WRONG_RATE", "1.0")
    problem = load_problems()[0]
    record = run_debate(problem)
    assert verify(problem, record.final_answer) is False


def test_verify_never_raises_on_garbage() -> None:
    problem = load_problems()[0]
    assert verify(problem, "this is not a valid answer at all") is False
    assert verify(problem, "") is False
