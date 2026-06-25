"""Stage 0.5 role-assignment tests: max comparative advantage + documented tiebreaks."""

from __future__ import annotations

import pytest

from debate.interfaces import Role, RoleSelfAssessment
from debate.roles import assign_roles


def _assess(agent_id: str, solver: float, judge: float) -> RoleSelfAssessment:
    return RoleSelfAssessment(
        agent_id=agent_id,
        solver_confidence=solver,
        judge_confidence=judge,
        role_preferences=[Role.SOLVER, Role.JUDGE],
        reasoning="x",
    )


def test_judge_is_max_comparative_advantage() -> None:
    # 'b' has the largest judge - solver gap (+0.5), even though 'd' is the best judge overall.
    assessments = [
        _assess("a", 0.9, 0.5),  # gap -0.4
        _assess("b", 0.4, 0.9),  # gap +0.5  <- winner
        _assess("c", 0.7, 0.6),  # gap -0.1
        _assess("d", 0.8, 0.95),  # gap +0.15 (higher judge_conf, smaller gap)
    ]
    result = assign_roles(assessments)
    assert result.judge_id == "b"
    assert result.solver_ids == ["a", "c", "d"]  # sorted, exactly 3


def test_assignment_is_deterministic() -> None:
    assessments = [
        _assess("formalist", 0.8, 0.7),
        _assess("lateral", 0.8, 0.7),
        _assess("checker", 0.8, 0.7),
        _assess("skeptic", 0.8, 0.7),
    ]
    first = assign_roles(assessments)
    second = assign_roles(list(reversed(assessments)))  # order must not matter
    assert first.judge_id == second.judge_id == "checker"  # all tied → smallest agent_id
    assert first.solver_ids == second.solver_ids == ["formalist", "lateral", "skeptic"]


def test_tiebreak_prefers_higher_judge_confidence() -> None:
    # equal gaps (all +0.1) → higher judge_confidence wins the judge seat
    assessments = [
        _assess("a", 0.5, 0.6),
        _assess("b", 0.8, 0.9),  # same +0.1 gap but highest judge_confidence
        _assess("c", 0.6, 0.7),
        _assess("d", 0.7, 0.8),
    ]
    assert assign_roles(assessments).judge_id == "b"


def test_requires_exactly_four() -> None:
    with pytest.raises(ValueError):
        assign_roles([_assess("a", 0.5, 0.5), _assess("b", 0.5, 0.5)])
