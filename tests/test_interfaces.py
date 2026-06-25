"""Day-1 contract tests: every frozen model round-trips, and the validators bite.

These guard the single most important invariant of the project — that the frozen data
contracts in ``interfaces.py`` serialize/validate cleanly — so all three workstreams can
rely on them. Fully offline.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from debate.interfaces import (
    AnswerType,
    Category,
    ChangeRecord,
    CostMeta,
    DebateRecord,
    ErrorType,
    Judgment,
    Problem,
    RefinedSolution,
    Review,
    ReviewError,
    Role,
    RoleAssignment,
    RoleSelfAssessment,
    Severity,
    Solution,
    StageTiming,
)


def _problem() -> Problem:
    return Problem(
        id="p1",
        text="What is 2 + 2?",
        category=Category.COMBINATORICS,
        ground_truth="4",
        answer_type=AnswerType.INTEGER,
        difficulty=1,
    )


def _full_record() -> DebateRecord:
    solver_ids = ["formalist", "lateral", "checker"]
    judge_id = "skeptic"
    assessments = [
        RoleSelfAssessment(
            agent_id=a,
            solver_confidence=0.8,
            judge_confidence=0.7,
            role_preferences=[Role.SOLVER, Role.JUDGE],
            reasoning="x",
        )
        for a in solver_ids + [judge_id]
    ]
    solutions = [
        Solution(solver_id=s, reasoning="r", answer="4", confidence=0.9) for s in solver_ids
    ]
    reviews = [
        Review(
            reviewer_id=r,
            target_solver_id=t,
            errors=[
                ReviewError(
                    location="Step 1",
                    error_type=ErrorType.LOGICAL,
                    severity=Severity.MINOR,
                    description="d",
                )
            ],
            overall_assessment="ok",
        )
        for r in solver_ids
        for t in solver_ids
        if r != t
    ]
    refinements = [
        RefinedSolution(
            solver_id=s,
            changes_made=[
                ChangeRecord(critique_location="Step 1", accepted=True, response="fixed")
            ],
            refined_reasoning="r2",
            refined_answer="4",
            confidence=0.95,
        )
        for s in solver_ids
    ]
    judgment = Judgment(
        winner_solver_id="formalist", final_answer="4", confidence=0.9, reasoning="best"
    )
    return DebateRecord(
        problem=_problem(),
        role_assessments=assessments,
        assignment=RoleAssignment(judge_id=judge_id, solver_ids=solver_ids, rationale="r"),
        solutions=solutions,
        reviews=reviews,
        refinements=refinements,
        judgment=judgment,
        final_answer="4",
        cost=CostMeta(),
        run_seed=123,
    )


@pytest.mark.parametrize(
    "obj",
    [
        _problem(),
        RoleSelfAssessment(
            agent_id="a",
            solver_confidence=0.5,
            judge_confidence=0.5,
            role_preferences=[Role.SOLVER, Role.JUDGE],
            reasoning="x",
        ),
        RoleAssignment(judge_id="d", solver_ids=["a", "b", "c"], rationale="r"),
        Solution(solver_id="a", reasoning="r", answer="4", confidence=0.9),
        ReviewError(
            location="Step 1",
            error_type=ErrorType.ARITHMETIC,
            severity=Severity.CRITICAL,
            description="d",
        ),
        Review(
            reviewer_id="a",
            target_solver_id="b",
            errors=[
                ReviewError(
                    location="L",
                    error_type=ErrorType.OTHER,
                    severity=Severity.MINOR,
                    description="d",
                )
            ],
            overall_assessment="ok",
        ),
        ChangeRecord(critique_location="Step 1", accepted=False, response="rebut"),
        RefinedSolution(solver_id="a", refined_reasoning="r", refined_answer="4", confidence=0.9),
        Judgment(winner_solver_id="a", final_answer="4", confidence=0.8, reasoning="r"),
        StageTiming(stage="stage1", agent_id="a"),
        CostMeta(),
        _full_record(),
    ],
)
def test_round_trip(obj: BaseModel) -> None:
    """Every model survives model_validate(model_dump()) byte-for-byte."""
    clone = type(obj).model_validate(obj.model_dump(mode="json"))
    assert clone.model_dump(mode="json") == obj.model_dump(mode="json")


def test_confidence_bounds_enforced() -> None:
    with pytest.raises(ValidationError):
        Solution(solver_id="a", reasoning="r", answer="4", confidence=1.5)


def test_review_requires_at_least_one_error() -> None:
    """The anti-sycophancy invariant: a Review with no located errors is invalid."""
    with pytest.raises(ValidationError):
        Review(reviewer_id="a", target_solver_id="b", errors=[], overall_assessment="looks good")


def test_debate_record_cardinality_enforced() -> None:
    """Wrong number of solutions/reviews/refinements must fail validation."""
    rec = _full_record()
    bad = rec.model_dump(mode="json")
    bad["solutions"] = bad["solutions"][:2]  # only 2 solvers
    with pytest.raises(ValidationError):
        DebateRecord.model_validate(bad)


def test_problem_difficulty_bounds() -> None:
    with pytest.raises(ValidationError):
        Problem(
            id="x",
            text="t",
            category=Category.LOGIC,
            ground_truth="1",
            answer_type=AnswerType.INTEGER,
            difficulty=9,
        )
