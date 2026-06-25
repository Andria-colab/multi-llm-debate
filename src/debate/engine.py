"""Engine entry point: ``run_debate(problem) -> DebateRecord``.

================================  STUB  =====================================
This is the Phase-0 STUB. It returns a *structurally valid* DebateRecord with
deterministic fake content so the Eval workstream can build the runner/metrics/plots
in parallel before the real orchestrator exists. The real Stages 0-4 orchestration
(client, agents, prompts, roles, concurrency) progressively replaces this function body
during Phase 1-2 — the import path ``from debate.engine import run_debate`` stays stable.

``DEBATE_STUB_WRONG_RATE`` (env, default 0.0) deterministically flips a fraction of
answers to "__wrong__" so Eval can confirm the metrics actually move.
=============================================================================
"""

from __future__ import annotations

import hashlib
import os

from .config import SETTINGS, per_problem_seed
from .interfaces import (
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
)

# Stub role layout: first 3 agents solve, last agent judges.
_AGENT_IDS = [a.agent_id for a in SETTINGS.agents]
_SOLVER_IDS = _AGENT_IDS[:3]
_JUDGE_ID = _AGENT_IDS[3]


def _stub_is_wrong(problem_id: str) -> bool:
    """Deterministic per-problem 'is this answer wrong?' decision for the stub."""
    rate = float(os.getenv("DEBATE_STUB_WRONG_RATE", "0.0"))
    if rate <= 0.0:
        return False
    bucket = int(hashlib.sha256(problem_id.encode()).hexdigest()[:4], 16) % 100
    return bucket < rate * 100


def run_debate(problem: Problem) -> DebateRecord:
    """STUB: return a valid DebateRecord. Replace with the real orchestrator (Stages 0-4)."""
    answer = "__wrong__" if _stub_is_wrong(problem.id) else problem.ground_truth

    role_assessments = [
        RoleSelfAssessment(
            agent_id=aid,
            solver_confidence=0.8,
            judge_confidence=0.7 if aid != _JUDGE_ID else 0.9,
            role_preferences=[Role.JUDGE, Role.SOLVER] if aid == _JUDGE_ID
            else [Role.SOLVER, Role.JUDGE],
            reasoning="stub self-assessment",
        )
        for aid in _AGENT_IDS
    ]

    assignment = RoleAssignment(
        judge_id=_JUDGE_ID, solver_ids=list(_SOLVER_IDS), rationale="stub assignment"
    )

    solutions = [
        Solution(solver_id=sid, reasoning="stub reasoning", answer=answer, confidence=0.85)
        for sid in _SOLVER_IDS
    ]

    reviews = [
        Review(
            reviewer_id=r,
            target_solver_id=t,
            errors=[
                ReviewError(
                    location="Step 1",
                    error_type=ErrorType.OTHER,
                    severity=Severity.MINOR,
                    description="stub located critique",
                )
            ],
            strengths=["stub strength tied to Step 1"],
            weaknesses=["stub weakness"],
            suggested_changes=["stub suggestion"],
            overall_assessment="promising_but_flawed",
        )
        for r in _SOLVER_IDS
        for t in _SOLVER_IDS
        if r != t
    ]

    refinements = [
        RefinedSolution(
            solver_id=sid,
            changes_made=[
                ChangeRecord(critique_location="Step 1", accepted=False, response="stub rebuttal")
            ],
            refined_reasoning="stub refined reasoning",
            refined_answer=answer,
            confidence=0.9,
        )
        for sid in _SOLVER_IDS
    ]

    winner_id = _SOLVER_IDS[0]
    final_answer = next(r.refined_answer for r in refinements if r.solver_id == winner_id)
    judgment = Judgment(
        winner_solver_id=winner_id,
        final_answer=final_answer,
        confidence=0.9,
        reasoning="stub judgment",
    )

    return DebateRecord(
        problem=problem,
        role_assessments=role_assessments,
        assignment=assignment,
        solutions=solutions,
        reviews=reviews,
        refinements=refinements,
        judgment=judgment,
        final_answer=final_answer,
        cost=CostMeta(),
        run_seed=per_problem_seed(problem.id),
    )
