"""Stage 0.5 — deterministic role assignment (pure Python, no LLM call).

Given the four agents' Stage-0 self-assessments, pick exactly one judge and three solvers.
The judge is the agent with the greatest *comparative* advantage at judging over solving —
``judge_confidence - solver_confidence`` — not merely the highest judge confidence, so an
agent that is great at everything still goes where its edge is largest (and strong solvers
stay solving). Determinism matters: same assessments must always yield the same assignment,
so every tie is broken by an explicit, documented rule.
"""

from __future__ import annotations

from .interfaces import RoleAssignment, RoleSelfAssessment


def assign_roles(assessments: list[RoleSelfAssessment]) -> RoleAssignment:
    """Choose 1 judge + 3 solvers from exactly four self-assessments.

    Judge = argmax of ``judge_confidence - solver_confidence``.
    Tiebreaks, in order: higher ``judge_confidence``; then higher ``solver_confidence``
    (a near-tie on the gap goes to the all-round stronger agent for the judge seat); then
    ``agent_id`` lexicographically (total order → fully reproducible).
    """
    if len(assessments) != 4:
        raise ValueError(f"role assignment needs exactly 4 assessments, got {len(assessments)}")
    ids = [a.agent_id for a in assessments]
    if len(set(ids)) != 4:
        raise ValueError(f"agent_ids must be unique, got {ids}")

    def judge_key(a: RoleSelfAssessment) -> tuple[float, float, float, str]:
        # min() picks the smallest key, so negate the "more is better" fields: largest
        # advantage first, then judge_confidence, then solver_confidence; finally the
        # lexicographically smallest agent_id breaks any remaining tie. The advantage is
        # rounded so float subtraction noise (0.8-0.7 != 0.9-0.8) can't decide the seat —
        # genuinely-equal gaps fall through to the judge_confidence tiebreak instead.
        advantage = round(a.judge_confidence - a.solver_confidence, 6)
        return (-advantage, -a.judge_confidence, -a.solver_confidence, a.agent_id)

    judge = min(assessments, key=judge_key)
    solver_ids = sorted(aid for aid in ids if aid != judge.agent_id)

    gap = judge.judge_confidence - judge.solver_confidence
    rationale = (
        f"judge={judge.agent_id}: max(judge_confidence - solver_confidence) = "
        f"{judge.judge_confidence:.2f} - {judge.solver_confidence:.2f} = {gap:+.2f}; "
        f"solvers={solver_ids}."
    )
    return RoleAssignment(judge_id=judge.agent_id, solver_ids=solver_ids, rationale=rationale)
