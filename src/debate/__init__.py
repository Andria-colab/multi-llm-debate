"""Multi-LLM Collaborative Debate System.

Re-exports the frozen data contracts so callers can ``from debate import Problem, ...``.
The workstream entry points (``run_debate``, ``load_problems``, ``verify``) live in their
own submodules and are imported directly from there to avoid import cycles during the
stub phase.
"""

from .interfaces import (
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

__all__ = [
    "AnswerType",
    "Category",
    "ChangeRecord",
    "CostMeta",
    "DebateRecord",
    "ErrorType",
    "Judgment",
    "Problem",
    "RefinedSolution",
    "Review",
    "ReviewError",
    "Role",
    "RoleAssignment",
    "RoleSelfAssessment",
    "Severity",
    "Solution",
    "StageTiming",
]
