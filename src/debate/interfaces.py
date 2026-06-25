"""Frozen data contracts for the Multi-LLM Collaborative Debate System.

Single source of truth for validation, JSON parsing, and Gemini structured-output
schemas. FROZEN in week 1 — all three workstreams (engine, dataset, eval) import from
here. Changing a field after freeze requires team sign-off because it breaks cached
records and the schemas sent to the model; bump ``DebateRecord.schema_version`` if you do.

Schema-safety rules (Gemini structured-output subset): only ``str``/``int``/``float``/
``bool``, ``Enum``/``Literal``, nested ``BaseModel``, ``list[...]``, and ``Optional``.
No ``dict[str, X]``, no unions of dissimilar models, no regex ``pattern``, no tuples.
Pydantic validators run client-side on ``.parsed`` and do NOT steer generation — but a
failed validation does trip the client's repair loop (exploited for Stage 2's
anti-sycophancy guarantee). Only the per-stage models (RoleSelfAssessment, Solution,
Review, RefinedSolution, Judgment) are passed as ``response_schema``; ``DebateRecord`` is
never sent to the model, so it may freely hold ``datetime`` and aggregate lists.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #
class Category(str, Enum):
    """The four problem domains (~6-7 problems each)."""

    COMBINATORICS = "combinatorics"  # incl. number theory
    PHYSICS = "physics"  # multi-step quantitative physics
    LOGIC = "logic"  # logic / constraint-satisfaction puzzles
    GAME_THEORY = "game_theory"  # strategic / game-theoretic


class AnswerType(str, Enum):
    """How ``answer`` is shaped, so the verifier/normalizer knows how to compare."""

    INTEGER = "integer"
    RATIONAL = "rational"  # fraction or decimal, compared exactly via sympy
    REAL = "real"  # float with tolerance (Problem.rel_tol)
    SYMBOLIC = "symbolic"  # expression, compared via sympy.simplify(a - b) == 0
    SET = "set"  # unordered collection, e.g. {2, 3, 5}
    TUPLE = "tuple"  # ordered, e.g. (x=2, y=3) -> canonicalized
    STRING = "string"  # exact normalized token (names, sequences); last resort


class Role(str, Enum):
    SOLVER = "solver"
    JUDGE = "judge"


class Severity(str, Enum):
    """Severity of a located critique; weights reviews and gates sycophancy."""

    CRITICAL = "critical"  # invalidates the final answer
    MAJOR = "major"  # significant flaw, answer may survive
    MINOR = "minor"  # cosmetic / small gap


class ErrorType(str, Enum):
    """Typed critique categories. Forces reviewers to classify, not hand-wave."""

    ARITHMETIC = "arithmetic"
    ALGEBRAIC = "algebraic"
    LOGICAL = "logical"  # invalid inference / unjustified step
    CONCEPTUAL = "conceptual"  # wrong formula / wrong principle
    MISREAD_PROBLEM = "misread_problem"
    UNSTATED_ASSUMPTION = "unstated_assumption"
    INCOMPLETE = "incomplete"  # missing case / step
    OTHER = "other"


# --------------------------------------------------------------------------- #
# Problem (Dataset -> Engine + Eval)
# --------------------------------------------------------------------------- #
class Problem(BaseModel):
    """A single verifiable problem. Produced by the Dataset workstream, consumed by the
    Engine (to solve) and Eval (to score). ``ground_truth`` is the canonical answer
    string; verification is always done by the programmatic verifier (``dataset.verify``),
    never by a bare string compare against this field.
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(..., description="Stable unique id, e.g. 'comb_001'.")
    text: str = Field(..., min_length=1, description="Full problem statement.")
    category: Category
    ground_truth: str = Field(..., description="Canonical correct answer, normalized form.")
    answer_type: AnswerType
    difficulty: int = Field(..., ge=1, le=5, description="1 (easy) .. 5 (hard).")
    concepts: list[str] = Field(default_factory=list, description="Tags, e.g. ['CRT'].")
    rel_tol: float = Field(1e-6, description="Tolerance used only for REAL answers.")
    source: str | None = Field(None, description="Provenance, e.g. 'original' / 'adapted:<ref>'.")
    gate_fail_rate: float | None = Field(
        None, ge=0.0, le=1.0, description="Filled by the difficulty gate (single-model fail rate)."
    )


# --------------------------------------------------------------------------- #
# Stage 0 — role self-assessment  /  Stage 0.5 — assignment (Engine)
# --------------------------------------------------------------------------- #
class RoleSelfAssessment(BaseModel):
    """One agent's Stage-0 output: how suited it thinks it is to each role on THIS
    problem. Confidences are independent per role (not required to sum to 1).
    ``agent_id`` is pinned engine-side AFTER the call — the model does not choose it.
    """

    agent_id: str = Field(..., description="Stable agent id, pinned engine-side post-parse.")
    solver_confidence: float = Field(
        ..., ge=0.0, le=1.0, description="P(this agent produces a fully correct answer)."
    )
    judge_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="P(this agent correctly picks the best of several mixed-quality candidates).",
    )
    role_preferences: list[Role] = Field(
        ..., description="Roles ranked best-first. Advisory only; assignment uses the scalars."
    )
    reasoning: str = Field(..., min_length=1, description="Brief justification of the confidences.")


class RoleAssignment(BaseModel):
    """Stage-0.5 deterministic result: exactly 3 solvers + 1 judge."""

    judge_id: str
    solver_ids: list[str] = Field(..., min_length=3, max_length=3)
    rationale: str = Field(..., description="Which rule/tiebreak chose the judge.")


# --------------------------------------------------------------------------- #
# Stage 1 — independent solution (Engine)
# --------------------------------------------------------------------------- #
class Solution(BaseModel):
    """One Solver's independent Stage-1 solution. No knowledge of other solvers.
    ``solver_id`` is stamped engine-side after parse (the model does not echo it).
    """

    solver_id: str = Field("", description="Stamped engine-side; not model-generated.")
    reasoning: str = Field(..., min_length=1, description="Full step-by-step working.")
    answer: str = Field(..., description="Final answer, in the problem's answer_type form.")
    confidence: float = Field(..., ge=0.0, le=1.0)


# --------------------------------------------------------------------------- #
# Stage 2 — peer review (Engine) — THE GRADED STAGE (15 pts)
# --------------------------------------------------------------------------- #
class ReviewError(BaseModel):
    """One located, typed, severity-tagged critique. The anti-sycophancy unit: a review
    MUST contain at least one of these pointing at a concrete location.
    """

    location: str = Field(..., min_length=1, description="Where, e.g. 'Step 3' or a quoted line.")
    error_type: ErrorType
    severity: Severity
    description: str = Field(..., min_length=1, description="What is wrong and why; be specific.")


class Review(BaseModel):
    """One Solver's Stage-2 review of ONE other solution. Six per problem (3 solvers x 2
    peers). The prompt + this schema's ``min_length=1`` on ``errors`` forbid vague praise.
    ``reviewer_id`` / ``target_solver_id`` are stamped engine-side.
    """

    reviewer_id: str = Field("", description="Stamped engine-side.")
    target_solver_id: str = Field("", description="solver_id of the reviewed solution; stamped.")
    errors: list[ReviewError] = Field(
        ..., min_length=1, description="Located/typed critiques. >=1 required (anti-sycophancy)."
    )
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    suggested_changes: list[str] = Field(default_factory=list)
    overall_assessment: str = Field(..., min_length=1)


# --------------------------------------------------------------------------- #
# Stage 3 — refinement (Engine)
# --------------------------------------------------------------------------- #
class ChangeRecord(BaseModel):
    """How the solver responded to ONE incoming critique: accept+fix or rebut+reason."""

    critique_location: str = Field(..., description="Echoes the ReviewError.location addressed.")
    accepted: bool = Field(..., description="True = critique accepted and fixed; False = rebutted.")
    response: str = Field(
        ..., min_length=1, description="If accepted: what changed. If rebutted: why it's wrong."
    )


class RefinedSolution(BaseModel):
    """One Solver's Stage-3 output after seeing its two reviews. Every incoming critique
    should be addressed by a ChangeRecord.
    """

    solver_id: str = Field("", description="Stamped engine-side.")
    changes_made: list[ChangeRecord] = Field(default_factory=list)
    refined_reasoning: str = Field(..., min_length=1)
    refined_answer: str = Field(..., description="Final post-refinement answer.")
    confidence: float = Field(..., ge=0.0, le=1.0)


# --------------------------------------------------------------------------- #
# Stage 4 — judgment (Engine)
# --------------------------------------------------------------------------- #
class Judgment(BaseModel):
    """The Judge's Stage-4 decision over the 3 refined solutions. ``final_answer`` is
    COPIED programmatically from the winning ``RefinedSolution.refined_answer`` by the
    orchestrator — the judge only chooses ``winner_solver_id``; never trust a
    judge-transcribed answer.
    """

    winner_solver_id: str
    final_answer: str = Field(
        "", description="Copied from winner.refined_answer by the orchestrator."
    )
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str = Field(..., min_length=1)


# --------------------------------------------------------------------------- #
# Metadata / cost / latency
# --------------------------------------------------------------------------- #
class StageTiming(BaseModel):
    """Per-call accounting. One entry per model call (Stage 0 x4, Stage 1 x3,
    Stage 2 x6, Stage 3 x3, Stage 4 x1 => up to 17 per problem).
    """

    stage: str = Field(..., description="e.g. 'stage1_solve', 'stage2_review'.")
    agent_id: str
    latency_s: float = Field(0.0, ge=0.0)
    prompt_tokens: int = Field(0, ge=0)
    output_tokens: int = Field(0, ge=0)  # candidates_token_count (visible)
    thinking_tokens: int = Field(0, ge=0)  # thoughts_token_count, tracked SEPARATELY
    total_tokens: int = Field(0, ge=0)
    cache_hit: bool = Field(False, description="True if served from disk cache, no API call.")
    retries: int = Field(0, ge=0, description="Retry attempts before success.")


class CostMeta(BaseModel):
    """Aggregate accounting for one DebateRecord."""

    total_latency_s: float = Field(0.0, ge=0.0)
    total_prompt_tokens: int = Field(0, ge=0)
    total_output_tokens: int = Field(0, ge=0)
    total_thinking_tokens: int = Field(0, ge=0)
    total_tokens: int = Field(0, ge=0)
    n_api_calls: int = Field(0, ge=0)  # excludes cache hits
    n_cache_hits: int = Field(0, ge=0)
    timings: list[StageTiming] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Full transcript (Engine -> Eval)
# --------------------------------------------------------------------------- #
class DebateRecord(BaseModel):
    """Complete, reproducible transcript of one problem through all stages. This is the
    Engine's deliverable (``run_debate(problem) -> DebateRecord``) and Eval's only input.
    Self-contained: re-serializable to JSON and replayable.
    """

    model_config = ConfigDict(frozen=False)

    schema_version: str = Field("1.0", description="Bump only with team sign-off.")
    problem: Problem

    role_assessments: list[RoleSelfAssessment] = Field(..., min_length=4, max_length=4)  # Stage 0
    assignment: RoleAssignment  # Stage 0.5
    solutions: list[Solution] = Field(..., min_length=3, max_length=3)  # Stage 1
    reviews: list[Review] = Field(..., min_length=6, max_length=6)  # Stage 2
    refinements: list[RefinedSolution] = Field(..., min_length=3, max_length=3)  # Stage 3
    judgment: Judgment  # Stage 4

    final_answer: str = Field(..., description="== judgment.final_answer; surfaced for Eval.")
    is_correct: bool | None = Field(None, description="Filled by Eval via verify().")
    cost: CostMeta = Field(default_factory=CostMeta)

    run_seed: int = Field(
        ..., description="Base seed for this problem; per-call seeds derived from it."
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
