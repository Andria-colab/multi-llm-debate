"""Metrics, the unified result row, and the on-disk results format.

This is the data + reporting layer of the Eval workstream. It is deliberately the *lowest*
module in the eval package (it imports nothing from ``baselines`` / ``plots`` / ``runner``)
so the other three can all depend on it without a cycle.

Three things live here:

* ``MethodResult`` — the common per-(method, problem) row that all three methods produce
  (single-call, majority-vote, full debate), so metrics and plots treat them identically.
* ``ResultsFile`` — the one machine-readable artifact the runner writes to ``results/`` and
  the plotter reads back. It carries everything needed to recompute every metric and
  re-render every figure *without* re-calling the API.
* Pure metric functions over those rows (accuracy / cost / latency) plus debate-only
  dynamics over raw ``DebateRecord`` (answer-change rate, critique acceptance, …). Every
  function here is a deterministic transform — no model calls, no network — so they are
  trivially unit-testable offline.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any, Callable

from pydantic import BaseModel, Field

from debate.dataset import verify
from debate.interfaces import CostMeta, DebateRecord, Problem, StageTiming

# The three compared methods, in canonical (plot/report) order, and their display labels.
METHODS: tuple[str, ...] = ("single", "voting", "debate")
METHOD_LABELS: dict[str, str] = {
    "single": "Single-call",
    "voting": "Majority-vote",
    "debate": "Full debate",
}

# Canonical artifact names under RESULTS_DIR (kept here so runner + plots agree on one name).
RESULTS_FILENAME = "results.json"
METRICS_FILENAME = "metrics.json"


# --------------------------------------------------------------------------- #
# The unified result row
# --------------------------------------------------------------------------- #
class MethodResult(BaseModel):
    """One method's outcome on one problem — the row every metric and plot consumes.

    Single-call, majority-vote and full-debate all collapse to this shape so they compare
    apples-to-apples. ``cost`` reuses the engine's ``CostMeta`` so there is exactly one cost
    accounting format across the whole project.
    """

    method: str  # one of METHODS
    problem: Problem  # full problem, so metrics can break down by category / difficulty
    answer: str
    is_correct: bool | None = None  # set by verify(); None only if scoring was skipped
    cost: CostMeta = Field(default_factory=CostMeta)
    votes: list[str] = Field(default_factory=list)  # majority-vote only: the per-voter answers
    n_voters: int = 0  # majority-vote only: number of voters


# --------------------------------------------------------------------------- #
# Shared, dependency-free helpers
# --------------------------------------------------------------------------- #
def aggregate_cost(timings: Sequence[StageTiming]) -> CostMeta:
    """Fold per-call timings into one ``CostMeta`` — the baselines' cost side.

    Deliberately mirrors ``engine._aggregate_cost`` field-for-field so a baseline's cost is
    accounted identically to a full debate's (an honest apples-to-apples comparison). Both
    just fold the *frozen* ``CostMeta`` contract, so they can drift only if that contract
    changes — keep the two in sync if it ever does.
    """
    return CostMeta(
        total_latency_s=sum(t.latency_s for t in timings),
        total_prompt_tokens=sum(t.prompt_tokens for t in timings),
        total_output_tokens=sum(t.output_tokens for t in timings),
        total_thinking_tokens=sum(t.thinking_tokens for t in timings),
        total_tokens=sum(t.total_tokens for t in timings),
        n_api_calls=sum(1 for t in timings if not t.cache_hit),
        n_cache_hits=sum(1 for t in timings if t.cache_hit),
        timings=list(timings),
    )


def answers_equivalent(problem: Problem, a: str, b: str) -> bool:
    """Do two answers mean the same thing for this problem?

    Byte-identical answers are always equivalent (the fast path) — this guarantees
    reflexivity even when the installed verifier cannot *parse* the answer. That matters: the
    real verifier (``dataset.verifiers``) returns ``False`` for ``verify(gt=a, a)`` whenever
    ``a`` is unparseable for the problem's ``answer_type`` (e.g. an INTEGER answer of
    ``"infinitely many"``), so without the fast path two identical-but-unparseable answers
    would read as *different*.

    Beyond exact equality it defers to the dataset verifier's per-``answer_type``
    normalization, so equivalence is only as strong as the installed verifier: the real
    verifier normalizes every type (``"5"`` == ``"5.0"``, ``"{1, 2}"`` == ``"{2,1}"``), while
    the bootstrap stub is exact-match for everything except REAL. It treats ``a`` as the
    ground truth and checks ``b`` against it (so for REAL the tolerance is anchored on ``a``);
    ``model_copy(update=...)`` rebuilds the frozen ``Problem`` with a swapped ``ground_truth``
    without going through ``__setattr__``. Never raises — ``verify`` is contractually total.
    """
    if a == b:
        return True
    probe = problem.model_copy(update={"ground_truth": a})
    return verify(probe, b)


# --------------------------------------------------------------------------- #
# Accuracy
# --------------------------------------------------------------------------- #
def _correct(r: MethodResult) -> bool:
    return r.is_correct is True


def accuracy(results: Sequence[MethodResult]) -> float:
    """Fraction correct (a ``None``/unscored row counts as not-correct). 0.0 on no rows."""
    if not results:
        return 0.0
    return sum(1 for r in results if _correct(r)) / len(results)


def _group_accuracy(
    results: Sequence[MethodResult], key: Callable[[MethodResult], str]
) -> dict[str, float]:
    groups: dict[str, list[MethodResult]] = {}
    for r in results:
        groups.setdefault(key(r), []).append(r)
    return {k: accuracy(v) for k, v in groups.items()}


def accuracy_by_category(results: Sequence[MethodResult]) -> dict[str, float]:
    """Accuracy split by ``problem.category`` (string-valued keys for JSON friendliness)."""
    return _group_accuracy(results, lambda r: r.problem.category.value)


def accuracy_by_difficulty(results: Sequence[MethodResult]) -> dict[str, float]:
    """Accuracy split by ``problem.difficulty`` (1..5), keyed by the stringified int."""
    return _group_accuracy(results, lambda r: str(r.problem.difficulty))


# --------------------------------------------------------------------------- #
# Report models
# --------------------------------------------------------------------------- #
class MethodMetrics(BaseModel):
    """All headline numbers for one method, ready to serialize and plot."""

    method: str
    n: int  # problems attempted by this method
    n_correct: int
    accuracy: float
    accuracy_by_category: dict[str, float] = Field(default_factory=dict)
    accuracy_by_difficulty: dict[str, float] = Field(default_factory=dict)

    # cost (thinking tokens tracked separately — they dominate cost on gemini-2.5-flash)
    total_tokens: int = 0
    total_prompt_tokens: int = 0
    total_output_tokens: int = 0
    total_thinking_tokens: int = 0
    mean_tokens: float = 0.0
    mean_prompt_tokens: float = 0.0
    mean_output_tokens: float = 0.0
    mean_thinking_tokens: float = 0.0
    total_api_calls: int = 0
    total_cache_hits: int = 0

    # latency (cache hits report ~0s, so a fully-cached rerun shows near-zero latency)
    total_latency_s: float = 0.0
    mean_latency_s: float = 0.0


class DebateDynamics(BaseModel):
    """Debate-only behavioural stats mined from the raw transcripts (nice for the writeup)."""

    n_records: int = 0
    answer_change_rate: float = 0.0  # P(a solver's refined answer differs from its stage-1 answer)
    critique_acceptance_rate: float = 0.0  # accepted ChangeRecords / all ChangeRecords
    mean_critiques_per_review: float = 0.0  # located errors per Stage-2 review
    severity_counts: dict[str, int] = Field(default_factory=dict)


class EvalReport(BaseModel):
    """The full comparison: one ``MethodMetrics`` per present method + debate dynamics."""

    methods: list[MethodMetrics] = Field(default_factory=list)
    dynamics: DebateDynamics | None = None

    def by_method(self) -> dict[str, MethodMetrics]:
        return {m.method: m for m in self.methods}


class ResultsFile(BaseModel):
    """The single artifact written to ``results/`` — reproducible and re-plottable.

    Holds the per-method rows (enough to recompute every metric), the raw debate records
    (enough to recompute dynamics and replay), and the pre-computed report (so a plotter can
    render without recomputing). Round-trips cleanly through ``model_dump_json`` /
    ``model_validate_json``.
    """

    schema_version: str = "1.0"
    created_at: str = ""
    n_problems: int = 0
    settings: dict[str, Any] = Field(default_factory=dict)
    results: dict[str, list[MethodResult]] = Field(default_factory=dict)
    debate_records: list[DebateRecord] = Field(default_factory=list)
    report: EvalReport | None = None


# --------------------------------------------------------------------------- #
# Computing the report
# --------------------------------------------------------------------------- #
def compute_method_metrics(method: str, results: Sequence[MethodResult]) -> MethodMetrics:
    """All metrics for one method over its rows. Safe on an empty list (returns zeros)."""
    n = len(results)
    n_correct = sum(1 for r in results if _correct(r))
    total_tokens = sum(r.cost.total_tokens for r in results)
    total_prompt = sum(r.cost.total_prompt_tokens for r in results)
    total_output = sum(r.cost.total_output_tokens for r in results)
    total_thinking = sum(r.cost.total_thinking_tokens for r in results)
    total_latency = sum(r.cost.total_latency_s for r in results)
    denom = n or 1  # avoid /0; with n==0 every numerator is 0 anyway
    return MethodMetrics(
        method=method,
        n=n,
        n_correct=n_correct,
        accuracy=accuracy(results),
        accuracy_by_category=accuracy_by_category(results),
        accuracy_by_difficulty=accuracy_by_difficulty(results),
        total_tokens=total_tokens,
        total_prompt_tokens=total_prompt,
        total_output_tokens=total_output,
        total_thinking_tokens=total_thinking,
        mean_tokens=total_tokens / denom,
        mean_prompt_tokens=total_prompt / denom,
        mean_output_tokens=total_output / denom,
        mean_thinking_tokens=total_thinking / denom,
        total_api_calls=sum(r.cost.n_api_calls for r in results),
        total_cache_hits=sum(r.cost.n_cache_hits for r in results),
        total_latency_s=total_latency,
        mean_latency_s=total_latency / denom,
    )


def compute_dynamics(records: Sequence[DebateRecord]) -> DebateDynamics:
    """Behavioural stats over the raw debate transcripts (Stage 2/3 anti-sycophancy signal)."""
    n_refine = n_changed = n_changes = n_accepted = 0
    n_reviews = n_errors = 0
    severity: Counter[str] = Counter()

    for rec in records:
        solution_by_id = {s.solver_id: s for s in rec.solutions}
        for ref in rec.refinements:
            n_refine += 1
            original = solution_by_id.get(ref.solver_id)
            if original is not None and not answers_equivalent(
                rec.problem, original.answer, ref.refined_answer
            ):
                n_changed += 1
            for change in ref.changes_made:
                n_changes += 1
                if change.accepted:
                    n_accepted += 1
        for review in rec.reviews:
            n_reviews += 1
            n_errors += len(review.errors)
            for err in review.errors:
                severity[err.severity.value] += 1

    return DebateDynamics(
        n_records=len(records),
        answer_change_rate=(n_changed / n_refine if n_refine else 0.0),
        critique_acceptance_rate=(n_accepted / n_changes if n_changes else 0.0),
        mean_critiques_per_review=(n_errors / n_reviews if n_reviews else 0.0),
        severity_counts=dict(severity),
    )


def build_report(
    results_by_method: Mapping[str, Sequence[MethodResult]],
    records: Sequence[DebateRecord] = (),
) -> EvalReport:
    """Assemble the full report: a ``MethodMetrics`` for each present method (in METHODS
    order, then any extras), plus debate dynamics when records are available."""
    ordered = list(METHODS) + [m for m in results_by_method if m not in METHODS]
    methods = [
        compute_method_metrics(m, results_by_method[m])
        for m in ordered
        if results_by_method.get(m)
    ]
    dynamics = compute_dynamics(records) if records else None
    return EvalReport(methods=methods, dynamics=dynamics)
