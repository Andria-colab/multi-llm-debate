"""The two comparison baselines: single-call and majority-vote.

Both reuse the engine's ``GeminiClient`` and the engine's Stage-1 solve prompt
(``stage1_solve`` + the persona system prompts), so they hit the same on-disk cache, honour
``DEBATE_OFFLINE``, and produce ``StageTiming``/``CostMeta`` that compare apples-to-apples
against a full debate. They run under their own ``stage`` names (``baseline_single`` /
``baseline_vote``) so their cache cells are distinct from the engine's Stage-1 cells — each
method pays for its own calls, keeping the cost comparison honest.

Both return the shared :class:`~debate.eval.metrics.MethodResult`, already scored via the
dataset verifier, so the runner collects them next to the full-debate rows unchanged.
"""

from __future__ import annotations

from collections.abc import Callable

from debate.agents import LLMClient
from debate.client import CallResult, GeminiClient
from debate.config import SETTINGS, Settings, per_problem_seed
from debate.dataset import verify
from debate.eval.metrics import MethodResult, aggregate_cost, answers_equivalent
from debate.interfaces import Problem, Solution
from debate.prompts import PERSONAS, stage1_solve

# The single-call baseline answers with the lowest-temperature, most rigorous persona so the
# comparison is "one careful model pass" vs. the full debate — not a lucky high-variance draw.
DEFAULT_SINGLE_AGENT = "formalist"
DEFAULT_N_VOTERS = 3


def _solution_answer(res: CallResult) -> str:
    """Pull the answer off a solve call. ``schema=Solution`` guarantees the parsed type."""
    parsed = res.parsed
    assert isinstance(parsed, Solution)  # client contract: schema=Solution -> Solution
    return parsed.answer


def _solve(
    client: LLMClient,
    problem: Problem,
    *,
    stage: str,
    agent_id: str,
    persona: str,
    temperature: float,
    seed: int,
) -> CallResult:
    """One structured Stage-1 solve call as a given persona (the shared baseline primitive)."""
    return client.call(
        stage=stage,
        agent_id=agent_id,
        problem_id=problem.id,
        system_prompt=PERSONAS[persona],
        user_prompt=stage1_solve(problem),
        schema=Solution,
        temperature=temperature,
        seed=seed,
    )


def run_single(
    problem: Problem,
    *,
    client: LLMClient | None = None,
    settings: Settings = SETTINGS,
    agent_id: str = DEFAULT_SINGLE_AGENT,
) -> MethodResult:
    """Single-call baseline: one solver answers the problem once."""
    client = client if client is not None else GeminiClient(settings)
    profile = next(
        (a for a in settings.agents if a.agent_id == agent_id), settings.agents[0]
    )
    seed = per_problem_seed(problem.id, settings) + profile.seed_offset
    res = _solve(
        client,
        problem,
        stage="baseline_single",
        agent_id=profile.agent_id,
        persona=profile.persona,
        temperature=profile.temperature,
        seed=seed,
    )
    answer = _solution_answer(res)
    return MethodResult(
        method="single",
        problem=problem,
        answer=answer,
        is_correct=verify(problem, answer),
        cost=aggregate_cost([res.timing]),
    )


def run_voting(
    problem: Problem,
    *,
    client: LLMClient | None = None,
    settings: Settings = SETTINGS,
    n_voters: int = DEFAULT_N_VOTERS,
) -> MethodResult:
    """Majority-vote baseline: ``n_voters`` independent solves, answer = the mode.

    Diversity comes from cycling the four configured personas (each with its own temperature)
    and a distinct, reproducible, INT32-safe per-voter seed. Answers are bucketed by semantic
    equivalence (``"5"`` == ``"5.0"``), not raw string match, and ties break deterministically.
    """
    if n_voters < 1:
        raise ValueError(f"n_voters must be >= 1, got {n_voters}")
    client = client if client is not None else GeminiClient(settings)
    profiles = settings.agents

    answers: list[str] = []
    timings = []
    for i in range(n_voters):
        profile = profiles[i % len(profiles)]
        # Synthetic per-voter id keeps the base seed < INT32 (per_problem_seed bounds it), then
        # the persona offset (<=404) stays within config's headroom. Distinct per voter, so
        # repeated personas (n_voters > 4) still resample rather than re-hitting one cache cell.
        seed = per_problem_seed(f"{problem.id}#vote{i}", settings) + profile.seed_offset
        agent_id = profile.agent_id if i < len(profiles) else f"{profile.agent_id}#{i}"
        res = _solve(
            client,
            problem,
            stage="baseline_vote",
            agent_id=agent_id,
            persona=profile.persona,
            temperature=profile.temperature,
            seed=seed,
        )
        answers.append(_solution_answer(res))
        timings.append(res.timing)

    winner = majority_answer(problem, answers)
    return MethodResult(
        method="voting",
        problem=problem,
        answer=winner,
        is_correct=verify(problem, winner),
        cost=aggregate_cost(timings),
        votes=answers,
        n_voters=len(answers),
    )


def majority_answer(
    problem: Problem,
    answers: list[str],
    *,
    equiv: Callable[[str, str], bool] | None = None,
) -> str:
    """The mode of ``answers``, bucketing by semantic equivalence (default: the verifier).

    Returns the representative (first-seen) answer of the largest bucket; ties break toward
    the earliest bucket, so the winner is deterministic for a given input order (``run_voting``
    builds ``answers`` in a fixed order). Byte-identical answers always share a bucket — even
    if ``equiv`` is non-reflexive on them — so unanimous unparseable votes still win. Bucketing
    is greedy first-match, so under a tolerance-based equivalence (REAL answers) it is only as
    well-defined as that tolerance is transitive: a cluster of near-ties within ``rel_tol`` is
    resolved by arrival order. Empty input -> empty string. ``equiv`` is injectable so tests
    (or callers) can pin the bucketing rule.
    """
    if not answers:
        return ""

    def _default_eq(a: str, b: str) -> bool:
        return answers_equivalent(problem, a, b)

    eq = equiv if equiv is not None else _default_eq

    reps: list[str] = []  # first-seen answer of each bucket
    counts: list[int] = []  # bucket sizes
    firsts: list[int] = []  # index of each bucket's first member
    for idx, ans in enumerate(answers):
        for j, rep in enumerate(reps):
            if rep == ans or eq(rep, ans):  # identical votes co-bucket even if eq isn't reflexive
                counts[j] += 1
                break
        else:  # no existing bucket matched -> open a new one
            reps.append(ans)
            counts.append(1)
            firsts.append(idx)

    best = max(range(len(reps)), key=lambda j: (counts[j], -firsts[j]))
    return reps[best]


def estimate_gate_fail_rate(
    problem: Problem,
    *,
    n: int = 5,
    client: LLMClient | None = None,
    settings: Settings = SETTINGS,
) -> float:
    """Optional difficulty-gate signal: single-model fail rate over ``n`` independent solves.

    Helps the Dataset owner fill ``Problem.gate_fail_rate``. Each attempt varies persona +
    seed so it is a genuine resample, not ``n`` cache hits of one call. Not part of the main
    sweep (it multiplies the call count) — call it explicitly when building the gate.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    client = client if client is not None else GeminiClient(settings)
    profiles = settings.agents
    fails = 0
    for i in range(n):
        profile = profiles[i % len(profiles)]
        seed = per_problem_seed(f"{problem.id}#gate{i}", settings) + profile.seed_offset
        res = _solve(
            client,
            problem,
            stage="baseline_gate",
            agent_id=f"{profile.agent_id}#{i}",
            persona=profile.persona,
            temperature=profile.temperature,
            seed=seed,
        )
        if not verify(problem, _solution_answer(res)):
            fails += 1
    return fails / n
