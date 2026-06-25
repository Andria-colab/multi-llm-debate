"""Engine entry point: ``run_debate(problem) -> DebateRecord``.

The real Stage 0→4 orchestrator. The import path ``from debate.engine import run_debate``
is the Engine workstream's stable deliverable.

    Stage 0    4 agents self-assess (solver vs judge suitability)        LLM ×4
    Stage 0.5  deterministic assignment → 3 solvers + 1 judge            pure Python
    Stage 1    3 solvers solve independently                            LLM ×3  (concurrent)
    Stage 2    each solver reviews the other two → 6 located critiques   LLM ×6  (concurrent)
    Stage 3    each solver accepts/rebuts its critiques and refines      LLM ×3  (concurrent)
    Stage 4    judge picks a winner among shuffled, anonymized answers   LLM ×1
               → the engine COPIES the winner's refined_answer as final

Every model call goes through the injected ``client`` (default: real ``GeminiClient``),
which handles caching, retries, and offline refusal. Identity (``*_id``) is stamped here,
never trusted from the model. Pass a ``FakeClient`` to run the whole pipeline offline.
"""

from __future__ import annotations

from random import Random
from typing import Callable, TypeVar

from .agents import Agent, LLMClient, build_agents
from .client import CallResult, GeminiClient
from .config import SETTINGS, Settings, per_problem_seed
from .interfaces import (
    CostMeta,
    DebateRecord,
    Judgment,
    Problem,
    RefinedSolution,
    Review,
    RoleSelfAssessment,
    Solution,
    StageTiming,
)
from .prompts import stage0_self_assess, stage1_solve, stage2_review, stage3_refine, stage4_judge
from .roles import assign_roles

T = TypeVar("T")


def _map_concurrent(
    fn: Callable[[T], CallResult], items: list[T], settings: Settings
) -> list[CallResult]:
    """Apply ``fn`` over ``items`` with bounded concurrency, preserving input order.

    Falls back to a plain sequential map when concurrency is disabled or unnecessary, so a
    single failure surfaces cleanly. Determinism is unaffected — each call's seed is fixed
    and results are reassembled by submission index, not completion order.
    """
    if settings.max_concurrency <= 1 or len(items) <= 1:
        return [fn(x) for x in items]

    from concurrent.futures import ThreadPoolExecutor

    results: list[CallResult | None] = [None] * len(items)
    with ThreadPoolExecutor(max_workers=settings.max_concurrency) as pool:
        futures = {pool.submit(fn, x): i for i, x in enumerate(items)}
        for future, idx in futures.items():
            results[idx] = future.result()  # re-raises any worker exception
    return [r for r in results if r is not None]


def _aggregate_cost(timings: list[StageTiming]) -> CostMeta:
    """Fold per-call timings into one record-level cost summary."""
    return CostMeta(
        total_latency_s=sum(t.latency_s for t in timings),
        total_prompt_tokens=sum(t.prompt_tokens for t in timings),
        total_output_tokens=sum(t.output_tokens for t in timings),
        total_thinking_tokens=sum(t.thinking_tokens for t in timings),
        total_tokens=sum(t.total_tokens for t in timings),
        n_api_calls=sum(1 for t in timings if not t.cache_hit),
        n_cache_hits=sum(1 for t in timings if t.cache_hit),
        timings=timings,
    )


def run_debate(
    problem: Problem, *, client: LLMClient | None = None, settings: Settings = SETTINGS
) -> DebateRecord:
    """Run one problem through all five stages and return a complete transcript."""
    client = client if client is not None else GeminiClient(settings)
    agents = build_agents(client, settings)
    by_id: dict[str, Agent] = {a.agent_id: a for a in agents}
    timings: list[StageTiming] = []

    # -- Stage 0: role self-assessment (×4) -------------------------------- #
    def assess(agent: Agent) -> CallResult:
        return agent.respond(
            stage="stage0_assess",
            problem_id=problem.id,
            user_prompt=stage0_self_assess(problem),
            schema=RoleSelfAssessment,
        )

    assessments: list[RoleSelfAssessment] = []
    for agent, res in zip(agents, _map_concurrent(assess, agents, settings)):
        assessment: RoleSelfAssessment = res.parsed  # type: ignore[assignment]
        assessment.agent_id = agent.agent_id  # identity is engine-owned
        assessments.append(assessment)
        timings.append(res.timing)

    # -- Stage 0.5: deterministic assignment ------------------------------- #
    assignment = assign_roles(assessments)
    solver_agents = [by_id[sid] for sid in assignment.solver_ids]
    judge_agent = by_id[assignment.judge_id]

    # -- Stage 1: independent solutions (×3) ------------------------------- #
    def solve(agent: Agent) -> CallResult:
        return agent.respond(
            stage="stage1_solve",
            problem_id=problem.id,
            user_prompt=stage1_solve(problem),
            schema=Solution,
        )

    solutions: list[Solution] = []
    for agent, res in zip(solver_agents, _map_concurrent(solve, solver_agents, settings)):
        sol: Solution = res.parsed  # type: ignore[assignment]
        sol.solver_id = agent.agent_id
        solutions.append(sol)
        timings.append(res.timing)
    solution_by_id = {s.solver_id: s for s in solutions}

    # -- Stage 2: peer review (×6) — each solver reviews the other two ----- #
    pairs = [
        (reviewer, target)
        for reviewer in solver_agents
        for target in solver_agents
        if reviewer.agent_id != target.agent_id
    ]

    def review(pair: tuple[Agent, Agent]) -> CallResult:
        reviewer, target = pair
        return reviewer.respond(
            stage="stage2_review",
            problem_id=problem.id,
            user_prompt=stage2_review(problem, solution_by_id[target.agent_id]),
            schema=Review,
        )

    reviews: list[Review] = []
    for (reviewer, target), res in zip(pairs, _map_concurrent(review, pairs, settings)):
        rv: Review = res.parsed  # type: ignore[assignment]
        rv.reviewer_id = reviewer.agent_id
        rv.target_solver_id = target.agent_id
        reviews.append(rv)
        timings.append(res.timing)

    # -- Stage 3: refinement (×3) — respond to incoming critiques ---------- #
    def refine(agent: Agent) -> CallResult:
        incoming = [r for r in reviews if r.target_solver_id == agent.agent_id]
        return agent.respond(
            stage="stage3_refine",
            problem_id=problem.id,
            user_prompt=stage3_refine(problem, solution_by_id[agent.agent_id], incoming),
            schema=RefinedSolution,
        )

    refinements: list[RefinedSolution] = []
    for agent, res in zip(solver_agents, _map_concurrent(refine, solver_agents, settings)):
        ref: RefinedSolution = res.parsed  # type: ignore[assignment]
        ref.solver_id = agent.agent_id
        refinements.append(ref)
        timings.append(res.timing)

    # -- Stage 4: judgment (×1) — shuffled + anonymized to fight position bias #
    shuffled = list(refinements)
    Random(per_problem_seed(problem.id)).shuffle(shuffled)
    labeled = [(f"candidate_{i + 1}", ref) for i, ref in enumerate(shuffled)]
    label_to_solver = {label: ref.solver_id for label, ref in labeled}

    judge_res = judge_agent.respond(
        stage="stage4_judge",
        problem_id=problem.id,
        user_prompt=stage4_judge(problem, labeled),
        schema=Judgment,
        thinking_budget=settings.judge_thinking_budget,
    )
    timings.append(judge_res.timing)
    judgment: Judgment = judge_res.parsed  # type: ignore[assignment]

    # Map the chosen anonymized label back to a real solver; copy the answer ourselves.
    winner_id = label_to_solver.get(judgment.winner_solver_id, labeled[0][1].solver_id)
    winner_refinement = next(r for r in refinements if r.solver_id == winner_id)
    judgment.winner_solver_id = winner_id
    judgment.final_answer = winner_refinement.refined_answer

    return DebateRecord(
        problem=problem,
        role_assessments=assessments,
        assignment=assignment,
        solutions=solutions,
        reviews=reviews,
        refinements=refinements,
        judgment=judgment,
        final_answer=winner_refinement.refined_answer,
        cost=_aggregate_cost(timings),
        run_seed=per_problem_seed(problem.id),
    )
