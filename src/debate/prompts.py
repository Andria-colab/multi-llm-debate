"""Persona system prompts + per-stage user-prompt builders.

Diversity on the single base model comes from four *personas* (system prompts) crossed
with per-agent temperature/seed (see ``config.AgentProfile``). ``PERSONAS`` is the key →
system-prompt map that ``config.AgentProfile.persona`` indexes into.

The per-stage builders turn frozen-contract objects into the user-turn text. They are
pure (no I/O) so the cache key — a hash of system+user prompt — is stable and any wording
change auto-invalidates exactly the affected cells.

Stage 2 (review) is the graded centerpiece: its prompt + the ``Review.errors``
``min_length=1`` schema constraint mechanically forbid "looks good" sycophancy.
"""

from __future__ import annotations

from .interfaces import AnswerType, Problem, RefinedSolution, Review, Solution

# --------------------------------------------------------------------------- #
# Persona system prompts (one base model, four thinking styles)
# --------------------------------------------------------------------------- #
_COMMON = (
    "You solve hard, verifiable problems in combinatorics/number theory, physics, logic, "
    "and game theory. You always show explicit, checkable reasoning and commit to a single "
    "final answer in exactly the requested format. You never fabricate steps you did not "
    "actually perform. Respond ONLY with JSON matching the provided schema."
)

PERSONAS: dict[str, str] = {
    "formalist": (
        "You are the FORMALIST. " + _COMMON + " Your style: definitions first, precise "
        "notation, and a rigorous derivation where every step follows from a stated rule or "
        "theorem. You prefer closed forms and exact arithmetic over estimation."
    ),
    "lateral": (
        "You are the LATERAL thinker. " + _COMMON + " Your style: reframe the problem, look "
        "for a bijection, symmetry, invariant, or change of variables that collapses the "
        "work. You explore an unconventional angle, then verify it rigorously before "
        "committing."
    ),
    "checker": (
        "You are the CHECKER. " + _COMMON + " Your style: solve, then independently "
        "re-derive or sanity-check the result by a second method, units, small cases, or "
        "bounds. You distrust arithmetic until you have confirmed it twice."
    ),
    "skeptic": (
        "You are the SKEPTIC. " + _COMMON + " Your style: hunt for the trap. Surface hidden "
        "assumptions, boundary/edge cases, and misreadings of the prompt before answering. "
        "You state every assumption you rely on explicitly."
    ),
}

# Judge uses whichever persona it was assigned; this extra instruction is appended for the
# judging turn so the assigned agent adopts an impartial evaluator stance.
JUDGE_PREAMBLE = (
    "For THIS task you are an impartial JUDGE, not a solver. Evaluate candidate solutions on "
    "correctness and rigor only. Ignore verbosity, formatting, and confidence claims. Do not "
    "let the order in which candidates are presented influence you."
)


# --------------------------------------------------------------------------- #
# Answer-format guidance (keyed by AnswerType so the model emits a comparable form)
# --------------------------------------------------------------------------- #
_ANSWER_HINTS: dict[AnswerType, str] = {
    AnswerType.INTEGER: "a single integer (digits only; no commas, words, or units)",
    AnswerType.RATIONAL: "an exact rational, as a fraction like 3/8 or an exact decimal",
    AnswerType.REAL: "a decimal number (no units); give enough digits to be unambiguous",
    AnswerType.SYMBOLIC: "a closed-form expression in standard notation, e.g. n*(n+1)/2",
    AnswerType.SET: "an unordered set in braces, e.g. {2, 3, 5}",
    AnswerType.TUPLE: "an ordered tuple in parentheses, e.g. (2, 3)",
    AnswerType.STRING: "the exact answer token, with no extra words",
}


def answer_format_hint(problem: Problem) -> str:
    """One line telling the model exactly how to shape ``answer`` for this problem."""
    return _ANSWER_HINTS[problem.answer_type]


def _problem_block(problem: Problem) -> str:
    return (
        f"PROBLEM (id={problem.id}, category={problem.category.value}, "
        f"difficulty={problem.difficulty}/5):\n{problem.text}\n\n"
        f"Required answer format: {answer_format_hint(problem)}."
    )


# --------------------------------------------------------------------------- #
# Stage 0 — role self-assessment
# --------------------------------------------------------------------------- #
def stage0_self_assess(problem: Problem) -> str:
    return (
        f"{_problem_block(problem)}\n\n"
        "Before anyone solves this, assess how well-suited YOU are to two roles on THIS "
        "specific problem:\n"
        "- solver_confidence: probability you would produce a fully correct answer.\n"
        "- judge_confidence: probability you would correctly pick the best of several "
        "mixed-quality candidate solutions.\n"
        "These are independent and need not sum to 1. Rank the roles best-first in "
        "role_preferences (advisory). Give a brief, honest justification in reasoning. "
        "Do not solve the problem yet. Do not set agent_id (it is assigned externally)."
    )


# --------------------------------------------------------------------------- #
# Stage 1 — independent solve
# --------------------------------------------------------------------------- #
def stage1_solve(problem: Problem) -> str:
    return (
        f"{_problem_block(problem)}\n\n"
        "Solve this problem independently and from scratch. Show full, step-by-step "
        "reasoning that another expert could check line by line. Then give your final "
        f"answer in the required format ({answer_format_hint(problem)}) and a calibrated "
        "confidence in [0,1]. Do not set solver_id."
    )


# --------------------------------------------------------------------------- #
# Stage 2 — peer review (THE GRADED STAGE)
# --------------------------------------------------------------------------- #
def stage2_review(problem: Problem, solution: Solution) -> str:
    return (
        f"{_problem_block(problem)}\n\n"
        "Below is ANOTHER solver's solution. Review it as a demanding referee.\n\n"
        f"--- SOLUTION UNDER REVIEW ---\n{solution.reasoning}\n\n"
        f"Final answer given: {solution.answer}\n"
        "--- END SOLUTION ---\n\n"
        "Produce a rigorous critique. RULES:\n"
        "1. You MUST report at least one located, typed, severity-tagged error in `errors`. "
        "Each error needs a concrete `location` (e.g. 'Step 3' or a short quoted phrase), an "
        "`error_type`, a `severity`, and a specific `description` of what is wrong and why.\n"
        "2. Vague praise is forbidden. 'Looks good' is not a valid review. If the solution "
        "appears fully correct, you must still identify its single weakest or least-justified "
        "step and file it as a MINOR error (e.g. error_type=incomplete or "
        "unstated_assumption) explaining what would make it airtight.\n"
        "3. Be concrete and located — never hand-wave. Then fill `strengths`, `weaknesses`, "
        "`suggested_changes`, and a one-line `overall_assessment`. Do not set reviewer_id or "
        "target_solver_id."
    )


# --------------------------------------------------------------------------- #
# Stage 3 — refinement
# --------------------------------------------------------------------------- #
def stage3_refine(problem: Problem, own_solution: Solution, incoming: list[Review]) -> str:
    critiques: list[str] = []
    for rv in incoming:
        for err in rv.errors:
            critiques.append(
                f"- [{err.severity.value}/{err.error_type.value}] at {err.location}: "
                f"{err.description}"
            )
    critiques_block = "\n".join(critiques) if critiques else "(no located critiques received)"
    return (
        f"{_problem_block(problem)}\n\n"
        "Here is YOUR earlier solution:\n"
        f"--- YOUR SOLUTION ---\n{own_solution.reasoning}\n"
        f"Final answer: {own_solution.answer}\n--- END ---\n\n"
        "Peers raised these located critiques of your solution:\n"
        f"{critiques_block}\n\n"
        "Respond to EACH critique with a ChangeRecord: set `critique_location` to the "
        "location addressed, `accepted=true` if you accept and fix it (say what changed) or "
        "`accepted=false` if you rebut it (say precisely why it is wrong). Then provide your "
        "`refined_reasoning`, your final `refined_answer` in the required format "
        f"({answer_format_hint(problem)}), and an updated confidence. Change your answer only "
        "if a critique genuinely warrants it. Do not set solver_id."
    )


# --------------------------------------------------------------------------- #
# Stage 4 — judgment (candidates anonymized + shuffled by the engine)
# --------------------------------------------------------------------------- #
def stage4_judge(problem: Problem, labeled_candidates: list[tuple[str, RefinedSolution]]) -> str:
    blocks: list[str] = []
    for label, cand in labeled_candidates:
        blocks.append(
            f"=== {label} ===\n{cand.refined_reasoning}\nFinal answer: {cand.refined_answer}\n"
        )
    candidates_block = "\n".join(blocks)
    labels = ", ".join(label for label, _ in labeled_candidates)
    return (
        f"{JUDGE_PREAMBLE}\n\n{_problem_block(problem)}\n\n"
        "Here are the candidate solutions, in randomized order and anonymized:\n\n"
        f"{candidates_block}\n"
        f"Choose the single best solution. Set `winner_solver_id` to exactly one of these "
        f"labels: {labels}. Judge correctness and rigor only; ignore length and ordering. "
        "Give a calibrated confidence and reasoning. Leave `final_answer` blank — it is "
        "copied from the winner programmatically."
    )
