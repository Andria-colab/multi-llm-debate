"""Phase-0 STUB for the Dataset workstream.

A few hand-written problems + an exact-match ``verify`` so the Engine and Eval can run
end-to-end before the real 25-problem dataset and programmatic verifiers (P2) exist.

When the real implementation lands, ``dataset/__init__.py`` switches its imports from
``._stub`` to ``.problems`` / ``.verifiers`` — a one-line change — and this file is removed.
"""

from __future__ import annotations

from ..interfaces import AnswerType, Category, Problem

_PROBLEMS: list[Problem] = [
    Problem(
        id="comb_stub_001",
        text="In how many ways can a 2x4 rectangle be tiled with 1x2 dominoes?",
        category=Category.COMBINATORICS,
        ground_truth="5",
        answer_type=AnswerType.INTEGER,
        difficulty=2,
        concepts=["tiling", "fibonacci"],
        source="original",
    ),
    Problem(
        id="game_stub_001",
        text=(
            "A second-price sealed-bid (Vickrey) auction has three bidders with private "
            "values 10, 7, and 5. Bidding their dominant strategy, how much does the winner pay?"
        ),
        category=Category.GAME_THEORY,
        ground_truth="7",
        answer_type=AnswerType.INTEGER,
        difficulty=2,
        concepts=["vickrey auction", "dominant strategy"],
        source="original",
    ),
    Problem(
        id="phys_stub_001",
        text=(
            "A 2 kg block slides from rest down a 30 degree frictionless incline for 3 s "
            "(g = 9.8 m/s^2). What is its final speed in m/s?"
        ),
        category=Category.PHYSICS,
        ground_truth="14.7",
        answer_type=AnswerType.REAL,
        difficulty=2,
        concepts=["kinematics", "incline"],
        rel_tol=1e-2,
        source="original",
    ),
]


def load_problems() -> list[Problem]:
    """STUB: return a handful of hand-written problems."""
    return list(_PROBLEMS)


def verify(problem: Problem, answer: str) -> bool:
    """STUB exact-match verifier. Never raises (garbage answer -> False).

    The real verifier (P2) re-derives/brute-forces answers and normalizes both sides.
    """
    try:
        if problem.answer_type == AnswerType.REAL:
            return abs(float(answer) - float(problem.ground_truth)) <= problem.rel_tol * max(
                1.0, abs(float(problem.ground_truth))
            )
        return answer.strip() == problem.ground_truth.strip()
    except Exception:
        return False
