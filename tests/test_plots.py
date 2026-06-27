"""Plot tests — render from a hand-built report and re-render from a saved results file.

We assert the figures get written (and the headline + cost charts exist), not their pixels.
Uses the headless Agg backend (selected inside ``debate.eval.plots``), so no display needed.
"""

from __future__ import annotations

from pathlib import Path

from debate.eval.metrics import (
    EvalReport,
    MethodResult,
    ResultsFile,
    build_report,
)
from debate.eval.plots import make_plots, replot
from debate.interfaces import AnswerType, Category, CostMeta, Problem


def _result(method: str, *, correct: bool, category: Category, difficulty: int) -> MethodResult:
    p = Problem(
        id=f"{method}_{category.value}_{difficulty}",
        text="synthetic",
        category=category,
        ground_truth="1",
        answer_type=AnswerType.INTEGER,
        difficulty=difficulty,
    )
    return MethodResult(
        method=method,
        problem=p,
        answer="1" if correct else "0",
        is_correct=correct,
        cost=CostMeta(total_tokens=100, total_thinking_tokens=60, total_output_tokens=30,
                      total_prompt_tokens=10),
    )


def _sample_report() -> EvalReport:
    rows = {
        "single": [
            _result("single", correct=True, category=Category.PHYSICS, difficulty=1),
            _result("single", correct=False, category=Category.LOGIC, difficulty=3),
        ],
        "voting": [
            _result("voting", correct=True, category=Category.PHYSICS, difficulty=1),
            _result("voting", correct=True, category=Category.LOGIC, difficulty=3),
        ],
        "debate": [
            _result("debate", correct=True, category=Category.PHYSICS, difficulty=1),
            _result("debate", correct=True, category=Category.LOGIC, difficulty=3),
        ],
    }
    return build_report(rows, [])


def test_make_plots_writes_all_figures(tmp_path: Path) -> None:
    written = make_plots(_sample_report(), tmp_path)
    names = {p.name for p in written}
    assert names == {
        "accuracy_by_method.png",
        "accuracy_by_category.png",
        "accuracy_by_difficulty.png",
        "cost_accuracy.png",
        "tokens_by_method.png",
    }
    for p in written:
        assert p.exists() and p.stat().st_size > 0


def test_make_plots_empty_report_writes_nothing(tmp_path: Path) -> None:
    assert make_plots(EvalReport(), tmp_path) == []


def test_replot_from_saved_results(tmp_path: Path) -> None:
    report = _sample_report()
    rf = ResultsFile(report=report, n_problems=2)
    results_path = tmp_path / "results.json"
    results_path.write_text(rf.model_dump_json(indent=2))

    figures_dir = tmp_path / "figs"
    written = replot(results_path, figures_dir)
    assert len(written) == 5
    assert (figures_dir / "accuracy_by_method.png").exists()


def test_replot_recomputes_report_when_absent(tmp_path: Path) -> None:
    # Older results file without an embedded report -> replot rebuilds it from the rows.
    rows = {
        "single": [_result("single", correct=True, category=Category.PHYSICS, difficulty=1)],
    }
    rf = ResultsFile(results=rows, report=None, n_problems=1)
    results_path = tmp_path / "results.json"
    results_path.write_text(rf.model_dump_json(indent=2))
    written = replot(results_path, tmp_path / "figs")
    assert (tmp_path / "figs" / "accuracy_by_method.png").exists()
    assert len(written) >= 1
