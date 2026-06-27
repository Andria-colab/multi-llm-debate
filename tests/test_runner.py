"""Runner end-to-end tests — the full sweep driven offline via FakeClient.

Asserts that ``run_all`` over the stub dataset writes the results JSON + figures, computes a
report for all three methods, is robust to a per-problem failure, and that the written
artifact round-trips (re-plottable without re-running).
"""

from __future__ import annotations

from pathlib import Path

from debate.client import CallResult, FakeClient
from debate.dataset import load_problems
from debate.eval.metrics import METHODS, ResultsFile
from debate.eval.runner import run_all, write_results
from debate.interfaces import Problem


def _factory(problem: Problem) -> FakeClient:
    """A correct-answer client per problem, so the whole offline sweep scores as correct."""
    return FakeClient(answer=problem.ground_truth)


def test_run_all_writes_results_and_figures(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    figures_dir = tmp_path / "figures"
    artifact = run_all(
        client_factory=_factory,
        results_dir=results_dir,
        figures_dir=figures_dir,
    )

    # results artifacts written
    assert (results_dir / "results.json").exists()
    assert (results_dir / "metrics.json").exists()

    # the headline figure + the cost/accuracy figure are a grading MUST
    pngs = {p.name for p in figures_dir.glob("*.png")}
    assert "accuracy_by_method.png" in pngs
    assert "cost_accuracy.png" in pngs
    assert len(pngs) >= 4

    # a report for all three methods, every problem correct under the truthful FakeClient
    assert artifact.report is not None
    by_method = artifact.report.by_method()
    assert set(by_method) == set(METHODS)
    n = len(load_problems())
    for method in METHODS:
        assert by_method[method].n == n
        assert by_method[method].accuracy == 1.0
    assert artifact.report.dynamics is not None
    assert artifact.report.dynamics.n_records == n


def test_run_all_is_robust_to_a_failing_problem(tmp_path: Path) -> None:
    problems = load_problems()
    boom_id = problems[0].id

    class BoomClient:
        def call(self, **kwargs: object) -> CallResult:
            raise RuntimeError("simulated model failure")

    def factory(problem: Problem) -> object:
        return BoomClient() if problem.id == boom_id else FakeClient(answer=problem.ground_truth)

    artifact = run_all(
        problems=problems,
        client_factory=factory,  # type: ignore[arg-type]
        results_dir=tmp_path / "results",
        figures_dir=tmp_path / "figures",
    )

    # the sweep survives the bad problem; the other problems still produce rows
    assert artifact.report is not None
    expected = len(problems) - 1
    for m in artifact.report.methods:
        assert m.n == expected
        assert m.accuracy == 1.0
    # the failing problem contributes no debate record
    assert artifact.report.dynamics is not None
    assert artifact.report.dynamics.n_records == expected


def test_results_file_round_trips(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    artifact = run_all(
        client_factory=_factory,
        results_dir=results_dir,
        figures_dir=tmp_path / "figures",
        make_figures=False,
    )
    reloaded = ResultsFile.model_validate_json((results_dir / "results.json").read_text())
    assert reloaded.n_problems == artifact.n_problems
    assert len(reloaded.debate_records) == len(load_problems())
    assert reloaded.report is not None
    assert {m.method for m in reloaded.report.methods} == set(METHODS)


def test_write_results_creates_dir(tmp_path: Path) -> None:
    nested = tmp_path / "deep" / "nested" / "results"
    rf = ResultsFile(n_problems=0)
    results_path, metrics_path = write_results(rf, nested)
    assert results_path.exists()
    assert metrics_path.exists()
