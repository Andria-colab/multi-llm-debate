"""``debate-run-all`` — the experiment runner (this module's ``main`` is the entry point).

Runs all problems through {single-call, majority-vote, full debate}, scores every answer with
the dataset verifier, writes one reproducible JSON artifact to ``results/``, and renders the
comparison figures to ``figures/``.

Design notes:

* **Idempotent & cheap to re-run.** Every model call is cached on disk by the client, so a
  second ``debate-run-all`` is fast and free — this runner adds *no* caching of its own.
* **Robust per problem.** Each method is wrapped in try/except so one bad run (e.g. a model
  ``ModelCallError``) is logged and skipped instead of killing the whole sweep. Metrics are
  computed over whatever succeeded.
* **Replayable.** The written JSON carries the per-method rows, the raw debate records, and
  the computed report — enough to recompute every metric and re-render every figure via
  ``debate.eval.plots`` without touching the API.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path

from debate.agents import LLMClient
from debate.client import GeminiClient
from debate.config import CACHE_NAMESPACE, FIGURES_DIR, RESULTS_DIR, SETTINGS, Settings
from debate.dataset import load_problems, verify
from debate.engine import run_debate
from debate.eval.baselines import DEFAULT_N_VOTERS, run_single, run_voting
from debate.eval.metrics import (
    METHODS,
    METRICS_FILENAME,
    RESULTS_FILENAME,
    EvalReport,
    MethodResult,
    ResultsFile,
    build_report,
)
from debate.eval.plots import make_plots
from debate.interfaces import DebateRecord, Problem

log = logging.getLogger("debate.eval.runner")

# A client factory lets every problem get its own client while staying injectable for tests
# (e.g. ``lambda p: FakeClient(answer=p.ground_truth)`` to run the whole sweep offline).
ClientFactory = Callable[[Problem], LLMClient]


def _record_to_result(record: DebateRecord) -> MethodResult:
    """Project a full-debate transcript onto the shared row the baselines also produce."""
    return MethodResult(
        method="debate",
        problem=record.problem,
        answer=record.final_answer,
        is_correct=record.is_correct,
        cost=record.cost,
    )


def run_all(
    *,
    problems: Sequence[Problem] | None = None,
    client_factory: ClientFactory | None = None,
    settings: Settings = SETTINGS,
    n_voters: int = DEFAULT_N_VOTERS,
    results_dir: Path = RESULTS_DIR,
    figures_dir: Path = FIGURES_DIR,
    make_figures: bool = True,
) -> ResultsFile:
    """Run the full comparison and write ``results/`` (+ ``figures/``). Returns the artifact.

    ``client_factory`` defaults to a fresh real ``GeminiClient`` per problem; inject a
    ``FakeClient`` factory to drive the entire sweep offline. Each method runs in its own
    try/except so a single failure is logged and skipped, never aborting the sweep.
    """
    problems = list(problems) if problems is not None else load_problems()

    def _default_factory(_problem: Problem) -> LLMClient:
        return GeminiClient(settings)

    factory: ClientFactory = client_factory if client_factory is not None else _default_factory

    results_by_method: dict[str, list[MethodResult]] = {m: [] for m in METHODS}
    records: list[DebateRecord] = []

    for problem in problems:
        log.info(
            "=== %s (%s, difficulty %d) ===", problem.id, problem.category.value, problem.difficulty
        )
        # One client per problem, reused by all three methods (distinct stage names keep their
        # cache cells separate, so sharing is safe and avoids re-initializing the SDK 3x).
        client = factory(problem)

        try:
            results_by_method["single"].append(
                run_single(problem, client=client, settings=settings)
            )
        except Exception:  # noqa: BLE001 - robustness: one bad run must not kill the sweep
            log.exception("single-call baseline failed for %s", problem.id)

        try:
            results_by_method["voting"].append(
                run_voting(problem, client=client, settings=settings, n_voters=n_voters)
            )
        except Exception:  # noqa: BLE001
            log.exception("majority-vote baseline failed for %s", problem.id)

        try:
            record = run_debate(problem, client=client, settings=settings)
            record.is_correct = verify(problem, record.final_answer)
            records.append(record)
            results_by_method["debate"].append(_record_to_result(record))
        except Exception:  # noqa: BLE001
            log.exception("full debate failed for %s", problem.id)

    report = build_report(results_by_method, records)
    # Persist the full effective configuration (minus the secret key) so results.json is
    # self-describing: personas/temperatures/seeds, thinking budgets, max_output_tokens, and
    # the cache namespace all change the cached outputs, so a reader needs them to reproduce.
    settings_dump = settings.model_dump(exclude={"api_key"})
    settings_dump["n_voters"] = n_voters
    settings_dump["cache_namespace"] = CACHE_NAMESPACE
    artifact = ResultsFile(
        created_at=datetime.now(timezone.utc).isoformat(),
        n_problems=len(problems),
        settings=settings_dump,
        results=results_by_method,
        debate_records=records,
        report=report,
    )

    write_results(artifact, results_dir)
    if make_figures:
        written = make_plots(report, figures_dir)
        log.info("Wrote %d figure(s) to %s", len(written), figures_dir)

    return artifact


def write_results(artifact: ResultsFile, results_dir: Path = RESULTS_DIR) -> tuple[Path, Path]:
    """Write the full artifact (``results.json``) + a slim report (``metrics.json``)."""
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / RESULTS_FILENAME
    results_path.write_text(artifact.model_dump_json(indent=2), encoding="utf-8")
    metrics_path = results_dir / METRICS_FILENAME
    report = artifact.report if artifact.report is not None else EvalReport()
    metrics_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return results_path, metrics_path


def _log_summary(report: EvalReport) -> None:
    for m in report.methods:
        log.info(
            "%-14s accuracy=%.1f%% (%d/%d)  mean_tokens=%.0f  mean_latency=%.1fs  api_calls=%d  cache_hits=%d",
            m.method,
            100 * m.accuracy,
            m.n_correct,
            m.n,
            m.mean_tokens,
            m.mean_latency_s,
            m.total_api_calls,
            m.total_cache_hits,
        )
    if report.dynamics is not None:
        d = report.dynamics
        log.info(
            "debate dynamics: answer_change=%.0f%%  critique_accept=%.0f%%  critiques/review=%.2f",
            100 * d.answer_change_rate,
            100 * d.critique_acceptance_rate,
            d.mean_critiques_per_review,
        )


def main() -> None:
    """Console-script entry point registered as ``debate-run-all`` in pyproject.toml."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )
    log.info("Running full comparison: {single, voting, full debate} over all problems …")
    artifact = run_all()
    if artifact.report is not None:
        _log_summary(artifact.report)
    log.info("Results -> %s   Figures -> %s", RESULTS_DIR, FIGURES_DIR)


if __name__ == "__main__":
    main()
