"""Comparison figures for the writeup — an explicit grading MUST.

Renders the headline charts from an :class:`~debate.eval.metrics.EvalReport`:

* **accuracy by method** — single vs majority-vote vs full debate (the headline);
* **accuracy by category** and **by difficulty**, grouped per method;
* **cost / accuracy trade-off** — mean tokens per problem vs accuracy (what the extra
  spend buys);
* **token mix by method** — stacked prompt / output / thinking (thinking dominates cost).

Plotting is intentionally decoupled from the runner: :func:`make_plots` takes a report
object (no I/O beyond saving), and :func:`replot` reloads the saved ``results/`` artifact and
re-renders — so figures can be regenerated without re-calling the API. ``python -m
debate.eval.plots`` does exactly that.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: render to files, never open a window (CI/servers safe)

import matplotlib.pyplot as plt  # noqa: E402  (must follow the backend selection)

from debate.config import FIGURES_DIR, RESULTS_DIR  # noqa: E402
from debate.eval.metrics import (  # noqa: E402
    METHOD_LABELS,
    EvalReport,
    ResultsFile,
    build_report,
)

# Stable per-method colours so every figure reads consistently.
_METHOD_COLORS: dict[str, str] = {
    "single": "#4C72B0",
    "voting": "#DD8452",
    "debate": "#55A868",
}
_FALLBACK_COLOR = "#888888"


def _label(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def _color(method: str) -> str:
    return _METHOD_COLORS.get(method, _FALLBACK_COLOR)


def _save(fig: plt.Figure, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Individual figures
# --------------------------------------------------------------------------- #
def _plot_accuracy_by_method(report: EvalReport, out_dir: Path) -> Path | None:
    methods = report.methods
    if not methods:
        return None
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = [_label(m.method) for m in methods]
    values = [m.accuracy for m in methods]
    colors = [_color(m.method) for m in methods]
    bars = ax.bar(labels, values, color=colors)
    for bar, m in zip(bars, methods):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{m.accuracy:.0%}\n({m.n_correct}/{m.n})",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy by method")
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, out_dir / "accuracy_by_method.png")


def _grouped_accuracy(
    report: EvalReport,
    out_dir: Path,
    *,
    field: str,
    keys: list[str],
    title: str,
    xlabel: str,
    filename: str,
    tick_labels: list[str] | None = None,
) -> Path | None:
    methods = report.methods
    if not methods or not keys:
        return None

    import numpy as np

    x = np.arange(len(keys))
    n_methods = len(methods)
    width = 0.8 / n_methods
    fig, ax = plt.subplots(figsize=(max(6, 1.3 * len(keys)), 4))
    for i, m in enumerate(methods):
        table: dict[str, float] = getattr(m, field)
        values = [table.get(k, 0.0) for k in keys]
        offset = (i - (n_methods - 1) / 2) * width
        ax.bar(x + offset, values, width=width, label=_label(m.method), color=_color(m.method))
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels if tick_labels is not None else keys)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, out_dir / filename)


def _plot_accuracy_by_category(report: EvalReport, out_dir: Path) -> Path | None:
    keys = sorted({k for m in report.methods for k in m.accuracy_by_category})
    return _grouped_accuracy(
        report,
        out_dir,
        field="accuracy_by_category",
        keys=keys,
        title="Accuracy by category",
        xlabel="Category",
        filename="accuracy_by_category.png",
    )


def _plot_accuracy_by_difficulty(report: EvalReport, out_dir: Path) -> Path | None:
    # difficulty keys are stringified ints -> sort numerically, not lexically
    keys = sorted(
        {k for m in report.methods for k in m.accuracy_by_difficulty},
        key=lambda k: int(k),
    )
    return _grouped_accuracy(
        report,
        out_dir,
        field="accuracy_by_difficulty",
        keys=keys,
        title="Accuracy by difficulty",
        xlabel="Difficulty (1=easy .. 5=hard)",
        filename="accuracy_by_difficulty.png",
        tick_labels=keys,
    )


def _plot_cost_accuracy(report: EvalReport, out_dir: Path) -> Path | None:
    methods = report.methods
    if not methods:
        return None
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for m in methods:
        ax.scatter(
            m.mean_tokens,
            m.accuracy,
            s=140,
            color=_color(m.method),
            edgecolors="black",
            linewidths=0.5,
            zorder=3,
        )
        ax.annotate(
            _label(m.method),
            (m.mean_tokens, m.accuracy),
            textcoords="offset points",
            xytext=(8, 6),
            fontsize=9,
        )
    ax.set_xlabel("Mean total tokens per problem")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(left=0)
    ax.set_title("Cost / accuracy trade-off")
    ax.grid(alpha=0.3)
    return _save(fig, out_dir / "cost_accuracy.png")


def _plot_tokens_by_method(report: EvalReport, out_dir: Path) -> Path | None:
    methods = report.methods
    if not methods:
        return None
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = [_label(m.method) for m in methods]
    prompt = [m.mean_prompt_tokens for m in methods]
    output = [m.mean_output_tokens for m in methods]
    thinking = [m.mean_thinking_tokens for m in methods]
    ax.bar(labels, prompt, label="prompt", color="#8C8C8C")
    ax.bar(labels, output, bottom=prompt, label="output", color="#4C72B0")
    bottom2 = [p + o for p, o in zip(prompt, output)]
    ax.bar(labels, thinking, bottom=bottom2, label="thinking", color="#C44E52")
    ax.set_ylabel("Mean tokens per problem")
    ax.set_title("Token mix by method (thinking dominates cost)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, out_dir / "tokens_by_method.png")


# --------------------------------------------------------------------------- #
# Public entry points
# --------------------------------------------------------------------------- #
def make_plots(report: EvalReport, figures_dir: Path = FIGURES_DIR) -> list[Path]:
    """Render every figure from ``report`` into ``figures_dir``; return the paths written.

    Skips any figure with no data to draw (returns fewer paths) rather than erroring, so a
    partial run still produces what it can.
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    candidates = [
        _plot_accuracy_by_method(report, figures_dir),
        _plot_accuracy_by_category(report, figures_dir),
        _plot_accuracy_by_difficulty(report, figures_dir),
        _plot_cost_accuracy(report, figures_dir),
        _plot_tokens_by_method(report, figures_dir),
    ]
    return [p for p in candidates if p is not None]


def replot(
    results_path: Path = RESULTS_DIR / "results.json",
    figures_dir: Path = FIGURES_DIR,
) -> list[Path]:
    """Reload a saved ``results/`` artifact and re-render the figures — no API calls.

    Uses the embedded report when present, else recomputes it from the saved rows/records so
    a metrics-code change re-renders correctly from old results.
    """
    rf = ResultsFile.model_validate_json(results_path.read_text(encoding="utf-8"))
    report = rf.report if rf.report is not None else build_report(rf.results, rf.debate_records)
    return make_plots(report, figures_dir)


if __name__ == "__main__":  # `python -m debate.eval.plots` -> re-render from results/
    written = replot()
    print(f"Wrote {len(written)} figure(s):")
    for p in written:
        print(f"  {p}")
