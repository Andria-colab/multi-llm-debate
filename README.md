# Multi-LLM Collaborative Debate System

Applied LLM Systems — final project. Three LLM *solvers* answer a hard, verifiable
problem independently, peer-review each other's work with located/typed critiques,
refine in response, and a fourth LLM *judge* picks the winning solution. We compare
this against a single-call baseline and a majority-vote baseline on 25 verifiable
problems across four categories (combinatorics/number-theory, physics, logic/constraint
puzzles, game theory).

> **Status:** Phase 0 (foundation). The frozen data contracts, config, cache, and stubs
> are in place; the engine, dataset, and evaluation modules are being built on top.

## Architecture

```
Stage 0    4 agents self-assess: better as Solver or Judge?      (LLM ×4)
Stage 0.5  Deterministic algorithm assigns 3 Solvers + 1 Judge   (pure Python)
Stage 1    3 Solvers solve independently                         (LLM ×3)
Stage 2    Each Solver reviews the other two → 6 critiques        (LLM ×6)   *graded*
Stage 3    Each Solver accepts/rebuts critiques and revises       (LLM ×3)
Stage 4    Judge reads refined solutions (shuffled), picks winner (LLM ×1)
           → engine COPIES the winning refined answer as the final answer
```

One `gemini-2.5-flash` model drives all four roles, differentiated by persona +
temperature + seed (a stated limitation vs. true multi-provider diversity).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"             # preferred: editable install + dev tools + `debate-run-all`
# or, plain pip:  pip install -r requirements-dev.txt   (runtime only: requirements.txt)
cp .env.example .env                # then put your Google AI Studio key in GEMINI_API_KEY
```

Get an API key at https://aistudio.google.com/apikey (free tier; has rate limits).

## Running

```bash
pytest -m "not live"        # offline test suite (no API calls)
debate-run-all              # run all 25 problems × {single, voting, full} → results/ + figures/
```

## Team / module ownership

| Person | Module | Public interface |
|---|---|---|
| Engine | `src/debate/{client,agents,prompts,roles,engine}.py` | `run_debate(problem) -> DebateRecord` |
| Dataset | `src/debate/dataset/` | `load_problems()`, `verify(problem, answer)` |
| Eval | `src/debate/eval/` | `debate-run-all` entry point, plots in `figures/` |

The shared contract lives in `src/debate/interfaces.py` (frozen — change only with team sign-off).
