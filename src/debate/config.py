"""Central settings. Read once at import. Env-overridable so CI / teammates don't
hard-code anything secret. Per-role temperature/seed are the persona-diversity levers
on the single base model (gemini-2.5-flash via the Gemini Developer API).
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from pydantic import BaseModel, Field

# Repo layout anchors (config.py lives at src/debate/config.py)
PKG_DIR = Path(__file__).resolve().parent  # src/debate
SRC_DIR = PKG_DIR.parent  # src
REPO_ROOT = SRC_DIR.parent  # repo root

# Load REPO_ROOT/.env into the process env BEFORE Settings reads os.getenv below, so a
# teammate's key in .env just works. override=False means a real shell env var (e.g. CI's
# DEBATE_OFFLINE=1) always wins over the file. python-dotenv is optional: without it, .env
# simply isn't auto-loaded (export the vars yourself).
try:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env", override=False)
except ImportError:  # pragma: no cover - dotenv is an optional convenience
    pass

DATA_DIR = REPO_ROOT / "data"
CACHE_DIR = Path(os.getenv("DEBATE_CACHE_DIR", str(DATA_DIR / "cache")))
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = REPO_ROOT / "figures"
LOGS_DIR = REPO_ROOT / "logs"
PROBLEMS_PATH = PKG_DIR / "dataset" / "problems.json"

# Cache namespace: bump to invalidate ALL caches at once (e.g. after a contract change).
CACHE_NAMESPACE = "1.0"


class AgentProfile(BaseModel):
    """Per-agent persona lever. Four agents, one base model.

    Diversity = persona (system prompt) + temperature + seed. ``persona`` is a key into
    ``debate.prompts.PERSONAS``; the system prompt itself lives there, not here.
    """

    agent_id: str
    persona: str
    temperature: float = Field(ge=0.0, le=2.0)
    seed_offset: int


class Settings(BaseModel):
    # --- model / API (Gemini Developer API, API-key auth) ---
    model: str = "gemini-2.5-flash"
    api_key: str = Field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))

    # --- generation defaults ---
    # 16k headroom: gemini-2.5-flash counts dynamic *thinking* tokens against this ceiling,
    # so 8k risks truncating the JSON body before the answer on hard problems.
    max_output_tokens: int = 16384
    thinking_budget: int = -1  # -1 dynamic, 0 off, 1..24576 fixed cap
    judge_thinking_budget: int = -1  # could lower for the cheap judge copy step

    # --- robustness ---
    offline: bool = Field(default_factory=lambda: bool(os.getenv("DEBATE_OFFLINE")))
    max_retries: int = 5
    backoff_base_s: float = 1.0
    backoff_max_s: float = 32.0
    request_timeout_s: float = 120.0

    # --- concurrency (keep LOW on the free tier to respect RPM) ---
    max_concurrency: int = 3

    # --- reproducibility ---
    base_seed: int = 7  # global; per-problem base = hash(base_seed, problem.id)

    # --- the four agents ---
    agents: list[AgentProfile] = Field(
        default_factory=lambda: [
            AgentProfile(
                agent_id="formalist", persona="formalist", temperature=0.15, seed_offset=101
            ),
            AgentProfile(agent_id="lateral", persona="lateral", temperature=0.85, seed_offset=202),
            AgentProfile(agent_id="checker", persona="checker", temperature=0.35, seed_offset=303),
            AgentProfile(agent_id="skeptic", persona="skeptic", temperature=0.60, seed_offset=404),
        ]
    )


SETTINGS = Settings()


# The Gemini API requires the generation seed to fit in a signed INT32. Keep the per-problem
# base comfortably under that max so adding any agent's seed_offset (max ~404) stays in range.
_INT32_MAX = 2_147_483_647
_SEED_SPACE = _INT32_MAX - 1000


def per_problem_seed(problem_id: str, settings: Settings = SETTINGS) -> int:
    """Deterministic base seed for a problem, derived from ``settings.base_seed`` + id.

    Bounded to ``_SEED_SPACE`` (< INT32 max, with headroom for the agent offsets) because the
    Gemini API rejects a seed that doesn't fit in a signed INT32. Takes ``settings`` (not the
    module global) so a caller threading a custom ``Settings`` — e.g. an independent re-run
    with a different ``base_seed`` — actually gets the seeds it asked for.
    """
    h = hashlib.sha256(f"{settings.base_seed}:{problem_id}".encode()).hexdigest()
    return int(h[:8], 16) % _SEED_SPACE


def seed_for(problem_id: str, agent_id: str, settings: Settings = SETTINGS) -> int:
    """Per-(problem, agent) seed: per-problem base + the agent's offset. Stable & reproducible.

    Looks the offset up in ``settings.agents`` (falling back to 0 for an unknown id). The
    engine seeds agents from their own ``AgentProfile.seed_offset`` directly, so it never
    hits that fallback even with a custom roster; this helper is for ad-hoc callers.
    """
    offset = next((a.seed_offset for a in settings.agents if a.agent_id == agent_id), 0)
    return per_problem_seed(problem_id, settings) + offset
