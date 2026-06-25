"""Disk cache for model-call outputs.

Every model call's parsed output is persisted on disk so a rerun never recomputes — the
concrete defense for the brief's "API limits won't be accepted as an excuse." The cache
key is a pure function of everything that affects the output, so editing a prompt or a
seed auto-invalidates exactly the affected cells (downstream stages embed upstream
outputs in their prompts, so changes cascade correctly).

On-disk layout (browsable, shardable):
    <CACHE_DIR>/<problem_id>/<stage>/<agent_id>__<key12>.json
Each file stores {key, model_cls, payload, usage, created_at}; ``usage`` is kept so a
cache hit can still report the tokens the original call cost (with cache_hit=True).

Test cassettes are just a checked-in cache directory using this same scheme.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from .config import CACHE_DIR, CACHE_NAMESPACE

M = TypeVar("M", bound=BaseModel)

_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def _safe(part: str) -> str:
    """Make an arbitrary id safe for a path segment."""
    return _SAFE.sub("_", part) or "_"


def make_key(
    *,
    problem_id: str,
    stage: str,
    agent_id: str,
    model: str,
    temperature: float,
    seed: int,
    thinking_budget: int,
    system_prompt: str,
    user_prompt: str,
    schema_name: str,
    namespace: str = CACHE_NAMESPACE,
) -> str:
    """Deterministic sha256 hex key. Pure function of everything that affects the output.

    ``inputs_hash`` covers the exact prompt strings + the schema *name* (not body) so that
    cosmetic docstring edits don't churn the cache; structural model changes are handled by
    bumping ``namespace`` (= CACHE_NAMESPACE) instead.
    """
    inputs_hash = hashlib.sha256(
        f"{system_prompt}\x00{user_prompt}\x00{schema_name}".encode()
    ).hexdigest()
    material = ":".join(
        [
            namespace,
            problem_id,
            stage,
            agent_id,
            model,
            f"{temperature:.6f}",
            str(seed),
            str(thinking_budget),
            inputs_hash,
        ]
    )
    return hashlib.sha256(material.encode()).hexdigest()


def cache_path(
    problem_id: str, stage: str, agent_id: str, key: str, root: Path = CACHE_DIR
) -> Path:
    """Return <root>/<problem_id>/<stage>/<agent_id>__<key[:12]>.json."""
    return (
        root
        / _safe(problem_id)
        / _safe(stage)
        / f"{_safe(agent_id)}__{key[:12]}.json"
    )


def cache_get(
    key: str,
    model_cls: type[M],
    *,
    problem_id: str,
    stage: str,
    agent_id: str,
    root: Path = CACHE_DIR,
) -> tuple[M, dict] | None:
    """Return (parsed_model, usage_dict) on hit (verifying stored full key == key), else None."""
    path = cache_path(problem_id, stage, agent_id, key, root)
    if not path.exists():
        return None
    try:
        blob = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if blob.get("key") != key:
        return None  # 12-char prefix collision (vanishingly rare) — treat as miss
    try:
        obj = model_cls.model_validate(blob["payload"])
    except Exception:
        return None
    return obj, blob.get("usage", {})


def cache_put(
    key: str,
    obj: M,
    usage: dict,
    *,
    problem_id: str,
    stage: str,
    agent_id: str,
    root: Path = CACHE_DIR,
) -> None:
    """Atomically write {key, model_cls, payload, usage, created_at} to cache_path.

    Atomic = write to <path>.tmp then os.replace, so a killed run can't leave a half file.
    """
    path = cache_path(problem_id, stage, agent_id, key, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = {
        "key": key,
        "model_cls": type(obj).__name__,
        "payload": obj.model_dump(mode="json"),
        "usage": usage,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(blob, indent=2))
    os.replace(tmp, path)
