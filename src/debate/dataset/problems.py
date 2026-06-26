from __future__ import annotations

import json

from ..config import PROBLEMS_PATH
from ..interfaces import Problem


def load_problems() -> list[Problem]:
    raw = json.loads(PROBLEMS_PATH.read_text(encoding="utf-8"))
    problems = [Problem.model_validate(p) for p in raw]
    assert len({p.id for p in problems}) == len(problems), "duplicate problem ids"
    return problems
