"""Dataset & verification workstream (P2).

Public interface: ``load_problems()`` and ``verify(problem, answer)``.

``load_problems`` reads the 25+ real problems from ``problems.json``; ``verify`` does
programmatic, per-``answer_type`` checking that never raises.
"""

from .problems import load_problems
from .verifiers import verify

__all__ = ["load_problems", "verify"]
