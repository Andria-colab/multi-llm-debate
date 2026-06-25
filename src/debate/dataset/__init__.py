"""Dataset & verification workstream (P2).

Public interface: ``load_problems()`` and ``verify(problem, answer)``.

STUB PHASE: these are re-exported from ``._stub``. When the real dataset lands, change the
import below to ``from .problems import load_problems`` / ``from .verifiers import verify``
(one-line swap) and delete ``_stub.py``.
"""

from ._stub import load_problems, verify

__all__ = ["load_problems", "verify"]
