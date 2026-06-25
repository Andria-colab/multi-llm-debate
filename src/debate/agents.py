"""Agent layer: bind a persona + temperature + per-(problem,agent) seed to the client.

An ``Agent`` is the thin object the orchestrator talks to. It owns its persona system
prompt and its generation levers, and exposes one method — ``respond`` — that builds the
deterministic seed for this (problem, agent) and delegates to the client's single call
choke point. Identity is engine-owned: ``respond`` returns the parsed object untouched and
the orchestrator stamps the ``*_id`` fields.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, TypeVar

from pydantic import BaseModel

from .config import SETTINGS, AgentProfile, Settings, seed_for
from .prompts import PERSONAS

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .client import CallResult

M = TypeVar("M", bound=BaseModel)


class LLMClient(Protocol):
    """The surface every client (real ``GeminiClient`` or ``FakeClient``) implements."""

    def call(
        self,
        *,
        stage: str,
        agent_id: str,
        problem_id: str,
        system_prompt: str,
        user_prompt: str,
        schema: type[M],
        temperature: float,
        seed: int,
        thinking_budget: int | None = ...,
    ) -> CallResult: ...


class Agent:
    """One persona bound to a client. Diversity = persona + temperature + seed."""

    def __init__(self, profile: AgentProfile, client: LLMClient) -> None:
        self.profile = profile
        self.client = client
        self.system_prompt = PERSONAS[profile.persona]

    @property
    def agent_id(self) -> str:
        return self.profile.agent_id

    def respond(
        self,
        *,
        stage: str,
        problem_id: str,
        user_prompt: str,
        schema: type[M],
        thinking_budget: int | None = None,
    ) -> CallResult:
        """Run one structured call as this agent. Seed is derived deterministically."""
        return self.client.call(
            stage=stage,
            agent_id=self.agent_id,
            problem_id=problem_id,
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            schema=schema,
            temperature=self.profile.temperature,
            seed=seed_for(problem_id, self.agent_id),
            thinking_budget=thinking_budget,
        )


def build_agents(client: LLMClient, settings: Settings = SETTINGS) -> list[Agent]:
    """Construct the four configured agents bound to ``client``."""
    return [Agent(profile, client) for profile in settings.agents]
