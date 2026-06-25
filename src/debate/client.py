"""LLM client: structured Gemini calls, cache, retry/backoff, and cost/latency tracking.

``GeminiClient.call`` is the single choke point for every model call in the system. It:
  1. computes the deterministic cache key (``cache.make_key``) and returns a cache hit
     immediately (no API call) — the "rerun never recomputes" robustness guarantee;
  2. refuses live calls when ``SETTINGS.offline`` is set (CI / tests) — raising
     ``OfflineError`` on a miss instead of touching the network;
  3. otherwise calls Gemini with the Pydantic model as ``response_schema``, runs a
     validate-and-repair loop (a failed parse re-asks the model — this is what enforces
     Stage 2's anti-sycophancy ``min_length=1``), with exponential backoff on quota/5xx;
  4. records a ``StageTiming`` (latency, tokens, thinking tokens, retries, cache_hit) for
     the caller to fold into ``CostMeta``.

``google-genai`` is imported lazily inside the live path so the package imports — and the
entire offline test suite runs — without the dependency installed.

``FakeClient`` is a deterministic, dependency-free stand-in used by the offline tests and
for wiring the orchestrator before a real API key exists.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar

from pydantic import BaseModel, ValidationError

from .cache import cache_get, cache_put, make_key
from .config import SETTINGS, Settings
from .interfaces import (
    ChangeRecord,
    ErrorType,
    Judgment,
    RefinedSolution,
    Review,
    ReviewError,
    Role,
    RoleSelfAssessment,
    Severity,
    Solution,
    StageTiming,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable

M = TypeVar("M", bound=BaseModel)


class OfflineError(RuntimeError):
    """Raised when a live API call is needed but ``SETTINGS.offline`` forbids it."""


class ModelCallError(RuntimeError):
    """Raised when the model never returns a schema-valid response within max_retries."""


class _NonRetryableResponse(RuntimeError):
    """A response re-asking cannot fix: truncated (MAX_TOKENS) or blocked (SAFETY/RECITATION)."""


@dataclass
class CallResult:
    """One model call's outcome: the validated object plus its accounting."""

    parsed: BaseModel
    timing: StageTiming


def _usage_to_dict(usage: object) -> dict[str, int]:
    """Normalize a Gemini ``usage_metadata`` object into our flat token dict."""
    g = lambda name: int(getattr(usage, name, 0) or 0)  # noqa: E731
    return {
        "prompt_tokens": g("prompt_token_count"),
        "output_tokens": g("candidates_token_count"),
        "thinking_tokens": g("thoughts_token_count"),
        "total_tokens": g("total_token_count"),
    }


def _timing_from_usage(
    *, stage: str, agent_id: str, usage: dict, latency_s: float, cache_hit: bool, retries: int
) -> StageTiming:
    return StageTiming(
        stage=stage,
        agent_id=agent_id,
        latency_s=latency_s,
        prompt_tokens=int(usage.get("prompt_tokens", 0)),
        output_tokens=int(usage.get("output_tokens", 0)),
        thinking_tokens=int(usage.get("thinking_tokens", 0)),
        total_tokens=int(usage.get("total_tokens", 0)),
        cache_hit=cache_hit,
        retries=retries,
    )


class GeminiClient:
    """Concrete client backed by the Gemini Developer API (API-key auth)."""

    def __init__(self, settings: Settings = SETTINGS) -> None:
        self.settings = settings
        self._genai_client: object | None = None  # lazily constructed

    # -- lazy SDK handle ---------------------------------------------------- #
    def _client(self) -> object:
        if self._genai_client is None:
            from google import genai  # lazy: package imports without google-genai installed
            from google.genai import types

            if not self.settings.api_key:
                raise OfflineError("GEMINI_API_KEY is not set; cannot make live calls.")
            # Wire request_timeout_s (SDK wants milliseconds) so a hung connection raises and
            # trips backoff instead of blocking the run forever.
            self._genai_client = genai.Client(
                api_key=self.settings.api_key,
                http_options=types.HttpOptions(timeout=int(self.settings.request_timeout_s * 1000)),
            )
        return self._genai_client

    # -- the one choke point ------------------------------------------------ #
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
        thinking_budget: int | None = None,
    ) -> CallResult:
        tb = self.settings.thinking_budget if thinking_budget is None else thinking_budget
        key = make_key(
            problem_id=problem_id,
            stage=stage,
            agent_id=agent_id,
            model=self.settings.model,
            temperature=temperature,
            seed=seed,
            thinking_budget=tb,
            max_output_tokens=self.settings.max_output_tokens,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema_name=schema.__name__,
        )

        hit = cache_get(key, schema, problem_id=problem_id, stage=stage, agent_id=agent_id)
        if hit is not None:
            obj, usage = hit
            timing = _timing_from_usage(
                stage=stage,
                agent_id=agent_id,
                usage=usage,
                latency_s=0.0,
                cache_hit=True,
                retries=0,
            )
            return CallResult(parsed=obj, timing=timing)

        if self.settings.offline:
            raise OfflineError(
                f"Cache miss for {stage}/{agent_id}/{problem_id} and offline mode is on."
            )

        obj, usage, latency_s, retries = self._generate_with_repair(
            stage=stage,
            agent_id=agent_id,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema=schema,
            temperature=temperature,
            seed=seed,
            thinking_budget=tb,
        )
        cache_put(key, obj, usage, problem_id=problem_id, stage=stage, agent_id=agent_id)
        timing = _timing_from_usage(
            stage=stage,
            agent_id=agent_id,
            usage=usage,
            latency_s=latency_s,
            cache_hit=False,
            retries=retries,
        )
        return CallResult(parsed=obj, timing=timing)

    # -- live generation with validate/repair + backoff --------------------- #
    def _generate_with_repair(
        self,
        *,
        stage: str,
        agent_id: str,
        system_prompt: str,
        user_prompt: str,
        schema: type[M],
        temperature: float,
        seed: int,
        thinking_budget: int,
    ) -> tuple[M, dict, float, int]:
        from google.genai import errors as genai_errors  # lazy
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=temperature,
            seed=seed,
            max_output_tokens=self.settings.max_output_tokens,
            response_mime_type="application/json",
            response_schema=schema,
            thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget),
        )

        started = time.monotonic()
        contents = user_prompt
        last_err: Exception | None = None
        for attempt in range(self.settings.max_retries):
            try:
                resp = self._call_api(contents=contents, config=config)
            except genai_errors.ClientError as exc:
                # 4xx is permanent (bad key, unknown model, malformed schema) EXCEPT 429
                # RESOURCE_EXHAUSTED, the free-tier quota limit, which is transient.
                if getattr(exc, "code", None) != 429:
                    raise
                last_err = exc
                self._sleep_backoff(attempt)
                continue
            except Exception as exc:  # ServerError (5xx) / network / timeout — transient
                last_err = exc
                self._sleep_backoff(attempt)
                continue

            usage = _usage_to_dict(getattr(resp, "usage_metadata", None))
            try:
                obj = self._parse(resp, schema)
            except _NonRetryableResponse as exc:
                # MAX_TOKENS / SAFETY / RECITATION: a re-ask with the same prompt can't fix it.
                raise ModelCallError(
                    f"{stage}/{agent_id}: unrecoverable response ({exc}). If MAX_TOKENS, raise "
                    f"Settings.max_output_tokens or lower thinking_budget."
                ) from exc
            except (ValidationError, ValueError) as exc:
                # schema-invalid (e.g. an empty Review.errors) -> trip the repair loop
                last_err = exc
                contents = (
                    f"{user_prompt}\n\nYour previous response was REJECTED as invalid: "
                    f"{exc}. Return ONLY valid JSON conforming to the schema."
                )
                continue
            return obj, usage, time.monotonic() - started, attempt

        raise ModelCallError(
            f"{stage}/{agent_id}: no valid response after {self.settings.max_retries} "
            f"attempts. Last error: {last_err}"
        )

    def _call_api(self, *, contents: str, config: object) -> object:
        return self._client().models.generate_content(  # type: ignore[attr-defined]
            model=self.settings.model, contents=contents, config=config
        )

    @staticmethod
    def _finish_reason(resp: object) -> str | None:
        candidates = getattr(resp, "candidates", None) or []
        if not candidates:
            return None
        reason = getattr(candidates[0], "finish_reason", None)
        return getattr(reason, "name", str(reason)) if reason is not None else None

    @classmethod
    def _parse(cls, resp: object, schema: type[M]) -> M:
        parsed = getattr(resp, "parsed", None)
        if isinstance(parsed, schema):
            return parsed
        # No parsed object: distinguish a fixable bad-JSON case from an unrecoverable one.
        reason = cls._finish_reason(resp)
        if reason in {"MAX_TOKENS", "SAFETY", "RECITATION"}:
            feedback = getattr(resp, "prompt_feedback", None)
            raise _NonRetryableResponse(f"finish_reason={reason}; prompt_feedback={feedback}")
        text = getattr(resp, "text", None)
        if not text:
            raise ValueError(f"empty model response (finish_reason={reason})")
        return schema.model_validate_json(text)

    def _sleep_backoff(self, attempt: int) -> None:
        delay = min(self.settings.backoff_base_s * (2**attempt), self.settings.backoff_max_s)
        time.sleep(delay)


# --------------------------------------------------------------------------- #
# FakeClient — deterministic, offline, no dependencies (tests + wiring)
# --------------------------------------------------------------------------- #
class FakeClient:
    """Returns deterministic schema-valid objects without any network or cache.

    ``answer`` is echoed as every Solution/RefinedSolution answer, so a full debate run
    yields a known final answer (set it to a problem's ground truth to make ``verify`` pass,
    or to a wrong token to make a metric move). Stage-4 winner is always the first candidate
    label the engine presents.
    """

    def __init__(self, answer: str = "STUB", *, judge_pick_label: str = "candidate_1") -> None:
        self.answer = answer
        self.judge_pick_label = judge_pick_label
        self.calls: list[tuple[str, str]] = []  # (stage, agent_id) — handy for assertions

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
        thinking_budget: int | None = None,
    ) -> CallResult:
        self.calls.append((stage, agent_id))
        obj = self._build(schema)
        return CallResult(
            parsed=obj,
            timing=StageTiming(stage=stage, agent_id=agent_id, cache_hit=False),
        )

    def _build(self, schema: type[M]) -> M:
        builders: dict[type[BaseModel], Callable[[], BaseModel]] = {
            RoleSelfAssessment: lambda: RoleSelfAssessment(
                agent_id="",
                solver_confidence=0.8,
                judge_confidence=0.7,
                role_preferences=[Role.SOLVER, Role.JUDGE],
                reasoning="fake self-assessment",
            ),
            Solution: lambda: Solution(
                solver_id="", reasoning="fake reasoning", answer=self.answer, confidence=0.8
            ),
            Review: lambda: Review(
                reviewer_id="",
                target_solver_id="",
                errors=[
                    ReviewError(
                        location="Step 1",
                        error_type=ErrorType.OTHER,
                        severity=Severity.MINOR,
                        description="fake located critique",
                    )
                ],
                overall_assessment="fake assessment",
            ),
            RefinedSolution: lambda: RefinedSolution(
                solver_id="",
                changes_made=[
                    ChangeRecord(
                        critique_location="Step 1", accepted=False, response="fake rebuttal"
                    )
                ],
                refined_reasoning="fake refined reasoning",
                refined_answer=self.answer,
                confidence=0.85,
            ),
            Judgment: lambda: Judgment(
                winner_solver_id=self.judge_pick_label,
                confidence=0.9,
                reasoning="fake judgment",
            ),
        }
        try:
            return builders[schema]()  # type: ignore[return-value]
        except KeyError:  # pragma: no cover - defensive
            raise ValueError(f"FakeClient has no builder for schema {schema.__name__}") from None
