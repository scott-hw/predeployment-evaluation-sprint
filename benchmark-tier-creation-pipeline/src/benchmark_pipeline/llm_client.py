"""
LLM client supporting both Anthropic and OpenAI providers.

Key design points:
- Provider selected via `provider` parameter: "anthropic" or "openai"
- Anthropic defaults: generator=claude-sonnet-4-6, validator=claude-haiku-4-5-20251001
- OpenAI defaults: generator=gpt-4o, validator=gpt-4o-mini
- Cache check before every call
- Rate limiter wraps every call
- Retry on 429 / 529
- Prompt caching via cache_control on Anthropic calls (no-op for OpenAI)
- Streaming for generation calls
- Never logs API keys
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Optional

from .cache import LLMCache
from .rate_limiter import RateLimiter, RateLimiterConfig
from .utils import strip_markdown_fences

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Provider-specific model defaults
# ---------------------------------------------------------------------------

ANTHROPIC_GENERATOR_MODEL = "claude-sonnet-4-6"
ANTHROPIC_VALIDATOR_MODEL = "claude-haiku-4-5-20251001"

OPENAI_GENERATOR_MODEL = "gpt-4o"
OPENAI_VALIDATOR_MODEL = "gpt-4o-mini"

# Aliases used throughout the rest of the codebase (default to Anthropic)
GENERATOR_MODEL = ANTHROPIC_GENERATOR_MODEL
VALIDATOR_MODEL = ANTHROPIC_VALIDATOR_MODEL

# Default token budgets
DEFAULT_GENERATION_MAX_TOKENS = 512
DEFAULT_VALIDATION_MAX_TOKENS = 768


def default_models_for_provider(provider: str) -> tuple[str, str]:
    """Return (generator_model, validator_model) defaults for a provider."""
    if provider == "openai":
        return OPENAI_GENERATOR_MODEL, OPENAI_VALIDATOR_MODEL
    return ANTHROPIC_GENERATOR_MODEL, ANTHROPIC_VALIDATOR_MODEL


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0

    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class LLMResponse:
    content: str
    parsed_json: Optional[dict] = None
    usage: TokenUsage = field(default_factory=TokenUsage)
    from_cache: bool = False
    model: str = ""
    stop_reason: str = ""


class LLMClient:
    """
    Async wrapper supporting both the Anthropic and OpenAI Python SDKs.

    Parameters
    ----------
    model : str
        Model string, e.g. "claude-sonnet-4-6" or "gpt-4o".
    provider : str
        "anthropic" (default) or "openai".
    temperature : float
        Sampling temperature.
    rate_limiter : RateLimiter
        Shared rate limiter instance.
    cache : LLMCache
        Disk cache instance. Pass None to disable caching.
    pricing : dict
        Pricing config loaded from llm_pricing.yaml.
    dry_run : bool
        If True, estimate token counts only — no actual completions.
    max_tokens : int
        Default max output tokens for this client.
    """

    def __init__(
        self,
        model: str,
        provider: str = "anthropic",
        temperature: float = 1.0,
        rate_limiter: Optional[RateLimiter] = None,
        cache: Optional[LLMCache] = None,
        pricing: Optional[dict] = None,
        dry_run: bool = False,
        max_tokens: int = DEFAULT_GENERATION_MAX_TOKENS,
    ) -> None:
        if provider not in ("anthropic", "openai"):
            raise ValueError(f"provider must be 'anthropic' or 'openai', got {provider!r}")

        self.model = model
        self.provider = provider
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.dry_run = dry_run
        self.pricing = pricing or {}

        self._client = None       # sync client (Anthropic only, for count_tokens)
        self._async_client = None # async client

        if provider == "anthropic":
            self._init_anthropic(dry_run)
        else:
            self._init_openai(dry_run)

        self._rate_limiter = rate_limiter or RateLimiter()
        self._cache = cache
        self._total_usage = TokenUsage()

    def _init_anthropic(self, dry_run: bool) -> None:
        import anthropic as _anthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key and not dry_run:
            raise ValueError(
                "ANTHROPIC_API_KEY environment variable is not set. "
                "Copy .env.example to .env and fill in your key."
            )
        if api_key:
            self._client = _anthropic.Anthropic(api_key=api_key)
            self._async_client = _anthropic.AsyncAnthropic(api_key=api_key)

    def _init_openai(self, dry_run: bool) -> None:
        try:
            import openai as _openai
        except ImportError:
            raise ImportError(
                "openai package is required for provider='openai'. "
                "Install it with: pip install openai"
            )
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key and not dry_run:
            raise ValueError(
                "OPENAI_API_KEY environment variable is not set. "
                "Copy .env.example to .env and fill in your key."
            )
        if api_key:
            self._async_client = _openai.AsyncOpenAI(api_key=api_key)

    # ------------------------------------------------------------------
    # Public async API
    # ------------------------------------------------------------------

    async def generate_json(
        self,
        prompt: str,
        schema_hint: Optional[dict] = None,
        max_tokens: Optional[int] = None,
        cached_system_prefix: Optional[str] = None,
    ) -> dict:
        """
        Make an LLM call, return parsed JSON dict.

        Parameters
        ----------
        prompt : str
            Full user prompt text.
        schema_hint : dict, optional
            Not used for the API call; informational only.
        max_tokens : int, optional
            Override instance max_tokens for this call.
        cached_system_prefix : str, optional
            Prepended as a system message. On Anthropic, activates prompt
            caching via cache_control. On OpenAI, sent as a plain system message.
        """
        effective_max_tokens = max_tokens or self.max_tokens
        cache_key = self._cache.make_key(
            self.model, self.temperature, effective_max_tokens, prompt
        ) if self._cache else None

        # 1. Cache check
        if self._cache and cache_key:
            entry = self._cache.get(cache_key)
            if entry:
                return entry["response"]

        # 2. Dry-run: count tokens only
        if self.dry_run:
            token_count = await self._count_tokens(prompt, cached_system_prefix)
            logger.info(
                "[dry-run] %s | ~%d input tokens (no API call)",
                self.model,
                token_count,
            )
            raise DryRunException(
                f"Dry-run mode: would send ~{token_count} tokens to {self.model}"
            )

        # 3. Real call with rate limiter + backoff
        async def _call():
            if self.provider == "anthropic":
                return await self._anthropic_completion(
                    prompt=prompt,
                    cached_system_prefix=cached_system_prefix,
                    max_tokens=effective_max_tokens,
                )
            else:
                return await self._openai_completion(
                    prompt=prompt,
                    system_prefix=cached_system_prefix,
                    max_tokens=effective_max_tokens,
                )

        raw_text, usage = await self._rate_limiter.execute_with_backoff(_call)

        # 4. Parse JSON
        cleaned = strip_markdown_fences(raw_text)
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            logger.error(
                "JSON parse error from %s: %s\nRaw response:\n%s",
                self.model,
                exc,
                raw_text[:500],
            )
            raise

        # 5. Store in cache
        if self._cache and cache_key:
            self._cache.set(
                cache_key,
                parsed,
                metadata={
                    "model": self.model,
                    "provider": self.provider,
                    "temperature": self.temperature,
                    "max_tokens": effective_max_tokens,
                },
            )

        # 6. Accumulate usage
        self._total_usage.input_tokens += usage.input_tokens
        self._total_usage.output_tokens += usage.output_tokens
        self._total_usage.cache_read_input_tokens += usage.cache_read_input_tokens
        self._total_usage.cache_creation_input_tokens += usage.cache_creation_input_tokens

        logger.debug(
            "%s | in=%d out=%d cache_read=%d cache_create=%d",
            self.model,
            usage.input_tokens,
            usage.output_tokens,
            usage.cache_read_input_tokens,
            usage.cache_creation_input_tokens,
        )
        return parsed

    async def estimate_call(
        self,
        prompt: str,
        expected_output_tokens: int = 256,
        cached_system_prefix: Optional[str] = None,
    ) -> "CostEstimate":
        """Estimate cost for a call without making it."""
        from .cost_estimator import CostEstimate

        input_tokens = await self._count_tokens(prompt, cached_system_prefix)
        return CostEstimate(
            input_tokens=input_tokens,
            output_tokens_estimate=expected_output_tokens,
            cost_usd=self._compute_cost(input_tokens, expected_output_tokens),
            model=self.model,
        )

    def get_total_usage(self) -> TokenUsage:
        return self._total_usage

    # ------------------------------------------------------------------
    # Anthropic backend
    # ------------------------------------------------------------------

    async def _anthropic_completion(
        self,
        prompt: str,
        cached_system_prefix: Optional[str],
        max_tokens: int,
    ) -> tuple[str, TokenUsage]:
        """Stream an Anthropic completion."""
        collected: list[str] = []
        usage = TokenUsage()

        messages = self._build_messages(prompt)
        system_blocks = self._build_anthropic_system(cached_system_prefix)

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if system_blocks:
            kwargs["system"] = system_blocks
        if self.temperature != 1.0:
            kwargs["temperature"] = self.temperature

        async with self._async_client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                collected.append(text)
            final = await stream.get_final_message()
            if final.usage:
                usage.input_tokens = final.usage.input_tokens
                usage.output_tokens = final.usage.output_tokens
                usage.cache_read_input_tokens = getattr(
                    final.usage, "cache_read_input_tokens", 0
                ) or 0
                usage.cache_creation_input_tokens = getattr(
                    final.usage, "cache_creation_input_tokens", 0
                ) or 0

        return "".join(collected), usage

    # ------------------------------------------------------------------
    # OpenAI backend
    # ------------------------------------------------------------------

    async def _openai_completion(
        self,
        prompt: str,
        system_prefix: Optional[str],
        max_tokens: int,
    ) -> tuple[str, TokenUsage]:
        """Call the OpenAI chat completions endpoint."""
        messages: list[dict] = []
        if system_prefix:
            messages.append({"role": "system", "content": system_prefix})
        messages.append({"role": "user", "content": prompt})

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": messages,
            "temperature": self.temperature,
        }

        response = await self._async_client.chat.completions.create(**kwargs)
        text = response.choices[0].message.content or ""
        usage = TokenUsage(
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
        )
        return text, usage

    # ------------------------------------------------------------------
    # Token counting
    # ------------------------------------------------------------------

    async def _count_tokens(
        self,
        prompt: str,
        cached_system_prefix: Optional[str] = None,
    ) -> int:
        """Estimate input token count. Falls back to char-based estimate if no client."""
        if self._async_client is None:
            # Offline dry-run fallback: ~4 chars per token
            total_chars = len(prompt)
            if cached_system_prefix:
                total_chars += len(cached_system_prefix)
            return max(1, total_chars // 4)

        if self.provider == "anthropic":
            return await self._anthropic_count_tokens(prompt, cached_system_prefix)
        else:
            return await self._openai_count_tokens(prompt, cached_system_prefix)

    async def _anthropic_count_tokens(
        self,
        prompt: str,
        cached_system_prefix: Optional[str],
    ) -> int:
        messages = self._build_messages(prompt)
        system_blocks = self._build_anthropic_system(cached_system_prefix)

        kwargs: dict[str, Any] = {"model": self.model, "messages": messages}
        if system_blocks:
            kwargs["system"] = system_blocks

        if self._client is None:
            total_chars = len(prompt) + (len(cached_system_prefix) if cached_system_prefix else 0)
            return max(1, total_chars // 4)

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self._client.messages.count_tokens(**kwargs),
        )
        return response.input_tokens

    async def _openai_count_tokens(
        self,
        prompt: str,
        system_prefix: Optional[str],
    ) -> int:
        """Estimate token count for OpenAI using tiktoken."""
        try:
            import tiktoken
            enc = tiktoken.encoding_for_model(self.model)
        except (ImportError, KeyError):
            # Fall back to char estimate
            total = len(prompt) + (len(system_prefix) if system_prefix else 0)
            return max(1, total // 4)

        total = len(enc.encode(prompt))
        if system_prefix:
            total += len(enc.encode(system_prefix))
        return total

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_messages(prompt: str) -> list[dict]:
        return [{"role": "user", "content": prompt}]

    @staticmethod
    def _build_anthropic_system(cached_prefix: Optional[str]) -> list[dict]:
        if not cached_prefix:
            return []
        return [
            {
                "type": "text",
                "text": cached_prefix,
                "cache_control": {"type": "ephemeral"},
            }
        ]

    def _compute_cost(self, input_tokens: int, output_tokens: int) -> float:
        model_pricing = self.pricing.get("models", {}).get(self.model, {})
        in_rate = model_pricing.get("input_per_million_tokens_usd", 0)
        out_rate = model_pricing.get("output_per_million_tokens_usd", 0)
        if isinstance(in_rate, str) or isinstance(out_rate, str):
            return 0.0  # placeholder pricing
        return (input_tokens / 1_000_000 * in_rate) + (output_tokens / 1_000_000 * out_rate)


# ---------------------------------------------------------------------------
# Dry-run sentinel exception
# ---------------------------------------------------------------------------

class DryRunException(Exception):
    """Raised in dry-run mode instead of making actual API calls."""
    pass
