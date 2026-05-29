"""
Cost estimation for dry-run mode.

DryRunEstimator renders all prompts, counts tokens via the Anthropic
count_tokens endpoint, and reports total cost without making any
generation calls.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Default fallback pricing if llm_pricing.yaml has placeholder strings
_FALLBACK_INPUT_PER_M = 3.0
_FALLBACK_OUTPUT_PER_M = 15.0


@dataclass
class CostEstimate:
    input_tokens: int
    output_tokens_estimate: int
    cost_usd: float
    model: str

    def __str__(self) -> str:
        return (
            f"CostEstimate(model={self.model!r}, "
            f"input={self.input_tokens}, "
            f"output_est={self.output_tokens_estimate}, "
            f"cost=${self.cost_usd:.4f})"
        )


@dataclass
class CallTypeStats:
    call_type: str
    model: str
    count: int = 0
    total_input_tokens: int = 0
    total_output_tokens_estimate: int = 0
    total_cost_usd: float = 0.0

    def add(self, estimate: CostEstimate) -> None:
        self.count += 1
        self.total_input_tokens += estimate.input_tokens
        self.total_output_tokens_estimate += estimate.output_tokens_estimate
        self.total_cost_usd += estimate.cost_usd


@dataclass
class DryRunReport:
    by_call_type: list[CallTypeStats] = field(default_factory=list)
    total_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens_estimate: int = 0
    total_cost_usd: float = 0.0
    wall_clock_estimate_minutes: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def print_summary(self) -> None:
        print("\n=== Dry-Run Cost Estimate ===")
        print(f"{'Call type':<35} {'Model':<35} {'Calls':>6} {'In tok':>10} {'Out tok est':>12} {'Cost USD':>10}")
        print("-" * 115)
        for s in self.by_call_type:
            print(
                f"{s.call_type:<35} {s.model:<35} {s.count:>6} "
                f"{s.total_input_tokens:>10,} {s.total_output_tokens_estimate:>12,} "
                f"${s.total_cost_usd:>9.4f}"
            )
        print("-" * 115)
        print(
            f"{'TOTAL':<35} {'':<35} {self.total_calls:>6} "
            f"{self.total_input_tokens:>10,} {self.total_output_tokens_estimate:>12,} "
            f"${self.total_cost_usd:>9.4f}"
        )
        print(f"\nEstimated wall-clock time: {self.wall_clock_estimate_minutes:.1f} minutes")
        if self.warnings:
            print("\nWarnings:")
            for w in self.warnings:
                print(f"  ! {w}")
        print()


def load_pricing(pricing_yaml: dict) -> dict[str, dict[str, float]]:
    """
    Parse llm_pricing.yaml content into a usable dict.

    Returns {model_name: {input_per_million_tokens_usd: float, output_per_million_tokens_usd: float}}
    Falls back to placeholder values for models with string prices.
    """
    models_raw = pricing_yaml.get("models", {})
    result: dict[str, dict[str, float]] = {}
    for model_name, rates in models_raw.items():
        try:
            in_rate = float(rates.get("input_per_million_tokens_usd", _FALLBACK_INPUT_PER_M))
            out_rate = float(rates.get("output_per_million_tokens_usd", _FALLBACK_OUTPUT_PER_M))
        except (TypeError, ValueError):
            logger.warning(
                "Pricing for %r has placeholder values — using fallback $%.2f/$%.2f per M tokens",
                model_name,
                _FALLBACK_INPUT_PER_M,
                _FALLBACK_OUTPUT_PER_M,
            )
            in_rate = _FALLBACK_INPUT_PER_M
            out_rate = _FALLBACK_OUTPUT_PER_M
        result[model_name] = {
            "input_per_million_tokens_usd": in_rate,
            "output_per_million_tokens_usd": out_rate,
        }
    return result


def compute_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    pricing: dict[str, dict[str, float]],
) -> float:
    model_rates = pricing.get(model, {})
    in_rate = model_rates.get("input_per_million_tokens_usd", _FALLBACK_INPUT_PER_M)
    out_rate = model_rates.get("output_per_million_tokens_usd", _FALLBACK_OUTPUT_PER_M)
    return (input_tokens / 1_000_000 * in_rate) + (output_tokens / 1_000_000 * out_rate)


class DryRunEstimator:
    """
    Estimates pipeline cost without making any generation API calls.

    Renders all prompts, counts tokens via the Anthropic count_tokens endpoint,
    then builds a DryRunReport.
    """

    def __init__(
        self,
        llm_client_generator,
        llm_client_validator,
        pricing: dict,
        requests_per_minute: int = 50,
    ) -> None:
        self._gen = llm_client_generator
        self._val = llm_client_validator
        self._pricing = load_pricing(pricing)
        self._rpm = requests_per_minute

    async def estimate(
        self,
        prompts: list[dict],
    ) -> DryRunReport:
        """
        Parameters
        ----------
        prompts : list of dicts with keys:
            - prompt_text: str
            - call_type: str  (e.g. "generation_T1", "validation")
            - model: str
            - expected_output_tokens: int
        """
        stats: dict[tuple[str, str], CallTypeStats] = {}
        warnings: list[str] = []
        total_calls = 0
        total_cost = 0.0
        total_in = 0
        total_out = 0

        for p in prompts:
            model = p["model"]
            call_type = p["call_type"]
            expected_out = p.get("expected_output_tokens", 256)
            prompt_text = p["prompt_text"]

            client = self._gen if model == self._gen.model else self._val
            try:
                token_count = await client._count_tokens(prompt_text)
            except Exception as exc:
                warnings.append(f"Token count failed for {call_type}/{model}: {exc}")
                token_count = 500  # fallback estimate

            cost = compute_cost(model, token_count, expected_out, self._pricing)
            estimate = CostEstimate(
                input_tokens=token_count,
                output_tokens_estimate=expected_out,
                cost_usd=cost,
                model=model,
            )

            key = (call_type, model)
            if key not in stats:
                stats[key] = CallTypeStats(call_type=call_type, model=model)
            stats[key].add(estimate)

            total_calls += 1
            total_in += token_count
            total_out += expected_out
            total_cost += cost

        # Wall clock: assume RPM limit
        wall_minutes = (total_calls / self._rpm) if self._rpm > 0 else 0.0

        # Check for placeholder pricing
        for model in (self._gen.model, self._val.model):
            rates = self._pricing.get(model, {})
            if not rates:
                warnings.append(f"No pricing data found for model {model!r}")

        report = DryRunReport(
            by_call_type=list(stats.values()),
            total_calls=total_calls,
            total_input_tokens=total_in,
            total_output_tokens_estimate=total_out,
            total_cost_usd=total_cost,
            wall_clock_estimate_minutes=wall_minutes,
            warnings=warnings,
        )
        return report
