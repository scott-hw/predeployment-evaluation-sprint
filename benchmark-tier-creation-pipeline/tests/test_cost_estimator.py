"""
Tests for the cost estimator.

Covers: CostEstimate calculation, pricing lookup, load_pricing,
        DryRunReport, fallback pricing for placeholder values.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from benchmark_pipeline.cost_estimator import (
    CostEstimate,
    CallTypeStats,
    DryRunReport,
    load_pricing,
    compute_cost,
    _FALLBACK_INPUT_PER_M,
    _FALLBACK_OUTPUT_PER_M,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

PRICING_DICT = {
    "models": {
        "claude-sonnet-4-6": {
            "input_per_million_tokens_usd": 3.0,
            "output_per_million_tokens_usd": 15.0,
        },
        "claude-haiku-4-5-20251001": {
            "input_per_million_tokens_usd": 0.8,
            "output_per_million_tokens_usd": 4.0,
        },
    }
}

PLACEHOLDER_PRICING = {
    "models": {
        "claude-sonnet-4-6": {
            "input_per_million_tokens_usd": "<verify_current>",
            "output_per_million_tokens_usd": "<verify_current>",
        },
    }
}


# ---------------------------------------------------------------------------
# load_pricing
# ---------------------------------------------------------------------------

class TestLoadPricing:
    def test_loads_numeric_pricing(self):
        pricing = load_pricing(PRICING_DICT)
        sonnet = pricing["claude-sonnet-4-6"]
        assert sonnet["input_per_million_tokens_usd"] == 3.0
        assert sonnet["output_per_million_tokens_usd"] == 15.0

    def test_loads_haiku_pricing(self):
        pricing = load_pricing(PRICING_DICT)
        haiku = pricing["claude-haiku-4-5-20251001"]
        assert haiku["input_per_million_tokens_usd"] == 0.8

    def test_placeholder_strings_use_fallback(self):
        pricing = load_pricing(PLACEHOLDER_PRICING)
        sonnet = pricing["claude-sonnet-4-6"]
        assert sonnet["input_per_million_tokens_usd"] == _FALLBACK_INPUT_PER_M
        assert sonnet["output_per_million_tokens_usd"] == _FALLBACK_OUTPUT_PER_M

    def test_empty_dict_returns_empty(self):
        pricing = load_pricing({})
        assert pricing == {}

    def test_missing_models_key_returns_empty(self):
        pricing = load_pricing({"other_key": "value"})
        assert pricing == {}


# ---------------------------------------------------------------------------
# compute_cost
# ---------------------------------------------------------------------------

class TestComputeCost:
    def test_zero_tokens_zero_cost(self):
        pricing = load_pricing(PRICING_DICT)
        cost = compute_cost("claude-sonnet-4-6", 0, 0, pricing)
        assert cost == 0.0

    def test_one_million_input_tokens(self):
        pricing = load_pricing(PRICING_DICT)
        cost = compute_cost("claude-sonnet-4-6", 1_000_000, 0, pricing)
        assert abs(cost - 3.0) < 1e-9

    def test_one_million_output_tokens(self):
        pricing = load_pricing(PRICING_DICT)
        cost = compute_cost("claude-sonnet-4-6", 0, 1_000_000, pricing)
        assert abs(cost - 15.0) < 1e-9

    def test_combined_input_output(self):
        pricing = load_pricing(PRICING_DICT)
        # 500k input @ $3/M = $1.50, 100k output @ $15/M = $1.50 => $3.00
        cost = compute_cost("claude-sonnet-4-6", 500_000, 100_000, pricing)
        assert abs(cost - 3.0) < 1e-6

    def test_haiku_cheaper_than_sonnet(self):
        pricing = load_pricing(PRICING_DICT)
        tokens = 100_000
        sonnet_cost = compute_cost("claude-sonnet-4-6", tokens, tokens, pricing)
        haiku_cost = compute_cost("claude-haiku-4-5-20251001", tokens, tokens, pricing)
        assert haiku_cost < sonnet_cost

    def test_unknown_model_uses_fallback(self):
        pricing = load_pricing(PRICING_DICT)
        cost = compute_cost("unknown-model", 1_000_000, 0, pricing)
        # Uses _FALLBACK_INPUT_PER_M
        assert cost == _FALLBACK_INPUT_PER_M


# ---------------------------------------------------------------------------
# CostEstimate dataclass
# ---------------------------------------------------------------------------

class TestCostEstimate:
    def test_str_representation(self):
        est = CostEstimate(
            input_tokens=1000,
            output_tokens_estimate=256,
            cost_usd=0.0042,
            model="claude-sonnet-4-6",
        )
        s = str(est)
        assert "claude-sonnet-4-6" in s
        assert "0.0042" in s

    def test_fields_accessible(self):
        est = CostEstimate(
            input_tokens=500,
            output_tokens_estimate=128,
            cost_usd=0.001,
            model="claude-haiku-4-5-20251001",
        )
        assert est.input_tokens == 500
        assert est.output_tokens_estimate == 128
        assert est.cost_usd == 0.001
        assert est.model == "claude-haiku-4-5-20251001"


# ---------------------------------------------------------------------------
# CallTypeStats
# ---------------------------------------------------------------------------

class TestCallTypeStats:
    def test_add_accumulates(self):
        stats = CallTypeStats(call_type="generation_T1", model="claude-sonnet-4-6")
        e1 = CostEstimate(input_tokens=1000, output_tokens_estimate=256, cost_usd=0.01, model="claude-sonnet-4-6")
        e2 = CostEstimate(input_tokens=2000, output_tokens_estimate=512, cost_usd=0.02, model="claude-sonnet-4-6")
        stats.add(e1)
        stats.add(e2)
        assert stats.count == 2
        assert stats.total_input_tokens == 3000
        assert stats.total_output_tokens_estimate == 768
        assert abs(stats.total_cost_usd - 0.03) < 1e-9


# ---------------------------------------------------------------------------
# DryRunReport
# ---------------------------------------------------------------------------

class TestDryRunReport:
    def test_print_summary_runs_without_error(self, capsys):
        stats = CallTypeStats(call_type="generation_T1", model="claude-sonnet-4-6")
        stats.add(CostEstimate(100, 50, 0.001, "claude-sonnet-4-6"))
        report = DryRunReport(
            by_call_type=[stats],
            total_calls=1,
            total_input_tokens=100,
            total_output_tokens_estimate=50,
            total_cost_usd=0.001,
            wall_clock_estimate_minutes=0.02,
            warnings=["Test warning"],
        )
        report.print_summary()
        captured = capsys.readouterr()
        assert "Dry-Run Cost Estimate" in captured.out
        assert "generation_T1" in captured.out
        assert "Test warning" in captured.out

    def test_empty_report(self, capsys):
        report = DryRunReport()
        report.print_summary()
        captured = capsys.readouterr()
        assert "Dry-Run Cost Estimate" in captured.out
