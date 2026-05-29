"""
Token counting, USD cost calculation, and the cost-cap tracker.

Update PRICE_PER_1M before running to reflect current provider pricing.
Values are (input_usd, output_usd) per 1 million tokens.
"""
import logging
import threading

logger = logging.getLogger(__name__)

# USD per 1M tokens — team should verify before each run.
PRICE_PER_1M: dict[str, tuple[float, float]] = {
    "claude-sonnet-4-6":          (3.00,  15.00),
    "claude-opus-4-7":            (15.00, 75.00),
    "claude-haiku-4-5-20251001":  (0.80,   4.00),
    "gpt-4o":                     (2.50,  10.00),
    "gpt-4o-mini":                (0.15,   0.60),
    "gemini-2.5-pro":             (1.25,  10.00),
    "gemini-2.5-flash":           (0.075,  0.30),
}

# Conservative per-call estimate used before a call to check the cap.
_EST_INPUT_TOKENS = 500
_EST_OUTPUT_TOKENS = 400


def token_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    if model not in PRICE_PER_1M:
        logger.warning("No price entry for %r; cost recorded as $0", model)
        return 0.0
    in_p, out_p = PRICE_PER_1M[model]
    return (input_tokens * in_p + output_tokens * out_p) / 1_000_000


def estimate_cost(model: str) -> float:
    return token_cost(model, _EST_INPUT_TOKENS, _EST_OUTPUT_TOKENS)


class CostTracker:
    """Thread-safe running cost total with a hard cap."""

    def __init__(self, max_cost_usd: float, db_path: str | None = None):
        self.max_cost_usd = max_cost_usd
        self._lock = threading.Lock()
        if db_path:
            from store import total_cost
            self._running = total_cost(db_path)
            logger.info("Resuming — already spent $%.4f of $%.2f cap", self._running, max_cost_usd)
        else:
            self._running = 0.0

    @property
    def running_total(self) -> float:
        with self._lock:
            return self._running

    def add(self, amount: float):
        with self._lock:
            self._running += amount

    def would_exceed(self, estimate: float) -> bool:
        with self._lock:
            return (self._running + estimate) > self.max_cost_usd
