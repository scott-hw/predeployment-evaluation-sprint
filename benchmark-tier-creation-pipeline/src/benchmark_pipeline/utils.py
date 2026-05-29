"""
Utility functions used across the pipeline.
"""

from __future__ import annotations

import logging
import os
import random
import re
import sys
from pathlib import Path
from typing import Optional

from .schemas import LanguageExample


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(level: str = "INFO") -> None:
    """Configure root logger with a clean format."""
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stderr)],
    )
    # Silence noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("anthropic").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# .env helper
# ---------------------------------------------------------------------------

def load_dotenv_from_project(base_dir: Optional[Path] = None) -> None:
    """Load .env from the benchmark-pipeline root (or given directory)."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    search = base_dir or Path(__file__).resolve().parents[3]
    env_file = search / ".env"
    if env_file.exists():
        load_dotenv(env_file)
        logging.getLogger(__name__).debug("Loaded .env from %s", env_file)
    else:
        # Try current working directory
        cwd_env = Path.cwd() / ".env"
        if cwd_env.exists():
            load_dotenv(cwd_env)


# ---------------------------------------------------------------------------
# String helpers
# ---------------------------------------------------------------------------

_FENCE_PATTERN = re.compile(
    r"^```(?:json|JSON)?\s*\n(.*?)\n```\s*$",
    re.DOTALL,
)


def strip_markdown_fences(text: str) -> str:
    """
    Remove ```json ... ``` or ``` ... ``` fences from an LLM response.

    Returns the inner content, stripped of leading/trailing whitespace.
    """
    stripped = text.strip()
    m = _FENCE_PATTERN.match(stripped)
    if m:
        return m.group(1).strip()
    # Also handle case where the response just starts with ```
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return stripped


# ---------------------------------------------------------------------------
# Language example sampling
# ---------------------------------------------------------------------------

def sample_examples(
    examples: list[LanguageExample],
    tier: str,
    n: int,
    seed: Optional[int] = None,
) -> list[LanguageExample]:
    """
    Sample up to `n` LanguageExamples for the given tier.

    Uses `seed` for reproducibility; if seed is None sampling is random.
    Never raises — returns fewer items if tier has fewer than n examples.
    """
    tier_examples = [e for e in examples if e.language_tier == tier]
    if not tier_examples:
        return []
    rng = random.Random(seed)
    k = min(n, len(tier_examples))
    return rng.sample(tier_examples, k)


# ---------------------------------------------------------------------------
# Topic distribution
# ---------------------------------------------------------------------------

def compute_topic_distribution(items: list) -> dict[str, int]:
    """
    Return a count of items per administrative topic tag.

    Each item may have multiple tags; each tag is counted once per item.
    """
    distribution: dict[str, int] = {}
    for item in items:
        tags = item.tags.get("administrative_topic_tags", []) if hasattr(item, "tags") else []
        for tag in tags:
            distribution[tag] = distribution.get(tag, 0) + 1
    return distribution


# ---------------------------------------------------------------------------
# Benchmark item ID
# ---------------------------------------------------------------------------

def format_benchmark_item_id(
    source_packet_id: str,
    tier: str,
    index: int,
) -> str:
    """
    Format: {source_packet_id}__{language_tier}__{NNNN}

    Index is 1-based and zero-padded to 4 digits.
    """
    return f"{source_packet_id}__{tier}__{index:04d}"
