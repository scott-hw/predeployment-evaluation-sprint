"""
Export benchmark items to JSONL, CSV, and JSON manifest.

Also writes rejected_candidates.jsonl for items that failed validation.
"""

from __future__ import annotations

import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .schemas import BenchmarkItem
from .utils import compute_topic_distribution

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# JSONL
# ---------------------------------------------------------------------------

def write_benchmark_jsonl(
    items: list[BenchmarkItem],
    output_dir: Path,
    filename: str = "benchmark.jsonl",
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / filename

    with out_path.open("w", encoding="utf-8") as fh:
        for item in items:
            fh.write(item.model_dump_json() + "\n")

    logger.info("Wrote %d benchmark items to %s", len(items), out_path)
    return out_path


# ---------------------------------------------------------------------------
# CSV (flattened)
# ---------------------------------------------------------------------------

_FLAT_FIELDS = [
    "benchmark_item_id",
    "source_packet_id",
    "language_tier",
    "question_text",
    "official_question",
    "normalized_resident_need",
    # Flattened source_answer fields
    "source_answer__official_answer",
    "source_answer__required_answer_elements",
    "source_answer__forbidden_claims",
    # Flattened tags
    "tags__administrative_topic_tags",
    "tags__source_agency",
    "tags__source_type",
    "tags__source_url",
    "tags__as_of_date",
    # Metadata
    "generation_metadata__generator_model",
    "generation_metadata__validator_model",
    "generation_metadata__validation_status",
    "generation_metadata__escalated",
    "generation_metadata__style_tags",
    "generation_metadata__language_example_ids_used",
    "generation_metadata__seed",
]


def _flatten_item(item: BenchmarkItem) -> dict[str, str]:
    """Flatten a BenchmarkItem to a flat dict for CSV output."""
    d: dict[str, str] = {}

    def _str(v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, list):
            return "; ".join(str(x) for x in v)
        return str(v)

    d["benchmark_item_id"] = item.benchmark_item_id
    d["source_packet_id"] = item.source_packet_id
    d["language_tier"] = item.language_tier
    d["question_text"] = item.question_text
    d["official_question"] = _str(item.official_question)
    d["normalized_resident_need"] = item.normalized_resident_need

    sa = item.source_answer
    d["source_answer__official_answer"] = _str(sa.get("official_answer"))
    d["source_answer__required_answer_elements"] = _str(sa.get("required_answer_elements"))
    d["source_answer__forbidden_claims"] = _str(sa.get("forbidden_claims"))

    tags = item.tags
    d["tags__administrative_topic_tags"] = _str(tags.get("administrative_topic_tags"))
    d["tags__source_agency"] = _str(tags.get("source_agency"))
    d["tags__source_type"] = _str(tags.get("source_type"))
    d["tags__source_url"] = _str(tags.get("source_url"))
    d["tags__as_of_date"] = _str(tags.get("as_of_date"))

    meta = item.generation_metadata
    d["generation_metadata__generator_model"] = _str(meta.get("generator_model"))
    d["generation_metadata__validator_model"] = _str(meta.get("validator_model"))
    d["generation_metadata__validation_status"] = _str(meta.get("validation_status"))
    d["generation_metadata__escalated"] = _str(meta.get("escalated"))
    d["generation_metadata__style_tags"] = _str(meta.get("style_tags"))
    d["generation_metadata__language_example_ids_used"] = _str(meta.get("language_example_ids_used"))
    d["generation_metadata__seed"] = _str(meta.get("seed"))

    return d


def write_benchmark_csv(
    items: list[BenchmarkItem],
    output_dir: Path,
    filename: str = "benchmark.csv",
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / filename

    if not items:
        logger.warning("No items to write to CSV.")
        with out_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=_FLAT_FIELDS)
            writer.writeheader()
        return out_path

    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_FLAT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for item in items:
            writer.writerow(_flatten_item(item))

    logger.info("Wrote CSV to %s (%d rows)", out_path, len(items))
    return out_path


# ---------------------------------------------------------------------------
# JSON manifest
# ---------------------------------------------------------------------------

def write_manifest(
    manifest_data: dict,
    output_dir: Path,
    filename: str = "manifest.json",
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / filename

    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest_data, fh, indent=2, ensure_ascii=False)

    logger.info("Wrote manifest to %s", out_path)
    return out_path


def build_manifest(
    items: list[BenchmarkItem],
    rejected: list[dict],
    config: dict,
    topic_weights: Optional[dict] = None,
) -> dict:
    """Build a manifest dict from the run."""
    topic_dist = compute_topic_distribution(items)

    tier_counts: dict[str, int] = {}
    for item in items:
        tier_counts[item.language_tier] = tier_counts.get(item.language_tier, 0) + 1

    manifest = {
        "generated_at": _now_iso(),
        "pipeline_version": "0.1.0",
        "total_items": len(items),
        "total_rejected": len(rejected),
        "tier_distribution": tier_counts,
        "topic_distribution": topic_dist,
        "config_snapshot": config,
        "warnings": [],
    }

    # Compare actual vs target topic distribution
    if topic_weights:
        target = topic_weights.get("topic_weights", {})
        actual_total = len(items)
        if actual_total > 0 and target:
            actual_pct = {k: v / actual_total for k, v in topic_dist.items()}
            for topic, target_pct in target.items():
                actual = actual_pct.get(topic, 0.0)
                delta = abs(actual - target_pct)
                if delta > 0.10:
                    manifest["warnings"].append(
                        f"Topic {topic!r}: target={target_pct:.1%}, "
                        f"actual={actual:.1%} (delta={delta:.1%})"
                    )

    return manifest


# ---------------------------------------------------------------------------
# Rejected candidates
# ---------------------------------------------------------------------------

def write_rejected_candidates(
    rejected: list[dict],
    output_dir: Path,
    filename: str = "rejected_candidates.jsonl",
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / filename

    with out_path.open("w", encoding="utf-8") as fh:
        for entry in rejected:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    logger.info("Wrote %d rejected candidates to %s", len(rejected), out_path)
    return out_path
