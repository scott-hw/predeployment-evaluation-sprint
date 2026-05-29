"""
I/O helpers for loading all pipeline inputs from disk.

Fail loudly on:
  - Missing required fields
  - Duplicate IDs
  - Topic tags not in taxonomy

Warn but continue if official_question is missing.
"""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Optional

import yaml

from .schemas import LanguageExample, SourcePacket, TopicTaxonomy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Source packets
# ---------------------------------------------------------------------------

def load_source_packets(
    path: Path,
    taxonomy: Optional[TopicTaxonomy] = None,
) -> list[SourcePacket]:
    """
    Load source packets from a JSONL file.

    Validates:
    - All required fields present (Pydantic)
    - Unique source_packet_id values
    - administrative_topic_tags exist in taxonomy (if provided)

    Warns if official_question is missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Source packets file not found: {path}")

    packets: list[SourcePacket] = []
    seen_ids: set[str] = set()
    valid_tags: Optional[set[str]] = (
        taxonomy.all_topic_keys() if taxonomy else None
    )

    with path.open("r", encoding="utf-8") as fh:
        for lineno, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"JSON parse error in {path} at line {lineno}: {exc}"
                ) from exc

            try:
                packet = SourcePacket(**data)
            except Exception as exc:
                raise ValueError(
                    f"SourcePacket validation error at line {lineno} "
                    f"(id={data.get('source_packet_id', '?')}): {exc}"
                ) from exc

            # Duplicate ID check
            if packet.source_packet_id in seen_ids:
                raise ValueError(
                    f"Duplicate source_packet_id {packet.source_packet_id!r} "
                    f"at line {lineno} in {path}"
                )
            seen_ids.add(packet.source_packet_id)

            # Missing official_question warning
            if not packet.official_question:
                logger.warning(
                    "source_packet_id=%s has no official_question — "
                    "T0 tier will be skipped for this packet.",
                    packet.source_packet_id,
                )

            # Tag validation
            if valid_tags is not None:
                unknown = set(packet.administrative_topic_tags) - valid_tags
                if unknown:
                    raise ValueError(
                        f"source_packet_id={packet.source_packet_id!r} has "
                        f"unknown topic tags: {sorted(unknown)}. "
                        f"Valid tags: {sorted(valid_tags)}"
                    )

            packets.append(packet)

    logger.info("Loaded %d source packets from %s", len(packets), path)
    return packets


# ---------------------------------------------------------------------------
# Topic taxonomy
# ---------------------------------------------------------------------------

def load_topic_taxonomy(path: Path) -> TopicTaxonomy:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Topic taxonomy file not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    if not raw or "administrative_topics" not in raw:
        raise ValueError(
            f"Topic taxonomy at {path} must have an 'administrative_topics' key."
        )

    try:
        taxonomy = TopicTaxonomy(**raw)
    except Exception as exc:
        raise ValueError(f"TopicTaxonomy validation error: {exc}") from exc

    logger.info("Loaded taxonomy with %d topics from %s", len(taxonomy.administrative_topics), path)
    return taxonomy


# ---------------------------------------------------------------------------
# Language examples
# ---------------------------------------------------------------------------

def load_language_examples(path: Path) -> list[LanguageExample]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Language examples file not found: {path}")

    examples: list[LanguageExample] = []
    seen_ids: set[str] = set()

    with path.open("r", encoding="utf-8") as fh:
        for lineno, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"JSON parse error in {path} at line {lineno}: {exc}"
                ) from exc

            try:
                ex = LanguageExample(**data)
            except Exception as exc:
                raise ValueError(
                    f"LanguageExample validation error at line {lineno}: {exc}"
                ) from exc

            if ex.language_example_id in seen_ids:
                raise ValueError(
                    f"Duplicate language_example_id {ex.language_example_id!r} "
                    f"at line {lineno}"
                )
            seen_ids.add(ex.language_example_id)
            examples.append(ex)

    logger.info("Loaded %d language examples from %s", len(examples), path)
    return examples


# ---------------------------------------------------------------------------
# YAML config loaders
# ---------------------------------------------------------------------------

def load_topic_weights(path: Path) -> dict:
    path = Path(path)
    if not path.exists():
        logger.warning("Topic weights file not found at %s — skipping", path)
        return {}
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    logger.debug("Loaded topic weights from %s", path)
    return data


def load_language_tiers(path: Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Language tiers file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    logger.debug("Loaded language tiers from %s", path)
    return data


def load_generation_config(path: Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Generation config not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    logger.debug("Loaded generation config from %s", path)
    return data


def load_llm_pricing(path: Path) -> dict:
    path = Path(path)
    if not path.exists():
        logger.warning("LLM pricing file not found at %s — using fallback pricing", path)
        return {}
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return data
