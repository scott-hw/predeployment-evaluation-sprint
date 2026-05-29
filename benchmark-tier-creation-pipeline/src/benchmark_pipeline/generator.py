"""
GenerationPipeline: orchestrates source packet -> benchmark item generation.

For each source_packet x tier:
  T0: copy official_question verbatim (no LLM)
  T1/T2/T3: LLM generation + validation, with retry budget

Benchmark item ID: {source_packet_id}__{language_tier}__{NNNN}
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from .schemas import (
    BenchmarkItem,
    GenerationOutput,
    LanguageExample,
    SourcePacket,
    TopicTaxonomy,
)
from .llm_client import LLMClient, GENERATOR_MODEL
from .validator import QuestionValidator
from .utils import format_benchmark_item_id, sample_examples

logger = logging.getLogger(__name__)


def _find_prompts_dir() -> Path:
    """
    Locate the prompts/ directory regardless of install mode.

    Search order:
    1. BENCHMARK_PIPELINE_ROOT env var
    2. Walk up from __file__ looking for a directory that contains prompts/
    3. Current working directory / prompts
    """
    import os
    root_env = os.environ.get("BENCHMARK_PIPELINE_ROOT")
    if root_env:
        p = Path(root_env) / "prompts"
        if p.exists():
            return p

    # Walk up from this file
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "prompts"
        if candidate.exists() and (candidate / "generate_question_variants.md").exists():
            return candidate

    # Final fallback: cwd/prompts
    return Path.cwd() / "prompts"


_PROMPTS_DIR = _find_prompts_dir()

TIER_ORDER = ["T0", "T1", "T2", "T3"]
LLM_TIERS = ["T1", "T2", "T3"]


def _load_prompt_template() -> str:
    p = _PROMPTS_DIR / "generate_question_variants.md"
    if p.exists():
        return p.read_text(encoding="utf-8")
    raise FileNotFoundError(f"Generation prompt not found at {p}")


def _load_tier_fragment(tier: str) -> str:
    p = _PROMPTS_DIR / "language_tier_fragments" / f"{tier}_*.md"
    # Find by prefix
    import glob
    matches = glob.glob(str(_PROMPTS_DIR / "language_tier_fragments" / f"{tier}_*.md"))
    if matches:
        return Path(matches[0]).read_text(encoding="utf-8")
    # Try exact name
    p2 = _PROMPTS_DIR / "language_tier_fragments" / f"{tier}.md"
    if p2.exists():
        return p2.read_text(encoding="utf-8")
    logger.warning("No tier fragment found for %s — using empty fragment", tier)
    return f"# Language tier {tier}\nGenerate a question in the {tier} style."


class GenerationPipeline:
    """
    Orchestrates full benchmark generation.

    Parameters
    ----------
    generator_client : LLMClient
        Uses GENERATOR_MODEL (Sonnet).
    validator : QuestionValidator
        Uses VALIDATOR_MODEL (Haiku) with optional Sonnet escalation.
    language_examples : list[LanguageExample]
        Pool of style examples to sample from.
    config : dict
        Merged generation config from generation_config.yaml.
    seed : int, optional
        Random seed for reproducible sampling.
    items_per_tier_per_packet : int
        Number of items to generate per tier per source packet.
    active_tiers : list[str], optional
        Which tiers to generate. Defaults to all four.
    dry_run : bool
        If True, skip LLM calls and return placeholder items.
    """

    def __init__(
        self,
        generator_client: LLMClient,
        validator: QuestionValidator,
        language_examples: list[LanguageExample],
        config: dict,
        seed: Optional[int] = None,
        items_per_tier_per_packet: int = 1,
        active_tiers: Optional[list[str]] = None,
        dry_run: bool = False,
    ) -> None:
        self._gen = generator_client
        self._validator = validator
        self._examples = language_examples
        self._config = config
        self._seed = seed
        self._items_per_tier = items_per_tier_per_packet
        self._active_tiers = active_tiers or TIER_ORDER
        self._dry_run = dry_run
        self._max_retries = config.get("max_regeneration_attempts", 2)
        self._examples_per_tier = config.get("language_examples_per_tier", 3)

        self._prompt_template = _load_prompt_template()
        self._tier_fragments: dict[str, str] = {}
        for tier in LLM_TIERS:
            if tier in self._active_tiers:
                self._tier_fragments[tier] = _load_tier_fragment(tier)

    async def run(
        self,
        packets: list[SourcePacket],
    ) -> tuple[list[BenchmarkItem], list[dict]]:
        """
        Run the full generation pipeline.

        Returns
        -------
        items : list[BenchmarkItem]
            Accepted benchmark items.
        rejected : list[dict]
            Rejected candidates with reason.
        """
        all_items: list[BenchmarkItem] = []
        all_rejected: list[dict] = []
        counters: dict[str, int] = {}  # (packet_id, tier) -> next index

        for packet in packets:
            for tier in self._active_tiers:
                for _ in range(self._items_per_tier):
                    key = f"{packet.source_packet_id}__{tier}"
                    idx = counters.get(key, 0) + 1
                    counters[key] = idx

                    item_id = format_benchmark_item_id(
                        packet.source_packet_id, tier, idx
                    )

                    if tier == "T0":
                        item = self._make_t0_item(packet, item_id)
                        if item:
                            all_items.append(item)
                        else:
                            all_rejected.append({
                                "benchmark_item_id": item_id,
                                "reason": "T0 skipped: official_question missing",
                                "source_packet_id": packet.source_packet_id,
                                "language_tier": "T0",
                            })
                    else:
                        item, rejected = await self._generate_with_retry(
                            packet=packet,
                            tier=tier,
                            item_id=item_id,
                            idx=idx,
                        )
                        if item:
                            all_items.append(item)
                        if rejected:
                            all_rejected.extend(rejected)

        logger.info(
            "Generation complete: %d accepted, %d rejected",
            len(all_items),
            len(all_rejected),
        )
        return all_items, all_rejected

    # ------------------------------------------------------------------
    # T0
    # ------------------------------------------------------------------

    def _make_t0_item(
        self,
        packet: SourcePacket,
        item_id: str,
    ) -> Optional[BenchmarkItem]:
        if not packet.official_question:
            logger.warning(
                "Skipping T0 for %s: official_question is None",
                packet.source_packet_id,
            )
            return None

        return BenchmarkItem(
            benchmark_item_id=item_id,
            source_packet_id=packet.source_packet_id,
            language_tier="T0",
            question_text=packet.official_question,
            official_question=packet.official_question,
            normalized_resident_need=packet.normalized_resident_need,
            source_answer={
                "official_answer": packet.official_answer,
                "required_answer_elements": packet.required_answer_elements,
                "forbidden_claims": packet.forbidden_claims,
            },
            tags={
                "administrative_topic_tags": packet.administrative_topic_tags,
                "source_agency": packet.source_agency,
                "source_type": packet.source_type,
                "source_url": packet.source_url,
                "as_of_date": packet.as_of_date,
            },
            generation_metadata={
                "generator_model": "none (verbatim copy)",
                "validator_model": "none (T0 not validated)",
                "validation_status": "T0_verbatim",
                "escalated": False,
                "style_tags": [],
                "language_example_ids_used": [],
                "seed": self._seed,
            },
        )

    # ------------------------------------------------------------------
    # T1/T2/T3 with retry
    # ------------------------------------------------------------------

    async def _generate_with_retry(
        self,
        packet: SourcePacket,
        tier: str,
        item_id: str,
        idx: int,
    ) -> tuple[Optional[BenchmarkItem], list[dict]]:
        rejected: list[dict] = []

        for attempt in range(self._max_retries + 1):
            # Sample language examples (vary seed per attempt to avoid identical retries)
            sample_seed = (
                None if self._seed is None
                else self._seed + idx * 100 + attempt
            )
            examples = sample_examples(
                self._examples,
                tier,
                self._examples_per_tier,
                seed=sample_seed,
            )

            # Build prompt
            prompt = self._render_generation_prompt(packet, tier, examples)
            source_packet_json = json.dumps(packet.model_dump(), ensure_ascii=False)

            # Dry-run mode
            if self._dry_run:
                item = self._make_placeholder_item(packet, tier, item_id, examples)
                return item, rejected

            # LLM call
            try:
                raw = await self._gen.generate_json(
                    prompt=prompt,
                    max_tokens=512,
                    cached_system_prefix=(
                        "You are generating benchmark questions for a "
                        "disaster-services chatbot evaluation. The source "
                        "packet below is the sole factual authority."
                    ),
                )
                gen_output = GenerationOutput(**raw)
            except Exception as exc:
                logger.error(
                    "Generation failed for %s tier=%s attempt=%d: %s",
                    packet.source_packet_id, tier, attempt, exc
                )
                rejected.append({
                    "benchmark_item_id": item_id,
                    "reason": f"Generation exception: {exc}",
                    "source_packet_id": packet.source_packet_id,
                    "language_tier": tier,
                    "attempt": attempt,
                })
                continue

            # Validate
            try:
                val_result = await self._validator.validate(
                    question_text=gen_output.question_text,
                    source_packet_json=source_packet_json,
                    language_tier=tier,
                    language_examples_json=json.dumps(
                        [e.model_dump() for e in examples], ensure_ascii=False
                    ),
                )
            except Exception as exc:
                logger.error(
                    "Validation failed for %s tier=%s attempt=%d: %s",
                    packet.source_packet_id, tier, attempt, exc
                )
                rejected.append({
                    "benchmark_item_id": item_id,
                    "reason": f"Validation exception: {exc}",
                    "source_packet_id": packet.source_packet_id,
                    "language_tier": tier,
                    "attempt": attempt,
                })
                continue

            if self._validator.is_acceptable(val_result):
                item = BenchmarkItem(
                    benchmark_item_id=item_id,
                    source_packet_id=packet.source_packet_id,
                    language_tier=tier,
                    question_text=gen_output.question_text,
                    official_question=packet.official_question,
                    normalized_resident_need=packet.normalized_resident_need,
                    source_answer={
                        "official_answer": packet.official_answer,
                        "required_answer_elements": packet.required_answer_elements,
                        "forbidden_claims": packet.forbidden_claims,
                    },
                    tags={
                        "administrative_topic_tags": packet.administrative_topic_tags,
                        "source_agency": packet.source_agency,
                        "source_type": packet.source_type,
                        "source_url": packet.source_url,
                        "as_of_date": packet.as_of_date,
                    },
                    generation_metadata={
                        "generator_model": GENERATOR_MODEL,
                        "validator_model": val_result.validator_model,
                        "validation_status": val_result.validation_output.validation_status,
                        "escalated": val_result.escalated,
                        "style_tags": gen_output.style_tags,
                        "language_example_ids_used": gen_output.language_example_ids_used,
                        "seed": self._seed,
                        "attempt": attempt,
                        "notes": gen_output.notes,
                    },
                )
                return item, rejected
            else:
                logger.info(
                    "%s tier=%s attempt=%d INVALID (%s)",
                    packet.source_packet_id,
                    tier,
                    attempt,
                    val_result.validation_output.validation_status,
                )
                rejected.append({
                    "benchmark_item_id": item_id,
                    "question_text": gen_output.question_text,
                    "reason": f"validation={val_result.validation_output.validation_status}",
                    "validation_detail": val_result.validation_output.model_dump(),
                    "source_packet_id": packet.source_packet_id,
                    "language_tier": tier,
                    "attempt": attempt,
                })

        # Exhausted retries
        logger.warning(
            "Exhausted %d retries for %s tier=%s — item will be rejected",
            self._max_retries + 1,
            packet.source_packet_id,
            tier,
        )
        return None, rejected

    def _render_generation_prompt(
        self,
        packet: SourcePacket,
        tier: str,
        examples: list[LanguageExample],
    ) -> str:
        source_packet_json = json.dumps(packet.model_dump(), indent=2, ensure_ascii=False)
        tier_fragment = self._tier_fragments.get(tier, "")
        examples_json = json.dumps(
            [e.model_dump() for e in examples],
            indent=2,
            ensure_ascii=False,
        )
        # Map tier to human-readable name
        tier_names = {
            "T1": "T1_clean_constituent",
            "T2": "T2_realistic_messy",
            "T3": "T3_high_friction",
        }
        tier_name = tier_names.get(tier, tier)

        return (
            self._prompt_template
            .replace("{source_packet_json}", source_packet_json)
            .replace("{tier_name}", tier_name)
            .replace("{tier_prompt_fragment}", tier_fragment)
            .replace("{language_examples_json}", examples_json)
        )

    def _make_placeholder_item(
        self,
        packet: SourcePacket,
        tier: str,
        item_id: str,
        examples: list[LanguageExample],
    ) -> BenchmarkItem:
        return BenchmarkItem(
            benchmark_item_id=item_id,
            source_packet_id=packet.source_packet_id,
            language_tier=tier,
            question_text=f"[DRY-RUN PLACEHOLDER] {packet.normalized_resident_need}",
            official_question=packet.official_question,
            normalized_resident_need=packet.normalized_resident_need,
            source_answer={
                "official_answer": packet.official_answer,
                "required_answer_elements": packet.required_answer_elements,
                "forbidden_claims": packet.forbidden_claims,
            },
            tags={
                "administrative_topic_tags": packet.administrative_topic_tags,
                "source_agency": packet.source_agency,
                "source_type": packet.source_type,
                "source_url": packet.source_url,
                "as_of_date": packet.as_of_date,
            },
            generation_metadata={
                "generator_model": GENERATOR_MODEL,
                "validator_model": "dry_run",
                "validation_status": "dry_run",
                "escalated": False,
                "style_tags": [],
                "language_example_ids_used": [e.language_example_id for e in examples],
                "seed": self._seed,
            },
        )
