"""
Tests for generation pipeline contracts.

Covers:
- T0 copies official_question verbatim (no LLM)
- T0 is skipped when official_question is None
- Benchmark item ID format
- Generation output parses as valid JSON matching GenerationOutput schema
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from benchmark_pipeline.schemas import (
    BenchmarkItem,
    GenerationOutput,
    SourcePacket,
)
from benchmark_pipeline.utils import format_benchmark_item_id
from benchmark_pipeline.generator import GenerationPipeline


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_packet(**overrides) -> SourcePacket:
    base = {
        "source_packet_id": "TEST_001",
        "source_title": "Test FAQ",
        "source_type": "faq",
        "source_agency": "TEST",
        "source_url": "https://example.gov/test",
        "as_of_date": "2025-01",
        "official_question": "Can I get help?",
        "official_answer": "Yes, you can get help by registering.",
        "normalized_resident_need": "Resident wants to know how to get help.",
        "required_answer_elements": ["Register to get help."],
        "forbidden_claims": ["No help is available."],
        "administrative_topic_tags": ["FEMA"],
        "content_validity_sources": [],
        "notes": None,
    }
    base.update(overrides)
    return SourcePacket(**base)


def make_pipeline(active_tiers=None, dry_run=False):
    """Build a GenerationPipeline with mocked LLM client and validator."""
    mock_client = MagicMock()
    mock_client.model = "claude-sonnet-4-6"
    mock_client.generate_json = AsyncMock(return_value={
        "question_text": "How do I register for help?",
        "style_tags": ["T1"],
        "language_example_ids_used": [],
        "notes": None,
    })

    mock_validator = MagicMock()
    mock_validator.validate = AsyncMock(return_value=MagicMock(
        validation_output=MagicMock(
            validation_status="valid",
            is_acceptable=lambda **kw: True,
        ),
        escalated=False,
        validator_model="claude-haiku-4-5-20251001",
    ))
    mock_validator.is_acceptable = MagicMock(return_value=True)

    config = {
        "max_regeneration_attempts": 1,
        "language_examples_per_tier": 2,
        "generator_max_tokens": 512,
        "validator_max_tokens": 768,
    }

    return GenerationPipeline(
        generator_client=mock_client,
        validator=mock_validator,
        language_examples=[],
        config=config,
        seed=42,
        items_per_tier_per_packet=1,
        active_tiers=active_tiers or ["T0", "T1", "T2", "T3"],
        dry_run=dry_run,
    )


# ---------------------------------------------------------------------------
# Benchmark item ID format
# ---------------------------------------------------------------------------

class TestBenchmarkItemIDFormat:
    def test_format_basic(self):
        item_id = format_benchmark_item_id("FEMA_RENTER_001", "T1", 1)
        assert item_id == "FEMA_RENTER_001__T1__0001"

    def test_format_zero_padding(self):
        item_id = format_benchmark_item_id("PKT_001", "T3", 42)
        assert item_id == "PKT_001__T3__0042"

    def test_format_four_digit_index(self):
        item_id = format_benchmark_item_id("PKT_001", "T0", 1000)
        assert item_id == "PKT_001__T0__1000"

    def test_benchmark_item_id_validation_accepts_valid(self):
        item = BenchmarkItem(
            benchmark_item_id="PKT_001__T1__0001",
            source_packet_id="PKT_001",
            language_tier="T1",
            question_text="Test question?",
            normalized_resident_need="Test need.",
            source_answer={
                "official_answer": "Answer.",
                "required_answer_elements": ["Element."],
                "forbidden_claims": ["Forbidden."],
            },
            tags={},
            generation_metadata={},
        )
        assert item.benchmark_item_id == "PKT_001__T1__0001"

    def test_benchmark_item_id_validation_rejects_malformed(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            BenchmarkItem(
                benchmark_item_id="BAD_ID",
                source_packet_id="PKT_001",
                language_tier="T1",
                question_text="Test?",
                normalized_resident_need="Test.",
                source_answer={
                    "official_answer": "A.",
                    "required_answer_elements": ["E."],
                    "forbidden_claims": ["F."],
                },
                tags={},
                generation_metadata={},
            )


# ---------------------------------------------------------------------------
# T0 verbatim copy
# ---------------------------------------------------------------------------

class TestT0VerbatimCopy:
    @pytest.mark.asyncio
    async def test_t0_copies_official_question_verbatim(self):
        pipeline = make_pipeline(active_tiers=["T0"])
        packet = make_packet(official_question="Can renters apply for FEMA?")
        items, rejected = await pipeline.run([packet])

        assert len(items) == 1
        assert items[0].language_tier == "T0"
        assert items[0].question_text == "Can renters apply for FEMA?"
        assert items[0].question_text == packet.official_question

    @pytest.mark.asyncio
    async def test_t0_skipped_when_no_official_question(self):
        pipeline = make_pipeline(active_tiers=["T0"])
        packet = make_packet(official_question=None)
        items, rejected = await pipeline.run([packet])

        assert len(items) == 0
        assert len(rejected) == 1
        assert "T0 skipped" in rejected[0]["reason"]

    @pytest.mark.asyncio
    async def test_t0_does_not_call_llm(self):
        pipeline = make_pipeline(active_tiers=["T0"])
        packet = make_packet()
        await pipeline.run([packet])
        # LLM client should never have been called for T0
        pipeline._gen.generate_json.assert_not_called()

    @pytest.mark.asyncio
    async def test_t0_metadata_marks_no_llm(self):
        pipeline = make_pipeline(active_tiers=["T0"])
        packet = make_packet()
        items, _ = await pipeline.run([packet])
        assert "verbatim" in items[0].generation_metadata["generator_model"]


# ---------------------------------------------------------------------------
# T1/T2/T3 LLM generation
# ---------------------------------------------------------------------------

class TestLLMGeneration:
    @pytest.mark.asyncio
    async def test_llm_tier_produces_item(self):
        pipeline = make_pipeline(active_tiers=["T1"])
        packet = make_packet()
        items, rejected = await pipeline.run([packet])
        assert len(items) == 1
        assert items[0].language_tier == "T1"

    @pytest.mark.asyncio
    async def test_llm_generates_four_items_all_tiers(self):
        pipeline = make_pipeline(active_tiers=["T0", "T1", "T2", "T3"])
        packet = make_packet()
        items, rejected = await pipeline.run([packet])
        tiers = {item.language_tier for item in items}
        assert "T0" in tiers
        assert "T1" in tiers
        assert "T2" in tiers
        assert "T3" in tiers

    @pytest.mark.asyncio
    async def test_rejected_when_validator_rejects(self):
        pipeline = make_pipeline(active_tiers=["T1"])
        pipeline._validator.is_acceptable = MagicMock(return_value=False)
        pipeline._validator.validate = AsyncMock(return_value=MagicMock(
            validation_output=MagicMock(
                validation_status="invalid",
                model_dump=lambda: {"validation_status": "invalid"},
            ),
            escalated=False,
            validator_model="claude-haiku-4-5-20251001",
        ))
        packet = make_packet()
        items, rejected = await pipeline.run([packet])
        assert len(items) == 0
        assert len(rejected) > 0


# ---------------------------------------------------------------------------
# GenerationOutput JSON parsing
# ---------------------------------------------------------------------------

class TestGenerationOutputParsing:
    def test_parses_valid_json(self):
        raw = {
            "question_text": "How do I apply for FEMA assistance as a renter?",
            "style_tags": ["T1", "complete_sentence"],
            "language_example_ids_used": ["EX_T1_001"],
            "notes": None,
        }
        go = GenerationOutput(**raw)
        assert go.question_text == "How do I apply for FEMA assistance as a renter?"

    def test_parses_minimal_json(self):
        raw = {"question_text": "Help with rent?"}
        go = GenerationOutput(**raw)
        assert go.style_tags == []

    def test_strip_markdown_fences(self):
        from benchmark_pipeline.utils import strip_markdown_fences
        fenced = '```json\n{"question_text": "Test?"}\n```'
        cleaned = strip_markdown_fences(fenced)
        parsed = json.loads(cleaned)
        assert parsed["question_text"] == "Test?"
