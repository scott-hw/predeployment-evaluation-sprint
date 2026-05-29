"""
Tests for Pydantic schema validation.

Covers: SourcePacket, LanguageExample, BenchmarkItem validation,
        required-field enforcement, duplicate ID detection,
        and tag validation.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

# Make sure the package is importable from the repo root
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from benchmark_pipeline.schemas import (
    BenchmarkItem,
    LanguageExample,
    SourcePacket,
    TopicTaxonomy,
    ValidationOutput,
    GenerationOutput,
)
from benchmark_pipeline.io import (
    load_source_packets,
    load_topic_taxonomy,
    load_language_examples,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

VALID_PACKET = {
    "source_packet_id": "TEST_001",
    "source_title": "Test FAQ",
    "source_type": "faq",
    "source_agency": "TEST_AGENCY",
    "source_url": "https://example.gov/test",
    "as_of_date": "2025-01-15",
    "official_question": "Can I get help?",
    "official_answer": "Yes, you can get help.",
    "normalized_resident_need": "Resident wants to know about help.",
    "required_answer_elements": ["You can get help."],
    "forbidden_claims": ["No help is available."],
    "administrative_topic_tags": ["FEMA"],
    "content_validity_sources": [],
    "notes": None,
}

VALID_TAXONOMY_YAML = """
administrative_topics:
  FEMA:
    label: "FEMA"
    description: "FEMA assistance programs."
    subtopics: []
  renter_assistance:
    label: "Renter Assistance"
    description: "Renter assistance programs."
    subtopics: []
"""


# ---------------------------------------------------------------------------
# SourcePacket validation
# ---------------------------------------------------------------------------

class TestSourcePacketValidation:
    def test_valid_packet(self):
        p = SourcePacket(**VALID_PACKET)
        assert p.source_packet_id == "TEST_001"
        assert p.source_type == "faq"

    def test_missing_required_field_raises(self):
        bad = {**VALID_PACKET}
        del bad["official_answer"]
        with pytest.raises(ValidationError):
            SourcePacket(**bad)

    def test_missing_required_answer_elements_raises(self):
        bad = {**VALID_PACKET, "required_answer_elements": []}
        with pytest.raises(ValidationError):
            SourcePacket(**bad)

    def test_missing_forbidden_claims_raises(self):
        bad = {**VALID_PACKET, "forbidden_claims": []}
        with pytest.raises(ValidationError):
            SourcePacket(**bad)

    def test_empty_source_packet_id_raises(self):
        bad = {**VALID_PACKET, "source_packet_id": "   "}
        with pytest.raises(ValidationError):
            SourcePacket(**bad)

    def test_invalid_date_format_raises(self):
        bad = {**VALID_PACKET, "as_of_date": "01/15/2025"}
        with pytest.raises(ValidationError):
            SourcePacket(**bad)

    def test_valid_date_yyyy_mm(self):
        p = SourcePacket(**{**VALID_PACKET, "as_of_date": "2025-01"})
        assert p.as_of_date == "2025-01"

    def test_optional_official_question_none(self):
        p = SourcePacket(**{**VALID_PACKET, "official_question": None})
        assert p.official_question is None

    def test_notes_optional(self):
        p = SourcePacket(**{**VALID_PACKET, "notes": "Some note."})
        assert p.notes == "Some note."


# ---------------------------------------------------------------------------
# LanguageExample validation
# ---------------------------------------------------------------------------

class TestLanguageExampleValidation:
    def test_valid_example(self):
        ex = LanguageExample(
            language_example_id="EX_001",
            source_family="fabricated",
            language_tier="T1",
            text="I need help with my rent after the disaster.",
            style_tags=["T1"],
        )
        # Short names are normalized to full canonical names
        assert ex.language_tier == "T1_clean_constituent"

    def test_valid_example_full_name(self):
        ex = LanguageExample(
            language_example_id="EX_002",
            source_family="fabricated",
            language_tier="T1_clean_constituent",
            text="I need help with my rent after the disaster.",
        )
        assert ex.language_tier == "T1_clean_constituent"

    def test_invalid_tier_raises(self):
        with pytest.raises(ValidationError):
            LanguageExample(
                language_example_id="EX_001",
                source_family="fabricated",
                language_tier="T0",  # T0 not allowed for examples
                text="Test.",
            )

    def test_t2_and_t3_valid(self):
        expected = {
            "T2": "T2_realistic_messy",
            "T3": "T3_high_friction",
        }
        for tier, full_name in expected.items():
            ex = LanguageExample(
                language_example_id=f"EX_{tier}_001",
                source_family="fabricated",
                language_tier=tier,
                text="Test text.",
            )
            # Short names are normalized to full canonical names
            assert ex.language_tier == full_name


# ---------------------------------------------------------------------------
# Duplicate ID detection via io.load_source_packets
# ---------------------------------------------------------------------------

class TestDuplicateIDs:
    def test_duplicate_source_packet_id_raises(self, tmp_path):
        p1 = json.dumps(VALID_PACKET)
        p2 = json.dumps({**VALID_PACKET, "administrative_topic_tags": ["FEMA"]})
        # Both have source_packet_id == "TEST_001"
        jsonl_file = tmp_path / "packets.jsonl"
        jsonl_file.write_text(p1 + "\n" + p2 + "\n")

        with pytest.raises(ValueError, match="Duplicate source_packet_id"):
            load_source_packets(jsonl_file)

    def test_unique_ids_accepted(self, tmp_path):
        p1 = json.dumps(VALID_PACKET)
        p2 = json.dumps({**VALID_PACKET, "source_packet_id": "TEST_002"})
        jsonl_file = tmp_path / "packets.jsonl"
        jsonl_file.write_text(p1 + "\n" + p2 + "\n")

        packets = load_source_packets(jsonl_file)
        assert len(packets) == 2


# ---------------------------------------------------------------------------
# Tag validation via taxonomy
# ---------------------------------------------------------------------------

class TestTagValidation:
    def _write_taxonomy(self, tmp_path: Path) -> Path:
        taxonomy_file = tmp_path / "taxonomy.yaml"
        taxonomy_file.write_text(VALID_TAXONOMY_YAML)
        return taxonomy_file

    def test_valid_tags_accepted(self, tmp_path):
        taxonomy_path = self._write_taxonomy(tmp_path)
        taxonomy = load_topic_taxonomy(taxonomy_path)
        packet_file = tmp_path / "packets.jsonl"
        packet_file.write_text(json.dumps(VALID_PACKET) + "\n")
        # VALID_PACKET uses tag "FEMA" which is in taxonomy
        packets = load_source_packets(packet_file, taxonomy=taxonomy)
        assert len(packets) == 1

    def test_unknown_tag_raises(self, tmp_path):
        taxonomy_path = self._write_taxonomy(tmp_path)
        taxonomy = load_topic_taxonomy(taxonomy_path)
        bad_packet = {**VALID_PACKET, "administrative_topic_tags": ["nonexistent_topic"]}
        packet_file = tmp_path / "packets.jsonl"
        packet_file.write_text(json.dumps(bad_packet) + "\n")
        with pytest.raises(ValueError, match="unknown topic tags"):
            load_source_packets(packet_file, taxonomy=taxonomy)


# ---------------------------------------------------------------------------
# ValidationOutput schema
# ---------------------------------------------------------------------------

class TestValidationOutput:
    def test_valid_status_values(self):
        for status in ("valid", "valid_with_minor_concerns", "invalid"):
            vo = ValidationOutput(
                validation_status=status,
                introduces_new_facts=False,
                answerable_from_source_packet=True,
                requires_external_facts=False,
                intent_clear_enough_to_score=True,
                language_tier_match=True,
                copied_from_language_examples=False,
                confidence="high",
            )
            assert vo.validation_status == status

    def test_invalid_status_raises(self):
        with pytest.raises(ValidationError):
            ValidationOutput(
                validation_status="maybe",
                introduces_new_facts=False,
                answerable_from_source_packet=True,
                requires_external_facts=False,
                intent_clear_enough_to_score=True,
                language_tier_match=True,
                copied_from_language_examples=False,
                confidence="high",
            )

    def test_is_acceptable(self):
        def make_vo(status, confidence="high"):
            return ValidationOutput(
                validation_status=status,
                introduces_new_facts=False,
                answerable_from_source_packet=True,
                requires_external_facts=False,
                intent_clear_enough_to_score=True,
                language_tier_match=True,
                copied_from_language_examples=False,
                confidence=confidence,
            )

        assert make_vo("valid").is_acceptable() is True
        assert make_vo("valid_with_minor_concerns").is_acceptable() is True
        assert make_vo("valid_with_minor_concerns").is_acceptable(exclude_minor_concerns=True) is False
        assert make_vo("invalid").is_acceptable() is False


# ---------------------------------------------------------------------------
# GenerationOutput schema
# ---------------------------------------------------------------------------

class TestGenerationOutput:
    def test_valid_output(self):
        go = GenerationOutput(
            question_text="Can I get help as a renter?",
            style_tags=["T1"],
            language_example_ids_used=["EX_T1_001"],
        )
        assert go.question_text == "Can I get help as a renter?"

    def test_empty_lists_default(self):
        go = GenerationOutput(question_text="Test?")
        assert go.style_tags == []
        assert go.language_example_ids_used == []
        assert go.notes is None
