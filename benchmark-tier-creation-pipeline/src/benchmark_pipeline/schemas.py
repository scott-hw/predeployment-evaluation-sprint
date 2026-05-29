"""
Pydantic v2 models for all pipeline data structures.

Pipeline invariant: SourcePacket is the sole factual authority.
LanguageExamples contribute style only.
"""

from __future__ import annotations

from typing import Any, ClassVar, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Source packet
# ---------------------------------------------------------------------------

class SourcePacket(BaseModel):
    source_packet_id: str
    source_title: str
    source_type: str  # e.g. "faq", "press_release", "policy_doc"
    source_agency: str
    source_url: str
    as_of_date: str  # ISO 8601 date string
    official_question: Optional[str] = None
    official_answer: str
    normalized_resident_need: str
    assumed_facts: list[str] = Field(default_factory=list)
    out_of_scope_facts: list[str] = Field(default_factory=list)
    required_answer_elements: list[str] = Field(min_length=1)
    forbidden_claims: list[str] = Field(min_length=1)
    administrative_topic_tags: list[str] = Field(default_factory=list)
    content_validity_sources: list[str] = Field(default_factory=list)
    notes: Optional[str] = None

    @field_validator("source_packet_id")
    @classmethod
    def id_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("source_packet_id must not be empty")
        return v

    @field_validator("as_of_date")
    @classmethod
    def date_format(cls, v: str) -> str:
        # Accept YYYY-MM-DD or YYYY-MM
        import re
        if not re.match(r"^\d{4}-\d{2}(-\d{2})?$", v):
            raise ValueError(f"as_of_date must be YYYY-MM-DD or YYYY-MM, got: {v!r}")
        return v


# ---------------------------------------------------------------------------
# Topic taxonomy
# ---------------------------------------------------------------------------

class TopicEntry(BaseModel):
    label: str
    description: str
    subtopics: list[str] = Field(default_factory=list)


class TopicTaxonomy(BaseModel):
    administrative_topics: dict[str, TopicEntry]

    def all_topic_keys(self) -> set[str]:
        return set(self.administrative_topics.keys())


# ---------------------------------------------------------------------------
# Language example
# ---------------------------------------------------------------------------

class LanguageExample(BaseModel):
    language_example_id: str
    source_family: str  # e.g. "fabricated", "reddit_r_losangeles"
    language_tier: str  # T1, T2, T3
    text: str
    style_tags: list[str] = Field(default_factory=list)
    notes: Optional[str] = None

    # Canonical tier name aliases — accept both short ("T1") and full ("T1_clean_constituent")
    _TIER_ALIASES: ClassVar[dict[str, str]] = {
        "T1": "T1_clean_constituent",
        "T2": "T2_realistic_messy",
        "T3": "T3_high_friction",
    }
    _VALID_TIERS: ClassVar[set[str]] = {
        "T1_clean_constituent",
        "T2_realistic_messy",
        "T3_high_friction",
        # short forms also accepted
        "T1", "T2", "T3",
    }

    @field_validator("language_tier")
    @classmethod
    def valid_tier(cls, v: str) -> str:
        if v not in cls._VALID_TIERS:
            raise ValueError(
                f"language_tier must be one of {sorted(cls._VALID_TIERS)}, got {v!r}"
            )
        # Normalize to full name
        return cls._TIER_ALIASES.get(v, v)


# ---------------------------------------------------------------------------
# Benchmark item
# ---------------------------------------------------------------------------

class BenchmarkItem(BaseModel):
    benchmark_item_id: str
    source_packet_id: str
    language_tier: str  # T0, T1, T2, T3
    question_text: str
    official_question: Optional[str] = None
    normalized_resident_need: str
    assumed_facts: list[str] = Field(default_factory=list)
    source_answer: dict[str, Any] = Field(default_factory=dict)
    tags: dict[str, Any] = Field(default_factory=dict)
    generation_metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("language_tier")
    @classmethod
    def valid_tier(cls, v: str) -> str:
        allowed = {"T0", "T1", "T2", "T3"}
        if v not in allowed:
            raise ValueError(f"language_tier must be one of {allowed}, got {v!r}")
        return v

    @field_validator("benchmark_item_id")
    @classmethod
    def id_format(cls, v: str) -> str:
        # Expected: {source_packet_id}__{language_tier}__{NNNN}
        parts = v.split("__")
        if len(parts) != 3:
            raise ValueError(
                f"benchmark_item_id must follow pattern "
                f"<source_packet_id>__<language_tier>__<NNNN>, got {v!r}"
            )
        return v


# ---------------------------------------------------------------------------
# LLM generation output (returned by generator LLM)
# ---------------------------------------------------------------------------

class GenerationOutput(BaseModel):
    question_text: str
    style_tags: list[str] = Field(default_factory=list)
    language_example_ids_used: list[str] = Field(default_factory=list)
    notes: Optional[str] = None


# ---------------------------------------------------------------------------
# LLM validation output (returned by validator LLM)
# ---------------------------------------------------------------------------

VALID_VALIDATION_STATUSES = {
    "valid",
    "valid_with_minor_concerns",
    "invalid",
}

VALID_CONFIDENCE_LEVELS = {"high", "medium", "low"}


class ValidationOutput(BaseModel):
    validation_status: str
    presuppositions_detected: list[str] = Field(default_factory=list)
    presuppositions_supported_by_packet: list[str] = Field(default_factory=list)
    unsupported_presuppositions: list[str] = Field(default_factory=list)
    introduces_new_facts: bool
    answerable_from_source_packet: bool
    requires_external_facts: bool
    intent_clear_enough_to_score: bool
    language_tier_match: bool
    copied_from_language_examples: bool
    confidence: str
    notes: Optional[str] = None

    @field_validator("validation_status")
    @classmethod
    def valid_status(cls, v: str) -> str:
        if v not in VALID_VALIDATION_STATUSES:
            raise ValueError(
                f"validation_status must be one of {VALID_VALIDATION_STATUSES}, got {v!r}"
            )
        return v

    @field_validator("confidence")
    @classmethod
    def valid_confidence(cls, v: str) -> str:
        if v not in VALID_CONFIDENCE_LEVELS:
            raise ValueError(
                f"confidence must be one of {VALID_CONFIDENCE_LEVELS}, got {v!r}"
            )
        return v

    def is_acceptable(self, exclude_minor_concerns: bool = False) -> bool:
        """Return True if item should be included in the benchmark."""
        if self.validation_status == "valid":
            return True
        if self.validation_status == "valid_with_minor_concerns":
            return not exclude_minor_concerns
        return False


# ---------------------------------------------------------------------------
# Validation result (internal — wraps ValidationOutput + escalation flag)
# ---------------------------------------------------------------------------

class ValidationResult(BaseModel):
    validation_output: ValidationOutput
    escalated: bool = False
    validator_model: str = ""
    raw_response: str = ""
