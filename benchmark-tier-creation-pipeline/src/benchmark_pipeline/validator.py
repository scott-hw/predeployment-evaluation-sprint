"""
Question variant validator.

Primary validator: claude-haiku-4-5-20251001 at temperature 0.0
Escalation: if confidence == "low", re-validate with claude-sonnet-4-6 at temperature 0.0

valid / valid_with_minor_concerns  -> include in benchmark
invalid                            -> retry generation (up to max_regeneration_attempts)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from .schemas import ValidationOutput, ValidationResult
from .llm_client import LLMClient, VALIDATOR_MODEL, GENERATOR_MODEL

logger = logging.getLogger(__name__)


def _find_prompts_dir() -> Path:
    import os
    root_env = os.environ.get("BENCHMARK_PIPELINE_ROOT")
    if root_env:
        p = Path(root_env) / "prompts"
        if p.exists():
            return p
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "prompts"
        if candidate.exists() and (candidate / "validate_question_variants.md").exists():
            return candidate
    return Path.cwd() / "prompts"


_PROMPTS_DIR = _find_prompts_dir()


def _load_prompt_template() -> str:
    p = _PROMPTS_DIR / "validate_question_variants.md"
    if p.exists():
        return p.read_text(encoding="utf-8")
    raise FileNotFoundError(f"Validation prompt not found at {p}")


class QuestionValidator:
    """
    Validates a generated question variant against its source packet.

    Parameters
    ----------
    primary_client : LLMClient
        Should use VALIDATOR_MODEL (Haiku) at temperature 0.0.
    secondary_client : LLMClient, optional
        Used for escalation; should use GENERATOR_MODEL (Sonnet) at temperature 0.0.
    exclude_minor_concerns : bool
        If True, valid_with_minor_concerns items are treated as invalid.
    """

    def __init__(
        self,
        primary_client: LLMClient,
        secondary_client: Optional[LLMClient] = None,
        exclude_minor_concerns: bool = False,
    ) -> None:
        self._primary = primary_client
        self._secondary = secondary_client
        self._exclude_minor = exclude_minor_concerns
        self._prompt_template = _load_prompt_template()

    async def validate(
        self,
        question_text: str,
        source_packet_json: str,
        language_tier: str,
        language_examples_json: str = "[]",
    ) -> ValidationResult:
        """
        Validate a generated question variant.

        Returns a ValidationResult with all fields populated.
        """
        prompt = self._render_prompt(
            question_text=question_text,
            source_packet_json=source_packet_json,
            language_tier=language_tier,
            language_examples_json=language_examples_json,
        )

        # Primary validation (Haiku)
        raw = await self._primary.generate_json(
            prompt=prompt,
            max_tokens=768,
        )
        try:
            vo = ValidationOutput(**raw)
        except Exception as exc:
            logger.error("ValidationOutput parse error: %s | raw=%s", exc, str(raw)[:300])
            # Return a fallback invalid result
            return ValidationResult(
                validation_output=ValidationOutput(
                    validation_status="invalid",
                    presuppositions_detected=[],
                    presuppositions_supported_by_packet=[],
                    unsupported_presuppositions=[],
                    introduces_new_facts=True,
                    answerable_from_source_packet=False,
                    requires_external_facts=True,
                    intent_clear_enough_to_score=False,
                    language_tier_match=False,
                    copied_from_language_examples=False,
                    confidence="low",
                    notes=f"Parse error: {exc}",
                ),
                escalated=False,
                validator_model=self._primary.model,
                raw_response=str(raw),
            )

        # Escalate if confidence is low
        escalated = False
        if vo.confidence == "low" and self._secondary is not None:
            logger.info("Escalating validation to %s (confidence=low)", GENERATOR_MODEL)
            secondary_raw = await self._secondary.generate_json(
                prompt=prompt,
                max_tokens=768,
            )
            try:
                vo = ValidationOutput(**secondary_raw)
                escalated = True
            except Exception as exc:
                logger.warning("Secondary validator parse error: %s", exc)

        return ValidationResult(
            validation_output=vo,
            escalated=escalated,
            validator_model=(
                self._secondary.model if escalated else self._primary.model
            ),
            raw_response=json.dumps(raw),
        )

    def is_acceptable(self, result: ValidationResult) -> bool:
        return result.validation_output.is_acceptable(
            exclude_minor_concerns=self._exclude_minor
        )

    def _render_prompt(
        self,
        question_text: str,
        source_packet_json: str,
        language_tier: str,
        language_examples_json: str,
    ) -> str:
        return (
            self._prompt_template
            .replace("{question_text}", question_text)
            .replace("{source_packet_json}", source_packet_json)
            .replace("{language_tier}", language_tier)
            .replace("{language_examples_json}", language_examples_json)
        )
