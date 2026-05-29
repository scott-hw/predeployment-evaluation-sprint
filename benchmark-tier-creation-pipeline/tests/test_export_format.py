"""
Tests for exporter: JSONL, CSV, and manifest writing and reading back.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from benchmark_pipeline.schemas import BenchmarkItem
from benchmark_pipeline.exporter import (
    write_benchmark_jsonl,
    write_benchmark_csv,
    write_manifest,
    write_rejected_candidates,
    build_manifest,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_item(
    item_id: str = "PKT_001__T1__0001",
    tier: str = "T1",
    question: str = "Can I get FEMA help as a renter?",
    packet_id: str = "PKT_001",
) -> BenchmarkItem:
    return BenchmarkItem(
        benchmark_item_id=item_id,
        source_packet_id=packet_id,
        language_tier=tier,
        question_text=question,
        official_question="Can renters apply?",
        normalized_resident_need="Resident wants to know about renter eligibility.",
        source_answer={
            "official_answer": "Yes, renters can apply.",
            "required_answer_elements": ["Renters are eligible."],
            "forbidden_claims": ["Renters are not eligible."],
        },
        tags={
            "administrative_topic_tags": ["FEMA", "renter_assistance"],
            "source_agency": "FEMA",
            "source_type": "faq",
            "source_url": "https://example.gov/fema",
            "as_of_date": "2025-01",
        },
        generation_metadata={
            "generator_model": "claude-sonnet-4-6",
            "validator_model": "claude-haiku-4-5-20251001",
            "validation_status": "valid",
            "escalated": False,
            "style_tags": ["T1"],
            "language_example_ids_used": ["EX_T1_001"],
            "seed": 42,
        },
    )


def make_items() -> list[BenchmarkItem]:
    return [
        make_item("PKT_001__T0__0001", "T0", "Can renters apply?"),
        make_item("PKT_001__T1__0001", "T1", "Can I get FEMA help as a renter?"),
        make_item("PKT_001__T2__0001", "T2", "so can renters get fema help too?"),
        make_item("PKT_001__T3__0001", "T3", "fema renter help??"),
    ]


# ---------------------------------------------------------------------------
# JSONL
# ---------------------------------------------------------------------------

class TestWriteBenchmarkJSONL:
    def test_writes_correct_number_of_lines(self, tmp_path):
        items = make_items()
        write_benchmark_jsonl(items, tmp_path)
        out = tmp_path / "benchmark.jsonl"
        lines = [l for l in out.read_text().splitlines() if l.strip()]
        assert len(lines) == len(items)

    def test_each_line_is_valid_json(self, tmp_path):
        items = make_items()
        write_benchmark_jsonl(items, tmp_path)
        out = tmp_path / "benchmark.jsonl"
        for line in out.read_text().splitlines():
            if line.strip():
                parsed = json.loads(line)
                assert "benchmark_item_id" in parsed

    def test_reads_back_as_benchmark_items(self, tmp_path):
        items = make_items()
        write_benchmark_jsonl(items, tmp_path)
        out = tmp_path / "benchmark.jsonl"
        loaded = []
        for line in out.read_text().splitlines():
            if line.strip():
                loaded.append(BenchmarkItem(**json.loads(line)))
        assert len(loaded) == len(items)
        assert loaded[0].benchmark_item_id == items[0].benchmark_item_id

    def test_empty_list_writes_empty_file(self, tmp_path):
        write_benchmark_jsonl([], tmp_path)
        out = tmp_path / "benchmark.jsonl"
        assert out.exists()
        assert out.read_text().strip() == ""

    def test_custom_filename(self, tmp_path):
        items = make_items()
        write_benchmark_jsonl(items, tmp_path, filename="my_benchmark.jsonl")
        assert (tmp_path / "my_benchmark.jsonl").exists()


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

class TestWriteBenchmarkCSV:
    def test_writes_csv_with_header(self, tmp_path):
        items = make_items()
        write_benchmark_csv(items, tmp_path)
        out = tmp_path / "benchmark.csv"
        with out.open() as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        assert len(rows) == len(items)

    def test_csv_has_required_columns(self, tmp_path):
        items = [make_item()]
        write_benchmark_csv(items, tmp_path)
        out = tmp_path / "benchmark.csv"
        with out.open() as fh:
            reader = csv.DictReader(fh)
            fieldnames = reader.fieldnames or []
        assert "benchmark_item_id" in fieldnames
        assert "language_tier" in fieldnames
        assert "question_text" in fieldnames
        assert "source_answer__official_answer" in fieldnames

    def test_csv_benchmark_item_id_matches(self, tmp_path):
        item = make_item()
        write_benchmark_csv([item], tmp_path)
        out = tmp_path / "benchmark.csv"
        with out.open() as fh:
            rows = list(csv.DictReader(fh))
        assert rows[0]["benchmark_item_id"] == item.benchmark_item_id

    def test_csv_nested_list_semicolon_joined(self, tmp_path):
        item = make_item()
        write_benchmark_csv([item], tmp_path)
        out = tmp_path / "benchmark.csv"
        with out.open() as fh:
            rows = list(csv.DictReader(fh))
        # tags__administrative_topic_tags should be "FEMA; renter_assistance"
        tags = rows[0]["tags__administrative_topic_tags"]
        assert "FEMA" in tags

    def test_empty_items_writes_header_only(self, tmp_path):
        write_benchmark_csv([], tmp_path)
        out = tmp_path / "benchmark.csv"
        with out.open() as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 0


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

class TestWriteManifest:
    def test_writes_valid_json(self, tmp_path):
        data = {"total_items": 4, "warnings": []}
        write_manifest(data, tmp_path)
        out = tmp_path / "manifest.json"
        loaded = json.loads(out.read_text())
        assert loaded["total_items"] == 4

    def test_manifest_roundtrip(self, tmp_path):
        items = make_items()
        manifest = build_manifest(items=items, rejected=[], config={"seed": 42})
        write_manifest(manifest, tmp_path)
        out = tmp_path / "manifest.json"
        loaded = json.loads(out.read_text())
        assert loaded["total_items"] == len(items)
        assert loaded["total_rejected"] == 0

    def test_manifest_tier_distribution(self, tmp_path):
        items = make_items()
        manifest = build_manifest(items=items, rejected=[], config={})
        assert manifest["tier_distribution"]["T0"] == 1
        assert manifest["tier_distribution"]["T1"] == 1
        assert manifest["tier_distribution"]["T2"] == 1
        assert manifest["tier_distribution"]["T3"] == 1

    def test_manifest_has_generated_at(self, tmp_path):
        manifest = build_manifest(items=[], rejected=[], config={})
        assert "generated_at" in manifest

    def test_manifest_topic_distribution(self, tmp_path):
        items = make_items()
        manifest = build_manifest(items=items, rejected=[], config={})
        dist = manifest["topic_distribution"]
        assert dist.get("FEMA", 0) == len(items)


# ---------------------------------------------------------------------------
# Rejected candidates
# ---------------------------------------------------------------------------

class TestWriteRejectedCandidates:
    def test_writes_jsonl(self, tmp_path):
        rejected = [
            {"benchmark_item_id": "PKT_001__T1__0001", "reason": "validation=invalid"},
            {"benchmark_item_id": "PKT_001__T2__0001", "reason": "generation exception"},
        ]
        write_rejected_candidates(rejected, tmp_path)
        out = tmp_path / "rejected_candidates.jsonl"
        lines = [l for l in out.read_text().splitlines() if l.strip()]
        assert len(lines) == 2

    def test_empty_rejected(self, tmp_path):
        write_rejected_candidates([], tmp_path)
        out = tmp_path / "rejected_candidates.jsonl"
        assert out.exists()
        assert out.read_text().strip() == ""
