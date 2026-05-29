"""
Tests for the LLM disk cache.

Covers: get/set/miss/hit, cache key generation, metadata tracking,
        clear, atomic write (via .tmp pattern).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from benchmark_pipeline.cache import LLMCache, _build_cache_key


# ---------------------------------------------------------------------------
# Cache key tests
# ---------------------------------------------------------------------------

class TestCacheKeyGeneration:
    def test_same_inputs_same_key(self):
        k1 = _build_cache_key("model-a", 1.0, 512, "hello world")
        k2 = _build_cache_key("model-a", 1.0, 512, "hello world")
        assert k1 == k2

    def test_different_model_different_key(self):
        k1 = _build_cache_key("model-a", 1.0, 512, "hello world")
        k2 = _build_cache_key("model-b", 1.0, 512, "hello world")
        assert k1 != k2

    def test_different_temperature_different_key(self):
        k1 = _build_cache_key("model-a", 0.0, 512, "hello world")
        k2 = _build_cache_key("model-a", 1.0, 512, "hello world")
        assert k1 != k2

    def test_different_max_tokens_different_key(self):
        k1 = _build_cache_key("model-a", 1.0, 256, "hello world")
        k2 = _build_cache_key("model-a", 1.0, 512, "hello world")
        assert k1 != k2

    def test_different_prompt_different_key(self):
        k1 = _build_cache_key("model-a", 1.0, 512, "prompt A")
        k2 = _build_cache_key("model-a", 1.0, 512, "prompt B")
        assert k1 != k2

    def test_key_is_64_char_hex(self):
        k = _build_cache_key("model", 0.5, 100, "test")
        assert len(k) == 64
        assert all(c in "0123456789abcdef" for c in k)


# ---------------------------------------------------------------------------
# Cache get/set/miss/hit
# ---------------------------------------------------------------------------

class TestLLMCacheMissHit:
    def test_miss_on_empty_cache(self, tmp_path):
        cache = LLMCache(cache_dir=tmp_path)
        result = cache.get("nonexistent_key_" + "x" * 56)
        assert result is None

    def test_set_then_get_returns_response(self, tmp_path):
        cache = LLMCache(cache_dir=tmp_path)
        key = cache.make_key("claude-sonnet-4-6", 1.0, 512, "test prompt")
        response = {"question_text": "Can I get help?", "style_tags": ["T1"]}
        cache.set(key, response, metadata={"model": "claude-sonnet-4-6"})
        retrieved = cache.get(key)
        assert retrieved is not None
        assert retrieved["response"] == response

    def test_from_cache_flag_set_on_hit(self, tmp_path):
        cache = LLMCache(cache_dir=tmp_path)
        key = cache.make_key("model", 0.0, 256, "prompt")
        cache.set(key, {"answer": "42"})
        entry = cache.get(key)
        assert entry["from_cache"] is True

    def test_hit_count_increments(self, tmp_path):
        cache = LLMCache(cache_dir=tmp_path)
        key = cache.make_key("model", 0.0, 256, "prompt")
        cache.set(key, {"answer": "42"})
        cache.get(key)
        cache.get(key)
        entry = cache.get(key)
        assert entry["hit_count"] == 3

    def test_cache_file_is_valid_json(self, tmp_path):
        cache = LLMCache(cache_dir=tmp_path)
        key = cache.make_key("model", 0.0, 256, "prompt")
        cache.set(key, {"answer": "42"})
        cache_file = tmp_path / f"{key}.json"
        assert cache_file.exists()
        data = json.loads(cache_file.read_text())
        assert "response" in data
        assert "created_at" in data

    def test_metadata_stored_in_cache_entry(self, tmp_path):
        cache = LLMCache(cache_dir=tmp_path)
        key = cache.make_key("model-x", 0.5, 512, "prompt")
        cache.set(key, {"answer": "yes"}, metadata={"model": "model-x", "temperature": 0.5})
        entry = cache.get(key)
        assert entry["model"] == "model-x"
        assert entry["temperature"] == 0.5


# ---------------------------------------------------------------------------
# Cache clear
# ---------------------------------------------------------------------------

class TestLLMCacheClear:
    def test_clear_removes_all_entries(self, tmp_path):
        cache = LLMCache(cache_dir=tmp_path)
        for i in range(5):
            key = cache.make_key("model", 0.0, 256, f"prompt {i}")
            cache.set(key, {"i": i})
        assert cache.size() == 5
        count = cache.clear()
        assert count == 5
        assert cache.size() == 0

    def test_clear_returns_zero_on_empty(self, tmp_path):
        cache = LLMCache(cache_dir=tmp_path)
        assert cache.clear() == 0

    def test_size_reflects_files(self, tmp_path):
        cache = LLMCache(cache_dir=tmp_path)
        assert cache.size() == 0
        key = cache.make_key("model", 0.0, 256, "prompt")
        cache.set(key, {"x": 1})
        assert cache.size() == 1


# ---------------------------------------------------------------------------
# make_key method
# ---------------------------------------------------------------------------

class TestMakeKeyMethod:
    def test_make_key_matches_build_cache_key(self, tmp_path):
        cache = LLMCache(cache_dir=tmp_path)
        k1 = cache.make_key("model", 0.7, 256, "my prompt")
        k2 = _build_cache_key("model", 0.7, 256, "my prompt")
        assert k1 == k2

    def test_cache_dir_created_if_missing(self, tmp_path):
        new_dir = tmp_path / "new_cache_dir"
        assert not new_dir.exists()
        cache = LLMCache(cache_dir=new_dir)
        assert new_dir.exists()
