"""
Disk-backed LLM response cache.

Cache key: SHA256(model_name + str(temperature) + str(max_tokens) + prompt_text)
Storage: data/cache/<cache_key>.json
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[3] / "data" / "cache"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_cache_key(
    model_name: str,
    temperature: float,
    max_tokens: int,
    prompt_text: str,
) -> str:
    """Return a 64-char hex SHA-256 digest used as the cache key."""
    raw = f"{model_name}{str(temperature)}{str(max_tokens)}{prompt_text}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class LLMCache:
    """Simple file-based cache for LLM responses."""

    def __init__(self, cache_dir: Optional[Path] = None) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else _DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.debug("LLMCache initialised at %s", self.cache_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def make_key(
        self,
        model_name: str,
        temperature: float,
        max_tokens: int,
        prompt_text: str,
    ) -> str:
        return _build_cache_key(model_name, temperature, max_tokens, prompt_text)

    def get(self, cache_key: str) -> Optional[dict]:
        """
        Return cached entry dict if present, else None.

        Side-effect: updates hit_count and last_accessed on hit.
        """
        path = self._path(cache_key)
        if not path.exists():
            logger.debug("Cache MISS: %s", cache_key[:16])
            return None

        try:
            with path.open("r", encoding="utf-8") as fh:
                entry = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Cache read error (%s): %s — treating as miss", cache_key[:16], exc)
            return None

        # Update access metadata
        entry.setdefault("hit_count", 0)
        entry["hit_count"] += 1
        entry["last_accessed"] = _now_iso()
        entry["from_cache"] = True
        self._write(cache_key, entry)

        logger.debug("Cache HIT : %s (hits=%d)", cache_key[:16], entry["hit_count"])
        return entry

    def set(
        self,
        cache_key: str,
        response_json: dict,
        metadata: Optional[dict] = None,
    ) -> None:
        """Persist a response to disk."""
        entry: dict = {
            "model": metadata.get("model", "") if metadata else "",
            "temperature": metadata.get("temperature", None) if metadata else None,
            "max_tokens": metadata.get("max_tokens", None) if metadata else None,
            "prompt_hash": cache_key,
            "hit_count": 0,
            "last_accessed": _now_iso(),
            "created_at": _now_iso(),
            "from_cache": False,
            "response": response_json,
        }
        if metadata:
            for k, v in metadata.items():
                if k not in entry:
                    entry[k] = v

        self._write(cache_key, entry)
        logger.debug("Cache SET : %s", cache_key[:16])

    def clear(self) -> int:
        """Delete all cached entries. Returns count deleted."""
        count = 0
        for p in self.cache_dir.glob("*.json"):
            try:
                p.unlink()
                count += 1
            except OSError as exc:
                logger.warning("Could not delete cache file %s: %s", p, exc)
        logger.info("Cache cleared: %d entries removed", count)
        return count

    def size(self) -> int:
        """Return number of cached entries on disk."""
        return sum(1 for _ in self.cache_dir.glob("*.json"))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _path(self, cache_key: str) -> Path:
        return self.cache_dir / f"{cache_key}.json"

    def _write(self, cache_key: str, entry: dict) -> None:
        path = self._path(cache_key)
        tmp = path.with_suffix(".tmp")
        try:
            with tmp.open("w", encoding="utf-8") as fh:
                json.dump(entry, fh, indent=2, ensure_ascii=False)
            tmp.replace(path)
        except OSError as exc:
            logger.error("Cache write error (%s): %s", cache_key[:16], exc)
            if tmp.exists():
                tmp.unlink(missing_ok=True)
