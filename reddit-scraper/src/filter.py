"""
filter.py — Stage 3: Question detection via regex pre-filter.

Marks records with is_question=True where the text:
  - Contains a '?', OR
  - Starts with a question-signaling word/phrase, OR
  - Is between min_chars and max_chars

No LLM; expect ~60-75% precision. False positives (rhetorical questions, rants)
are fine — the Markdown report is meant to be eyeballed.

Usage:
    python src/filter.py
"""

import argparse
import re
import sys
import pathlib
import logging

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from src.storage import get_db, load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# Question-opening word/phrase pattern (case-insensitive, at start of body or within first 200 chars)
_QUESTION_START = re.compile(
    r"\b(how|where|what|when|can|does|do|is|are|should|anyone|has anyone|"
    r"where do|how do|do i|am i|will i|can i|who|which|why|would|could|"
    r"has anyone|did anyone|does anyone|is there|are there|"
    r"anyone know|does anyone know|would anyone)\b",
    re.IGNORECASE,
)


def is_question(text: str, min_chars: int, max_chars: int) -> bool:
    if not text:
        return False
    n = len(text)
    if n < min_chars or n > max_chars:
        return False
    if "?" in text:
        return True
    # Check first 200 chars for question-opening phrases
    snippet = text[:200]
    if _QUESTION_START.search(snippet):
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description="Mark question-candidate records")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--reset", action="store_true", help="Re-run filter from scratch")
    args = parser.parse_args()

    cfg = load_config(args.config)
    conn = get_db(cfg["paths"]["db_path"])
    min_chars = cfg["filter"]["min_chars"]
    max_chars = cfg["filter"]["max_chars"]

    if args.reset:
        conn.execute("UPDATE records SET is_question = NULL")
        conn.commit()

    # Fetch records not yet filtered
    rows = conn.execute("""
        SELECT id, body FROM records
        WHERE is_question IS NULL AND body IS NOT NULL
    """).fetchall()

    log.info("Filtering %d records (min=%d, max=%d chars)...", len(rows), min_chars, max_chars)

    questions = 0
    batch = []
    for row_id, body in rows:
        result = is_question(body, min_chars, max_chars)
        batch.append((result, row_id))
        if result:
            questions += 1
        if len(batch) >= 1000:
            conn.executemany("UPDATE records SET is_question = ? WHERE id = ?", batch)
            batch = []

    if batch:
        conn.executemany("UPDATE records SET is_question = ? WHERE id = ?", batch)

    conn.commit()

    total = len(rows)
    pct = questions / total * 100 if total else 0
    log.info("Done. Questions: %d / %d (%.1f%%)", questions, total, pct)
    conn.close()


if __name__ == "__main__":
    main()
