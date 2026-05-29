"""
tag.py — Stage 6: Keyword-based tagging + language detection.

Reads from records WHERE is_question = TRUE.
Writes program_area_tags, question_type_tags, and lang columns.

Tagging: multi-label, case-insensitive substring matching on body_clean (or body).
Language: langdetect — marks short/noisy texts as 'unknown'.

Usage:
    python src/tag.py
    python src/tag.py --reset   # re-tag everything
"""

import argparse
import sys
import pathlib
import logging

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from src.storage import get_db, load_config, load_keywords

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

try:
    from langdetect import detect, LangDetectException
    _LANGDETECT_OK = True
except ImportError:
    log.warning("langdetect not installed — all languages will be marked 'unknown'")
    _LANGDETECT_OK = False


def detect_language(text: str) -> str:
    if not _LANGDETECT_OK or not text or len(text) < 40:
        return "unknown"
    try:
        return detect(text)
    except Exception:
        return "unknown"


def apply_keyword_tags(text: str, keyword_dict: dict[str, list[str]]) -> list[str]:
    """Return list of category labels that have at least one keyword match in text."""
    if not text:
        return []
    text_lower = text.lower()
    matched = []
    for label, keywords in keyword_dict.items():
        for kw in keywords:
            if kw.lower() in text_lower:
                matched.append(label)
                break  # one match per label is enough
    return matched


def main():
    parser = argparse.ArgumentParser(description="Tag question records with program area and question type labels")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--keywords", default="keywords.yaml")
    parser.add_argument("--reset", action="store_true", help="Re-tag all records from scratch")
    parser.add_argument("--all", action="store_true", help="Tag all records, not just questions")
    args = parser.parse_args()

    cfg = load_config(args.config)
    kw_data = load_keywords(args.keywords)
    conn = get_db(cfg["paths"]["db_path"])

    program_kws: dict[str, list[str]] = kw_data.get("program_area", {})
    question_kws: dict[str, list[str]] = kw_data.get("question_type", {})

    if args.reset:
        conn.execute("UPDATE records SET lang = NULL, program_area_tags = NULL, question_type_tags = NULL")
        conn.commit()

    if args.all:
        where = "body IS NOT NULL AND lang IS NULL"
    else:
        where = "is_question = TRUE AND body IS NOT NULL AND lang IS NULL"

    rows = conn.execute(f"""
        SELECT id, COALESCE(body_clean, body) as text
        FROM records
        WHERE {where}
    """).fetchall()

    log.info("Tagging %d records...", len(rows))

    batch = []
    tagged = 0
    for row_id, text in rows:
        lang = detect_language(text)
        pa_tags = apply_keyword_tags(text, program_kws)
        qt_tags = apply_keyword_tags(text, question_kws)
        batch.append((lang, pa_tags, qt_tags, row_id))
        tagged += 1
        if len(batch) >= 500:
            conn.executemany(
                "UPDATE records SET lang = ?, program_area_tags = ?, question_type_tags = ? WHERE id = ?",
                batch
            )
            batch = []
            if tagged % 5000 == 0:
                log.info("  ...%d tagged", tagged)

    if batch:
        conn.executemany(
            "UPDATE records SET lang = ?, program_area_tags = ?, question_type_tags = ? WHERE id = ?",
            batch
        )

    conn.commit()
    log.info("Tagging complete: %d records", tagged)

    # Summary
    print("\n=== Program Area Distribution ===")
    rows = conn.execute("""
        SELECT tag, COUNT(*) as n
        FROM (
            SELECT unnest(program_area_tags) as tag
            FROM records WHERE is_question = TRUE AND program_area_tags IS NOT NULL
        )
        GROUP BY tag ORDER BY n DESC LIMIT 20
    """).fetchall()
    for tag, n in rows:
        print(f"  {tag:<25} {n:>5}")

    print("\n=== Language Distribution ===")
    rows = conn.execute("""
        SELECT lang, COUNT(*) as n
        FROM records WHERE is_question = TRUE
        GROUP BY lang ORDER BY n DESC LIMIT 10
    """).fetchall()
    for lang, n in rows:
        print(f"  {lang:<10} {n:>5}")

    conn.close()


if __name__ == "__main__":
    main()
