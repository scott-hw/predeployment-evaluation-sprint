"""
normalize.py — Stage 2: Flatten raw JSON bundles into the DuckDB `records` table.

Reads from data/raw/{subreddit}/{post_id}.json
Writes to the `records` table in DuckDB.

Usage:
    python src/normalize.py
    python src/normalize.py --subreddit Altadena
"""

import argparse
import json
import pathlib
import sys
import logging

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from src.storage import get_db, load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

_REMOVED = {"[deleted]", "[removed]", "", None}


def _parse_utc(val) -> str | None:
    """Coerce created_utc (float, int, or ISO string) to ISO string for DuckDB."""
    if val is None:
        return None
    if isinstance(val, str):
        # Already ISO-ish: "2025-01-07T12:34:56.000Z" → strip Z for DuckDB
        return val.replace("Z", "").replace("T", " ")
    # Unix timestamp
    import datetime
    return datetime.datetime.fromtimestamp(float(val), tz=datetime.timezone.utc).replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S")


def _body_text(post_or_comment: dict) -> str:
    """Extract body text from a post or comment dict."""
    text = post_or_comment.get("selftext") or post_or_comment.get("body") or ""
    return text.strip()


def _is_removed(text: str) -> bool:
    return text in _REMOVED or not text


def normalize_bundle(bundle: dict, subreddit: str) -> list[dict]:
    """Convert a raw post+comments bundle into a list of record dicts."""
    records = []

    post = bundle.get("post", {})
    post_id = post.get("id")
    if not post_id:
        return []

    title = (post.get("title") or "").strip()
    body = _body_text(post)

    # Skip deleted/removed posts with no title either
    if _is_removed(body) and not title:
        return []

    post_text = f"{title} {body}".strip()
    records.append({
        "id": post_id,
        "parent_id": None,
        "subreddit": subreddit,
        "type": "post",
        "title": title or None,
        "body": post_text or None,
        "created_utc": _parse_utc(post.get("created_utc")),
        "score": post.get("score", 0),
        "depth": 0,
        "has_question_mark": "?" in post_text,
    })

    for c in bundle.get("comments", []):
        if c.get("kind") == "more":
            continue
        body = _body_text(c)
        if _is_removed(body):
            continue
        comment_id = c.get("id")
        if not comment_id:
            continue
        parent_raw = c.get("parent_id") or ""
        # parent_id may be "t1_xyz" (comment) or "t3_xyz" (post) — strip prefix
        parent_id = parent_raw.split("_", 1)[-1] if "_" in parent_raw else parent_raw or None
        depth = c.get("_depth", c.get("depth", 1))

        records.append({
            "id": comment_id,
            "parent_id": parent_id,
            "subreddit": subreddit,
            "type": "comment",
            "title": None,
            "body": body,
            "created_utc": _parse_utc(c.get("created_utc")),
            "score": c.get("score", 0),
            "depth": depth,
            "has_question_mark": "?" in body,
        })

    return records


def _upsert_records(conn, records: list[dict]) -> int:
    """Insert records, skipping any that already exist (by id)."""
    inserted = 0
    for r in records:
        try:
            conn.execute("""
                INSERT OR IGNORE INTO records
                    (id, parent_id, subreddit, type, title, body, created_utc, score, depth, has_question_mark)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                r["id"], r["parent_id"], r["subreddit"], r["type"],
                r["title"], r["body"], r["created_utc"], r["score"],
                r["depth"], r["has_question_mark"],
            ])
            inserted += 1
        except Exception as e:
            log.debug("Skipping record %s: %s", r["id"], e)
    return inserted


def main():
    parser = argparse.ArgumentParser(description="Normalize raw JSON bundles into DuckDB records")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--subreddit", help="Process only this subreddit's raw files")
    args = parser.parse_args()

    cfg = load_config(args.config)
    conn = get_db(cfg["paths"]["db_path"])
    raw_dir = pathlib.Path(cfg["paths"]["raw_dir"])

    if not raw_dir.exists():
        log.error("Raw data directory not found: %s — run collect.py first", raw_dir)
        sys.exit(1)

    subdirs = sorted(raw_dir.iterdir())
    if args.subreddit:
        subdirs = [d for d in subdirs if d.name.lower() == args.subreddit.lower()]

    grand_total = 0
    for sub_dir in subdirs:
        if not sub_dir.is_dir():
            continue
        subreddit = sub_dir.name
        json_files = list(sub_dir.glob("*.json"))
        log.info("[%s] Normalizing %d files...", subreddit, len(json_files))
        sub_total = 0
        for jf in json_files:
            try:
                bundle = json.loads(jf.read_text(encoding="utf-8"))
                recs = normalize_bundle(bundle, subreddit)
                inserted = _upsert_records(conn, recs)
                sub_total += inserted
            except Exception as e:
                log.warning("Failed to process %s: %s", jf, e)
        conn.commit()
        log.info("[%s] Inserted %d records", subreddit, sub_total)
        grand_total += sub_total

    log.info("Normalization complete. Total records: %d", grand_total)

    # Quick summary
    summary = conn.execute("""
        SELECT type, COUNT(*) as n FROM records GROUP BY type ORDER BY type
    """).fetchall()
    for row in summary:
        print(f"  {row[0]}: {row[1]:,}")

    conn.close()


if __name__ == "__main__":
    main()
