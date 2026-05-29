"""
backfill_comments.py — Fetch and ingest comments for govt-tagged question posts only.

Pulls comment trees from Arctic Shift for posts already tagged with government-service
program areas, writes them into the raw JSON bundles, and inserts them into DuckDB.
Skips any post whose raw file already has comments.

Usage:
    python backfill_comments.py
"""

import json
import pathlib
import sys
import time
import logging
import datetime

import duckdb
import requests

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from src.storage import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

BASE_URL = "https://arctic-shift.photon-reddit.com"
GOVT_TAGS = ['fema_ia', 'sba', 'dua', 'dmv', 'tax', 'd_snap', 'debris', 'permits', 'insurance', 'utilities']

_REMOVED = {"[deleted]", "[removed]", "", None}


# ── Arctic Shift helpers ───────────────────────────────────────────────────────

def fetch_comment_tree(session, post_id, delay, timeout):
    params = {"link_id": f"t3_{post_id}", "limit": 25000}
    for attempt in range(5):
        try:
            resp = session.get(f"{BASE_URL}/api/comments/tree", params=params, timeout=timeout)
            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", 10)) + 1
                log.warning("Rate limited — sleeping %ds", wait)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            time.sleep(delay)
            return resp.json().get("data") or []
        except requests.RequestException as e:
            log.warning("Request error (attempt %d/5): %s", attempt + 1, e)
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Failed after 5 attempts for post {post_id}")


def flatten_comments(nodes, flat, depth=0):
    for node in nodes:
        if not isinstance(node, dict) or node.get("kind") == "more":
            continue
        # Arctic Shift wraps comment fields under "data"; unwrap if present
        inner = node.get("data") or node
        comment = {k: v for k, v in inner.items() if k != "replies"}
        comment["_depth"] = depth
        flat.append(comment)
        replies = inner.get("replies") or node.get("replies") or []
        flatten_comments(replies, flat, depth + 1)


# ── Normalize helpers (mirrors normalize.py) ──────────────────────────────────

def _parse_utc(val):
    if val is None:
        return None
    if isinstance(val, str):
        return val.replace("Z", "").replace("T", " ")
    return datetime.datetime.fromtimestamp(float(val), tz=datetime.timezone.utc).replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S")


def comments_to_records(comments, subreddit):
    records = []
    for c in comments:
        if c.get("kind") == "more":
            continue
        body = (c.get("body") or "").strip()
        if body in _REMOVED or not body:
            continue
        comment_id = c.get("id")
        if not comment_id:
            continue
        parent_raw = c.get("parent_id") or ""
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


def upsert_records(conn, records):
    inserted = 0
    for r in records:
        try:
            conn.execute("""
                INSERT OR IGNORE INTO records
                    (id, parent_id, subreddit, type, title, body, created_utc, score, depth, has_question_mark)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [r["id"], r["parent_id"], r["subreddit"], r["type"],
                  r["title"], r["body"], r["created_utc"], r["score"],
                  r["depth"], r["has_question_mark"]])
            inserted += 1
        except Exception as e:
            log.debug("Skipping %s: %s", r["id"], e)
    return inserted


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    cfg = load_config("config.yaml")
    db_path = cfg["paths"]["db_path"]
    raw_dir = pathlib.Path(cfg["paths"]["raw_dir"])
    delay   = cfg["arctic_shift"]["request_delay"]
    timeout = cfg["arctic_shift"]["timeout"]

    conn = duckdb.connect(db_path)

    # Pull the post IDs we care about
    tag_filter = " OR ".join([f"list_contains(program_area_tags, '{t}')" for t in GOVT_TAGS])
    rows = conn.execute(f"""
        SELECT id, subreddit
        FROM records
        WHERE is_question = TRUE AND ({tag_filter})
        ORDER BY subreddit, id
    """).fetchall()

    log.info("Found %d govt-tagged question posts to backfill comments for", len(rows))

    session = requests.Session()
    session.headers["User-Agent"] = "eaton-question-pipeline/1.0 (research)"

    total_fetched  = 0
    total_inserted = 0
    skipped        = 0

    for i, (post_id, subreddit) in enumerate(rows, 1):
        raw_file = raw_dir / subreddit / f"{post_id}.json"

        # Check if raw file already has comments
        if raw_file.exists():
            bundle = json.loads(raw_file.read_text(encoding="utf-8"))
            if bundle.get("comments"):
                skipped += 1
                continue  # already have comments for this post

        # Fetch from API
        try:
            flat = []
            tree = fetch_comment_tree(session, post_id, delay, timeout)
            flatten_comments(tree, flat)
        except Exception as e:
            log.warning("[%d/%d] Failed to fetch comments for %s: %s", i, len(rows), post_id, e)
            continue

        # Update raw file
        if raw_file.exists():
            bundle = json.loads(raw_file.read_text(encoding="utf-8"))
            bundle["comments"] = flat
            raw_file.write_text(json.dumps(bundle, ensure_ascii=False))

        # Insert into DB
        records = comments_to_records(flat, subreddit)
        n = upsert_records(conn, records)
        total_fetched  += len(flat)
        total_inserted += n

        if i % 25 == 0 or i == len(rows):
            conn.commit()
            log.info("[%d/%d] post %s — %d comments fetched, %d inserted (running total: %d)",
                     i, len(rows), post_id, len(flat), n, total_inserted)

    conn.commit()
    log.info("Done. Skipped %d (already had comments). Fetched %d comments, inserted %d records.",
             skipped, total_fetched, total_inserted)

    summary = conn.execute("SELECT type, COUNT(*) as n FROM records GROUP BY type ORDER BY type").fetchall()
    for row in summary:
        print(f"  {row[0]}: {row[1]:,}")

    conn.close()


if __name__ == "__main__":
    main()
