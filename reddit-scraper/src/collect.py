"""
collect.py — Stage 1: Collect posts and comments from Reddit via Arctic Shift API.

Two collection patterns:
  - Full subreddit pull: all posts from r/Altadena, r/Pasadena in the date window.
  - Keyword search: title-keyword-filtered posts from r/losangeles, r/California, etc.

Usage:
    python src/collect.py                  # run all subreddits from config
    python src/collect.py --subreddit Altadena   # single subreddit
    python src/collect.py --coverage-check       # just count posts, don't save
"""

import argparse
import json
import pathlib
import sys
import time
import datetime
import logging
import os

import requests

# Add project root to path so we can import storage
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from src.storage import get_db, load_config, upsert_collection_log, get_collection_checkpoint

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

BASE_URL = "https://arctic-shift.photon-reddit.com"


# ---------------------------------------------------------------------------
# Arctic Shift HTTP helpers
# ---------------------------------------------------------------------------

def _get(session: requests.Session, endpoint: str, params: dict, delay: float, timeout: int) -> dict:
    """GET with retry on rate-limit or transient errors."""
    url = f"{BASE_URL}{endpoint}"
    for attempt in range(5):
        try:
            resp = session.get(url, params=params, timeout=timeout)
            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", 10)) + 1
                log.warning("Rate limited — sleeping %ds", wait)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            time.sleep(delay)
            return resp.json()
        except requests.RequestException as e:
            log.warning("Request error (attempt %d/5): %s", attempt + 1, e)
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Failed after 5 attempts: {url} {params}")


def fetch_posts_page(
    session: requests.Session,
    subreddit: str,
    after: str,
    before: str,
    limit: int,
    delay: float,
    timeout: int,
    title_keyword: str | None = None,
) -> list[dict]:
    """Fetch one page of posts from Arctic Shift, sorted ascending by created_utc."""
    params = {
        "subreddit": subreddit,
        "after": after,
        "before": before,
        "sort": "asc",
        "limit": limit,
    }
    if title_keyword:
        # Arctic Shift title search — works when subreddit is specified
        params["title"] = title_keyword
    data = _get(session, "/api/posts/search", params, delay, timeout)
    return data.get("data") or []


def fetch_comment_tree(
    session: requests.Session,
    post_id: str,
    max_comments: int,
    delay: float,
    timeout: int,
) -> list[dict]:
    """Fetch all comments for a post as a flat list."""
    params = {
        "link_id": f"t3_{post_id}",
        "limit": min(max_comments, 25000),
    }
    data = _get(session, "/api/comments/tree", params, delay, timeout)
    comments_raw = data.get("data") or []
    flat = []
    _flatten_comments(comments_raw, flat, depth=0)
    return flat


def _flatten_comments(nodes: list, flat: list, depth: int) -> None:
    """Recursively flatten the comment tree into a list, skipping 'more' stubs."""
    for node in nodes:
        if not isinstance(node, dict):
            continue
        if node.get("kind") == "more":
            continue  # collapsed stub — skip
        # Extract comment fields
        comment = {k: v for k, v in node.items() if k != "replies"}
        comment["_depth"] = depth
        flat.append(comment)
        replies = node.get("replies") or []
        _flatten_comments(replies, flat, depth + 1)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_post_bundle(raw_dir: pathlib.Path, subreddit: str, post: dict, comments: list[dict]) -> pathlib.Path:
    """Save post + its comments as a single JSON file. Returns the file path."""
    sub_dir = raw_dir / subreddit
    sub_dir.mkdir(parents=True, exist_ok=True)
    post_id = post["id"]
    bundle = {"post": post, "comments": comments}
    out_path = sub_dir / f"{post_id}.json"
    out_path.write_text(json.dumps(bundle, ensure_ascii=False))
    return out_path


# ---------------------------------------------------------------------------
# Collection runners
# ---------------------------------------------------------------------------

def _utc_to_iso(utc_val) -> str:
    """Convert created_utc (float, int, or ISO string) to an ISO date string for the API."""
    if isinstance(utc_val, str):
        # Arctic Shift sometimes returns ISO strings like "2025-01-07T12:34:56.000Z"
        # We want just the date for the `after` param
        return utc_val[:10]
    # Unix timestamp
    return datetime.datetime.fromtimestamp(float(utc_val), tz=datetime.timezone.utc).replace(tzinfo=None).strftime("%Y-%m-%dT%H:%M:%S")


def collect_full_pull(
    conn,
    session: requests.Session,
    subreddit: str,
    cfg: dict,
    raw_dir: pathlib.Path,
    fetch_comments: bool,
) -> int:
    """Collect all posts (+ comments) from a subreddit in the date window."""
    after_cfg = cfg["collection"]["after"]
    before_cfg = cfg["collection"]["before"]
    page_size = cfg["arctic_shift"]["page_size"]
    delay = cfg["arctic_shift"]["request_delay"]
    timeout = cfg["arctic_shift"]["timeout"]
    max_comments = cfg["collection"]["max_comments_per_post"]

    # Resume from checkpoint if available
    _, last_utc = get_collection_checkpoint(conn, subreddit, "full_pull")
    after = _utc_to_iso(last_utc) if last_utc else after_cfg

    total = 0
    seen_ids: set[str] = set()

    log.info("[%s] Starting full pull (after=%s, before=%s)", subreddit, after, before_cfg)

    while True:
        posts = fetch_posts_page(session, subreddit, after, before_cfg, page_size, delay, timeout)
        if not posts:
            log.info("[%s] No more posts — full pull complete. Total: %d", subreddit, total)
            break

        new_posts = [p for p in posts if p["id"] not in seen_ids]
        if not new_posts:
            log.info("[%s] All posts in page already seen — stopping.", subreddit)
            break

        for post in new_posts:
            seen_ids.add(post["id"])
            comments = []
            if fetch_comments:
                try:
                    comments = fetch_comment_tree(session, post["id"], max_comments, delay, timeout)
                except Exception as e:
                    log.warning("[%s] Comment fetch failed for %s: %s", subreddit, post["id"], e)
            save_post_bundle(raw_dir, subreddit, post, comments)
            total += 1
            if total % 50 == 0:
                log.info("[%s] Collected %d posts...", subreddit, total)

        last_post = posts[-1]
        last_utc_val = last_post.get("created_utc", 0)
        upsert_collection_log(
            conn, subreddit, "full_pull", None,
            last_post["id"], float(last_utc_val) if not isinstance(last_utc_val, str) else 0,
            total
        )
        conn.commit()

        if len(posts) < page_size:
            log.info("[%s] Last page — full pull complete. Total: %d", subreddit, total)
            break

        # Advance the cursor: use the timestamp of the last post
        after = _utc_to_iso(last_utc_val)

    return total


def collect_keyword_search(
    conn,
    session: requests.Session,
    subreddit: str,
    keywords: list[str],
    cfg: dict,
    raw_dir: pathlib.Path,
    fetch_comments: bool,
) -> int:
    """Collect posts matching any keyword (via title search) in the date window."""
    after_cfg = cfg["collection"]["after"]
    before_cfg = cfg["collection"]["before"]
    page_size = cfg["arctic_shift"]["page_size"]
    delay = cfg["arctic_shift"]["request_delay"]
    timeout = cfg["arctic_shift"]["timeout"]
    max_comments = cfg["collection"]["max_comments_per_post"]

    total = 0
    seen_ids: set[str] = set()  # deduplicate across keywords

    for keyword in keywords:
        log.info("[%s] Keyword search: '%s'", subreddit, keyword)

        # Resume checkpoint per keyword
        _, last_utc = get_collection_checkpoint(conn, subreddit, "keyword_search", keyword)
        after = _utc_to_iso(last_utc) if last_utc else after_cfg

        keyword_count = 0

        while True:
            posts = fetch_posts_page(
                session, subreddit, after, before_cfg, page_size, delay, timeout,
                title_keyword=keyword
            )
            if not posts:
                break

            new_posts = [p for p in posts if p["id"] not in seen_ids]

            for post in new_posts:
                seen_ids.add(post["id"])
                comments = []
                if fetch_comments:
                    try:
                        comments = fetch_comment_tree(session, post["id"], max_comments, delay, timeout)
                    except Exception as e:
                        log.warning("[%s] Comment fetch failed for %s: %s", subreddit, post["id"], e)
                save_post_bundle(raw_dir, subreddit, post, comments)
                keyword_count += 1
                total += 1

            last_post = posts[-1]
            last_utc_val = last_post.get("created_utc", 0)
            upsert_collection_log(
                conn, subreddit, "keyword_search", keyword,
                last_post["id"], float(last_utc_val) if not isinstance(last_utc_val, str) else 0,
                keyword_count
            )
            conn.commit()

            if len(posts) < page_size:
                break

            after = _utc_to_iso(last_utc_val)

        log.info("[%s] Keyword '%s' done: %d new posts", subreddit, keyword, keyword_count)

    log.info("[%s] Keyword searches complete. Total new: %d", subreddit, total)
    return total


# ---------------------------------------------------------------------------
# Coverage check
# ---------------------------------------------------------------------------

def coverage_check(session: requests.Session, cfg: dict) -> None:
    """Print post counts for all target subreddits without saving anything.

    Uses a sample probe (first + last page of the window) since the aggregate
    endpoint times out for large/busy subreddits.
    """
    after = cfg["collection"]["after"]
    before = cfg["collection"]["before"]
    delay = cfg["arctic_shift"]["request_delay"]
    timeout = cfg["arctic_shift"]["timeout"]

    print("\n=== Coverage Check ===")
    print(f"Window: {after} → {before}\n")

    def sample_count(sub: str, title_kw: str | None = None) -> str:
        """Return a coverage indicator: first/last page sizes."""
        params_first = {"subreddit": sub, "after": after, "before": before,
                        "sort": "asc", "limit": 5}
        params_last = {"subreddit": sub, "after": after, "before": before,
                       "sort": "desc", "limit": 5}
        if title_kw:
            params_first["title"] = title_kw
            params_last["title"] = title_kw
        try:
            r1 = session.get(f"{BASE_URL}/api/posts/search", params=params_first, timeout=timeout)
            time.sleep(delay)
            first_posts = (r1.json().get("data") or []) if r1.ok else []
            r2 = session.get(f"{BASE_URL}/api/posts/search", params=params_last, timeout=timeout)
            time.sleep(delay)
            last_posts = (r2.json().get("data") or []) if r2.ok else []

            if not first_posts:
                return "  0 results found  ⚠"
            first_ts = first_posts[0].get("created_utc", "?")
            last_ts = last_posts[0].get("created_utc", "?") if last_posts else "?"

            def fmt_ts(ts):
                if isinstance(ts, (int, float)):
                    return datetime.datetime.fromtimestamp(float(ts), tz=datetime.timezone.utc).replace(tzinfo=None).strftime("%Y-%m-%d")
                return str(ts)[:10]

            return f"  ✓ has data ({fmt_ts(first_ts)} … {fmt_ts(last_ts)})"
        except Exception as e:
            return f"  ERROR: {e}"

    for sub in cfg["collection"].get("full_pulls", []):
        indicator = sample_count(sub)
        print(f"  r/{sub:<20} (full pull)   {indicator}")

    for entry in cfg["collection"].get("keyword_searches", []):
        sub = entry["subreddit"]
        kws = entry["keywords"]
        # Check with the first keyword as a representative probe
        indicator = sample_count(sub, title_kw=kws[0])
        print(f"  r/{sub:<20} (keyword: '{kws[0]}') {indicator}")
    print()


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Collect Eaton fire Reddit posts via Arctic Shift")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--subreddit", help="Collect only this subreddit")
    parser.add_argument("--coverage-check", action="store_true", help="Just print post counts, don't collect")
    parser.add_argument("--no-comments", action="store_true", help="Skip comment fetching (faster)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    raw_dir = pathlib.Path(cfg["paths"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers["User-Agent"] = "eaton-question-pipeline/1.0 (research; contact: researcher@example.com)"

    if args.coverage_check:
        coverage_check(session, cfg)
        return

    conn = get_db(cfg["paths"]["db_path"])
    fetch_comments = cfg["collection"]["fetch_comments"] and not args.no_comments

    grand_total = 0

    # Full subreddit pulls
    for sub in cfg["collection"].get("full_pulls", []):
        if args.subreddit and args.subreddit.lower() != sub.lower():
            continue
        n = collect_full_pull(conn, session, sub, cfg, raw_dir, fetch_comments)
        grand_total += n

    # Keyword searches
    for entry in cfg["collection"].get("keyword_searches", []):
        sub = entry["subreddit"]
        if args.subreddit and args.subreddit.lower() != sub.lower():
            continue
        n = collect_keyword_search(
            conn, session, sub, entry["keywords"], cfg, raw_dir, fetch_comments
        )
        grand_total += n

    log.info("Collection complete. Grand total: %d posts", grand_total)
    conn.close()


if __name__ == "__main__":
    main()
