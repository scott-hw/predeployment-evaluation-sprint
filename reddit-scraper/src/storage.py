"""
storage.py — DuckDB schema initialization and connection helpers.
"""

import duckdb
import pathlib
import yaml


def get_db(db_path: str) -> duckdb.DuckDBPyConnection:
    """Open (or create) the pipeline DuckDB database."""
    pathlib.Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(db_path)
    _init_schema(conn)
    return conn


def _init_schema(conn: duckdb.DuckDBPyConnection) -> None:
    """Create tables if they don't already exist."""

    conn.execute("""
        CREATE TABLE IF NOT EXISTS collection_log (
            subreddit       TEXT NOT NULL,
            collection_type TEXT NOT NULL,   -- 'full_pull' | 'keyword_search'
            keyword         TEXT NOT NULL DEFAULT '',  -- '' for full pulls
            last_post_id    TEXT,
            last_utc        DOUBLE,
            post_count      INTEGER DEFAULT 0,
            updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (subreddit, collection_type, keyword)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS records (
            id                  TEXT PRIMARY KEY,
            parent_id           TEXT,
            subreddit           TEXT,
            type                TEXT,          -- 'post' | 'comment'
            title               TEXT,
            body                TEXT,
            created_utc         TIMESTAMP,
            score               INTEGER,
            depth               INTEGER,
            has_question_mark   BOOLEAN,

            -- populated by filter.py
            is_question         BOOLEAN,

            -- populated by cleanup.py
            body_clean          TEXT,

            -- populated by tag.py
            lang                TEXT,
            program_area_tags   TEXT[],
            question_type_tags  TEXT[],

            -- populated by cluster.py
            cluster_id          INTEGER,
            cluster_size        INTEGER,
            is_exemplar         BOOLEAN
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS raw_posts (
            post_id     TEXT PRIMARY KEY,
            subreddit   TEXT,
            raw_json    TEXT,
            fetched_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_keywords(keywords_path: str = "keywords.yaml") -> dict:
    with open(keywords_path) as f:
        return yaml.safe_load(f)


def upsert_collection_log(
    conn: duckdb.DuckDBPyConnection,
    subreddit: str,
    collection_type: str,
    keyword: str | None,
    last_post_id: str,
    last_utc: float,
    post_count: int,
) -> None:
    kw = keyword or ""
    conn.execute("""
        INSERT OR REPLACE INTO collection_log
            (subreddit, collection_type, keyword, last_post_id, last_utc, post_count, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
    """, [subreddit, collection_type, kw, last_post_id, last_utc, post_count])


def get_collection_checkpoint(
    conn: duckdb.DuckDBPyConnection,
    subreddit: str,
    collection_type: str,
    keyword: str | None = None,
) -> tuple[str | None, float | None]:
    """Return (last_post_id, last_utc) for resuming a collection run."""
    kw = keyword or ""
    row = conn.execute("""
        SELECT last_post_id, last_utc
        FROM collection_log
        WHERE subreddit = ? AND collection_type = ? AND keyword = ?
    """, [subreddit, collection_type, kw]).fetchone()
    if row:
        return row[0], row[1]
    return None, None
