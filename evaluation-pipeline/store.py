"""
SQLite persistence layer. All reads/writes go through this module.

Generation key = hash(model_id, system_prompt_id, question_id, temperature)
Judgment key   = hash(generation_key, judge_id, metric_id)

Re-running the same command resumes safely because inserts use INSERT OR IGNORE.
Each thread opens its own connection (thread-local); all writes are serialized
via a module-level lock to prevent WAL conflicts.
"""
import hashlib
import logging
import sqlite3
import threading
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

_local = threading.local()
_write_lock = threading.Lock()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS generations (
    id                        TEXT PRIMARY KEY,
    model_id                  TEXT NOT NULL,
    system_prompt_id          TEXT NOT NULL,
    question_id               TEXT NOT NULL,
    temperature               REAL NOT NULL,
    question                  TEXT NOT NULL,
    language_tier             TEXT NOT NULL,
    domain                    TEXT NOT NULL,
    required_answer_elements  TEXT NOT NULL,
    forbidden_claims          TEXT NOT NULL,
    violated_forbidden_claim  TEXT,
    official_answer           TEXT NOT NULL,
    response_text             TEXT NOT NULL,
    input_tokens              INTEGER NOT NULL,
    output_tokens             INTEGER NOT NULL,
    cost_usd                  REAL NOT NULL,
    created_at                TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS judgments (
    id              TEXT PRIMARY KEY,
    generation_id   TEXT NOT NULL,
    judge_id        TEXT NOT NULL,
    metric_id       TEXT NOT NULL,
    name            TEXT NOT NULL,
    value           REAL,
    data_type       TEXT NOT NULL,
    comment         TEXT,
    input_tokens    INTEGER,
    output_tokens   INTEGER,
    cost_usd        REAL,
    error           TEXT,
    created_at      TEXT NOT NULL,
    FOREIGN KEY (generation_id) REFERENCES generations(id)
);
"""


def make_key(*parts) -> str:
    combined = "|".join(str(p) for p in parts)
    return hashlib.sha256(combined.encode()).hexdigest()


def _conn(db_path: str) -> sqlite3.Connection:
    if getattr(_local, "db_path", None) != db_path:
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.row_factory = sqlite3.Row
        _local.conn = conn
        _local.db_path = db_path
    return _local.conn


def init_db(db_path: str):
    conn = _conn(db_path)
    conn.executescript(_SCHEMA)
    conn.commit()


def generation_exists(db_path: str, key: str) -> bool:
    return _conn(db_path).execute(
        "SELECT 1 FROM generations WHERE id = ?", (key,)
    ).fetchone() is not None


def save_generation(db_path: str, record: dict):
    import json as _json
    with _write_lock:
        conn = _conn(db_path)
        conn.execute(
            """INSERT OR IGNORE INTO generations
               (id, model_id, system_prompt_id, question_id, temperature,
                question, language_tier, domain,
                required_answer_elements, forbidden_claims, violated_forbidden_claim,
                official_answer, response_text, input_tokens, output_tokens, cost_usd, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                record["id"], record["model_id"], record["system_prompt_id"],
                record["question_id"], record["temperature"],
                record["question"], record["language_tier"], record["domain"],
                _json.dumps(record["required_answer_elements"]),
                _json.dumps(record["forbidden_claims"]),
                record.get("violated_forbidden_claim"),
                record["official_answer"],
                record["response_text"], record["input_tokens"],
                record["output_tokens"], record["cost_usd"],
                record["created_at"],
            ),
        )
        conn.commit()


def judgment_exists(db_path: str, key: str) -> bool:
    return _conn(db_path).execute(
        "SELECT 1 FROM judgments WHERE id = ?", (key,)
    ).fetchone() is not None


def save_judgment(db_path: str, record: dict):
    with _write_lock:
        conn = _conn(db_path)
        conn.execute(
            """INSERT OR IGNORE INTO judgments
               (id, generation_id, judge_id, metric_id, name, value, data_type,
                comment, input_tokens, output_tokens, cost_usd, error, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                record["id"], record["generation_id"], record["judge_id"],
                record["metric_id"], record["name"], record.get("value"),
                record["data_type"], record.get("comment"),
                record.get("input_tokens"), record.get("output_tokens"),
                record.get("cost_usd"), record.get("error"),
                record["created_at"],
            ),
        )
        conn.commit()


def total_cost(db_path: str) -> float:
    conn = _conn(db_path)
    gen = conn.execute("SELECT COALESCE(SUM(cost_usd),0) FROM generations").fetchone()[0]
    jdg = conn.execute(
        "SELECT COALESCE(SUM(cost_usd),0) FROM judgments WHERE error IS NULL"
    ).fetchone()[0]
    return gen + jdg


def load_generations(db_path: str) -> list[dict]:
    import json as _json
    rows = _conn(db_path).execute("SELECT * FROM generations").fetchall()
    result = []
    for r in rows:
        d = dict(r)
        d["required_answer_elements"] = _json.loads(d["required_answer_elements"])
        d["forbidden_claims"] = _json.loads(d["forbidden_claims"])
        result.append(d)
    return result


def load_judgments(db_path: str) -> list[dict]:
    rows = _conn(db_path).execute("SELECT * FROM judgments").fetchall()
    return [dict(r) for r in rows]
