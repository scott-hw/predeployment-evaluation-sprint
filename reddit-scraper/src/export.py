"""
export.py — Stage 8: Export to Parquet + Markdown report.

Parquet: eaton_questions.parquet — all question records with all columns.
Report: eaton_report.md — grouped by program_area, top clusters with exemplar text.

Usage:
    python src/export.py
    python src/export.py --report-only   # skip Parquet
"""

import argparse
import sys
import pathlib
import logging
import datetime as _dt
from datetime import datetime

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from src.storage import get_db, load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def export_parquet(conn, out_path: pathlib.Path) -> int:
    """Export all question records to Parquet. Returns row count."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("Exporting to %s...", out_path)
    conn.execute(f"""
        COPY (
            SELECT
                id,
                parent_id,
                subreddit,
                type,
                title,
                COALESCE(body_clean, body) AS body,
                created_utc,
                score,
                depth,
                has_question_mark,
                lang,
                program_area_tags,
                question_type_tags,
                cluster_id,
                cluster_size,
                is_exemplar
            FROM records
            WHERE is_question = TRUE
            ORDER BY subreddit, created_utc
        ) TO '{out_path}' (FORMAT PARQUET)
    """)
    n = conn.execute("SELECT COUNT(*) FROM records WHERE is_question = TRUE").fetchone()[0]
    log.info("Exported %d question records to %s", n, out_path)
    return n


def generate_report(conn, out_path: pathlib.Path) -> None:
    """Generate a Markdown report grouped by program_area."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Gather overall stats
    total_records = conn.execute("SELECT COUNT(*) FROM records").fetchone()[0]
    total_questions = conn.execute("SELECT COUNT(*) FROM records WHERE is_question = TRUE").fetchone()[0]
    subreddits = conn.execute("""
        SELECT subreddit, COUNT(*) as n FROM records WHERE is_question = TRUE
        GROUP BY subreddit ORDER BY n DESC
    """).fetchall()

    # Questions with no program_area tag
    untagged = conn.execute("""
        SELECT COUNT(*) FROM records
        WHERE is_question = TRUE AND (program_area_tags IS NULL OR len(program_area_tags) = 0)
    """).fetchone()[0]

    # All program area tags
    pa_tags = conn.execute("""
        SELECT tag, COUNT(*) as n
        FROM (
            SELECT unnest(program_area_tags) as tag
            FROM records WHERE is_question = TRUE AND program_area_tags IS NOT NULL
        )
        GROUP BY tag ORDER BY n DESC
    """).fetchall()

    lines = [
        f"# Eaton Fire Question Mining — Report",
        f"",
        f"Generated: {_dt.datetime.now(_dt.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"",
        f"## Overview",
        f"",
        f"| Metric | Count |",
        f"|--------|-------|",
        f"| Total records (posts + comments) | {total_records:,} |",
        f"| Detected as questions | {total_questions:,} |",
        f"| Questions with no program tag | {untagged:,} |",
        f"",
        f"### By Subreddit",
        f"",
        f"| Subreddit | Questions |",
        f"|-----------|-----------|",
    ]
    for sub, n in subreddits:
        lines.append(f"| r/{sub} | {n:,} |")

    lines += [
        f"",
        f"### By Program Area (questions can have multiple tags)",
        f"",
        f"| Program Area | Count |",
        f"|--------------|-------|",
    ]
    for tag, n in pa_tags:
        lines.append(f"| {tag} | {n:,} |")

    # Per-area cluster sections
    lines += ["", "---", "", "## Top Clusters by Program Area", ""]
    lines.append("*(Exemplar = longest question in the cluster. Size = number of similar questions.)*")
    lines.append("")

    if not pa_tags:
        lines.append("*No program area tags found. Run tag.py first.*")
    else:
        for tag, total_n in pa_tags:
            lines += [f"### {tag.replace('_', ' ').title()} ({total_n:,} questions)", ""]

            # Top clusters for this tag
            clusters = conn.execute("""
                SELECT r.cluster_id, r.cluster_size, r.subreddit, r.created_utc,
                       COALESCE(r.body_clean, r.body) as body
                FROM records r
                WHERE r.is_question = TRUE
                  AND r.is_exemplar = TRUE
                  AND r.cluster_id >= 0
                  AND list_contains(r.program_area_tags, ?)
                ORDER BY r.cluster_size DESC
                LIMIT 15
            """, [tag]).fetchall()

            if not clusters:
                # Fallback: just show individual questions (no cluster assignment yet)
                rows = conn.execute("""
                    SELECT subreddit, created_utc, COALESCE(body_clean, body)
                    FROM records
                    WHERE is_question = TRUE AND list_contains(program_area_tags, ?)
                    ORDER BY score DESC NULLS LAST
                    LIMIT 10
                """, [tag]).fetchall()
                for sub, ts, body in rows:
                    preview = (body or "")[:300].replace("\n", " ")
                    lines.append(f"- **r/{sub}** — {preview}")
                lines.append("")
                continue

            for cid, csize, sub, ts, body in clusters:
                preview = (body or "")[:400].replace("\n", " ")
                ts_str = str(ts)[:10] if ts else "?"
                lines.append(f"**Cluster {cid}** ({csize} similar) — r/{sub}, {ts_str}")
                lines.append(f"> {preview}")
                lines.append("")

    # Untagged questions sample
    lines += ["---", "", "## Sample Untagged Questions (no program area matched)", ""]
    untagged_sample = conn.execute("""
        SELECT subreddit, COALESCE(body_clean, body)
        FROM records
        WHERE is_question = TRUE
          AND (program_area_tags IS NULL OR len(program_area_tags) = 0)
        ORDER BY score DESC NULLS LAST
        LIMIT 20
    """).fetchall()
    for sub, body in untagged_sample:
        preview = (body or "")[:200].replace("\n", " ")
        lines.append(f"- **r/{sub}** — {preview}")

    report = "\n".join(lines)
    out_path.write_text(report, encoding="utf-8")
    log.info("Report written to %s", out_path)


def main():
    parser = argparse.ArgumentParser(description="Export questions to Parquet and Markdown report")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--report-only", action="store_true", help="Skip Parquet export")
    args = parser.parse_args()

    cfg = load_config(args.config)
    conn = get_db(cfg["paths"]["db_path"])

    if not args.report_only:
        export_parquet(conn, pathlib.Path(cfg["paths"]["parquet_out"]))

    generate_report(conn, pathlib.Path(cfg["paths"]["report_out"]))
    conn.close()


if __name__ == "__main__":
    main()
