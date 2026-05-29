# Eaton Fire Question Mining Pipeline

Surfaces real questions Californians asked on Reddit after the Eaton Fire (Jan 2025). No API keys. No LLM calls. Runs entirely on a single machine.

## What it does

1. **Collect** — pulls posts and comment trees from Reddit via the [Arctic Shift](https://arctic-shift.photon-reddit.com) archive (no Reddit credentials)
2. **Normalize** — flattens raw JSON into a DuckDB table
3. **Filter** — regex-based question detection (`?`, question-opening words, length bounds)
4. **Cleanup** — strips HTML entities, mojibake, Reddit markdown
5. **Tag** — keyword-dictionary labeling by program area (FEMA, insurance, debris, etc.) and question type
6. **Cluster** — local sentence embeddings (`paraphrase-multilingual-MiniLM-L12-v2`) + HDBSCAN to group similar questions
7. **Export** — Parquet file + Markdown report grouped by program area

## Data sources

| Subreddit | Mode |
|-----------|------|
| r/Altadena | Full pull (all posts Jan 7 – Jun 30 2025) |
| r/Pasadena | Full pull |
| r/losangeles | Keyword search (`eaton fire`, `altadena fire`, …) |
| r/California | Keyword search |
| r/Insurance | Keyword search |
| r/personalfinance | Keyword search |

> **Note:** r/EatonFire and r/LACountyFires have zero posts in the Arctic Shift archive. The keyword-search subs cover the same content.

## Setup

```bash
pip install -r requirements.txt

# For clustering (heavy — installs PyTorch ~2 GB):
pip install sentence-transformers hdbscan
```

## Usage

### Quick start (full pipeline)

```bash
# 1. Verify coverage before committing to a long collection:
python run_pipeline.py --coverage-check

# 2. Run everything:
python run_pipeline.py
```

### Stage by stage

```bash
# Collect r/Altadena only, without fetching comments (fast sanity check):
python src/collect.py --subreddit Altadena --no-comments

# Resume from a specific stage (e.g., if collection is done):
python run_pipeline.py --from normalize

# Run only the export (re-generate report after editing keywords.yaml):
python run_pipeline.py --only export
```

### Coverage check

```bash
python run_pipeline.py --coverage-check
```
Probes the first and last page of each subreddit/keyword combination — shows the date range of actual coverage without downloading anything.

## Iterating on keywords

After eyeballing the Markdown report, edit [keywords.yaml](keywords.yaml) and re-run tagging and export:

```bash
python src/tag.py --reset
python src/export.py --report-only
```

## Outputs

| File | Description |
|------|-------------|
| `data/raw/{subreddit}/{post_id}.json` | Raw post + comment bundle per post |
| `data/eaton_pipeline.duckdb` | DuckDB database with all pipeline state |
| `data/processed/eaton_questions.parquet` | Flat export of all question records |
| `data/processed/eaton_report.md` | **Start here** — Markdown report grouped by program area |

## Timing estimates (with comments enabled)

Fetching comment trees dominates. r/Altadena has ~1,400 posts, each needing a comment-tree request:

| Mode | Estimated time |
|------|----------------|
| `--no-comments` all subreddits | ~5 min |
| Full pipeline with comments | 2–4 hours |
| Clustering only (after collect) | 10–30 min (CPU) |

The collector is resumable — kill it and restart; it picks up from the last checkpoint.

## What you lose without an LLM

- ~25–40% noise in "questions" (rhetorical, rants ending in `?`)
- Keyword tagging misses paraphrases ("the place where you get food stamps" won't hit `d_snap`)
- Clusters help but different phrasings of the same intent may land in separate clusters

All fixable by iterating on `keywords.yaml` after eyeballing the report.

## Project layout

```
eaton-question-pipeline/
  requirements.txt
  config.yaml          ← subreddits, dates, API settings
  keywords.yaml        ← edit this to tune tags
  run_pipeline.py      ← orchestrator
  src/
    collect.py         ← Stage 1
    normalize.py       ← Stage 2
    filter.py          ← Stage 3
    cleanup.py         ← Stage 4 (cleanup)
    tag.py             ← Stage 5
    cluster.py         ← Stage 6 (needs sentence-transformers)
    export.py          ← Stage 7
    storage.py         ← DuckDB schema + helpers
  data/
    raw/               ← per-post JSON bundles
    processed/         ← Parquet + report
  notebooks/
    01_explore.ipynb   ← EDA: volume over time, top questions by score
```
