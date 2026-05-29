# Pre-Deployment Diagnostic Evaluation for Public Sector Chatbots

Replication code for *Pre-Deployment Diagnostic Evaluation for Public Sector Chatbots*. The paper evaluates a California state agency's Claude-based Digital Assistant against commercial models on post-disaster resource navigation questions derived from the January 2025 Eaton Fire.

**Note**: Exact experiment is not replicable, as the system prompt used is not public. Uploading a separate system prompt is possible to replicate a similar but non-identical procedure.

## Repository structure

```
reddit-scraper/                   # Pipeline 1 — mine constituent questions from Reddit
benchmark-tier-creation-pipeline/ # Pipeline 2 — generate the four-tier benchmark
evaluation-pipeline/              # Pipeline 3 — run models and score responses
```

The three pipelines run in sequence. Their outputs feed into each other as described below.

## Data flow

```
reddit-scraper/
  → data/processed/eaton_report.md      (qualitative input: topics & tone)

benchmark-tier-creation-pipeline/
  data/inputs/source_packets/
  data/inputs/language_examples/        (tone examples informed by Reddit output)
  → data/outputs/benchmark.jsonl

evaluation-pipeline/
  benchmark/benchmark.jsonl             (copy of benchmark-tier-creation output)
  → runs/results.db
  → runs/results.csv
```

## Pipeline 1 — Reddit scraper

Surfaces real questions Californians asked on Reddit after the Eaton fire. Used qualitatively to validate benchmark topic coverage and to provide few-shot tone examples for Tier 2 question generation. **Does not require any API keys.**

See [`reddit-scraper/README.md`](reddit-scraper/README.md) for setup and usage.

## Pipeline 2 — Benchmark generation

Takes 90 official FAQ source packets and generates four language-register variants of each question (360 items total):

| Tier | Style | LLM-generated? |
|------|-------|----------------|
| T0 | FAQ verbatim | No |
| T1 | Clean constituent | Yes |
| T2 | Realistic / messy | Yes |
| T3 | False-assumption embedded | Yes |

The generated `benchmark.jsonl` is already included in `evaluation-pipeline/benchmark/` and does not need to be regenerated to replicate the evaluation. Run this pipeline only if you want to modify the source packets or regeneration parameters.

See [`benchmark-tier-creation-pipeline/README.md`](benchmark-tier-creation-pipeline/README.md) for setup and usage.

## Pipeline 3 — Evaluation

Sends every benchmark question to each model × system-prompt combination, then scores responses with two LLM judges (Claude Sonnet 4.6 and GPT-4o) on three metrics:

| Metric | Description |
|--------|-------------|
| `required_elements` | Proportion of required answer elements present (0–1) |
| `forbidden_elements` | Proportion of forbidden claims present — lower is better (0–1) |
| `identifies_false_claim` | Whether the model corrected the embedded false assumption (T3 only, 0/1) |

**This is the pipeline to run to replicate the paper's results.**

See [`evaluation-pipeline/README.md`](evaluation-pipeline/README.md) for setup and usage.

```
