# Evaluation Pipeline

Runs the chatbot evaluation described in Hwang, Jiang & Canessa (2026). For each benchmark item × model × system-prompt combination, the pipeline generates a response then scores it with two LLM judges.

## Layout

```
evaluation-pipeline/
  config.yaml          # single config file — edit models, prompts, judges, metrics here
  .env.example         # API key template
  requirements.txt
  benchmark/           # benchmark JSONL files (benchmark.jsonl already included)
  prompts/
    sys_minimal.txt              # no-prompt baseline system message
    sys_variation.txt            # government-persona system prompt
    judge_required_elements.txt  # scores proportion of required elements present
    judge_forbidden_elements.txt # scores proportion of forbidden claims present
    judge_identifies_false_claim.txt  # scores whether false assumption was caught (T3 only)
  providers.py         # Anthropic / OpenAI / Google adapter
  generate.py          # Phase 1: generate model responses
  judge.py             # Phase 2: score responses with LLM judges
  store.py             # SQLite persistence
  costs.py             # token counting and cost cap
  analyze.py           # export tidy CSV and print summary table
  run.py               # CLI orchestrator
  runs/                # output directory (created automatically)
    results.db         # SQLite database
    results.csv        # tidy CSV, one row per judgment
```

## Setup

**1. Python 3.11+**

```bash
cd evaluation-pipeline
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**2. API keys**

```bash
cp .env.example .env
# Fill in ANTHROPIC_API_KEY, OPENAI_API_KEY, and GOOGLE_API_KEY
```

Or export directly:

```bash
export ANTHROPIC_API_KEY=...
export OPENAI_API_KEY=...
export GOOGLE_API_KEY=...    # only needed if running Gemini as a generation model
```

## Running

All commands run from inside `evaluation-pipeline/`.

```bash
# Estimate job count and cost without calling any APIs
python run.py all --dry-run

# Run both phases end-to-end
python run.py all

# Run phases separately
python run.py generate
python run.py judge

# Pilot: cap at 20 jobs and $2
python run.py all --limit 20 --max-cost 2.0
```

Runs are **resumable** — completed jobs are skipped. Interrupt and restart at any time.

## Configuration

Edit `config.yaml` to change models, system prompts, judges, metrics, cost cap, and concurrency. The file is the single source of truth for a run; no code changes are needed to add a model or swap a judge.

### Models

The paper evaluated three generation models, each run with and without the government system prompt:

| `id` | Provider | Model |
|------|----------|-------|
| `claude` | anthropic | claude-sonnet-4-6 |
| `gpt` | openai | gpt-4o |
| `gemini` | google | gemini-2.5-pro |

### System prompts

| `id` | File | Description |
|------|------|-------------|
| `none` | `sys_minimal.txt` | Bare "helpful assistant" prompt |
| `with_prompt` | `sys_variation.txt` | Government-persona prompt mimicking the Digital Assistant's system prompt |

### Metrics

| `id` | Judges | T3 only? |
|------|--------|----------|
| `required_elements` | claude, gpt | No |
| `forbidden_elements` | claude, gpt | No |
| `identifies_false_claim` | claude, gpt | Yes |

### Cost cap

`max_cost_usd` in `config.yaml` sets a hard spend limit. The pipeline stops gracefully when the next call would exceed it. Update `costs.py` → `PRICE_PER_1M` before running to reflect current provider pricing.

## Benchmark format

`benchmark/benchmark.jsonl` contains one JSON object per line. Each object has:

- `benchmark_item_id` — unique ID (e.g. `DEBRIS_PHASES_001__T0__0001`)
- `question_text` — the question sent to the model
- `language_tier` — `T0`, `T1`, `T2`, or `T3`
- `source_answer.required_answer_elements` — list used by the `required_elements` judge
- `source_answer.forbidden_claims` — list used by the `forbidden_elements` judge
- `generation_metadata.violated_forbidden_claim` — the embedded false claim for T3 items

## Output

- `runs/results.db` — SQLite database with all generations and judgments
- `runs/results.csv` — tidy CSV with columns: `model`, `system_prompt`, `question_id`, `language_tier`, `domain`, `judge`, `metric`, `value`, `data_type`

The CSV is designed for direct import into R or Python for the mixed-effects model analysis described in the paper.
