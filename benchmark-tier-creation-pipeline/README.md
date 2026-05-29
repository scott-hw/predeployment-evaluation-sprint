# Benchmark Generation Pipeline

A Python pipeline that generates multi-tier evaluation benchmarks for a disaster-services chatbot, using the Eaton Fire (January 2025, Los Angeles County) as the target context.

---

## Overview

Each **source packet** describes exactly one Q&A (a resident question + official agency answer). The pipeline generates four **language tier** variants per packet:

| Tier | Name | Description | LLM? |
|------|------|-------------|------|
| T0 | FAQ Verbatim | Official question copied verbatim | No |
| T1 | Clean Constituent | Grammatically complete, polite register | Yes |
| T2 | Realistic Messy | Casual but complete; minor typos ok | Yes |
| T3 | High Friction | Fragmented, SMS shorthand | Yes |

Default: 1 item × 4 tiers × N packets = 4N benchmark items.

**Default provider**: Anthropic
- Generator: `claude-sonnet-4-6`
- Validator: `claude-haiku-4-5-20251001` (escalates to Sonnet on low confidence)

**OpenAI provider** (opt-in via `--provider openai`):
- Generator: `gpt-4o`
- Validator: `gpt-4o-mini`

---

## Pipeline Invariants

1. **JSON in, JSON out.** All inputs JSONL/YAML on disk. All outputs JSONL, CSV, JSON manifest.
2. **No external HTTP except the LLM API.** No URL fetching, no scraping.
3. **Source packet is sole factual authority.** Language examples contribute style only.
4. **Different generator and validator models** to prevent shared blind spots.
5. **All LLM calls cacheable.** Cache key = SHA256(model, temperature, max_tokens, prompt).
6. **Determinism where achievable.** `--seed` controls sampling and example selection.

---

## Repository Structure

```
benchmark-pipeline/
  README.md
  pyproject.toml
  .env.example
  config/
    generation_config.yaml        # Main pipeline config
    topic_weights.example.yaml    # Target topic distribution (example)
    language_tiers.yaml           # Tier definitions
    llm_pricing.yaml              # Pricing (must be verified before use)
  data/
    inputs/
      source_packets/             # JSONL source packets
      topic_taxonomy/             # YAML topic taxonomy
      language_examples/          # JSONL style examples
    intermediate/                 # Scratch files
    cache/                        # LLM response cache (disk)
    outputs/                      # Generated benchmark files
  schemas/                        # JSON Schema Draft-07 for all data types
  prompts/
    generate_question_variants.md
    validate_question_variants.md
    language_tier_fragments/      # Per-tier style instructions
  src/benchmark_pipeline/         # Python package
  tests/                          # pytest tests
```

---

## Quickstart

### 1. Install

```bash
cd benchmark-pipeline
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Configure API key

```bash
cp .env.example .env
# For Anthropic (default): set ANTHROPIC_API_KEY=sk-ant-...
# For OpenAI:              set OPENAI_API_KEY=sk-...
```

To use the OpenAI provider, also install the optional dependency:

```bash
pip install -e ".[openai]"
```

### 3. Dry-run (cost estimate only, no API generation calls)

```bash
benchmark-pipeline generate --dry-run
```

### 4. Validate your source packets

```bash
benchmark-pipeline validate-source-packets \
  --source-packets data/inputs/source_packets/source_packets.example.jsonl \
  --taxonomy data/inputs/topic_taxonomy/topics.example.yaml
```

### 5. Run full generation (uses example data)

```bash
benchmark-pipeline generate \
  --source-packets data/inputs/source_packets/source_packets.example.jsonl \
  --taxonomy data/inputs/topic_taxonomy/topics.example.yaml \
  --language-examples data/inputs/language_examples/language_examples.example.jsonl \
  --seed 42 \
  --output-dir data/outputs
```

### 6. Summarize results

```bash
benchmark-pipeline summarize --benchmark data/outputs/benchmark.jsonl
```

---

## CLI Reference

### `benchmark-pipeline generate`

```
Options:
  --source-packets PATH       Source packets JSONL [default: example file]
  --taxonomy PATH             Topic taxonomy YAML [default: example file]
  --language-examples PATH    Language examples JSONL [default: example file]
  --output-dir PATH           Output directory [default: data/outputs]
  --config PATH               Generation config YAML
  --pricing PATH              LLM pricing YAML
  --topic-weights PATH        Topic weights YAML (optional)
  --dry-run                   Estimate cost only; no LLM calls
  --max-source-packets N      Limit number of source packets processed
  --topics T1,T2              Comma-separated topic filter
  --language-tiers T1,T2      Comma-separated tier filter
  --items-per-tier N          Items per tier per packet [default: 1]
  --no-cache                  Disable disk cache
  --seed N                    Random seed for reproducibility
  --exclude-minor-concerns    Reject valid_with_minor_concerns items
  --provider anthropic|openai LLM provider [default: anthropic]
  --generator-model NAME      Override generator model name
  --validator-model NAME      Override validator model name
```

**Provider examples:**

```bash
# Anthropic (default)
benchmark-pipeline generate --source-packets data/inputs/source_packets/source_packets.jsonl ...

# OpenAI with provider defaults (gpt-4o generator, gpt-4o-mini validator)
benchmark-pipeline generate --provider openai ...

# OpenAI with explicit model overrides
benchmark-pipeline generate --provider openai \
  --generator-model gpt-4o \
  --validator-model gpt-4o-mini \
  ...
```

### `benchmark-pipeline validate-source-packets`

Validates JSONL against the Pydantic schema and topic taxonomy. Exits non-zero on any error.

### `benchmark-pipeline validate-benchmark`

Validates output benchmark JSONL against the BenchmarkItem schema.

### `benchmark-pipeline summarize`

Prints tier distribution, topic distribution, and item counts.

---

## Input Formats

### Source Packets (`data/inputs/source_packets/*.jsonl`)

One JSON object per line. See `schemas/source_packet.schema.json` for the full schema.

Key fields:
- `source_packet_id` — unique identifier (e.g. `FEMA_RENTER_001`)
- `official_question` — used verbatim for T0; optional but recommended
- `official_answer` — sole factual authority; only facts here can appear in generated questions
- `required_answer_elements` — chatbot must mention these to score correctly
- `forbidden_claims` — chatbot must not assert these
- `administrative_topic_tags` — must match keys in the active taxonomy

### Topic Taxonomy (`data/inputs/topic_taxonomy/*.yaml`)

```yaml
administrative_topics:
  FEMA:
    label: "FEMA Individual Assistance"
    description: "..."
    subtopics: [...]
```

### Language Examples (`data/inputs/language_examples/*.jsonl`)

Style examples only — no factual content. Fields: `language_example_id`, `source_family`, `language_tier` (T1/T2/T3), `text`, `style_tags`.

---

## Output Formats

All outputs written to `data/outputs/` (configurable):

| File | Description |
|------|-------------|
| `benchmark.jsonl` | One BenchmarkItem per line |
| `benchmark.csv` | Flattened CSV for spreadsheet analysis |
| `manifest.json` | Run metadata: counts, topic distribution, warnings |
| `rejected_candidates.jsonl` | Items that failed validation after all retries |

---

## Configuration

### `config/generation_config.yaml`

Key settings:
- `generator_temperature`: temperature for generation calls (default: 1.0)
- `items_per_tier_per_packet`: items generated per tier (default: 1)
- `max_regeneration_attempts`: retries before rejecting an item (default: 2)
- `exclude_minor_concerns`: whether `valid_with_minor_concerns` → rejected (default: false)
- `rate_limiter.requests_per_minute`: API rate limit (default: 50)

### `config/llm_pricing.yaml`

**Must be verified before running cost estimates.** The file ships with `<verify_current>` placeholders that trigger fallback pricing. Check current rates at:
- Anthropic: https://docs.anthropic.com/en/api/overview
- OpenAI: https://openai.com/api/pricing/

---

## Caching

All LLM responses are cached to `data/cache/` by default. Cache key is:

```
SHA256(model_name + str(temperature) + str(max_tokens) + prompt_text)
```

To bypass the cache for a run: `--no-cache`
To clear the cache programmatically:
```python
from benchmark_pipeline.cache import LLMCache
LLMCache().clear()
```

---

## Running Tests

```bash
pip install pytest pytest-asyncio
pytest tests/ -v
```

Test coverage:
- `test_schema_validation.py` — Pydantic validation, duplicate IDs, tag checks
- `test_generation_contract.py` — T0 verbatim copy, item ID format, JSON parsing
- `test_export_format.py` — JSONL/CSV/manifest write and read-back
- `test_cache.py` — cache get/set/miss/hit/clear
- `test_rate_limiter.py` — semaphore, bucket, backoff
- `test_cost_estimator.py` — pricing lookup, cost calculation

---

## Adding New Source Packets

1. Create a JSONL entry in `data/inputs/source_packets/source_packets.jsonl`
2. Ensure all `administrative_topic_tags` exist in `data/inputs/topic_taxonomy/topics.yaml`
3. Run `benchmark-pipeline validate-source-packets` to check before generation
4. Run `benchmark-pipeline generate --dry-run` to estimate cost

---

## Factual Authority Invariant

The pipeline enforces this invariant at multiple layers:

1. **Prompt**: "The source packet below is the sole factual authority. Do not introduce facts not present in it."
2. **Hard constraints list** in the generation prompt enumerates forbidden fact types.
3. **Validator**: checks `introduces_new_facts` and `unsupported_presuppositions`.
4. **Rejected candidates log**: every failed validation is recorded with the reason.

Researchers reviewing the benchmark should check `rejected_candidates.jsonl` for patterns suggesting the generator is systematically drifting from source packets.

---

## Prompt Caching

When using the **Anthropic provider**, the pipeline uses prompt caching (`cache_control: ephemeral`) on the system prefix of generation calls. This reduces input token costs for repeated calls against the same source packet. The cache hit rate is reported in token usage logs.

The **OpenAI provider** does not use prompt caching; the system prefix is sent as a plain system message on each call.

---

## Security Notes

- API keys are read from `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` environment variables (or `.env` file). Only the key matching the active provider is required.
- Keys are **never logged** — the logging setup in `llm_client.py` explicitly omits them.
- The pipeline makes no outbound HTTP except to the active provider's API endpoint (`api.anthropic.com` or `api.openai.com`).
- Source packet URLs are stored as metadata only and are never fetched.

---

## License

See repository root for license information.
