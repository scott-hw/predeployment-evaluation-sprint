"""
Phase 1 — Generation.

For each (model × system_prompt × benchmark_item), call the model and persist
the response. Re-runs skip already-completed work via content-hash keys.
"""
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import yaml

from costs import CostTracker, estimate_cost, token_cost
from providers import generate as api_generate
from store import generation_exists, init_db, make_key, save_generation

logger = logging.getLogger(__name__)


def _resolve(base: Path, rel: str) -> Path:
    return (base / rel).resolve()


def load_benchmark(benchmark_dir: Path) -> list[dict]:
    items = []
    for path in sorted(benchmark_dir.rglob("*.jsonl")):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                tags = entry.get("tags", {})
                topic_tags = tags.get("administrative_topic_tags", [])
                sa = entry.get("source_answer", {})
                items.append({
                    "id": entry["benchmark_item_id"],
                    "question": entry["question_text"],
                    "language_tier": entry["language_tier"],
                    "domain": topic_tags[0] if topic_tags else "",
                    "required_answer_elements": sa.get("required_answer_elements", []),
                    "forbidden_claims": sa.get("forbidden_claims", []),
                    "violated_forbidden_claim": entry.get("generation_metadata", {}).get("violated_forbidden_claim"),
                    "official_answer": sa.get("official_answer", ""),
                })
    return items


def run_generation(
    config_path: str,
    limit: int | None = None,
    max_cost: float | None = None,
    dry_run: bool = False,
):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    base = Path(config_path).parent
    benchmark_dir = _resolve(base, cfg["benchmark_dir"])
    output_dir = _resolve(base, cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    db_path = str(output_dir / "results.db")

    temperature = cfg.get("temperature", 0)
    concurrency = cfg.get("concurrency", 4)
    max_cost_usd = max_cost if max_cost is not None else cfg["max_cost_usd"]

    init_db(db_path)
    tracker = CostTracker(max_cost_usd, db_path)

    items = load_benchmark(benchmark_dir)
    logger.info("Loaded %d benchmark items from %s", len(items), benchmark_dir)

    system_prompts: dict[str, str] = {}
    for sp in cfg["system_prompts"]:
        with open(_resolve(base, sp["file"])) as f:
            system_prompts[sp["id"]] = f.read()

    # Build full job list
    jobs = []
    for model_cfg in cfg["models"]:
        for sp_cfg in cfg["system_prompts"]:
            for item in items:
                key = make_key(model_cfg["id"], sp_cfg["id"], item["id"], temperature)
                jobs.append({
                    "gen_key": key,
                    "model_cfg": model_cfg,
                    "sp_cfg": sp_cfg,
                    "item": item,
                })

    if cfg.get("max_items"):
        jobs = jobs[: cfg["max_items"]]
    if limit is not None:
        jobs = jobs[:limit]

    pending = [j for j in jobs if not generation_exists(db_path, j["gen_key"])]
    done_count = len(jobs) - len(pending)
    logger.info("%d total jobs — %d done, %d pending", len(jobs), done_count, len(pending))

    if dry_run:
        estimated = sum(estimate_cost(j["model_cfg"]["model"]) for j in pending)
        print(f"Dry run: {len(pending)} pending generation jobs, estimated cost ${estimated:.2f}")
        return

    completed = 0
    stopped = False

    def run_job(job: dict) -> dict | None:
        nonlocal stopped
        if stopped:
            return None

        model_cfg = job["model_cfg"]
        sp_cfg = job["sp_cfg"]
        item = job["item"]
        key = job["gen_key"]

        if tracker.would_exceed(estimate_cost(model_cfg["model"])):
            logger.warning(
                "Cost cap $%.2f reached at $%.4f — stopping",
                tracker.max_cost_usd,
                tracker.running_total,
            )
            stopped = True
            return None

        try:
            result = api_generate(
                provider=model_cfg["provider"],
                model=model_cfg["model"],
                system=system_prompts[sp_cfg["id"]],
                user=item["question"],
                temperature=temperature,
            )
        except Exception as e:
            logger.error("Generation failed [model=%s sp=%s q=%s]: %s",
                         model_cfg["id"], sp_cfg["id"], item["id"], e)
            return None

        cost = token_cost(model_cfg["model"], result["input_tokens"], result["output_tokens"])
        tracker.add(cost)

        record = {
            "id": key,
            "model_id": model_cfg["id"],
            "system_prompt_id": sp_cfg["id"],
            "question_id": item["id"],
            "temperature": temperature,
            "question": item["question"],
            "language_tier": item["language_tier"],
            "domain": item["domain"],
            "required_answer_elements": item["required_answer_elements"],
            "forbidden_claims": item["forbidden_claims"],
            "violated_forbidden_claim": item.get("violated_forbidden_claim"),
            "official_answer": item["official_answer"],
            "response_text": result["text"],
            "input_tokens": result["input_tokens"],
            "output_tokens": result["output_tokens"],
            "cost_usd": cost,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        save_generation(db_path, record)
        logger.debug("Saved generation %s… (model=%s $%.4f)", key[:8], model_cfg["id"], cost)
        return record

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {executor.submit(run_job, j): j for j in pending}
        for future in as_completed(futures):
            if future.result() is not None:
                completed += 1

    remaining = len(pending) - completed
    print(
        f"Generation: {completed} new responses saved, {done_count} already existed, "
        f"{remaining} skipped. Total spend so far: ${tracker.running_total:.4f}"
    )
