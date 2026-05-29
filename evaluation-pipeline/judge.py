"""
Phase 2 — Judging.

For each (stored generation × metric × assigned judge), call the judge model
and persist a score. Re-runs skip already-completed judgments.

Judge prompts never reveal which model produced a response (anonymized judging).
Each prompt instructs the judge to return {"reasoning": "...", "value": ...};
reasoning-before-value yields better-calibrated scores.
"""
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import yaml

from costs import CostTracker, estimate_cost, token_cost
from providers import generate as api_generate
from store import (
    init_db,
    judgment_exists,
    load_generations,
    make_key,
    save_judgment,
)

logger = logging.getLogger(__name__)

_FENCE_RE = re.compile(r"^```[a-z]*\n?|\n?```$", re.MULTILINE)


def _resolve(base: Path, rel: str) -> Path:
    return (base / rel).resolve()


def _parse_response(text: str, metric_cfg: dict) -> tuple[float | None, str | None, str | None]:
    """
    Parse and validate a judge response.
    Returns (value, comment, error_message).
    value is None on any failure.
    """
    cleaned = _FENCE_RE.sub("", text).strip()
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError as e:
        return None, None, f"JSON parse error: {e} — raw: {cleaned[:200]!r}"

    if "value" not in obj:
        return None, None, f"Missing 'value' key in: {cleaned[:200]!r}"

    comment = obj.get("reasoning", "")
    raw = obj["value"]
    out_type = metric_cfg["output_type"]

    if out_type == "boolean":
        if isinstance(raw, bool):
            return float(raw), comment, None
        if isinstance(raw, str) and raw.lower() in ("true", "false"):
            return float(raw.lower() == "true"), comment, None
        return None, None, f"Expected boolean, got {raw!r}"

    if out_type == "numeric":
        try:
            val = float(raw)
        except (TypeError, ValueError):
            return None, None, f"Expected numeric, got {raw!r}"
        lo, hi = metric_cfg.get("range", [0, 5])
        if not (lo <= val <= hi):
            return None, None, f"Value {val} out of range [{lo}, {hi}]"
        return val, comment, None

    return None, None, f"Unknown output_type: {out_type!r}"


def run_judging(
    config_path: str,
    limit: int | None = None,
    max_cost: float | None = None,
    dry_run: bool = False,
):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    base = Path(config_path).parent
    output_dir = _resolve(base, cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    db_path = str(output_dir / "results.db")

    concurrency = cfg.get("concurrency", 4)
    max_cost_usd = max_cost if max_cost is not None else cfg["max_cost_usd"]

    init_db(db_path)
    tracker = CostTracker(max_cost_usd, db_path)

    judges: dict[str, dict] = {j["id"]: j for j in cfg["judges"]}
    metrics: list[dict] = cfg["metrics"]

    prompt_templates: dict[str, str] = {}
    for m in metrics:
        with open(_resolve(base, m["prompt"])) as f:
            prompt_templates[m["id"]] = f.read()

    generations = load_generations(db_path)
    logger.info("Loaded %d generation records", len(generations))

    # Build full job list
    jobs = []
    for gen in generations:
        for metric in metrics:
            if metric.get("requires_expected_answer"):
                continue
            required_tier = metric.get("requires_tier")
            if required_tier and gen.get("language_tier") != required_tier:
                continue
            for judge_id in metric["judges"]:
                judge_cfg = judges[judge_id]
                jdg_key = make_key(gen["id"], judge_id, metric["id"])
                jobs.append({
                    "jdg_key": jdg_key,
                    "gen": gen,
                    "metric": metric,
                    "judge_cfg": judge_cfg,
                    "template": prompt_templates[metric["id"]],
                })

    if limit is not None:
        jobs = jobs[:limit]

    pending = [j for j in jobs if not judgment_exists(db_path, j["jdg_key"])]
    done_count = len(jobs) - len(pending)
    logger.info("%d total judgment jobs — %d done, %d pending", len(jobs), done_count, len(pending))

    if dry_run:
        estimated = sum(estimate_cost(j["judge_cfg"]["model"]) for j in pending)
        print(f"Dry run: {len(pending)} pending judgment jobs, estimated cost ${estimated:.2f}")
        return

    completed = 0
    stopped = False

    def run_job(job: dict) -> dict | None:
        nonlocal stopped
        if stopped:
            return None

        gen = job["gen"]
        metric = job["metric"]
        judge_cfg = job["judge_cfg"]
        template = job["template"]
        jdg_key = job["jdg_key"]

        if tracker.would_exceed(estimate_cost(judge_cfg["model"])):
            logger.warning(
                "Cost cap $%.2f reached at $%.4f — stopping",
                tracker.max_cost_usd,
                tracker.running_total,
            )
            stopped = True
            return None

        # Fill prompt — no model identity included (anonymized judging)
        required = "\n".join(f"- {e}" for e in gen["required_answer_elements"])
        forbidden = "\n".join(f"- {e}" for e in gen["forbidden_claims"])
        fill = {
            "question": gen["question"],
            "response": gen["response_text"],
            "required_answer_elements": required,
            "forbidden_claims": forbidden,
            "violated_forbidden_claim": gen.get("violated_forbidden_claim") or "",
        }
        prompt = template.format(**fill)

        data_type = "NUMERIC"

        # Attempt the call; retry once on parse/validation failure
        in_tok = out_tok = 0
        cost = 0.0
        final_error: str | None = None

        for attempt in range(2):
            try:
                result = api_generate(
                    provider=judge_cfg["provider"],
                    model=judge_cfg["model"],
                    system="You are an evaluation assistant. Follow the instructions exactly and return only the JSON object requested.",
                    user=prompt,
                    temperature=0,
                )
                in_tok = result["input_tokens"]
                out_tok = result["output_tokens"]
                cost = token_cost(judge_cfg["model"], in_tok, out_tok)

                value, comment, parse_error = _parse_response(result["text"], metric)
                if parse_error is None:
                    tracker.add(cost)
                    record = {
                        "id": jdg_key,
                        "generation_id": gen["id"],
                        "judge_id": judge_cfg["id"],
                        "metric_id": metric["id"],
                        "name": metric["id"],
                        "value": value,
                        "data_type": data_type,
                        "comment": comment,
                        "input_tokens": in_tok,
                        "output_tokens": out_tok,
                        "cost_usd": cost,
                        "error": None,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    }
                    save_judgment(db_path, record)
                    logger.debug(
                        "Saved judgment %s… metric=%s value=%s",
                        jdg_key[:8], metric["id"], value,
                    )
                    return record
                else:
                    final_error = parse_error
                    logger.warning("Attempt %d parse error [%s]: %s", attempt + 1, jdg_key[:8], parse_error)

            except Exception as e:
                final_error = str(e)
                logger.error("Judge call failed attempt %d [%s]: %s", attempt + 1, jdg_key[:8], e)

        # Both attempts failed — record an error row so the run can continue
        tracker.add(cost)
        record = {
            "id": jdg_key,
            "generation_id": gen["id"],
            "judge_id": judge_cfg["id"],
            "metric_id": metric["id"],
            "name": metric["id"],
            "value": None,
            "data_type": data_type,
            "comment": None,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "cost_usd": cost,
            "error": final_error,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        save_judgment(db_path, record)
        return None

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {executor.submit(run_job, j): j for j in pending}
        for future in as_completed(futures):
            if future.result() is not None:
                completed += 1

    skipped = len(pending) - completed
    print(
        f"Judging: {completed} scores saved, {done_count} already existed, "
        f"{skipped} skipped/failed. Total spend so far: ${tracker.running_total:.4f}"
    )
