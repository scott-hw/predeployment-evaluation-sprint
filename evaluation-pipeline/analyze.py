"""
Read generation and judgment stores, write a tidy CSV, and print a summary table.

CSV schema (one row per judgment):
  model, system_prompt, question_id, language, domain, judge, metric, value, data_type

Summary table: mean score per (model, system_prompt, metric).
"""
import csv
import logging
from collections import defaultdict
from pathlib import Path

import yaml

from store import load_generations, load_judgments

logger = logging.getLogger(__name__)


def _resolve(base: Path, rel: str) -> Path:
    return (base / rel).resolve()


def run_analysis(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    base = Path(config_path).parent
    output_dir = _resolve(base, cfg["output_dir"])
    db_path = str(output_dir / "results.db")

    generations = {g["id"]: g for g in load_generations(db_path)}
    all_judgments = load_judgments(db_path)
    good = [j for j in all_judgments if j["error"] is None and j["value"] is not None]
    errors = len(all_judgments) - len(good)

    if errors:
        logger.warning("%d judgment rows have errors and are excluded from analysis", errors)

    # Tidy CSV
    csv_path = output_dir / "results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model", "system_prompt", "question_id", "language_tier", "domain",
            "judge", "metric", "value", "data_type",
        ])
        for j in good:
            gen = generations.get(j["generation_id"])
            if gen is None:
                logger.warning("Judgment %s references unknown generation — skipping", j["id"][:8])
                continue
            writer.writerow([
                gen["model_id"],
                gen["system_prompt_id"],
                gen["question_id"],
                gen["language_tier"],
                gen["domain"],
                j["judge_id"],
                j["metric_id"],
                j["value"],
                j["data_type"],
            ])

    print(f"Wrote {len(good)} rows to {csv_path}")

    # Summary table
    scores: dict[tuple, list[float]] = defaultdict(list)
    for j in good:
        gen = generations.get(j["generation_id"])
        if gen is None:
            continue
        key = (gen["model_id"], gen["system_prompt_id"], j["metric_id"])
        scores[key].append(j["value"])

    if not scores:
        print("No scored judgments to summarize.")
        return

    print("\nMean scores by (model, system_prompt, metric):")
    col = f"{'Model':<12}  {'Prompt':<22}  {'Metric':<18}  {'N':>6}  {'Mean':>7}"
    print(col)
    print("-" * len(col))
    for (model, prompt, metric), vals in sorted(scores.items()):
        mean = sum(vals) / len(vals)
        print(f"{model:<12}  {prompt:<22}  {metric:<18}  {len(vals):>6}  {mean:>7.3f}")
