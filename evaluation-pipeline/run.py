#!/usr/bin/env python3
"""
CLI orchestrator for the multilingual LLM evaluation pipeline.

Usage:
  python run.py generate [--config PATH] [--limit N] [--max-cost X] [--dry-run]
  python run.py judge    [--config PATH] [--limit N] [--max-cost X] [--dry-run]
  python run.py all      [--config PATH] [--limit N] [--max-cost X] [--dry-run]
"""
import argparse
import logging
import os
import sys
from pathlib import Path


def _load_dotenv(config_path: str):
    env_path = Path(config_path).parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())


def main():
    parser = argparse.ArgumentParser(
        description="Multilingual LLM Evaluation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "phase",
        choices=["generate", "judge", "all"],
        help="Which phase(s) to run",
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml (default: config.yaml)")
    parser.add_argument("--limit", type=int, default=None, metavar="N", help="Cap job count (pilot mode)")
    parser.add_argument("--max-cost", type=float, default=None, metavar="X", help="Override max_cost_usd")
    parser.add_argument("--dry-run", action="store_true", help="Estimate job count and cost without calling APIs")
    args = parser.parse_args()

    config_path = str(Path(args.config).resolve())

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    _load_dotenv(config_path)

    # Ensure pipeline modules are importable regardless of working directory
    pipeline_dir = str(Path(config_path).parent)
    if pipeline_dir not in sys.path:
        sys.path.insert(0, pipeline_dir)

    kwargs = dict(config_path=config_path, limit=args.limit, max_cost=args.max_cost, dry_run=args.dry_run)

    if args.phase in ("generate", "all"):
        from generate import run_generation
        run_generation(**kwargs)

    if args.phase in ("judge", "all"):
        from judge import run_judging
        run_judging(**kwargs)

    if args.phase == "all" and not args.dry_run:
        from analyze import run_analysis
        run_analysis(config_path)


if __name__ == "__main__":
    main()
