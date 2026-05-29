"""
run_pipeline.py — Orchestrator for the Eaton Fire question mining pipeline.

Runs all stages in sequence. Each stage is idempotent — re-running skips
already-processed records.

Usage:
    python run_pipeline.py                  # run all stages
    python run_pipeline.py --from normalize # start from a specific stage
    python run_pipeline.py --only collect   # run only one stage
    python run_pipeline.py --coverage-check # just verify Arctic Shift coverage
"""

import argparse
import subprocess
import sys
import pathlib
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

ROOT = pathlib.Path(__file__).parent

STAGES = [
    ("collect",    "src/collect.py"),
    ("normalize",  "src/normalize.py"),
    ("filter",     "src/filter.py"),
    ("cleanup",    "src/cleanup.py"),
    ("tag",        "src/tag.py"),
    ("cluster",    "src/cluster.py"),
    ("export",     "src/export.py"),
]

STAGE_NAMES = [s[0] for s in STAGES]


def run_stage(name: str, script: str, extra_args: list[str] = None) -> bool:
    cmd = [sys.executable, str(ROOT / script)] + (extra_args or [])
    log.info("=== Stage: %s ===", name.upper())
    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0:
        log.error("Stage '%s' failed (exit %d)", name, result.returncode)
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Run the Eaton fire question pipeline")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--from", dest="from_stage", choices=STAGE_NAMES,
                        help="Start from this stage (skip earlier stages)")
    parser.add_argument("--only", choices=STAGE_NAMES,
                        help="Run only this stage")
    parser.add_argument("--coverage-check", action="store_true",
                        help="Run collect.py --coverage-check and exit")
    parser.add_argument("--no-comments", action="store_true",
                        help="Pass --no-comments to collect stage (faster)")
    parser.add_argument("--subreddit", help="Collect/normalize only this subreddit")
    args = parser.parse_args()

    if args.coverage_check:
        run_stage("collect (coverage check)", "src/collect.py",
                  ["--config", args.config, "--coverage-check"])
        return

    active_stages = STAGES
    if args.only:
        active_stages = [(n, s) for n, s in STAGES if n == args.only]
    elif args.from_stage:
        idx = STAGE_NAMES.index(args.from_stage)
        active_stages = STAGES[idx:]

    config_args = ["--config", args.config]

    for name, script in active_stages:
        extra = list(config_args)
        if name == "collect":
            if args.no_comments:
                extra.append("--no-comments")
            if args.subreddit:
                extra += ["--subreddit", args.subreddit]
        elif name == "normalize" and args.subreddit:
            extra += ["--subreddit", args.subreddit]

        ok = run_stage(name, script, extra)
        if not ok:
            log.error("Pipeline aborted at stage '%s'.", name)
            sys.exit(1)

    log.info("Pipeline complete.")


if __name__ == "__main__":
    main()
