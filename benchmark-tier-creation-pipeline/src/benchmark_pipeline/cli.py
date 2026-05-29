"""
CLI entry point for the benchmark generation pipeline.

Usage:
    benchmark-pipeline generate [OPTIONS]
    benchmark-pipeline validate-source-packets [OPTIONS]
    benchmark-pipeline validate-benchmark [OPTIONS]
    benchmark-pipeline summarize [OPTIONS]
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Optional

import click

from .utils import setup_logging, load_dotenv_from_project


def _find_project_root() -> Path:
    """
    Locate the benchmark-pipeline project root regardless of install mode.

    Search order:
    1. BENCHMARK_PIPELINE_ROOT env var
    2. Walk up from __file__ looking for a directory that contains config/ and data/
    3. Current working directory
    """
    import os
    root_env = os.environ.get("BENCHMARK_PIPELINE_ROOT")
    if root_env:
        return Path(root_env)

    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "config").exists() and (parent / "data").exists():
            return parent

    return Path.cwd()


_PROJECT_ROOT = _find_project_root()
_CONFIG_DIR = _PROJECT_ROOT / "config"
_DATA_DIR = _PROJECT_ROOT / "data"


def _default(rel: str) -> Path:
    return _PROJECT_ROOT / rel


# ---------------------------------------------------------------------------
# Main group
# ---------------------------------------------------------------------------

@click.group()
@click.option("--log-level", default="INFO", show_default=True, help="Logging level.")
@click.pass_context
def main(ctx: click.Context, log_level: str) -> None:
    """Benchmark generation pipeline for disaster-services chatbot evaluation."""
    setup_logging(log_level)
    load_dotenv_from_project(_PROJECT_ROOT)
    ctx.ensure_object(dict)


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------

@main.command()
@click.option(
    "--source-packets",
    "source_packets_path",
    default=str(_default("data/inputs/source_packets/source_packets.example.jsonl")),
    show_default=True,
    help="Path to source packets JSONL.",
)
@click.option(
    "--taxonomy",
    "taxonomy_path",
    default=str(_default("data/inputs/topic_taxonomy/topics.example.yaml")),
    show_default=True,
    help="Path to topic taxonomy YAML.",
)
@click.option(
    "--language-examples",
    "language_examples_path",
    default=str(_default("data/inputs/language_examples/language_examples.example.jsonl")),
    show_default=True,
    help="Path to language examples JSONL.",
)
@click.option(
    "--output-dir",
    default=str(_default("data/outputs")),
    show_default=True,
    help="Output directory.",
)
@click.option(
    "--config",
    "config_path",
    default=str(_default("config/generation_config.yaml")),
    show_default=True,
    help="Path to generation_config.yaml.",
)
@click.option(
    "--pricing",
    "pricing_path",
    default=str(_default("config/llm_pricing.yaml")),
    show_default=True,
    help="Path to llm_pricing.yaml.",
)
@click.option(
    "--language-tiers",
    "language_tiers_path",
    default=str(_default("config/language_tiers.yaml")),
    show_default=True,
    help="Path to language_tiers.yaml config.",
)
@click.option(
    "--topic-weights",
    "topic_weights_path",
    default=str(_default("config/topic_weights.example.yaml")),
    show_default=True,
    help="Path to topic_weights YAML (optional).",
)
@click.option("--dry-run", is_flag=True, default=False, help="Estimate cost only; no LLM calls.")
@click.option("--max-source-packets", "max_packets", default=None, type=int, help="Limit source packets.")
@click.option("--topics", default=None, help="Comma-separated topic filter (e.g. FEMA,renter_assistance).")
@click.option("--tiers", "tiers_filter", default=None, help="Comma-separated tier filter (e.g. T1_clean_constituent,T2_realistic_messy).")
@click.option("--items-per-tier", "items_per_tier", default=1, show_default=True, type=int, help="Items to generate per tier per packet.")
@click.option("--no-cache", is_flag=True, default=False, help="Disable disk cache.")
@click.option("--seed", default=None, type=int, help="Random seed.")
@click.option("--exclude-minor-concerns", is_flag=True, default=False, help="Exclude valid_with_minor_concerns items.")
@click.option(
    "--provider",
    default="anthropic",
    show_default=True,
    type=click.Choice(["anthropic", "openai"], case_sensitive=False),
    help="LLM provider to use.",
)
@click.option("--generator-model", "generator_model", default=None, help="Override generator model name.")
@click.option("--validator-model", "validator_model", default=None, help="Override validator model name.")
def generate(
    source_packets_path: str,
    taxonomy_path: str,
    language_examples_path: str,
    output_dir: str,
    config_path: str,
    pricing_path: str,
    language_tiers_path: str,
    topic_weights_path: str,
    dry_run: bool,
    max_packets: Optional[int],
    topics: Optional[str],
    tiers_filter: Optional[str],
    items_per_tier: int,
    no_cache: bool,
    seed: Optional[int],
    exclude_minor_concerns: bool,
    provider: str,
    generator_model: Optional[str],
    validator_model: Optional[str],
) -> None:
    """Run the full benchmark generation pipeline."""
    asyncio.run(
        _async_generate(
            source_packets_path=Path(source_packets_path),
            taxonomy_path=Path(taxonomy_path),
            language_examples_path=Path(language_examples_path),
            output_dir=Path(output_dir),
            config_path=Path(config_path),
            pricing_path=Path(pricing_path),
            language_tiers_path=Path(language_tiers_path),
            topic_weights_path=Path(topic_weights_path),
            dry_run=dry_run,
            max_packets=max_packets,
            topics_filter=topics.split(",") if topics else None,
            tiers_filter=tiers_filter.split(",") if tiers_filter else None,
            items_per_tier=items_per_tier,
            use_cache=not no_cache,
            seed=seed,
            exclude_minor_concerns=exclude_minor_concerns,
            provider=provider,
            generator_model=generator_model,
            validator_model=validator_model,
        )
    )


async def _async_generate(
    source_packets_path: Path,
    taxonomy_path: Path,
    language_examples_path: Path,
    output_dir: Path,
    config_path: Path,
    pricing_path: Path,
    language_tiers_path: Path,
    topic_weights_path: Path,
    dry_run: bool,
    max_packets: Optional[int],
    topics_filter: Optional[list[str]],
    tiers_filter: Optional[list[str]],
    items_per_tier: int,
    use_cache: bool,
    seed: Optional[int],
    exclude_minor_concerns: bool,
    provider: str = "anthropic",
    generator_model: Optional[str] = None,
    validator_model: Optional[str] = None,
) -> None:
    import logging
    logger = logging.getLogger(__name__)

    from .io import (
        load_source_packets,
        load_topic_taxonomy,
        load_language_examples,
        load_generation_config,
        load_llm_pricing,
        load_topic_weights,
        load_language_tiers,
    )
    from .cache import LLMCache
    from .rate_limiter import RateLimiter, RateLimiterConfig
    from .llm_client import LLMClient, default_models_for_provider
    from .validator import QuestionValidator
    from .generator import GenerationPipeline, TIER_ORDER
    from .exporter import (
        write_benchmark_jsonl,
        write_benchmark_csv,
        write_manifest,
        write_rejected_candidates,
        build_manifest,
    )
    from .cost_estimator import DryRunEstimator, load_pricing

    # Resolve model names: CLI overrides > provider defaults
    default_gen, default_val = default_models_for_provider(provider)
    resolved_gen_model = generator_model or default_gen
    resolved_val_model = validator_model or default_val

    logger.info("Provider: %s | generator: %s | validator: %s",
                provider, resolved_gen_model, resolved_val_model)

    # Load configs
    config = load_generation_config(config_path)
    pricing_raw = load_llm_pricing(pricing_path)
    topic_weights = load_topic_weights(topic_weights_path)
    # Load language tiers config (overrides defaults in generation_config.yaml if provided)
    language_tiers_config = load_language_tiers(language_tiers_path) if language_tiers_path.exists() else {}
    logger.debug("Language tiers config loaded from %s", language_tiers_path)

    # Load inputs
    taxonomy = load_topic_taxonomy(taxonomy_path)
    packets = load_source_packets(source_packets_path, taxonomy=taxonomy)
    examples = load_language_examples(language_examples_path)

    # Apply filters
    if topics_filter:
        packets = [
            p for p in packets
            if any(t in p.administrative_topic_tags for t in topics_filter)
        ]
        logger.info("After topic filter: %d packets", len(packets))

    if max_packets:
        packets = packets[:max_packets]
        logger.info("After --max-source-packets: %d packets", len(packets))

    active_tiers = tiers_filter or TIER_ORDER
    if tiers_filter:
        # Ensure T0 is included only if explicitly requested
        logger.info("Active tiers: %s", active_tiers)

    if not packets:
        click.echo("No source packets to process after filters.", err=True)
        sys.exit(1)

    # Build clients
    cache = LLMCache() if use_cache else None
    rl_config = RateLimiterConfig(
        requests_per_minute=config.get("rate_limiter", {}).get("requests_per_minute", 50),
        max_concurrent_requests=config.get("rate_limiter", {}).get("max_concurrent_requests", 5),
    )
    rate_limiter = RateLimiter(rl_config)

    client_kwargs = dict(
        provider=provider,
        rate_limiter=rate_limiter,
        cache=cache,
        pricing=pricing_raw,
        dry_run=dry_run,
    )

    gen_client = LLMClient(
        model=resolved_gen_model,
        temperature=config.get("generator_temperature", 1.0),
        max_tokens=config.get("generator_max_tokens", 512),
        **client_kwargs,
    )

    val_client_primary = LLMClient(
        model=resolved_val_model,
        temperature=0.0,
        max_tokens=config.get("validator_max_tokens", 768),
        **client_kwargs,
    )

    val_client_secondary = LLMClient(
        model=resolved_gen_model,
        temperature=0.0,
        max_tokens=config.get("validator_max_tokens", 768),
        **client_kwargs,
    )

    validator = QuestionValidator(
        primary_client=val_client_primary,
        secondary_client=val_client_secondary,
        exclude_minor_concerns=exclude_minor_concerns,
    )

    pipeline = GenerationPipeline(
        generator_client=gen_client,
        validator=validator,
        language_examples=examples,
        config=config,
        seed=seed,
        items_per_tier_per_packet=items_per_tier,
        active_tiers=active_tiers,
        dry_run=dry_run,
    )

    if dry_run:
        click.echo("=== DRY-RUN MODE: no LLM generation calls will be made ===\n")
        # Build prompt list for estimation
        from .generator import _load_prompt_template, _load_tier_fragment, LLM_TIERS
        import json as _json

        prompt_list = []
        tmpl = _load_prompt_template()
        for packet in packets:
            for tier in active_tiers:
                if tier == "T0":
                    continue
                fragment = _load_tier_fragment(tier)
                from .utils import sample_examples as _sample
                tier_examples = _sample(examples, tier, config.get("language_examples_per_tier", 3), seed)
                prompt = (
                    tmpl
                    .replace("{source_packet_json}", _json.dumps(packet.model_dump(), indent=2))
                    .replace("{tier_name}", tier)
                    .replace("{tier_prompt_fragment}", fragment)
                    .replace("{language_examples_json}", _json.dumps([e.model_dump() for e in tier_examples], indent=2))
                )
                prompt_list.append({
                    "prompt_text": prompt,
                    "call_type": f"generation_{tier}",
                    "model": resolved_gen_model,
                    "expected_output_tokens": config.get("generator_max_tokens", 512),
                })
                # Validator call
                prompt_list.append({
                    "prompt_text": f"Validate: {packet.normalized_resident_need}",
                    "call_type": f"validation_{tier}",
                    "model": resolved_val_model,
                    "expected_output_tokens": config.get("validator_max_tokens", 768),
                })

        estimator = DryRunEstimator(
            llm_client_generator=gen_client,
            llm_client_validator=val_client_primary,
            pricing=pricing_raw,
            requests_per_minute=rl_config.requests_per_minute,
        )
        report = await estimator.estimate(prompt_list)
        report.print_summary()
        click.echo(
            f"Would generate up to {len(packets) * len(active_tiers) * items_per_tier} items "
            f"({len(packets)} packets × {len(active_tiers)} tiers × {items_per_tier} items/tier)."
        )
        return

    # Run pipeline
    items, rejected = await pipeline.run(packets)

    # Write outputs
    out_path = Path(output_dir)
    write_benchmark_jsonl(items, out_path)
    write_benchmark_csv(items, out_path)
    write_rejected_candidates(rejected, out_path)

    manifest = build_manifest(
        items=items,
        rejected=rejected,
        config={
            "seed": seed,
            "items_per_tier": items_per_tier,
            "active_tiers": active_tiers,
            "source_packets_path": str(source_packets_path),
            "max_packets": max_packets,
        },
        topic_weights=topic_weights,
    )
    write_manifest(manifest, out_path)

    click.echo(f"\nGeneration complete.")
    click.echo(f"  Accepted items : {len(items)}")
    click.echo(f"  Rejected items : {len(rejected)}")
    click.echo(f"  Output dir     : {out_path}")

    if manifest.get("warnings"):
        click.echo("\nManifest warnings:")
        for w in manifest["warnings"]:
            click.echo(f"  ! {w}")


# ---------------------------------------------------------------------------
# validate-source-packets
# ---------------------------------------------------------------------------

@main.command("validate-source-packets")
@click.option(
    "--source-packets",
    "source_packets_path",
    default=str(_default("data/inputs/source_packets/source_packets.example.jsonl")),
    show_default=True,
)
@click.option(
    "--taxonomy",
    "taxonomy_path",
    default=str(_default("data/inputs/topic_taxonomy/topics.example.yaml")),
    show_default=True,
)
def validate_source_packets(source_packets_path: str, taxonomy_path: str) -> None:
    """Validate source packets JSONL against schema and taxonomy."""
    from .io import load_source_packets, load_topic_taxonomy
    import jsonschema, json

    schema_path = _PROJECT_ROOT / "schemas" / "source_packet.schema.json"

    try:
        taxonomy = load_topic_taxonomy(Path(taxonomy_path))
        packets = load_source_packets(Path(source_packets_path), taxonomy=taxonomy)
    except Exception as exc:
        click.echo(f"ERROR: {exc}", err=True)
        sys.exit(1)

    # Additional JSON Schema validation
    if schema_path.exists():
        with schema_path.open() as fh:
            schema = json.load(fh)
        errors = 0
        for p in packets:
            try:
                jsonschema.validate(p.model_dump(), schema)
            except jsonschema.ValidationError as exc:
                click.echo(f"Schema error in {p.source_packet_id}: {exc.message}", err=True)
                errors += 1
        if errors:
            sys.exit(1)

    click.echo(f"OK — {len(packets)} source packets valid.")


# ---------------------------------------------------------------------------
# validate-benchmark
# ---------------------------------------------------------------------------

@main.command("validate-benchmark")
@click.option(
    "--benchmark",
    "benchmark_path",
    default=str(_default("data/outputs/benchmark.jsonl")),
    show_default=True,
)
def validate_benchmark(benchmark_path: str) -> None:
    """Validate output benchmark JSONL."""
    import json as _json
    from .schemas import BenchmarkItem

    path = Path(benchmark_path)
    if not path.exists():
        click.echo(f"ERROR: File not found: {path}", err=True)
        sys.exit(1)

    items = []
    errors = 0
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = _json.loads(line)
                item = BenchmarkItem(**data)
                items.append(item)
            except Exception as exc:
                click.echo(f"Line {lineno}: {exc}", err=True)
                errors += 1

    if errors:
        click.echo(f"\nFAIL — {errors} validation error(s).", err=True)
        sys.exit(1)

    click.echo(f"OK — {len(items)} benchmark items valid.")


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------

@main.command()
@click.option(
    "--benchmark",
    "benchmark_path",
    default=str(_default("data/outputs/benchmark.jsonl")),
    show_default=True,
)
def summarize(benchmark_path: str) -> None:
    """Print summary statistics for a benchmark JSONL file."""
    import json as _json
    from .schemas import BenchmarkItem
    from .utils import compute_topic_distribution

    path = Path(benchmark_path)
    if not path.exists():
        click.echo(f"ERROR: File not found: {path}", err=True)
        sys.exit(1)

    items = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(BenchmarkItem(**_json.loads(line)))
            except Exception:
                pass

    if not items:
        click.echo("No valid benchmark items found.")
        return

    tier_counts: dict[str, int] = {}
    packet_ids: set[str] = set()
    for item in items:
        tier_counts[item.language_tier] = tier_counts.get(item.language_tier, 0) + 1
        packet_ids.add(item.source_packet_id)

    topic_dist = compute_topic_distribution(items)

    click.echo(f"\n{'='*50}")
    click.echo(f"Benchmark Summary: {path.name}")
    click.echo(f"{'='*50}")
    click.echo(f"Total items        : {len(items)}")
    click.echo(f"Source packets     : {len(packet_ids)}")
    click.echo(f"\nTier distribution:")
    for tier in sorted(tier_counts):
        click.echo(f"  {tier:<6} : {tier_counts[tier]}")
    click.echo(f"\nTopic distribution:")
    for topic, count in sorted(topic_dist.items(), key=lambda x: -x[1]):
        click.echo(f"  {topic:<35} : {count}")
    click.echo()
