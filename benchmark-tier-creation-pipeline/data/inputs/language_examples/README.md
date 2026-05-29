# Language Examples

This directory contains JSONL files of language style examples.

## Purpose

Language examples teach the generator LLM the **register and style** of each
language tier (T1, T2, T3). They are NOT factual sources.

The pipeline enforces this via:
1. The generator prompt's hard constraint: "Do not copy any factual content from them."
2. The validator's `copied_from_language_examples` check.

## File format

JSONL — one `LanguageExample` JSON object per line.
Lines starting with `#` are ignored.

## Sourcing policy

- `source_family: "fabricated"` — examples written by pipeline authors for testing.
- Real user-generated content (e.g., Reddit posts) must be de-identified and have
  all factual claims removed or replaced with `[PLACEHOLDER]` before being added here.
- Never include real names, phone numbers, addresses, dollar amounts, or agency names
  from real user-generated content.

## Schema

See `schemas/language_example.schema.json`.
