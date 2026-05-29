# Source Packets

This directory contains JSONL files where each line is a JSON object conforming to
`schemas/source_packet.schema.json`.

## File format

- **JSONL** (one JSON object per line).
- Lines starting with `#` are comments and are ignored by the loader.
- Each object must have a unique `source_packet_id`.

## Naming convention

`source_packets.jsonl` — production file (not committed; add to `.gitignore`).
`source_packets.example.jsonl` — three example packets for testing and onboarding.

## Required fields

See `schemas/source_packet.schema.json` for the full schema.
The `source_packet_id` must be globally unique across all JSONL files you use in a run.

## Factual authority invariant

The pipeline treats each source packet as the **sole factual authority** for all
benchmark items generated from it. LLM generators may only use facts present in
`official_answer`, `required_answer_elements`, `assumed_facts`, and the other
structured fields — never facts from language examples or general knowledge.
