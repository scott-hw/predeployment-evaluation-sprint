# Topic Taxonomy

This directory contains the administrative topic taxonomy used to tag and validate
source packets and benchmark items.

## File format

`topics.yaml` — production taxonomy (commit to version control).
`topics.example.yaml` — example taxonomy included in this repository.

## Schema

See `schemas/topic_taxonomy.schema.json`.

## Usage

- Every `administrative_topic_tags` value in source packets must appear as a key
  in `administrative_topics` in the active taxonomy file.
- The pipeline fails loudly on unknown tags.
- Add new topics here before adding them to source packets.

## Eaton Fire topic coverage

The example taxonomy covers the main service categories relevant to the
January 2025 Eaton Fire disaster response, based on the types of questions
survivors and renters typically have after a major wildfire in Los Angeles County.
