# Narration simplification: completed design record

Date: October 19, 2026.

This record summarizes the implemented narration design. The supported data
flow, command contract, limits, validation boundaries, and offline reader are
documented in the [architecture guide](ARCHITECTURE.md).

## Design decisions

- Keep compilation, original patches, function comparisons, and source
  availability evidence separate from generated prose.
- Store each source buffer once in a compact, revision-bound packet. Resolve
  direct citations to original source lines. Account for every changed region
  and preserve complete supplied diffs.
- Use one writing request for a small change. For a larger change, use bounded
  parallel writing and at most one editorial request. Compiler groups and
  dependencies guide planning; the story may organize concepts in a different
  order.
- Keep a required source-grounded **Before the code** preamble. Use clear
  simple English, define necessary jargon at first use, retain exact technical
  terms, and mark code names and paths with backticks. Structural checks do
  not prove that prose is true.
- Bound generation at 12 provider attempts, four concurrent calls, a 240-second
  deadline, and 1 MiB of response and saved annotation data. Persist attempt
  state separately from optional payload caching.
- Save a flat `diffstory.report.v2` with the packet, story, and bounded
  generation metadata. Export optional annotations without duplicating source.
  Keep public rendering, report loading, and annotation application as
  independent validation boundaries. Reuse a validation result only in the
  same call that produced or checked that report.
- Keep output HTML self-contained and offline. Do not execute repository code
  or claim that tests ran.

## Completion boundary

The current CLI supports `--narrate` for Git, GitHub, and saved snapshots, with
OpenAI Responses or Codex CLI providers and explicit source-transfer consent.
Manual compiler annotations and evidence export remain available without a
provider call. The removed generation architectures and stored-report formats
are not supported by the current reader.

This design record describes structural behavior. It does not establish
provider latency, generated-prose quality, semantic correctness, or test
execution. See the architecture guide for current command examples and
contract details.
