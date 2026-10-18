# Narration simplification plan

Date: October 15, 2026.

Keep the source analysis and offline reader. Replace the narration engine with
a bounded, two-stage pipeline, then remove the superseded machinery. The goal
is a useful, source-linked explanation of a complex changeset within a few
minutes, with a small and predictable number of model requests.

This is an implementation plan. The request counts, runtime limits, and size
limits below are proposed defaults. Live measurements must establish the
completion times that the supported providers can achieve.

**Evidence for the change**

The local review found these results:

- The bundled demo contains 7,449 source bytes. Its default v2 preflight plans
  13 first-stage requests and reserves 76,526 output tokens. Later narration
  stages add requests.
- Changing only the fixed partition count from 16 to 1 reduces the demo to
  two first-stage requests, but still reserves 73,006 output tokens. These
  numbers are output reservations, not measured provider usage.
- The existing local 24-file PR fixture contains 150 changes, 31 groups, and
  1,709,147 source bytes. Compilation took 6.8 seconds and produced 5,587,128
  bytes of compact report JSON. Narration preflight did not finish during the
  bounded probe; it was interrupted while repeatedly serializing candidate
  batches.
- The compiler test suite passed all 80 tests. The review made no live model
  requests, so it did not establish provider latency or narration quality.

The [request planner](../diffstory/narrative_plan.py) reserves output in
proportion to source bytes, partitions requests into fixed hash buckets, and
repeatedly serializes growing candidate batches. The evidence builder also
creates narration work for unchanged context. The
[v2 coordinator](../diffstory/narrative_v2.py) then runs detail, planning,
section, and document stages. Retry limits apply to individual batch lineages;
there is no run-wide call or elapsed-time ceiling.

These findings support replacing the narration path. A batching adjustment
alone does not address the excessive output contract or the later stages.

**Implementation sequence**

1. **Define the output and performance contract.**

   A report must explain the main change, walk through its important
   mechanisms, and place each explanation beside supporting code. Related
   changes can share an explanation. Every changed region must remain
   accessible and accounted for. A source fragment does not require its own
   paragraph.

   Preserve the required **Before the code** preamble. It must introduce the
   project or relevant subsystem, explain its purpose and basic concepts,
   then orient the reader to the change. Keep it separate from the short
   lead and implementation tour. Preserve paragraphs, conceptual sketches,
   and the separate 4,000-character prose and sketch allowances.

   All writing and editorial stages must use roughly 80% ASD-STE100-style
   English, with natural rhythm and exact technical terms. Explain necessary
   jargon and expand acronyms before use. Assume the reader can read code
   but does not already know the project. Performance simplification must
   preserve these narration requirements.

   Establish three benchmark fixtures: the bundled demo, the existing local
   24-file PR, and a larger synthetic changeset. Keep private fixture source
   and generated reports outside the public file set.

   Measure compilation, request preparation, provider time, validation, and
   rendering separately. Record total requests, provider-reported tokens,
   peak memory, and disk bytes for reports and caches. Bind results to the
   tested code and fixture revisions. Synthetic providers can establish local
   overhead and structural correctness; live runs must establish narration
   quality and provider latency.

   Acceptance: a small, reproducible baseline command reports these metrics
   without sending source unless live generation is explicitly selected.

2. **Build one compact changeset packet.**

   Retain pinned revisions, diff hunks, symbol matching, source coordinates,
   and dependency hints from the existing compiler. Send changed code with
   enough surrounding context to explain it. Include complete definitions
   when needed.

   Store source once and reference it by IDs and ranges. Preserve warnings
   for unavailable or incomplete source. Generated files and repetitive
   mechanical edits can receive a collective explanation, with their
   complete supplied diffs still available in the report. Do not silently
   drop changed regions to fit a budget.

   Use incremental size accounting during preparation. Serialize each
   completed batch once, then check the complete provider request against
   its capacity. Avoid repeated serialization of growing candidate batches.

   Acceptance: packet tests show that all changed regions are represented,
   citations resolve to the correct revision and range, and unchanged
   context does not create separate narration jobs.

3. **Replace the four-stage narration hierarchy.**

   For a small changeset, use one request to write the complete walkthrough.

   For a large changeset, use a few parallel requests to write source-linked
   passages for coherent parts of the change. Use one final editorial
   request to edit the complete source-free story, choose the reading order,
   and write its opening, connections, and closing. Give that request the
   complete validated writer prose, orientations, section titles and
   summaries, normalized passage IDs, citation counts, relevant static test
   IDs, and author context. Do not include source buffers or duplicate exact
   citation lists. The editor may revise and merge prose. The host must require
   an exact partition of the original passage IDs, then retain the union of
   their change IDs, region IDs, and exact citations. Remap opening and review
   references through merged passage IDs. Reject excessive citations or
   annotation bytes; never trim evidence to fit.

   Bound each complete source-free writer projection at 24,576 provider-wire
   bytes, including prose and metadata. Before any writer receives source,
   prepare the empty editor request once and reserve its bytes plus one maximum
   projection per writer and list separators. Use actual provider serialization
   and escaping for this bound. Check the completed editor request against both
   provider capacity and the preflight bound.

   New walkthrough v3 stories require a concrete before/after summary, an
   optional evidence-backed impact, a short route to mechanism/tests/
   compatibility passages, exact author-description excerpts for reported
   checks, and specific linked review checks. The host owns the analyzer's
   verified-run count; generated prose cannot supply it. Archived v1/v2 story
   shapes remain readable.

   Let the model organize the story around concepts. Compiler groups and
   dependencies supply evidence and ordering hints. Eliminate separate
   model calls for planning cards, every section, and repeated summary levels.

   Acceptance: the demo produces a complete walkthrough in one successful
   request. The complex fixture uses bounded parallel writing followed by
   one editorial request. Both preserve coverage and source links through
   the existing reader.

4. **Use one small runtime with explicit limits.**

   Start with these defaults:

   | Control | Proposed default |
   |---|---:|
   | Small changeset | 1 request |
   | Complex changeset | 4–8 writing requests + 1 editorial request |
   | Maximum requests, including retries | 12 |
   | Concurrent requests | 4 |
   | Generation deadline | 4 minutes |
   | Model-produced annotation data | At most 1 MiB |

   The generation deadline covers local narration preparation, provider
   work, and assembly. Measure ingestion and final output separately as
   well as in the end-to-end runtime.

   Allocate retries from the same run budget. Pass the remaining deadline
   to each provider call. Reject an oversized plan before sending source,
   with a clear explanation of the exceeded limit. A limit failure must not
   produce a report labeled as complete.

   Keep a simple request-keyed cache and a small run record for completed
   results, attempts, and usage. Bind cache entries to the exact evidence,
   provider, model, prompt, and schema. Resuming the same run must retain
   spent attempts. Avoid recursive retry and split trees.

   Acceptance: tests exercise concurrent budget reservation, deadline
   enforcement, failed calls, exhausted retries, cache invalidation, and
   resumption. No failure path can silently exceed the run budget.

5. **Flatten storage and validation.**

   The report needs source, changes, story passages, citations, and basic
   generation metadata: provider, model, revisions, usage, and completion
   state.

   Validate schema, revisions, citation ranges, coverage, output sizes, and
   safe rendering at input boundaries. Preserve the visible
   “model-generated · unverified” label. Structural validation must not be
   presented as proof that the prose is true.

   Remove recursive statement ancestry, reconstructed production proofs,
   session capability machinery, and the general-purpose shared-record
   graph from new generation. Their processing and storage costs are not
   needed for a source-linked walkthrough.

   Keep the offline reader's prose/code flow, navigation, original line
   numbers, and lazy expansion. Show each passage's actual additions and
   deletions by default, with full definitions and exact cited ranges
   available beside them. Preserve access to complete supplied diffs.
   Preserve inline code styling for backticked names and paths across the
   narrative, including emphasized text and explanatory change notes.
   Keep old saved reports readable through an isolated compatibility path.

   Acceptance: new artifacts reload without reconstructing generation
   history. Output and cache size grow with source and prose, rather than
   repeated provenance. Browser checks cover source links, expansion,
   offline operation, and safe display of untrusted text.

6. **Prove the replacement, switch the default, and remove obsolete code.**

   Require a real complex-PR run within a few minutes, with recorded calls,
   tokens, local processing time, and artifact sizes. Review the generated
   story beside its cited code. Check that explanations describe the actual
   mechanisms and distinguish author intent, source evidence, and uncertainty.

   Retain regression tests for missing changes, invalid citations, wrong
   revisions, timeouts, exhausted budgets, cache invalidation, saved-report
   reload, and offline browser behavior. Preserve compiler tests for
   matching, source availability, and original line coordinates.

   After acceptance, retire the old generation paths and their
   implementation-specific tests. Reduce the 5,399-line benchmark script
   to a focused performance harness. Keep useful existing work when it
   supports the new contract. Align README, architecture documentation,
   and CLI defaults with the single supported generation path.

   Acceptance: the new path is the default, the superseded generators are
   removed, compatibility is limited to reading saved artifacts, and the
   narration implementation is substantially smaller.

**Milestone order**

- [ ] **Baseline and one-request walkthrough.** Define the benchmark
  contract, build the compact packet, and deliver a complete one-request
  story through the existing reader.
- [ ] **Bounded complex-PR generation.** Add parallel writing, the editorial
  pass, shared run limits, and simple cache recovery. Measure live latency
  and review source fidelity.
- [ ] **Storage cleanup and replacement.** Flatten new artifacts, retain
  saved-report compatibility, verify the reader, switch the default, and
  remove superseded machinery.

The main session owns the plan, integration decisions, and acceptance.
Workers own scoped edits under the repository's agent-routing rules. Use an
independent review before accepting the replacement. Existing uncommitted
work must be inspected and preserved unless a specific change is part of
the accepted replacement.

After narration meets its targets, measure GitHub ingestion against total
runtime. Its source downloads are currently serial. Add bounded parallel
downloads if ingestion materially delays the complete workflow, while
preserving pinned revisions, source limits, and incomplete-source reporting.
