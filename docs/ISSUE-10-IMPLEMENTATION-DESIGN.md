# Issue #10 implementation design

This document is a design. The code changes and verification below are not
implemented or completed by this document. It records the prior design for
[issue #10](https://github.com/epistemic-ai/diffstory/issues/10).

## Scope and current behavior

Use evidence for each file to classify unmatched definitions. Keep the current
symbol matcher, change kinds, schema versions, and ID formulas. Put the
explanation in the existing `change.basis` field.

The prior design's in-memory check confirmed that an unrelated Markdown change
changes `removed` to `observed_base`, and `added` to `observed_head`. Raw hunks
remain. Current code inspection confirms the cause: `_collect_symbol_changes`
in [analysis.py](../diffstory/analysis.py) uses warnings and excerpt scope for
the whole snapshot. `extract` puts text-only notes into that warning stream.
Thus, a limit in one file changes claims about another file.

The check also found a render failure with warnings. In
[render.py](../diffstory/render.py), `_validate_report_collections` requires
warning objects, but the compiler emits strings. Fix this in the same change
so reports with unresolved parse cases can render.

Confirmed additions and removals apply to the supplied file. They do not prove
that a definition or its behavior is absent elsewhere in the repository.

## Snapshot evidence contract

Add optional `file_evidence` to `diffstory.snapshot.v1`. It is a list of file
comparison records. Each record has exactly two fields, `base` and `head`.
Each field contains a side object. For example:

```json
{
  "file_evidence": [
    {
      "base": {"path": "config.py", "state": "supplied", "coverage": "full"},
      "head": {"path": "config.py", "state": "supplied", "coverage": "full"}
    }
  ]
}
```

This example shows the new field only; the snapshot still requires its normal
schema, metadata, and fragments. The allowed side objects are:

| State | Exact fields | Meaning and valid values |
|---|---|---|
| `supplied` | `path`, `state`, `coverage` | At least one source fragment is present. `coverage` is `full` or `partial`. |
| `absent` | `path`, `state` | The file is confirmed absent at this revision. No source fragment is present. |
| `unavailable` | `path`, `state`, `reason` | Source was not supplied. File absence is not established. `reason` is one code below. |

`path` is a nonempty repository-relative string. `state`, `coverage`, and
`reason` are strings from their listed sets. Do not add `coverage` or `reason`
to states that do not permit them. These records describe source availability;
parse results belong in the internal index, not this snapshot field.

| Reason code | Meaning |
|---|---|
| `not_supplied` | No source was supplied for this side. |
| `size_limit` | The source exceeds the per-file size limit. |
| `binary_or_non_utf8` | The source is binary or cannot be decoded as UTF-8. |
| `unsupported_object` | The source object is not a supported regular file. |
| `invalid_content_size` | The declared content size is invalid. |
| `unsupported_encoding` | The returned content encoding is not supported. |
| `read_failed` | A source read failed without proof of absence. This code does not change existing fatal-error rules. |

Keep human error details in warnings. Never derive a reason code from warning
text. For an added file, use the new path on both sides and base `absent`. For a
deleted file, use the old path on both sides and head `absent`. For a rename,
use the old base path and new head path. This pair is the rename link; do not
add a rename flag.

### Validation

Validate the field in `analysis.py` before classification:

- If present, `file_evidence` must be a list. Each record must contain exactly
  `base` and `head`. Each side must be an object with exactly the fields for its
  state. Reject unknown states, coverage values, and reason codes.
- Reject empty paths, absolute paths, NUL, and `.` or `..` path components.
  Compare paths exactly. Do not resolve paths through the filesystem.
- Each `(side, path)` can occur in at most one record.
- Every fragment must belong to exactly one `supplied` side. `absent` and
  `unavailable` sides must have no fragments. When the field is present, it
  must cover all fragments; do not mix explicit evidence and silent fallback.
- A supplied side must have at least one fragment. Empty text counts.
- `full` requires exactly one fragment starting at integer line 1. Reject
  boolean line values. An explicit excerpt scope conflicts with full coverage.
  Allow omitted fragment scope, `full`, or `complete` because the new field
  makes the whole-file claim.
- `partial` permits several nonoverlapping fragments. Do not infer full
  coverage from adjacent ranges. Keep positive-integer line and overlap
  checks for all fragments.
- Reject a record with both sides absent. Two unavailable sides are allowed.
- Keep the existing rejection of snapshots with no fragments, except a
  declared empty comparison (`meta.changed_files == 0`). Evidence records alone
  do not bypass this check.
- Apply the region checks below before raw changes are collected.

## Produce evidence in Git and GitHub ingestion

In [ingest.py](../diffstory/ingest.py), create records from the changed-file
list before reading source. Update each side from the structured read result.
Preserve byte accounting, source limits, revision checks, and fetch rules.

| Producer | Required behavior |
|---|---|
| Local Git | `A` establishes base `absent`; `D` establishes head `absent`. A successful read gives `supplied/full`, including a zero-byte file. Each existing skip branch gives `unavailable` with the applicable reason code. |
| GitHub | `added` and `removed` establish the corresponding absent side. `renamed` links `previous_filename` to `filename`; require the previous path. Successful reads give `supplied/full`. Record each skipped side with its reason. |

Keep local Git's `--no-renames`. It supplies delete/add records. The symbol
matcher detects moves before unmatched classification. Adding Git rename
detection would change status parsing and raw-file grouping without being
needed for this fix.

For GitHub, read base source at the merge base and head source from the current
head repository, as today. Return structured read status from the file-read
helpers and carry it through `_read_github_tasks`. Set the reason in the branch
that detects the failure. Preserve the distinction between a fragment, a
warning, the source-byte count, and the availability result.

Aggregate-limit and Git command failures remain fatal. GitHub API errors,
incomplete file lists, revision changes, and aggregate-limit failures also
remain fatal. A failed request does not establish absence. Do not convert
these fatal errors to `read_failed` records.

## Parse status and internal coverage index

Change `extract` to return an explicit status alongside symbols, imports, and
its optional note. Use `(symbols, imports, status, note)`:

| Value | Valid values and use |
|---|---|
| `symbols` | Existing extracted symbol records; retain their IDs and `fragment_id`. |
| `imports` | Existing import records for the fragment. |
| `status` | `ok`: Python parsed successfully; `text_only`: no Python AST analysis; `failed`: Python parsing failed. |
| `note` | A string for a text-only or parse-failure explanation, or `None`. It is not a status signal. |

Build an internal coverage index in `_prepare_snapshot`, keyed by `(side,
path)`. Each entry holds the paired file record, source state, coverage,
fragment IDs, parse status for each fragment, and unavailable reason. States,
coverage, and reasons use the sets above. Coverage applies only to supplied
source; a reason applies only to unavailable source. Absent and unavailable
sides have no fragment IDs or parse results. Parse status uses the three
values above. Retain each partial fragment's result so one parse failure does
not discard valid symbols from another fragment.

Use `symbol.fragment_id` to find the source fragment and its file evidence.
Do not add the evidence key to symbol IDs. A side is **full and parsed** only
when its full-coverage claim passes validation and its Python source has
status `ok`. Empty Python source qualifies. Failed parsing leaves the source
state `supplied`; it does not imply absence.

### Function contracts

Use Google-style Python docstrings for changed functions and any new helpers.
Document each type's meaning, fields, valid values, and use. State these
preconditions, behavior, and postconditions in the relevant contracts:

| Function or function set | Preconditions | Essential behavior and postconditions |
|---|---|---|
| `_validate_snapshot_input` and evidence/range validators | Snapshot input is untrusted. | Validate the envelope, evidence, paths, ranges, and limits. Return accepted data or raise `ValueError`; do not silently repair evidence. |
| `extract` | Fragment identity and range are validated. | Parse without importing or executing source. Return symbols, imports, explicit status, and optional note. Keep the source-size failure fatal. |
| `_index_snapshot_fragments`, `_prepare_snapshot` | Use validated snapshot data. | Build fragment, symbol, import, and coverage indexes; separate notes and warnings. Keep ID formulas and line offsets unchanged. |
| Git/GitHub read helpers | Path, side, revision, and file task are known. | Return source availability with reason, source data, warning, and byte count as applicable. Preserve network/read effects and fatal errors. |
| `from_git`, `from_github`, `_read_github_tasks` | Revisions and the changed-file list pass existing checks. | Produce evidence for each side and carry structured read results through aggregation. Enforce existing aggregate limits. |
| `_collect_symbol_changes` and basis helpers | Matching is complete; indexes and parse status are available. | Keep matched kinds; classify unmatched symbols with the predicate below. Produce deterministic basis text and existing coverage lookups. |
| `_collect_raw_changes` | Region uniqueness and record links are validated. | Pair supplied text by region without losing a fragment. Preserve offsets, hunks, and raw source rows. |
| `evidence_packet` | Input is a compiled report, including a legacy report. | Include notes, with `[]` when absent, and preserve basis and warnings. |
| `_validate_report_collections` | Report input is untrusted. | Validate object collections separately from string lists. Accept absent notes; reject invalid warnings or notes. |
| Reader badge, figure, and appendix functions | Report passed validation. | Show the labels and escaped basis below. Keep basis visible after lazy rendering and with authored passages. |

## Classify unmatched definitions after matching

Run `match_symbols` first. Keep its matching rules and all matched
classifications. For each remaining unmatched symbol, use this exact predicate:

```text
confirmed =
    own side is supplied, full, and successfully parsed
    AND (
        counterpart side is confirmed absent
        OR counterpart side is supplied, full, and successfully parsed
    )
    AND no unresolved same-name, same-node-type declaration remains
        in the counterpart file
```

The last guard checks **all** extracted counterpart symbols, including matched
symbols. It prevents a failed one-to-one match between repeated declarations
from becoming a false removal or addition. Do not add a matching heuristic.

If the predicate passes, emit `removed` for a base symbol or `added` for a head
symbol. Otherwise emit `observed_base` or `observed_head`, respectively.

| Relevant file evidence | Unmatched base definition | Unmatched head definition |
|---|---|---|
| Both sides supplied, full, parsed; declaration guard passes | `removed` | `added` |
| Base supplied, full, parsed; head absent | `removed` | None |
| Base absent; head supplied, full, parsed | None | `added` |
| Either relevant side partial, text-only, failed, or unavailable | `observed_base`, if extracted | `observed_head`, if extracted |
| Declaration guard fails | `observed_base`, if extracted | `observed_head`, if extracted |
| No extracted definitions | None | None |

Several partial fragments can yield symbols even if another fragment fails.
Their unmatched symbols remain observed. Limits in unrelated files have no
effect on this predicate. Remove `warnings`, global excerpt detection, and
the unused `known_added`/`known_removed` escape paths from
`_collect_symbol_changes`. Consume the coverage index instead.

### Old snapshots without `file_evidence`

- Pair files by the same path. Use a shared explicit region to link different
  paths only when that link is unique and consistent. Conflicting links remain
  unresolved.
- Infer full coverage only from one fragment at line 1 with explicit
  `scope="full"`, and only when snapshot scope is not `selected excerpts`.
- Treat `scope="complete"` as partial unless new evidence establishes full
  coverage. The current line-20 fixture must remain partial.
- A missing side is `unavailable/not_supplied`. Never infer absence from missing
  fragments, metadata counts, warnings, or generated diff rows.

Old snapshots with full base/head source can still confirm definition
additions and removals within those files. Old one-sided snapshots become
observed unless their producer supplies explicit absence evidence.

## Raw diff and region behavior

Keep `region` as the raw-diff pairing key. File evidence selects a counterpart
through the base/head paths. If a fragment has no explicit region, use its
record's head path as the region; this also pairs renamed files. Preserve
explicit regions for excerpts.

Require full base/head fragments in one record to share a region. Reject
duplicate `(region, side)` fragments instead of allowing `_collect_raw_changes`
to overwrite one. Multiple excerpts must have distinct regions. A shared
region must not join fragments from different evidence records.

Keep line offsets and hunk generation unchanged. Region matching aligns the
supplied text; it does not prove full coverage. Added and deleted rows must
remain in symbol hunks and `raw_files` when classification becomes observed.

## Basis text, notes, warnings, and reader behavior

Generate basis text in Python from structured evidence. Use these templates,
with the definition's name and own-side path substituted:

| Case | Exact basis template |
|---|---|
| Removal; both files parsed | `Definition NAME was removed from the supplied file PATH. Both file versions are fully supplied and parsed. No counterpart was matched.` |
| Addition; both files parsed | `Definition NAME was added to the supplied file PATH. Both file versions are fully supplied and parsed. No counterpart was matched.` |
| Whole-file removal | `Definition NAME was removed from the supplied file PATH. The base file is fully supplied and parsed. The file is confirmed absent at the head revision.` |
| Whole-file addition | `Definition NAME was added to the supplied file PATH. The file is confirmed absent at the base revision. The head file is fully supplied and parsed.` |
| Unresolved base symbol | `Definition NAME is present in the supplied base source for PATH. Removal from this file is unresolved: REASON.` |
| Unresolved head symbol | `Definition NAME is present in the supplied head source for PATH. Addition to this file is unresolved: REASON.` |

For linked different paths, include `Base path: OLD. Head path: NEW.` Append
this sentence to each confirmed case: `This result does not establish whether
the definition or its behavior exists elsewhere in the repository.`

Build unresolved reasons from state, coverage, parse status, and the declaration
guard. Use plain text such as:

- `The head source was not supplied; file absence is not confirmed.`
- `Only part of the base file is supplied.`
- `The head Python source could not be parsed.`
- `The head source is unavailable because it exceeds the file size limit.`
- `A definition with the same name and type remains in the other file version, but the matcher did not form one pair.`

If several conditions block confirmation, list reasons in base-then-head order.
Keep the output deterministic. Do not inspect warning wording.

Add optional `report.notes`, a list of strings. Send `text_only` notes there at
extraction time. Keep parse failures and skipped-source warnings in
`report.warnings`, also a list of strings. Preserve incoming legacy warning
strings; do not parse or rewrite them to determine severity.

In `render.py`, remove `warnings` from the list-of-objects check. Validate it as
a list of strings. Validate notes the same way when present. In the evidence
appendix, show notes under **Analysis notes** and warnings under **Limits of
this input**. Include notes in `evidence_packet`, with an empty default for old
reports. No change is needed in [narrative.py](../diffstory/narrative.py): its
source pieces already include `basis`.

Use these labels in Python's `LABELS` and [app.js](../diffstory/assets/app.js):

| Kind | Reader label |
|---|---|
| `added` | `Added to file` |
| `removed` | `Removed from file` |
| `observed_head` | `Addition unresolved` |
| `observed_base` | `Removal unresolved` |

Handle these four kinds before the current `is_test` badge shortcut. Test
definitions must also show their addition or removal status. Keep the existing
test-source treatment for other kinds.

Render each affected change's `basis` as escaped text inside its code figure,
immediately below the caption. Reuse `.code-note`. Put the explanation outside
`.code-body`, which lazy rendering replaces. This must cover default passages,
authored passages, and supporting changes. Preserve authored text; it cannot
suppress the compiler explanation. Remove the old `observed_head` sentence
from `defaultPassages`: not every unresolved case lacks counterpart source.
Keep raw hunks, definition toggles, and source links working as today.

## Compatibility and IDs

Keep snapshot, report, and annotation schemas at v1. Keep all ID formulas.
Because `_append_change` hashes the kind, a corrected kind changes the change
ID. Group IDs remain stable when path and theme remain stable.

Old annotations that cite changed IDs must fail existing validation and be
regenerated. Do not translate stale IDs by name. Old reports must still render,
with notes defaulting to empty. Rendering must not reclassify an old report;
recompile its source snapshot to obtain the corrected kinds.

## Ordered implementation and file ownership

Use one writer per file set. Finish each step before dependent work. This
document authorizes no source edits by itself; these are the bounded steps for
the implementation assignment.

| Order | Owned files | Required result |
|---|---|---|
| 1 | [analysis.py](../diffstory/analysis.py), [test_analysis.py](../tests/test_analysis.py) | Validate evidence and regions. Add parse status and the coverage index. Replace unmatched classification. Add basis, labels, notes, and evidence-packet support. |
| 2 | [ingest.py](../diffstory/ingest.py), [test_integration.py](../tests/test_integration.py) | Produce side evidence with structured reasons. Keep limits, fetch rules, and Git move handling. |
| 3 | [render.py](../diffstory/render.py), [app.js](../diffstory/assets/app.js), [test_reader.py](../tests/test_reader.py) | Fix warning validation. Show file labels and basis beside code. Show notes separately. |
| 4 | [browser_smoke.py](../tests/browser_smoke.py) | Add a small synthetic page that checks actual labels, explanations, and source rows. |
| 5 | [make_demo.py](../examples/make_demo.py), generated `examples/demo.*` | Add evidence from the complete synthetic before/after maps. Regenerate snapshot, annotations, report, and HTML together. |

Keep the existing `snap()` test helper as legacy input. Add explicit evidence
only to tests that need it. Do not silently make every omitted test-side file
absent. CI checks consistency of the generated demo artifacts.

## Focused test matrix

Each test function must name and document one concern: state what could go
wrong. Put input/expected pairs in a matrix and loop over them, using subtests
where useful. Rows below define separate concerns; related variants belong in
that concern's matrix. For directional cases, test both addition and removal.

| Concern / suggested test name | Input variants | Expected result |
|---|---|---|
| `test_unrelated_input_preserves_kind` | Add/delete a constant in full, parsed `config.py`; repeat with changed Markdown, changed `BUILD.bazel`, an unrelated parse failure, and an incoming warning | Target stays `added`/`removed`. |
| `test_relevant_limits_leave_kind_observed` | Missing side; unavailable side; partial source; line-20 `complete` source; counterpart parse failure | `observed_head`/`observed_base` with the relevant basis reason. |
| `test_empty_python_is_full_parsed` | Full empty base versus definition at head; reverse | `added`/`removed`. |
| `test_explicit_absence_confirms_kind` | Explicit absent base/head versus a full parsed file | Whole-file addition/removal is confirmed. |
| `test_legacy_missing_source_is_unavailable` | Omit base/head with no evidence, including metadata or warning hints | Observed kind; absence is not inferred. |
| `test_legacy_scope_limits_full_coverage` | Explicit `full` at line 1; omitted scope; `complete`; `selected excerpts` snapshot | Only the first case can establish legacy full coverage. |
| `test_legacy_region_link_must_be_unique` | Unique shared region across different paths; conflicting links | Unique link pairs files; conflict cannot confirm a kind. |
| `test_matching_preserves_moves` | Existing move and rename fixtures | Existing matched classifications remain. |
| `test_linked_rename_uses_counterpart_evidence` | Linked rename with an unmatched definition; repeat with counterpart unavailable | Confirm only with full parsed counterpart and a passing declaration guard. |
| `test_repeated_declaration_blocks_confirmation` | Same-name, same-node-type counterpart remains, including an already matched symbol | Unmatched declaration remains observed. |
| `test_partial_parse_failure_keeps_valid_symbols` | Multiple partial fragments; one parses and one fails | Keep valid symbols; their unmatched kinds remain observed. |
| `test_evidence_shape_is_strict` | Wrong list/record/side type; extra fields; bad state, coverage, or reason; missing reason | `ValueError`. |
| `test_evidence_paths_are_relative` | Empty, absolute, NUL, `.` or `..` components; valid relative path | Reject each invalid path; accept the valid path. |
| `test_evidence_side_is_unique` | Repeated `(side, path)`; unique sides | Reject duplicate; accept unique sides. |
| `test_evidence_covers_fragments` | Uncovered fragment; source on absent/unavailable side; supplied side without a fragment; supplied empty text | Reject the first three; accept empty text. |
| `test_full_coverage_claim_is_valid` | Several fragments; non-1 start; boolean start; excerpt scope; one line-1 fragment with omitted/full/complete scope | Reject false full claims; accept the last scope variants with explicit full evidence. |
| `test_partial_ranges_do_not_overlap` | Overlapping ranges; adjacent ranges | Reject overlap; keep adjacent ranges partial. |
| `test_absence_record_is_valid` | Both sides absent; two unavailable sides in a snapshot with other source | Reject first; accept second. |
| `test_empty_snapshot_rule_is_preserved` | No fragments with/without declared empty comparison | Accept only the declared empty comparison. |
| `test_region_side_is_unique` | Duplicate `(region, side)` | `ValueError`; no silent overwrite. |
| `test_region_links_stay_within_record` | Full pair with different regions; shared region across records | `ValueError`. |
| `test_raw_rows_survive_classification` | Confirmed and unresolved additions/removals, including unrelated input variants | Symbol hunks and `raw_files` retain added/deleted rows and offsets. |
| `test_partial_regions_keep_all_rows` | Several nonoverlapping excerpts with distinct regions | All supplied changed rows remain. |
| `test_ingest_maps_status_to_evidence` | Git A/D/read/skip; GitHub added/removed/renamed/read/skip; empty read | Exact side states, linked paths, and branch reason codes; empty read is supplied/full. |
| `test_github_rename_requires_previous_path` | Renamed status with/without `previous_filename` | Valid link or failure; no guessed previous path. |
| `test_ingest_fatal_errors_stay_fatal` | Existing Git/API/list/revision/aggregate-limit failure fixtures | Existing fatal outcomes; no false absence. |
| `test_report_string_lists_render` | Text-only notes; parse-warning strings; old report without notes; malformed string lists | Valid reports render; old notes default empty; malformed lists fail. |
| `test_evidence_packet_includes_notes` | New report with notes; old report without notes | Supplied notes or `[]`. |
| `test_annotations_preserve_evidence` | Valid authored annotations | Kinds and basis remain intact. |
| `test_stale_change_ids_are_rejected` | Annotations that cite IDs from the old kind | Existing validation fails; regenerate annotations. |
| `test_basis_reasons_are_deterministic` | Multiple base/head blockers; linked different paths | Stable base-then-head reasons and both paths. |
| Browser: labels | All four kinds, including test definitions | Exact reader labels above. |
| Browser: basis visibility | Default/authored passages and supporting changes, before/after lazy load | Compiler basis stays visible; authored text stays intact. |
| Browser: source rows | Added/deleted definitions and raw-hunk appendix | Added/deleted lines and raw hunks remain visible. |
| Browser: basis escaping | Basis with HTML/script-like text | Text displays; no script executes. |
| Browser: narrow layout | Long basis at narrow viewport | No horizontal page overflow. |

### Verification commands for implementation

Run each command separately and check its result before the next command.
These are planned checks, not recorded passes.

```sh
python -m unittest discover -s tests -p test_analysis.py -v
```

```sh
python -m unittest discover -s tests -p test_integration.py -v
```

```sh
python -m unittest discover -s tests -p test_reader.py -v
```

```sh
python examples/make_demo.py
```

```sh
python tests/browser_smoke.py
```

Then run the normal full unit suite and Ruff checks:

```sh
python -m unittest discover -s tests -v
```

```sh
python -m ruff check .
```

```sh
python -m ruff format --check .
```

Use the generated-demo consistency check from [CI](../.github/workflows/ci.yml)
on the completed change's Git baseline, after generation:

```sh
git diff --exit-code -- examples/demo.html examples/demo.snapshot.json examples/demo.annotations.json examples/demo.report.json
```

During implementation, those files will have intended changes from the old
baseline. Review them together and confirm that a second generation makes no
further changes. Do not report intended artifact updates as a consistency pass
against the old baseline.

## Settled decisions and risks

- Use optional v1 evidence. Require full parsed source for confirmed file
  claims. Keep local Git delete/add handling. Reuse `basis` for explanations.
- Evidence can state a whole-file claim, but validation cannot prove that an
  external producer supplied every byte. Producers must report coverage
  correctly; partial coverage must stay explicit.
- Conservative legacy handling changes some old classifications and their
  change IDs. Regenerate affected annotations and all demo artifacts together.
- Keep evidence linkage separate from raw regions and symbol matching.
  Reject ambiguous region ownership to prevent lost or mispaired source.
- Preserve fatal input and fetch checks. Source availability, Python parse
  success, and repository-wide behavior are distinct claims.

No broad redesign is required. Implementation is complete only when these
contracts and focused checks hold, and the normal verification checks pass.
