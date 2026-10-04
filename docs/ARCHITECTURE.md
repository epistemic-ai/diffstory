# Internal contract

```text
Committed Git objects / pinned GitHub sources / saved snapshot
    → Python AST units + independent textual hunks
    → conservative cross-file declaration pairs
    → calls / references / imports / test references
    → semantic chapters + prerequisite graph
    → SCC condensation + deterministic reading heuristic
    → baseline narrative + optional revision-bound annotations
    → source-bound prose/code passages
    → continuous HTML document + lazy inline code + machine-readable report
```

## Evidence and presentation are separate

A Git hunk is not the identity of a conceptual change. A moved definition is represented by a single change with both source spans and an independently computed paired delta. The raw source view retains file-oriented hunks, including imports and code that was not parsed into symbols.

Matching uses four stages: unique same-path/name matches; unique exact-AST cross-file matches; unique function/class matches with only the declaration name and leading docstring excluded; and conservative same-name candidates with a structural similarity threshold. Ambiguous duplicates remain unpaired. Similarity is evidence for correspondence, not evidence for preserved behavior. A same-path implementation edit is not automatically called a behavioral regression.

Module environment differences are deliberately not erased. Renaming a reference inside a body, modifying a literal, or changing a default/decorator prevents exact-match classification.

## Snapshot shape

```json
{
  "schema": "diffstory.snapshot.v1",
  "meta": {
    "repository": "owner/repo",
    "base_sha": "<full effective base commit>",
    "head_sha": "<full head commit>",
    "scope": "changed files",
    "changed_files": 1
  },
  "fragments": [
    {
      "path": "pkg/module.py",
      "side": "head",
      "text": "def f(x):\n    return x\n",
      "start_line": 1,
      "scope": "full"
    }
  ],
  "file_evidence": [{
    "base": {"path": "pkg/module.py", "state": "absent"},
    "head": {"path": "pkg/module.py", "state": "supplied", "coverage": "full"}
  }],
  "warnings": []
}
```

Pydantic models in `models.py` validate snapshot fields and source states. Each state has its own fields: an absent file has no coverage or reason, an unavailable file requires a reason, and supplied source requires coverage. `evidence.py` then checks these claims against the actual fragments and parse results. The compiler receives named indexes instead of a positional bundle. Legacy inference stays separate from explicit file claims.

`file_evidence` records the source state of each changed path on both revisions. A side is `absent`, `unavailable` with a reason, or `supplied` with `full` coverage. Git and GitHub ingestion create these records from pinned source reads. A copy or rename keeps its old path on the base side and its new path on the head side. The path link lets the compiler compare the correct files. A missing or skipped read cannot prove that a definition is absent.

Use `scope: "selected excerpts"` for partial snapshots. Each excerpt retains its original line offset. When multiple excerpts are supplied from the same file, set a distinct `region` to identify each before/after pair; source ranges must not overlap. Full snapshots include both sides of modified files and the existing side of truly added/deleted files. A confirmed Python definition addition or removal needs full, parsed source for the relevant file pair. A warning about another file, including a Markdown file, does not weaken that evidence. Missing source and parse failures on the relevant file leave the counterpart unresolved. Legacy snapshots without `file_evidence` use conservative coverage rules.

The report includes `changes`, `groups`, `edges`, `symbol_edges`, `tests`, `raw_files`, `cycles`, `warnings`, `stats`, `meta` and `method`. Stable IDs join evidence and narrative. Line URLs bind to a commit, not a moving branch. The renderer rejects malformed identifiers, numeric fields, diff tags and missing group references, escapes source/prose and prevents literal script terminators in embedded JSON.

## Annotation handoff

```json
{
  "schema": "diffstory.annotations.v1",
  "base_sha": "<same effective base as the report>",
  "head_sha": "<same head as the report>",
  "document": {
    "preamble": "A source-grounded overview of the whole change, before the code tour.",
    "lead": "A short source-grounded opening paragraph.",
    "closing": "The mental model to retain."
  },
  "steps": [{
    "group_id": "<existing group ID>",
    "title": "Explain the abstraction before its callers",
    "intent": "What this change accomplishes, grounded in source.",
    "why_now": "What the previous step established and what this one adds.",
    "takeaway": "The mental model needed for the next step.",
    "invariants": ["A concrete property to verify"],
    "questions": ["A review question, not an invented defect"],
    "evidence_change_ids": ["<change ID belonging to this group>"],
    "transition": "A sentence leading to the next idea.",
    "passages": [{
      "text": "This paragraph explains `the_symbol` immediately before its code.",
      "change_ids": ["<change ID belonging to this group>"],
      "view": "definition",
      "focus": {"start": 120, "end": 130}
    }]
  }]
}
```

The optional preamble introduces the whole change before the code tour. Model narration requires it. It can include one compact conceptual sketch; a two-path decision sketch uses a three-line plain-text flow chart. The preamble allows at most 4,000 characters of prose and a separate 4,000 characters for all complete fenced text sketches, including their fences. The whole field is limited to 8,000 characters. Unclosed fences count as prose. Other document prose has a 6,000-character limit. The provider JSON schema comes from the same Pydantic document model used for response validation. The annotation, generated-report, and renderer checks share its field rules and limits.

The compiler rejects a revision mismatch, missing/out-of-group evidence IDs, oversized prose and invalid field types. Annotation fields cannot replace structural matches or claim test execution. The current validator does not prove that a natural-language claim follows from the source. Human review remains necessary.

When an annotation supplies a step for every group, the step array's order is the reading order. Narrated reports may choose a story order, but the validator keeps prerequisite groups before their dependents and groups in a dependency cycle adjacent. The generated step order is applied to the report's `groups` array, so the contents list and rendered document use the same path. Partial manual annotations retain the report's baseline order.

Manually authored annotations remain backward compatible: passages and group coverage are optional. Generated annotations add a top-level `generation` object with schema `diffstory.generation.v1`. It records `origin: "model_generated"`, `verification: "unverified"`, provider/model, exact revisions, measured usage, expected/completed group and change IDs, chunk coverage, and sanitized errors. It contains no credentials. Generated annotations must have exactly one step for every group, cite every group change, and include nonempty source-bound passages covering every change. Passage focus ranges must fit both the inspected source slice and the report's original source lines. Incomplete generated candidates are rejected; existing manual annotation behavior is unchanged.

When generated annotations are applied, the validated `generation` object is copied into the report JSON. Each generated group receives `provenance: "model-generated · unverified"`. Rendering checks the generation revisions, usage shape, chunk completion, group/change coverage, passages, and provenance label again. The reader displays both a document-level warning naming the provider/model and a per-step generated label. Re-rendering a generated report preserves this warning.

## Continuous reader

`document`, `passages` and `transition` are optional, backward-compatible annotation fields. A focus range must fall inside one supplied head definition (or base when no head exists); ranges refer to original source lines, not snippet-relative offsets. A diff view cannot accept a definition focus. The passage `view` field remains validated for compatibility and focus checks but does not set the reader's initial mode. Repeated references in different passages are allowed for discussing a signature and body separately. Missing references do not remove units: the reader exposes them under Supporting changes. The renderer also validates annotations in a precompiled report.

The UI renders every section and prose passage into one document. Story passages open in diff view; **Read definition** switches a block in place and applies any validated focus range. Test source remains a definition, and raw appendix hunks remain diffs. Narration instructions require backticks around code identifiers, paths, commands and literal values. The renderer styles explicit backtick spans and automatically formats known source names, paths, short variable names and command flags when an annotation omits the markers. An IntersectionObserver defers offscreen code previews; manual load controls provide a fallback. A bounded preview is the default for long definitions/diffs. User expansion adds up to 160 display rows per action. No network fetch is involved: all source is embedded as escaped JSON. There is no side pane, tabbed dashboard or step router. A small Contents popover jumps to stable section anchors, while native page scrolling is preserved.

Supporting units and original raw hunks are created on demand. Source/prose is escaped before display, and the page's CSP disallows network connections, external scripts and object loads.

## Deliberate prototype boundaries

Python AST support is implemented with the standard library, so parse support follows the interpreter running the CLI. A newer-syntax file falls back to textual evidence. Classes are atomic. Static resolution respects common import aliases and avoids simple local/parameter shadows, but does not claim whole-program completeness.

The default narrative uses deterministic templates. The demo adds authored domain-specific annotations. The opt-in `--narrate` path supports the OpenAI Responses API and the user's signed-in Codex CLI. Both providers return structured output and assemble validated source-bound passages with hierarchical summaries. OpenAI requests use its selected model capacity; known Codex overrides use published model capacity, while the CLI default is packed against a conservative 400,000-token context bound. A planning call chooses a story order from group summaries; dependency constraints and cycle groups are enforced before section prose is written. Long source lines are split into bounded slices without dropping text. If a leaf response omits change citations, Diffstory may make one bounded repair request for that chunk. The Codex adapter runs `codex exec` from a temporary directory, in a read-only sandbox, with local shell, browser, app, and plugin tools disabled. Each provider call has a 900-second timeout. Both adapters use the standard library for transport. Shared data validation uses Pydantic. Successful calls contribute provider-reported input, cached input, and output token counts to a run summary; no run-wide token, call-count, or elapsed-time budget is imposed.

Git and GitHub ingestion enforce a 64 MB aggregate source limit by default, and snapshot compilation enforces the same limit. The annotation manifest and report retain revisions, measured usage, and chunk coverage without storing credentials. The model-generated warning states that prose remains unverified.

There is no background agent, GitHub write access, production deployment, formal equivalence engine, live coverage ingestion, or automatic merge recommendation. The UI does not execute supplied repository source. JavaScript executes only the local report interface.

## High-value next increments

First add full-repository symbol indexing with explicit resolution confidence and unchanged callers. Then add statement-level extraction matching (for shared helpers cut out of larger functions), followed by test-run artifact ingestion tied to exact revisions. Additional narrative adapters may be added later, but must remain downstream of evidence analysis and must not rewrite classifications or validation status.

## Primary implementation references

- Python AST API: https://docs.python.org/3/library/ast.html
- GitHub pull-request file pagination: https://docs.github.com/en/rest/pulls/pulls#list-pull-requests-files
- GitHub contents API: https://docs.github.com/en/rest/repos/contents
