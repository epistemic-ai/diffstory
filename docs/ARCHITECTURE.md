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
  "warnings": []
}
```

Use `scope: "selected excerpts"` for partial snapshots. Each excerpt retains its original line offset. When multiple excerpts are supplied from the same file, set a distinct `region` to identify each before/after pair; source ranges must not overlap. Full snapshots include both sides of modified files and the existing side of truly added/deleted files. Missing data, parse failures and skip warnings cause conservative unresolved-counterpart labels instead of definitive addition/removal claims.

The report includes `changes`, `groups`, `edges`, `symbol_edges`, `tests`, `raw_files`, `cycles`, `warnings`, `stats`, `meta` and `method`. Stable IDs join evidence and narrative. Line URLs bind to a commit, not a moving branch. The renderer rejects malformed identifiers, numeric fields, diff tags and missing group references, escapes source/prose and prevents literal script terminators in embedded JSON.

## Annotation handoff

```json
{
  "schema": "diffstory.annotations.v1",
  "base_sha": "<same effective base as the report>",
  "head_sha": "<same head as the report>",
  "document": {
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

The compiler rejects a revision mismatch, missing/out-of-group evidence IDs, oversized prose and invalid field types. Annotation fields cannot replace structural matches or claim test execution. The current validator does not prove that a natural-language claim follows from the source. Human review remains necessary.

## Continuous reader

`document`, `passages` and `transition` are optional, backward-compatible annotation fields. A focus range must fall inside one supplied head definition (or base when no head exists); ranges refer to original source lines, not snippet-relative offsets. A diff view cannot accept a definition focus. Repeated references in different passages are allowed for discussing a signature and body separately. Missing references do not remove units: the reader exposes them under Supporting changes. The renderer also validates annotations in a precompiled report.

The UI renders every section and prose passage into one document. An IntersectionObserver defers offscreen code previews; manual load controls provide a fallback. A bounded preview is the default for long definitions/diffs. User expansion adds up to 160 display rows per action. No network fetch is involved: all source is embedded as escaped JSON. There is no side pane, tabbed dashboard or step router. A small Contents popover jumps to stable section anchors, while native page scrolling is preserved.

Per-section audit material, linked test source, supporting units and original raw hunks are created on demand. The review JSON schema and revision-based storage key remain v1. Notes from the earlier prototype can be imported when the exact revisions and group IDs match. Source/prose is escaped before display, and the page's CSP disallows network connections, external scripts and object loads.

## Deliberate prototype boundaries

Python AST support is implemented with the standard library, so parse support follows the interpreter running the CLI. A newer-syntax file falls back to textual evidence. Classes are atomic. Static resolution respects common import aliases and avoids simple local/parameter shadows, but does not claim whole-program completeness.

The default narrative uses deterministic templates. The demo adds authored domain-specific annotations. The evidence/annotation interface permits an LLM-assisted step, but automatic provider integration, API billing, model retries and model evals are not implemented.

There is no background agent, GitHub write access, production deployment, formal equivalence engine, live coverage ingestion, or automatic merge recommendation. The UI does not execute supplied repository source. JavaScript executes only the local report interface.

## High-value next increments

First add full-repository symbol indexing with explicit resolution confidence and unchanged callers. Then add statement-level extraction matching (for shared helpers cut out of larger functions), followed by test-run artifact ingestion tied to exact revisions. An automated narrative provider should remain downstream of evidence analysis and must not be allowed to rewrite classifications or validation status.

## Primary implementation references

- Python AST API: https://docs.python.org/3/library/ast.html
- GitHub pull-request file pagination: https://docs.github.com/en/rest/pulls/pulls#list-pull-requests-files
- GitHub contents API: https://docs.github.com/en/rest/repos/contents
