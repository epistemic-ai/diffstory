# Internal contract

```text
Original Git/GitHub patches + pinned source
    → retained file patches + separate function comparisons
    → Python AST units
    → conservative cross-file declaration pairs
    → calls / references / imports / test references
    → semantic chapters + prerequisite graph
    → SCC condensation + deterministic reading heuristic
    → baseline narrative + optional revision-bound annotations
    → source-bound prose/code passages
    → continuous HTML document + lazy inline code + machine-readable report
```

## Evidence and presentation are separate

### Original patch and function comparison

The original patch shows what changed in each file. Git ingestion captures the
output of `git diff`; GitHub ingestion retains each file's `patch` from the
[PR files API](https://docs.github.com/en/rest/pulls/pulls#list-pull-requests-files).
We keep that text and its line grouping. We check the rows against
the captured file versions; we do not run another diff algorithm to replace
Git's or GitHub's choices. An incomplete or mismatched patch is rejected.

The **function comparison** is an extra view. It compares the earlier and
later versions of a matched function, even when they are in different files.
It is generated from source and labeled separately. It does not replace the
original patch.

Passages can show excerpts from that original patch. Selecting a smaller
range must not choose a different alignment of its lines. The complete
captured patch remains available under **Original source hunks**.

For example, suppose `normalize_name()` moves from `users.py` to `text.py`.
Both files already exist. The original file patch shows:

```diff
# users.py
-def normalize_name(name):
-    return name.lower()

# text.py
+def normalize_name(name):
+    return name.strip().lower()
```

The optional function comparison shows the small edit more directly:

```diff
 def normalize_name(name):
-    return name.lower()
+    return name.strip().lower()
```

The function moved, and it now removes surrounding spaces before lowercasing.
For a 200-line function, this view avoids reading the whole function twice.
Keep both views: the original patch records the PR; the function comparison
helps explain a move or edit. A proposed match still needs review when the
tool cannot confirm that the two definitions belong together.

If GitHub omits a file patch, record it as unavailable. Do not silently replace
it with a generated diff. Source can still support the separate function
comparison. Snapshots must include a patch record for each file, even when
the patch text is unavailable. Older snapshots without these records are
rejected; capture them again with the current Git or GitHub command.
The retained original text also
keeps file headers and no-newline markers that are not ordinary code rows.

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
  "patches": [{
    "path": "pkg/module.py",
    "origin": "git",
    "patch": "@@ -0,0 +1,2 @@\n+def f(x):\n+    return x\n"
  }],
  "warnings": []
}
```

Pydantic models in `models.py` validate snapshot fields and source states. Each state has its own fields: an absent file has no coverage or reason, an unavailable file requires a reason, and supplied source requires coverage. `evidence.py` then checks these claims against the actual fragments and parse results. The compiler receives named indexes instead of a positional bundle.

Only the current snapshot contract is supported. `file_evidence` is required. It records the source state of each changed path on both revisions. A side is `absent`, `unavailable` with a reason, or `supplied` with `full` coverage. Git and GitHub ingestion create these records from pinned source reads. A copy or rename keeps its old path on the base side and its new path on the head side. The path link lets the compiler compare the correct files. A missing or skipped read cannot prove that a definition is absent.

Use `scope: "selected excerpts"` for partial snapshots and set each supplied excerpt side to `coverage: "partial"` in its file evidence. If the other revision was not supplied, mark it `unavailable` with a reason. Each excerpt retains its original line offset. When multiple excerpts are supplied from the same file, set a distinct `region` to identify each before/after pair; source ranges must not overlap. Full snapshots include both sides of modified files and the existing side of truly added/deleted files. A confirmed Python definition addition or removal needs full, parsed source for the relevant file pair. A warning about another file, including a Markdown file, does not weaken that evidence. Missing source and parse failures on the relevant file leave the counterpart unresolved. Snapshots without `file_evidence` are rejected.

The report includes `changes`, `groups`, `edges`, `symbol_edges`, `tests`, `raw_files`, `cycles`, `notes`, `warnings`, `stats`, `meta` and `method`. Stable IDs join evidence and narrative. Line URLs bind to a commit, not a moving branch. The renderer rejects malformed identifiers, numeric fields, diff tags and missing group references, escapes source/prose and prevents literal script terminators in embedded JSON.

Snapshots require `patches`, a list of `{path, origin, patch}` records.
The path is the later path from the file evidence; `origin` is `git` or
`github`; `patch` contains the original text or is null when unavailable.
The snapshot's pinned revisions apply to these records too. Patch paths must
be unique, with one record for every file evidence pair. Total retained patch
text has a separate 64,000,000-byte cap. It does not raise the source-byte limit.

## Annotation handoff and current walkthrough

Compiler output uses `diffstory.report.v1`. Human-authored prose can be saved
separately as `diffstory.annotations.v1` and applied only to compiler output
with matching base and head revisions. The example below shows the main shape;
fields that need source context use existing group and change IDs.

```json
{
  "schema": "diffstory.annotations.v1",
  "base_sha": "<same effective base as the report>",
  "head_sha": "<same head as the report>",
  "document": {
    "preamble": "A source-grounded overview before the code tour.",
    "lead": "A short opening paragraph.",
    "closing": "The mental model to retain."
  },
  "steps": [{
    "group_id": "<existing group ID>",
    "title": "Explain one part of the change",
    "intent": "What this change does, based on source.",
    "why_now": "How this part relates to the reading path.",
    "takeaway": "The useful mental model.",
    "invariants": ["A concrete property to verify"],
    "questions": ["A review question, not an invented defect"],
    "evidence_change_ids": ["<change ID in this group>"],
    "passages": [{
      "text": "This paragraph explains `the_symbol` beside its code.",
      "change_ids": ["<change ID in this group>"],
      "view": "definition",
      "focus": {"start": 120, "end": 130}
    }]
  }]
}
```

The manual `preamble` is optional. In generated narration, **Before the code**
is required. It introduces the system's purpose, concrete inputs, operation,
outputs, and the roles of its main parts before implementation details. It
starts with a plain sentence about supplied context. If the author did not give
a goal, it describes what the evidence shows without guessing at motivation.
It can name relevant classes with simple, source-grounded roles. It defines
necessary technical terms when they first appear. It does not list files, give
a code tour, or add unrelated review questions.

The preamble follows ASD-STE100 principles as a strong guide while keeping a
natural reading rhythm and exact technical terms. The prose limit is 4,000
characters. Complete fenced sketches have a separate 4,000-character limit,
including fences; the combined field limit is 8,000 characters. A sketch is
optional. Generated prose puts identifiers, paths, commands, and literal
values in backticks. Manual annotations may be partial. Their revision and ID
checks do not prove that a prose claim is true.

The `git`, `github`, and `snapshot` commands accept `--narrate`. The current
flat report uses `diffstory.report.v2`; its source packet uses
`diffstory.packet.v1`; its annotation export uses
`diffstory.annotations.walkthrough.v1`. The packet stores each source buffer
once. Passages refer to exact revisions and inclusive original line ranges.
Every compiler change and changed diff region must be covered by at least one
passage. Full supplied diffs remain available in the report. The writer can
organize the explanation around concepts and combine related changes. Compiler
groups and dependency edges are evidence and ordering hints. Validation checks
references and coverage; it does not enforce compiler prerequisite order.

A successful generated report records provider, model, destination, exact
revisions, evidence digest, usage, writing and editorial request counts, and
cache counts. It contains no credentials. The renderer labels model prose
**model-generated · unverified**. Validation checks shape, size, source links,
citations, and changed-region coverage. It does not assess the truth of the
prose or claim that repository tests ran.

### Provider consent, limits, and state

OpenAI Responses is the default provider; `--provider codex` selects the
signed-in Codex CLI. OpenAI uses an API key from `OPENAI_API_KEY` by default,
or the environment variable named by `--api-key-env`. Codex uses `codex exec`
and saved CLI sign-in. OpenAI currently supports `gpt-6.1-sol`; Codex accepts
the known-capacity overrides `gpt-6-astra`, `gpt-6.1-sol`, `gpt-6-luna`, and
`gpt-5.3-codex`. Unknown Codex overrides are rejected before source transfer.
Before sending source, the CLI displays the provider destination, revision
scope, and planned request count, then asks for consent.
`--yes` skips the prompt when the user has already approved that transfer. The
source cap defaults to 64,000,000 bytes across both revisions and can be
lowered with `--max-source-bytes`.

The generation runtime allows at most 12 provider attempts including retries,
four concurrent calls, and 240 seconds for preparation and generation. It
allows 1 MiB of total canonical response bytes per run and a separate 1 MiB
saved-annotation limit. Interactive consent wait is outside the deadline. A
failed request or limit does not produce a complete report. The CLI prints
provider-reported input, cached input, and output token totals. These counts do
not show that the generated explanation is correct.

Run attempts and the original deadline are stored durably. The default state
path replaces the output suffix with `.narration-state`; `--state-dir` selects
a path. Resume uses the recorded attempt budget and deadline. `--fresh-state`
requires a new path and retains old state. Payload cache data is separate: by
default it replaces the output suffix with `.narration-cache`. `--cache-dir`
selects a path and `--no-cache` disables payload reuse without disabling run
state. Cache entries are bound to prepared requests and revalidated before
reuse. The unresolved Codex default model disables persistent caching; choose
an explicit supported `--model` to enable it.

### Supported command flow

```bash
# Compile local changes and write HTML plus a .report.json file.
diffstory git --repo . --base BASE --head HEAD --out walkthrough.html

# Read a pull request. Add --narrate only to send its prepared source to a provider.
diffstory github 'OWNER/REPO#NUMBER' --narrate --out walkthrough.html

# Compile a saved diffstory.snapshot.v1 input.
diffstory snapshot input.snapshot.json --out walkthrough.html

# Export source evidence without a model call.
diffstory evidence walkthrough.report.json --out evidence.json

# Apply revision-bound manual or saved walkthrough annotations to compiler data.
diffstory render compiler.report.json --annotations annotations.json --out annotated.html
```

Every compile or render writes a standalone HTML page and adjacent
`.report.json`. Narration also writes a candidate `.annotations.json` unless
`--annotations-out` changes its path. The manual evidence flow makes no
provider call. The output page embeds report data, CSS, JavaScript, and the
brand mark. It works offline; it does not execute supplied repository code or
fetch source. Explicit source links open GitHub only when a reader clicks them.
The OpenAI request uses `store: false`, but that setting is not a zero-retention
promise. Data controls of the selected provider still apply.

The reader presents prose beside code and diff evidence. Source and prose are
escaped before display, and the page's content security policy blocks network
connections, external scripts, and object loads. It preserves original source
line numbers and supports lazy code previews and expansion. Structural checks
and offline display do not replace human review or tests.

## Deliberate prototype boundaries

Python AST support follows the interpreter running the CLI. Newer syntax falls
back to textual evidence. Classes are atomic. Static resolution handles common
import aliases and simple local or parameter shadows, but it does not claim
whole-program completeness. Dynamic dispatch, reflection, unchanged external
callers, and whole-program behavior are not fully resolved.

The tool does not index the whole repository, match extracted statements,
ingest verified test-run artifacts, execute supplied code, or issue a
merge-safety verdict. Test references are not measured coverage or passing
tests. Identical ASTs do not prove identical behavior.
