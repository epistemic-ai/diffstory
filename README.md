<p><img src="diffstory/assets/epistemic-mark.svg" width="38" alt="Epistemic AI"></p>

# Diffstory

**Read the change, not just the diff.**

A local-first literate code-review reader by [Epistemic AI](https://www.epistemic.ai). Diffstory turns committed code changes into one continuous document: **prose → source → prose → source**. Moved definitions appear together; large blocks expand where you are reading.

[Explore the demo](examples/demo.html) · [Architecture](docs/ARCHITECTURE.md) · [Contributing](CONTRIBUTING.md) · [Security](SECURITY.md)

[![Python test coverage](https://github.com/epistemic-ai/diffstory/raw/python-coverage-comment-action-data/badge.svg)](https://github.com/epistemic-ai/diffstory/tree/python-coverage-comment-action-data)

![Diffstory's single-page reader](docs/reader/desktop.png)

## Try it

Download or clone this repository and open **`examples/demo.html`** in a browser. Download the HTML first; GitHub's file view does not execute it. No server, account, API key, web fonts, CDN, or internet connection is needed to read it. The example is original synthetic code, not an excerpt from a private repository.

To analyze your own checkout, use Python 3.10+ and Git. Install the CLI from this source checkout first:

```bash
python -m pip install .
diffstory --version
```

Then create a walkthrough:

```bash
diffstory git \
  --repo /path/to/your/project \
  --base main --head HEAD \
  --out walkthrough.html
```

Pydantic validates source evidence and narration data. Build and browser-test tools are development-only dependencies.

## A reading path, not a dashboard

The reader keeps each explanation close to the source, with original line
numbers. File diffs open by default. **Before definition** and **After
definition** show the supplied code; **Compare function versions** opens the
separate, generated comparison. Current compiler reports can also use
**Read definition** to switch between a diff and its supplied source. Code
identifiers, paths, commands, and literal values use inline code styling. Long
blocks expand where you are reading, at most 160 display rows per action. Code
scrolls with the document.

**Contents** searches sections and symbols without filtering the document away. Reader options collapse expanded code or download the analysis JSON.

“Lazy loading” means deferred rendering from embedded data. The complete supplied snapshot is still inside the HTML. Treat reports, snapshots, and exported evidence as source-code-bearing files.

## GitHub pull requests

```bash
diffstory github 'OWNER/REPO#NUMBER' \
  --out walkthrough.html \
  --save-snapshot walkthrough.snapshot.json
```

Replace `OWNER/REPO#NUMBER` with a real PR. For private repositories, Diffstory first uses `GITHUB_TOKEN` (or the variable selected by `--token-env NAME`); if that variable is unset, it reuses the signed-in GitHub CLI account through `gh auth token`. Run `gh auth login` once if needed. Public PRs can be read without credentials, subject to GitHub rate limits. Do not paste tokens into issues, commits, commands saved to a shared history, or reports.

The CLI performs read-only requests, paginates changed files, pins both revisions, and rejects a PR that changes during retrieval. It does not inherit credentials from a chat application. The local Git comparison defaults to **merge-base(base, head) → head**; use `--two-dot` for an endpoint comparison. Uncommitted and untracked work is not included.

File diffs keep the original Git or GitHub patch and its line grouping.
**Compare function versions** is a separate, generated view: it can show a
function moving between files while changing just one line. It helps explain
the edit without replacing the PR's patch. See the
[moved-function example](docs/ARCHITECTURE.md#original-patch-and-function-comparison).
If an original patch is unavailable, the reader says so.

Snapshots must include `file_evidence` with an explicit source state for both revisions of each file, plus one `patches` record per file. A patch record contains the original Git/GitHub patch or null when unavailable. The Git and GitHub commands supply these records. Older snapshots that omit them are rejected; capture them again with the current command.

Every compilation writes standalone HTML plus an adjacent `.report.json`. Binary, oversized, unsupported, or missing source is explicitly reported; it is not silently treated as a complete analysis.

## What the compiler knows

Python functions, async functions, classes, and assignments are compared using the standard-library AST. Pairing distinguishes identical ASTs, declaration renames, edited move candidates, ordinary edits, additions, removals, and unresolved counterparts. Literals, internal names, defaults, decorators, and string whitespace are not erased to obtain a convenient match.

Static calls, references, import aliases, and test references form a dependency graph. Strongly connected components handle cycles; deterministic priorities produce a reproducible baseline order. A generated walkthrough can arrange concepts in a different order. Compiler groups and dependencies are evidence and ordering hints; validation does not require a concept to appear after its prerequisites. This is a comprehension heuristic, not a proof of an optimal order.

Other languages, newer Python syntax unsupported by your interpreter, and unclassified module-level code remain available as textual hunks. Classes are atomic units. Dynamic dispatch, reflection, unchanged external callers, and whole-program behavior are not fully resolved.

**Identical ASTs are not a proof of identical behavior. Test references are not measured coverage or passing tests.** Diffstory does not execute repository source, run its tests, or issue a merge-safety verdict.

## Add better prose without surrendering the evidence

The CLI creates deterministic baseline explanations. Domain-specific prose is a separate, revision-bound annotation layer. The bundled synthetic walkthrough includes authored annotations to demonstrate the intended reading experience.

```bash
diffstory evidence compiler.report.json --out narrative-evidence.json
# Write annotations.json from the documented evidence schema.
diffstory render compiler.report.json \
  --annotations annotations.json --out narrated.html
```

Evidence packets contain source code. Send them only to an approved model or environment. The manual annotation workflow makes no provider call and needs no API key. Annotation validation locks revisions and evidence IDs; it cannot establish that every prose claim is true. See [the annotation contract](docs/ARCHITECTURE.md#annotation-handoff-and-current-walkthrough).

### Optional model-written walkthrough

The `git`, `github`, and `snapshot` commands accept `--narrate`. Diffstory
builds a bounded packet from the supplied source, then asks OpenAI Responses or
the signed-in Codex CLI to write source-linked explanations. OpenAI is the
default provider. It uses the OpenAI API key from `OPENAI_API_KEY` (or the
variable selected by `--api-key-env`); a ChatGPT subscription does not provide
an API key. Codex uses `codex exec` and saved CLI sign-in. Use `--provider
codex` to select it. Both providers receive only the prepared evidence request.
OpenAI currently supports `gpt-6.1-sol`. Codex accepts the known-capacity
models `gpt-6-astra`, `gpt-6.1-sol`, `gpt-6-luna`, and `gpt-5.3-codex`.
Unknown Codex overrides are rejected before source transfer.

```bash
diffstory github 'OWNER/REPO#NUMBER' --narrate --out walkthrough.html
diffstory git --repo . --base BASE --head HEAD --narrate --out walkthrough.html
diffstory snapshot input.snapshot.json --narrate --yes --out walkthrough.html
```

Before the first model request, Diffstory displays the provider destination,
revision scope, and planned request count, then asks for consent. `--yes`
skips that prompt and means the source transfer was approved in advance. No
model request occurs without `--narrate`. The aggregate source limit defaults
to 64,000,000 bytes across both revisions; `--max-source-bytes` can lower it.

The writer receives each source buffer once in the packet. Passages cite the
exact revision and inclusive source lines. Every compiler change and changed
diff region must be accounted for by at least one passage. Complete supplied
diffs remain in the report for inspection. Compiler groups and dependencies
guide preparation, but the writer can organize the story around concepts and
the validator does not enforce prerequisite order.

The walkthrough uses a small or parallel writing pass and at most one
editorial request. It allows at most 12 attempts, including retries, up to four
concurrent calls, a 240-second preparation-and-generation deadline, a 1 MiB
total response-byte budget per run, and a separate 1 MiB saved-annotation
limit. Confirmation time is not part of the deadline. If a limit or request
fails, Diffstory does not promote new report or HTML outputs.

Run attempts and the original deadline are recorded in a durable state
directory. By default it is the output path with `.narration-state` in place of
the file suffix. `--state-dir` selects an existing or new location;
`--fresh-state NEW_DIRECTORY` starts a new budget while retaining the old
state. Payload caching is separate. Its default is the output path with
`.narration-cache` in place of the suffix. Use `--cache-dir` to select a
location or `--no-cache` to disable payload reuse. Turning off the cache does
not turn off attempt state. Cached results are bound to prepared requests and
revalidated before use. Codex's unresolved default model disables the cache;
an explicit supported `--model` enables it.

The CLI prints provider-reported input, cached input, and output token totals.
The report and candidate annotation export record provider, model, revisions,
usage, request counts, and cache counts. They do not store credentials. A
successful run writes HTML, `.report.json`, and `.annotations.json`; the
annotation export contains the story and metadata but no source packet.

The report uses `diffstory.report.v2`; the optional annotation export uses
`diffstory.annotations.walkthrough.v1`. Compiler reports use
`diffstory.report.v1`, and manual annotations use `diffstory.annotations.v1`.
Use `diffstory evidence REPORT.json --out evidence.json` to export the compact
source packet without making a provider call. Use `diffstory render REPORT.json
--annotations ANNOTATIONS.json --out narrated.html` to apply manual or saved
walkthrough annotations to matching compiler source.

Generated narration requires a **Before the code** preamble. It first explains
the system's purpose, inputs, operation, outputs, and main roles in simple
English. It gives a source-grounded overview before file or function details.
The writing follows ASD-STE100 principles as a guide, keeps the natural rhythm
of prose, defines necessary jargon at first use, and puts code names, paths,
commands, and literal values in backticks. The preamble has separate 4,000
character limits for prose and complete fenced sketches. Structural checks
enforce fields, citations, coverage, and size. They do not prove that prose is
true, that behavior is correct, or that tests ran. Generated prose is labeled
**model-generated · unverified**.

The reader is a self-contained HTML page. It does not send report data over the
network. Source links open GitHub only when a reader clicks them. OpenAI
requests set `store: false`; this is not a zero-retention promise. Review the
[OpenAI API data controls](https://developers.openai.com/api/docs/guides/your-data)
before sending confidential source. Codex CLI runs follow the signed-in
account's current data and retention controls.

## Develop and verify

Development uses Python 3.13, pinned in `.python-version`. The package supports
Python 3.10 and newer. `uv sync --locked` creates `.venv` and installs the
project with its development tools from `uv.lock`; `uv run --locked` runs each
command in that environment without manual activation.

```bash
uv sync --locked
uv run --locked python -m unittest discover -s tests -v
uv run --locked python examples/make_demo.py
uv run --locked python scripts/check_release.py
uv run --locked python -m playwright install chromium
uv run --locked python tests/browser_smoke.py
uv run --locked python tests/browser_walkthrough.py
uv build
uv run --locked twine check dist/*
```

For a system Chromium installation, set `DIFFSTORY_CHROMIUM` to its executable path. Browser tests exercise rendering, navigation, expansion, responsive layout, and offline behavior. They do not claim tests passed in the analyzed repository. See [verification](docs/VERIFICATION.md) for the checks actually run on this release.

## Public examples and privacy

Only original synthetic source belongs in public fixtures. Generated review files are ignored by default, with explicit exceptions for `examples/demo.*`. `scripts/check_release.py` checks the public file set for known private-fixture identifiers and common credential patterns before publishing. It is a defense-in-depth check, not a comprehensive secret detector.

The analyzer is read-only and the reader does not contact external services. Explicitly clicked source links open GitHub. The release helper in `scripts/publish.sh` is a separate, opt-in GitHub write operation; it is never called by the analyzer.

## Status and license

**0.3.0 · alpha.** Usable for local review and experimentation; not a substitute for tests or human review. Diffstory supports local change analysis and optional bounded model-written walkthroughs. Whole-repository indexing, statement-level extraction matching, and verified test-artifact ingestion are not supported.

Software and original synthetic examples: [MIT](LICENSE). Epistemic AI trademarks and the supplied logo have separate terms in [NOTICE](NOTICE) and [Branding](docs/BRANDING.md). No font files are bundled.
