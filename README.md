<p><img src="diffstory/assets/epistemic-mark.svg" width="38" alt="Epistemic AI"></p>

# Diffstory

**Read the change, not just the diff.**

A local-first literate code-review reader by [Epistemic AI](https://www.epistemic.ai). Diffstory turns committed code changes into one continuous document: **prose → source → prose → source**. Moved definitions appear together; large blocks expand where you are reading.

[Explore the demo](examples/demo.html) · [Architecture](docs/ARCHITECTURE.md) · [Contributing](CONTRIBUTING.md) · [Security](SECURITY.md)

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
separate, generated comparison. Older reports retain their **Read definition**
control. Code identifiers, paths, commands, and literal values use inline code
styling. Long blocks expand where you are reading, at most 160 display rows
per action. Code scrolls with the document.

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
If an original patch is unavailable, the reader says so. Older saved inputs
that contain only source text show a labeled **Source comparison**.

Snapshots must include `file_evidence` with an explicit source state for both revisions of each file, plus one `patches` record per file. A patch record contains the original Git/GitHub patch or null when unavailable. The Git and GitHub commands supply these records. Older snapshots that omit them are rejected; capture them again with the current command.

Every compilation writes standalone HTML plus an adjacent `.report.json`. Binary, oversized, unsupported, or missing source is explicitly reported; it is not silently treated as a complete analysis.

## What the compiler knows

Python functions, async functions, classes, and assignments are compared using the standard-library AST. Pairing distinguishes identical ASTs, declaration renames, edited move candidates, ordinary edits, additions, removals, and unresolved counterparts. Literals, internal names, defaults, decorators, and string whitespace are not erased to obtain a convenient match.

Static calls, references, import aliases, and test references form a dependency graph. Strongly connected components handle cycles; deterministic priorities produce a reproducible baseline order. Narrated reports may choose a different story order while keeping prerequisites ahead of their dependents. This is a comprehension heuristic, not a proof of an optimal order.

Other languages, newer Python syntax unsupported by your interpreter, and unclassified module-level code remain available as textual hunks. Classes are atomic units. Dynamic dispatch, reflection, unchanged external callers, and whole-program behavior are not fully resolved.

**Identical ASTs are not a proof of identical behavior. Test references are not measured coverage or passing tests.** Diffstory does not execute repository source, run its tests, or issue a merge-safety verdict.

## Add better prose without surrendering the evidence

The CLI creates deterministic baseline explanations. Domain-specific prose is a separate, revision-bound annotation layer. The bundled synthetic walkthrough includes authored annotations to demonstrate the intended reading experience.

```bash
diffstory evidence walkthrough.report.json --out narrative-evidence.json
# Write annotations.json from the documented evidence schema.
diffstory render walkthrough.report.json \
  --annotations annotations.json --out narrated.html
```

Evidence packets contain source code. Send them only to an approved model or environment. The manual annotation workflow makes no provider call and needs no API key. Annotation validation locks revisions and evidence IDs; it cannot establish that every prose claim is true. See [the annotation contract](docs/ARCHITECTURE.md#annotation-handoff).

### Opt in to model narration

The `git` and `github` commands accept `--narrate` to generate candidate annotations. By default, `--provider openai` calls the OpenAI Responses API with [GPT-6.1 Sol](https://developers.openai.com/api/docs/models/gpt-6.1-sol). This is a separate API integration; a ChatGPT subscription does not supply an API key. Set `OPENAI_API_KEY` with your credential manager, or select another environment variable with `--api-key-env`. Diffstory never accepts the key as a command argument or writes it to generated files.

```bash
diffstory github 'OWNER/REPO#NUMBER' \
  --narrate \
  --out walkthrough.html
```

To use your signed-in Codex CLI account instead, install and sign in to the
[Codex CLI](https://developers.openai.com/codex/cli) with `codex login`, then
select `--provider codex`:

```bash
diffstory github 'OWNER/REPO#NUMBER' \
  --narrate \
  --provider codex \
  --out walkthrough.html
```

This route runs `codex exec` with the Codex CLI's saved sign-in and default
model. You can pass `--model MODEL_ID` for a supported Codex model. Known
overrides (`gpt-6-astra`, `gpt-6.1-sol`, `gpt-6-luna`, and `gpt-5.3-codex`)
use their published context and output capacities; the CLI default is packed
against a conservative 400,000-token context bound. Unknown overrides are
rejected before source transfer until capacity metadata is available. This
route does not need an OpenAI API key; access and usage follow the signed-in
ChatGPT or workspace account. The CLI receives only the prepared evidence
request, runs from a temporary directory with read-only access, and has local
shell, browser, app, and plugin tools disabled. See the [Codex non-interactive
mode documentation](https://developers.openai.com/codex/non-interactive-mode)
for how `codex exec` uses saved authentication and structured output.

Codex CLI does not expose a per-call output-token cap. Diffstory lets Codex
complete each request, validates its structured response, and records the
input, cached input, and output token counts reported by the CLI.

Before the first model request, Diffstory displays the destination, PR revisions, source scope, and planned request count. It asks for confirmation. Use `--yes` only when you have already approved that transfer, such as in an automation job. Without `--narrate`, GitHub ingestion still reads the PR over the network but no model request is made.

Narration packs bounded evidence chunks to fit the selected model's context and uses multi-level summaries for larger changes. Long source lines are split into slices without dropping source text. After a successful run, the CLI prints provider-reported input tokens, cached input tokens, and output tokens; these totals are also stored in the annotations and report. `--max-source-bytes` can lower the hard 64 MB cap on captured source across both revisions. Excess source is rejected before narration, and the GitHub reader stops fetching files once the configured cap is exceeded.

In the default v2 architecture, there are no run-wide token, call-count, or elapsed-time ceilings. Each individual provider call has a 15-minute timeout to prevent a stalled request from hanging indefinitely; a run may make as many calls as its source requires.

In default v2, independent narration tasks run concurrently. The default limit is four active tasks. Set `--concurrency N` to a positive integer to change the limit.

In default v2, Diffstory saves validated narration results as checkpoints by default. The default directory is `Path(out).with_suffix('.narration-cache')`; for `walkthrough.html`, this is `walkthrough.narration-cache/`. Use `--cache-dir PATH` to choose another directory or `--no-cache` to disable persistent checkpoints. These options are mutually exclusive.

A default v2 checkpoint is reused only when the run matches: the exact base and head revisions, report source and pull-request context, provider and model, prepared prompt and schema, output reservation, and validation version. Diffstory excludes only the report's `captured_at` timestamp from this identity. It validates each saved result again before reuse. Progress reports the current stage, active calls, completed and planned tasks, reused results, provider calls, and elapsed time. A reused-result count is separate from current provider calls and provider-reported cached input tokens.

### Opt-in walkthrough architecture

The default architecture remains v2. To use the bounded, flat walkthrough
architecture, select it explicitly:

```bash
diffstory github 'OWNER/REPO#NUMBER' \
  --narrate \
  --architecture walkthrough \
  --out walkthrough.html
```

Walkthrough mode allows at most 12 provider attempts across writers and the
editor, including retries; each request can be attempted at most twice. It
allows 1 to 4 concurrent calls, rejects higher concurrency, and applies one
240-second deadline to local preparation and generation. Interactive
confirmation wait does not use the budget. The shared 1 MiB response-byte
limit counts canonical JSON response bytes before validation, including
responses that fail validation and accepted cached results. The separate 1 MiB
saved-story limit remains in force.

If a request or limit fails, walkthrough mode stops submitting new work, lets
active calls finish, and does not write new output files. It retains
attempts, measured usage, response bytes, and accepted-result digests in its
durable run state. The default state directory is
`Path(out).with_suffix('.narration-state')`; set `--state-dir PATH` to choose a
different one. Resume with the same state directory to retain the original
attempts and deadline. A resume does not start a new 240-second period. Use
`--fresh-state NEW_DIRECTORY` only to start a new budget in a new directory;
the old state stays in place. Incompatible run-state versions are rejected;
use a new state directory instead of reusing an unknown accounting format.

Payload caching is separate from run state. The default cache directory is
`Path(out).with_suffix('.narration-cache')`; use `--cache-dir PATH` to choose
another directory or `--no-cache` to disable cached result payloads. Disabling
the cache does not disable durable attempt and deadline state. Cache results
are revalidated before reuse. If a durable run already accepted a result for a
request, resume accepts only that result digest; a changed cache result is
rejected. A cache hit counts against the response-byte budget once per run, and
reusing that exact result on resume does not charge it again. With Codex's
unresolved default model, the cache is disabled; pass an explicit `--model` to
enable it.

In default v2, the CLI proceeds without persistent checkpoints when `--provider codex` uses the unresolved default model. Set `--model MODEL_ID` to enable them. Diffstory rejects `--cache-dir` when Codex has no explicit model. If a stage fails, default v2 stops submitting tasks, waits for active calls to finish, and keeps valid checkpoints so a later run can reuse them.

Successful runs write HTML, `.report.json`, and candidate `.annotations.json` files. Each generated passage is tied to a change ID and checked against its group and source range. The generated report preserves its provider/model, revisions, measured usage, and chunk coverage, and displays a visible **model-generated · unverified** warning. It does not validate whether prose is true.

Generated narration uses ASD-STE100 writing principles as a strong guide while keeping a natural reading rhythm. A required **Before the code** section introduces the whole change before the code walkthrough. It can use several short paragraphs and one compact visual. The model selects the view when the evidence supports it. The reader draws a two-path decision as a flow chart with arrows and outcome boxes; pseudocode and other sketches stay in a text block. The preamble can reach about one page for a larger change. Prose has a 4,000-character limit; complete fenced sketches have a separate 4,000-character allowance, including their fences.

The OpenAI API request sets `store: false`, but this is not a zero-retention promise. OpenAI's API data controls, abuse-monitoring retention, organization settings, and applicable exceptions govern what the API retains; check the current [OpenAI API data controls](https://developers.openai.com/api/docs/guides/your-data) before sending confidential source. The API calls are documented in the [Responses API](https://developers.openai.com/api/docs/guides/structured-outputs) documentation. Codex CLI runs follow the signed-in ChatGPT or workspace account's current data and retention controls.

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
uv build
uv run --locked twine check dist/*
```

For a system Chromium installation, set `DIFFSTORY_CHROMIUM` to its executable path. Browser tests exercise rendering, navigation, expansion, responsive layout, and offline behavior. They do not claim tests passed in the analyzed repository. See [verification](docs/VERIFICATION.md) for the checks actually run on this release.

## Public examples and privacy

Only original synthetic source belongs in public fixtures. Generated review files are ignored by default, with explicit exceptions for `examples/demo.*`. `scripts/check_release.py` checks the public file set for known private-fixture identifiers and common credential patterns before publishing. It is a defense-in-depth check, not a comprehensive secret detector.

The analyzer is read-only and the reader does not contact external services. Explicitly clicked source links open GitHub. The release helper in `scripts/publish.sh` is a separate, opt-in GitHub write operation; it is never called by the analyzer.

## Status and license

**0.3.0 · alpha.** Usable for local review and experimentation; not a substitute for tests or human review. Automatic model narration, whole-repository indexing, statement-level extraction matching, and verified test-artifact ingestion remain future work.

Software and original synthetic examples: [MIT](LICENSE). Epistemic AI trademarks and the supplied logo have separate terms in [NOTICE](NOTICE) and [Branding](docs/BRANDING.md). No font files are bundled.
