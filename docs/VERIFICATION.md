# Verification for 0.3.0

Checks executed locally on September 24, 2026, using Python 3.13.5,
Node.js 22.16.0, Playwright 1.57.0, and system Chromium.

| Check | Actual result |
|---|---|
| Python unit and integration suite | 81 tests passed |
| Headless browser suite | 69 assertions passed |
| JavaScript parse check | `node --check` passed |
| GitHub configuration YAML | All YAML parsed |
| Public-input guard | Passed on the prepared public file set |
| Publishing helper | Tested against a fake GitHub CLI and real local Git repositories; create/push/remote-SHA verification succeeded without a network request |
| Helper retry | Refuses an existing local Git history |
| Packaging | Wheel and source distribution built with the installed setuptools backend |
| Fresh wheel install | Offline install in an isolated virtual environment succeeded; `pip check` passed |
| Installed rendering | Packaged CLI reproduced the public demo HTML byte-for-byte |
| Package contents | HTML/CSS/JavaScript/SVG assets and license metadata present; no font files or private demo inputs included |

The browser checks cover desktop, 390px mobile, and 320px narrow layouts;
continuous prose/code sections; keyboard contents navigation; inline code/diff
switching; lazy rendering and incremental expansion; all 802 lines of a large
synthetic definition; 800 added and 800 removed diff rows; source/test disclosures;
revision-bound notes and invalid-import rejection. No page JavaScript errors or
runtime network requests occurred in the demo test. Screenshots are in `reader/`.

The installed CLI and source checkout use the same parser, templates and embedded
brand asset. The release ZIP contains the reviewed source file set and a
SHA-256 manifest. Run `python scripts/check_release.py --manifest` to verify it.

## Not claimed by these results

The repository has not been created or pushed by this preparation session.
The available GitHub connector has no repository-creation operation, and the
local environment has no authenticated GitHub CLI. The publication helper is
ready for a user-authorized local run.

GitHub-hosted CI, other Python versions in the CI matrix, public package-registry
publication, and a comprehensive security audit have not been performed here.
The normal release instructions include `python -m build` and `twine check`;
the build frontend and Twine were not available in this offline environment, so
packaging was tested through the installed build backend and direct metadata /
installation checks instead.

Tests above validate Diffstory itself, not the tests in a repository being
analyzed. The synthetic example's test definitions are source evidence; its
report correctly records zero test runs verified by the analyzer.

## Human review for generated narration

The model-backed generation path is opt-in and is not exercised by the offline
test suite. Before accepting a provider, model, or prompt change, generate a
walkthrough with each changed provider from the synthetic fixture and from a
source snapshot approved for that use. Review every generated passage beside
its rendered code. Score each dimension from 0 to 2: 0 means incorrect or
missing, 1 means partly useful or too vague, and 2 means accurate and useful.

| Dimension | Review question |
|---|---|
| Evidence fidelity | Does each prose claim follow from its cited change and visible source range? |
| Coverage | Does every expected group and change have a useful explanation, without unsupported steps? |
| Coherence | Do the opening, sections, transitions, and closing form a clear walkthrough? |
| Uncertainty | Does the prose distinguish static references from executed tests or verified behavior? |
| Reading order | Do explanations follow the compiler's dependency order and explain why the next group follows? |

Reject a candidate for any invented test result, unsupported behavior claim, or
prose citation that does not support the claim, regardless of the total score.
This review assesses readability and evidence fidelity; it does not certify
the model's prose as true.
