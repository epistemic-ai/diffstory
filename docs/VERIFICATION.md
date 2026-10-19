# Verification for 0.3.0

Local verification completed on October 19, 2026. It used Python 3.13.5,
project-pinned Ruff 0.16.10, Twine 7.0.0, Playwright 1.63.0, and Chromium
launched through Playwright.

| Check | Command or method | Result |
|---|---|---|
| Python suite | `.venv/bin/python -B -m unittest discover -s tests -q` | 248 tests passed |
| Ruff lint | `uv run --locked ruff check .` | Passed; all checks passed |
| Ruff format | `uv run --locked ruff format --check .` | Passed; 50 files already formatted |
| JavaScript syntax | `node --check diffstory/assets/app.js` | Passed |
| Public release inputs | `.venv/bin/python -B scripts/check_release.py` | Passed; 76 public files checked |
| Flat reader browser suite | `env PYTHONPATH=.:tests .venv/bin/python -B tests/browser_walkthrough.py` | 26 tests passed |
| Browser smoke suite | `env PYTHONPATH=.:tests .venv/bin/python -B tests/browser_smoke.py` | 143 assertions passed; no JavaScript errors or runtime network requests |
| Demo freshness | Regenerated the demo and compared its four checked-in artifacts with current generator and renderer output | All four matched exactly |
| Offline walkthrough probes | Synthetic fixture provider, without a live model | Demo: 1 call over 7,449 source bytes. Larger synthetic change: 6 calls over 3,623,872 source bytes. |

Ruff 0.16.10 passed lint and format checks. Use these locked project commands
to reproduce the checks:

```sh
uv run --locked ruff --version
uv run --locked ruff check .
uv run --locked ruff format --check .
```

The local lint run used the cached pinned tool; the commands above are the
project-managed reproduction commands.

The browser checks used the public synthetic demo and synthetic test inputs.
Playwright's Chromium launcher needed approved process permission on macOS.
No private report or private browser artifact was inspected. The smoke suite
checks the packaged reader's interaction, layout, safe display, and offline
behavior. These checks do not test source repositories analyzed by Diffstory.

An offline package build passed after the source and README updates. Twine 7.0.0
accepted both the wheel and source archive. The archive audit found 111 members;
all 23 current Python and asset files were byte-identical in both archives, and
the current README and docs matched the source archive. Retired modules and
test/harness files, and private workflow state, were absent. A CLI loaded from
the extracted wheel in a temporary working directory reproduced the public
demo HTML exactly. This was not a fresh pip installation.

No fresh GitHub Actions result for this cleanup is included here. No package
registry publication or comprehensive security audit was performed. The local
release-input check is a defense-in-depth guard, not a complete secret or
security audit.

## Human review for generated walkthroughs

The offline fixtures exercise request structure, citations, coverage, and safe
rendering. They do not establish model prose quality, source semantics, or
provider latency. This cleanup did not change prompts or transports, and no
live provider run was made.

For a future provider or prompt change, review generated passages beside their
cited source. Check that:

- Each claim follows from its cited revision and original source lines.
- Every compiler change and changed region has a useful explanation; related
  concepts may share a passage.
- The story order and transitions help a reader follow the change. Compiler
  groups and dependency edges are hints; validation does not enforce
  prerequisite order.
- The prose separates author-reported intent and static evidence from executed
  tests or verified behavior. It must not invent test results.

Structural validation checks shapes, citations, coverage, and size. It does not
prove that generated prose is true or that repository tests ran.
