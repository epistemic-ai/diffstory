# Contributing to Diffstory

Keep the reading experience simple and the evidence explicit.

## Local setup

Use Git and [uv](https://docs.astral.sh/uv/). Development uses Python 3.13,
as pinned in `.python-version`; the package supports Python 3.10 and newer.
From a checkout:

```bash
uv sync --locked
uv run --locked python -m unittest discover -s tests -v
uv run --locked python examples/make_demo.py
uv run --locked python scripts/check_release.py
```

`uv sync` creates `.venv`; `uv run` uses it without manual activation. The
`--locked` flag checks that the committed lockfile matches the project
metadata. To change dependencies, use `uv add` or edit `pyproject.toml` and run
`uv lock`, then commit both metadata and `uv.lock`.

For reader changes, install Chromium and run the browser check:

```bash
uv run --locked python -m playwright install chromium
uv run --locked python tests/browser_smoke.py
```

A system browser can be selected through `DIFFSTORY_CHROMIUM`. Screenshots are
written under `docs/reader/` and must show only the synthetic demo.

## Pull requests

Explain the problem and the intended behavior. Add a regression test before changing matching or ingestion rules. Keep analyzer output deterministic and preserve source line numbers. Update the demo and docs when a reader interaction or report contract changes. Do not normalize literals or internal references away to increase matching rates.

Contributions must be code or documentation you have permission to submit under this repository's MIT license. Do not include private code, user notes, credentials, customer data, proprietary diagrams, or private screenshots. Reproduce bugs with small synthetic examples. Generated reports from private repositories must stay outside the public tree.

## Project boundaries

The compiler may parse source; it must not import or execute it. Narrative annotations may explain evidence; they must not rewrite classifications or fabricate test results. Expanding a large diff must not discard lines. Avoid external scripts, fonts, analytics, and framework dependencies in the standalone reader.

Discuss substantial schema changes in an issue before implementation. Backward compatibility for exported review notes depends on schema, exact revisions, and group identifiers.
