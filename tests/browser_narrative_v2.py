"""Exercise authored v2 reports through the public pipeline and Chromium DOM.

Run with: env PYTHONPATH=.:tests .venv/bin/python -B tests/browser_narrative_v2.py
Only local authored fixtures are used. No provider transport or private report
is read. Playwright and Chromium must already be available. Each run writes
reports, screenshots and its assertion summary to a fresh private temp folder.
"""

from __future__ import annotations

import itertools
import json
import os
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

from playwright.sync_api import sync_playwright
from test_narrative import FakeProvider
from test_narrative import two_group_report
from test_narrative_v2 import CASE_AFTER
from test_narrative_v2 import CASE_BEFORE
from test_narrative_v2 import CASE_EFFECT
from test_narrative_v2 import STRIP_AFTER
from test_narrative_v2 import STRIP_BEFORE
from test_narrative_v2 import STRIP_EFFECT
from test_reader import UNSAFE_TEXT
from test_reader import chunked_reader_report
from test_reader import exact_reader_options
from test_reader import exact_reader_provider
from test_reader import exact_reader_snapshot
from test_reader import offline_reader_report
from test_reader import reader_golden_report

from diffstory.analysis import apply_annotations
from diffstory.analysis import compile_snapshot
from diffstory.narrative import Narrator
from diffstory.narrative_manifest import LedgerParameters
from diffstory.narrative_v2 import V2Options
from diffstory.render import render

if TYPE_CHECKING:
    from playwright.sync_api import Browser
    from playwright.sync_api import Locator
    from playwright.sync_api import Page


class Checks:
    """Count passing assertions and leave a precise failure message."""

    def __init__(self) -> None:
        """Start an empty assertion count."""
        self.count = 0

    def __call__(self, value: object, message: str) -> None:
        """Require a condition and count it after success."""
        assert value, message
        self.count += 1


def plain(text: str) -> str:
    """Remove the two supported inline formatting delimiters."""
    return re.sub(
        r"\*\*([^*\n]+)\*\*", r"\1", re.sub(r"`([^`\n]+)`", r"\1", text)
    )


def ref_key(ref: dict) -> str:
    """Index a validated reference without relying on display IDs."""
    return json.dumps(ref, sort_keys=True)


def details_of(typed: dict) -> list[dict]:
    """Return every retained detail, including unassigned context."""
    return [d for s in typed["sections"] for d in s["details"]] + typed[
        "context_details"
    ]


def proof_endpoints(typed: dict) -> dict[str, set[str]]:
    """Independently resolve every inherited branch to exact catalog entries."""
    citations = {ref_key(c["reference"]): c["key"] for c in typed["citations"]}
    observations = {
        (d["unit_key"], o["id"]): o
        for d in details_of(typed)
        for o in d["observations"]
    }
    inherited = {
        ref_key(p["reference"]): p for p in typed["support_provenance"]
    }
    resolved = {}
    for root in inherited:
        stack = [(root, False)]
        while stack:
            key, ready = stack.pop()
            if key in resolved:
                continue
            statement = inherited[key]["statement"]
            children = [
                ref_key(r)
                for r in statement["basis"]
                if r["kind"] == "statement"
            ]
            if not ready:
                stack.append((key, True))
                stack.extend(
                    (child, False)
                    for child in children
                    if child not in resolved
                )
                continue
            refs = list(statement["citations"])
            for claim in statement["claims"]:
                refs.extend(
                    observations[claim["unit_key"], claim["observation_id"]][
                        "citations"
                    ]
                )
            refs.extend(
                r for r in statement["basis"] if r["kind"] == "source_slice"
            )
            resolved[key] = {citations[ref_key(r)] for r in refs}
            for child in children:
                resolved[key].update(resolved[child])
    return {
        inherited[key]["key"]: endpoints for key, endpoints in resolved.items()
    }


def field_texts(node: Locator, field: str) -> list[str]:
    """Read actual DOM text, including CR and terminal whitespace."""
    return node.locator(f'[data-field="{field}"]').evaluate_all(
        "nodes => nodes.map(n => n.textContent)"
    )


def check_details(page: Page, report: dict, check: Checks) -> None:
    """Check every retained detail field and source link in its own visible node."""
    typed = report["narrative_v2"]
    details = details_of(typed)
    catalog = {ref_key(c["reference"]): c["key"] for c in typed["citations"]}
    check(
        page.locator(".v2-detail").count() == len(details),
        "every retained detail appears once",
    )
    for detail in details:
        node = page.locator(f'[data-detail="{detail["unit_key"]}"]')
        check(node.is_visible(), "retained detail is visible")
        for field in ("before", "after", "effect", "uncertainty"):
            check(
                field_texts(node, field)
                == [plain(e[field]) for e in detail["explanations"]],
                f"all detail {field} DOM text",
            )
        for field, family in (
            ("observation", "observations"),
            ("question", "questions"),
        ):
            check(
                field_texts(node, field)
                == [plain(o["text"]) for o in detail[family]],
                f"all detail {family} DOM text",
            )
        limits = [
            limit for e in detail["explanations"] for limit in e["limitations"]
        ]
        check(
            field_texts(node, "reason")
            == [plain(limit["reason"].replace("_", " ")) for limit in limits],
            "readable evidence limits",
        )
        for index, limit in enumerate(limits):
            text = node.locator(".v2-limitation").nth(index).text_content()
            check(
                limit["side"].title() in text and limit["path"] in text,
                "limitation side and path",
            )
            check(
                "does not establish absence or behavior" in text,
                "limitation does not claim absence",
            )
        expected = {
            catalog[ref_key(ref)]
            for family in ("explanations", "observations", "questions")
            for item in detail[family]
            for ref in item["citations"]
        }
        if not any(
            detail[family]
            for family in ("explanations", "observations", "questions")
        ):
            check(
                "Supplied context was retained" in node.text_content(),
                "empty acknowledgement is displayed",
            )
            expected = {
                c["key"]
                for c in typed["citations"]
                if c["reference"].get("unit_key") == detail["unit_key"]
            }
            check(
                expected,
                "acknowledged context has authoritative supplied-source links",
            )
        actual = set(
            node.locator("a[data-citation]").evaluate_all(
                "nodes => nodes.map(n => n.dataset.citation)"
            )
        )
        check(
            actual == expected,
            "detail source links use the exact authoritative catalog",
        )


def check_prose(page: Page, report: dict, check: Checks) -> None:
    """Check document, sections, terms, attribution and all inherited endpoints."""
    typed = report["narrative_v2"]
    statements = []
    terms = []
    for family in ("lead", "preamble", "closing"):
        statements.extend(typed["document"][family])
    terms.extend(typed["document"]["terminology"])
    for section in typed["sections"]:
        prose = section["prose"]
        heading = page.locator(f"#heading-{section['group_id']}")
        check(
            heading.text_content() == plain(prose["title"]["text"]) + "#",
            "typed section title",
        )
        title_support = page.locator(
            f'[id="{section["group_id"]}"] .v2-title-support'
        )
        check(
            title_support.locator(
                ':scope > [data-field="effect"]'
            ).all_text_contents()
            == (
                []
                if prose["title"]["effect"] is None
                else [plain(prose["title"]["effect"])]
            ),
            "heading supported effect is retained",
        )
        check(
            title_support.locator(
                ':scope > [data-field="uncertainty"]'
            ).all_text_contents()
            == [plain(text) for text in prose["title"]["uncertainty"]],
            "heading uncertainty is retained",
        )
        check(
            prose["title"]["kind"].replace("editorial", "Reading guidance")
            in title_support.text_content()
            if prose["title"]["kind"] == "editorial"
            else title_support.is_visible(),
            "heading provenance is visible",
        )
        for family in ("orientation", "synthesis", "transition"):
            statements.extend(prose[family])
        terms.extend(prose["terminology"])
    statements.extend(term["definition"] for term in terms)
    actual = page.locator(".v2-statement").evaluate_all("""nodes => nodes.map(n => ({
        kind: n.dataset.kind,
        text: n.querySelector(':scope > [data-field="text"]').textContent,
        effect: n.querySelector(':scope > .v2-field > [data-field="effect"]')?.textContent ?? null,
        uncertainty: [...n.querySelectorAll(':scope > .v2-field > [data-field="uncertainty"]')].map(x => x.textContent)
    }))""")
    expected = [
        {
            "kind": s["kind"],
            "text": plain(s["text"]),
            "effect": None if s["effect"] is None else plain(s["effect"]),
            "uncertainty": [plain(t) for t in s["uncertainty"]],
        }
        for s in statements
    ]
    check(
        Counter(map(ref_key, actual)) == Counter(map(ref_key, expected)),
        "every document/section/term statement and field is displayed exactly",
    )
    check(
        Counter(page.locator(".v2-term h4").all_text_contents())
        == Counter(plain(t["term"]) for t in terms),
        "all terminology labels",
    )
    for kind, label in (
        ("author_motivation", "Author-reported motivation"),
        ("editorial", "Reading guidance"),
        ("uncertainty", "Uncertainty"),
    ):
        nodes = page.locator(f'.v2-statement[data-kind="{kind}"]')
        check(nodes.count(), f"fixture exercises {kind}")
        for text in nodes.locator(":scope > .v2-label").all_text_contents():
            check(
                label in text and "not implementation evidence" in text,
                f"{kind} attribution is not proof",
            )
    endpoints = proof_endpoints(typed)
    provenance = {p["key"]: p for p in typed["support_provenance"]}
    catalog = {c["key"]: c for c in typed["citations"]}
    check(
        page.locator(".v2-inherited").count(),
        "inherited statements are exercised",
    )
    for node in page.locator(".v2-inherited").all():
        key = node.get_attribute("data-inherited")
        statement = provenance[key]["statement"]
        check(
            node.locator(':scope > [data-field="text"]').text_content()
            == plain(statement["text"]),
            "inherited statement retains its actual supported text",
        )
        check(
            node.locator(
                ':scope > .v2-field > [data-field="effect"]'
            ).all_text_contents()
            == (
                []
                if statement["effect"] is None
                else [plain(statement["effect"])]
            ),
            "inherited effect is retained",
        )
        check(
            node.locator(
                ':scope > .v2-field > [data-field="uncertainty"]'
            ).all_text_contents()
            == [plain(text) for text in statement["uncertainty"]],
            "inherited uncertainty is retained",
        )
        actual = set(
            node.locator("a[data-citation]").evaluate_all(
                "nodes => nodes.map(n => n.dataset.citation)"
            )
        )
        check(
            actual == endpoints[key],
            "all inherited proof branches retain exact endpoints",
        )
        check(
            "Inherited statement" in node.text_content(),
            "inherited statement provenance is explicit",
        )
        for link in node.locator("a[data-citation]").all():
            citation = catalog[link.get_attribute("data-citation")]
            check(
                link.get_attribute("data-consumption")
                == citation["consumption"],
                "inherited endpoint keeps direct/slice consumption",
            )
            label = (
                "Exact source slice"
                if citation["consumption"] == "source_slice"
                else "Direct source"
            )
            check(
                "Inherited · " + label in link.text_content(),
                "readable inherited source label",
            )


def expected_rows(
    fragment: dict, start: int, end: int
) -> list[tuple[int, str]]:
    """Find physical rows from the full original fragment, never a cut prefix."""
    raw = fragment["text"].encode()
    boundaries = [0, *(m.end() for m in re.finditer(rb"\r\n|\r|\n", raw))]
    if boundaries[-1] != len(raw):
        boundaries.append(len(raw))
    return [
        (
            fragment["start_line"] + index,
            raw[max(start, a) : min(end, b)].decode(),
        )
        for index, (a, b) in enumerate(itertools.pairwise(boundaries))
        if a < end and b > start
    ]


def dom_rows(block: Locator) -> list[tuple[int, str]]:
    """Read source strings and gutters from the actual DOM."""
    return [
        tuple(row)
        for row in block.locator(".code-line").evaluate_all(
            "nodes => nodes.map(n => [Number(n.querySelector('.line-no').textContent), n.querySelector('.code-text').textContent])"
        )
    ]


def preview_rows(rows: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """Apply the reader's documented compact preview size to expected rows."""
    return rows if len(rows) <= 34 else rows[:19] + rows[-5:]


def check_sources(  # noqa: C901, PLR0915 # One interaction sequence per immutable citation.
    page: Page, report: dict, check: Checks, *, sample: bool = False
) -> dict:
    """Inspect exact DOM spans; the chunk stress case samples one per fragment."""
    typed = report["narrative_v2"]
    fragments = {f["key"]: f for f in report["source_evidence"]["fragments"]}
    check(
        page.locator(".exact-code .code-line").count() == 0,
        "closed source catalog creates no source DOM eagerly",
    )
    full_seen = set()
    halves = 0
    same_line = {}
    chunked = 0
    empty = 0
    selected = typed["citations"]
    if sample:
        selected = list({c["fragment_key"]: c for c in selected}.values())
    for citation in selected:
        key = citation["key"]
        fragment = fragments[citation["fragment_key"]]
        raw = fragment["text"].encode()
        start, end = citation["byte_start"], citation["byte_end"]
        source = page.locator(f"#source-{key}")
        block = source.locator(".exact-code")
        # Use the reader's own source link navigation where a visible claim uses it.
        links = page.locator(f'a[data-citation="{key}"]')
        if links.count():
            links.first.click()
        else:
            source.locator("summary").click()
        block.scroll_into_view_if_needed()
        block.evaluate(
            "node => node.querySelector('[data-load-preview]')?.click()"
        )
        check(
            block.get_attribute("data-loaded") == "true",
            "source is loaded on inspection",
        )
        for attribute, value in (
            ("fragment", citation["fragment_key"]),
            ("side", citation["side"]),
            ("byte-start", str(start)),
            ("byte-end", str(end)),
        ):
            check(
                block.get_attribute("data-" + attribute) == value,
                "source block exact binding: " + attribute,
            )
        rows = expected_rows(fragment, start, end)
        actual = dom_rows(block)
        check(
            actual == preview_rows(rows),
            "actual DOM excerpt and original physical line numbers",
        )
        if len(rows) <= 34:
            check(
                "".join(text for _, text in actual).encode() == raw[start:end],
                "actual DOM cited span equals intended UTF-8 bytes",
            )
        source_link = block.locator(".code-source-link")
        check(
            source_link.get_attribute("href") == citation["url"],
            "current exact source URL",
        )
        check(
            citation["revision"] in citation["url"],
            "source URL uses current revision",
        )
        check(
            citation["side"].title() in source_link.text_content(),
            "source URL explicitly labels its side",
        )
        check(
            f"L{citation['line_start']}\u2013{citation['line_end']}"
            in source.locator("summary").text_content(),
            "current bound physical lines are displayed",
        )
        check(
            citation["side"] == fragment["side"],
            "base cannot fall back to head",
        )
        for offset in (start, end):
            halves += raw[max(0, offset - 1) : offset + 1] == b"\r\n"
        if citation["line_start"] == citation["line_end"]:
            same_line.setdefault(
                (citation["side"], citation["line_start"]), set()
            ).add((start, end))
        if not raw:
            empty += 1
            check(
                block.locator(".v2-empty-source").is_visible(),
                "empty supplied fragment remains inspectable",
            )
            check(
                "whole file or its counterpart" in block.text_content(),
                "partial empty fragment has no whole-file absence claim",
            )
            check(
                actual == [],
                "empty supplied source DOM has exactly zero source bytes",
            )
        elif citation["fragment_key"] not in full_seen:
            full_seen.add(citation["fragment_key"])
            full = expected_rows(fragment, 0, len(raw))
            block.locator("[data-expand]").click()
            check(
                dom_rows(block) == full[:160],
                "first full-source chunk preserves exact original text",
            )
            if len(full) > 160:
                chunked += 1
                check(
                    block.locator(".code-line").count() == 160,
                    "large source only creates its first DOM chunk",
                )
            loaded = 160
            while block.locator("[data-expand]").count():
                block.locator("[data-expand]").click()
                loaded += 160
                check(
                    dom_rows(block) == full[:loaded],
                    "subsequent source chunk equals exact original prefix",
                )
            check(
                "".join(text for _, text in dom_rows(block)).encode() == raw,
                "actual full-source DOM equals all original UTF-8 bytes",
            )
            block.locator("[data-collapse]").click()
            check(
                dom_rows(block) == preview_rows(rows),
                "collapse returns to exact cited excerpt",
            )
        source.locator("summary").click()
    return {
        "catalog_entries": len(typed["citations"]),
        "inspected_entries": len(selected),
        "full_fragments": len(full_seen),
        "empty_fragments": empty,
        "half_crlf_endpoints": halves,
        "same_line_ranges": max(map(len, same_line.values()), default=0),
        "chunked_fragments": chunked,
    }


def check_page(  # noqa: PLR0913 # Fixture data, browser and evidence destination stay explicit.
    browser: Browser,
    report: dict,
    html: str,
    directory: Path,
    check: Checks,
    *,
    sample_sources: bool = False,
) -> dict:
    """Load an offline fixture and verify display, sources, errors and network."""
    page = browser.new_page(viewport={"width": 1280, "height": 900})
    page.set_default_timeout(6500)
    errors, console_errors, requests = [], [], []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.on(
        "console",
        lambda message: console_errors.append(message.text)
        if message.type == "error"
        else None,
    )
    page.on("request", lambda request: requests.append(request.url))
    page.set_content(html, wait_until="domcontentloaded")
    config = report["generation"]["request_configuration"]
    notice = page.locator(".generation-notice").text_content()
    check(
        f"Provider: {config['name']} · model: {config['model']}." in notice,
        "notice uses validated request configuration labels",
    )
    check(
        "unverified" in notice and "undefined" not in notice,
        "model interpretation stays visibly unverified",
    )
    check_details(page, report, check)
    check_prose(page, report, check)
    text = page.locator("body").text_content()
    for mechanism in (
        STRIP_BEFORE,
        STRIP_AFTER,
        STRIP_EFFECT,
        CASE_BEFORE,
        CASE_AFTER,
        CASE_EFFECT,
    ):
        check(
            mechanism in text,
            "concrete before/after normalization mechanism retained",
        )
    check(
        UNSAFE_TEXT in text,
        "script-shaped prose, Unicode, CRLF and trailing spaces remain exact DOM text",
    )
    screenshot = directory / "reader-top.png"
    page.screenshot(path=str(screenshot))
    sources = check_sources(page, report, check, sample=sample_sources)
    check(
        page.evaluate("typeof window.COMPROMISED === 'undefined'"),
        "source and prose never execute",
    )
    check(
        page.locator(
            ".v2-detail script, .v2-statement script, .exact-code script"
        ).count()
        == 0,
        "escaped markup never creates script DOM",
    )
    page.locator("#evidence > summary").click()
    page.locator('#evidence[data-ready="true"]').wait_for(state="visible")
    check(
        "0 runs verified by this analyzer"
        in page.locator("#evidence").text_content(),
        "compiler test-status authority remains visible",
    )
    page.set_viewport_size({"width": 390, "height": 844})
    page.evaluate("window.scrollTo(0, 0)")
    check(
        page.evaluate(
            "document.documentElement.scrollWidth <= window.innerWidth"
        ),
        "typed prose fits the mobile reading column",
    )
    mobile = directory / "reader-mobile.png"
    page.screenshot(path=str(mobile))
    check(not errors, f"no browser page errors: {errors}")
    check(not console_errors, f"no browser console errors: {console_errors}")
    check(not requests, f"no runtime network requests: {requests}")
    page.close()
    return {
        **sources,
        "page_errors": errors,
        "console_errors": console_errors,
        "network_requests": requests,
        "screenshots": [str(screenshot), str(mobile)],
    }


def check_legacy_notice(
    browser: Browser, directory: Path, check: Checks
) -> None:
    """Retain the established provider/model notice for actual legacy generation."""
    source = two_group_report()
    annotations = Narrator(
        FakeProvider(),
        concurrency=1,
        recovery_dir=directory / "legacy-recovery",
    ).generate(source)
    report = apply_annotations(source, annotations)
    page = browser.new_page()
    errors, requests = [], []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.on("request", lambda request: requests.append(request.url))
    page.set_content(render(report), wait_until="domcontentloaded")
    generation = report["generation"]
    notice = page.locator(".generation-notice").text_content()
    check(
        f"Provider: {generation['provider']} · model: {generation['model']}."
        in notice,
        "legacy provider/model notice remains correct",
    )
    check("unverified" in notice, "legacy narration remains unverified")
    check(
        page.locator(".v2-detail").count() == 0,
        "legacy output never enters the typed v2 branch",
    )
    check(
        not errors and not requests,
        "legacy notice page has no errors or runtime network",
    )
    page.close()


def main() -> None:
    """Run generated, offline-roundtripped and shifted authored reader fixtures."""
    directory = Path(
        tempfile.mkdtemp(prefix="diffstory-browser-narrative-v2-")
    )
    check = Checks()
    golden = offline_reader_report(
        reader_golden_report(), directory / "golden"
    )
    chunked = offline_reader_report(
        chunked_reader_report(),
        directory / "chunked",
        options=V2Options(
            ledger=LedgerParameters(max_span_bytes=4096, max_unit_spans=128)
        ),
    )
    snapshot = exact_reader_snapshot()
    original = offline_reader_report(
        compile_snapshot(snapshot),
        directory / "exact",
        provider=exact_reader_provider(),
        options=exact_reader_options(),
        cache=directory / "cache",
    )
    for fragment in snapshot["fragments"]:
        fragment["start_line"] += 30
    snapshot["meta"].update(base_sha="c" * 40, head_sha="d" * 40)
    shifted = offline_reader_report(
        compile_snapshot(snapshot),
        directory / "shifted",
        provider=exact_reader_provider(),
        options=exact_reader_options(),
        cache=directory / "cache",
    )
    check(
        shifted[2]["fixture_attempts"] == 0,
        "current revision presentation reuses validated results without new model requests",
    )
    check(
        original[0]["generation"]["producers"]
        == shifted[0]["generation"]["producers"],
        "reuse preserves actual original producer identities",
    )
    for old, new in zip(
        original[0]["narrative_v2"]["citations"],
        shifted[0]["narrative_v2"]["citations"],
        strict=True,
    ):
        check(
            old["key"] == new["key"],
            "revision rebinding preserves citation identity",
        )
        check(
            new["line_start"] == old["line_start"] + 30
            and new["line_end"] == old["line_end"] + 30,
            "all current lines shift by exactly thirty",
        )
        check(
            new["url"] != old["url"],
            "current URL changes with source revision",
        )
    results = {}
    with sync_playwright() as playwright:
        options = {"headless": True}
        if os.environ.get("DIFFSTORY_CHROMIUM"):
            options["executable_path"] = os.environ["DIFFSTORY_CHROMIUM"]
        browser = playwright.chromium.launch(**options)
        for name, (report, html, evidence) in (
            ("golden", golden),
            ("exact", original),
            ("shifted", shifted),
            ("chunked", chunked),
        ):
            results[name] = {
                "pipeline": evidence,
                **check_page(
                    browser,
                    report,
                    html,
                    directory / name,
                    check,
                    sample_sources=name == "chunked",
                ),
            }
        check_legacy_notice(browser, directory, check)
        browser.close()
    check(
        results["golden"]["empty_fragments"] == 2,
        "both genuinely empty supplied sides are inspected",
    )
    for name in ("exact", "shifted"):
        check(
            results[name]["half_crlf_endpoints"] > 0,
            "fixture exercises both halves of CRLF",
        )
        check(
            results[name]["same_line_ranges"] > 1,
            "same-line source slices remain distinct",
        )
    check(
        results["chunked"]["chunked_fragments"] == 2,
        "both large source sides are inspected lazily",
    )
    result = {
        "status": "PASS",
        "assertions": check.count,
        "authored_fixture_only": True,
        "live_provider_calls": 0,
        "results": results,
    }
    (directory / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(  # noqa: T201 # Runnable browser harness result.
        json.dumps(
            {
                "assertions": check.count,
                "result": str(directory / "result.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
