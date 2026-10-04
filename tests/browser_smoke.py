"""Literate reader browser tests. Optional dev dependencies: Playwright + Chromium.

Uses set_content because some managed environments disallow file:// navigation.
The offline HTML has no runtime network or framework dependency.
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from playwright.sync_api import expect
from playwright.sync_api import sync_playwright

if TYPE_CHECKING:
    from collections.abc import Callable

    from playwright.sync_api import Browser

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# The package path is prepared first so the file also runs as a direct script.
from diffstory.analysis import (  # noqa: E402, I001
    apply_annotations,
    compile_snapshot,
)
from diffstory.render import render  # noqa: E402


def file_evidence_report() -> dict:
    """Build a reader fixture with confirmed, unresolved, and text-only files.

    Returns:
        A compiled report with unsafe path text and an unrelated source warning.
    """
    rows = [
        (
            "config.py",
            "def keep():\n    return 1\n",
            "def keep():\n    return 1\n\ndef alpha():\n    return 2\n\ndef beta():\n    return 3\n",
            None,
        ),
        ("removed.py", "def removed():\n    return 4\n", None, None),
        (
            "unresolved_head.py",
            None,
            "def maybe_new():\n    return 5\n",
            "not_supplied",
        ),
        (
            "unresolved_base.py",
            "def maybe_old():\n    return 6\n",
            None,
            "size_limit",
        ),
        (
            "tests/test_status.py",
            None,
            "def test_new_contract():\n    assert True\n",
            None,
        ),
        (
            "source<svg onload=alert(1)>.py",
            None,
            "def escaped_basis():\n    return 7\n",
            None,
        ),
        ("README.md", None, "Notes stay separate from limits.\n", None),
    ]
    fragments, evidence = [], []
    for path, base_text, head_text, unavailable_reason in rows:
        pair = {}
        for side, source in (("base", base_text), ("head", head_text)):
            if source is not None:
                pair[side] = {
                    "path": path,
                    "state": "supplied",
                    "coverage": "full",
                }
                fragments.append(
                    {
                        "path": path,
                        "side": side,
                        "start_line": 1,
                        "scope": "full",
                        "text": source,
                    }
                )
            elif unavailable_reason:
                pair[side] = {
                    "path": path,
                    "state": "unavailable",
                    "reason": unavailable_reason,
                }
            else:
                pair[side] = {"path": path, "state": "absent"}
        evidence.append(pair)
    return compile_snapshot(
        {
            "schema": "diffstory.snapshot.v1",
            "meta": {
                "title": "File evidence smoke page",
                "base_sha": "c" * 40,
                "head_sha": "d" * 40,
                "scope": "changed files",
                "changed_files": len(rows),
                "input": "synthetic example",
            },
            "fragments": fragments,
            "file_evidence": evidence,
            "warnings": ["A counterpart file could not be read."],
        }
    )


def check_file_evidence(
    browser: Browser, check: Callable, smoke_report: dict
) -> None:
    """Check that classification and basis survive rendering and source expansion.

    Args:
        browser: Open Chromium instance.
        check: Assertion callback that counts successful checks.
        smoke_report: Compiled file-evidence fixture.
    """
    smoke_changes = {
        (change.get("after") or change.get("before"))["name"]: change
        for change in smoke_report["changes"]
    }
    expected_labels = {
        "alpha": ("added", "Added to file"),
        "beta": ("added", "Added to file"),
        "removed": ("removed", "Removed from file"),
        "maybe_new": ("observed_head", "Addition unresolved"),
        "maybe_old": ("observed_base", "Removal unresolved"),
        "test_new_contract": ("added", "Added to file"),
        "escaped_basis": ("added", "Added to file"),
    }
    check(
        all(
            smoke_changes[name]["kind"] == kind
            for name, (kind, _) in expected_labels.items()
        ),
        "synthetic report has each file status, including test source",
    )
    proof_page = browser.new_page(viewport={"width": 1280, "height": 900})
    proof_page.set_content(render(smoke_report), wait_until="domcontentloaded")
    for name, (kind, label) in expected_labels.items():
        change_record = smoke_changes[name]
        figure = proof_page.locator(
            f'[data-changes~="{change_record["id"]}"]'
        ).first
        check(
            figure.locator(".classification").inner_text() == label,
            f"reader label for {name} ({kind})",
        )
        expected_basis = change_record["basis"]
        check(
            figure.locator(".code-basis").inner_text() == expected_basis,
            f"basis appears below caption for {name}",
        )
        if name == "alpha":
            figure.evaluate(
                "(el) => el.querySelector('[data-load-preview]')?.click()"
            )
            expect(figure).to_have_attribute("data-loaded", "true")
            check(
                figure.locator(".code-line.add .code-text")
                .first.inner_text()
                .find("def alpha")
                >= 0,
                "added source row appears in the default passage",
            )
            check(
                figure.locator(".code-basis").inner_text() == expected_basis,
                "basis remains visible after lazy code loading",
            )
    escaped_figure = proof_page.locator(
        f'[data-changes~="{smoke_changes["escaped_basis"]["id"]}"]'
    )
    check(
        escaped_figure.locator(".code-basis script, .code-basis svg").count()
        == 0,
        "script-like basis text stays escaped",
    )
    proof_page.locator("#evidence summary").click(no_wait_after=True)
    expect(proof_page.locator("#evidence")).to_contain_text("Analysis notes")
    expect(proof_page.locator("#evidence")).to_contain_text(
        "Limits of this input"
    )
    proof_page.set_viewport_size({"width": 320, "height": 700})
    check(
        proof_page.evaluate(
            "document.documentElement.scrollWidth <= innerWidth"
        ),
        "long basis fits a narrow viewport",
    )

    proof_page.close()


def check_authored_preamble(
    browser: Browser, check: Callable, smoke_report: dict
) -> None:
    """Check that authored prose and sketches retain text without executing markup.

    Args:
        browser: Open Chromium instance.
        check: Assertion callback that counts successful checks.
        smoke_report: Compiled file-evidence fixture used for source passages.
    """
    smoke_changes = {
        (change.get("after") or change.get("before"))["name"]: change
        for change in smoke_report["changes"]
    }
    config_group = next(
        group
        for group in smoke_report["groups"]
        if set(group["change_ids"])
        & {smoke_changes["alpha"]["id"], smoke_changes["beta"]["id"]}
    )
    authored = apply_annotations(
        smoke_report,
        {
            "schema": "diffstory.annotations.v1",
            "base_sha": smoke_report["meta"]["base_sha"],
            "head_sha": smoke_report["meta"]["head_sha"],
            "document": {
                "preamble": (
                    "This change connects file evidence to the reader.\n"
                    "Map: report → sections → source.\n\n"
                    "The reader keeps the explanation near its source.\n\n"
                    "```text\n"
                    "Evidence <svg onload=alert(1)>\n"
                    "  ↓\n"
                    "Reader\n"
                    "```\n\n"
                    "The sections follow this overview."
                ),
            },
            "steps": [
                {
                    "group_id": config_group["id"],
                    "title": "Read the supplied file changes",
                    "evidence_change_ids": config_group["change_ids"],
                    "passages": [
                        {
                            "text": "Authored explanation stays visible beside the source.",
                            "change_ids": [smoke_changes["alpha"]["id"]],
                            "view": "diff",
                        },
                    ],
                },
            ],
        },
    )
    authored_page = browser.new_page(viewport={"width": 1280, "height": 900})
    authored_page.set_content(render(authored), wait_until="domcontentloaded")
    preamble = authored_page.locator(".preamble")
    check(preamble.count() == 1, "optional preamble is visible")
    check(
        preamble.locator("h2").text_content() == "Before the code",
        "preamble has its section heading",
    )
    preamble_text = preamble.locator(".preamble-body").inner_text()
    check(
        "This change connects file evidence to the reader.\n"
        "Map: report → sections → source." in preamble_text,
        "preamble keeps authored line breaks",
    )
    check(
        preamble.locator(".preamble-body p").count() == 3,
        "preamble separates its prose into paragraphs",
    )
    check(
        preamble.locator(".preamble-sketch pre").inner_text()
        == "Evidence <svg onload=alert(1)>\n  ↓\nReader",
        "preamble shows its conceptual sketch",
    )
    check(
        preamble.locator(".preamble-sketch code").evaluate(
            "(el) => getComputedStyle(el).backgroundColor"
        )
        == "rgba(0, 0, 0, 0)",
        "conceptual sketch uses plain diagram styling",
    )
    check(
        preamble.locator("svg, script").count() == 0,
        "preamble markup stays escaped",
    )
    check(
        preamble.locator(".preamble-body p").first.evaluate(
            "(el) => getComputedStyle(el).whiteSpace"
        )
        == "pre-line",
        "preamble preserves line breaks in the reader",
    )
    check(
        preamble.evaluate(
            "(el) => Boolean(el.compareDocumentPosition("
            "document.querySelector('.code-block')) & "
            "Node.DOCUMENT_POSITION_FOLLOWING)"
        ),
        "preamble appears before the first source block",
    )
    authored_section = authored_page.locator(
        f'[data-story-section="{config_group["id"]}"]'
    )
    check(
        "Authored explanation stays visible beside the source."
        in authored_section.locator(".prose").inner_text(),
        "authored passage remains intact",
    )
    alpha_figure = authored_section.locator(
        f'[data-changes~="{smoke_changes["alpha"]["id"]}"]'
    )
    check(
        alpha_figure.locator(".code-basis").inner_text()
        == smoke_changes["alpha"]["basis"],
        "basis appears beside an authored passage",
    )
    beta = smoke_changes["beta"]
    support = authored_section.locator("[data-extra]")
    check(support.count() == 1, "uncited changes remain supporting source")
    support.locator("summary").click(no_wait_after=True)
    expect(support).to_have_attribute("data-ready", "true")
    beta_figure = support.locator(f'[data-changes~="{beta["id"]}"]')
    check(
        beta_figure.locator(".code-basis").inner_text() == beta["basis"],
        "basis appears for a supporting change",
    )
    beta_figure.evaluate(
        "(el) => el.querySelector('[data-load-preview]')?.click()"
    )
    expect(beta_figure).to_have_attribute("data-loaded", "true")
    check(
        beta_figure.locator(".code-basis").inner_text() == beta["basis"],
        "supporting basis survives lazy loading",
    )
    authored_page.close()


def check_decision_flowchart(
    browser: Browser, check: Callable, smoke_report: dict
) -> None:
    """Check decision formatting, escaped labels, mobile fit, and text fallback.

    Args:
        browser: Open Chromium instance.
        check: Assertion callback that counts successful checks.
        smoke_report: Valid report used to host the conceptual sketches.
    """
    authored = dict(smoke_report)
    authored["document"] = {}
    authored["document"]["preamble"] = (
        "This change adds a second credential source.\n\n"
        "```text\n"
        "Credential lookup\n"
        "  ├─ Environment token <svg onload=alert(1)> → use first\n"
        "  └─ Empty token → try client → use its credential\n"
        "```"
    )
    diagram_page = browser.new_page(viewport={"width": 390, "height": 844})
    diagram_page.set_content(render(authored), wait_until="domcontentloaded")
    diagram = diagram_page.locator(".preamble-flowchart")
    check(diagram.count() == 1, "decision sketch becomes a flow chart")
    check(
        diagram.locator(".flowchart-decision polygon").count() == 1
        and diagram.locator(".flowchart-connectors").count() == 1,
        "flow chart has a decision node and directional connectors",
    )
    check(
        "Environment token <svg onload=alert(1)>"
        in diagram.locator(".flowchart-condition").first.inner_text(),
        "flow chart keeps branch conditions",
    )
    check(
        "try client → use its credential"
        in diagram.locator(".flowchart-outcome").last.inner_text(),
        "flow chart keeps complete outcomes",
    )
    check(
        diagram.locator(
            "svg[onload], script, .flowchart-condition svg"
        ).count()
        == 0,
        "flow chart labels stay escaped",
    )
    check(
        not diagram_page.evaluate(
            "document.documentElement.scrollWidth > innerWidth"
        ),
        "flow chart does not overflow a mobile page",
    )
    text_sketches = [
        (
            "reversed branch markers",
            "Credential lookup\n  └─ Empty token → try client\n  ├─ Environment token → use first",
        ),
        (
            "empty first condition",
            "Credential lookup\n  ├─ → first → second\n  └─ Empty token → try client",
        ),
        (
            "empty outcome",
            "Credential lookup\n  ├─ Token → \n  └─ Empty token → try client",
        ),
        (
            "long decision label",
            "x" * 37
            + "\n  ├─ Token → use first\n  └─ Empty token → try client",
        ),
        (
            "long outcome",
            "Credential lookup\n  ├─ Token → "
            + "x" * 121
            + "\n  └─ Empty token → try client",
        ),
    ]
    for name, sketch in text_sketches:
        authored["document"]["preamble"] = (
            f"Choose a path.\n\n```text\n{sketch}\n```"
        )
        diagram_page.set_content(
            render(authored), wait_until="domcontentloaded"
        )
        check(
            diagram_page.locator(".preamble-flowchart").count() == 0
            and diagram_page.locator(".preamble-sketch pre").inner_text()
            == sketch,
            f"{name} stays in the text sketch",
        )
    diagram_page.close()


def main() -> None:  # noqa: PLR0915  # Sequential reader interactions share page state.
    """Exercise the offline reader in Chromium and capture browser screenshots.

    The smoke contract covers narrative structure, file labels and basis,
    accessible navigation, source expansion, responsive layouts, safe
    embedding, and absence of runtime network requests.

    Side Effects:
        Launches Playwright Chromium and writes screenshots to a temporary
        directory. Raises an assertion or browser error on failure.
    """
    html = (ROOT / "examples/demo.html").read_text(encoding="utf-8")
    report = json.loads((ROOT / "examples/demo.report.json").read_text())
    screenshot_dir = Path(tempfile.gettempdir()) / "diffstory-browser-smoke"
    screenshot_dir.mkdir(parents=True, exist_ok=True)

    def change(name: str) -> dict:
        """Find a report change by its before or after symbol name.

        Args:
            name: Symbol name to locate.

        Returns:
            The matching report change mapping.
        """
        return next(
            c
            for c in report["changes"]
            if (c.get("after") or c.get("before"))["name"] == name
        )

    parsing_index = next(
        i
        for i, g in enumerate(report["groups"])
        if g["theme"] == "result parsing"
    )
    parsing_group = report["groups"][parsing_index]
    assertions = 0

    def check(value: object, message: str) -> None:
        """Assert one browser invariant and count it for the final summary.

        Args:
            value: Condition that must be true.
            message: Description shown when the assertion fails.

        Side Effects:
            Increments the enclosing assertion count or raises
            ``AssertionError``.
        """
        nonlocal assertions
        assert value, message
        assertions += 1

    with sync_playwright() as p:
        options = {"headless": True}
        executable = os.environ.get("DIFFSTORY_CHROMIUM")
        if executable:
            options["executable_path"] = executable
        browser = p.chromium.launch(**options)
        page = browser.new_page(
            viewport={"width": 1440, "height": 1050}, accept_downloads=True
        )
        page.set_default_timeout(6500)
        errors = []
        requests = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("request", lambda r: requests.append(r.url))
        page.set_content(html, wait_until="domcontentloaded")
        page.wait_for_timeout(150)
        check(
            page.title() == f"Diffstory · {report['meta']['title']} · Reader",
            "title",
        )
        check(
            page.locator(".preamble").count() == 0,
            "legacy reports without a preamble keep the existing layout",
        )
        check(
            page.locator(".story-section").count() == 10,
            "all sections present simultaneously",
        )
        check(
            page.locator('[role="tablist"],.sidebar,.navfooter').count() == 0,
            "no dashboard, tabs or pager",
        )
        check(
            page.locator("#contentsPanel").is_hidden()
            and page.locator("#morePanel").is_hidden(),
            "quiet initial header",
        )
        check(
            page.locator(".code-block[data-loaded]").count()
            < page.locator(".code-block").count(),
            "offscreen code is deferred",
        )
        check(
            page.locator("#story .passage").count() > 15,
            "multiple prose-code passages",
        )
        check(
            page.locator("#story .passage").evaluate_all(
                '(nodes)=>nodes.every(n=>n.querySelector(".prose") && n.querySelector(".code-block"))'
            ),
            "each passage binds prose to code",
        )
        check(
            "Original synthetic example"
            in page.locator(".document-scope").inner_text(),
            "snapshot boundary visible",
        )
        check(
            page.evaluate(
                "document.documentElement.scrollWidth <= innerWidth"
            ),
            "desktop no document overflow",
        )
        check(
            page.locator(".brandmark svg").count() == 1,
            "real Epistemic AI mark embedded",
        )
        check(
            page.locator(".edition").inner_text() == "EPISTEMIC AI",
            "Epistemic AI publisher label",
        )
        check(
            page.evaluate(
                'getComputedStyle(document.documentElement).getPropertyValue("--accent").trim()'
            )
            == "#2563eb",
            "brand palette",
        )

        smoke_report = file_evidence_report()
        check_file_evidence(browser, check, smoke_report)
        check_authored_preamble(browser, check, smoke_report)
        check_decision_flowchart(browser, check, smoke_report)

        page.screenshot(
            path=str(screenshot_dir / "desktop.png"), full_page=False
        )

        # Find is a contents filter, never a replacement of the document.
        page.keyboard.press("/")
        check(page.locator("#contentsPanel").is_visible(), "keyboard contents")
        check(
            page.locator("#search").evaluate(
                "(el)=>el===document.activeElement"
            ),
            "search focus",
        )
        page.locator("#search").fill("parse_tags")
        check(page.locator(".contents-item").count() == 1, "symbol search")
        check(
            page.locator(".story-section").count() == 10,
            "search leaves whole document intact",
        )
        page.locator(".contents-item").click(no_wait_after=True)
        page.wait_for_timeout(150)
        check(
            page.locator("#contentsPanel").is_hidden(),
            "contents closes after jump",
        )
        check(
            parsing_group["id"] in page.evaluate("location.hash"),
            "section deep link",
        )
        check(
            f"{parsing_index + 1:02d} / 10"
            in page.locator("#readingPosition").inner_text(),
            "reading position tracks scroll",
        )
        page.screenshot(
            path=str(screenshot_dir / "parsing.png"), full_page=False
        )

        section = page.locator(".story-section").nth(parsing_index)
        code = section.locator(
            f'[data-changes~="{change("parse_tags")["id"]}"]'
        )
        code.scroll_into_view_if_needed()
        page.wait_for_function(
            '(id)=>document.getElementById(id).dataset.loaded === "true"',
            arg=code.get_attribute("id"),
        )
        check(
            code.locator(".code-line.delete").count() > 0
            and code.locator(".code-line.add").count() > 0,
            "paired diff is the default view",
        )
        code.locator("[data-switch]").click(no_wait_after=True)
        check(
            "return tags" in code.inner_text(),
            "definition view shows the implementation",
        )
        check(
            page.locator(".story-section").count() == 10,
            "changing code view does not navigate",
        )
        code.locator("[data-switch]").click(no_wait_after=True)
        check(
            code.locator(".code-line.delete").count() > 0
            and code.locator(".code-line.add").count() > 0,
            "return to paired diff",
        )

        large = (
            page.locator(".story-section").first.locator(".code-block").first
        )
        large.scroll_into_view_if_needed()
        page.wait_for_timeout(100)
        initial = large.locator(".code-line").count()
        check(initial <= 24, "long diff uses bounded preview")
        check(
            "Load full diff" in large.inner_text(),
            "explicit in-place diff expansion",
        )
        large.locator("[data-expand]").click(no_wait_after=True)
        expected = sum(
            len(h["rows"]) for h in change("construct_query")["hunks"]
        )
        check(
            large.locator(".code-line").count() == expected,
            "full diff expands without lost lines",
        )
        check(
            large.locator(".code-region").evaluate(
                "(el)=>el.scrollHeight<=el.clientHeight+1"
            ),
            "no nested vertical code scrollbar",
        )
        large.locator("[data-collapse]").click(no_wait_after=True)
        check(
            large.locator(".code-line").count() == initial,
            "collapse restores source-focused excerpt",
        )
        page.locator("#contentsBtn").click(no_wait_after=True)
        page.keyboard.press("Escape")
        check(
            page.locator("#contentsPanel").is_hidden(),
            "escape closes contents",
        )
        check(
            page.locator("#contentsBtn").evaluate(
                "(el)=>el===document.activeElement"
            ),
            "escape restores focus",
        )

        # Every semantic unit remains reachable, even when not central to the prose.
        for detail in page.locator("[data-extra]").all():
            detail.evaluate("(el)=>el.open=true")
        page.wait_for_timeout(120)
        represented = page.locator("[data-changes]").evaluate_all(
            '(nodes)=>[...new Set(nodes.flatMap(n=>n.dataset.changes.split(" ").filter(Boolean)))]'
        )
        check(
            set(represented) == {c["id"] for c in report["changes"]},
            "all units retained in main or supporting source",
        )

        repeated_report = json.loads(json.dumps(report))
        repeated_group = repeated_report["groups"][parsing_index]
        repeated_passage = next(
            passage
            for passage in repeated_group["narrative"]["passages"]
            if len(passage["change_ids"]) == 1
        )
        repeated_change_id = repeated_passage["change_ids"][0]
        repeated_group["narrative"]["passages"].append(dict(repeated_passage))
        expected_occurrences = sum(
            repeated_change_id in passage["change_ids"]
            for passage in repeated_group["narrative"]["passages"]
        )
        repeated_page = browser.new_page()
        repeated_page.set_content(
            render(repeated_report), wait_until="domcontentloaded"
        )
        repeated_blocks = (
            repeated_page.locator(".story-section")
            .nth(parsing_index)
            .locator(f'[data-changes~="{repeated_change_id}"]')
            .count()
        )
        check(
            repeated_blocks == expected_occurrences,
            "repeated citations render their code for every passage",
        )
        repeated_page.close()

        page.locator("#moreBtn").click(no_wait_after=True)
        page.locator('#morePanel a[href="#source-hunks"]').click(
            no_wait_after=True
        )
        expect(page.locator("#source-hunks")).to_have_attribute("open", "")
        check(
            page.locator(".raw-file").count() == 8,
            "original hunks grouped by eight unique paths",
        )
        page.locator(".raw-file").first.locator("summary").first.click(
            no_wait_after=True
        )
        expect(
            page.locator(".raw-file").first.locator(".code-block").first
        ).to_be_visible()
        check(
            page.locator(".raw-file").first.locator(".code-block").count() > 0,
            "raw hunks load inline",
        )
        page.locator("#moreBtn").click(no_wait_after=True)
        page.locator('#morePanel a[href="#evidence"]').click(
            no_wait_after=True
        )
        check(
            page.locator("#evidence").get_attribute("open") is not None,
            "evidence lives in the same document",
        )
        check(
            "rendered lazily from that embedded data"
            in page.locator("#evidence").inner_text(),
            "offline lazy-rendering disclosure",
        )
        check(
            report["meta"]["head_sha"]
            in page.locator("#evidence").inner_text(),
            "pinned head available",
        )
        check(
            page.locator(".story-section").count() == 10,
            "appendices do not replace narrative",
        )

        # Responsive layout, including expanded source.
        page.set_viewport_size({"width": 390, "height": 844})
        page.evaluate("scrollTo(0,0)")
        page.wait_for_timeout(100)
        page.screenshot(
            path=str(screenshot_dir / "mobile.png"), full_page=False
        )
        check(
            page.evaluate(
                "document.documentElement.scrollWidth <= innerWidth"
            ),
            "mobile top no overflow",
        )
        for index in (0, 1, 3, 8, 9):
            page.locator(".story-section").nth(index).evaluate(
                '(el)=>el.scrollIntoView({block:"start"})'
            )
            page.wait_for_timeout(80)
            check(
                page.evaluate(
                    "document.documentElement.scrollWidth <= innerWidth"
                ),
                f"mobile section {index + 1} no overflow",
            )
        page.set_viewport_size({"width": 320, "height": 700})
        check(
            page.evaluate(
                "document.documentElement.scrollWidth <= innerWidth"
            ),
            "narrow 320px no overflow",
        )
        check(not errors, "JavaScript errors: " + str(errors))
        check(not requests, "Unexpected network requests: " + str(requests))

        # Stress a genuinely oversized diff, not only the small example snapshot.
        source = (
            "def construct_large_query():\n"
            + "".join(f"    value_{i} = {i}\n" for i in range(800))
            + "    return value_799\n"
        )
        new = source.replace("value_799 = 799", "value_799 = 800")
        snapshot = {
            "schema": "diffstory.snapshot.v1",
            "meta": {
                "base_sha": "a" * 40,
                "head_sha": "b" * 40,
                "changed_files": 1,
                "scope": "complete",
            },
            "fragments": [
                {
                    "side": "base",
                    "path": "large.py",
                    "text": source,
                    "start_line": 1,
                    "scope": "complete",
                },
                {
                    "side": "head",
                    "path": "large.py",
                    "text": new,
                    "start_line": 1,
                    "scope": "complete",
                },
            ],
        }
        stress = browser.new_page(viewport={"width": 1280, "height": 950})
        stress.set_content(
            render(compile_snapshot(snapshot)), wait_until="domcontentloaded"
        )
        block = stress.locator(".code-block").first
        block.scroll_into_view_if_needed()
        stress.wait_for_timeout(100)
        if block.locator("[data-switch]").inner_text() == "Read definition":
            block.locator("[data-switch]").click(no_wait_after=True)
        check(
            block.locator(".code-line").count() == 24,
            "oversized definition uses bounded preview",
        )
        block.locator("[data-expand]").click(no_wait_after=True)
        check(
            block.locator(".code-line").count() == 160,
            "first expansion bounded to chunk",
        )
        check(
            "remaining" in block.inner_text(),
            "remaining source is not silently discarded",
        )
        while block.locator("[data-expand]").count():
            block.locator("[data-expand]").click(no_wait_after=True)
        check(
            block.locator(".code-line").count() == 802,
            "all 802 original lines recoverable",
        )
        block.locator("[data-collapse]").click(no_wait_after=True)
        check(
            block.locator(".code-line").count() == 24,
            "huge source collapses back to bounded preview",
        )
        check(
            stress.evaluate(
                "document.documentElement.scrollWidth <= innerWidth"
            ),
            "large source layout bounded",
        )
        stress.close()
        # Oversized paired diff: all changed rows are recovered in bounded batches.
        snapshot["fragments"][1]["text"] = re.sub(
            r"= (\d+)", lambda m: "= " + str(int(m.group(1)) + 1000), source
        )
        big_report = compile_snapshot(snapshot)
        stress = browser.new_page(viewport={"width": 1280, "height": 950})
        stress.set_content(render(big_report), wait_until="domcontentloaded")
        block = stress.locator(".code-block").first
        block.scroll_into_view_if_needed()
        stress.wait_for_timeout(100)
        check(
            block.locator(".code-line").count() <= 24,
            "oversized diff is initially folded",
        )
        check(
            "Load full diff" in block.inner_text(),
            "oversized diff has explicit load action",
        )
        block.locator("[data-expand]").click(no_wait_after=True)
        check(
            block.locator(".code-line").count() <= 160,
            "diff expansion is bounded",
        )
        while block.locator("[data-expand]").count():
            block.locator("[data-expand]").click(no_wait_after=True)
        expected = sum(
            len(h["rows"]) for h in big_report["changes"][0]["hunks"]
        )
        check(
            block.locator(".code-line").count() == expected,
            "every diff row can be loaded",
        )
        check(
            block.locator(".code-line.add").count() == 800,
            "all added lines retained",
        )
        check(
            block.locator(".code-line.delete").count() == 800,
            "all removed lines retained",
        )
        stress.close()
        browser.close()
    print(
        f"Browser reader: {assertions} assertions passed; no page JavaScript errors or network requests in demo."
    )


if __name__ == "__main__":
    main()
