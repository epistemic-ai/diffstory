"""Run offline flat-report reader checks with Playwright and Chromium.

Run with: .venv/bin/python tests/browser_walkthrough.py
Synthetic prose checks structure and safe display; it does not measure model
quality or live provider latency.
"""

from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from test_analysis import snap  # noqa: E402
from test_walkthrough import FixtureProvider  # noqa: E402

from diffstory.analysis import compile_snapshot  # noqa: E402
from diffstory.compact_packet import packet_identity  # noqa: E402
from diffstory.render import render  # noqa: E402
from diffstory.walkthrough import WalkthroughNarrator  # noqa: E402


def source_path_index(report: dict) -> dict[str, str]:
    """Index packet source IDs by repository path for synthetic passages."""
    return {
        source["id"]: source["path"] for source in report["packet"]["sources"]
    }


def embed_report_data(html: str, report: dict) -> str:
    """Replace one embedded report payload for reader startup rejection checks."""
    marker = '<script type="application/json" id="report-data">'
    start = html.index(marker) + len(marker)
    end = html.index("</script>", start)
    payload = json.dumps(report, ensure_ascii=False, separators=(",", ":"))
    payload = (
        payload.replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )
    return html[:start] + payload + html[end:]


def passage_for_regions(
    passage_id: str,
    text: str,
    regions: list[dict],
) -> dict:
    """Build a story passage with exact references from selected diff regions."""
    citations = []
    seen = set()
    change_ids = []
    for region in regions:
        for citation in region["citations"]:
            key = (citation["source_id"], citation["start"], citation["end"])
            if key not in seen:
                citations.append(citation)
                seen.add(key)
        for change_id in region["change_ids"]:
            if change_id not in change_ids:
                change_ids.append(change_id)
    return {
        "id": passage_id,
        "text": text,
        "change_ids": change_ids,
        "region_ids": [region["id"] for region in regions],
        "citations": citations,
    }


def feature_report() -> dict:
    """Build a public flat report for reader interactions and review status."""
    before_lines = (
        ["def calculate():"]
        + [f"    value_{index} = {index}" for index in range(1, 161)]
        + ["    return 1"]
    )
    after_lines = list(before_lines)
    for index in [*range(40, 86), *range(110, 132)]:
        after_lines[index] = f"    value_{index} = {index * 10}"
    after_lines[-1] = "    return 2"
    before_large = "\n".join(before_lines) + "\n"
    after_large = "\n".join(after_lines) + "\n"
    before = {
        "large.py": before_large,
        "removed.py": "def old_widget():\n    return 4\n",
        "tests/test_large.py": (
            "def test_calculate():\n    assert calculate() == 1\n"
        ),
        "old_move.py": "def old_name():\n    return 7\n",
    }
    after = {
        "large.py": after_large,
        "new.py": "def new_widget():\n    return 5\n",
        "tests/test_large.py": (
            "def test_calculate():\n    assert calculate() == 2\n"
        ),
        "new_move.py": "# moved source\n\ndef new_name():\n    return 7\n",
    }
    compiled = compile_snapshot(
        snap(
            before,
            after,
            repository="owner/project",
            title="Reader feature fixture",
            description="Author says: 2 tests passed locally.",
        )
    )
    report = WalkthroughNarrator(FixtureProvider()).generate(compiled)
    packet = report["packet"]
    paths = source_path_index(report)
    regions = packet["regions"]
    large_regions = [
        region
        for region in regions
        if paths[region["citations"][0]["source_id"]] == "large.py"
    ]
    large_regions.sort(key=lambda region: region["hunk"])
    main_passages = [
        passage_for_regions(
            'middle/first "<img>',
            "The middle dispatch path changed. Literal <img src=x onerror=window.pwned=1> stays inert; marker-sentence-72 confirms the cited case.",
            large_regions[:1],
        ),
        passage_for_regions(
            "middle-second",
            "The zero-byte guard now rejects an empty response.",
            large_regions[1:],
        ),
    ]

    def citation_at(path: str, side: str, line: int) -> dict:
        source = next(
            source
            for source in packet["sources"]
            if source["path"] == path and source["side"] == side
        )
        return {
            "source_id": source["id"],
            "revision": source["revision"],
            "start": line,
            "end": line,
        }

    main_passages[0]["citations"] = [
        citation_at("large.py", "base", 65),
        citation_at("large.py", "head", 65),
    ]
    main_passages[1]["citations"] = [
        citation_at("large.py", "base", 121),
        citation_at("large.py", "head", 121),
        citation_at("large.py", "base", 162),
        citation_at("large.py", "head", 162),
    ]
    remaining = [region for region in regions if region not in large_regions]
    test_change = next(
        change
        for change in packet["changes"]
        if paths[(change.get("after") or change.get("before"))["source_id"]]
        == "tests/test_large.py"
    )
    test_regions = [
        region
        for region in remaining
        if test_change["id"] in region["change_ids"]
    ]
    test_passage = passage_for_regions(
        "test-source",
        "The changed test expectation records the new return value.",
        test_regions,
    )
    extra_passages = []
    used_regions = {region["id"] for region in large_regions + test_regions}
    main_change_ids = set(main_passages[0]["change_ids"])
    for change in packet["changes"]:
        if (
            change["id"] == test_change["id"]
            or change["id"] in main_change_ids
        ):
            continue
        related = [
            region
            for region in remaining
            if change["id"] in region["change_ids"]
            and region["id"] not in used_regions
        ]
        if not related:
            continue
        extra_passages.append(
            passage_for_regions(
                f"case-{len(extra_passages) + 1}",
                f"The {change['kind']} case remains visible with its own source side.",
                related,
            )
        )
        used_regions.update(region["id"] for region in related)
    unassigned = [
        region for region in regions if region["id"] not in used_regions
    ]
    if unassigned:
        extra_passages.append(
            passage_for_regions(
                "remaining-evidence",
                "The remaining source regions stay available for review.",
                unassigned,
            )
        )

    passages = [*main_passages, test_passage, *extra_passages]
    report["story"]["sections"] = [
        {
            "id": "reader-cases",
            "title": "Reader cases",
            "summary": "Synthetic browser evidence for the offline reader.",
            "passages": passages,
        }
    ]
    report["story"]["quick_summary"] = {
        "before": "The earlier `calculate()` comparison reports the old result.",
        "after": "The later `calculate()` comparison reports the updated result.",
        "passage_ids": [main_passages[0]["id"], main_passages[1]["id"]],
    }
    report["story"]["impact"] = {
        "text": "The revised `calculate()` definition returns a different value.",
        "passage_ids": [main_passages[1]["id"]],
    }
    report["story"]["route"] = [
        {"kind": "mechanism", "passage_ids": [main_passages[0]["id"]]},
        {"kind": "tests", "passage_ids": [test_passage["id"]]},
        {"kind": "compatibility", "passage_ids": [extra_passages[0]["id"]]},
    ]
    report["story"]["author_reported_checks"] = ["2 tests passed locally"]
    report["story"]["review_checks"] = [
        {
            "text": "Confirm the changed test covers the relevant boundary case.",
            "passage_ids": [test_passage["id"], main_passages[1]["id"]],
            "test_ids": [packet["analysis"]["tests"][0]["id"]],
        }
    ]
    assert {change["id"] for change in packet["changes"]} == {
        change_id
        for passage in passages
        for change_id in passage["change_ids"]
    }
    return report


def grouped_report() -> dict:
    """Build one passage that links several edits in the same source file."""
    before = (
        "def alpha():\n    return 1\n\n"
        "def beta():\n    return 2\n\n"
        "def gamma():\n    return 3\n"
    )
    after = (
        before.replace("return 1", "return 11")
        .replace("return 2", "return 22")
        .replace("return 3", "return 33")
    )
    compiled = compile_snapshot(
        snap(
            {"mechanisms.py": before},
            {"mechanisms.py": after},
            repository="owner/project",
        )
    )
    report = WalkthroughNarrator(FixtureProvider()).generate(compiled)
    passage = passage_for_regions(
        "mechanical-changes",
        "These related definitions carry the same file-level mechanism.",
        report["packet"]["regions"],
    )
    report["story"]["sections"] = [
        {
            "id": "grouped",
            "title": "Related definitions",
            "summary": "A compact same-file fixture.",
            "passages": [passage],
        }
    ]
    report["story"]["quick_summary"]["passage_ids"] = [passage["id"]]
    report["story"]["route"] = [
        {"kind": "mechanism", "passage_ids": [passage["id"]]}
    ]
    report["story"]["impact"] = None
    report["story"]["review_checks"] = []
    return report


def mixed_group_report() -> dict:
    """Keep routine same-file edits beside their shared partial-source caveat."""
    path = "mechanisms.py"
    before = "def alpha():\n    return 1\n\ndef beta():\n    return 3\n"
    after = "def alpha():\n    return 2\n\ndef beta():\n    return 4\n"
    compiled = compile_snapshot(
        snap(
            {path: before},
            {path: after},
            repository="owner/project",
            overrides={
                ("base", path): {
                    "path": path,
                    "state": "supplied",
                    "coverage": "partial",
                }
            },
            fragment_options={("base", path): {"scope": "excerpt"}},
        )
    )
    report = WalkthroughNarrator(FixtureProvider()).generate(compiled)
    packet = report["packet"]
    passage = passage_for_regions(
        "mixed-evidence",
        "The related definitions share this partial source file.",
        packet["regions"],
    )
    report["story"]["sections"] = [
        {
            "id": "mixed-evidence",
            "title": "Partial source",
            "summary": "A routine change and an unresolved declaration share one file.",
            "passages": [passage],
        }
    ]
    report["story"]["quick_summary"]["passage_ids"] = [passage["id"]]
    report["story"]["impact"] = None
    report["story"]["route"] = [
        {"kind": "mechanism", "passage_ids": [passage["id"]]}
    ]
    report["story"]["review_checks"] = []
    report["story"]["author_reported_checks"] = []
    return report


class BrowserWalkthroughTests(unittest.TestCase):
    """Keep the flat reader offline, source-linked, expandable, and inert."""

    @classmethod
    def setUpClass(cls) -> None:
        """Start Chromium and construct one public synthetic saved report."""
        lines = [
            "def calculate():",
            *[f"    value_{i} = {i}" for i in range(200)],
            "    return 1",
        ]
        before = "\n".join(lines) + "\n"
        after = before.replace("return 1", "return 2")
        snapshot = snap(
            {"a.py": before},
            {"a.py": after},
            repository="owner/project",
            title="Browser fixture",
        )
        snapshot["file_evidence"].append(
            {
                side: {
                    "path": "missing.py",
                    "state": "unavailable",
                    "reason": "not_supplied",
                }
                for side in ("base", "head")
            }
        )
        snapshot["patches"].append(
            {"path": "missing.py", "origin": "github", "patch": None}
        )
        compiled = compile_snapshot(snapshot)
        compiled["changes"][0]["basis"] = "Compare `calculate` in `a.py`."
        cls.report = WalkthroughNarrator(FixtureProvider()).generate(compiled)
        cls.sketch = (
            "Saved versions\n    ↓\nComparison\n    ↓\nLinked explanation"
        )
        cls.report["story"]["preamble"] += (
            "\n\nTreat <img src=x onerror=window.pwned=1> as data."
            f"\n\n```text\n{cls.sketch}\n```"
        )
        passage = cls.report["story"]["sections"][0]["passages"][0]
        passage["text"] = (
            "Review </script><img src=x onerror=window.pwned=1> as inert text."
        )
        passage["citations"] = [
            {
                "source_id": source["id"],
                "revision": source["revision"],
                "start": 1,
                "end": len(lines),
            }
            for source in cls.report["packet"]["sources"]
        ]
        cls.html = render(cls.report)
        cls.feature_report = feature_report()
        cls.feature_html = render(cls.feature_report)
        cls.grouped_report = grouped_report()
        cls.grouped_html = render(cls.grouped_report)
        cls.mixed_report = mixed_group_report()
        cls.mixed_html = render(cls.mixed_report)
        cls.playwright = sync_playwright().start()
        cls.browser = cls.playwright.chromium.launch()

    @classmethod
    def tearDownClass(cls) -> None:
        """Close all Chromium processes after the checks finish."""
        cls.browser.close()
        cls.playwright.stop()

    def setUp(self) -> None:
        """Load a fresh offline page and capture browser errors and transfers."""
        self.context = self.browser.new_context(accept_downloads=True)
        self.page = self.context.new_page()
        self.page.set_default_timeout(5_000)
        self.errors = []
        self.requests = []
        self.page.on("pageerror", lambda error: self.errors.append(str(error)))
        self.page.on(
            "request", lambda request: self.requests.append(request.url)
        )
        self.page.route("**/*", lambda route: route.abort())
        self.page.set_content(self.html, wait_until="load")
        self.page.locator(".story-section").wait_for()

    def tearDown(self) -> None:
        """Release page state so one check cannot affect another."""
        self.context.close()

    def assert_passage_links_navigate(
        self, links: list, passage_ids: list[str]
    ) -> None:
        """Check that each visible label points to its actual passage."""
        story_passages = self.feature_report["story"]["sections"][0][
            "passages"
        ]
        page_passages = self.page.locator(".story-section .passage")
        self.assertEqual(len(links), len(passage_ids))
        for link, passage_id in zip(links, passage_ids, strict=True):
            index = next(
                i
                for i, passage in enumerate(story_passages)
                if passage["id"] == passage_id
            )
            target = page_passages.nth(index)
            target_id = target.get_attribute("id")
            self.assertEqual(link.get_attribute("href"), "#" + target_id)
            link.click()
            self.assertEqual(
                self.page.evaluate("location.hash"), "#" + target_id
            )
            self.assertTrue(target.is_visible())

    def test_original_patch_and_function_comparison_are_separate(self) -> None:
        """Keep PR grouping by default; make generated comparisons opt-in."""
        original = "@@ -2 +2 @@\n-    return 1\n+    return 2\n"
        for supplied in (original, None):
            with self.subTest(original_available=supplied is not None):
                snapshot = snap(
                    {"a.py": "def calculate():\n    return 1\n"},
                    {"a.py": "def calculate():\n    return 2\n"},
                    repository="owner/project",
                )
                snapshot["patches"] = [
                    {"path": "a.py", "origin": "github", "patch": supplied}
                ]
                report = WalkthroughNarrator(FixtureProvider()).generate(
                    compile_snapshot(snapshot)
                )
                self.page.set_content(render(report), wait_until="load")
                block = self.page.locator(".passage > .code-block").first
                block.scroll_into_view_if_needed()
                if block.locator("[data-load-preview]").count():
                    block.locator("[data-load-preview]").evaluate(
                        "(button) => button.click()"
                    )
                expected = (
                    "GitHub patch excerpt"
                    if supplied
                    else "Original patch unavailable"
                )
                self.assertIn(
                    expected, block.locator(".code-footer").inner_text()
                )
                self.assertEqual(
                    block.locator(".code-line.context").count(), 0
                )
                self.assertEqual(
                    block.locator(".code-line.add").count(),
                    1 if supplied else 0,
                )
                block.get_by_role(
                    "button", name="Compare function versions"
                ).click()
                self.assertIn(
                    "Function comparison · generated",
                    block.locator(".code-footer").inner_text(),
                )
                self.assertIn(
                    "def calculate()",
                    block.locator(".code-line.context").inner_text(),
                )
                self.assertIn(
                    "return 1", block.locator(".code-line.delete").inner_text()
                )
                self.assertIn(
                    "return 2", block.locator(".code-line.add").inner_text()
                )
                block.get_by_role(
                    "button", name="View diff", exact=True
                ).click()
                self.assertIn(
                    expected, block.locator(".code-footer").inner_text()
                )
                self.assertEqual(
                    block.locator(".code-line.context").count(), 0
                )
                if supplied:
                    self.page.locator("#source-hunks > summary").click()
                    raw = self.page.locator(".raw-file").first
                    raw.locator(":scope > summary").click()
                    raw.locator(".original-patch > summary").click()
                    self.assertEqual(
                        raw.locator(".original-patch pre").text_content(),
                        original,
                    )
        self.assertEqual(self.requests, [])

    def test_cross_file_patch_rows_and_comparison_keep_source_coordinates(
        self,
    ) -> None:
        """Resolve file-patch and function rows from their separate sources."""
        moved_change = next(
            change
            for change in self.feature_report["packet"]["changes"]
            if change["kind"] == "moved_renamed"
        )
        paths = source_path_index(self.feature_report)
        self.assertEqual(
            paths[moved_change["before"]["source_id"]], "old_move.py"
        )
        self.assertEqual(
            paths[moved_change["after"]["source_id"]], "new_move.py"
        )
        self.assertNotEqual(
            moved_change["before"]["source_id"],
            moved_change["after"]["source_id"],
        )
        self.assertNotEqual(
            moved_change["before"]["start"], moved_change["after"]["start"]
        )
        self.page.set_content(self.feature_html, wait_until="load")
        moved = self.page.locator(
            f'[data-changes~="{moved_change["id"]}"]'
        ).first
        moved.scroll_into_view_if_needed()
        moved.get_by_role("button", name="View diff", exact=True).click()
        old_delete = moved.locator(".code-line.delete").filter(
            has_text="def old_name()"
        )
        new_add = moved.locator(".code-line.add").filter(
            has_text="def new_name()"
        )
        old_delete.first.wait_for()
        new_add.first.wait_for()
        self.assertIn("def old_name()", old_delete.first.inner_text())
        self.assertIn("def new_name()", new_add.first.inner_text())
        self.assertIn(
            "1", old_delete.first.locator(".line-no").all_inner_texts()
        )
        self.assertIn("3", new_add.first.locator(".line-no").all_inner_texts())

        moved.get_by_role(
            "button", name="Compare function versions", exact=True
        ).click()
        old_delete = moved.locator(".code-line.delete").filter(
            has_text="def old_name()"
        )
        new_add = moved.locator(".code-line.add").filter(
            has_text="def new_name()"
        )
        old_delete.first.wait_for()
        new_add.first.wait_for()
        self.assertIn(
            "1", old_delete.first.locator(".line-no").all_inner_texts()
        )
        self.assertIn("3", new_add.first.locator(".line-no").all_inner_texts())
        self.assertEqual(self.requests, [])

    def test_source_links_bind_exact_revisions_and_original_lines(
        self,
    ) -> None:
        """Cited code must expose the correct side and original line permalink."""
        block = self.page.locator(".passage > .code-block").first
        block.scroll_into_view_if_needed()
        block.locator(".code-source-link").first.wait_for()
        for revision in ["a" * 40, "b" * 40]:
            with self.subTest(revision=revision):
                hrefs = block.locator(".code-source-link").evaluate_all(
                    "nodes => nodes.map(node => node.href)"
                )
                self.assertTrue(
                    any(
                        f"/blob/{revision}/a.py#L1-L202" in href
                        for href in hrefs
                    )
                )

    def test_code_expands_beyond_initial_preview(self) -> None:
        """Lazy code previews must leave the complete source reachable."""
        self.page.set_content(self.feature_html, wait_until="load")
        minimum = 162
        block = self.page.locator(".passage > .code-block").first
        block.scroll_into_view_if_needed()
        block.get_by_role(
            "button", name="After definition", exact=True
        ).click()
        block.locator(".load-bar").wait_for()
        block.locator(".load-bar").click()
        self.assertGreaterEqual(
            block.locator(".code-line").count(), min(minimum, 160)
        )
        if block.locator(".load-bar").count():
            block.locator(".load-bar").click()
        self.assertEqual(block.locator(".code-line").count(), minimum)
        self.assertIn("return 2", block.inner_text())
        block.get_by_role("button", name="View diff", exact=True).click()
        while block.locator(".load-bar").count():
            block.locator(".load-bar").click()
        self.assertIn("return 1", block.locator(".code-lines").inner_text())
        self.assertIn("return 2", block.locator(".code-lines").inner_text())

    def test_passage_shows_additions_and_deletions_by_default(self) -> None:
        """Reading a passage must reveal its actual edits without opening an appendix."""
        block = self.page.locator(".passage > .code-block").first
        block.scroll_into_view_if_needed()
        block.locator(".code-line.delete").wait_for()
        cases = [
            ("delete", "return 1", "\u2212", ["202", ""]),
            ("add", "return 2", "+", ["", "202"]),
        ]
        for tag, text, sign, numbers in cases:
            with self.subTest(tag=tag):
                row = block.locator(f".code-line.{tag}")
                self.assertIn(text, row.locator(".code-text").inner_text())
                self.assertEqual(row.locator(".code-sign").inner_text(), sign)
                self.assertEqual(
                    row.locator(".line-no").all_inner_texts(), numbers
                )
        self.assertTrue(
            block.get_by_role(
                "button", name="Before definition", exact=True
            ).is_visible()
        )
        self.assertTrue(
            block.get_by_role(
                "button", name="After definition", exact=True
            ).is_visible()
        )

    def test_exact_cited_source_remains_available_beside_the_diff(
        self,
    ) -> None:
        """Opening citations must keep both pinned sources and line ranges."""
        citations = self.page.locator(".passage .citation-sources").first
        self.assertFalse(citations.evaluate("node => node.open"))
        citations.locator("summary").click()
        cases = [("base", "return 1"), ("head", "return 2")]
        for index, (side, text) in enumerate(cases):
            with self.subTest(side=side):
                block = citations.locator(".code-block").nth(index)
                block.scroll_into_view_if_needed()
                block.locator(".code-line").first.wait_for()
                self.assertIn(
                    "L1\N{EN DASH}202",
                    block.locator(".code-footer").inner_text(),
                )
                self.assertIn(text, block.inner_text())
                revision = "a" * 40 if side == "base" else "b" * 40
                self.assertIn(
                    f"/blob/{revision}/a.py#L1-L202",
                    block.locator(".code-source-link").first.get_attribute(
                        "href"
                    ),
                )
                self.assertEqual(
                    block.get_by_role("button", name="View diff").count(), 0
                )

    def test_untrusted_prose_is_displayed_as_text(self) -> None:
        """Source-like markup must not become HTML or execute handlers."""
        cases = [
            (".passage .prose", "<img src=x onerror=window.pwned=1>"),
            (".preamble-body", "<img src=x onerror=window.pwned=1>"),
        ]
        for selector, expected in cases:
            with self.subTest(selector=selector):
                self.assertIn(
                    expected, self.page.locator(selector).inner_text()
                )
                self.assertEqual(
                    self.page.locator(f"{selector} img").count(), 0
                )
                self.assertIsNone(self.page.evaluate("window.pwned"))

    def test_narrative_keeps_inline_code_styling_and_escapes_markup(
        self,
    ) -> None:
        """Code names and paths must keep their styling, including inside emphasis."""
        report = copy.deepcopy(self.report)
        section = report["story"]["sections"][0]
        text = (
            "Read `diffstory/models.py` and **`UnavailableFile`**; treat "
            "`<img src=x onerror=window.pwned=1>` as text."
        )
        fields = [
            (report["story"], "preamble", ".preamble-body > p"),
            (report["story"], "introduction", ".lead"),
            (section, "summary", ".section-context"),
            (section["passages"][0], "text", ".passage .prose"),
            (report["story"], "closing", ".closing"),
        ]
        for record, field, _selector in fields:
            record[field] = text
        self.page.set_content(render(report), wait_until="load")
        self.page.locator(".story-section").wait_for()
        expected = [
            "diffstory/models.py",
            "UnavailableFile",
            "<img src=x onerror=window.pwned=1>",
        ]
        for _record, _field, selector in fields:
            with self.subTest(selector=selector):
                narrative = self.page.locator(selector)
                self.assertEqual(
                    narrative.locator("code").all_inner_texts(), expected
                )
                self.assertEqual(
                    narrative.locator("strong code").inner_text(),
                    "UnavailableFile",
                )
                self.assertNotIn("`", narrative.inner_text())
                self.assertEqual(narrative.locator("img").count(), 0)
                code = narrative.locator("code").first
                self.assertNotEqual(
                    code.evaluate("node => getComputedStyle(node).fontFamily"),
                    narrative.evaluate(
                        "node => getComputedStyle(node).fontFamily"
                    ),
                )
        self.assertEqual(
            self.page.locator(".code-basis, .code-bases").count(), 0
        )
        self.assertEqual(
            json.loads(self.page.locator("#report-data").text_content()),
            report,
        )
        self.assertIsNone(self.page.evaluate("window.pwned"))

    def test_concept_preamble_precedes_code_and_keeps_paragraphs_and_sketch(
        self,
    ) -> None:
        """The project introduction must survive projection before any code tour."""
        preamble = self.page.locator(".preamble")
        self.assertEqual(preamble.count(), 1)
        self.assertEqual(
            preamble.locator("h2").inner_text().casefold(), "before the code"
        )
        expected = self.report["story"]["preamble"].split("\n\n")[:3]
        self.assertEqual(
            preamble.locator(".preamble-body > p").all_inner_texts(), expected
        )
        self.assertEqual(
            preamble.locator("pre code").inner_text(), self.sketch
        )
        self.assertTrue(
            preamble.evaluate(
                "node => Boolean(node.compareDocumentPosition(document.querySelector('.code-block')) & Node.DOCUMENT_POSITION_FOLLOWING)"
            )
        )
        self.assertEqual(
            self.page.locator(".lead").inner_text(),
            self.report["story"]["introduction"],
        )

    def test_offline_reader_preserves_hunks_and_unverified_label(self) -> None:
        """The reader must need no network and retain complete supplied diffs."""
        for selector, text in [
            (".generation-notice", "unverified"),
            ("#source-hunks summary", "Original source hunks"),
        ]:
            with self.subTest(selector=selector):
                self.assertIn(text, self.page.locator(selector).inner_text())
        self.page.locator("#source-hunks summary").click()
        self.page.locator("#source-hunks .raw-file summary").click()
        raw = self.page.locator("#source-hunks .code-block").first
        raw.scroll_into_view_if_needed()
        raw.locator(".code-line").first.wait_for()
        self.assertEqual(raw.locator(".code-line").count(), 5)
        for text in ["return 1", "return 2"]:
            self.assertIn(text, raw.inner_text())
        self.assertEqual(self.errors, [])
        self.assertEqual(self.requests, [])

    def test_unavailable_files_are_not_counted_as_supplied_source(
        self,
    ) -> None:
        """Missing file records must not inflate the supplied-source count."""
        expected = "1 source paths"
        self.assertIn(
            expected, self.page.locator(".document-meta").inner_text()
        )

    def test_download_keeps_the_flat_artifact(self) -> None:
        """Downloading must not serialize a display projection or duplicate source."""
        expected = "diffstory.report.v2"
        self.page.locator("#moreBtn").click()
        with self.page.expect_download() as pending:
            self.page.locator("#downloadReport").click()
        data = json.loads(Path(pending.value.path()).read_text())
        self.assertEqual(data["schema"], expected)
        embedded = json.loads(self.page.locator("#report-data").text_content())
        self.assertEqual(data, embedded)

    def test_passage_citations_focus_middle_edits_and_full_diff_stays_available(
        self,
    ) -> None:
        """Two citations in one large definition must select their own edits."""
        self.page.set_content(self.feature_html, wait_until="load")
        self.page.locator(".story-section").wait_for()
        expected = [
            (65, "value_64 = 64", "value_64 = 640"),
            (121, "value_120 = 120", "value_120 = 1200"),
        ]
        for index, (line, before_text, after_text) in enumerate(expected):
            with self.subTest(passage=index):
                block = self.page.locator(".passage > .code-block").nth(index)
                block.scroll_into_view_if_needed()
                block.locator(".code-line.delete").first.wait_for()
                changed = block.locator(
                    ".code-line.add, .code-line.delete"
                ).evaluate_all(
                    "nodes => nodes.map(node => ({"
                    "tag: node.classList.contains('add') ? 'add' : 'delete',"
                    "numbers: [...node.querySelectorAll('.line-no.dual')].map(n => n.textContent),"
                    "text: node.querySelector('.code-text').textContent"
                    "}))"
                )
                self.assertTrue(
                    any(
                        row["tag"] == "delete"
                        and row["numbers"][0] == str(line)
                        and row["text"].strip() == before_text
                        for row in changed
                    ),
                    f"cited before edit at original line {line} is visible",
                )
                self.assertTrue(
                    any(
                        row["tag"] == "add"
                        and row["numbers"][1] == str(line)
                        and row["text"].strip() == after_text
                        for row in changed
                    ),
                    f"cited after edit at original line {line} is visible",
                )

        block = self.page.locator(".passage > .code-block").first
        while block.locator(".load-bar").count():
            block.locator(".load-bar").click()
        complete = block.locator(".code-line").evaluate_all(
            "nodes => nodes.map(node => node.textContent)"
        )
        for text in (
            "value_40 = 40",
            "value_40 = 400",
            "value_131 = 131",
            "value_131 = 1310",
            "return 1",
            "return 2",
        ):
            self.assertTrue(any(text in row for row in complete), text)
        self.assertGreater(len(complete), 100)
        self.assertEqual(self.requests, [])

    def test_wide_citation_preview_stays_bounded_and_expands_to_full_diff(
        self,
    ) -> None:
        """A broad cited range stays compact but never drops full diff rows."""
        report = copy.deepcopy(self.feature_report)
        passage = report["story"]["sections"][0]["passages"][0]
        for citation in passage["citations"]:
            citation["start"] = 1
            citation["end"] = 162
        self.page.set_content(render(report), wait_until="load")
        self.page.locator(".story-section").wait_for()
        block = self.page.locator(".passage > .code-block").first
        block.scroll_into_view_if_needed()
        block.locator(".code-line.delete").first.wait_for()
        self.assertLessEqual(block.locator(".code-line").count(), 24)
        self.assertIn("cited diff lines folded", block.inner_text())
        while block.locator(".load-bar").count():
            block.locator(".load-bar").click()
        complete = block.locator(".code-lines").inner_text()
        for text in (
            "value_40 = 40",
            "value_40 = 400",
            "value_85 = 85",
            "value_85 = 850",
        ):
            self.assertIn(text, complete)
        self.assertGreater(block.locator(".code-line").count(), 100)
        self.assertEqual(self.requests, [])

    def test_passage_search_returns_a_safe_sentence_and_navigates_to_it(
        self,
    ) -> None:
        """Search must find passage-only prose and link to its stable anchor."""
        self.page.set_content(self.feature_html, wait_until="load")
        self.page.locator(".story-section").wait_for()
        search = self.page.locator("#search")
        self.assertIn(
            "passage", search.get_attribute("placeholder").casefold()
        )
        self.page.locator("#contentsBtn").click()
        search.fill("zero-byte guard")
        result = self.page.locator("#contentsList .passage-result").first
        result.wait_for()
        self.assertIn(
            "The zero-byte guard now rejects an empty response.",
            result.inner_text(),
        )
        target_id = self.page.locator(".passage").nth(1).get_attribute("id")
        self.assertEqual(result.get_attribute("href"), f"#{target_id}")
        self.assertNotIn("middle-second", result.inner_text())
        result.click()
        self.assertEqual(self.page.evaluate("location.hash"), f"#{target_id}")
        self.assertTrue(self.page.locator(f"#{target_id}").is_visible())

        self.page.locator("#contentsBtn").click()
        search.fill("<img src=x onerror=window.pwned=1>")
        hostile_result = self.page.locator("#contentsList .passage-result")
        hostile_result.wait_for()
        self.assertIn(
            "<img src=x onerror=window.pwned=1>",
            hostile_result.first.inner_text(),
        )
        self.assertEqual(self.page.locator("#contentsList img").count(), 0)
        self.assertEqual(self.page.locator(".passage img").count(), 0)
        self.assertIsNone(self.page.evaluate("window.pwned"))

        search.fill("Reader cases")
        section_id = self.page.locator(".story-section").first.get_attribute(
            "id"
        )
        self.assertEqual(
            self.page.locator(
                f'#contentsList .contents-item[href="#{section_id}"]'
            ).count(),
            1,
        )

    def test_passage_search_keeps_a_late_dotted_path_visible(self) -> None:
        """A dotted path in a later sentence stays in the matching result."""
        report = copy.deepcopy(self.feature_report)
        passage = report["story"]["sections"][0]["passages"][0]
        passage["text"] = (
            "A short opening sentence. "
            + ("Validate supplied evidence carefully " * 18)
            + "in `diffstory/models.py` before review."
        )
        self.page.set_content(render(report), wait_until="load")
        self.page.locator(".story-section").wait_for()
        self.page.locator("#contentsBtn").click()
        self.page.locator("#search").fill("diffstory/models.py")
        result = self.page.locator("#contentsList .passage-result").first
        result.wait_for()

        self.assertIn("diffstory/models.py", result.inner_text())
        self.assertNotIn("A short opening sentence", result.inner_text())
        self.assertEqual(result.locator("img").count(), 0)
        target_id = self.page.locator(".passage").first.get_attribute("id")
        self.assertEqual(result.get_attribute("href"), f"#{target_id}")
        result.click()
        self.assertEqual(self.page.evaluate("location.hash"), f"#{target_id}")
        self.assertTrue(self.page.locator(f"#{target_id}").is_visible())

    def test_passage_search_keeps_a_late_long_sentence_query_visible(
        self,
    ) -> None:
        """A late search term stays visible with its matching sentence."""
        report = copy.deepcopy(self.feature_report)
        passage = report["story"]["sections"][0]["passages"][0]
        passage["text"] = (
            "An unrelated opening sentence. "
            + ("Validate supplied evidence carefully " * 18)
            + "The late marker `zero-byte-sentinel` identifies this evidence."
        )
        self.page.set_content(render(report), wait_until="load")
        self.page.locator(".story-section").wait_for()
        self.page.locator("#contentsBtn").click()
        self.page.locator("#search").fill("zero-byte-sentinel")
        result = self.page.locator("#contentsList .passage-result").first
        result.wait_for()

        self.assertIn("zero-byte-sentinel", result.inner_text())
        self.assertNotIn("An unrelated opening sentence", result.inner_text())
        target_id = self.page.locator(".passage").first.get_attribute("id")
        self.assertEqual(result.get_attribute("href"), f"#{target_id}")
        result.click()
        self.assertEqual(self.page.evaluate("location.hash"), f"#{target_id}")
        self.assertTrue(self.page.locator(f"#{target_id}").is_visible())

    def test_opening_summary_status_and_evidence_order(self) -> None:  # noqa: PLR0915 # One integrated navigation scenario.
        """Section links form distinct rows and navigate to exact passages."""
        section_title = (
            "Reader cases for the `calculate()` compatibility path, "
            "response reconciliation rules, neighboring state recovery "
            "branches, documented fallbacks, concurrent updates, retry "
            "behavior, and review conclusions"
        )
        visible_title = section_title.replace("`", "")
        layout_report = copy.deepcopy(self.feature_report)
        layout_report["story"]["sections"][0]["title"] = section_title
        self.page.set_content(render(layout_report), wait_until="load")
        self.page.set_viewport_size({"width": 1280, "height": 900})
        self.page.locator(".story-section").wait_for()
        opening = self.page.locator(".walkthrough-opening")
        self.assertEqual(opening.count(), 1)
        order = self.page.evaluate("""() => {
          const nodes = ['.preamble', '.walkthrough-opening', '.story-section']
            .map(selector => document.querySelector(selector));
          return nodes.map(node => [...node.parentElement.children].indexOf(node));
        }""")
        self.assertEqual(order, sorted(order))
        summary = opening.locator(".quick-summary")
        self.assertIn("earlier calculate() comparison", summary.inner_text())
        self.assertIn("later calculate() comparison", summary.inner_text())
        self.assertEqual(
            summary.locator(".summary-pair code").all_inner_texts(),
            ["calculate()", "calculate()"],
        )
        self.assertNotIn("`", summary.inner_text())
        summary_links = summary.locator("a[href^='#passage-']").all()
        self.assertEqual(len(summary_links), 2)
        self.assertEqual(
            [link.inner_text() for link in summary_links],
            [
                f"{visible_title} · paragraph 1",
                f"{visible_title} · paragraph 2",
            ],
        )
        self.assertEqual(
            summary_links[0].locator("code").all_inner_texts(),
            ["calculate()"],
        )
        arrow_content = summary_links[0].evaluate(
            "(node) => getComputedStyle(node, '::after').content"
        )
        self.assertIn("→", arrow_content)
        self.assertNotEqual(arrow_content, "none")
        self.assertEqual(
            summary.locator(".evidence-label").evaluate(
                "(node) => getComputedStyle(node).textTransform"
            ),
            "uppercase",
        )
        impact_links = opening.locator(
            ".walkthrough-impact a[href^='#passage-']"
        ).all()
        self.assertEqual(
            [link.inner_text() for link in impact_links], [visible_title]
        )
        route_links = opening.locator(
            ".reading-route a[href^='#passage-']"
        ).all()
        self.assertEqual(
            [link.inner_text() for link in route_links],
            [visible_title, visible_title, visible_title],
        )
        self.assertEqual(
            opening.locator(".route-label").first.evaluate(
                "(node) => getComputedStyle(node).textTransform"
            ),
            "uppercase",
        )
        for anchor in summary_links + impact_links + route_links:
            self.assertEqual(
                anchor.evaluate(
                    "(node) => getComputedStyle(node).textTransform"
                ),
                "none",
            )
        self.assertLess(
            summary.locator(".evidence-label").bounding_box()["y"],
            summary.locator(".passage-links").bounding_box()["y"],
        )
        for route_row in opening.locator(".reading-route p").all():
            self.assertLess(
                route_row.locator(".route-label").bounding_box()["y"],
                route_row.locator(".passage-links").bounding_box()["y"],
            )

        def check_rows_fit_viewport() -> None:
            viewport_width = self.page.evaluate("innerWidth")
            self.assertLessEqual(
                self.page.evaluate("document.documentElement.scrollWidth"),
                viewport_width,
            )
            for group in self.page.locator(".passage-links").all():
                bounds = group.bounding_box()
                self.assertIsNotNone(bounds)
                rects = group.locator("a").evaluate_all(
                    """nodes => nodes.map(node => {
                      const box = node.getBoundingClientRect();
                      return {
                        top: box.top, bottom: box.bottom,
                        left: box.left, right: box.right, height: box.height
                      };
                    })"""
                )
                self.assertTrue(rects)
                for rect in rects:
                    self.assertAlmostEqual(rect["left"], bounds["x"], delta=1)
                    self.assertAlmostEqual(
                        rect["right"],
                        bounds["x"] + bounds["width"],
                        delta=1,
                    )
                    self.assertLessEqual(rect["right"], viewport_width)
                    self.assertGreaterEqual(rect["height"], 44)
                for index in range(1, len(rects)):
                    self.assertGreater(
                        rects[index]["top"] - rects[index - 1]["bottom"], 0
                    )

        check_rows_fit_viewport()
        self.assertGreater(summary_links[0].bounding_box()["height"], 44)
        self.page.set_viewport_size({"width": 320, "height": 800})
        check_rows_fit_viewport()
        self.assertTrue(
            all(link.bounding_box()["height"] > 44 for link in summary_links)
        )
        summary_links[0].hover()
        self.assertEqual(
            summary_links[0].evaluate(
                "(node) => getComputedStyle(node).textDecorationLine"
            ),
            "underline",
        )
        summary_links[0].focus()
        self.page.keyboard.press("Tab")
        focused_link = self.page.locator(
            ".passage-links a:focus-visible"
        ).first
        self.assertTrue(focused_link.count())
        self.assertEqual(
            focused_link.evaluate(
                "(node) => getComputedStyle(node).outlineStyle"
            ),
            "solid",
        )
        route_ids = [
            passage_id
            for item in self.feature_report["story"]["route"]
            for passage_id in item["passage_ids"]
        ]
        self.assert_passage_links_navigate(
            summary_links,
            self.feature_report["story"]["quick_summary"]["passage_ids"],
        )
        self.assert_passage_links_navigate(
            impact_links,
            self.feature_report["story"]["impact"]["passage_ids"],
        )
        self.assert_passage_links_navigate(route_links, route_ids)
        self.assertIn("0 test runs", opening.inner_text().casefold())
        self.assertIn("verified by diffstory", opening.inner_text().casefold())
        author = opening.locator(".author-reported")
        self.assertIn(
            "not independently verified", author.inner_text().casefold()
        )
        author.locator("summary").click()
        self.assertIn("2 tests passed locally", author.inner_text())

    def test_unindexed_test_source_remains_available_and_is_described_honestly(
        self,
    ) -> None:
        """Empty test indexing must not imply that cited test source is absent."""
        report = copy.deepcopy(self.feature_report)
        report["packet"]["analysis"]["tests"] = []
        report["generation"]["evidence_digest"] = packet_identity(
            report["packet"]
        )
        for check in report["story"]["review_checks"]:
            check["test_ids"] = []
        self.page.set_content(render(report), wait_until="load")
        self.page.locator(".story-section").wait_for()

        opening = self.page.locator(".walkthrough-opening")
        status = opening.inner_text()
        self.assertIn("0 test runs verified by Diffstory.", status)
        self.assertIn(
            "No individual test definitions are listed separately.", status
        )
        self.assertIn(
            "Test code may still be available in the cited source.", status
        )
        self.assertNotIn("No static test source is attached", status)
        author = opening.locator(".author-reported")
        self.assertIn(
            "not independently verified",
            author.locator("summary").inner_text(),
        )
        author.locator("summary").click()
        self.assertIn("2 tests passed locally", author.inner_text())

        review = self.page.locator(".review-check")
        self.assertEqual(review.locator(".review-test-source").count(), 0)
        passage = self.page.locator(".passage").nth(2)
        source = passage.locator(".code-block").first
        source.scroll_into_view_if_needed()
        source.locator(".code-line").first.wait_for()
        self.assertIn("test_calculate", source.inner_text())
        self.assertEqual(self.requests, [])

    def test_review_checks_link_to_passages_and_static_test_source(
        self,
    ) -> None:
        """Review work links to exact prose and inspectable local test code."""
        self.page.set_content(self.feature_html, wait_until="load")
        self.page.locator(".story-section").wait_for()
        review = self.page.locator(".review-check")
        self.assertEqual(review.count(), 1)
        self.assertIn("Confirm the changed test", review.inner_text())
        review_links = review.locator("a[href^='#passage-']")
        self.assertEqual(review_links.count(), 2)
        links = review_links.all()
        self.assertEqual(
            [link.inner_text() for link in links],
            ["Reader cases · paragraph 3", "Reader cases · paragraph 2"],
        )
        self.assertEqual(
            review.locator(".evidence-label").evaluate(
                "(node) => getComputedStyle(node).textTransform"
            ),
            "uppercase",
        )
        self.assertLess(
            review.locator(".evidence-label").bounding_box()["y"],
            review.locator(".review-links").bounding_box()["y"],
        )
        for anchor in links:
            self.assertEqual(
                anchor.evaluate(
                    "(node) => getComputedStyle(node).textTransform"
                ),
                "none",
            )
        self.assert_passage_links_navigate(
            links,
            self.feature_report["story"]["review_checks"][0]["passage_ids"],
        )
        source_details = review.locator(".review-test-source")
        self.assertEqual(source_details.count(), 1)
        source_details.locator("summary").click()
        source = source_details.locator(".code-block")
        source.scroll_into_view_if_needed()
        source.locator(".code-line").first.wait_for()
        self.assertIn("test_calculate", source.inner_text())
        source_link = source.locator(".code-source-link")
        self.assertIn(
            "/blob/" + "b" * 40 + "/tests/test_large.py#L1-L2",
            source_link.first.get_attribute("href"),
        )
        self.assertIn("static", review.inner_text().casefold())
        self.assertEqual(self.requests, [])

    def test_opening_is_responsive_and_null_impact_is_not_established(
        self,
    ) -> None:
        """The opening fits a phone viewport and labels unsupported impact."""
        self.page.set_content(self.feature_html, wait_until="load")
        self.page.locator(".story-section").wait_for()
        self.page.set_viewport_size({"width": 390, "height": 844})
        self.assertEqual(
            self.page.locator(".summary-pair")
            .evaluate("node => getComputedStyle(node).gridTemplateColumns")
            .count(" "),
            0,
        )
        self.assertLessEqual(
            self.page.evaluate("document.documentElement.scrollWidth"),
            self.page.evaluate("window.innerWidth"),
        )
        grouped = self.context.new_page()
        try:
            grouped.route("**/*", lambda route: route.abort())
            grouped.set_content(self.grouped_html, wait_until="load")
            grouped.locator(".story-section").wait_for()
            self.assertEqual(
                grouped.locator(
                    ".walkthrough-opening .walkthrough-impact"
                ).count(),
                1,
            )
            self.assertIn(
                "not established",
                grouped.locator(".walkthrough-opening .walkthrough-impact")
                .inner_text()
                .casefold(),
            )
        finally:
            grouped.close()

    def test_archived_v2_story_is_rejected_by_render_validation(
        self,
    ) -> None:
        """A retired flat-story generation marker is rejected by the renderer."""
        archived = copy.deepcopy(self.feature_report)
        for field in (
            "quick_summary",
            "impact",
            "route",
            "author_reported_checks",
            "review_checks",
        ):
            archived["story"].pop(field)
        archived["generation"]["version"] = (
            "diffstory.generation.walkthrough.v2"
        )
        with self.assertRaises(ValueError):
            render(archived)

    def test_reader_startup_rejects_unsupported_report_formats(self) -> None:
        """The browser refuses unknown and retired data instead of falling back."""
        cases = [
            (
                "unknown schema",
                {"schema": "diffstory.report.v99"},
                "Unsupported embedded report schema",
            ),
            (
                "shared envelope",
                {
                    "schema": "diffstory.report.v1",
                    "storage_version": "diffstory.report-storage.v1",
                    "artifact_kind": "report",
                    "root": {},
                    "tables": {},
                },
                "Unsupported shared report envelope",
            ),
            (
                "compiler generation marker",
                {
                    "schema": "diffstory.report.v1",
                    "generation": {"version": "diffstory.generation.v1"},
                },
                "Unsupported generated fields in compiler report",
            ),
            (
                "compiler narrative marker",
                {
                    "schema": "diffstory.report.v1",
                    "narrative_v2": {"version": "diffstory.presentation.v2"},
                },
                "Unsupported generated fields in compiler report",
            ),
            (
                "missing flat generation",
                {"schema": "diffstory.report.v2"},
                "Unsupported walkthrough generation version",
            ),
            (
                "archived flat generation",
                {
                    "schema": "diffstory.report.v2",
                    "generation": {
                        "version": "diffstory.generation.walkthrough.v2"
                    },
                },
                "Unsupported walkthrough generation version",
            ),
        ]
        for label, report, expected in cases:
            with self.subTest(label=label):
                self.errors.clear()
                self.page.set_content(
                    embed_report_data(self.html, report),
                    wait_until="load",
                )
                self.assertTrue(self.errors, "reader startup must fail")
                self.assertIn(expected, self.errors[-1])
                self.assertEqual(
                    self.page.locator(".story-section").count(), 0
                )

    def test_flat_diff_blocks_show_only_available_before_and_after_definitions(
        self,
    ) -> None:
        """Definition controls must show the correct local source side."""
        self.page.set_content(self.feature_html, wait_until="load")
        self.page.locator(".story-section").wait_for()
        packet = self.feature_report["packet"]

        def block_for(kind: str) -> object:
            change = next(
                change
                for change in packet["changes"]
                if change["kind"] == kind
            )
            block = self.page.locator(
                f'.code-block[data-changes~="{change["id"]}"]'
            ).first
            block.scroll_into_view_if_needed()
            block.locator(".code-line").first.wait_for()
            return block

        modified = block_for("modified")
        before = modified.get_by_role(
            "button", name="Before definition", exact=True
        )
        after = modified.get_by_role(
            "button", name="After definition", exact=True
        )
        before.click()
        self.assertEqual(before.get_attribute("aria-pressed"), "true")
        self.assertIn("value_64 = 64", modified.inner_text())
        self.assertNotIn("value_64 = 640", modified.inner_text())
        self.assertIn(
            "65", modified.locator(".code-line .line-no").all_inner_texts()
        )
        initial_rows = modified.locator(".code-line").count()
        after.click()
        self.assertEqual(after.get_attribute("aria-pressed"), "true")
        self.assertIn("value_64 = 640", modified.inner_text())
        self.assertNotIn("value_64 = 64\n", modified.inner_text())
        self.assertEqual(modified.locator(".code-line").count(), initial_rows)
        self.assertIn(
            "65", modified.locator(".code-line .line-no").all_inner_texts()
        )
        modified.get_by_role(
            "button", name="After definition", exact=True
        ).click()
        modified.get_by_role(
            "button", name="Load full after definition"
        ).click()
        self.assertGreater(
            modified.locator(".code-line").count(), initial_rows
        )
        modified.get_by_role(
            "button", name="Before definition", exact=True
        ).click()
        self.assertEqual(modified.locator(".code-line").count(), initial_rows)
        self.assertIn("value_64 = 64", modified.inner_text())
        modified.get_by_role("button", name="View diff", exact=True).click()
        self.assertTrue(modified.locator(".code-line.delete").count())
        self.assertTrue(modified.locator(".code-line.add").count())

        added = block_for("added")
        self.assertEqual(
            added.get_by_role(
                "button", name="Before definition", exact=True
            ).count(),
            0,
        )
        added.get_by_role(
            "button", name="After definition", exact=True
        ).click()
        self.assertIn("new_widget", added.inner_text())
        removed = block_for("removed")
        self.assertEqual(
            removed.get_by_role(
                "button", name="After definition", exact=True
            ).count(),
            0,
        )
        removed.get_by_role(
            "button", name="Before definition", exact=True
        ).click()
        self.assertIn("old_widget", removed.inner_text())

        moved = block_for("moved_renamed")
        moved.get_by_role(
            "button", name="Before definition", exact=True
        ).click()
        self.assertIn("old_move.py", moved.locator(".filename").inner_text())
        self.assertIn("old_name", moved.inner_text())
        moved.get_by_role(
            "button", name="After definition", exact=True
        ).click()
        self.assertIn("new_move.py", moved.locator(".filename").inner_text())
        self.assertIn("new_name", moved.inner_text())
        self.assertIn(
            "3", moved.locator(".code-line .line-no").all_inner_texts()
        )
        self.assertEqual(self.requests, [])

    def test_same_file_changes_share_one_diff_without_basis_clutter(
        self,
    ) -> None:
        """Routine grouped edits stay compact while retaining source links."""
        self.page.set_content(self.grouped_html, wait_until="load")
        self.page.locator(".story-section").wait_for()
        changes = self.grouped_report["packet"]["changes"]
        blocks = self.page.locator(".passage > .code-block")
        self.assertEqual(blocks.count(), 1)
        block = blocks.first
        block.scroll_into_view_if_needed()
        block.locator(".code-line.add").first.wait_for()
        self.assertEqual(
            block.locator(".classification").inner_text(),
            "3 related changes",
        )
        self.assertNotIn("moved declarations", block.inner_text())
        added = [
            text.strip()
            for text in block.locator(
                ".code-line.add .code-text"
            ).all_inner_texts()
        ]
        deleted = [
            text.strip()
            for text in block.locator(
                ".code-line.delete .code-text"
            ).all_inner_texts()
        ]
        for value in ("return 11", "return 22", "return 33"):
            self.assertIn(value, added)
        for value in ("return 1", "return 2", "return 3"):
            self.assertIn(value, deleted)
        self.assertEqual(
            block.locator(".code-basis, .code-bases, .code-caveat").count(), 0
        )
        embedded = json.loads(self.page.locator("#report-data").text_content())
        self.assertEqual(
            [change["basis"] for change in embedded["packet"]["changes"]],
            [change["basis"] for change in changes],
        )
        for change in changes:
            self.assertEqual(
                self.page.locator(f"#unit-{change['id']}").count(), 1
            )
        while block.locator(".load-bar").count():
            block.locator(".load-bar").click()
        complete = block.locator(".code-lines").inner_text()
        for value in ("return 11", "return 22", "return 33"):
            self.assertIn(value, complete)
        self.assertEqual(
            self.page.locator(".passage > .code-block").count(), 1
        )
        self.assertEqual(self.requests, [])

    def test_grouped_routine_change_keeps_material_source_caveat(self) -> None:
        """A routine first change must not hide grouped partial-source caveats."""
        self.page.set_content(self.mixed_html, wait_until="load")
        self.page.locator(".story-section").wait_for()
        changes = self.mixed_report["packet"]["changes"]
        self.assertGreaterEqual(len(changes), 2)
        self.assertTrue(
            all(change["kind"] == "modified" for change in changes)
        )
        block = self.page.locator(".passage > .code-block").first
        self.assertEqual(
            self.page.locator(".passage > .code-block").count(), 1
        )
        block.scroll_into_view_if_needed()
        block.get_by_role(
            "button", name="Compare function versions", exact=True
        ).click()
        block.locator(".code-line.add").first.wait_for()
        self.assertEqual(block.locator(".code-basis, .code-bases").count(), 0)
        caveats = block.locator(".code-caveat")
        self.assertGreaterEqual(caveats.count(), 1)
        caveat_text = " ".join(caveats.all_inner_texts()).casefold()
        self.assertIn("only part", caveat_text)
        for change in changes:
            self.assertNotIn(change["basis"], block.inner_text())

        if block.locator("[data-load-preview]").count():
            block.locator("[data-load-preview]").click()
        while block.locator(".load-bar").count():
            block.locator(".load-bar").click()
        self.assertIn(
            "only part", block.locator(".code-caveat").inner_text().casefold()
        )
        self.assertIn("return 2", block.locator(".code-lines").inner_text())
        for change in changes:
            self.assertEqual(
                self.page.locator(f"#unit-{change['id']}").count(), 1
            )
        self.assertEqual(self.requests, [])


if __name__ == "__main__":
    unittest.main()
