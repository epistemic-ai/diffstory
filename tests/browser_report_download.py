"""Check report downloads and browser format routing in local Chromium.

Run from the repository with ``PYTHONPATH=.:tests``. The check uses one
count-1 authored report and a local fixture provider. An optional artifact
parent and Chromium executable come from ``DIFFSTORY_VERIFICATION_DIR`` and
``DIFFSTORY_CHROMIUM``. Generated files stay in a unique invocation directory.
"""

from __future__ import annotations

import copy
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from typing import TYPE_CHECKING
from typing import ClassVar
from urllib.parse import urlsplit

from playwright.sync_api import Browser
from playwright.sync_api import BrowserContext
from playwright.sync_api import Page
from playwright.sync_api import Playwright
from playwright.sync_api import Request
from playwright.sync_api import Route
from playwright.sync_api import sync_playwright
from test_cli_narration_v2 import FixtureProvider
from test_narrative_v2 import golden_report

from diffstory.narrative import Narrator
from diffstory.narrative_plan import canonical_bytes
from diffstory.narrative_session import ReportSession
from diffstory.render import iter_render_validated

if TYPE_CHECKING:
    from collections.abc import Iterable

MAX_REPORT_BYTES = 64 * 1024 * 1024
REPORT_DATA_OPEN = b'<script type="application/json" id="report-data">'
LARGE_INTEGER = 9_007_199_254_740_993
NO_STORAGE_VERSION = object()

ESCAPE_CASE_MATRIX = [
    (
        "five script-safe characters and Unicode",
        "<&>\u2028\u2029 café 東京",
        "<&>\u2028\u2029 café 東京",
    ),
    (
        "literal escape-looking text",
        r"\u0026 \u003c \u003e \u2028 \u2029",
        r"\u0026 \u003c \u003e \u2028 \u2029",
    ),
    ("one backslash before ampersand", "\\&", "\\&"),
    ("two backslashes before less-than", "\\\\<", "\\\\<"),
    (
        "three backslashes before a line separator",
        "\\" * 3 + "\u2029",
        "\\" * 3 + "\u2029",
    ),
]


def _collect(chunks: Iterable[bytes]) -> bytes:
    """Collect a trusted report writer stream for one small browser fixture.

    Args:
        chunks: Ordered output chunks from a bounded ReportSession writer.

    Returns:
        The exact concatenation of the chunks.

    Preconditions:
        Every item is a byte string from one trusted writer call.
    Postconditions:
        The return value preserves every input byte and its order.
    Essential behavior:
        Keeps test collection outside the product's streaming writer.
    """
    output = bytearray()
    for chunk in chunks:
        if type(chunk) is not bytes:
            msg = "report writer returned a non-byte chunk"
            raise TypeError(msg)
        output.extend(chunk)
    return bytes(output)


def _script_payload(html: bytes) -> bytes:
    """Extract the exact report-data payload from generated self-contained HTML.

    Args:
        html: Bytes produced by the trusted self-contained HTML writer.

    Returns:
        The UTF-8 bytes between the unique report-data script tags.

    Preconditions:
        ``html`` contains one report-data script with a closing script tag.
    Postconditions:
        The returned bytes are unchanged from the generated HTML payload.
    Essential behavior:
        Measures the text read by the browser before JSON parsing.
    """
    if html.count(REPORT_DATA_OPEN) != 1:
        msg = "HTML does not contain one report-data script"
        raise AssertionError(msg)
    start = html.index(REPORT_DATA_OPEN) + len(REPORT_DATA_OPEN)
    end = html.find(b"</script>", start)
    if end < start:
        msg = "report-data script has no closing tag"
        raise AssertionError(msg)
    payload = html[start:end]
    payload.decode("utf-8")
    return payload


def _renderer_escape_json(raw_json: bytes) -> bytes:
    """Apply the five documented script-safety substitutions to JSON text.

    Args:
        raw_json: Canonical UTF-8 JSON bytes from the report writer.

    Returns:
        JSON bytes with the renderer's five script-safety escapes.

    Preconditions:
        ``raw_json`` is valid UTF-8 JSON text.
    Postconditions:
        JSON values stay unchanged when the returned bytes are parsed.
    Essential behavior:
        Provides an output oracle for the HTML writer's exact substitutions.
    """
    text = raw_json.decode("utf-8")
    escaped = (
        text.replace("&", r"\u0026")
        .replace("<", r"\u003c")
        .replace(">", r"\u003e")
        .replace("\u2028", r"\u2028")
        .replace("\u2029", r"\u2029")
    )
    return escaped.encode("utf-8")


def _replace_report_payload(html: bytes, report: dict) -> bytes:
    """Build one controlled expanded-page fixture with a changed script value.

    Args:
        html: Trusted expanded HTML for the authored report.
        report: A detached expanded report with one routing case applied.

    Returns:
        HTML whose report-data payload contains the supplied report.

    Preconditions:
        ``html`` has one report-data script and ``report`` is a dict.
    Postconditions:
        Only the report-data payload changes; the HTML shell remains intact.
    Essential behavior:
        Creates the unsupported-version browser case without asking the Python
        renderer to accept an invalid storage envelope.
    """
    if type(report) is not dict:
        msg = "expanded browser fixture must be a dictionary"
        raise TypeError(msg)
    start = html.index(REPORT_DATA_OPEN) + len(REPORT_DATA_OPEN)
    end = html.find(b"</script>", start)
    if end < start or html.count(REPORT_DATA_OPEN) != 1:
        msg = "trusted HTML has no unique report-data payload"
        raise AssertionError(msg)
    payload = _renderer_escape_json(canonical_bytes(report))
    return html[:start] + payload + html[end:]


class BrowserReportDownloadTests(unittest.TestCase):
    """Exercise one local report across actual browser boundaries.

    Class members:
        browser: Local Chromium instance used only for generated HTML.
        expanded_html: Trusted expanded HTML for the one authored report.
        expanded_report: Detached logical report used for routing fixtures.
        provider_calls: Count of requests sent to the local FixtureProvider.
        report_data: Checked expanded report returned by the writer session.
        shared_html: Trusted self-contained shared report HTML.
        shared_json: Canonical shared output from the same checked handle.
        summaries: Numeric and boolean browser results; contains no report text.
        verification_dir: Fresh private directory for this test invocation.
    """

    browser: ClassVar[Browser | None] = None
    expanded_html: ClassVar[bytes] = b""
    expanded_report: ClassVar[dict] = {}
    expanded_path: ClassVar[Path | None] = None
    playwright: ClassVar[Playwright | None] = None
    provider_calls: ClassVar[int] = 0
    report_data: ClassVar[dict] = {}
    shared_html: ClassVar[bytes] = b""
    shared_json: ClassVar[bytes] = b""
    shared_path: ClassVar[Path | None] = None
    summaries: ClassVar[list[dict]] = []
    started_at: ClassVar[float] = 0.0
    setup_complete: ClassVar[bool] = False
    temporary_directory: ClassVar[tempfile.TemporaryDirectory | None] = None
    verification_dir: ClassVar[Path]

    @classmethod
    def _new_verification_directory(cls) -> Path:
        """Create fresh state for one invocation without reusing old budgets.

        Returns:
            A unique directory for this invocation's fixture, recovery state,
            browser files, and source-free summary.

        Preconditions:
            The optional artifact parent in ``DIFFSTORY_VERIFICATION_DIR`` is
            writable when it is set.
        Postconditions:
            The returned directory did not exist before this invocation.
        Essential behavior:
            Keeps prior recovery state intact and out of the current run.
        """
        parent_value = os.environ.get("DIFFSTORY_VERIFICATION_DIR")
        if parent_value:
            parent = Path(parent_value).expanduser()
            parent.mkdir(parents=True, exist_ok=True)
            return Path(
                tempfile.mkdtemp(prefix="browser-boundary-", dir=parent)
            )
        cls.temporary_directory = tempfile.TemporaryDirectory(
            prefix="diffstory-browser-boundary-"
        )
        return Path(cls.temporary_directory.name)

    @classmethod
    def _finalize_invocation(cls) -> None:
        """Write source-free results and release only this run's temp state.

        Preconditions:
            ``setUpClass`` created this invocation's verification directory.
        Postconditions:
            The summary stays in a configured artifact parent, or the system
            temporary directory is removed when no parent was configured.
        Essential behavior:
            Never removes a previous run or a caller-supplied parent directory.
        """
        summary = {
            "provider_calls": cls.provider_calls,
            "fixture": "golden_report_count_1",
            "provider": "local FixtureProvider",
            "setup_complete": cls.setup_complete,
            "shared_bytes": len(cls.shared_json),
            "expanded_html_bytes": len(cls.expanded_html),
            "shared_html_bytes": len(cls.shared_html),
            "browser_cases": cls.summaries,
            "runtime_seconds": round(time.perf_counter() - cls.started_at, 3)
            if cls.started_at
            else None,
        }
        (cls.verification_dir / "browser-boundary-summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        if cls.temporary_directory is not None:
            cls.temporary_directory.cleanup()
            cls.temporary_directory = None

    @classmethod
    def setUpClass(cls) -> None:
        """Generate one count-1 report and launch local Chromium.

        Preconditions:
            The repository virtualenv and Playwright dependency are available.
            ``DIFFSTORY_CHROMIUM`` may select an installed local executable.
        Postconditions:
            One local fixture report and its expanded/shared HTML are ready.
        Essential behavior:
            Reuses one generated proof graph for every browser boundary case.
        """
        cls.started_at = time.perf_counter()
        cls.verification_dir = cls._new_verification_directory()
        cls.addClassCleanup(cls._finalize_invocation)
        sys.stdout.write(f"verification_dir={cls.verification_dir}\n")
        sys.stdout.flush()

        source_report = golden_report(1)
        source_report["storage_version"] = "compiler-format-v7"
        source_report["opaque_browser_probe"] = {
            "$ref": "literal compiler metadata",
            "storage_version": "diffstory.report-storage.v88",
            "large_integer": LARGE_INTEGER,
            "cases": [
                {"name": name, "value": value}
                for name, value, _expected in ESCAPE_CASE_MATRIX
            ],
        }
        provider = FixtureProvider()
        narrator = Narrator(
            provider,
            architecture="v2",
            recovery_dir=cls.verification_dir / "recovery",
            concurrency=1,
        )
        narrator.preview(source_report)
        writer_session = narrator.capture_session(source_report)
        try:
            generation = narrator.generate_validated(
                source_report, session=writer_session
            )
            report_handle = writer_session.apply(generation)
            cls.shared_json = _collect(
                writer_session.iter_report_json(
                    report_handle,
                    report_storage="shared",
                    max_output_bytes=MAX_REPORT_BYTES,
                )
            )
            expanded_json = _collect(
                writer_session.iter_report_json(
                    report_handle,
                    report_storage="expanded",
                    max_output_bytes=MAX_REPORT_BYTES,
                )
            )
            cls.shared_html = _collect(
                iter_render_validated(
                    writer_session,
                    report_handle,
                    report_storage="shared",
                    max_output_bytes=MAX_REPORT_BYTES,
                )
            )
            cls.expanded_html = _collect(
                iter_render_validated(
                    writer_session,
                    report_handle,
                    report_storage="expanded",
                    max_output_bytes=MAX_REPORT_BYTES,
                )
            )
            cls.expanded_report = json.loads(expanded_json)
            cls.report_data = writer_session.report_data(report_handle)
        finally:
            cls.provider_calls = len(provider.calls)
            writer_session.close()

        cls.shared_path = cls.verification_dir / "shared-report.html"
        cls.shared_path.write_bytes(cls.shared_html)
        cls.expanded_path = cls.verification_dir / "expanded-report.html"
        cls.expanded_path.write_bytes(cls.expanded_html)
        cls.playwright = sync_playwright().start()
        cls.addClassCleanup(cls.playwright.stop)
        launch_options = {}
        chromium_override = os.environ.get("DIFFSTORY_CHROMIUM")
        if chromium_override:
            executable = Path(chromium_override).expanduser()
            if not executable.is_file():
                msg = "DIFFSTORY_CHROMIUM is not an installed executable"
                raise FileNotFoundError(msg)
            launch_options["executable_path"] = str(executable)
        cls.browser = cls.playwright.chromium.launch(
            **launch_options,
            headless=True,
            timeout=15_000,
            args=[
                "--disable-background-networking",
                "--disable-component-update",
                "--disable-default-apps",
                "--disable-sync",
                "--no-default-browser-check",
                "--no-first-run",
            ],
        )
        cls.addClassCleanup(cls.browser.close)
        cls.setup_complete = True

    def _open_page(
        self, name: str, html: bytes
    ) -> tuple[Page, BrowserContext, list[str], list[str], list[str]]:
        """Open one generated local page with external HTTP blocked and counted.

        Args:
            name: Safe short fixture label used for its private HTML filename.
            html: Self-contained HTML bytes for one browser case.

        Returns:
            A tuple of page, context, attempted requests, page errors, and
            console errors for the caller to assert and close.

        Preconditions:
            ``html`` is generated from the one count-1 report fixture.
        Postconditions:
            The page has loaded from a local file with network schemes blocked.
        Essential behavior:
            Makes browser network attempts observable and unable to leave local
            Chromium.
        """
        html_path = self.verification_dir / f"{name}.html"
        html_path.write_bytes(html)
        context = self.browser.new_context(accept_downloads=True)
        context.set_default_timeout(6_000)

        def block_network(route: Route) -> None:
            """Block network schemes while allowing the local file page.

            Args:
                route: One browser request intercepted by this context.

            Preconditions:
                The request has a parseable URL scheme.
            Postconditions:
                Network schemes are aborted; local schemes continue.
            Essential behavior:
                Prevents external access without treating ``file://`` as
                network traffic.
            """
            scheme = urlsplit(route.request.url).scheme.lower()
            if scheme in {"http", "https", "ws", "wss"}:
                route.abort()
            else:
                route.continue_()

        context.route("**/*", block_network)
        route_web_socket = getattr(context, "route_web_socket", None)
        if route_web_socket is not None:
            route_web_socket("**/*", lambda websocket: websocket.close())
        page = context.new_page()
        requests: list[str] = []
        page_errors: list[str] = []
        console_errors: list[str] = []

        def record_network_request(request: Request) -> None:
            """Record only attempted network schemes for this page.

            Args:
                request: One browser request event.

            Preconditions:
                ``request.url`` is the browser-provided request URL.
            Postconditions:
                The list contains network URLs and excludes local file URLs.
            Essential behavior:
                Counts external attempts without counting the document load.
            """
            if urlsplit(request.url).scheme in {"http", "https", "ws", "wss"}:
                requests.append(request.url)

        page.on("request", record_network_request)
        page.on("pageerror", lambda error: page_errors.append(str(error)))
        page.on(
            "console",
            lambda message: console_errors.append(message.text)
            if message.type == "error"
            else None,
        )
        page.goto(html_path.as_uri(), wait_until="load", timeout=10_000)
        return page, context, requests, page_errors, console_errors

    def _assert_source_preview(self, page: Page) -> bool:
        """Check one available source preview against supplied source lines.

        Args:
            page: A loaded shared report page for the one authored fixture.

        Returns:
            ``True`` when one source preview exists and renders supplied text;
            ``False`` only when the report has no source section.

        Preconditions:
            The page has loaded its self-contained report payload.
        Postconditions:
            An available preview is opened and its rendered lines are checked.
        Essential behavior:
            Confirms source preview output without expanding unrelated blocks.
        """
        source_sections = page.locator(".v2-source")
        if not source_sections.count():
            return False
        source_section = source_sections.first
        source_section.locator("summary").click()
        source_section.evaluate(
            """section => {
                const button = section.querySelector('button[data-load-preview]');
                if (button) button.click();
            }"""
        )
        rendered_lines = source_section.locator(".code-line .code-text")
        rendered_lines.first.wait_for(state="visible")
        rendered_source = rendered_lines.all_text_contents()
        supplied_lines = {
            line.strip()
            for fragment in self.report_data["source_evidence"]["fragments"]
            for line in fragment["text"].splitlines()
            if line.strip()
        }
        self.assertTrue(
            any(
                text.strip()
                and any(text.strip() in line for line in supplied_lines)
                for text in rendered_source
            ),
            "the browser did not display supplied source lines",
        )
        return True

    def test_shared_download_restores_the_exact_canonical_report_bytes(
        self,
    ) -> None:
        r"""Concern: escaped HTML script text must not corrupt a shared download.

        Input/expected matrix:
        * All five escaped characters plus Unicode -> the same logical string.
        * Literal ``\\u`` escape-looking text -> the same literal backslashes.
        * One, two, and three backslashes before escaped targets -> exact values.
        * Integer 9007199254740993 -> the same exact canonical integer token.
        """
        payload = _script_payload(self.shared_html)
        self.assertEqual(payload, _renderer_escape_json(self.shared_json))
        for escape in (
            b"\\u0026",
            b"\\u003c",
            b"\\u003e",
            b"\\u2028",
            b"\\u2029",
        ):
            with self.subTest(escaped_token=escape.decode("ascii")):
                self.assertIn(escape, payload)
        self.assertRegex(
            self.shared_json,
            rb'"large_integer"\s*:\s*9007199254740993',
        )

        page, context, requests, page_errors, console_errors = self._open_page(
            "shared-report", self.shared_html
        )
        downloaded_path = self.verification_dir / "actual-download.json"
        try:
            headings = page.locator(".story-section h2")
            self.assertGreater(headings.count(), 0)
            self.assertTrue(headings.first.inner_text().strip())
            source_rendered = self._assert_source_preview(page)

            page.locator("#moreBtn").click()
            with page.expect_download(timeout=10_000) as download_info:
                page.locator("#downloadReport").click()
            download_info.value.save_as(str(downloaded_path))
            downloaded_bytes = downloaded_path.read_bytes()
            self.assertEqual(downloaded_bytes, self.shared_json)

            imported_session, imported_generation = (
                ReportSession.import_indexed(
                    downloaded_bytes,
                    max_input_bytes=MAX_REPORT_BYTES,
                )
            )
            try:
                imported_handle = imported_session.apply(imported_generation)
                imported_report = imported_session.report_data(imported_handle)
                actual_probe = imported_report["opaque_browser_probe"]
                expected_cases = [
                    {"name": name, "value": expected}
                    for name, _value, expected in ESCAPE_CASE_MATRIX
                ]
                for index, (_name, value, expected) in enumerate(
                    ESCAPE_CASE_MATRIX
                ):
                    with self.subTest(matrix_case=index):
                        self.assertEqual(value, expected)
                        self.assertEqual(
                            actual_probe["cases"][index]["value"], expected
                        )
                self.assertEqual(actual_probe["cases"], expected_cases)
                self.assertEqual(actual_probe["large_integer"], LARGE_INTEGER)
                self.assertEqual(
                    actual_probe["$ref"], "literal compiler metadata"
                )
                self.assertEqual(
                    canonical_bytes(imported_report["source_evidence"]),
                    canonical_bytes(self.report_data["source_evidence"]),
                )
            finally:
                imported_session.close()

            self.assertEqual(requests, [])
            self.assertEqual(page_errors, [])
            self.assertEqual(console_errors, [])
            self.summaries.append(
                {
                    "case": "shared_download",
                    "network_requests": len(requests),
                    "page_errors": len(page_errors),
                    "console_errors": len(console_errors),
                    "download_bytes": len(downloaded_bytes),
                    "byte_equal": downloaded_bytes == self.shared_json,
                    "fresh_import_and_apply": True,
                    "source_rendered": source_rendered,
                }
            )
        finally:
            context.close()

    def test_reader_claims_only_reserved_top_level_storage_versions(
        self,
    ) -> None:
        """Concern: ordinary expanded metadata must not select shared storage.

        Input/expected matrix:
        * No top-level version -> ordinary expanded report renders.
        * ``compiler-format-v7`` string -> ordinary metadata remains intact.
        * Numeric top-level version -> ordinary expanded report renders.
        * Nested reserved text and a literal ``$ref`` -> ordinary metadata.
        * ``diffstory.report-storage.v99`` top-level string -> browser error.
        """
        base_report = copy.deepcopy(self.expanded_report)
        probe = base_report["opaque_browser_probe"]
        self.assertEqual(
            probe["storage_version"], "diffstory.report-storage.v88"
        )
        self.assertEqual(probe["$ref"], "literal compiler metadata")
        cases = [
            (
                "reader-control",
                NO_STORAGE_VERSION,
                {
                    "has_storage_version": False,
                    "storage_version": None,
                    "nested_storage_version": "diffstory.report-storage.v88",
                    "literal_ref": "literal compiler metadata",
                },
                None,
            ),
            (
                "reader-ordinary-string",
                "compiler-format-v7",
                {
                    "has_storage_version": True,
                    "storage_version": "compiler-format-v7",
                    "nested_storage_version": "diffstory.report-storage.v88",
                    "literal_ref": "literal compiler metadata",
                },
                None,
            ),
            (
                "reader-nonstring-version",
                7,
                {
                    "has_storage_version": True,
                    "storage_version": 7,
                    "nested_storage_version": "diffstory.report-storage.v88",
                    "literal_ref": "literal compiler metadata",
                },
                None,
            ),
            (
                "reader-unsupported-version",
                "diffstory.report-storage.v99",
                None,
                "Unsupported embedded report storage version",
            ),
        ]
        for name, version, expected, expected_error in cases:
            with self.subTest(case=name):
                report = copy.deepcopy(base_report)
                if version is NO_STORAGE_VERSION:
                    report.pop("storage_version", None)
                else:
                    report["storage_version"] = version
                html = _replace_report_payload(self.expanded_html, report)
                page, context, requests, page_errors, console_errors = (
                    self._open_page(name, html)
                )
                try:
                    if expected_error is None:
                        self.assertGreater(
                            page.locator(".story-section h2").count(), 0
                        )
                        actual = page.locator("#report-data").evaluate(
                            """element => {
                              const report = JSON.parse(element.textContent);
                              const probe = report.opaque_browser_probe;
                              return {
                                has_storage_version: Object.prototype.hasOwnProperty.call(report, 'storage_version'),
                                storage_version: report.storage_version ?? null,
                                nested_storage_version: probe.storage_version,
                                literal_ref: probe['$ref']
                              };
                            }"""
                        )
                        self.assertEqual(actual, expected)
                        self.assertEqual(page_errors, [])
                        self.assertEqual(console_errors, [])
                    else:
                        self.assertTrue(
                            any(
                                expected_error in error
                                for error in page_errors
                            ),
                            page_errors,
                        )
                    self.assertEqual(requests, [])
                    self.summaries.append(
                        {
                            "case": name,
                            "network_requests": len(requests),
                            "page_errors": len(page_errors),
                            "console_errors": len(console_errors),
                            "expected_error": expected_error is not None,
                            "passed": True,
                        }
                    )
                finally:
                    context.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
