"""Check shared report storage in local Chromium with authored fixtures only.

Run with ``PYTHONPATH=.:tests`` and ``DIFFSTORY_CHROMIUM`` set to an already
cached Chromium executable. The script makes trusted expanded/shared outputs,
loads them offline, and writes all generated artifacts beneath
``DIFFSTORY_VERIFICATION_DIR`` when that variable is set.
"""

from __future__ import annotations

import copy
import json
import os
import signal
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from browser_narrative_v2 import Checks
from browser_narrative_v2 import check_page
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright
from test_reader import UNSAFE_TEXT
from test_reader import MixedTerminologyReaderFixtureProvider
from test_reader import encode_omitted_terminology_defaults
from test_reader import exact_reader_options
from test_reader import exact_reader_provider
from test_reader import exact_reader_snapshot
from test_reader import reader_golden_report

from diffstory.analysis import compile_snapshot
from diffstory.narrative import Narrator
from diffstory.narrative_session import ReportSession
from diffstory.render import iter_render_validated
from diffstory.report_storage import STORAGE_VERSION
from diffstory.report_storage import parse_report_storage

if TYPE_CHECKING:
    from playwright.sync_api import Browser
    from playwright.sync_api import Page


MAX_BYTES = 64 * 1024 * 1024
TABLES = sorted(
    (
        "results",
        "plans",
        "evidence_refs",
        "source_targets",
        "resolved_citations",
        "urls",
        "runs",
        "invocations",
        "requests",
        "producers",
        "receipts",
        "readers",
        "inherited_provenance",
    )
)
REPORT_DATA_OPEN = '<script type="application/json" id="report-data">'


def collect(chunks: object) -> bytes:
    """Join a trusted byte stream for a browser fixture.

    Preconditions: ``chunks`` yields only byte strings from a bounded writer.
    Postconditions: Returns those bytes in their original order.
    Essential behavior: Keeps test-only collection outside the product writer.
    """
    output = bytearray()
    for chunk in chunks:  # type: ignore[union-attr]
        if type(chunk) is not bytes:
            msg = "writer returned a non-byte chunk"
            raise TypeError(msg)
        output.extend(chunk)
    return bytes(output)


def embedded_json(html: bytes) -> dict:
    """Read the generated application/json payload from one authored HTML.

    Preconditions: ``html`` is output from the trusted self-contained writer.
    Postconditions: Returns the parsed embedded JSON object.
    Essential behavior: Confirms HTML escaping preserves a complete JSON value.
    """
    text = html.decode("utf-8")
    start = text.index(REPORT_DATA_OPEN) + len(REPORT_DATA_OPEN)
    end = text.index("</script>", start)
    value = json.loads(text[start:end])
    if type(value) is not dict:
        msg = "embedded report data is not an object"
        raise AssertionError(msg)
    return value


def trusted_reader_outputs(  # noqa: C901, PLR0915 # One source-free trusted writer/import lifecycle.
    source_report: dict,
    directory: Path,
    *,
    provider: object | None = None,
    options: object | None = None,
) -> tuple[dict, bytes, bytes, bytes, dict]:
    """Create a validated report and both HTML modes through session writers.

    Preconditions: ``source_report`` is an authored compiler report; optional
        provider and options use the existing deterministic local test helpers.
    Postconditions: Returns its expanded oracle, complete shared JSON, expanded
        HTML, shared HTML, and numeric storage facts after an independent reload.
    Essential behavior: The same shared bytes are independently imported and
        applied before either HTML stream is produced.
    """
    directory.mkdir(parents=True, exist_ok=True)
    source = copy.deepcopy(source_report)
    source["opaque_browser_probe"] = {
        "$ref": "literal compiler metadata",
        "large_text": "",
    }
    recovery = directory / "recovery"
    narrator = Narrator(
        provider or MixedTerminologyReaderFixtureProvider(),
        architecture="v2",
        recovery_dir=recovery,
        concurrency=1,
        v2_options=options,
    )
    narrator.preview(source)
    writer_session = narrator.capture_session(source)
    imported_session = None
    try:
        generation = narrator.generate_validated(
            source, session=writer_session
        )
        report_handle = writer_session.apply(generation)
        expanded_json = collect(
            writer_session.iter_report_json(
                report_handle,
                report_storage="expanded",
                max_output_bytes=MAX_BYTES,
            )
        )
        writer_shared_json = collect(
            writer_session.iter_report_json(
                report_handle,
                report_storage="shared",
                max_output_bytes=MAX_BYTES,
            )
        )
        expanded_oracle = json.loads(expanded_json)
        if expanded_oracle != writer_session.report_data(report_handle):
            msg = "expanded writer differs from its checked report"
            raise AssertionError(msg)

        writer_envelope = json.loads(writer_shared_json)
        writer_rows = [
            row
            for row in writer_envelope["tables"]["results"]
            if row["kind"] in {"section", "document"}
        ]
        if not any(row["record"].get("terminology") for row in writer_rows):
            msg = (
                "trusted writer omitted the authored nonempty terminology case"
            )
            raise AssertionError(msg)
        if not any(
            row["record"].get("terminology") == [] for row in writer_rows
        ):
            msg = "trusted writer omitted the authored empty-default case"
            raise AssertionError(msg)
        if parse_report_storage(writer_shared_json).artifact_kind != "report":
            msg = "trusted shared writer did not emit a report envelope"
            raise AssertionError(msg)

        # Re-encode detached logical data with empty defaults omitted before
        # importing it. The strict writer's typed normalization fills them.
        shared_json = encode_omitted_terminology_defaults(expanded_oracle)

        imported_session, imported_generation = ReportSession.import_indexed(
            shared_json,
            max_input_bytes=MAX_BYTES,
        )
        imported_handle = imported_session.apply(imported_generation)
        roundtrip_shared = collect(
            imported_session.iter_report_json(
                imported_handle,
                report_storage="shared",
                max_output_bytes=MAX_BYTES,
            )
        )
        if roundtrip_shared != shared_json:
            msg = "offline import changed omitted shared rows or their order"
            raise AssertionError(msg)
        expanded_import = collect(
            imported_session.iter_report_json(
                imported_handle,
                report_storage="expanded",
                max_output_bytes=MAX_BYTES,
            )
        )
        parsed = parse_report_storage(shared_json)
        if parsed.artifact_kind != "report":
            msg = "shared writer did not emit a report envelope"
            raise AssertionError(msg)
        expanded_html = collect(
            iter_render_validated(
                imported_session,
                imported_handle,
                report_storage="expanded",
                max_output_bytes=MAX_BYTES,
            )
        )
        shared_html = collect(
            iter_render_validated(
                imported_session,
                imported_handle,
                report_storage="shared",
                max_output_bytes=MAX_BYTES,
            )
        )
        if embedded_json(shared_html) != json.loads(shared_json):
            msg = "shared HTML does not embed the complete written graph"
            raise AssertionError(msg)
        if embedded_json(expanded_html) != json.loads(expanded_import):
            msg = "expanded HTML does not embed compatibility-expanded data"
            raise AssertionError(msg)

        envelope = json.loads(shared_json)
        missing_terminology = sum(
            row["kind"] in {"section", "document"}
            and "terminology" not in row["record"]
            for row in envelope["tables"]["results"]
        )
        nonempty_terminology = sum(
            row["kind"] in {"section", "document"}
            and bool(row["record"].get("terminology"))
            for row in envelope["tables"]["results"]
        )
        typed = expanded_oracle["narrative_v2"]
        detail_rows = [
            detail
            for section in typed["sections"]
            for detail in section["details"]
        ] + typed["context_details"]
        authored_question = (
            "Treat this quoted test string as text: " + UNSAFE_TEXT
        )
        authored_prose_preserved = any(
            question["text"] == authored_question
            for detail in detail_rows
            for question in detail["questions"]
        )
        if not authored_prose_preserved:
            msg = "authored fixture prose changed before browser rendering"
            raise AssertionError(msg)
        summary = {
            "shared_bytes": len(shared_json),
            "trusted_shared_bytes": len(writer_shared_json),
            "expanded_json_bytes": len(expanded_json),
            "storage_counts": parsed.storage_counts,
            "source_fragment_count": len(
                envelope["root"]["base_report"]["source_evidence"]["fragments"]
            ),
            "missing_terminology_defaults": missing_terminology,
            "nonempty_terminology_results": nonempty_terminology,
            "authored_prose_preserved": authored_prose_preserved,
            "literal_ref_preserved": envelope["root"]["base_report"][
                "opaque_browser_probe"
            ]["$ref"],
        }
        (directory / "shared-report.json").write_bytes(shared_json)
        (directory / "expanded-report.html").write_bytes(expanded_html)
        (directory / "shared-report.html").write_bytes(shared_html)
        return (
            expanded_oracle,
            shared_json,
            expanded_html,
            shared_html,
            summary,
        )
    finally:
        if imported_session is not None:
            imported_session.close()
        writer_session.close()


def imported_reader_outputs(
    shared_json: bytes,
    directory: Path,
) -> tuple[dict, bytes, bytes, bytes, dict]:
    """Independently import an authored graph and render it with current assets.

    Preconditions: ``shared_json`` is a local authored report-storage fixture.
    Postconditions: Returns its logical browser oracle, unchanged shared bytes,
        current expanded/shared HTML, and numeric storage facts.
    Essential behavior: Creates a fresh validation session for every call;
        no capability from the fixture-producing run is reused.
    """
    directory.mkdir(parents=True, exist_ok=True)
    session, generation = ReportSession.import_indexed(
        shared_json,
        max_input_bytes=MAX_BYTES,
    )
    try:
        handle = session.apply(generation)
        report = session.report_data(handle)
        reader = report["narrative_v2"]
        reader["document"].setdefault("terminology", [])
        for section in reader["sections"]:
            section["prose"].setdefault("terminology", [])
        roundtrip = collect(
            session.iter_report_json(
                handle,
                report_storage="shared",
                max_output_bytes=MAX_BYTES,
            )
        )
        if roundtrip != shared_json:
            msg = "fresh import changed authored shared rows or their order"
            raise AssertionError(msg)
        expanded_json = collect(
            session.iter_report_json(
                handle,
                report_storage="expanded",
                max_output_bytes=MAX_BYTES,
            )
        )
        expanded_html = collect(
            iter_render_validated(
                session,
                handle,
                report_storage="expanded",
                max_output_bytes=MAX_BYTES,
            )
        )
        shared_html = collect(
            iter_render_validated(
                session,
                handle,
                report_storage="shared",
                max_output_bytes=MAX_BYTES,
            )
        )
        envelope = json.loads(shared_json)
        if embedded_json(expanded_html) != json.loads(expanded_json):
            msg = "current expanded HTML lost compatibility-expanded data"
            raise AssertionError(msg)
        if embedded_json(shared_html) != envelope:
            msg = "current shared HTML lost the authored report graph"
            raise AssertionError(msg)
        result_rows = envelope["tables"]["results"]
        missing_terminology = sum(
            row["kind"] in {"section", "document"}
            and "terminology" not in row["record"]
            for row in result_rows
        )
        nonempty_terminology = sum(
            row["kind"] in {"section", "document"}
            and bool(row["record"].get("terminology"))
            for row in result_rows
        )
        detail_rows = [
            detail
            for section in reader["sections"]
            for detail in section["details"]
        ] + reader["context_details"]
        authored_question = (
            "Treat this quoted test string as text: " + UNSAFE_TEXT
        )
        authored_prose_preserved = any(
            question["text"] == authored_question
            for detail in detail_rows
            for question in detail["questions"]
        )
        if not authored_prose_preserved:
            msg = "fresh import changed authored fixture prose"
            raise AssertionError(msg)
        summary = {
            "shared_bytes": len(shared_json),
            "expanded_json_bytes": len(expanded_json),
            "storage_counts": parse_report_storage(shared_json).storage_counts,
            "source_fragment_count": len(
                envelope["root"]["base_report"]["source_evidence"]["fragments"]
            ),
            "missing_terminology_defaults": missing_terminology,
            "nonempty_terminology_results": nonempty_terminology,
            "authored_prose_preserved": authored_prose_preserved,
            "literal_ref_preserved": envelope["root"]["base_report"][
                "opaque_browser_probe"
            ]["$ref"],
        }
        (directory / "shared-report.json").write_bytes(shared_json)
        (directory / "expanded-report.html").write_bytes(expanded_html)
        (directory / "shared-report.html").write_bytes(shared_html)
        return report, shared_json, expanded_html, shared_html, summary
    finally:
        session.close()


def check_shared_summary(summary: dict, report: dict, check: Checks) -> None:
    """Check typed envelope closure and natural omitted defaults.

    Preconditions: ``summary`` came from the application/json node in a shared
        report page, and ``report`` is its independently imported oracle.
    Postconditions: Every storage table, proof family, source fragment, literal
        compiler marker, and omitted terminology default is accounted for.
    Essential behavior: Checks only numeric counts and a test marker; it does
        not retain or print graph rows or narrative text.
    """
    check(summary["shared"], "the browser loaded a shared report envelope")
    check(
        summary["storageVersion"] == STORAGE_VERSION,
        "the browser sees report-storage.v1",
    )
    check(
        summary["artifactKind"] == "report",
        "the embedded graph remains a report",
    )
    check(
        summary["tableNames"] == TABLES,
        "all thirteen typed tables remain embedded",
    )
    check(
        summary["sourceFragmentCount"]
        == len(report["source_evidence"]["fragments"]),
        "all supplied source fragments remain in the browser payload",
    )
    check(
        summary["requestCount"] > 0,
        "the complete request proof remains embedded",
    )
    check(
        summary["producerCount"] > 0,
        "the complete producer proof remains embedded",
    )
    check(
        summary["receiptCount"] > 0,
        "the complete receipt proof remains embedded",
    )
    check(
        summary["literalCompilerRef"] == "literal compiler metadata",
        "a compiler $ref-shaped value stays literal",
    )
    check(
        summary["omittedTerminologyDefaults"] > 0,
        "the fixture keeps optional terminology defaults absent in stored rows",
    )


def visible_dom_text(browser: Browser, html: bytes) -> str:
    """Return story text without non-rendered JSON or executable script text.

    Preconditions: ``html`` is one trusted self-contained authored fixture.
    Postconditions: Returns its DOM text in order, excluding script nodes.
    Essential behavior: Keeps expanded/shared comparisons focused on rendered
        content rather than their intentionally different embedded payloads.
    """
    page = browser.new_page(viewport={"width": 1280, "height": 900})
    try:
        page.set_content(html.decode("utf-8"), wait_until="domcontentloaded")
        return page.evaluate("""() => {
          const body = document.body.cloneNode(true);
          body.querySelectorAll('script').forEach(node => node.remove());
          return body.textContent;
        }""")
    finally:
        page.close()


def _metric_page(  # noqa: C901, PLR0912, PLR0915 # One measured page has one serial load/click/capture lifecycle.
    browser: Browser,
    report: dict,
    html: bytes,
    *,
    expected_shared: bool,
) -> tuple[dict, dict]:
    """Measure startup and one full-source interaction on one browser page.

    Preconditions: ``html`` is trusted writer output for ``report`` and
        ``browser`` is a local Playwright Chromium instance.
    Postconditions: Returns numeric test-hook values and an in-memory DOM
        signature that the caller compares but does not serialize.
    Essential behavior: Records attempted requests and rejects page or console
        errors while preserving the exact cited preview and full source.
    """
    fragments = {
        fragment["key"]: fragment
        for fragment in report["source_evidence"]["fragments"]
    }
    citation = next(
        item
        for item in report["narrative_v2"]["citations"]
        if fragments[item["fragment_key"]]["text"]
    )
    metric_html = html.decode("utf-8")
    report_script = '<script type="application/json" id="report-data">'
    if metric_html.count(report_script) != 1:
        msg = "trusted reader output has no unique report-data script"
        raise AssertionError(msg)
    metric_bootstrap = (
        "<script>window.__DIFFSTORY_TEST_METRICS__ = true;</script>"
    )
    metric_html = metric_html.replace(
        report_script, metric_bootstrap + report_script, 1
    )
    page: Page = browser.new_page(viewport={"width": 1280, "height": 900})
    page.set_default_timeout(6500)
    requests: list[str] = []
    page_errors: list[str] = []
    console_errors: list[str] = []
    page.on("request", lambda request: requests.append(request.url))
    page.on("pageerror", lambda error: page_errors.append(str(error)))
    page.on(
        "console",
        lambda message: console_errors.append(message.text)
        if message.type == "error"
        else None,
    )
    page.set_content(metric_html, wait_until="domcontentloaded")
    page.locator(".story-section").first.wait_for(state="visible")
    citation_link = page.locator(f'a[data-citation="{citation["key"]}"]').first
    if citation_link.count():
        citation_link.click()
    else:
        page.locator(f"#source-{citation['key']} > summary").click()
    block = page.locator(f"#source-{citation['key']} .exact-code")
    if block.get_attribute("data-loaded") != "true":
        preview = block.locator("[data-load-preview]")
        if preview.count():
            try:
                preview.click(timeout=1500)
            except PlaywrightTimeoutError:
                if block.get_attribute("data-loaded") != "true":
                    raise
        elif block.get_attribute("data-loaded") != "true":
            msg = "source preview was neither rendered nor available to load"
            raise AssertionError(msg)
    expand = block.locator("[data-expand]")
    if expand.count():
        expand.click()
    metrics = page.evaluate("window.__DIFFSTORY_READER_METRICS__ || null")
    signature = page.evaluate("""() => ({
      text: (() => {
        const body = document.body.cloneNode(true);
        body.querySelectorAll('script').forEach(node => node.remove());
        return body.textContent;
      })(),
      fields: [...document.querySelectorAll('[data-field]')].map(node => [node.dataset.field, node.textContent]),
      links: [...document.querySelectorAll('a[href]')].map(node => [node.getAttribute('href'), node.textContent, node.dataset.citation || null, node.dataset.consumption || null]),
      sections: [...document.querySelectorAll('.story-section')].map(node => node.id),
      contextDetails: [...document.querySelectorAll('.v2-context .v2-detail')].map(node => node.dataset.detail),
      catalogSources: [...document.querySelectorAll('.v2-catalog .v2-source')].map(node => node.id.replace(/^source-/, '')),
      anchors: [...document.querySelectorAll('[id]')].map(node => node.id),
      sources: [...document.querySelectorAll('.v2-source')].map(node => [node.id, node.querySelector('summary')?.textContent || '', node.querySelector('.exact-code')?.dataset.fragment || '', node.querySelector('.exact-code')?.dataset.side || '', node.querySelector('.code-source-link')?.getAttribute('href') || ''])
    })""")
    if requests or page_errors or console_errors:
        page.close()
        msg = f"browser requests/errors: {len(requests)}/{len(page_errors)}/{len(console_errors)}"
        raise AssertionError(msg)
    typed = report["narrative_v2"]
    expected_context = [
        detail["unit_key"] for detail in typed["context_details"]
    ]
    expected_catalog = [citation["key"] for citation in typed["citations"]]
    if signature["contextDetails"] != expected_context:
        page.close()
        msg = "retained context detail descendants differ from stored order"
        raise AssertionError(msg)
    if signature["catalogSources"] != expected_catalog:
        page.close()
        msg = "source catalog descendants differ from stored order"
        raise AssertionError(msg)
    if not isinstance(metrics, dict):
        page.close()
        msg = "test-only browser metrics were not published"
        raise AssertionError(msg)
    if any(
        value is not None and type(value) not in {int, float}
        for value in metrics.values()
    ):
        page.close()
        msg = "browser metrics exposed a nonnumeric value"
        raise AssertionError(msg)
    if (
        bool(metrics["storageResultRecordCount"] is not None)
        != expected_shared
    ):
        page.close()
        msg = "browser metrics report the wrong storage path"
        raise AssertionError(msg)
    if metrics["embeddedJSONBytes"] <= 0 or metrics["jsonParseMs"] < 0:
        page.close()
        msg = "browser parse metrics are invalid"
        raise AssertionError(msg)
    if metrics["sectionProjectionCount"] != len(
        report["narrative_v2"]["sections"]
    ):
        page.close()
        msg = "browser section projection count is incomplete"
        raise AssertionError(msg)
    if metrics["sourceDecodeRenderCount"] <= 0:
        page.close()
        msg = "browser did not measure cited source loading"
        raise AssertionError(msg)
    page.close()
    numeric = {
        **metrics,
        "request_count": len(requests),
        "page_error_count": len(page_errors),
        "console_error_count": len(console_errors),
    }
    return numeric, signature


def _timeout_handler(_signum: int, _frame: object) -> None:
    """Stop one browser fidelity run at its 300-second contract limit."""
    msg = "browser fidelity run exceeded 300 seconds"
    raise TimeoutError(msg)


def _run() -> None:  # noqa: C901, PLR0912, PLR0915 # One capped serial browser acceptance run.
    """Run local writer, Chromium parity, closure and fixture measurements.

    Preconditions: Playwright and the cached Chromium binary are available.
    Postconditions: The private verification folder contains only generated
        authored-fixture HTML, screenshots, numeric metrics, and summaries.
    Essential behavior: No network service, live provider, private report, or
        historical benchmark input is opened.
    """
    parent = Path(
        os.environ.get("DIFFSTORY_VERIFICATION_DIR", tempfile.gettempdir())
    )
    parent.mkdir(parents=True, exist_ok=True)
    output = Path(
        tempfile.mkdtemp(prefix="diffstory-m6u3-browser-", dir=parent)
    )
    check = Checks()

    fixture_root = os.environ.get("DIFFSTORY_AUTHORED_FIXTURE_ROOT")
    if fixture_root:
        authored = Path(fixture_root)
        golden = imported_reader_outputs(
            (authored / "golden-small" / "shared-report.json").read_bytes(),
            output / "golden-small",
        )
        exact = imported_reader_outputs(
            (authored / "exact" / "shared-report.json").read_bytes(),
            output / "exact",
        )
    else:
        golden = trusted_reader_outputs(
            reader_golden_report(), output / "golden-small"
        )
        exact = trusted_reader_outputs(
            compile_snapshot(exact_reader_snapshot()),
            output / "exact",
            provider=exact_reader_provider(),
            options=exact_reader_options(),
        )
    fixture_cases = [
        ("golden", golden),
        ("exact", exact),
    ]
    fidelity: dict[str, dict] = {}
    measurements: list[dict] = []

    with sync_playwright() as playwright:
        launch_options = {"headless": True}
        executable = os.environ.get("DIFFSTORY_CHROMIUM")
        if executable:
            launch_options["executable_path"] = executable
        browser = playwright.chromium.launch(**launch_options)
        try:
            for fixture_name, pair in fixture_cases:
                (
                    report,
                    shared_json,
                    expanded_html,
                    shared_html,
                    source_summary,
                ) = pair
                signatures = {}
                visible_texts = {}
                summaries = {}
                for storage, html in (
                    ("expanded", expanded_html),
                    ("shared", shared_html),
                ):
                    page_dir = output / fixture_name / storage
                    page_dir.mkdir(parents=True, exist_ok=True)
                    page_result = check_page(
                        browser,
                        report,
                        html.decode("utf-8"),
                        page_dir,
                        check,
                        capture_dom_signature=True,
                        capture_embedded_summary=storage == "shared",
                    )
                    signatures[storage] = page_result.pop("_dom_signature")
                    signatures[storage].pop("text")
                    visible_texts[storage] = visible_dom_text(browser, html)
                    if storage == "shared":
                        summaries[storage] = page_result.pop(
                            "_embedded_summary"
                        )
                        check_shared_summary(summaries[storage], report, check)
                    fidelity[f"{fixture_name}_{storage}"] = {
                        "catalog_entries": page_result["catalog_entries"],
                        "inspected_entries": page_result["inspected_entries"],
                        "full_fragments": page_result["full_fragments"],
                        "empty_fragments": page_result["empty_fragments"],
                        "half_crlf_endpoints": page_result[
                            "half_crlf_endpoints"
                        ],
                        "same_line_ranges": page_result["same_line_ranges"],
                        "chunked_fragments": page_result["chunked_fragments"],
                        "request_count": len(page_result["network_requests"]),
                        "page_error_count": len(page_result["page_errors"]),
                        "console_error_count": len(
                            page_result["console_errors"]
                        ),
                        "screenshot_count": len(page_result["screenshots"]),
                    }
                check(
                    signatures["expanded"] == signatures["shared"]
                    and visible_texts["expanded"] == visible_texts["shared"],
                    f"{fixture_name} expanded/shared rendered text, fields, labels, URLs, and anchors match",
                )
                if fixture_name == "golden":
                    for storage in ("expanded", "shared"):
                        check(
                            fidelity[f"{fixture_name}_{storage}"][
                                "empty_fragments"
                            ]
                            == 2,
                            f"{storage} path preserves both empty supplied fragments",
                        )
                    check(
                        source_summary["missing_terminology_defaults"] > 0,
                        "trusted shared fixture contains omitted optional result defaults",
                    )
                    check(
                        source_summary["nonempty_terminology_results"] > 0,
                        "trusted shared fixture also contains nonempty terminology",
                    )
                    check(
                        source_summary["authored_prose_preserved"],
                        "trusted writer preserves the fixture's authored prose",
                    )
                else:
                    for storage in ("expanded", "shared"):
                        row = fidelity[f"{fixture_name}_{storage}"]
                        check(
                            row["half_crlf_endpoints"] > 0,
                            f"{storage} keeps CRLF byte endpoints exact",
                        )
                        check(
                            row["same_line_ranges"] > 1,
                            f"{storage} keeps distinct same-line source slices",
                        )

            for fixture_name, pair in fixture_cases:
                report, shared_json, expanded_html, shared_html, _summary = (
                    pair
                )
                paired_metrics = {}
                paired_signatures = {}
                for storage, html in (
                    ("expanded", expanded_html),
                    ("shared", shared_html),
                ):
                    metric_values, signature = _metric_page(
                        browser,
                        report,
                        html,
                        expected_shared=storage == "shared",
                    )
                    paired_metrics[storage] = metric_values
                    paired_signatures[storage] = signature
                check(
                    paired_signatures["expanded"]
                    == paired_signatures["shared"],
                    f"{fixture_name} browser DOM remains equal across storage modes",
                )
                for storage in ("expanded", "shared"):
                    values = paired_metrics[storage]
                    check(
                        values["request_count"] == 0,
                        f"{fixture_name} {storage} made no network attempts",
                    )
                    check(
                        values["page_error_count"] == 0,
                        f"{fixture_name} {storage} had no page errors",
                    )
                    check(
                        values["console_error_count"] == 0,
                        f"{fixture_name} {storage} had no console errors",
                    )
                    check(
                        values["readerReferenceCount"] > 0,
                        f"{fixture_name} {storage} counted reader selectors",
                    )
                    check(
                        values["readerResultCount"] > 0,
                        f"{fixture_name} {storage} counted reader results",
                    )
                    measurements.append(
                        {
                            "fixture": fixture_name,
                            "sharedReportBytes": len(shared_json),
                            "storage": storage,
                            **values,
                        }
                    )
        finally:
            browser.close()

    json_sizes = [
        row["embeddedJSONBytes"]
        for row in measurements
        if row["storage"] == "expanded"
    ]
    check(
        json_sizes == sorted(json_sizes),
        "the authored exact fixture is larger than the golden fixture",
    )
    measurement_artifact = {
        "source_free": True,
        "reused_authored_fixtures": bool(fixture_root),
        "fixtures": [fixture_name for fixture_name, _pair in fixture_cases],
        "measurements": measurements,
    }
    (output / "measurements.json").write_text(
        json.dumps(measurement_artifact, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    result = {
        "status": "PASS",
        "assertions": check.count,
        "authored_fixture_only": True,
        "live_provider_calls": 0,
        "network_attempts": 0,
        "fixture_pages": len(fidelity),
        "measurement_rows": len(measurements),
        "fidelity": fidelity,
    }
    (output / "result.json").write_text(
        json.dumps(result, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    """Apply the hard cap and run the source-free Chromium harness.

    Preconditions: The caller has supplied local Playwright and cached Chromium.
    Postconditions: The harness stops within 300 seconds and leaves only its
        authored-fixture artifacts under the configured verification folder.
    Essential behavior: No browser or provider work continues past the run cap.
    """
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(300)
    try:
        _run()
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    main()
