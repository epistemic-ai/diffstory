"""Check shared report storage in local Chromium with authored fixtures only.

Run with ``PYTHONPATH=.:tests`` and ``DIFFSTORY_CHROMIUM`` set to an already
cached Chromium executable. The script makes trusted expanded/shared outputs,
loads them offline, and writes all generated artifacts beneath
``DIFFSTORY_VERIFICATION_DIR`` when that variable is set.
"""

from __future__ import annotations

import contextlib
import copy
import json
import math
import os
import signal
import statistics
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from browser_narrative_v2 import Checks
from browser_narrative_v2 import check_page
from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright
from test_narrative_v2 import golden_report
from test_reader import UNSAFE_TEXT
from test_reader import MixedTerminologyReaderFixtureProvider
from test_reader import encode_omitted_terminology_defaults
from test_reader import exact_reader_options
from test_reader import exact_reader_provider
from test_reader import exact_reader_snapshot
from test_reader import reader_golden_report

from diffstory.analysis import compile_snapshot
from diffstory.narrative import Narrator
from diffstory.narrative_manifest import GENERATION_STORAGE_VERSION
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
REPORT_DATA_OPEN_BYTES = REPORT_DATA_OPEN.encode("ascii")


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
    value = json.loads(_embedded_script_payload_bytes(html).decode("utf-8"))
    if type(value) is not dict:
        msg = "embedded report data is not an object"
        raise AssertionError(msg)
    return value


def _embedded_script_payload_bytes(html: bytes) -> bytes:
    """Extract the exact UTF-8 payload consumed by the report-data script.

    Preconditions: ``html`` is one trusted self-contained writer output.
    Postconditions: Returns bytes between its unique report-data opening tag
        and the following closing script tag, without JSON re-serialization.
    Essential behavior: Measures HTML-escaped script text at the same boundary
        as the browser's ``TextEncoder(script.textContent)`` call.
    """
    if html.count(REPORT_DATA_OPEN_BYTES) != 1:
        msg = "trusted reader output has no unique report-data script"
        raise AssertionError(msg)
    start = html.index(REPORT_DATA_OPEN_BYTES) + len(REPORT_DATA_OPEN_BYTES)
    end = html.find(b"</script>", start)
    if end < start:
        msg = "trusted reader output has no closing report-data script tag"
        raise AssertionError(msg)
    payload = html[start:end]
    payload.decode("utf-8")
    return payload


def trusted_reader_outputs(  # noqa: C901, PLR0915 # One source-free trusted writer/import lifecycle.
    source_report: dict,
    directory: Path,
    *,
    provider: object | None = None,
    options: object | None = None,
    include_html_escape_probe: bool = False,
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
    browser_probe = {
        "$ref": "literal compiler metadata",
        "large_text": "",
    }
    if include_html_escape_probe:
        browser_probe["html_escape_probe"] = "<&> café \u2028\u2029"
    source["opaque_browser_probe"] = browser_probe
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
            "expanded_embedded_script_bytes": len(
                _embedded_script_payload_bytes(expanded_html)
            ),
            "shared_embedded_script_bytes": len(
                _embedded_script_payload_bytes(shared_html)
            ),
            "expanded_generation_storage_version": expanded_oracle[
                "generation"
            ].get("storage_version"),
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
            "expanded_embedded_script_bytes": len(
                _embedded_script_payload_bytes(expanded_html)
            ),
            "shared_embedded_script_bytes": len(
                _embedded_script_payload_bytes(shared_html)
            ),
            "expanded_generation_storage_version": json.loads(expanded_json)[
                "generation"
            ].get("storage_version"),
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


def _retained_heap(page: Page) -> tuple[int | None, int | None]:
    """Collect supported retained JavaScript heap values after source expansion.

    Preconditions: ``page`` is a live local Chromium page after its measured
        source interaction.
    Postconditions: Returns retained V8 heap and backing-storage bytes, with
        null values when Chromium does not support the CDP fields.
    Essential behavior: Forces CDP garbage collection before reading usage; it
        does not treat the result as a peak or include native DOM memory.
    """
    client = None
    try:
        client = page.context.new_cdp_session(page)
        client.send("HeapProfiler.enable")
        client.send("HeapProfiler.collectGarbage")
        usage = client.send("Runtime.getHeapUsage")
    except PlaywrightError:
        return None, None
    finally:
        if client is not None:
            with contextlib.suppress(PlaywrightError):
                client.detach()
    values = []
    for field in (
        "usedSize",
        "backingStorageSize",
    ):
        value = usage.get(field)
        if type(value) not in {int, float} or not math.isfinite(value):
            values.append(None)
        else:
            values.append(round(value))
    return values[0], values[1]


def _median_range(values: list[object]) -> dict:
    """Summarize numeric repeated samples without inventing unavailable values.

    Preconditions: ``values`` contains finite numbers or explicit ``None``.
    Postconditions: Returns median, minimum, maximum, and count for numeric
        observations; all three statistics are null when none are available.
    Essential behavior: Keeps each raw sample in the artifact and summarizes
        only matching fixture, format, and metric rows.
    """
    samples = [
        value
        for value in values
        if type(value) in {int, float} and math.isfinite(value)
    ]
    if not samples:
        return {"median": None, "min": None, "max": None, "sample_count": 0}
    return {
        "median": statistics.median(samples),
        "min": min(samples),
        "max": max(samples),
        "sample_count": len(samples),
    }


def _summarize_measurement_group(
    rows: list[dict], expected_repeats: int
) -> dict:
    """Summarize one complete fixture/storage repeat group.

    Preconditions: ``rows`` contains only one fixture/storage pair and the
        expected repeat count is from 1 through 5.
    Postconditions: Returns numeric median/range/count summaries, with null
        statistics only when every observation for a metric is unavailable.
    Essential behavior: Rejects missing, extra, malformed, or partly null
        observations before any median can imply a complete sample group.
    """
    if not 1 <= expected_repeats <= 5 or len(rows) != expected_repeats:
        msg = (
            "a fixture/storage summary requires exactly its expected repeat "
            f"count ({len(rows)} observed, {expected_repeats} expected)"
        )
        raise AssertionError(msg)
    row_metadata = {
        "fixture",
        "repeat",
        "format_position",
        "format_order",
        "storage",
        "retained_heap_supported",
    }
    metric_keys = set(rows[0]) - row_metadata
    if any(set(row) - row_metadata != metric_keys for row in rows):
        msg = "a fixture/storage repeat group has inconsistent metric fields"
        raise AssertionError(msg)
    support_values = [row.get("retained_heap_supported") for row in rows]
    if any(type(value) is not bool for value in support_values):
        msg = "a fixture/storage repeat group has invalid retained-heap flags"
        raise AssertionError(msg)

    metrics = {}
    for key in sorted(metric_keys):
        values = [row[key] for row in rows]
        if any(
            value is not None
            and (type(value) not in {int, float} or not math.isfinite(value))
            for value in values
        ):
            msg = (
                f"measurement {key} contains a non-finite or nonnumeric value"
            )
            raise AssertionError(msg)
        available_count = sum(value is not None for value in values)
        if available_count not in {0, expected_repeats}:
            msg = f"measurement {key} has only a partial set of observations"
            raise AssertionError(msg)
        metrics[key] = _median_range(values)
    return {
        "metrics": metrics,
        "retained_heap_supported_samples": sum(support_values),
        "repeat_count": len(rows),
    }


def _smoke_measurement_grouping() -> int:
    """Check the closed summary cases before starting the browser matrix.

    Preconditions: The local Python runtime is available; no browser is needed.
    Postconditions: Three explicit input/expected cases pass or raise early.
    Essential behavior: Prevents row-level grouping, unavailable-metric
        coercion, and incomplete repeat groups from yielding plausible output.
    """
    cases = [
        {
            "concern": (
                "three numeric observations retain their median, range, and count"
            ),
            "rows": [
                {"value": 10, "retained_heap_supported": True},
                {"value": 30, "retained_heap_supported": True},
                {"value": 20, "retained_heap_supported": False},
            ],
            "expected": {
                "metrics": {
                    "value": {
                        "median": 20,
                        "min": 10,
                        "max": 30,
                        "sample_count": 3,
                    }
                },
                "retained_heap_supported_samples": 2,
                "repeat_count": 3,
            },
        },
        {
            "concern": (
                "an unavailable metric stays null with zero numeric samples"
            ),
            "rows": [
                {"value": None, "retained_heap_supported": False},
                {"value": None, "retained_heap_supported": False},
                {"value": None, "retained_heap_supported": False},
            ],
            "expected": {
                "metrics": {
                    "value": {
                        "median": None,
                        "min": None,
                        "max": None,
                        "sample_count": 0,
                    }
                },
                "retained_heap_supported_samples": 0,
                "repeat_count": 3,
            },
        },
    ]
    for case in cases:
        actual = _summarize_measurement_group(case["rows"], 3)
        if actual != case["expected"]:
            msg = f"measurement grouping smoke failed: {case['concern']}"
            raise AssertionError(msg)

    missing_observation_rows = [
        {"value": 10, "retained_heap_supported": True},
        {"value": 20, "retained_heap_supported": True},
    ]
    try:
        _summarize_measurement_group(missing_observation_rows, 3)
    except AssertionError:
        return len(cases) + 1
    msg = (
        "a missing observation must fail instead of producing a partial median"
    )
    raise AssertionError(msg)


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
        signature that the caller compares but does not serialize. Each call
        creates a new page and isolated browser context.
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
        lambda message: (
            console_errors.append(message.text)
            if message.type == "error"
            else None
        ),
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
    retained_js_heap, retained_backing_storage = _retained_heap(page)
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
    if (
        metrics["readerReferenceCount"] <= 0
        or metrics["readerResultCount"] <= 0
    ):
        page.close()
        msg = (
            "browser did not count the reader result and reference projections"
        )
        raise AssertionError(msg)
    page.close()
    numeric = {
        **metrics,
        "request_count": len(requests),
        "page_error_count": len(page_errors),
        "console_error_count": len(console_errors),
        "retained_js_heap_used_bytes": retained_js_heap,
        "retained_backing_storage_bytes": retained_backing_storage,
        "retained_heap_supported": retained_js_heap is not None,
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
    check(
        _smoke_measurement_grouping() == 3,
        "closed measurement grouping cases pass before browser startup",
    )

    (
        escape_smoke_report,
        escape_smoke_shared_json,
        escape_smoke_expanded_html,
        escape_smoke_shared_html,
        escape_smoke_summary,
    ) = trusted_reader_outputs(
        reader_golden_report(),
        output / "html-escape-smoke",
        include_html_escape_probe=True,
    )
    escape_probe = escape_smoke_report["opaque_browser_probe"].get(
        "html_escape_probe"
    )
    check(
        escape_probe == "<&> café \u2028\u2029",
        "the authored byte smoke includes HTML-significant and Unicode text",
    )
    escape_smoke_facts = {}
    for storage, raw_bytes, html, expected_value in (
        (
            "expanded",
            escape_smoke_summary["expanded_json_bytes"],
            escape_smoke_expanded_html,
            escape_smoke_report,
        ),
        (
            "shared",
            escape_smoke_summary["shared_bytes"],
            escape_smoke_shared_html,
            json.loads(escape_smoke_shared_json),
        ),
    ):
        embedded_bytes = len(_embedded_script_payload_bytes(html))
        check(
            raw_bytes != embedded_bytes,
            f"{storage} HTML-significant Unicode fixture separates raw JSON and embedded script byte counts",
        )
        check(
            embedded_json(html) == expected_value,
            f"{storage} HTML escaping preserves the exact logical JSON value",
        )
        escape_smoke_facts[storage] = {
            "raw_compact_json_bytes": raw_bytes,
            "embedded_script_utf8_bytes": embedded_bytes,
            "escape_overhead_bytes": embedded_bytes - raw_bytes,
            "logical_values_match": True,
        }

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
    check(
        fixture_cases[1][1][4]["expanded_json_bytes"]
        > fixture_cases[0][1][4]["expanded_json_bytes"],
        "the authored exact fixture is larger than the golden fixture",
    )
    repeat_value = os.environ.get("DIFFSTORY_BROWSER_REPEATS")
    try:
        repeat_count = int(repeat_value or "1")
    except ValueError as error:
        msg = "DIFFSTORY_BROWSER_REPEATS must be an integer from 1 through 5"
        raise ValueError(msg) from error
    repeated_matrix = repeat_value is not None
    if not 1 <= repeat_count <= 5:
        msg = "DIFFSTORY_BROWSER_REPEATS must be an integer from 1 through 5"
        raise ValueError(msg)
    scale_matrix = (
        {"count": 2, "expected_sections": 2},
        {"count": 4, "expected_sections": 4},
        {"count": 8, "expected_sections": 8},
    )
    measurement_fixtures = fixture_cases
    if repeated_matrix:
        measurement_fixtures = []
        for scale in scale_matrix:
            count = scale["count"]
            pair = trusted_reader_outputs(
                golden_report(count),
                output / f"golden-count-{count}",
                provider=MixedTerminologyReaderFixtureProvider(),
            )
            report = pair[0]
            fixture_name = f"golden_count_{count}"
            check(
                len(report["narrative_v2"]["sections"])
                == scale["expected_sections"],
                f"golden_report count {count} yields {scale['expected_sections']} compiler-backed scale sections",
            )
            measurement_fixtures.append((fixture_name, pair))
        measurement_fixtures.append(("exact", exact))

    fixture_facts = {}
    for fixture_name, pair in measurement_fixtures:
        report, shared_json, _expanded_html, _shared_html, source_summary = (
            pair
        )
        storage_counts = source_summary["storage_counts"]
        check(
            source_summary["expanded_generation_storage_version"]
            == GENERATION_STORAGE_VERSION,
            f"{fixture_name} expanded bytes use compact generation-storage.v1",
        )
        check(
            source_summary["expanded_json_bytes"] > 0
            and source_summary["shared_bytes"] == len(shared_json)
            and source_summary["source_fragment_count"]
            == len(report["source_evidence"]["fragments"])
            and source_summary["source_fragment_count"] > 0,
            f"{fixture_name} has matching byte counts and complete source closure",
        )
        proof_tables = (
            "runs",
            "invocations",
            "requests",
            "producers",
            "receipts",
        )
        check(
            all(storage_counts.get(table, 0) > 0 for table in proof_tables),
            f"{fixture_name} independently imported report keeps all proof table families",
        )
        expanded_bytes = source_summary["expanded_json_bytes"]
        shared_bytes = len(shared_json)
        fixture_facts[fixture_name] = {
            "source_fragment_count": source_summary["source_fragment_count"],
            "expanded_generation_storage_version": GENERATION_STORAGE_VERSION,
            "expanded_json_bytes": expanded_bytes,
            "shared_report_bytes": shared_bytes,
            "raw_compact_json_bytes": {
                "expanded": expanded_bytes,
                "shared": shared_bytes,
            },
            "embedded_script_utf8_bytes": {
                "expanded": source_summary["expanded_embedded_script_bytes"],
                "shared": source_summary["shared_embedded_script_bytes"],
            },
            "html_escape_overhead_bytes": {
                "expanded": source_summary["expanded_embedded_script_bytes"]
                - expanded_bytes,
                "shared": source_summary["shared_embedded_script_bytes"]
                - shared_bytes,
            },
            "shared_minus_expanded_bytes": shared_bytes - expanded_bytes,
            "shared_delta_percent": round(
                (shared_bytes - expanded_bytes) * 100 / expanded_bytes, 2
            ),
            "storage_counts": storage_counts,
        }
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

            for fixture_index, (fixture_name, pair) in enumerate(
                measurement_fixtures
            ):
                report, shared_json, expanded_html, shared_html, _summary = (
                    pair
                )
                html_by_storage = {
                    "expanded": expanded_html,
                    "shared": shared_html,
                }
                for repeat_index in range(repeat_count):
                    format_order = ["expanded", "shared"]
                    if (fixture_index + repeat_index) % 2:
                        format_order.reverse()
                    paired_signatures = {}
                    for format_position, storage in enumerate(format_order, 1):
                        metric_values, signature = _metric_page(
                            browser,
                            report,
                            html_by_storage[storage],
                            expected_shared=storage == "shared",
                        )
                        paired_signatures[storage] = signature
                        expected_bytes = fixture_facts[fixture_name][
                            "embedded_script_utf8_bytes"
                        ][storage]
                        check(
                            metric_values["embeddedJSONBytes"]
                            == expected_bytes,
                            f"{fixture_name} repeat {repeat_index + 1} {storage} browser bytes equal the exact trusted HTML script payload",
                        )
                        measurements.append(
                            {
                                "fixture": fixture_name,
                                "repeat": repeat_index + 1,
                                "format_position": format_position,
                                "format_order": format_order,
                                "storage": storage,
                                **metric_values,
                            }
                        )
                    check(
                        paired_signatures["expanded"]
                        == paired_signatures["shared"],
                        f"{fixture_name} repeat {repeat_index + 1} DOM text, fields, labels, links, context, catalog, anchors, and exact source match",
                    )
        finally:
            browser.close()

    expected_measurement_rows = len(measurement_fixtures) * 2 * repeat_count
    check(
        len(measurements) == expected_measurement_rows,
        "the repeated matrix contains every fixture, format, and observation",
    )
    measurement_summary = {}
    for fixture_name, _pair in measurement_fixtures:
        format_summaries = {}
        for storage in ("expanded", "shared"):
            rows = [
                row
                for row in measurements
                if row["fixture"] == fixture_name and row["storage"] == storage
            ]
            check(
                len(rows) == repeat_count,
                f"{fixture_name} {storage} has exactly {repeat_count} measured observations",
            )
            format_summaries[storage] = _summarize_measurement_group(
                rows, repeat_count
            )
        measurement_summary[fixture_name] = format_summaries
    measurement_artifact = {
        "source_free": True,
        "reused_authored_fixtures": bool(fixture_root),
        "measurement_mode": (
            "opt-in-repeat-matrix"
            if repeated_matrix
            else "default-single-pass"
        ),
        "repeat_count": repeat_count,
        "scale_input_matrix": list(scale_matrix) if repeated_matrix else [],
        "fixtures": [
            fixture_name for fixture_name, _pair in measurement_fixtures
        ],
        "fixture_facts": fixture_facts,
        "html_escape_byte_smoke": escape_smoke_facts,
        "measurements": measurements,
        "summary_by_fixture_and_format": measurement_summary,
        "retained_heap_method": (
            "Chrome CDP HeapProfiler.collectGarbage then Runtime.getHeapUsage "
            "after source expansion; retained heap is not peak or allocation "
            "memory, and does not include native DOM memory. Backing storage "
            "is reported only when Chromium provides that counter."
        ),
        "performance_memory_scope": (
            "Existing performance.memory fields are coarse current snapshots "
            "after parse, index, and render; they are not peak measurements."
        ),
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
        "measurement_mode": measurement_artifact["measurement_mode"],
        "repeat_count": repeat_count,
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
