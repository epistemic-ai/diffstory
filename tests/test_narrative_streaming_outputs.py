"""Source-free checks for session-bound streaming JSON and HTML output."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pydantic import BaseModel

from test_narrative_v2 import FixtureProvider
from test_narrative_v2 import golden_report
from test_reader import MixedTerminologyReaderFixtureProvider
from test_reader import encode_omitted_terminology_defaults
from test_reader import reader_golden_report

from diffstory import report_storage
from diffstory.analysis import apply_annotations
from diffstory.analysis import validate_generation
from diffstory.narrative import Narrator
from diffstory.narrative_manifest import GenerationManifest
from diffstory.narrative_manifest import persisted_manifest
from diffstory.narrative_manifest import stored_generation
from diffstory.narrative_plan import canonical_bytes
from diffstory.narrative_session import ReportSession
from diffstory.narrative_session import ValidatedGeneration
from diffstory.narrative_session import _CapturedInput
from diffstory.render import iter_render_validated
from diffstory.render import render
from diffstory.render import validate_report
from diffstory.report_storage import ReportStorageView
from diffstory.report_storage import encode_report_storage
from diffstory.report_storage import expand_report_storage
from diffstory.report_storage import parse_report_storage

_MAX_OUTPUT = 8 * 1024 * 1024
_MAX_CHUNK = 16 * 1024


class NarrativeStreamingOutputTests(unittest.TestCase):
    """Use one authored source-free generation for each output concern."""

    def setUp(self) -> None:
        """Create the approved preview, indexed handles, and exact oracles."""
        self.recovery = tempfile.TemporaryDirectory()
        self.addCleanup(self.recovery.cleanup)
        self.report = golden_report()
        self.report.setdefault("notes", []).append(
            "CRLF stays exact: first line\r\nsecond line; Unicode: café 雪."
        )
        self.report["opaque_metadata"] = {
            "$ref": "literal compiler metadata",
            "source_text": "<script>literal</script> & CRLF\r\n",
            "large_value": "λ" * 262_144,
        }

        def script_shaped_prose(
            _provider: FixtureProvider, data: dict, response: dict
        ) -> dict:
            """Add HTML-significant text without changing the support graph."""
            if data["stage"] == "document":
                response["items"][0]["preamble"][0]["text"] += (
                    " </script><script>alert(1)</script> & <>\u2028\u2029\r\n"
                )
            return response

        self.provider = FixtureProvider(script_shaped_prose)
        self.narrator = Narrator(
            self.provider,
            architecture="v2",
            recovery_dir=self.recovery.name,
            concurrency=1,
        )
        self.narrator.preview(self.report)
        self.session = self.narrator.capture_session(self.report)
        self.addCleanup(self.session.close)
        self.generation = self.narrator.generate_validated(
            self.report, session=self.session
        )
        self.report_handle = self.session.apply(self.generation)
        self.proof = self.session._registered_payload(  # noqa: SLF001 # Use the exact session-issued proof for test oracles.
            self.generation, ValidatedGeneration, "generation"
        )
        self.expected_report = self.session.report_data(self.report_handle)
        self.expected_annotations = expand_report_storage(
            self.proof.graph, max_output_bytes=_MAX_OUTPUT
        )
        self.expected_annotations["generation"] = stored_generation(
            persisted_manifest(self.expected_annotations["generation"])
        )

    def _collect(self, chunks: object) -> tuple[bytes, int]:
        """Consume byte chunks while tracking the largest allocation."""
        output = bytearray()
        largest = 0
        for chunk in chunks:  # type: ignore[union-attr]
            self.assertIs(type(chunk), bytes)
            largest = max(largest, len(chunk))
            output.extend(chunk)
        return bytes(output), largest

    def test_expanded_stream_matches_report_and_annotation_oracles(
        self,
    ) -> None:
        """Concern: streaming must preserve compatibility defaults and values.

        Input/expected matrix:
        * A fresh indexed report handle -> canonical equality with detached
          ``report_data`` and generation-storage.v1 compact invocations.
        * Its indexed generation handle -> canonical equality with the
          annotation candidate reconstructed by the established expansion
          and stored-generation compatibility oracle.
        * Both streams -> bounded byte chunks and exactly two full capture
          checks, at stream start and completion.
        """
        cases = [
            (
                "report",
                lambda: self.session.iter_report_json(
                    self.report_handle,
                    report_storage="expanded",
                    max_output_bytes=_MAX_OUTPUT,
                ),
                self.expected_report,
            ),
            (
                "annotations",
                lambda: self.session.iter_annotations_json(
                    self.generation,
                    report_storage="expanded",
                    max_output_bytes=_MAX_OUTPUT,
                ),
                self.expected_annotations,
            ),
        ]
        original_check = _CapturedInput.assert_capture_current
        check_count = 0

        def counted_check(
            captured: _CapturedInput, *args: object, **kwargs: object
        ) -> None:
            """Count actual complete capture checks at stream boundaries."""
            nonlocal check_count
            check_count += 1
            original_check(captured, *args, **kwargs)

        with (
            patch.object(
                _CapturedInput, "assert_capture_current", counted_check
            ),
            patch.object(
                ReportSession,
                "report_data",
                side_effect=AssertionError("stream called report_data"),
            ),
            patch.object(
                report_storage,
                "expand_report_storage",
                side_effect=AssertionError("stream expanded the graph"),
            ),
            patch.object(
                report_storage,
                "encode_report_storage",
                side_effect=AssertionError(
                    "stream re-encoded a full artifact"
                ),
            ),
            patch.object(
                GenerationManifest,
                "model_validate_json",
                side_effect=AssertionError("stream built a full manifest"),
            ),
        ):
            for name, make_stream, expected in cases:
                with self.subTest(artifact=name):
                    data, largest = self._collect(make_stream())
                    self.assertLessEqual(largest, _MAX_CHUNK)
                    self.assertEqual(
                        canonical_bytes(json.loads(data)),
                        canonical_bytes(expected),
                    )

        self.assertEqual(check_count, 4)

    def test_shared_streams_are_canonical_closed_graphs_and_reload_offline(
        self,
    ) -> None:
        """Concern: shared streams must retain exact closure and revalidate offline.

        Input/expected matrix:
        * Fresh report handle -> canonical shared bytes equal the independent
          report-storage encoder over the effective applied report.
        * Fresh generation handle -> canonical shared bytes equal the encoder
          over the annotation compatibility candidate.
        * Shared report bytes -> a new offline session independently proves,
          applies, and streams the same logical report.
        * Shared annotation bytes plus the matching compiler report -> an
          independent session proves, applies, and derives the same report.
        """
        expected_report_storage = encode_report_storage(
            self.expected_report, "report"
        )
        expected_annotation_storage = encode_report_storage(
            self.expected_annotations, "annotations"
        )
        report_storage_bytes, report_chunk = self._collect(
            self.session.iter_report_json(
                self.report_handle,
                report_storage="shared",
                max_output_bytes=_MAX_OUTPUT,
            )
        )
        annotation_storage_bytes, annotation_chunk = self._collect(
            self.session.iter_annotations_json(
                self.generation,
                report_storage="shared",
                max_output_bytes=_MAX_OUTPUT,
            )
        )
        self.assertLessEqual(report_chunk, _MAX_CHUNK)
        self.assertLessEqual(annotation_chunk, _MAX_CHUNK)
        self.assertEqual(report_storage_bytes, expected_report_storage)
        self.assertEqual(annotation_storage_bytes, expected_annotation_storage)
        for payload, artifact_kind in (
            (report_storage_bytes, "report"),
            (annotation_storage_bytes, "annotations"),
        ):
            with self.subTest(artifact=artifact_kind):
                parsed = parse_report_storage(payload)
                self.assertEqual(parsed.artifact_kind, artifact_kind)
                self.assertEqual(len(parsed.storage_counts), 13)
                self.assertEqual(
                    set(parsed.storage_counts),
                    set(report_storage._TABLES),  # noqa: SLF001 # Require the full closed table set.
                )

        report_session, imported_report_generation = (
            ReportSession.import_indexed(
                report_storage_bytes,
                max_input_bytes=len(report_storage_bytes),
            )
        )
        self.addCleanup(report_session.close)
        imported_report = report_session.apply(imported_report_generation)
        self.assertEqual(
            canonical_bytes(report_session.report_data(imported_report)),
            canonical_bytes(self.expected_report),
        )
        imported_report_json, _ = self._collect(
            report_session.iter_report_json(
                imported_report, max_output_bytes=_MAX_OUTPUT
            )
        )
        self.assertEqual(
            canonical_bytes(json.loads(imported_report_json)),
            canonical_bytes(self.expected_report),
        )
        with patch.object(
            report_storage,
            "_check_graph_and_addresses",
            side_effect=AssertionError("writer rechecked graph closure"),
        ):
            derived_annotations, _ = self._collect(
                report_session.iter_annotations_json(
                    imported_report_generation,
                    max_output_bytes=_MAX_OUTPUT,
                )
            )
            derived_annotation_storage, _ = self._collect(
                report_session.iter_annotations_json(
                    imported_report_generation,
                    report_storage="shared",
                    max_output_bytes=_MAX_OUTPUT,
                )
            )
        self.assertEqual(
            canonical_bytes(json.loads(derived_annotations)),
            canonical_bytes(self.expected_annotations),
        )
        self.assertEqual(
            derived_annotation_storage, expected_annotation_storage
        )

        annotation_session, imported_annotation_generation = (
            ReportSession.import_indexed(
                annotation_storage_bytes,
                compiler_report=self.report,
                max_input_bytes=len(annotation_storage_bytes),
            )
        )
        self.addCleanup(annotation_session.close)
        imported_annotation_report = annotation_session.apply(
            imported_annotation_generation
        )
        self.assertEqual(
            canonical_bytes(
                annotation_session.report_data(imported_annotation_report)
            ),
            canonical_bytes(self.expected_report),
        )
        roundtrip_annotation_bytes, _ = self._collect(
            annotation_session.iter_annotations_json(
                imported_annotation_generation,
                report_storage="shared",
                max_output_bytes=_MAX_OUTPUT,
            )
        )
        self.assertEqual(roundtrip_annotation_bytes, annotation_storage_bytes)

    def test_imported_omitted_defaults_keep_lossless_shared_bytes(
        self,
    ) -> None:
        """Concern: compatibility normalization must not alter shared raw fields.

        Input/expected matrix:
        * Annotation storage with omitted default version and provenance ->
          shared writer preserves the exact canonical input bytes and fields.
        * The same indexed generation in expanded mode -> exact equality with
          the existing expansion plus ``stored_generation`` oracle, including
          the default provenance and generation-storage.v1 representation.
        * Unit-key and producer arrays -> retain their validated order.
        """
        encoded = encode_report_storage(
            self.expected_annotations, "annotations"
        )
        omitted = json.loads(encoded)
        omitted["root"]["generation"].pop("version")
        omitted["root"]["generation"].pop("provenance")
        omitted_bytes = canonical_bytes(omitted)
        imported, generation = ReportSession.import_indexed(
            omitted_bytes,
            compiler_report=self.report,
            max_input_bytes=len(omitted_bytes),
        )
        self.addCleanup(imported.close)
        shared, _ = self._collect(
            imported.iter_annotations_json(
                generation,
                report_storage="shared",
                max_output_bytes=_MAX_OUTPUT,
            )
        )
        self.assertEqual(shared, omitted_bytes)

        raw_expanded = expand_report_storage(
            parse_report_storage(omitted_bytes), max_output_bytes=_MAX_OUTPUT
        )
        expected_expanded = copy.deepcopy(raw_expanded)
        expected_generation = expected_expanded["generation"]
        expected_generation.setdefault("version", "diffstory.generation.v2")
        expected_generation.setdefault("provenance", "unverified")
        expected_expanded["generation"] = stored_generation(
            persisted_manifest(expected_generation)
        )
        output, _ = self._collect(
            imported.iter_annotations_json(
                generation, max_output_bytes=_MAX_OUTPUT
            )
        )
        actual = json.loads(output)
        self.assertEqual(
            canonical_bytes(actual), canonical_bytes(expected_expanded)
        )
        self.assertEqual(
            actual["generation"]["version"], "diffstory.generation.v2"
        )
        self.assertEqual(
            actual["generation"]["storage_version"],
            "diffstory.generation-storage.v1",
        )
        self.assertIn("provenance", actual["generation"])
        self.assertEqual(
            actual["generation"]["unit_keys"],
            raw_expanded["generation"]["unit_keys"],
        )
        self.assertEqual(
            actual["generation"]["producers"],
            expected_expanded["generation"]["producers"],
        )

    def test_html_stream_preserves_data_and_escapes_script_context(
        self,
    ) -> None:
        """Concern: prose and metadata must stay data inside the HTML script tag.

        Input/expected matrix:
        * Script-shaped generated prose and Unicode separators -> valid
          embedded JSON with the original parsed text and no raw script closer.
        * Literal compiler ``$ref`` metadata and CRLF -> exact values after
          parsing, with no schema-legal pointer rewriting.
        * Expanded and shared HTML -> complete report data/envelope under the
          existing network-free self-contained template.
        """
        cases = [
            ("expanded", self.expected_report),
            (
                "shared",
                json.loads(
                    b"".join(
                        self.session.iter_report_json(
                            self.report_handle,
                            report_storage="shared",
                            max_output_bytes=_MAX_OUTPUT,
                        )
                    )
                ),
            ),
        ]
        suffix = " </script><script>alert(1)</script> & <>\u2028\u2029\r\n"
        for storage, expected in cases:
            with self.subTest(storage=storage):
                html_bytes, largest = self._collect(
                    iter_render_validated(
                        self.session,
                        self.report_handle,
                        report_storage=storage,
                        max_output_bytes=_MAX_OUTPUT,
                    )
                )
                self.assertLessEqual(largest, _MAX_CHUNK)
                html = html_bytes.decode("utf-8")
                start_tag = '<script type="application/json" id="report-data">'
                payload_start = html.index(start_tag) + len(start_tag)
                payload_end = html.index("</script>", payload_start)
                payload = html[payload_start:payload_end]
                self.assertNotIn("</script><script>alert(1)", payload)
                self.assertIn("\\u003c", payload)
                self.assertIn("\\u0026", payload)
                self.assertIn("\\u2028", payload)
                self.assertIn("\\u2029", payload)
                self.assertIn("connect-src 'none'", html)
                embedded = json.loads(payload)
                if storage == "expanded":
                    self.assertEqual(
                        canonical_bytes(embedded), canonical_bytes(expected)
                    )
                    text = embedded["generation"]["document"]["preamble"][0][
                        "text"
                    ]
                    self.assertTrue(text.endswith(suffix))
                    self.assertEqual(
                        embedded["opaque_metadata"]["$ref"],
                        "literal compiler metadata",
                    )
                    self.assertEqual(
                        embedded["opaque_metadata"]["source_text"],
                        "<script>literal</script> & CRLF\r\n",
                    )
                else:
                    self.assertEqual(
                        canonical_bytes(embedded), canonical_bytes(expected)
                    )
                    self.assertEqual(
                        embedded["root"]["base_report"]["opaque_metadata"][
                            "$ref"
                        ],
                        "literal compiler metadata",
                    )

    def test_output_limits_count_all_bytes_and_reject_invalid_limits(
        self,
    ) -> None:
        """Concern: writers must reject over-cap output without truncating.

        Input/expected matrix:
        * Exact serialized JSON or HTML byte count in both storage modes ->
          complete stream succeeds.
        * One byte below the exact count or a zero-byte cap -> consumption
          raises ``ValueError`` instead of returning truncated output.
        * Boolean or negative caps and unknown storage values -> rejected
          before the iterator can emit bytes.
        * HTML capacity sized for raw JSON but not its script escaping ->
          rejected after accounting for assets and escape overhead.
        """
        report_bytes, _ = self._collect(
            self.session.iter_report_json(
                self.report_handle, max_output_bytes=_MAX_OUTPUT
            )
        )
        annotation_bytes, _ = self._collect(
            self.session.iter_annotations_json(
                self.generation, max_output_bytes=_MAX_OUTPUT
            )
        )
        html_bytes, _ = self._collect(
            iter_render_validated(
                self.session,
                self.report_handle,
                max_output_bytes=_MAX_OUTPUT,
            )
        )
        shared_report_bytes, _ = self._collect(
            self.session.iter_report_json(
                self.report_handle,
                report_storage="shared",
                max_output_bytes=_MAX_OUTPUT,
            )
        )
        shared_annotation_bytes, _ = self._collect(
            self.session.iter_annotations_json(
                self.generation,
                report_storage="shared",
                max_output_bytes=_MAX_OUTPUT,
            )
        )
        shared_html_bytes, _ = self._collect(
            iter_render_validated(
                self.session,
                self.report_handle,
                report_storage="shared",
                max_output_bytes=_MAX_OUTPUT,
            )
        )
        exact_json, _ = self._collect(
            self.session.iter_report_json(
                self.report_handle,
                max_output_bytes=len(report_bytes),
            )
        )
        exact_annotations, _ = self._collect(
            self.session.iter_annotations_json(
                self.generation,
                max_output_bytes=len(annotation_bytes),
            )
        )
        exact_html, _ = self._collect(
            iter_render_validated(
                self.session,
                self.report_handle,
                max_output_bytes=len(html_bytes),
            )
        )
        exact_shared_report, _ = self._collect(
            self.session.iter_report_json(
                self.report_handle,
                report_storage="shared",
                max_output_bytes=len(shared_report_bytes),
            )
        )
        exact_shared_annotations, _ = self._collect(
            self.session.iter_annotations_json(
                self.generation,
                report_storage="shared",
                max_output_bytes=len(shared_annotation_bytes),
            )
        )
        exact_shared_html, _ = self._collect(
            iter_render_validated(
                self.session,
                self.report_handle,
                report_storage="shared",
                max_output_bytes=len(shared_html_bytes),
            )
        )
        self.assertEqual(exact_json, report_bytes)
        self.assertEqual(exact_annotations, annotation_bytes)
        self.assertEqual(exact_html, html_bytes)
        self.assertEqual(exact_shared_report, shared_report_bytes)
        self.assertEqual(exact_shared_annotations, shared_annotation_bytes)
        self.assertEqual(exact_shared_html, shared_html_bytes)

        failing = [
            (
                "json below cap",
                lambda: self.session.iter_report_json(
                    self.report_handle, max_output_bytes=len(report_bytes) - 1
                ),
            ),
            (
                "json zero cap",
                lambda: self.session.iter_report_json(
                    self.report_handle, max_output_bytes=0
                ),
            ),
            (
                "annotation below cap",
                lambda: self.session.iter_annotations_json(
                    self.generation,
                    max_output_bytes=len(annotation_bytes) - 1,
                ),
            ),
            (
                "shared report below cap",
                lambda: self.session.iter_report_json(
                    self.report_handle,
                    report_storage="shared",
                    max_output_bytes=len(shared_report_bytes) - 1,
                ),
            ),
            (
                "shared annotations below cap",
                lambda: self.session.iter_annotations_json(
                    self.generation,
                    report_storage="shared",
                    max_output_bytes=len(shared_annotation_bytes) - 1,
                ),
            ),
            (
                "annotation zero cap",
                lambda: self.session.iter_annotations_json(
                    self.generation, max_output_bytes=0
                ),
            ),
            (
                "html zero cap",
                lambda: iter_render_validated(
                    self.session, self.report_handle, max_output_bytes=0
                ),
            ),
            (
                "html below cap",
                lambda: iter_render_validated(
                    self.session,
                    self.report_handle,
                    max_output_bytes=len(html_bytes) - 1,
                ),
            ),
            (
                "shared HTML below cap",
                lambda: iter_render_validated(
                    self.session,
                    self.report_handle,
                    report_storage="shared",
                    max_output_bytes=len(shared_html_bytes) - 1,
                ),
            ),
        ]
        for name, make_stream in failing:
            with self.subTest(case=name), self.assertRaises(ValueError):
                b"".join(make_stream())

        invalid = [
            ("boolean cap", True, "expanded"),
            ("negative cap", -1, "expanded"),
            ("unknown format", _MAX_OUTPUT, "future"),
        ]
        for name, limit, storage in invalid:
            with self.subTest(case=name), self.assertRaises(ValueError):
                self.session.iter_report_json(
                    self.report_handle,
                    report_storage=storage,
                    max_output_bytes=limit,
                )

        other_invalid = [
            (
                "annotation boolean cap",
                lambda: self.session.iter_annotations_json(
                    self.generation, max_output_bytes=True
                ),
            ),
            (
                "annotation negative cap",
                lambda: self.session.iter_annotations_json(
                    self.generation, max_output_bytes=-1
                ),
            ),
            (
                "annotation unknown format",
                lambda: self.session.iter_annotations_json(
                    self.generation,
                    report_storage="future",
                    max_output_bytes=_MAX_OUTPUT,
                ),
            ),
            (
                "HTML boolean cap",
                lambda: iter_render_validated(
                    self.session, self.report_handle, max_output_bytes=True
                ),
            ),
            (
                "HTML negative cap",
                lambda: iter_render_validated(
                    self.session, self.report_handle, max_output_bytes=-1
                ),
            ),
            (
                "HTML unknown format",
                lambda: iter_render_validated(
                    self.session,
                    self.report_handle,
                    report_storage="future",
                    max_output_bytes=_MAX_OUTPUT,
                ),
            ),
        ]
        for name, make_stream in other_invalid:
            with self.subTest(case=name), self.assertRaises(ValueError):
                make_stream()

        html = html_bytes.decode("utf-8")
        marker = '<script type="application/json" id="report-data">'
        payload_start = html.index(marker) + len(marker)
        payload_end = html.index("</script>", payload_start)
        escaped_length = len(html[payload_start:payload_end].encode("utf-8"))
        prefix_and_suffix = len(html_bytes) - escaped_length
        raw_json_length = len(report_bytes)
        self.assertGreater(escaped_length, raw_json_length)
        escape_only_cap = prefix_and_suffix + raw_json_length
        with self.assertRaises(ValueError):
            b"".join(
                iter_render_validated(
                    self.session,
                    self.report_handle,
                    max_output_bytes=escape_only_cap,
                )
            )

        prefix_length = payload_start
        partial = bytearray()
        with self.assertRaises(ValueError):
            for chunk in iter_render_validated(
                self.session,
                self.report_handle,
                max_output_bytes=prefix_length + _MAX_CHUNK,
            ):
                partial.extend(chunk)
        self.assertGreaterEqual(len(partial), prefix_length)
        self.assertLess(len(partial), len(html_bytes))

    def test_capability_checks_reject_foreign_partial_and_closed_streams(
        self,
    ) -> None:
        """Concern: JSON streams must not transfer or outlive session authority.

        Input/expected matrix:
        * A copied report handle -> copy rejection before a stream is created.
        * A handle used by another session or a ``ParsedRecords`` token ->
          rejection before any output chunk.
        * A stream paused after one chunk and then closed -> the next
          consumption rejects the expired session.
        """
        with self.assertRaises(TypeError):
            copy.copy(self.report_handle)
        foreign = ReportSession.capture(
            self.report, self.session.request_configuration_data()
        )
        self.addCleanup(foreign.close)
        for name, stream in (
            (
                "foreign handle",
                foreign.iter_report_json(
                    self.report_handle, max_output_bytes=_MAX_OUTPUT
                ),
            ),
            (
                "partial parse handle",
                self.session.iter_annotations_json(
                    self.session.parse_import(
                        self.expected_annotations["generation"],
                        candidate=self.expected_annotations,
                    ),  # type: ignore[arg-type]
                    max_output_bytes=_MAX_OUTPUT,
                ),
            ),
        ):
            with self.subTest(case=name), self.assertRaises(ValueError):
                next(stream)

        isolated = tempfile.TemporaryDirectory()
        self.addCleanup(isolated.cleanup)
        provider = FixtureProvider()
        narrator = Narrator(
            provider,
            architecture="v2",
            recovery_dir=isolated.name,
            concurrency=1,
        )
        local_report = golden_report()
        narrator.preview(local_report)
        session = narrator.capture_session(local_report)
        self.addCleanup(session.close)
        generation = narrator.generate_validated(local_report, session=session)
        report_handle = session.apply(generation)
        paused = session.iter_report_json(
            report_handle, max_output_bytes=_MAX_OUTPUT
        )
        first = next(paused)
        self.assertTrue(first)
        session.close()
        with self.assertRaisesRegex(ValueError, "session_closed"):
            next(paused)

    def test_final_boundary_rejects_proof_mutation_after_streaming(
        self,
    ) -> None:
        """Concern: a mutated proof must not receive a successful end signal.

        Input/expected matrix:
        * Exact report handle and intact witness -> stream starts normally.
        * Witness changed after the first output chunk -> iteration raises at
          the final full proof boundary instead of returning ``StopIteration``.
        """
        stream = self.session.iter_report_json(
            self.report_handle, max_output_bytes=_MAX_OUTPUT
        )
        first = next(stream)
        self.assertTrue(first)
        original_witness = self.proof.witness
        object.__setattr__(self.proof, "witness", "0" * 64)
        emitted = len(first)
        try:
            with self.assertRaisesRegex(
                ValueError, "stale_session_capability"
            ):
                for chunk in stream:
                    emitted += len(chunk)
        finally:
            object.__setattr__(self.proof, "witness", original_witness)
        self.assertEqual(emitted, len(canonical_bytes(self.expected_report)))

    def test_large_strings_and_arrays_keep_output_chunks_bounded(  # noqa: PLR0915 # Keep the scaling probe and its measurements together.
        self,
    ) -> None:
        """Concern: output allocation must not scale with one embedded string.

        Input/expected matrix:
        * A 256 KiB compiler metadata string and ordinary indexed records ->
          output size grows with the data while every emitted chunk remains
          at most 16 KiB.
        * The writer's canonical serializer -> never passes a whole long
          string to ``canonical_bytes`` or builds the full report buffer.
        * Generation and presentation output -> normalize each selected typed
          row once, while nested materializations are counted separately; no
          full expansion, full manifest parse, or compatibility adapter runs.
        """
        original_canonical_bytes = report_storage.canonical_bytes
        large_scalar_calls = []

        def guarded_canonical_bytes(value: object) -> bytes:
            """Detect whole large string serialization in the output path."""
            if type(value) is str and len(value) > 1024:
                large_scalar_calls.append(len(value))
                msg = "whole long string passed to canonical_bytes"
                raise AssertionError(msg)
            return original_canonical_bytes(value)

        materialized = []
        selected_materialized = []
        selected_models = []
        materialize_depth = 0
        original_materialize = ReportStorageView._materialize  # noqa: SLF001 # Count selected normalization work in the stream.
        original_selected_model = report_storage._materialized_model  # noqa: SLF001 # Count actual strict selected-row adapters.

        def counted_materialize(
            view: ReportStorageView,
            table: str,
            identifier: str,
            memo: dict[str, dict],
        ) -> dict:
            """Record selected-record normalization without retaining rows."""
            nonlocal materialize_depth
            materialized.append(table)
            if materialize_depth == 0:
                selected_materialized.append(table)
            materialize_depth += 1
            try:
                return original_materialize(view, table, identifier, memo)
            finally:
                materialize_depth -= 1

        def counted_selected_model(  # noqa: PLR0913 # Mirror the exact internal adapter signature for truthful counting.
            view: ReportStorageView,
            pointer: object,
            *,
            table: str,
            kind: str,
            model_type: type[BaseModel],
            compact_invocation: bool = False,
        ) -> Iterator[bytes]:
            """Count strict selected-row adapters without retaining models."""
            selected_models.append((table, kind, model_type.__name__))
            yield from original_selected_model(
                view,
                pointer,
                table=table,
                kind=kind,
                model_type=model_type,
                compact_invocation=compact_invocation,
            )

        total = 0
        largest = 0
        with (
            patch.object(
                report_storage, "canonical_bytes", guarded_canonical_bytes
            ),
            patch.object(
                ReportStorageView, "_materialize", counted_materialize
            ),
            patch.object(
                report_storage, "_materialized_model", counted_selected_model
            ),
            patch.object(
                report_storage,
                "expand_report_storage",
                side_effect=AssertionError("stream expanded the full graph"),
            ),
            patch.object(
                GenerationManifest,
                "model_validate_json",
                side_effect=AssertionError("stream parsed a full manifest"),
            ),
            patch.object(
                ReportSession,
                "report_data",
                side_effect=AssertionError(
                    "stream called compatibility expansion"
                ),
            ),
        ):
            for chunk in self.session.iter_report_json(
                self.report_handle, max_output_bytes=_MAX_OUTPUT
            ):
                self.assertIs(type(chunk), bytes)
                largest = max(largest, len(chunk))
                total += len(chunk)

        self.assertGreater(total, 262_144)
        self.assertLessEqual(largest, _MAX_CHUNK)
        self.assertEqual(large_scalar_calls, [])
        self.assertTrue(materialized)
        self.assertGreater(materialized.count("producers"), 0)
        self.assertGreater(materialized.count("results"), 0)
        arrays = ("details", "planning", "sections", "producers", "receipts")
        generation_normalizations = 3 + sum(
            len(self.expected_annotations["generation"][field])
            for field in arrays
        )
        reader = self.expected_report["narrative_v2"]
        presentation_normalizations = (
            1
            + len(reader["context_details"])
            + len(reader["citations"])
            + len(reader["support_provenance"])
            + sum(
                1 + len(section["details"]) for section in reader["sections"]
            )
        )
        selected_normalizations = (
            generation_normalizations + presentation_normalizations
        )
        self.assertEqual(len(selected_models), selected_normalizations)
        self.assertEqual(len(selected_materialized), selected_normalizations)
        nested_materializations = len(materialized) - selected_normalizations
        self.assertGreater(nested_materializations, 0)
        self.assertGreater(presentation_normalizations, 0)
        self.assertEqual(
            selected_materialized.count("producers"),
            len(self.expected_annotations["generation"]["producers"]),
        )
        producer_rows = self.expected_annotations["generation"]["producers"]
        peer_dependencies = sum(
            sum(
                dependency["kind"] == "peer"
                for dependency in producer["invocation"][
                    "dynamic_dependencies"
                ]
            )
            for producer in producer_rows
        )
        self.assertEqual(
            peer_dependencies,
            sum(
                len(producer["invocation"]["batch"]["item_keys"])
                for producer in producer_rows
            ),
        )


def _omit_supported_statement_defaults(
    statement: dict, omitted: dict[str, int]
) -> None:
    """Remove only empty optional statement values from a detached fixture.

    Preconditions:
        ``statement`` is one locally authored supported statement from an
        expanded fixture; ``omitted`` records the changed default fields.
    Postconditions:
        Empty arrays and ``None`` values use their existing model defaults.
    Essential behavior:
        Nonempty support, effects, uncertainty, and author provenance remain
        unchanged for the subsequent independent import.
    """
    if statement.get("kind") in {"implementation_claim", "author_motivation"}:
        return
    defaults = {
        "claims": [],
        "citations": [],
        "basis": [],
        "effect": None,
        "uncertainty": [],
        "author_context_digest": None,
    }
    for field, default in defaults.items():
        if field in statement and statement[field] == default:
            statement.pop(field)
            omitted[field] = omitted.get(field, 0) + 1


def _omit_reader_result_defaults(
    result: dict, omitted: dict[str, int]
) -> None:
    """Remove optional empty defaults from one detached result and its prose.

    Preconditions:
        ``result`` is an expanded local generation or reader result.
    Postconditions:
        Only model-defined empty terminology and supported-statement defaults
        are absent; all nonempty authored content remains present.
    Essential behavior:
        Visits only declared result and statement fields, not arbitrary maps.
    """
    if result.get("terminology") == []:
        result.pop("terminology")
        omitted["terminology"] = omitted.get("terminology", 0) + 1
    for field in (
        "abstraction",
        "title",
        "orientation",
        "synthesis",
        "transition",
        "preamble",
        "lead",
        "closing",
    ):
        value = result.get(field)
        if isinstance(value, dict):
            _omit_supported_statement_defaults(value, omitted)
        elif type(value) in {list, tuple}:
            for statement in value:
                if isinstance(statement, dict):
                    _omit_supported_statement_defaults(statement, omitted)
    terminology = result.get("terminology", ())
    for term in terminology:
        definition = term.get("definition")
        if isinstance(definition, dict):
            _omit_supported_statement_defaults(definition, omitted)


def _omitted_defaults_fixture() -> tuple[dict, dict, bytes, dict[str, int]]:
    """Build source-free shared bytes with valid optional defaults omitted.

    Preconditions:
        Local fixture providers and the existing detached omission encoder
        are available to the test suite.
    Postconditions:
        Returns an unchanged compiler report, its fully typed current report,
        canonical independently importable shared bytes, and omission counts.
    Essential behavior:
        Generates through the public session flow, then edits only a detached
        expanded copy before re-encoding a new shared fixture.
    """
    compiler_report = reader_golden_report()
    compiler_report["opaque_browser_probe"] = {
        "$ref": "literal compiler metadata"
    }
    with tempfile.TemporaryDirectory() as directory:
        narrator = Narrator(
            MixedTerminologyReaderFixtureProvider(),
            architecture="v2",
            recovery_dir=Path(directory) / "recovery",
            concurrency=1,
        )
        narrator.preview(compiler_report)
        writer_session = narrator.capture_session(compiler_report)
        try:
            generation = narrator.generate_validated(
                compiler_report, session=writer_session
            )
            report_handle = writer_session.apply(generation)
            current_report = writer_session.report_data(report_handle)
        finally:
            writer_session.close()

    omitted_bytes = encode_omitted_terminology_defaults(current_report)
    detached = expand_report_storage(
        parse_report_storage(omitted_bytes), max_output_bytes=128 * 1024 * 1024
    )
    omitted: dict[str, int] = {}
    generation_data = detached["generation"]
    for result in generation_data["sections"]:
        _omit_reader_result_defaults(result, omitted)
    _omit_reader_result_defaults(generation_data["document"], omitted)
    reader = detached["narrative_v2"]
    omitted["terminology"] = int("terminology" not in reader["document"])
    omitted["terminology"] += sum(
        "terminology" not in section["prose"] for section in reader["sections"]
    )
    _omit_reader_result_defaults(reader["document"], omitted)
    for section in reader["sections"]:
        _omit_reader_result_defaults(section["prose"], omitted)
    for inherited in reader["support_provenance"]:
        _omit_supported_statement_defaults(inherited["statement"], omitted)
    omitted_bytes = encode_report_storage(detached, "report")
    return compiler_report, current_report, omitted_bytes, omitted


class OmittedDefaultPublicBoundaryTests(unittest.TestCase):
    """Check raw default omissions through independent report consumers."""

    @classmethod
    def setUpClass(cls) -> None:
        """Create one authored fixture shared by both boundary concerns."""
        (
            cls.compiler_report,
            cls.current_report,
            cls.shared_bytes,
            cls.omitted,
        ) = _omitted_defaults_fixture()

    def test_expanded_report_defaults_match_current_public_assembly(
        self,
    ) -> None:
        """Concern: valid raw omissions must not break expanded report consumers.

        Input/expected matrix:
        * Imported shared rows omit empty terminology, statement arrays,
          effect, uncertainty, and author defaults.
        * ``report_data`` and streamed expanded JSON -> match each other and
          the report independently produced by public ``apply_annotations``.
        * Both expanded values -> pass independent public validation and render.
        * Shared output -> retains the imported tables and omitted row fields.
        """
        for field in (
            "terminology",
            "claims",
            "citations",
            "basis",
            "effect",
            "uncertainty",
            "author_context_digest",
        ):
            with self.subTest(omitted_field=field):
                self.assertGreater(self.omitted.get(field, 0), 0)

        raw_expansion = expand_report_storage(
            parse_report_storage(self.shared_bytes),
            max_output_bytes=128 * 1024 * 1024,
        )
        raw_reader = raw_expansion["narrative_v2"]
        self.assertNotIn("terminology", raw_reader["document"])
        self.assertTrue(
            any(
                "terminology" not in section["prose"]
                for section in raw_reader["sections"]
            )
        )
        self.assertTrue(
            any(
                section["prose"].get("terminology")
                for section in raw_reader["sections"]
            )
        )
        raw_storage = json.loads(self.shared_bytes)
        session, generation = ReportSession.import_indexed(
            self.shared_bytes, max_input_bytes=len(self.shared_bytes)
        )
        self.addCleanup(session.close)
        report_handle = session.apply(generation)
        report_data = session.report_data(report_handle)
        expanded_stream = json.loads(
            b"".join(
                session.iter_report_json(
                    report_handle,
                    report_storage="expanded",
                    max_output_bytes=128 * 1024 * 1024,
                )
            )
        )
        annotation_candidate = json.loads(
            b"".join(
                session.iter_annotations_json(
                    generation,
                    report_storage="expanded",
                    max_output_bytes=128 * 1024 * 1024,
                )
            )
        )
        validate_generation(
            self.compiler_report, annotation_candidate["generation"]
        )
        independently_applied = apply_annotations(
            self.compiler_report, annotation_candidate
        )

        for name, candidate in (
            ("report_data", report_data),
            ("expanded_stream", expanded_stream),
        ):
            with self.subTest(public_value=name):
                self.assertEqual(
                    canonical_bytes(candidate),
                    canonical_bytes(independently_applied),
                )
                validate_report(candidate)
                html = render(candidate)
                marker = '<script type="application/json" id="report-data">'
                start = html.index(marker) + len(marker)
                end = html.index("</script>", start)
                self.assertEqual(
                    canonical_bytes(json.loads(html[start:end])),
                    canonical_bytes(candidate),
                )
        self.assertIsNone(report_data["groups"][-1]["next_id"])
        self.assertEqual(
            canonical_bytes(report_data), canonical_bytes(expanded_stream)
        )

        html_stream = b"".join(
            iter_render_validated(
                session,
                report_handle,
                report_storage="expanded",
                max_output_bytes=128 * 1024 * 1024,
            )
        ).decode("utf-8")
        marker = '<script type="application/json" id="report-data">'
        start = html_stream.index(marker) + len(marker)
        end = html_stream.index("</script>", start)
        self.assertEqual(
            canonical_bytes(json.loads(html_stream[start:end])),
            canonical_bytes(expanded_stream),
        )

        shared_output = b"".join(
            session.iter_report_json(
                report_handle,
                report_storage="shared",
                max_output_bytes=128 * 1024 * 1024,
            )
        )
        actual_storage = json.loads(shared_output)
        self.assertEqual(actual_storage["tables"], raw_storage["tables"])
        self.assertEqual(canonical_bytes(actual_storage), shared_output)
        self.assertEqual(
            "terminology" in actual_storage["tables"]["results"][0]["record"],
            "terminology" in raw_storage["tables"]["results"][0]["record"],
        )

    def test_expanded_annotation_defaults_apply_and_shared_rows_stay_raw(
        self,
    ) -> None:
        """Concern: annotation compatibility output must keep strict proof checks.

        Input/expected matrix:
        * An imported report with valid omitted result defaults -> expanded
          annotation JSON applies only with its matching compiler report.
        * Shared annotation JSON -> preserves the imported result table rows
          while deriving its annotation root and reader closure.
        * A fresh import of that annotation envelope -> completes the same
          proof and public application without default-normalizing shared rows.
        """
        original_storage = json.loads(self.shared_bytes)
        session, generation = ReportSession.import_indexed(
            self.shared_bytes, max_input_bytes=len(self.shared_bytes)
        )
        self.addCleanup(session.close)
        report_handle = session.apply(generation)
        report_data = session.report_data(report_handle)
        expanded_annotations_bytes = b"".join(
            session.iter_annotations_json(
                generation,
                report_storage="expanded",
                max_output_bytes=128 * 1024 * 1024,
            )
        )
        expanded_annotations = json.loads(expanded_annotations_bytes)
        validate_generation(
            self.compiler_report, expanded_annotations["generation"]
        )
        independently_applied = apply_annotations(
            self.compiler_report, expanded_annotations
        )
        self.assertEqual(
            canonical_bytes(independently_applied),
            canonical_bytes(report_data),
        )
        self.assertEqual(expanded_annotations["document"]["terminology"], [])
        self.assertTrue(
            any(
                "terminology" not in step["prose"]
                for step in expand_report_storage(
                    parse_report_storage(
                        b"".join(
                            session.iter_annotations_json(
                                generation,
                                report_storage="shared",
                                max_output_bytes=128 * 1024 * 1024,
                            )
                        )
                    ),
                    max_output_bytes=128 * 1024 * 1024,
                )["steps"]
            )
        )

        shared_annotations_bytes = b"".join(
            session.iter_annotations_json(
                generation,
                report_storage="shared",
                max_output_bytes=128 * 1024 * 1024,
            )
        )
        shared_annotations = json.loads(shared_annotations_bytes)
        shared_tables = shared_annotations["tables"]
        self.assertEqual(
            shared_tables["results"], original_storage["tables"]["results"]
        )
        for table in set(shared_tables) - {"readers"}:
            self.assertEqual(
                shared_tables[table], original_storage["tables"][table]
            )
        self.assertEqual(
            canonical_bytes(shared_annotations), shared_annotations_bytes
        )
        parse_report_storage(shared_annotations_bytes)

        imported_annotations, imported_generation = (
            ReportSession.import_indexed(
                shared_annotations_bytes,
                compiler_report=self.compiler_report,
                max_input_bytes=len(shared_annotations_bytes),
            )
        )
        self.addCleanup(imported_annotations.close)
        imported_report_handle = imported_annotations.apply(
            imported_generation
        )
        self.assertEqual(
            canonical_bytes(
                imported_annotations.report_data(imported_report_handle)
            ),
            canonical_bytes(independently_applied),
        )


if __name__ == "__main__":
    unittest.main()
