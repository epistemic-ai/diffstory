"""Offline CLI contracts for shared report storage and staged outputs."""

from __future__ import annotations

import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from unittest.mock import patch

from test_cli_narration_v2 import FixtureProvider
from test_cli_narration_v2 import source_snapshot
from test_narrative_v2 import golden_report

from diffstory import narrative_plan
from diffstory import narrative_session
from diffstory import narrative_v2
from diffstory.analysis import compile_snapshot
from diffstory.analysis import evidence_packet
from diffstory.cli import MAX_HTML_BYTES
from diffstory.cli import CliError
from diffstory.cli import _build_parser
from diffstory.cli import _write_evidence_packet
from diffstory.cli import main
from diffstory.cli import read_json
from diffstory.render import render
from diffstory.render import validate_report

STORAGE_VERSION = "diffstory.report-storage.v1"
SHARED_MARKER = '<script type="application/json" id="report-data">'


def _shared_json(path: Path) -> tuple[bytes, dict]:
    """Read a generated shared export as bytes and its test-only JSON view."""
    raw = path.read_bytes()
    return raw, json.loads(raw)


def _embedded_report(path: Path) -> dict:
    """Read the self-contained report graph from an authored CLI HTML file."""
    html = path.read_text(encoding="utf-8")
    start = html.index(SHARED_MARKER) + len(SHARED_MARKER)
    end = html.index("</script>", start)
    return json.loads(html[start:end])


class CliReportStorageTests(unittest.TestCase):
    """Exercise public CLI import, export, and staging with local fixtures."""

    def setUp(self) -> None:
        """Create isolated paths and one source-free compiler input."""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.snapshot = source_snapshot()

    def run_cli(
        self,
        command: list[str],
        *,
        provider: FixtureProvider | None = None,
    ) -> tuple[int, str, str, FixtureProvider]:
        """Run one CLI command with only its input and provider stubbed."""
        selected_provider = provider or FixtureProvider()
        with (
            patch(
                "diffstory.cli.from_github",
                return_value=copy.deepcopy(self.snapshot),
            ),
            patch(
                "diffstory.cli.OpenAIResponsesProvider",
                return_value=selected_provider,
            ),
            patch(
                "diffstory.cli.CodexCLIProvider",
                return_value=selected_provider,
            ),
            redirect_stdout(io.StringIO()) as output,
            redirect_stderr(io.StringIO()) as errors,
        ):
            code = main(command)
        return code, output.getvalue(), errors.getvalue(), selected_provider

    def fresh_command(self, out: Path, *options: str) -> list[str]:
        """Return one in-process-provider v2 generation command."""
        return [
            "github",
            "owner/repo#1",
            "--narrate",
            "--yes",
            "--concurrency",
            "1",
            "--out",
            str(out),
            *options,
        ]

    def test_expanded_default_and_shared_fresh_outputs_keep_proof(
        self,
    ) -> None:
        """Keep the default expanded and opt-in shared outputs complete.

        Concern: CLI output must not drop the independently checkable v2 proof
            or candidate while selecting a different persisted representation.
        Input/expected matrix:
        * No storage option -> expanded report and annotations; typed public
          validation succeeds and generation-storage.v1 remains present.
        * Shared option -> canonical report and annotation envelopes; HTML embeds
          the same complete report graph and every export has v2 proof data.
        """
        expanded_out = self.root / "expanded.html"
        shared_out = self.root / "shared.html"
        cases = [
            (expanded_out, [], False),
            (shared_out, ["--report-storage", "shared"], True),
        ]
        for out, options, shared in cases:
            with self.subTest(storage="shared" if shared else "expanded"):
                with (
                    patch(
                        "diffstory.narrative_session._snapshot_report",
                        wraps=narrative_session._snapshot_report,  # noqa: SLF001
                    ) as capture_snapshot,
                    patch(
                        "diffstory.narrative_v2.validate_production",
                        wraps=narrative_v2.validate_production,
                    ) as full_production_validation,
                ):
                    code, output, errors, provider = self.run_cli(
                        self.fresh_command(out, *options)
                    )
                self.assertEqual(code, 0, errors)
                self.assertGreater(len(provider.calls), 0)
                self.assertIn("actual attempts", output)
                self.assertEqual(capture_snapshot.call_count, 1)
                self.assertEqual(full_production_validation.call_count, 0)
                report_path = out.with_suffix(".report.json")
                annotation_path = out.with_suffix(".annotations.json")
                self.assertIn(
                    f"Candidate annotations: {annotation_path}", output
                )
                if shared:
                    report_bytes, report_envelope = _shared_json(report_path)
                    annotation_bytes, annotation_envelope = _shared_json(
                        annotation_path
                    )
                    self.assertEqual(
                        report_envelope["storage_version"], STORAGE_VERSION
                    )
                    self.assertEqual(
                        report_envelope["artifact_kind"], "report"
                    )
                    self.assertEqual(
                        annotation_envelope["storage_version"], STORAGE_VERSION
                    )
                    self.assertEqual(
                        annotation_envelope["artifact_kind"], "annotations"
                    )
                    self.assertEqual(_embedded_report(out), report_envelope)
                    self.assertNotIn("narrative_v2", report_envelope["root"])
                    self.assertGreater(len(report_bytes), 0)
                    self.assertGreater(len(annotation_bytes), 0)
                else:
                    report = json.loads(
                        report_path.read_text(encoding="utf-8")
                    )
                    candidate = json.loads(
                        annotation_path.read_text(encoding="utf-8")
                    )
                    self.assertEqual(
                        report["generation"]["version"],
                        "diffstory.generation.v2",
                    )
                    self.assertEqual(
                        report["generation"]["storage_version"],
                        "diffstory.generation-storage.v1",
                    )
                    self.assertEqual(
                        candidate["schema"], "diffstory.annotations.v2"
                    )
                    self.assertEqual(_embedded_report(out), report)
                    validate_report(report)
                    self.assertEqual(render(report), out.read_text())

    def test_ordinary_report_metadata_does_not_select_shared_import(
        self,
    ) -> None:
        """Keep ordinary metadata literal when its keys precede report fields.

        Concern: A report with compiler metadata must not be mistaken for a
            shared envelope or changed by a recursive marker search.
        Input/expected matrix:
        * Top-level artifact_kind first with prose containing storage markers
          and a literal $ref -> render succeeds and saves the same values.
        * Top-level non-storage storage_version first -> render succeeds and
          saves that metadata unchanged.
        * Other artifact_kind value first -> render succeeds as ordinary data.
        """
        compiler_report = golden_report(1)
        compiler_report["meta"]["description"] = (
            "Literal text: diffstory.report-storage.v1 and "
            '{"$ref":"not-a-storage-reference"}.'
        )
        cases = [
            (
                "artifact-kind literal",
                {
                    "artifact_kind": "literal compiler metadata",
                    **compiler_report,
                },
            ),
            (
                "non-storage version metadata",
                {"storage_version": "compiler-format-v7", **compiler_report},
            ),
            (
                "other artifact-kind value",
                {"artifact_kind": "report", **compiler_report},
            ),
        ]
        for name, report in cases:
            with self.subTest(case=name):
                validate_report(report)
                expected_report = json.loads(json.dumps(report))
                source = self.root / f"{name.replace(' ', '-')}.json"
                source.write_text(json.dumps(report), encoding="utf-8")
                out = self.root / f"{name.replace(' ', '-')}.html"
                code, _, errors, provider = self.run_cli(
                    ["render", str(source), "--out", str(out)]
                )
                self.assertEqual(code, 0, errors)
                self.assertEqual(provider.calls, [])
                saved = json.loads(
                    out.with_suffix(".report.json").read_text(encoding="utf-8")
                )
                self.assertEqual(saved, expected_report)
                self.assertEqual(
                    out.read_text(encoding="utf-8"), render(report)
                )

    def test_report_claims_are_imported_strictly_without_ordinary_fallback(
        self,
    ) -> None:
        """Reject malformed or unsupported top-level shared storage claims.

        Concern: A claimed report-storage envelope must not gain the ordinary
            dictionary path when it is malformed, noncanonical, unsupported,
            or has the wrong artifact kind.
        Input/expected matrix:
        * Canonical shared report plus leading whitespace -> safe rejection.
        * Truncated shared report and malformed claimed header -> safe rejection.
        * Unsupported storage version and wrong artifact kind -> safe rejection.
        * Every failure leaves HTML and report destinations absent.
        """
        source_out = self.root / "claimed-source.html"
        code, _, errors, _ = self.run_cli(
            self.fresh_command(source_out, "--report-storage", "shared")
        )
        self.assertEqual(code, 0, errors)
        canonical_path = source_out.with_suffix(".report.json")
        canonical_raw, envelope = _shared_json(canonical_path)

        wrong_kind = copy.deepcopy(envelope)
        wrong_kind["artifact_kind"] = "compiler-report"
        unsupported = copy.deepcopy(envelope)
        unsupported["storage_version"] = "diffstory.report-storage.v99"
        reordered = {
            "storage_version": envelope["storage_version"],
            **{
                key: value
                for key, value in envelope.items()
                if key != "storage_version"
            },
        }
        cases = [
            ("leading whitespace", b" \n" + canonical_raw),
            ("truncated", canonical_raw[:-3]),
            (
                "malformed header",
                b'{"artifact_kind":"report","storage_version":"'
                + STORAGE_VERSION.encode("ascii")
                + b'","root":',
            ),
            (
                "escaped discriminator",
                b'{"artifact_kind":"report","storage\\u005fversion":'
                b'"diffstory\\u002ereport-storage.v1","root":',
            ),
            (
                "wrong kind",
                json.dumps(wrong_kind, separators=(",", ":")).encode("utf-8"),
            ),
            (
                "unsupported version",
                json.dumps(unsupported, separators=(",", ":")).encode("utf-8"),
            ),
            (
                "noncanonical order",
                json.dumps(reordered, separators=(",", ":")).encode("utf-8"),
            ),
        ]
        for name, raw in cases:
            with self.subTest(case=name):
                source = self.root / f"claimed-{name.replace(' ', '-')}.json"
                source.write_bytes(raw)
                out = self.root / f"rejected-{name.replace(' ', '-')}.html"
                code, _, errors, provider = self.run_cli(
                    ["render", str(source), "--out", str(out)]
                )
                self.assertEqual(code, 2)
                self.assertEqual(provider.calls, [])
                self.assertFalse(out.exists())
                self.assertFalse(out.with_suffix(".report.json").exists())
                self.assertNotIn(STORAGE_VERSION, errors)

    def test_offline_shared_report_and_annotation_reload_in_both_formats(
        self,
    ) -> None:
        """Import shared bytes independently and keep annotations source-bound.

        Concern: Offline CLI consumers must revalidate shared bytes and require
            the compiler report that matches an annotation export.
        Input/expected matrix:
        * Shared report -> shared output preserves canonical bytes; expanded
          output passes public report validation and embeds that same graph.
        * Shared annotations plus matching compiler report -> shared and
          expanded applied reports both retain the complete generated story.
        * Same annotations plus a changed compiler head -> fail with no output.
        """
        source_out = self.root / "source.html"
        code, _, errors, _ = self.run_cli(
            self.fresh_command(
                source_out,
                "--report-storage",
                "shared",
            )
        )
        self.assertEqual(code, 0, errors)
        shared_report_path = source_out.with_suffix(".report.json")
        shared_annotations_path = source_out.with_suffix(".annotations.json")
        original_bytes = shared_report_path.read_bytes()

        for storage in ("shared", "expanded"):
            with self.subTest(report_storage=storage):
                reloaded = self.root / f"report-{storage}.html"
                with patch(
                    "diffstory.narrative_v2.validate_indexed_production",
                    wraps=narrative_v2.validate_indexed_production,
                ) as indexed_validation:
                    code, _, errors, provider = self.run_cli(
                        [
                            "render",
                            str(shared_report_path),
                            "--out",
                            str(reloaded),
                            "--report-storage",
                            storage,
                        ]
                    )
                self.assertEqual(code, 0, errors)
                self.assertEqual(provider.calls, [])
                self.assertEqual(indexed_validation.call_count, 1)
                saved_path = reloaded.with_suffix(".report.json")
                saved_bytes = saved_path.read_bytes()
                if storage == "shared":
                    self.assertEqual(saved_bytes, original_bytes)
                    self.assertEqual(
                        _embedded_report(reloaded),
                        json.loads(original_bytes),
                    )
                else:
                    saved = json.loads(saved_bytes)
                    validate_report(saved)
                    self.assertEqual(_embedded_report(reloaded), saved)

        compiler_report = compile_snapshot(self.snapshot)
        compiler_path = self.root / "compiler.report.json"
        compiler_path.write_text(
            json.dumps(
                compiler_report,
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        for storage in ("shared", "expanded"):
            with self.subTest(annotation_storage=storage):
                applied = self.root / f"annotations-{storage}.html"
                code, _, errors, provider = self.run_cli(
                    [
                        "render",
                        str(compiler_path),
                        "--annotations",
                        str(shared_annotations_path),
                        "--out",
                        str(applied),
                        "--report-storage",
                        storage,
                    ]
                )
                self.assertEqual(code, 0, errors)
                self.assertEqual(provider.calls, [])
                output_report = applied.with_suffix(".report.json")
                if storage == "shared":
                    _, envelope = _shared_json(output_report)
                    self.assertEqual(envelope["artifact_kind"], "report")
                    self.assertEqual(_embedded_report(applied), envelope)
                else:
                    report = json.loads(output_report.read_text())
                    validate_report(report)
                    self.assertEqual(_embedded_report(applied), report)

        changed_compiler = copy.deepcopy(compiler_report)
        changed_compiler["meta"]["head_sha"] = "c" * 40
        changed_path = self.root / "changed-compiler.report.json"
        changed_path.write_text(json.dumps(changed_compiler), encoding="utf-8")
        rejected = self.root / "wrong-source.html"
        code, _, errors, provider = self.run_cli(
            [
                "render",
                str(changed_path),
                "--annotations",
                str(shared_annotations_path),
                "--out",
                str(rejected),
            ]
        )
        self.assertEqual(code, 2)
        self.assertEqual(provider.calls, [])
        self.assertIn("dependencies_changed", errors)
        self.assertFalse(rejected.exists())

    def test_non_generation_commands_do_not_report_candidate_annotations(
        self,
    ) -> None:
        """Create candidate paths only when a command produced a candidate.

        Concern: Render, snapshot, and supplied-annotation conversions must
            not print or create an annotations output that was never produced.
        Input/expected matrix:
        * Snapshot and compiler-only render -> no candidate path is printed.
        * Offline shared report render -> no candidate path is printed.
        * Supplied shared annotations applied to their compiler report -> the
          applied report is written without a second candidate file.
        """
        snapshot_path = self.root / "input.snapshot.json"
        snapshot_path.write_text(json.dumps(self.snapshot), encoding="utf-8")
        compiler_path = self.root / "compiler.json"
        compiler_report = compile_snapshot(self.snapshot)
        compiler_path.write_text(json.dumps(compiler_report), encoding="utf-8")

        shared_source = self.root / "shared-source.html"
        code, _, errors, _ = self.run_cli(
            self.fresh_command(shared_source, "--report-storage", "shared")
        )
        self.assertEqual(code, 0, errors)
        shared_report = shared_source.with_suffix(".report.json")
        shared_annotations = shared_source.with_suffix(".annotations.json")

        cases = [
            (
                "snapshot",
                ["snapshot", str(snapshot_path)],
            ),
            (
                "compiler render",
                ["render", str(compiler_path)],
            ),
            (
                "offline shared report",
                ["render", str(shared_report)],
            ),
            (
                "supplied annotations",
                [
                    "render",
                    str(compiler_path),
                    "--annotations",
                    str(shared_annotations),
                ],
            ),
        ]
        for name, command in cases:
            with self.subTest(case=name):
                out = self.root / f"no-candidate-{name.replace(' ', '-')}.html"
                code, output, errors, provider = self.run_cli(
                    [*command, "--out", str(out)]
                )
                self.assertEqual(code, 0, errors)
                self.assertEqual(provider.calls, [])
                self.assertNotIn("Candidate annotations:", output)
                self.assertTrue(out.exists())
                self.assertTrue(out.with_suffix(".report.json").exists())
                self.assertFalse(out.with_suffix(".annotations.json").exists())

    def test_shared_output_staging_preserves_three_existing_files_on_caps(
        self,
    ) -> None:
        """Do not publish report, annotations, or HTML before all caps pass.

        Concern: A report or whole-HTML size failure must leave each existing
            destination byte-identical and remove all sibling stage files.
        Input/expected matrix:
        * Report cap below the emitted report -> all three original files stay.
        * HTML cap below packaged HTML -> all three originals still stay.
        * Both failures use a warm authored CLI run with provider calls blocked.
        """
        out = self.root / "staged.html"
        command = self.fresh_command(
            out,
            "--report-storage",
            "shared",
        )
        code, _, errors, _ = self.run_cli(command)
        self.assertEqual(code, 0, errors)
        destinations = [
            out,
            out.with_suffix(".report.json"),
            out.with_suffix(".annotations.json"),
        ]
        original = [path.read_bytes() for path in destinations]

        for limit_options in (
            ["--max-report-bytes", "1"],
            ["--max-html-bytes", "1"],
        ):
            with self.subTest(options=limit_options):
                provider = FixtureProvider()
                with patch.object(
                    provider,
                    "complete_reply",
                    side_effect=AssertionError("Warm run must not send"),
                ):
                    code, _, errors, _ = self.run_cli(
                        [*command, *limit_options],
                        provider=provider,
                    )
                self.assertEqual(code, 2)
                self.assertEqual(provider.calls, [])
                self.assertTrue(
                    "--max-report-bytes" in errors
                    or "--max-html-bytes" in errors
                )
                self.assertEqual(
                    [path.read_bytes() for path in destinations], original
                )
                self.assertEqual(list(self.root.glob(".*.tmp")), [])

    def test_shared_evidence_export_uses_the_proved_expanded_report(  # noqa: PLR0915
        self,
    ) -> None:
        """Use the checked applied base without expanding the full report.

        Concern: Shared evidence export must preserve public packet values
            without materializing generation and presentation models that the
            packet discards.
        Input/expected matrix:
        * Shared v2 report -> fresh import and checked applied base -> packet
          equal to public ``evidence_packet`` of a separately expanded report.
        * Packet byte cap exactly at serialized size -> succeeds; one byte
          below -> fails before output promotion.
        * The one indexed validator call occurs at fresh import ingress; full
          validators and ledger/index builders do not run during serialization
          or the final current check.
        """
        out = self.root / "evidence-source.html"
        code, _, errors, _ = self.run_cli(
            self.fresh_command(out, "--report-storage", "shared")
        )
        self.assertEqual(code, 0, errors)
        shared_path = out.with_suffix(".report.json")
        expanded_out = self.root / "evidence-expanded.html"
        code, _, errors, provider = self.run_cli(
            [
                "render",
                str(shared_path),
                "--out",
                str(expanded_out),
                "--report-storage",
                "expanded",
            ]
        )
        self.assertEqual(code, 0, errors)
        self.assertEqual(provider.calls, [])
        expanded_report = json.loads(
            expanded_out.with_suffix(".report.json").read_text()
        )

        packet = evidence_packet(expanded_report)
        expected_bytes = json.dumps(
            packet,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        exact_stream = io.BytesIO()
        _write_evidence_packet(
            exact_stream,
            packet,
            max_bytes=len(expected_bytes),
        )
        self.assertEqual(exact_stream.getvalue(), expected_bytes)
        with self.assertRaisesRegex(CliError, "--max-report-bytes"):
            _write_evidence_packet(
                io.BytesIO(),
                packet,
                max_bytes=len(expected_bytes) - 1,
            )

        evidence_path = self.root / "shared-evidence.json"
        real_writer = _write_evidence_packet
        callback_mocks: list[Mock] = []
        writer_counts: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

        def callback_counts() -> tuple[int, ...]:
            """Return current full proof and ledger/index callback counts.

            Postconditions:
                The tuple reports function calls only, not per-row dispatch.
            """
            return tuple(callback.call_count for callback in callback_mocks)

        def observe_packet_writer(
            stream: io.BytesIO,
            packet_value: object,
            *,
            max_bytes: int,
        ) -> None:
            """Record proof/build callbacks across bounded packet serialization.

            Args:
                stream: Staged evidence packet file.
                packet_value: Owned or ordinary JSON packet.
                max_bytes: Exact packet cap passed by the CLI.

            Postconditions:
                Records the callback counts before and after serialization.
            """
            before = callback_counts()
            real_writer(stream, packet_value, max_bytes=max_bytes)
            writer_counts.append((before, callback_counts()))

        with (
            patch(
                "diffstory.narrative_v2.validate_indexed_production",
                wraps=narrative_v2.validate_indexed_production,
            ) as indexed_validation,
            patch(
                "diffstory.narrative_v2.validate_production",
                wraps=narrative_v2.validate_production,
            ) as full_validation,
            patch(
                "diffstory.narrative_v2.build_evidence_ledger",
                wraps=narrative_v2.build_evidence_ledger,
            ) as v2_ledger_build,
            patch(
                "diffstory.narrative_v2.build_ledger_index",
                wraps=narrative_v2.build_ledger_index,
            ) as v2_index_build,
            patch(
                "diffstory.narrative_plan.build_evidence_ledger",
                wraps=narrative_plan.build_evidence_ledger,
            ) as plan_ledger_build,
            patch(
                "diffstory.narrative_plan.build_ledger_index",
                wraps=narrative_plan.build_ledger_index,
            ) as plan_index_build,
            patch(
                "diffstory.narrative_session.ReportSession.report_data",
                side_effect=AssertionError("Shared evidence must not expand"),
            ),
            patch(
                "diffstory.report_storage.expand_report_storage",
                side_effect=AssertionError("Shared evidence must not expand"),
            ),
            patch(
                "diffstory.report_storage._materialize_template",
                side_effect=AssertionError(
                    "Shared evidence must not copy a graph"
                ),
            ),
            patch(
                "diffstory.narrative_manifest.V2Presentation.model_validate",
                side_effect=AssertionError(
                    "Shared evidence must not build a full model"
                ),
            ),
            patch(
                "diffstory.cli._write_evidence_packet",
                side_effect=observe_packet_writer,
            ),
        ):
            callback_mocks.extend(
                (
                    indexed_validation,
                    full_validation,
                    v2_ledger_build,
                    v2_index_build,
                    plan_ledger_build,
                    plan_index_build,
                )
            )
            code, _, errors, provider = self.run_cli(
                ["evidence", str(shared_path), "--out", str(evidence_path)]
            )
        self.assertEqual(code, 0, errors)
        self.assertEqual(provider.calls, [])
        self.assertEqual(indexed_validation.call_count, 1)
        self.assertEqual(full_validation.call_count, 0)
        self.assertEqual(len(writer_counts), 1)
        self.assertEqual(writer_counts[0][0], writer_counts[0][1])
        self.assertEqual(callback_counts(), writer_counts[0][1])
        self.assertEqual(evidence_path.read_bytes(), expected_bytes)
        self.assertEqual(
            json.loads(evidence_path.read_text()),
            json.loads(json.dumps(evidence_packet(expanded_report))),
        )

        existing = self.root / "preserved-evidence.json"
        existing.write_bytes(b"keep this packet")
        real_writer = _write_evidence_packet

        def fail_below_output_cap(
            stream: io.BytesIO,
            packet_value: object,
            *,
            max_bytes: int,
        ) -> None:
            """Force a packet-size failure after the stage file is created."""
            real_writer(stream, packet_value, max_bytes=min(max_bytes, 1))

        with patch(
            "diffstory.cli._write_evidence_packet",
            side_effect=fail_below_output_cap,
        ):
            code, _, errors, _ = self.run_cli(
                ["evidence", str(shared_path), "--out", str(existing)]
            )
        self.assertEqual(code, 2)
        self.assertIn("--max-report-bytes", errors)
        self.assertEqual(existing.read_bytes(), b"keep this packet")
        self.assertEqual(list(self.root.glob(".*.tmp")), [])

    def test_unknown_evidence_input_keeps_existing_destination(self) -> None:
        """Keep unknown or malformed evidence input safe and non-destructive.

        Concern: An unrecognized JSON object must not be written as an evidence
            packet or expose its input values in a CLI error.
        Input/expected matrix:
        * Wrong report schema with a marker value -> safe rejection, old output
          bytes unchanged, and no sibling temporary file remains.
        * Truncated JSON -> safe malformed-input rejection with the same
          preservation and cleanup behavior.
        """
        destination = self.root / "existing-evidence.json"
        destination.write_bytes(b"existing evidence")
        cases = [
            (
                "wrong schema",
                json.dumps(
                    {
                        "schema": "wrong-schema",
                        "marker": "DO_NOT_ECHO_REPORT_INPUT",
                    }
                ).encode("utf-8"),
            ),
            ("truncated JSON", b'{"schema":"'),
        ]
        for name, raw in cases:
            with self.subTest(case=name):
                source = self.root / f"unknown-{name.replace(' ', '-')}.json"
                source.write_bytes(raw)
                code, _, errors, provider = self.run_cli(
                    ["evidence", str(source), "--out", str(destination)]
                )
                self.assertEqual(code, 2)
                self.assertEqual(provider.calls, [])
                self.assertNotIn("DO_NOT_ECHO_REPORT_INPUT", errors)
                self.assertEqual(
                    destination.read_bytes(), b"existing evidence"
                )
                self.assertEqual(list(self.root.glob(".*.tmp")), [])

    def test_exact_report_and_html_limits_and_shared_legacy_rejection(
        self,
    ) -> None:
        """Honor exact configured caps and refuse proof-free shared labels.

        Concern: Explicit limits must apply to emitted bytes, and shared output
            must not relabel manual or compiler-only data as a proved report.
        Input/expected matrix:
        * Exact shared report and HTML byte sizes -> offline render succeeds.
        * One byte below the input report cap -> reject before output creation.
        * Compiler-only input with shared requested -> clear v2-proof error.
        * Parser default is expanded and zero HTML limit is invalid.
        """
        defaults = _build_parser().parse_args(
            ["render", "input.json", "--out", "output.html"]
        )
        self.assertEqual(defaults.report_storage, "expanded")
        self.assertEqual(defaults.max_html_bytes, MAX_HTML_BYTES)
        with (
            redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit),
        ):
            _build_parser().parse_args(
                [
                    "render",
                    "input.json",
                    "--out",
                    "output.html",
                    "--max-html-bytes",
                    "0",
                ]
            )

        source_out = self.root / "exact-source.html"
        code, _, errors, _ = self.run_cli(
            self.fresh_command(source_out, "--report-storage", "shared")
        )
        self.assertEqual(code, 0, errors)
        shared_path = source_out.with_suffix(".report.json")
        shared_size = shared_path.stat().st_size
        html_size = source_out.stat().st_size
        exact_out = self.root / "exact.html"
        code, _, errors, provider = self.run_cli(
            [
                "render",
                str(shared_path),
                "--out",
                str(exact_out),
                "--report-storage",
                "shared",
                "--max-report-bytes",
                str(shared_size),
                "--max-html-bytes",
                str(html_size),
            ]
        )
        self.assertEqual(code, 0, errors)
        self.assertEqual(provider.calls, [])
        self.assertEqual(exact_out.stat().st_size, html_size)
        self.assertEqual(
            exact_out.with_suffix(".report.json").stat().st_size,
            shared_size,
        )

        too_small = self.root / "below.html"
        code, _, errors, _ = self.run_cli(
            [
                "render",
                str(shared_path),
                "--out",
                str(too_small),
                "--max-report-bytes",
                str(shared_size - 1),
            ]
        )
        self.assertEqual(code, 2)
        self.assertIn("byte limit", errors)
        self.assertFalse(too_small.exists())

        compiler_path = self.root / "compiler.json"
        compiler_path.write_text(
            json.dumps(compile_snapshot(self.snapshot)), encoding="utf-8"
        )
        rejected = self.root / "compiler-only.html"
        code, _, errors, _ = self.run_cli(
            [
                "render",
                str(compiler_path),
                "--out",
                str(rejected),
                "--report-storage",
                "shared",
            ]
        )
        self.assertEqual(code, 2)
        self.assertIn("complete v2 generation proof", errors)
        self.assertFalse(rejected.exists())

    def test_input_growth_after_stat_stays_within_the_exact_limit(
        self,
    ) -> None:
        """Do not trust stat when a report file grows before its bounded read.

        Concern: A file that grows after the initial size check must not pass
            the configured read limit.
        Input/expected matrix:
        * Stat says one byte but the read returns five bytes with a four-byte
          cap -> reject the complete file without parsing it.
        """
        input_path = self.root / "growing.json"
        input_path.write_bytes(b"{}")
        with (
            patch(
                "diffstory.cli.Path.stat",
                return_value=SimpleNamespace(st_size=1),
            ),
            patch(
                "diffstory.cli.Path.open",
                return_value=io.BytesIO(b"12345"),
            ),
            self.assertRaisesRegex(CliError, "configured 4 byte limit"),
        ):
            read_json(str(input_path), max_bytes=4)

    def test_output_path_conflicts_are_rejected_before_any_stage(self) -> None:
        """Do not let a saved snapshot overwrite the report destination."""
        snapshot_path = self.root / "input.snapshot.json"
        snapshot_path.write_text(json.dumps(self.snapshot), encoding="utf-8")
        out = self.root / "conflict.html"
        report_path = out.with_suffix(".report.json")
        report_path.write_bytes(b"existing report")

        code, _, errors, _ = self.run_cli(
            [
                "snapshot",
                str(snapshot_path),
                "--out",
                str(out),
                "--save-snapshot",
                str(report_path),
            ]
        )

        self.assertEqual(code, 2)
        self.assertIn("must use different paths", errors)
        self.assertFalse(out.exists())
        self.assertEqual(report_path.read_bytes(), b"existing report")
        self.assertEqual(list(self.root.glob(".*.tmp")), [])

    def test_saved_snapshot_keeps_legacy_bytes_and_joins_staging(self) -> None:
        """Keep snapshot serialization stable and clean stages after a write error.

        Concern: Save-snapshot bytes and requested report/HTML destinations
            must share the staging boundary without changing source JSON limits.
        Input/expected matrix:
        * Valid snapshot save -> the prior indented UTF-8 JSON bytes are kept.
        * Snapshot parent is a regular file -> staged report and HTML are
          removed and existing destinations remain byte-identical.
        """
        snapshot_path = self.root / "input.snapshot.json"
        snapshot_path.write_text(json.dumps(self.snapshot), encoding="utf-8")
        saved_snapshot = self.root / "saved.snapshot.json"
        out = self.root / "snapshot.html"
        code, _, errors, _ = self.run_cli(
            [
                "snapshot",
                str(snapshot_path),
                "--out",
                str(out),
                "--save-snapshot",
                str(saved_snapshot),
            ]
        )
        self.assertEqual(code, 0, errors)
        self.assertEqual(
            saved_snapshot.read_bytes(),
            json.dumps(
                self.snapshot,
                ensure_ascii=False,
                indent=2,
            ).encode("utf-8"),
        )

        blocked_parent = self.root / "not-a-directory"
        blocked_parent.write_bytes(b"keep blocker")
        failing_out = self.root / "snapshot-failure.html"
        failing_report = failing_out.with_suffix(".report.json")
        failing_out.write_bytes(b"keep html")
        failing_report.write_bytes(b"keep report")
        code, _, errors, _ = self.run_cli(
            [
                "snapshot",
                str(snapshot_path),
                "--out",
                str(failing_out),
                "--save-snapshot",
                str(blocked_parent / "saved.snapshot.json"),
            ]
        )
        self.assertEqual(code, 2)
        self.assertIn("storage", errors)
        self.assertEqual(failing_out.read_bytes(), b"keep html")
        self.assertEqual(failing_report.read_bytes(), b"keep report")
        self.assertEqual(blocked_parent.read_bytes(), b"keep blocker")
        self.assertEqual(list(self.root.glob(".*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
