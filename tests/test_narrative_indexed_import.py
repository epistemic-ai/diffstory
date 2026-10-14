"""Source-free tests for independent indexed report storage imports."""

from __future__ import annotations

import copy
import json
import pickle
import tempfile
import unittest
from types import MappingProxyType
from unittest.mock import patch

from test_narrative_runtime import changed_record
from test_narrative_v2 import FixtureProvider
from test_narrative_v2 import golden_report

from diffstory.analysis import apply_annotations
from diffstory.narrative import Narrator
from diffstory.narrative_contracts import ClaimRef
from diffstory.narrative_manifest import GenerationManifest
from diffstory.narrative_manifest import expanded_generation
from diffstory.narrative_manifest import result_digest
from diffstory.narrative_plan import canonical_bytes
from diffstory.narrative_plan import content_digest
from diffstory.narrative_runtime import ArtifactRuntimeError
from diffstory.narrative_session import ReportSession
from diffstory.narrative_session import ValidatedGeneration
from diffstory.report_storage import _logical_record_for_validation
from diffstory.report_storage import _reader_logical_chunks
from diffstory.report_storage import _streamed_storage_address
from diffstory.report_storage import encode_report_storage
from diffstory.report_storage import expand_report_storage
from diffstory.report_storage import parse_report_storage


class IndexedStorageImportTests(unittest.TestCase):
    """Use one complete synthetic generation and no live providers."""

    @classmethod
    def setUpClass(cls) -> None:
        """Prepare complete source-free report and annotation storage bytes."""
        cls.report = golden_report()
        with tempfile.TemporaryDirectory() as directory:
            cls.annotations = Narrator(
                FixtureProvider(),
                architecture="v2",
                recovery_dir=directory,
                concurrency=1,
            ).generate(cls.report)
        cls.applied = apply_annotations(cls.report, cls.annotations)
        cls.report_bytes = encode_report_storage(cls.applied, "report")
        cls.annotation_bytes = encode_report_storage(
            cls.annotations, "annotations"
        )
        cls.expanded_report = {
            **cls.applied,
            "generation": expanded_generation(cls.applied["generation"]),
        }
        cls.expanded_annotations = {
            **copy.deepcopy(cls.annotations),
            "generation": expanded_generation(cls.annotations["generation"]),
        }

    @staticmethod
    def _proof(
        session: ReportSession, generation: ValidatedGeneration
    ) -> object:
        """Select the private proof payload for structural unit assertions."""
        return session._registered_payload(  # noqa: SLF001 # Tests inspect the exact issued proof registration.
            generation, ValidatedGeneration, "generation"
        )

    @staticmethod
    def _rehashed_reader_substitution(encoded: bytes) -> bytes:
        """Duplicate one unused-by-parser reader selector and readdress it."""
        envelope = json.loads(encoded)
        indexes = {
            table: {entry["id"]: entry for entry in entries}
            for table, entries in envelope["tables"].items()
        }
        reader_id = envelope["root"]["reader"]["$ref"]
        entry = indexes["readers"][reader_id]
        reader = entry["record"]
        selector = next(
            name
            for name in ("citations", "support_provenance")
            if reader[name]
        )
        reader[selector].append(copy.deepcopy(reader[selector][0]))
        changed_id = _streamed_storage_address(
            "reader", _reader_logical_chunks(reader, indexes)
        )
        entry["id"] = changed_id
        envelope["root"]["reader"]["$ref"] = changed_id
        envelope["tables"]["readers"].sort(key=lambda item: item["id"])
        return canonical_bytes(envelope)

    @staticmethod
    def _rehashed_reader_order_substitution(encoded: bytes) -> bytes:
        """Reverse ordered sections and recalculate the reader content ID."""
        envelope = json.loads(encoded)
        indexes = {
            table: {entry["id"]: entry for entry in entries}
            for table, entries in envelope["tables"].items()
        }
        reader_id = envelope["root"]["reader"]["$ref"]
        entry = indexes["readers"][reader_id]
        section_field = (
            "sections"
            if entry["record"]["shape"] == "presentation"
            else "steps"
        )
        entry["record"][section_field].reverse()
        changed_id = _streamed_storage_address(
            "reader",
            _reader_logical_chunks(entry["record"], indexes),
        )
        entry["id"] = changed_id
        envelope["root"]["reader"]["$ref"] = changed_id
        envelope["tables"]["readers"].sort(key=lambda item: item["id"])
        return canonical_bytes(envelope)

    def test_report_and_annotation_import_share_rows_and_match_expanded_oracle(
        self,
    ) -> None:
        """Concern: exact imported data and shared application rows.

        Input/expected matrix:
        * Complete report bytes -> one independent proof and detached oracle
          parity; application graph retains the exact captured compiler base.
        * Complete annotation bytes plus matching compiler report -> one
          independent proof; report handoff shares result and audit row objects.
        * Either indexed generation passed to apply -> explicit M6 renderer
          seal deferral, with no ``ValidatedReport`` issued.
        """
        expansions = [
            expand_report_storage(
                parse_report_storage(self.report_bytes),
                max_output_bytes=128 * 1024 * 1024,
            ),
            expand_report_storage(
                parse_report_storage(self.annotation_bytes),
                max_output_bytes=128 * 1024 * 1024,
            ),
        ]
        self.assertEqual(
            canonical_bytes(expansions[0]),
            canonical_bytes(self.expanded_report),
        )
        self.assertEqual(
            canonical_bytes(expansions[1]),
            canonical_bytes(self.expanded_annotations),
        )

        for payload, compiler_report, expected_kind in (
            (self.report_bytes, None, "report"),
            (self.annotation_bytes, self.report, "annotations"),
        ):
            with self.subTest(artifact=expected_kind):
                with (
                    patch(
                        "diffstory.report_storage.expand_report_storage",
                        side_effect=AssertionError(
                            "import expanded the graph"
                        ),
                    ),
                    patch(
                        "diffstory.narrative_v2.validate_production",
                        side_effect=AssertionError(
                            "import used the full oracle"
                        ),
                    ),
                    patch.object(
                        GenerationManifest,
                        "model_validate",
                        side_effect=AssertionError(
                            "import built a full manifest"
                        ),
                    ),
                ):
                    session, generation = ReportSession.import_indexed(
                        payload,
                        compiler_report=compiler_report,
                        max_input_bytes=len(payload),
                    )
                try:
                    proof = self._proof(session, generation)
                    self.assertEqual(
                        type(proof.records).__name__, "GenerationRecordSet"
                    )
                    self.assertEqual(
                        type(proof).__name__, "_IndexedGenerationProof"
                    )
                    self.assertIsNotNone(proof.application_payload)
                    handoff = proof.application_payload
                    self.assertIs(
                        handoff.base_report,
                        session._captured_input.report_view,  # noqa: SLF001 # Confirm exact owned compiler-base identity.
                    )
                    if expected_kind == "report":
                        self.assertIs(handoff.graph, proof.graph)
                        self.assertNotEqual(
                            proof.report_digest,
                            content_digest(handoff.original_run_bytes),
                        )
                        oracle_graph = expand_report_storage(
                            handoff.graph,
                            max_output_bytes=128 * 1024 * 1024,
                        )
                        self.assertEqual(
                            canonical_bytes(oracle_graph),
                            canonical_bytes(self.expanded_report),
                        )
                    else:
                        self.assertIsNot(handoff.graph, proof.graph)
                        for table, source_rows in proof.graph._tables.items():  # noqa: SLF001 # Compare every retained table row by physical identity.
                            destination_rows = handoff.graph._tables[table]  # noqa: SLF001
                            for identifier in set(source_rows) & set(
                                destination_rows
                            ):
                                self.assertIs(
                                    source_rows[identifier],
                                    destination_rows[identifier],
                                )
                        self.assertEqual(
                            canonical_bytes(
                                expand_report_storage(
                                    proof.graph,
                                    max_output_bytes=128 * 1024 * 1024,
                                )
                            ),
                            canonical_bytes(self.expanded_annotations),
                        )
                        application_report = expand_report_storage(
                            handoff.graph,
                            max_output_bytes=128 * 1024 * 1024,
                        )
                        captured_base = canonical_bytes(
                            session.captured_report()
                        )
                        overlay = json.loads(canonical_bytes(handoff.overlay))
                        application_report["document"] = overlay["document"]
                        groups_by_id = {
                            group["id"]: group
                            for group in application_report["groups"]
                        }
                        for group_overlay in overlay["groups"]:
                            group = groups_by_id[group_overlay["group_id"]]
                            for field in (
                                "title",
                                "number",
                                "next_id",
                                "narrative",
                            ):
                                group[field] = group_overlay[field]
                        self.assertEqual(
                            canonical_bytes(application_report),
                            canonical_bytes(self.expanded_report),
                        )
                        application_report["meta"]["description"] = (
                            "detached expansion mutation"
                        )
                        self.assertEqual(
                            canonical_bytes(session.captured_report()),
                            captured_base,
                        )
                        with self.assertRaises(TypeError):
                            handoff.overlay["document"] = {}
                    with self.assertRaisesRegex(
                        ValueError, "indexed_report_renderer_seal_deferred"
                    ):
                        session.apply(generation)
                finally:
                    session.close()

    def test_report_import_captures_compiler_base_once_without_thawing(
        self,
    ) -> None:
        """Concern: import peak memory avoids another complete base copy.

        Input/expected matrix:
        * Complete report storage -> parser capture visits each base value
          once, does not thaw or resnapshot the base, and transfers one exact
          immutable object to the session, graph, and application payload.
        """
        from diffstory import narrative_session
        from diffstory import report_storage

        parser = report_storage._parse_report_storage_for_import  # noqa: SLF001 # Retain the real byte parser for capture identity evidence.
        strict_load = report_storage._strict_json_load  # noqa: SLF001 # Observe the actual parser-owned source base.
        capture_base = report_storage._capture_import_base_report  # noqa: SLF001
        capture_value = report_storage._capture_import_base_value  # noqa: SLF001
        thaw = report_storage._thaw  # noqa: SLF001
        snapshot = narrative_session._snapshot_report  # noqa: SLF001
        parsed_values = []
        raw_bases = []
        thawed_bases = []
        captured_nodes = []

        def retain_parsed(*args: object, **kwargs: object) -> object:
            """Record the real parser result without replacing its checks."""
            parsed = parser(*args, **kwargs)
            parsed_values.append(parsed)
            return parsed

        def retain_loaded(*args: object, **kwargs: object) -> object:
            """Record the actual mutable compiler base from strict JSON."""
            result = strict_load(*args, **kwargs)
            raw_bases.append(result[0]["root"]["base_report"])
            return result

        def count_capture_value(value: object) -> object:
            """Count every node traversed by the actual base capture."""
            captured_nodes.append(value)
            return capture_value(value)

        def count_base_thaw(value: object) -> object:
            """Record thaw calls that target either form of this exact base."""
            if any(value is base for base in raw_bases) or any(
                value is parsed.root.get("base_report")
                for parsed in parsed_values
            ):
                thawed_bases.append(value)
            return thaw(value)

        def count_report_snapshot(value: object) -> object:
            """Expose any full report recapture during this import."""
            return snapshot(value)

        with (
            patch.object(
                report_storage,
                "_parse_report_storage_for_import",
                side_effect=retain_parsed,
            ),
            patch.object(
                report_storage,
                "_strict_json_load",
                side_effect=retain_loaded,
            ),
            patch.object(
                report_storage,
                "_capture_import_base_report",
                wraps=capture_base,
            ) as base_captures,
            patch.object(
                report_storage,
                "_capture_import_base_value",
                side_effect=count_capture_value,
            ),
            patch.object(report_storage, "_thaw", side_effect=count_base_thaw),
            patch.object(
                narrative_session,
                "_snapshot_report",
                side_effect=count_report_snapshot,
            ) as report_snapshots,
        ):
            session, generation = ReportSession.import_indexed(
                self.report_bytes,
                max_input_bytes=len(self.report_bytes),
            )
        try:
            proof = self._proof(session, generation)
            parsed_base = parsed_values[0].root["base_report"]
            expected_base = {
                key: value
                for key, value in self.applied.items()
                if key not in {"generation", "narrative_v2"}
            }
            self.assertEqual(len(parsed_values), 1)
            self.assertEqual(base_captures.call_count, 1)
            self.assertEqual(
                len(captured_nodes), self._json_node_count(expected_base)
            )
            self.assertEqual(
                sum(
                    type(value) in {dict, list, tuple}
                    for value in captured_nodes
                ),
                self._json_container_count(expected_base),
            )
            self.assertEqual(thawed_bases, [])
            self.assertEqual(report_snapshots.call_count, 0)
            captured_base = (
                session._captured_input.report_view  # noqa: SLF001 # Inspect the private owner for identity evidence.
            )
            self.assertIs(parsed_base, captured_base)
            self.assertIs(parsed_base, proof.graph._root["base_report"])  # noqa: SLF001 # The validated report graph shares the exact base.
            self.assertIs(parsed_base, proof.application_payload.base_report)
        finally:
            session.close()

    def test_public_capture_still_rejects_arbitrary_mapping_views(
        self,
    ) -> None:
        """Concern: private parsed ownership does not widen public capture.

        Input/expected matrix:
        * Read-only mapping view of a compiler report -> rejected by the public
          capture boundary with the existing report-object type error.
        """
        configuration = self.expanded_annotations["generation"][
            "request_configuration"
        ]
        with self.assertRaisesRegex(TypeError, "report_must_be_json_object"):
            ReportSession.capture(
                MappingProxyType(dict(self.report)), configuration
            )

    @staticmethod
    def _json_node_count(value: object) -> int:
        """Count JSON values visited by one complete recursive capture pass."""
        if type(value) is dict:
            return 1 + sum(
                IndexedStorageImportTests._json_node_count(item)
                for item in value.values()
            )
        if type(value) in {list, tuple}:
            return 1 + sum(
                IndexedStorageImportTests._json_node_count(item)
                for item in value
            )
        return 1

    @staticmethod
    def _json_container_count(value: object) -> int:
        """Count immutable report containers allocated by one capture pass."""
        if type(value) is dict:
            return 1 + sum(
                IndexedStorageImportTests._json_container_count(item)
                for item in value.values()
            )
        if type(value) in {list, tuple}:
            return 1 + sum(
                IndexedStorageImportTests._json_container_count(item)
                for item in value
            )
        return 0

    def test_import_requires_matching_compiler_input_and_exact_byte_limit(
        self,
    ) -> None:
        """Concern: new-session import inputs are explicit and bounded.

        Input/expected matrix:
        * Annotation bytes without a compiler report -> required-input error.
        * Annotation bytes with a changed source revision -> dependency error.
        * Report bytes with an unnecessary compiler report -> unused-input error.
        * Either valid artifact with a cap one byte below its size -> size error.
        * Valid report bytes under an exact-sized cap -> a live session and
          registered ``ValidatedGeneration``.
        """
        cases = (
            (
                self.annotation_bytes,
                None,
                len(self.annotation_bytes),
                "compiler_report_required",
            ),
            (
                self.annotation_bytes,
                {
                    **self.report,
                    "meta": {**self.report["meta"], "head_sha": "wrong"},
                },
                len(self.annotation_bytes),
                "dependencies_changed",
            ),
            (
                self.report_bytes,
                self.report,
                len(self.report_bytes),
                "compiler_report_not_used_for_report",
            ),
            (
                self.report_bytes,
                None,
                len(self.report_bytes) - 1,
                "storage_input_too_large",
            ),
        )
        for payload, compiler_report, limit, reason in cases:
            with (
                self.subTest(reason=reason),
                self.assertRaisesRegex(
                    (ValueError, ArtifactRuntimeError), reason
                ),
            ):
                ReportSession.import_indexed(
                    payload,
                    compiler_report=compiler_report,
                    max_input_bytes=limit,
                )

        with self.assertRaisesRegex(ValueError, "invalid_input_limit"):
            ReportSession.import_indexed(
                self.report_bytes,
                max_input_bytes=True,
            )

        session, generation = ReportSession.import_indexed(
            self.report_bytes,
            max_input_bytes=len(self.report_bytes),
        )
        try:
            self.assertIsInstance(generation, ValidatedGeneration)
            self.assertIsNotNone(self._proof(session, generation))
        finally:
            session.close()

    def test_omitted_defaults_remain_omitted_and_header_scalars_stay_strict(
        self,
    ) -> None:
        """Concern: JSON default presence and scalar types survive import.

        Input/expected matrix:
        * Annotation generation with omitted default provenance -> accepted
          and the proof header keeps provenance absent.
        * Applied report with an omitted final ``next_id`` whose value is
          ``None`` -> accepted under the existing group-field semantics.
        * Header usage count encoded as ``true`` -> rejected as a strict
          scalar mismatch.
        * Reordered ledger unit keys -> rejected against the captured index.
        """
        omitted_header = copy.deepcopy(self.expanded_annotations)
        omitted_header["generation"].pop("provenance")
        omitted_bytes = encode_report_storage(omitted_header, "annotations")
        session, generation = ReportSession.import_indexed(
            omitted_bytes,
            compiler_report=self.report,
            max_input_bytes=len(omitted_bytes),
        )
        try:
            proof = self._proof(session, generation)
            self.assertNotIn("provenance", proof.records.header)
            expanded = expand_report_storage(
                proof.graph, max_output_bytes=128 * 1024 * 1024
            )
            self.assertNotIn("provenance", expanded["generation"])
        finally:
            session.close()

        omitted_group_default = copy.deepcopy(self.applied)
        final_group = omitted_group_default["groups"][-1]
        self.assertIsNone(final_group.get("next_id"))
        final_group.pop("next_id", None)
        report_bytes = encode_report_storage(omitted_group_default, "report")
        session, generation = ReportSession.import_indexed(
            report_bytes, max_input_bytes=len(report_bytes)
        )
        session.close()

        strict_scalar = json.loads(self.annotation_bytes)
        strict_scalar["root"]["generation"]["current_usage"]["calls"] = True
        strict_scalar_bytes = canonical_bytes(strict_scalar)
        with self.assertRaises(ValueError):
            ReportSession.import_indexed(
                strict_scalar_bytes,
                compiler_report=self.report,
                max_input_bytes=len(strict_scalar_bytes),
            )

        wrong_order = json.loads(self.annotation_bytes)
        wrong_order["root"]["generation"]["unit_keys"].reverse()
        wrong_order_bytes = canonical_bytes(wrong_order)
        with self.assertRaises((ValueError, ArtifactRuntimeError)):
            ReportSession.import_indexed(
                wrong_order_bytes,
                compiler_report=self.report,
                max_input_bytes=len(wrong_order_bytes),
            )

    def test_rehashed_result_receipt_and_request_peers_need_current_proof(
        self,
    ) -> None:
        """Concern: readdressing stored peers cannot replace producer proof.

        Input/expected matrix:
        * Unsupported section claim with a matching new result receipt digest
          and freshly encoded storage addresses -> import rejects citation or
          support derivation.
        * Changed producer request input bound with storage re-encoding ->
          import rejects the request metadata against the current plan.
        * A reversed serialized dependency array -> expansion preserves its
          order exactly while typed proof uses the existing canonical peer
          semantics.
        """
        manifest = GenerationManifest.model_validate_json(
            canonical_bytes(self.expanded_annotations["generation"])
        )
        original_section = manifest.sections[0]
        unsupported = changed_record(
            original_section.synthesis[0],
            claims=(
                ClaimRef(
                    unit_key=manifest.details[0].unit_key,
                    observation_id="0" * 64,
                ),
            ),
        )
        changed_section = changed_record(
            original_section, synthesis=(unsupported,)
        )
        changed_manifest = changed_record(
            manifest,
            sections=(changed_section, *manifest.sections[1:]),
            receipts=tuple(
                changed_record(
                    receipt, result_digest=result_digest(changed_section)
                )
                if receipt.item_key == changed_section.item_key
                else receipt
                for receipt in manifest.receipts
            ),
        )
        candidate = copy.deepcopy(self.expanded_annotations)
        candidate["generation"] = changed_manifest.model_dump(mode="json")
        candidate["steps"][0]["prose"] = changed_section.model_dump(
            mode="json"
        )
        changed_result_bytes = encode_report_storage(candidate, "annotations")
        self.assertEqual(
            parse_report_storage(changed_result_bytes).artifact_kind,
            "annotations",
        )
        with self.assertRaises((ValueError, ArtifactRuntimeError)):
            ReportSession.import_indexed(
                changed_result_bytes,
                compiler_report=self.report,
                max_input_bytes=len(changed_result_bytes),
            )

        first_producer = manifest.producers[0]
        first_request = first_producer.requests[0]
        altered_producer = changed_record(
            first_producer,
            requests=(
                changed_record(
                    first_request,
                    batch=changed_record(
                        first_request.batch,
                        input_bound=first_request.batch.input_bound + 1,
                    ),
                ),
                *first_producer.requests[1:],
            ),
        )
        altered_manifest = changed_record(
            manifest,
            producers=(altered_producer, *manifest.producers[1:]),
            receipts=tuple(
                changed_record(receipt, producer_key=altered_producer.key)
                if receipt.producer_key == first_producer.key
                else receipt
                for receipt in manifest.receipts
            ),
        )
        candidate = copy.deepcopy(self.expanded_annotations)
        candidate["generation"] = altered_manifest.model_dump(mode="json")
        altered_request_bytes = encode_report_storage(candidate, "annotations")
        self.assertEqual(
            parse_report_storage(altered_request_bytes).artifact_kind,
            "annotations",
        )
        with self.assertRaises((ValueError, ArtifactRuntimeError)):
            ReportSession.import_indexed(
                altered_request_bytes,
                compiler_report=self.report,
                max_input_bytes=len(altered_request_bytes),
            )

        dependency = next(
            item
            for item in first_producer.invocation.dependencies
            if item.kind == "peer"
        )
        changed_dependencies = tuple(
            changed_record(item, digest="0" * 64)
            if item.kind in {"peer", "evidence"} and item.key == dependency.key
            else item
            for item in first_producer.invocation.dependencies
        )
        altered_peer_producer = changed_record(
            first_producer,
            invocation=changed_record(
                first_producer.invocation,
                dependencies=changed_dependencies,
            ),
        )
        peer_artifact_keys = altered_peer_producer.invocation.artifact_keys()
        altered_peer_manifest = changed_record(
            manifest,
            producers=(
                altered_peer_producer,
                *manifest.producers[1:],
            ),
            receipts=tuple(
                changed_record(
                    receipt,
                    producer_key=altered_peer_producer.key,
                    artifact_key=peer_artifact_keys[receipt.item_key],
                )
                if receipt.producer_key == first_producer.key
                else receipt
                for receipt in manifest.receipts
            ),
        )
        candidate = copy.deepcopy(self.expanded_annotations)
        candidate["generation"] = altered_peer_manifest.model_dump(mode="json")
        altered_peer_bytes = encode_report_storage(candidate, "annotations")
        self.assertEqual(
            parse_report_storage(altered_peer_bytes).artifact_kind,
            "annotations",
        )
        with self.assertRaises((ValueError, ArtifactRuntimeError)):
            ReportSession.import_indexed(
                altered_peer_bytes,
                compiler_report=self.report,
                max_input_bytes=len(altered_peer_bytes),
            )

        reordered = copy.deepcopy(self.expanded_annotations)
        peer_order = reordered["generation"]["producers"][0]["invocation"][
            "dependencies"
        ]
        peer_order.reverse()
        reordered_bytes = encode_report_storage(reordered, "annotations")
        expanded_reordered = expand_report_storage(
            parse_report_storage(reordered_bytes),
            max_output_bytes=128 * 1024 * 1024,
        )
        self.assertEqual(
            canonical_bytes(expanded_reordered), canonical_bytes(reordered)
        )
        session, generation = ReportSession.import_indexed(
            reordered_bytes,
            compiler_report=self.report,
            max_input_bytes=len(reordered_bytes),
        )
        try:
            proof = self._proof(session, generation)
            expanded_import = expand_report_storage(
                proof.graph, max_output_bytes=128 * 1024 * 1024
            )
            self.assertEqual(
                canonical_bytes(expanded_import),
                canonical_bytes(reordered),
            )
        finally:
            session.close()

    def test_empty_and_partial_storage_never_registers_generation_authority(
        self,
    ) -> None:
        """Concern: empty or partial row closure cannot issue a generation.

        Input/expected matrix:
        * Complete annotations represented by a non-byte storage view ->
          rejected at the bytes-only import boundary.
        * Empty detail selectors -> rejected for incomplete ledger coverage.
        * Partial planning selectors -> rejected for incomplete plan coverage.
        * Empty producer selectors -> rejected for incomplete producer coverage.
        """
        with self.assertRaisesRegex(ValueError, "storage_input_must_be_bytes"):
            ReportSession.import_indexed(
                parse_report_storage(self.annotation_bytes),
                compiler_report=self.report,
                max_input_bytes=len(self.annotation_bytes),
            )

        mutations = (
            ("details", []),
            (
                "planning",
                json.loads(self.annotation_bytes)["root"]["generation"][
                    "planning"
                ][:-1],
            ),
            ("producers", []),
        )
        for field, value in mutations:
            with self.subTest(field=field):
                envelope = json.loads(self.annotation_bytes)
                envelope["root"]["generation"][field] = value
                raw = canonical_bytes(envelope)
                with self.assertRaises((ValueError, ArtifactRuntimeError)):
                    ReportSession.import_indexed(
                        raw,
                        compiler_report=self.report,
                        max_input_bytes=len(raw),
                    )

    def test_normalized_url_rows_reject_unmodeled_fields(self) -> None:
        """Concern: compact URL parsing must retain the closed row schema.

        Input/expected matrix:
        * One normalized URL row with exactly ``value`` -> strict row model
          validation accepts the logical URL.
        * The same row with an extra normalized field -> strict row
          validation rejects the unmodeled field before projecting it away.
        """
        valid_tables = {
            "urls": {
                "url-id": {
                    "record": {"value": "https://example.test/"},
                    "kind": "url",
                }
            }
        }
        logical, _, _ = _logical_record_for_validation(
            valid_tables, "urls", "url-id"
        )
        self.assertEqual(logical, {"value": "https://example.test/"})

        invalid_tables = {
            "urls": {
                "url-id": {
                    "record": {
                        "value": "https://example.test/",
                        "unmodeled": "extra",
                    },
                    "kind": "url",
                }
            }
        }
        with self.assertRaisesRegex(ValueError, "invalid_url_record"):
            _logical_record_for_validation(invalid_tables, "urls", "url-id")

    def test_independent_audit_dispatch_runs_once_without_full_expansion(
        self,
    ) -> None:
        """Concern: one indexed proof retains actual work and shared audits.

        Input/expected matrix:
        * One complete annotation artifact -> one indexed production proof,
          repeated existing stage-audit calls, and one validation count per
          unique stored row family.
        * One current compiler report -> one ledger, one ledger index, and one
          compiler-fact scan; the exact source owner metadata fingerprint
          count and full report digest bytes are also measured.
        * Expansion, legacy full validation, and full manifest construction
          during the import -> immediate test failure.
        """
        from diffstory import narrative_session
        from diffstory import narrative_v2
        from diffstory import report_storage
        from diffstory.narrative_hierarchy import _IndexedSourceState
        from diffstory.narrative_plan import build_evidence_ledger
        from diffstory.narrative_plan import build_ledger_index
        from diffstory.narrative_v2 import planning_waves

        audit = narrative_v2._audit_stage_records  # noqa: SLF001 # Wrap the shared proof algorithm to distinguish real stage audits.
        indexed = narrative_v2.validate_indexed_production
        fixture_manifest = GenerationManifest.model_validate_json(
            canonical_bytes(self.expanded_annotations["generation"])
        )
        expected_audits = 1 + len(planning_waves(fixture_manifest.plan)) + 2
        metadata_fingerprint = _IndexedSourceState.current_metadata_fingerprint
        metadata_scans = []
        typed_record = report_storage._validate_typed_table_record  # noqa: SLF001 # Count actual strict row model validation calls.
        reader_record = report_storage._validate_reader_record  # noqa: SLF001 # Count skeleton validation separately from table dispatch.
        digest_inputs = []
        original_digest = narrative_session.canonical_digest

        def measure_digest(value: object) -> str:
            """Count exact objects streamed by the captured raw hasher."""
            digest_inputs.append(value)
            return original_digest(value)

        def measure_metadata_fingerprint(
            state: _IndexedSourceState,
        ) -> str:
            """Count full metadata witness streams for the one source owner."""
            metadata_scans.append(state)
            return metadata_fingerprint(state)

        with (
            patch.object(
                narrative_v2,
                "_audit_stage_records",
                wraps=audit,
            ) as audit_calls,
            patch.object(
                narrative_v2,
                "validate_indexed_production",
                wraps=indexed,
            ) as indexed_calls,
            patch(
                "diffstory.narrative_v2.build_evidence_ledger",
                wraps=build_evidence_ledger,
            ) as ledgers,
            patch(
                "diffstory.narrative_v2.build_ledger_index",
                wraps=build_ledger_index,
            ) as indexes,
            patch(
                "diffstory.narrative_session.compiler_fact_signature",
                wraps=narrative_session.compiler_fact_signature,
            ) as fact_scans,
            patch.object(
                narrative_session,
                "canonical_digest",
                side_effect=measure_digest,
            ),
            patch.object(
                _IndexedSourceState,
                "current_metadata_fingerprint",
                new=measure_metadata_fingerprint,
            ),
            patch.object(
                report_storage,
                "_validate_typed_table_record",
                wraps=typed_record,
            ) as typed_calls,
            patch.object(
                report_storage,
                "_validate_reader_record",
                wraps=reader_record,
            ) as reader_calls,
            patch(
                "diffstory.report_storage.expand_report_storage",
                side_effect=AssertionError("import expanded the graph"),
            ),
            patch(
                "diffstory.narrative_v2.validate_production",
                side_effect=AssertionError("import used the full oracle"),
            ),
            patch.object(
                GenerationManifest,
                "model_validate",
                side_effect=AssertionError("import built a full manifest"),
            ),
        ):
            session, generation = ReportSession.import_indexed(
                self.annotation_bytes,
                compiler_report=self.report,
                max_input_bytes=len(self.annotation_bytes),
            )
        try:
            proof = self._proof(session, generation)
            self.assertEqual(indexed_calls.call_count, 1)
            self.assertEqual(audit_calls.call_count, expected_audits)
            self.assertEqual(ledgers.call_count, 1)
            self.assertEqual(indexes.call_count, 1)
            self.assertEqual(fact_scans.call_count, 1)
            self.assertEqual(len(metadata_scans), 69)
            self.assertEqual(len({id(state) for state in metadata_scans}), 1)
            raw_hash_passes = sum(
                value is session._captured_input.report_view  # noqa: SLF001 # Count actual full-root hasher inputs.
                for value in digest_inputs
            )
            self.assertEqual(raw_hash_passes, 5)
            raw_report_bytes = len(
                canonical_bytes(session._captured_input.report_view)  # noqa: SLF001 # Measure the exact report stream size after import.
            )
            expected_raw_report_bytes = len(canonical_bytes(self.report))
            self.assertEqual(raw_report_bytes, expected_raw_report_bytes)
            self.assertEqual(
                raw_hash_passes * raw_report_bytes,
                5 * expected_raw_report_bytes,
            )
            counts = proof.graph._typed_validation_counts  # noqa: SLF001 # Read compact parser counts rather than model expansions.
            rows = proof.graph._tables  # noqa: SLF001 # Compare actual typed validation against unique table rows.
            self.assertEqual(
                typed_calls.call_count,
                sum(
                    len(entries)
                    for table, entries in rows.items()
                    if table != "readers"
                ),
            )
            self.assertEqual(reader_calls.call_count, len(rows["readers"]) + 1)
            self.assertEqual(
                sum(counts.values()),
                sum(len(entries) for entries in rows.values()),
            )
            self.assertEqual(
                counts,
                session._arena._validation_counts,  # noqa: SLF001 # Parser facts seed this exact session-owned arena.
            )
        finally:
            session.close()

    def test_rehashed_reader_substitution_fails_current_derivation(
        self,
    ) -> None:
        """Concern: content addresses cannot grant reader completion authority.

        Input/expected matrix:
        * Reordered duplicate citation/provenance selector with a recomputed
          reader address -> structural storage parse succeeds.
        * The same self-consistent graph imported with its matching compiler
          report -> independent reader comparison rejects it.
        * Reversed section order with a new reader address -> strict shared
          relationships reject the wrong ordered row binding.
        """
        changed = self._rehashed_reader_substitution(self.annotation_bytes)
        self.assertEqual(
            parse_report_storage(changed).artifact_kind, "annotations"
        )
        with self.assertRaisesRegex(
            (ValueError, ArtifactRuntimeError), "reader_mismatch"
        ):
            ReportSession.import_indexed(
                changed,
                compiler_report=self.report,
                max_input_bytes=len(changed),
            )

        reordered = self._rehashed_reader_order_substitution(
            self.annotation_bytes
        )
        with self.assertRaisesRegex(ValueError, "reader_section_mismatch"):
            parse_report_storage(reordered)
        with self.assertRaises((ValueError, ArtifactRuntimeError)):
            ReportSession.import_indexed(
                reordered,
                compiler_report=self.report,
                max_input_bytes=len(reordered),
            )

    def test_caller_mutation_close_copy_and_foreign_handles_do_not_transfer(
        self,
    ) -> None:
        """Concern: import proof and source bindings remain session-owned.

        Input/expected matrix:
        * Annotation bytes plus one caller compiler mapping -> one complete
          caller capture; later mutation cannot change the owned report.
        * Copy or pickle a session/token -> rejected.
        * Apply the exact token in a second imported session -> rejected.
        * Close the issuing session -> its token and retained registry expire.
        """
        from diffstory import narrative_session

        original = copy.deepcopy(self.report)
        snapshot_report = narrative_session._snapshot_report  # noqa: SLF001 # Count actual caller-base capture work.
        with patch.object(
            narrative_session,
            "_snapshot_report",
            wraps=snapshot_report,
        ) as snapshot_calls:
            session, generation = ReportSession.import_indexed(
                self.annotation_bytes,
                compiler_report=original,
                max_input_bytes=len(self.annotation_bytes),
            )
        self.assertEqual(snapshot_calls.call_count, 1)
        self.assertIs(snapshot_calls.call_args.args[0], original)
        captured = session._captured_input  # noqa: SLF001 # Retain a test handle to check close-time alias clearing.
        source_binding = captured._source_binding  # noqa: SLF001
        source_state = source_binding._source_state  # noqa: SLF001 # Inspect the operation-local citation cache that close must expire.
        source_scope = source_state.details_context._citation_scope  # noqa: SLF001
        foreign, _foreign_generation = ReportSession.import_indexed(
            self.report_bytes,
            max_input_bytes=len(self.report_bytes),
        )
        try:
            original["meta"]["head_sha"] = "caller-mutated"
            self.assertEqual(
                canonical_bytes(session.captured_report()),
                canonical_bytes(self.report),
            )
            with self.assertRaises(TypeError):
                copy.copy(session)
            with self.assertRaises(TypeError):
                pickle.dumps(session)
            with self.assertRaises(TypeError):
                copy.copy(generation)
            with self.assertRaisesRegex(
                ValueError, "invalid_session_capability"
            ):
                foreign.apply(generation)
            with self.assertRaisesRegex(
                ValueError, "indexed_report_renderer_seal_deferred"
            ):
                session.apply(generation)
        finally:
            foreign.close()
            session.close()
        with self.assertRaisesRegex(ValueError, "session_closed"):
            session.apply(generation)
        self.assertEqual(session._registrations, {})  # noqa: SLF001 # Closing drops the exact capability registry.
        self.assertIsNone(session._arena)  # noqa: SLF001 # Closing drops imported row aliases.
        self.assertIsNone(captured._source_binding)  # noqa: SLF001 # Closing removes the source adapter alias.
        self.assertTrue(source_scope.closed)
        self.assertIsNone(captured._ledger)  # noqa: SLF001 # Closing removes ledger and result-index aliases.


if __name__ == "__main__":
    unittest.main()
