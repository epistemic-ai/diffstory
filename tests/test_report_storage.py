"""Lossless structural storage tests using only local synthetic reports."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from collections import Counter
from typing import TYPE_CHECKING
from unittest.mock import patch

if TYPE_CHECKING:
    from pydantic import BaseModel

from test_narrative_runtime import changed_record
from test_narrative_v2 import FixtureProvider
from test_narrative_v2 import golden_report

from diffstory import report_storage
from diffstory.analysis import apply_annotations
from diffstory.narrative import Narrator
from diffstory.narrative_contracts import ClaimRef
from diffstory.narrative_contracts import DetailResult
from diffstory.narrative_contracts import DocumentResult
from diffstory.narrative_manifest import GenerationManifest
from diffstory.narrative_manifest import InheritedProvenance
from diffstory.narrative_manifest import PresentationSection
from diffstory.narrative_manifest import ResolvedCitation
from diffstory.narrative_manifest import V2Presentation
from diffstory.narrative_manifest import expanded_generation
from diffstory.narrative_manifest import result_digest
from diffstory.narrative_manifest import validate_generation
from diffstory.narrative_plan import canonical_bytes
from diffstory.narrative_runtime import ArtifactRuntimeError
from diffstory.render import render
from diffstory.render import validate_report
from diffstory.report_storage import STORAGE_VERSION
from diffstory.report_storage import ReportReaderProjection
from diffstory.report_storage import ReportRecordGraph
from diffstory.report_storage import ReportStorageView
from diffstory.report_storage import encode_report_storage
from diffstory.report_storage import expand_report_storage
from diffstory.report_storage import parse_report_storage
from diffstory.report_storage import storage_address


def _envelope(encoded: bytes) -> dict:
    return json.loads(encoded)


def _raw(envelope: dict) -> bytes:
    return canonical_bytes(envelope)


def _entry(envelope: dict, table: str, kind: str | None = None) -> dict:
    return next(
        item
        for item in envelope["tables"][table]
        if kind is None or item["kind"] == kind
    )


def _repoint(
    value: object, old: str, new: str, *, kind: str | None = None
) -> object:
    if type(value) is dict:
        if set(value) == {"$ref", "kind"} and value["$ref"] == old:
            return {"$ref": new, "kind": kind or value["kind"]}
        return {
            key: _repoint(item, old, new, kind=kind)
            for key, item in value.items()
        }
    if type(value) is list:
        return [_repoint(item, old, new, kind=kind) for item in value]
    return value


class ReportStorageTests(unittest.TestCase):
    """Use proof-valid expanded v2 and generation-storage.v1 inputs."""

    @classmethod
    def setUpClass(cls) -> None:
        """Generate one proof-valid synthetic report with the local provider."""
        cls.report = golden_report()
        cls.provider = FixtureProvider()
        with tempfile.TemporaryDirectory() as directory:
            cls.annotations = Narrator(
                cls.provider,
                architecture="v2",
                recovery_dir=directory,
                concurrency=1,
            ).generate(cls.report)
        cls.applied = apply_annotations(cls.report, cls.annotations)
        cls.expanded_annotations = {
            **cls.annotations,
            "generation": expanded_generation(cls.annotations["generation"]),
        }
        cls.expanded_report = {
            **cls.applied,
            "generation": expanded_generation(cls.applied["generation"]),
        }

    @staticmethod
    def _typed_model(model: type, value: dict) -> BaseModel:
        return model.model_validate_json(canonical_bytes(value))

    def _typed_annotation_reader(self, value: dict) -> ReportReaderProjection:
        return ReportReaderProjection.from_annotations(
            document=self._typed_model(DocumentResult, value["document"]),
            steps=tuple(
                self._typed_model(PresentationSection, item)
                for item in value["steps"]
            ),
            context_details=tuple(
                self._typed_model(DetailResult, item)
                for item in value["context_details"]
            ),
            citations=tuple(
                self._typed_model(ResolvedCitation, item)
                for item in value["citations"]
            ),
            support_provenance=tuple(
                self._typed_model(InheritedProvenance, item)
                for item in value["support_provenance"]
            ),
        )

    def _typed_report_graph(
        self, generation_data: dict | None = None
    ) -> ReportRecordGraph:
        manifest = self._typed_model(
            GenerationManifest,
            generation_data or self.expanded_report["generation"],
        )
        presentation = self._typed_model(
            V2Presentation, self.applied["narrative_v2"]
        )
        base_report = {
            key: value
            for key, value in self.applied.items()
            if key not in {"generation", "narrative_v2"}
        }
        return ReportRecordGraph.from_typed(
            manifest,
            ReportReaderProjection.from_presentation(presentation),
            artifact_kind="report",
            base_report=base_report,
        )

    def assert_round_trip(
        self, value: dict, kind: str, *, expected: dict | None = None
    ) -> tuple[bytes, dict]:
        """Require expansion to match the exact logical compatibility oracle."""
        encoded = encode_report_storage(value, kind)
        view = parse_report_storage(encoded)
        expanded = expand_report_storage(
            view, max_output_bytes=128 * 1024 * 1024
        )
        self.assertEqual(
            canonical_bytes(expanded),
            canonical_bytes(expected if expected is not None else value),
        )
        self.assertEqual(
            parse_report_storage(encoded).byte_length, len(encoded)
        )
        return encoded, expanded

    def _coherent_reader_record_change(  # noqa: PLR0913 # Keep mutation schema inputs explicit.
        self,
        *,
        table: str,
        reader_field: str,
        logical_field: str,
        kind: str,
        change_logical: object,
        change_stored: object,
    ) -> dict:
        """Return an envelope with the record and reader addresses rebuilt."""
        envelope = _envelope(encode_report_storage(self.applied, "report"))
        old_reader_id = envelope["root"]["reader"]["$ref"]
        reader_entry = next(
            item
            for item in envelope["tables"]["readers"]
            if item["id"] == old_reader_id
        )
        reader_record = reader_entry["record"]
        old_record_id = reader_record[reader_field][0]["$ref"]
        affected = [
            index
            for index, pointer in enumerate(reader_record[reader_field])
            if pointer["$ref"] == old_record_id
        ]
        self.assertTrue(affected)
        record_entry = next(
            item
            for item in envelope["tables"][table]
            if item["id"] == old_record_id
        )
        stored_record = copy.deepcopy(record_entry["record"])
        change_stored(stored_record)

        logical_reader = copy.deepcopy(self.applied["narrative_v2"])
        for index in affected:
            change_logical(logical_reader[logical_field][index])
        new_record_id = storage_address(
            kind, logical_reader[logical_field][affected[0]]
        )
        record_entry["id"] = new_record_id
        record_entry["record"] = stored_record
        envelope = _repoint(envelope, old_record_id, new_record_id)
        envelope["tables"][table].sort(key=lambda item: item["id"])

        new_reader_id = storage_address("reader", logical_reader)
        reader_entry = next(
            item
            for item in envelope["tables"]["readers"]
            if item["id"] == old_reader_id
        )
        reader_entry["id"] = new_reader_id
        envelope = _repoint(envelope, old_reader_id, new_reader_id)
        envelope["tables"]["readers"].sort(key=lambda item: item["id"])
        return envelope

    def test_expanded_and_compact_generation_inputs_have_exact_oracles(
        self,
    ) -> None:
        """Round-trip both generation forms and validate through public readers."""
        annotations_bytes, expanded_annotations = self.assert_round_trip(
            self.annotations, "annotations", expected=self.expanded_annotations
        )
        self.assertEqual(
            canonical_bytes(expanded_annotations),
            canonical_bytes(self.expanded_annotations),
        )
        report_bytes, expanded_report = self.assert_round_trip(
            self.applied, "report", expected=self.expanded_report
        )
        self.assertEqual(
            canonical_bytes(expanded_report),
            canonical_bytes(self.expanded_report),
        )
        self.assertEqual(
            expanded_report["source_evidence"], self.applied["source_evidence"]
        )
        self.assertEqual(
            apply_annotations(self.report, expanded_annotations), self.applied
        )
        validate_report(copy.deepcopy(expanded_report))
        self.assertIn(
            "diffstory.presentation.v2", render(copy.deepcopy(expanded_report))
        )
        # Also encode already-expanded semantic v2 inputs without adding defaults.
        self.assertEqual(
            canonical_bytes(
                expand_report_storage(
                    parse_report_storage(
                        encode_report_storage(
                            self.expanded_annotations, "annotations"
                        )
                    ),
                    max_output_bytes=128 * 1024 * 1024,
                )
            ),
            canonical_bytes(self.expanded_annotations),
        )
        self.assertEqual(
            canonical_bytes(
                expand_report_storage(
                    parse_report_storage(
                        encode_report_storage(self.expanded_report, "report")
                    ),
                    max_output_bytes=128 * 1024 * 1024,
                )
            ),
            canonical_bytes(self.expanded_report),
        )
        self.assertGreater(len(annotations_bytes), 0)
        self.assertGreater(len(report_bytes), 0)

    def test_missing_defaults_and_peer_array_order_are_preserved(self) -> None:
        """Keep absent defaults and peer dependency order exactly as supplied."""
        expanded = copy.deepcopy(self.expanded_annotations)
        expanded["generation"].pop("provenance")
        dependencies = expanded["generation"]["producers"][0]["invocation"][
            "dependencies"
        ]
        dependencies.reverse()
        encoded = encode_report_storage(expanded, "annotations")
        view = parse_report_storage(encoded)
        actual = expand_report_storage(
            view, max_output_bytes=128 * 1024 * 1024
        )
        self.assertEqual(canonical_bytes(actual), canonical_bytes(expanded))
        self.assertNotIn("provenance", actual["generation"])
        self.assertEqual(
            actual["generation"]["producers"][0]["invocation"]["dependencies"],
            dependencies,
        )

        wrong_scalar = _envelope(encoded)
        wrong_scalar["root"]["generation"]["current_usage"]["calls"] = True
        with self.assertRaises(ValueError):
            parse_report_storage(_raw(wrong_scalar))

    def test_canonical_storage_round_trip_is_byte_exact(self) -> None:
        """Require canonical stored bytes and reject noncanonical JSON framing."""
        encoded = encode_report_storage(self.applied, "report")
        envelope = _envelope(encoded)
        self.assertEqual(envelope["storage_version"], STORAGE_VERSION)
        self.assertEqual(encoded, canonical_bytes(envelope))
        self.assertEqual(
            parse_report_storage(encoded).byte_length,
            len(encoded),
        )
        with self.assertRaisesRegex(ValueError, "noncanonical_storage_json"):
            parse_report_storage(b" " + encoded)

    def test_shared_results_provenance_and_indexed_access(self) -> None:
        """Share proof and reader results and resolve subsets without expansion."""
        encoded = encode_report_storage(self.applied, "report")
        envelope = _envelope(encoded)
        generation = envelope["root"]["generation"]
        reader_pointer = envelope["root"]["reader"]["$ref"]
        reader = next(
            entry["record"]
            for entry in envelope["tables"]["readers"]
            if entry["id"] == reader_pointer
        )
        self.assertEqual(
            generation["document"]["$ref"], reader["document"]["$ref"]
        )
        generation_detail_ids = {
            item["$ref"] for item in generation["details"]
        }
        reader_detail_ids = {
            item["$ref"]
            for section in reader["sections"]
            for item in section["details"]
        }
        self.assertTrue(generation_detail_ids & reader_detail_ids)
        self.assertLess(
            len(envelope["tables"]["results"]),
            len(generation["details"])
            + len(generation["planning"])
            + len(generation["sections"])
            + 1
            + len(reader["context_details"])
            + sum(
                len(section["details"]) + 1 for section in reader["sections"]
            ),
        )
        for item in envelope["tables"]["inherited_provenance"]:
            record = item["record"]
            self.assertEqual(
                set(record),
                {"key", "reference", "result", "consumption", "receipt"},
            )
            self.assertNotIn("statement", record)

        view = parse_report_storage(encoded)
        with patch(
            "diffstory.report_storage.expand_report_storage",
            side_effect=AssertionError("indexed accessor expanded the report"),
        ):
            sections = list(view.iter_sections())
            self.assertEqual(
                len(sections), len(self.applied["narrative_v2"]["sections"])
            )
            self.assertEqual(
                sections,
                self.applied["narrative_v2"]["sections"],
            )
            self.assertEqual(
                view.citations(),
                tuple(self.expanded_report["narrative_v2"]["citations"]),
            )
            self.assertEqual(
                view.provenance(),
                tuple(
                    self.expanded_report["narrative_v2"]["support_provenance"]
                ),
            )
            audit = view.generation_audit()
            self.assertEqual(audit["version"], "diffstory.generation.v2")
            self.assertEqual(
                len(audit["receipts"]),
                len(self.expanded_report["generation"]["receipts"]),
            )

        if sections:
            detached = view.section_details(sections[0]["group_id"])
            detached[0]["explanations"][0]["before"] = "mutated"
            self.assertEqual(
                view.section_details(sections[0]["group_id"]),
                sections[0]["details"],
            )
        first_result = next(iter(envelope["tables"]["results"]))
        original = view.result(first_result["id"], kind=first_result["kind"])
        mutated = copy.deepcopy(original)
        mutated["version"] = "changed"
        self.assertNotEqual(
            view.result(first_result["id"], kind=first_result["kind"]), mutated
        )

        compact_baseline = len(canonical_bytes(self.applied))
        result_occurrences = (
            len(generation["details"])
            + len(generation["planning"])
            + len(generation["sections"])
            + 1
            + len(reader["context_details"])
            + sum(
                len(section["details"]) + 1 for section in reader["sections"]
            )
        )
        self.assertGreater(
            result_occurrences, len(envelope["tables"]["results"])
        )
        self.assertGreater(compact_baseline, 0)
        self.assertGreater(len(encoded), 0)
        self.assertLess(len(encoded), compact_baseline)

    def test_section_details_resolves_only_the_requested_section(self) -> None:
        """Resolve only selected details, without loading any section prose."""
        from diffstory.report_storage import parse_report_storage

        envelope = _envelope(encode_report_storage(self.applied, "report"))
        reader_id = envelope["root"]["reader"]["$ref"]
        reader = next(
            item["record"]
            for item in envelope["tables"]["readers"]
            if item["id"] == reader_id
        )
        self.assertGreaterEqual(len(reader["sections"]), 2)
        target_index = 1
        target = reader["sections"][target_index]
        requested_ids = {item["$ref"] for item in target["details"]}
        self.assertTrue(requested_ids)
        other_result_ids = {
            pointer["$ref"]
            for index, section in enumerate(reader["sections"])
            if index != target_index
            for pointer in (section["prose"], *section["details"])
        }
        other_result_ids.update(
            section["prose"]["$ref"] for section in reader["sections"]
        )
        resolved: list[tuple[str, str]] = []
        original = ReportStorageView._materialize  # noqa: SLF001 # Spy on the private indexed resolver.

        def tracked(
            view: ReportStorageView, table: str, identifier: str, memo: dict
        ) -> dict:
            resolved.append((table, identifier))
            return original(view, table, identifier, memo)

        view = parse_report_storage(
            encode_report_storage(self.applied, "report")
        )
        with patch.object(ReportStorageView, "_materialize", new=tracked):
            actual = view.section_details(target["group_id"])
        resolved_result_ids = {
            identifier for table, identifier in resolved if table == "results"
        }
        self.assertEqual(resolved_result_ids, requested_ids)
        self.assertTrue(resolved_result_ids.isdisjoint(other_result_ids))
        self.assertEqual(
            actual,
            self.applied["narrative_v2"]["sections"][target_index]["details"],
        )

    def test_typed_graph_builds_without_expanded_codec_roundtrip(self) -> None:
        """Build report and annotation graphs directly from typed records."""
        original_canonical = report_storage.canonical_bytes

        def guarded_canonical(value: object) -> bytes:
            if (
                type(value) is dict
                and {"sections", "document", "support_provenance"}
                <= value.keys()
                and type(value["document"]) is dict
                and value["document"].get("version")
                == "diffstory.document-result.v2"
            ):
                message = "graph serialized a complete reader"
                raise AssertionError(message)
            return original_canonical(value)

        report_generation = self._typed_model(
            GenerationManifest, self.expanded_report["generation"]
        )
        presentation = self._typed_model(
            V2Presentation, self.applied["narrative_v2"]
        )
        base_report = {
            key: value
            for key, value in self.applied.items()
            if key not in {"generation", "narrative_v2"}
        }
        base_report["custom_compiler_metadata"] = {
            "literal": {
                "$ref": "not-a-storage-address",
                "kind": "detail",
            }
        }
        report_input = {
            **self.applied,
            "custom_compiler_metadata": base_report[
                "custom_compiler_metadata"
            ],
        }
        expected_report = {
            **self.expanded_report,
            "custom_compiler_metadata": base_report[
                "custom_compiler_metadata"
            ],
        }
        with (
            patch(
                "diffstory.report_storage.encode_report_storage",
                side_effect=AssertionError("graph used the expanded encoder"),
            ),
            patch(
                "diffstory.report_storage.parse_report_storage",
                side_effect=AssertionError("graph reparsed storage bytes"),
            ),
            patch(
                "diffstory.report_storage.expand_report_storage",
                side_effect=AssertionError("graph expanded the report"),
            ),
            patch(
                "diffstory.report_storage.canonical_bytes",
                side_effect=guarded_canonical,
            ),
            patch.object(
                PresentationSection,
                "model_dump",
                side_effect=AssertionError(
                    "graph dumped a full reader section"
                ),
            ),
            patch(
                "diffstory.report_storage._model_snapshot",
                side_effect=AssertionError("graph used model serialization"),
            ),
        ):
            report_graph = ReportRecordGraph.from_typed(
                report_generation,
                ReportReaderProjection.from_presentation(presentation),
                artifact_kind="report",
                base_report=base_report,
            )

        self.assertEqual(
            report_graph.to_bytes(),
            encode_report_storage(report_input, "report"),
        )
        self.assertEqual(
            canonical_bytes(
                expand_report_storage(
                    report_graph, max_output_bytes=128 * 1024 * 1024
                )
            ),
            canonical_bytes(expected_report),
        )

        annotation_reader = self._typed_annotation_reader(self.annotations)
        metadata = {
            key: self.annotations[key]
            for key in ("schema", "base_sha", "head_sha")
        }
        compact_manifest = self._typed_model(
            GenerationManifest, self.annotations["generation"]
        )
        with (
            patch(
                "diffstory.report_storage.encode_report_storage",
                side_effect=AssertionError("graph used the expanded encoder"),
            ),
            patch(
                "diffstory.report_storage.parse_report_storage",
                side_effect=AssertionError("graph reparsed storage bytes"),
            ),
            patch(
                "diffstory.report_storage.expand_report_storage",
                side_effect=AssertionError("graph expanded annotations"),
            ),
            patch(
                "diffstory.report_storage.canonical_bytes",
                side_effect=guarded_canonical,
            ),
            patch.object(
                PresentationSection,
                "model_dump",
                side_effect=AssertionError("graph dumped an annotation step"),
            ),
            patch(
                "diffstory.report_storage._model_snapshot",
                side_effect=AssertionError("graph used model serialization"),
            ),
        ):
            compact_graph = ReportRecordGraph.from_typed(
                compact_manifest,
                annotation_reader,
                artifact_kind="annotations",
                metadata=metadata,
            )

        expanded_annotations_graph = ReportRecordGraph.from_typed(
            self._typed_model(
                GenerationManifest,
                self.expanded_annotations["generation"],
            ),
            annotation_reader,
            artifact_kind="annotations",
            metadata=metadata,
        )
        self.assertEqual(
            compact_graph.to_bytes(), expanded_annotations_graph.to_bytes()
        )
        self.assertEqual(
            compact_graph.to_bytes(),
            encode_report_storage(self.annotations, "annotations"),
        )
        self.assertEqual(
            canonical_bytes(
                expand_report_storage(
                    compact_graph, max_output_bytes=128 * 1024 * 1024
                )
            ),
            canonical_bytes(self.expanded_annotations),
        )

    def test_graph_references_share_records_and_select_one_section(
        self,
    ) -> None:
        """Use typed references for shared results and resolve only selected details."""
        graph = self._typed_report_graph()
        generation_details = graph.generation_result_refs["details"]
        reader_projection = graph.reader_projection
        reader_details = tuple(
            reference
            for section in reader_projection.sections
            for reference in section.details
        )
        shared = next(
            reference
            for reference in reader_details
            if reference in generation_details
        )
        matching_generation_ref = next(
            reference
            for reference in generation_details
            if reference == shared
        )
        self.assertIs(shared, matching_generation_ref)
        self.assertEqual(shared.storage_id, matching_generation_ref.storage_id)
        self.assertEqual(
            graph.result_record(shared),
            graph.result_record(matching_generation_ref),
        )

        selected = reader_projection.sections[1]
        requested_ids = {item.storage_id for item in selected.details}
        resolved: list[tuple[str, str]] = []
        original = ReportStorageView._materialize  # noqa: SLF001 # Track one-section indexed resolution.

        def tracked(
            view: ReportStorageView, table: str, identifier: str, memo: dict
        ) -> dict:
            resolved.append((table, identifier))
            return original(view, table, identifier, memo)

        with patch.object(ReportStorageView, "_materialize", new=tracked):
            selected_by_id = graph.reader_section(selected.group_id)
            details = graph.section_details(selected.group_id)
        self.assertEqual(selected_by_id, selected)
        resolved_result_ids = {
            identifier for table, identifier in resolved if table == "results"
        }
        self.assertEqual(resolved_result_ids, requested_ids)
        self.assertEqual(
            details,
            self.applied["narrative_v2"]["sections"][1]["details"],
        )

    def test_graph_owns_detached_primitive_records(self) -> None:
        """Caller models and detached accessor mutations cannot change the graph."""
        generation = self._typed_model(
            GenerationManifest, self.expanded_report["generation"]
        )
        presentation = self._typed_model(
            V2Presentation, self.applied["narrative_v2"]
        )
        base_report = {
            key: copy.deepcopy(value)
            for key, value in self.applied.items()
            if key not in {"generation", "narrative_v2"}
        }
        graph = ReportRecordGraph.from_typed(
            generation,
            ReportReaderProjection.from_presentation(presentation),
            artifact_kind="report",
            base_report=base_report,
        )
        before_bytes = graph.to_bytes()
        original_detail = generation.details[0]
        object.__setattr__(original_detail, "unit_key", "0" * 64)
        base_report["meta"]["title"] = "mutated caller dictionary"
        detached_base = graph.record(
            "readers",
            graph._reader_id(),  # noqa: SLF001 # Confirm owned address data.
        )
        self.assertNotIn(
            "mutated caller dictionary",
            canonical_bytes(detached_base).decode(),
        )
        self.assertEqual(graph.to_bytes(), before_bytes)

        reference = graph.generation_result_refs["details"][0]
        detached = graph.result_record(reference)
        detached["explanations"][0]["before"] = "mutated accessor"
        self.assertNotEqual(
            graph.result_record(reference)["explanations"][0]["before"],
            "mutated accessor",
        )
        stored = graph.record("results", reference.storage_id)
        stored["unit_key"] = "0" * 64
        self.assertNotEqual(
            graph.record("results", reference.storage_id)["unit_key"],
            "0" * 64,
        )
        self.assertEqual(graph.to_bytes(), before_bytes)

    def test_unique_record_validation_is_not_repeated_by_relationship_checks(
        self,
    ) -> None:
        """Validate every unique table record once and reuse typed models later."""
        encoded = encode_report_storage(self.applied, "report")
        envelope = _envelope(encoded)
        expected_record_calls = Counter(
            f"{table}:{entry['kind']}"
            for table, entries in envelope["tables"].items()
            for entry in entries
        )
        validation_calls: Counter[str] = Counter()
        model_calls: Counter[str] = Counter()
        original_record_validation = (
            report_storage._validate_typed_table_record  # noqa: SLF001
        )
        original_model = report_storage._model  # noqa: SLF001

        def tracked_record_validation(
            table: str, kind: str, logical: dict
        ) -> BaseModel | None:
            validation_calls[f"{table}:{kind}"] += 1
            return original_record_validation(table, kind, logical)

        def tracked_model(
            model: type[BaseModel], logical: object
        ) -> BaseModel:
            model_calls[model.__name__] += 1
            return original_model(model, logical)

        with (
            patch(
                "diffstory.report_storage._validate_typed_table_record",
                side_effect=tracked_record_validation,
            ),
            patch(
                "diffstory.report_storage._model", side_effect=tracked_model
            ),
        ):
            view = parse_report_storage(encoded)

        self.assertEqual(validation_calls, expected_record_calls)
        self.assertEqual(
            view.typed_validation_counts, dict(expected_record_calls)
        )
        model_by_table_kind = {
            "results": report_storage._RESULT_MODELS,  # noqa: SLF001
            "plans": {"story_plan": report_storage.StoryPlan},
            "evidence_refs": {
                "span_ref": report_storage.SpanRef,
                "source_slice_ref": report_storage.SourceSliceRef,
                "statement_ref": report_storage.StatementRef,
            },
            "resolved_citations": {
                "resolved_citation": report_storage.ResolvedCitation
            },
            "runs": {"run": report_storage.RunIdentity},
            "invocations": {"invocation": report_storage.ArtifactInvocation},
            "requests": {"request": report_storage.PreparedMetadata},
            "producers": {"producer": report_storage.ProducerDescriptor},
            "receipts": {"receipt": report_storage.ResultReceipt},
            "inherited_provenance": {
                "inherited_provenance": report_storage.InheritedProvenance
            },
        }
        expected_model_calls: Counter[str] = Counter()
        for table, kinds in model_by_table_kind.items():
            for kind, model in kinds.items():
                expected_model_calls[model.__name__] += expected_record_calls[
                    f"{table}:{kind}"
                ]
        self.assertEqual(
            Counter(
                {name: model_calls[name] for name in expected_model_calls}
            ),
            expected_model_calls,
        )

    def test_live_graph_runs_unique_table_validators_once(self) -> None:
        """Count executed typed validators for direct construction as well."""
        encoded = encode_report_storage(self.applied, "report")
        envelope = _envelope(encoded)
        expected_record_calls = Counter(
            f"{table}:{entry['kind']}"
            for table, entries in envelope["tables"].items()
            for entry in entries
        )
        validation_calls: Counter[str] = Counter()
        model_calls: Counter[str] = Counter()
        original_record_validation = (
            report_storage._validate_typed_table_record  # noqa: SLF001
        )
        original_model = report_storage._model  # noqa: SLF001

        def tracked_record_validation(
            table: str, kind: str, logical: dict
        ) -> BaseModel | None:
            validation_calls[f"{table}:{kind}"] += 1
            return original_record_validation(table, kind, logical)

        def tracked_model(
            model: type[BaseModel], logical: object
        ) -> BaseModel:
            model_calls[model.__name__] += 1
            return original_model(model, logical)

        with (
            patch(
                "diffstory.report_storage._validate_typed_table_record",
                side_effect=tracked_record_validation,
            ),
            patch(
                "diffstory.report_storage._model", side_effect=tracked_model
            ),
        ):
            graph = self._typed_report_graph()

        self.assertEqual(validation_calls, expected_record_calls)
        self.assertEqual(
            graph.typed_validation_counts, dict(expected_record_calls)
        )
        model_by_table_kind = {
            "results": report_storage._RESULT_MODELS,  # noqa: SLF001
            "plans": {"story_plan": report_storage.StoryPlan},
            "evidence_refs": {
                "span_ref": report_storage.SpanRef,
                "source_slice_ref": report_storage.SourceSliceRef,
                "statement_ref": report_storage.StatementRef,
            },
            "resolved_citations": {
                "resolved_citation": report_storage.ResolvedCitation
            },
            "runs": {"run": report_storage.RunIdentity},
            "invocations": {"invocation": report_storage.ArtifactInvocation},
            "requests": {"request": report_storage.PreparedMetadata},
            "producers": {"producer": report_storage.ProducerDescriptor},
            "receipts": {"receipt": report_storage.ResultReceipt},
            "inherited_provenance": {
                "inherited_provenance": report_storage.InheritedProvenance
            },
        }
        expected_model_calls: Counter[str] = Counter()
        for table, kinds in model_by_table_kind.items():
            for kind, model in kinds.items():
                expected_model_calls[model.__name__] += expected_record_calls[
                    f"{table}:{kind}"
                ]
        actual_table_model_calls = Counter(
            {name: model_calls[name] for name in expected_model_calls}
        )
        self.assertEqual(actual_table_model_calls, expected_model_calls)
        self.assertEqual(
            model_calls["SpanRef"],
            expected_record_calls["evidence_refs:span_ref"],
        )
        self.assertEqual(
            model_calls["StatementRef"],
            expected_record_calls["evidence_refs:statement_ref"],
        )

    def test_live_typed_capture_rejects_bypassed_scalar_and_relation_changes(
        self,
    ) -> None:
        """Read raw nested attributes before any serializer can coerce them."""

        def graph_for(
            manifest: GenerationManifest, presentation: V2Presentation
        ) -> ReportRecordGraph:
            base_report = {
                key: value
                for key, value in self.applied.items()
                if key not in {"generation", "narrative_v2"}
            }
            return ReportRecordGraph.from_typed(
                manifest,
                ReportReaderProjection.from_presentation(presentation),
                artifact_kind="report",
                base_report=base_report,
            )

        manifest = self._typed_model(
            GenerationManifest, self.expanded_report["generation"]
        )
        presentation = self._typed_model(
            V2Presentation, self.applied["narrative_v2"]
        )
        object.__setattr__(presentation.citations[0], "byte_start", True)
        with self.assertRaisesRegex(ValueError, "byte_start"):
            graph_for(manifest, presentation)

        manifest = self._typed_model(
            GenerationManifest, self.expanded_report["generation"]
        )
        presentation = self._typed_model(
            V2Presentation, self.applied["narrative_v2"]
        )
        object.__setattr__(presentation.citations[0], "unexpected", "value")
        with self.assertRaisesRegex(
            ValueError, "invalid_typed_storage_record"
        ):
            graph_for(manifest, presentation)

        manifest = self._typed_model(
            GenerationManifest, self.expanded_report["generation"]
        )
        presentation = self._typed_model(
            V2Presentation, self.applied["narrative_v2"]
        )
        object.__setattr__(manifest.receipts[0], "result_digest", "0" * 64)
        with self.assertRaisesRegex(ValueError, "receipt_result_mismatch"):
            graph_for(manifest, presentation)

        manifest = self._typed_model(
            GenerationManifest, self.expanded_report["generation"]
        )
        presentation = self._typed_model(
            V2Presentation, self.applied["narrative_v2"]
        )
        sent_receipt = next(
            receipt
            for receipt in manifest.receipts
            if receipt.disposition == "sent"
        )
        producer = next(
            item
            for item in manifest.producers
            if item.key == sent_receipt.producer_key
        )
        prior_key = producer.key
        object.__setattr__(
            producer.original_run,
            "report_signature",
            "changed-report-signature",
        )
        changed_key = producer.key
        for receipt in manifest.receipts:
            if receipt.producer_key == prior_key:
                object.__setattr__(receipt, "producer_key", changed_key)
        with self.assertRaisesRegex(ValueError, "sent_run_mismatch"):
            graph_for(manifest, presentation)

        manifest = self._typed_model(
            GenerationManifest, self.expanded_report["generation"]
        )
        presentation = self._typed_model(
            V2Presentation, self.applied["narrative_v2"]
        )
        sent_requests = {
            (receipt.producer_key, receipt.request_key)
            for receipt in manifest.receipts
            if receipt.disposition == "sent"
        }
        self.assertGreater(len(sent_requests), 0)
        object.__setattr__(
            manifest.current_usage, "calls", len(sent_requests) - 1
        )
        with self.assertRaisesRegex(ValueError, "sent_usage_mismatch"):
            graph_for(manifest, presentation)

    def test_live_typed_capture_checks_reader_metadata_and_order(self) -> None:
        """Keep the indexed reader skeleton bound to its ordered generation."""

        def graph_for(presentation: V2Presentation) -> ReportRecordGraph:
            manifest = self._typed_model(
                GenerationManifest, self.expanded_report["generation"]
            )
            base_report = {
                key: value
                for key, value in self.applied.items()
                if key not in {"generation", "narrative_v2"}
            }
            return ReportRecordGraph.from_typed(
                manifest,
                ReportReaderProjection.from_presentation(presentation),
                artifact_kind="report",
                base_report=base_report,
            )

        presentation = self._typed_model(
            V2Presentation, self.applied["narrative_v2"]
        )
        object.__setattr__(presentation, "provenance", "verified")
        with self.assertRaisesRegex(ValueError, "invalid_reader_metadata"):
            graph_for(presentation)

        presentation = self._typed_model(
            V2Presentation, self.applied["narrative_v2"]
        )
        self.assertGreaterEqual(len(presentation.sections), 2)
        object.__setattr__(
            presentation, "sections", tuple(reversed(presentation.sections))
        )
        with self.assertRaisesRegex(ValueError, "reader_section_mismatch"):
            graph_for(presentation)

    def test_typed_constructor_defaults_match_canonical_models(self) -> None:
        """Include default schema versions omitted by fresh model constructors."""
        manifest = self._typed_model(
            GenerationManifest, self.expanded_report["generation"]
        )
        source_detail = manifest.details[0]
        fresh_detail = DetailResult(
            **{
                name: getattr(source_detail, name)
                for name in type(source_detail).model_fields
                if name != "version"
            }
        )
        source_section = manifest.sections[0]
        fresh_section = type(source_section)(
            **{
                name: getattr(source_section, name)
                for name in type(source_section).model_fields
                if name != "version"
            }
        )
        manifest_fields = {
            name: getattr(manifest, name)
            for name in GenerationManifest.model_fields
            if name not in {"version", "provenance"}
        }
        manifest_fields["details"] = (fresh_detail, *manifest.details[1:])
        manifest_fields["sections"] = (fresh_section, *manifest.sections[1:])
        fresh_manifest = GenerationManifest(**manifest_fields)
        self.assertNotIn("version", fresh_manifest.model_fields_set)
        self.assertNotIn("provenance", fresh_manifest.model_fields_set)
        self.assertNotIn("version", fresh_detail.model_fields_set)
        self.assertNotIn("version", fresh_section.model_fields_set)

        presentation = self._typed_model(
            V2Presentation, self.applied["narrative_v2"]
        )
        source_reader_section = presentation.sections[0]
        fresh_reader_section = type(source_reader_section)(
            group_id=source_reader_section.group_id,
            group_key=source_reader_section.group_key,
            number=source_reader_section.number,
            next_id=source_reader_section.next_id,
            prose=fresh_section,
            details=source_reader_section.details,
        )
        fresh_presentation = V2Presentation(
            version=presentation.version,
            provenance=presentation.provenance,
            document=presentation.document,
            sections=(fresh_reader_section, *presentation.sections[1:]),
            context_details=presentation.context_details,
            citations=presentation.citations,
            support_provenance=presentation.support_provenance,
        )
        base_report = {
            key: value
            for key, value in self.applied.items()
            if key not in {"generation", "narrative_v2"}
        }
        graph = ReportRecordGraph.from_typed(
            fresh_manifest,
            ReportReaderProjection.from_presentation(fresh_presentation),
            artifact_kind="report",
            base_report=base_report,
        )
        expanded = expand_report_storage(
            parse_report_storage(graph.to_bytes()),
            max_output_bytes=128 * 1024 * 1024,
        )
        self.assertEqual(
            canonical_bytes(expanded["generation"]),
            canonical_bytes(fresh_manifest),
        )
        self.assertEqual(
            canonical_bytes(expanded), canonical_bytes(self.expanded_report)
        )

        missing_defaults = copy.deepcopy(self.expanded_report["generation"])
        missing_defaults.pop("version")
        missing_defaults.pop("provenance")
        parsed_manifest = GenerationManifest.model_validate_json(
            canonical_bytes(missing_defaults)
        )
        self.assertNotIn("version", parsed_manifest.model_fields_set)
        parsed_graph = ReportRecordGraph.from_typed(
            parsed_manifest,
            ReportReaderProjection.from_presentation(presentation),
            artifact_kind="report",
            base_report=base_report,
        )
        parsed_expanded = expand_report_storage(
            parse_report_storage(parsed_graph.to_bytes()),
            max_output_bytes=128 * 1024 * 1024,
        )
        self.assertEqual(
            canonical_bytes(parsed_expanded["generation"]),
            canonical_bytes(parsed_manifest),
        )

    def test_transformed_citations_validate_closed_logical_records(
        self,
    ) -> None:
        """Reject coherent citation mutations after normalized reconstruction."""
        cases = (
            (
                "unknown field",
                lambda logical: logical.update({"unknown_extra": True}),
                lambda stored: stored.update({"unknown_extra": True}),
                "invalid_normalized_resolved_citation",
            ),
            (
                "boolean coordinate",
                lambda logical: logical.update({"byte_start": True}),
                lambda stored: stored.update({"byte_start": True}),
                "byte_start",
            ),
        )
        for name, change_logical, change_stored, error in cases:
            with self.subTest(name=name):
                envelope = self._coherent_reader_record_change(
                    table="resolved_citations",
                    reader_field="citations",
                    logical_field="citations",
                    kind="resolved_citation",
                    change_logical=change_logical,
                    change_stored=change_stored,
                )
                with self.assertRaisesRegex(ValueError, error):
                    parse_report_storage(_raw(envelope))

        for field, value in (
            ("path", "shadowed/path"),
            ("side", "base"),
            ("revision", "shadowed-revision"),
            ("fragment_key", "0" * 64),
        ):
            with self.subTest(shadowed_field=field):
                envelope = self._coherent_reader_record_change(
                    table="resolved_citations",
                    reader_field="citations",
                    logical_field="citations",
                    kind="resolved_citation",
                    change_logical=lambda _logical: None,
                    change_stored=lambda stored, field=field, value=value: (
                        stored.update({field: value})
                    ),
                )
                with self.assertRaisesRegex(
                    ValueError, "invalid_normalized_resolved_citation"
                ):
                    parse_report_storage(_raw(envelope))

    def test_transformed_inherited_provenance_validates_closed_records(
        self,
    ) -> None:
        """Reject coherent unknown and shadowed fields on inherited records."""
        unknown = self._coherent_reader_record_change(
            table="inherited_provenance",
            reader_field="support_provenance",
            logical_field="support_provenance",
            kind="inherited_provenance",
            change_logical=lambda logical: logical.update(
                {"unknown_extra": True}
            ),
            change_stored=lambda stored: stored.update(
                {"unknown_extra": True}
            ),
        )
        with self.assertRaisesRegex(
            ValueError, "invalid_normalized_inherited_provenance"
        ):
            parse_report_storage(_raw(unknown))

        provenance = self.applied["narrative_v2"]["support_provenance"][0]
        copied_statement = self._coherent_reader_record_change(
            table="inherited_provenance",
            reader_field="support_provenance",
            logical_field="support_provenance",
            kind="inherited_provenance",
            change_logical=lambda _logical: None,
            change_stored=lambda stored: stored.update(
                {"statement": copy.deepcopy(provenance["statement"])}
            ),
        )
        with self.assertRaisesRegex(
            ValueError, "invalid_normalized_inherited_provenance"
        ):
            parse_report_storage(_raw(copied_statement))

    def test_literal_ref_shaped_compiler_metadata_stays_literal(self) -> None:
        """Keep compiler metadata that resembles a pointer as literal data."""
        report = copy.deepcopy(self.applied)
        report["custom_compiler_metadata"] = {
            "literal": {"$ref": "not-a-storage-address", "kind": "detail"}
        }
        expected = {
            **self.expanded_report,
            "custom_compiler_metadata": report["custom_compiler_metadata"],
        }
        encoded, expanded = self.assert_round_trip(
            report, "report", expected=expected
        )
        self.assertEqual(
            expanded["custom_compiler_metadata"],
            report["custom_compiler_metadata"],
        )
        self.assertGreater(len(encoded), 0)

    def test_explicit_null_urls_and_unicode_source_evidence_are_exact(
        self,
    ) -> None:
        """Retain explicit null URLs and exact citation identities."""
        annotations = copy.deepcopy(self.annotations)
        for citation in annotations["citations"]:
            citation["url"] = None
        encoded = encode_report_storage(annotations, "annotations")
        view = parse_report_storage(encoded)
        self.assertTrue(view.citations())
        self.assertTrue(
            all(citation["url"] is None for citation in view.citations())
        )
        expanded = expand_report_storage(
            view, max_output_bytes=128 * 1024 * 1024
        )
        self.assertTrue(
            all(citation["url"] is None for citation in expanded["citations"])
        )
        self.assertEqual(
            [item["key"] for item in expanded["citations"]],
            [item["key"] for item in annotations["citations"]],
        )

    def test_unicode_mixed_line_source_slices_and_inherited_paths_roundtrip(
        self,
    ) -> None:
        """Round-trip Unicode source slices and statement-path provenance."""
        from diffstory.narrative_plan import RequestPackingLimits
        from diffstory.narrative_v2 import V2Options

        text = ('é😀\r\nCR\rLF\nLS\u2028PS\u2029"\\  ' * 700) + "terminal  "
        report = golden_report(1)
        report["meta"]["title"] = "Authored title 😀 " * 2200
        report["meta"]["repository"] = "example/project/" * 1800
        report["meta"]["description"] = "Author motivation\r\n" * 2200
        report["groups"][0]["theme"] = "Compiler heading " * 2500

        def expanded_fixture(
            _provider: FixtureProvider, data: dict, response: dict
        ) -> dict:
            if data["stage"] == "detail":
                item = response["items"][0]
                item["explanations"][0]["before"] = text
                item["observations"][0]["text"] = (
                    "Observation β\r\n\u2028  " * 700
                )
                item["observations"].extend(
                    {
                        "text": f"Additional observation {index}.",
                        "citations": list(
                            item["observations"][0]["citations"]
                        ),
                    }
                    for index in range(32)
                )
            elif data["stage"] == "planning":
                for output, item in zip(
                    response["items"], data["items"], strict=True
                ):
                    if item["support"]["author"] and not any(
                        item["support"][field]
                        for field in ("claims", "citations", "basis")
                    ):
                        output["abstraction"][0]["kind"] = "author_motivation"
                        output["abstraction"][0]["text"] = (
                            "The author reports a motivation for the change."
                        )
            return response

        options = V2Options(
            card_wire_bytes=1024,
            card_support_bytes=2048,
            prose_wire_bytes=4096,
            packing=RequestPackingLimits(stable_partition_count=1),
        )
        provider = FixtureProvider(expanded_fixture)
        provider.context_tokens = 40000
        provider.max_output_tokens = 15000
        with tempfile.TemporaryDirectory() as directory:
            annotations = Narrator(
                provider,
                architecture="v2",
                concurrency=1,
                v2_options=options,
                recovery_dir=directory,
            ).generate(report)
        slices = [
            item
            for item in annotations["citations"]
            if item["consumption"] == "source_slice"
        ]
        self.assertTrue(slices)
        self.assertTrue(annotations["support_provenance"])
        encoded = encode_report_storage(annotations, "annotations")
        expanded = expand_report_storage(
            parse_report_storage(encoded), max_output_bytes=128 * 1024 * 1024
        )
        expected = {
            **annotations,
            "generation": expanded_generation(annotations["generation"]),
        }
        self.assertEqual(canonical_bytes(expanded), canonical_bytes(expected))
        self.assertEqual(
            [
                item
                for item in expanded["citations"]
                if item["consumption"] == "source_slice"
            ],
            slices,
        )

    def test_assigned_and_unassigned_context_empty_acknowledgements_roundtrip(
        self,
    ) -> None:
        """Retain context assignments and empty source acknowledgements."""
        from test_narrative_boundaries import generate
        from test_narrative_boundaries import retained_context_snapshot

        from diffstory.analysis import compile_snapshot
        from diffstory.narrative_manifest import LedgerParameters
        from diffstory.narrative_plan import build_evidence_ledger
        from diffstory.narrative_plan import build_ledger_index
        from diffstory.narrative_v2 import V2Options

        report = compile_snapshot(retained_context_snapshot())
        options = V2Options(
            ledger=LedgerParameters(max_span_bytes=9, max_unit_spans=2)
        )
        candidate = generate(
            report, options=options, provider=FixtureProvider()
        )
        applied = apply_annotations(report, candidate)
        ledger = build_evidence_ledger(report, **options.ledger.model_dump())
        index = build_ledger_index(ledger)
        assigned = {
            unit for group in index.groups.values() for unit in group.unit_keys
        }
        context_keys = {
            item["unit_key"] for item in candidate["context_details"]
        }
        self.assertTrue(context_keys)
        self.assertEqual(
            context_keys,
            {
                item["unit_key"]
                for item in candidate["generation"]["details"]
                if item["unit_key"] not in assigned
            },
        )
        self.assertTrue(
            any(
                not detail["explanations"]
                and not detail["observations"]
                and not detail["questions"]
                for section in candidate["steps"]
                for detail in section["details"]
            )
        )
        empty_links = [
            item
            for item in candidate["citations"]
            if item["path"] == "empty.txt"
        ]
        self.assertEqual(len(empty_links), 2)
        encoded = encode_report_storage(applied, "report")
        expanded = expand_report_storage(
            parse_report_storage(encoded), max_output_bytes=128 * 1024 * 1024
        )
        expected = {
            **applied,
            "generation": expanded_generation(applied["generation"]),
        }
        self.assertEqual(canonical_bytes(expanded), canonical_bytes(expected))

    def test_expansion_cap_is_checked_before_full_output(self) -> None:
        """Reject amplified output before building the complete expanded report."""
        view = parse_report_storage(
            encode_report_storage(self.applied, "report")
        )
        with self.assertRaisesRegex(ValueError, "expanded_report_too_large"):
            expand_report_storage(view, max_output_bytes=1)
        expanded = expand_report_storage(
            view, max_output_bytes=128 * 1024 * 1024
        )
        self.assertEqual(
            canonical_bytes(expanded), canonical_bytes(self.expanded_report)
        )

    def test_coherent_unsupported_claim_and_request_metadata_need_current_proof(
        self,
    ) -> None:
        """Keep coherent forged claims and request metadata subject to current proof."""
        manifest = GenerationManifest.model_validate_json(
            canonical_bytes(self.expanded_annotations["generation"])
        )
        original = manifest.sections[0]
        unsupported = changed_record(
            original.synthesis[0],
            claims=(
                ClaimRef(
                    unit_key=manifest.details[0].unit_key,
                    observation_id="0" * 64,
                ),
            ),
        )
        section = changed_record(original, synthesis=(unsupported,))
        changed = changed_record(
            manifest,
            sections=(section, *manifest.sections[1:]),
            receipts=tuple(
                changed_record(receipt, result_digest=result_digest(section))
                if receipt.item_key == section.item_key
                else receipt
                for receipt in manifest.receipts
            ),
        )
        candidate = copy.deepcopy(self.expanded_annotations)
        candidate["generation"] = changed.model_dump(mode="json")
        candidate["steps"][0]["prose"] = section.model_dump(mode="json")
        encoded = encode_report_storage(candidate, "annotations")
        structurally_valid = parse_report_storage(encoded)
        expanded = expand_report_storage(
            structurally_valid, max_output_bytes=128 * 1024 * 1024
        )
        self.assertEqual(
            canonical_bytes(expanded["generation"]),
            canonical_bytes(changed.model_dump(mode="json")),
        )
        with self.assertRaisesRegex(ArtifactRuntimeError, "citation"):
            validate_generation(
                self.report,
                GenerationManifest.model_validate_json(
                    canonical_bytes(expanded["generation"])
                ),
                candidate=expanded,
            )

        old = manifest.producers[0]
        first = old.requests[0]
        producer = changed_record(
            old,
            requests=(
                changed_record(
                    first,
                    batch=changed_record(
                        first.batch, input_bound=first.batch.input_bound + 1
                    ),
                ),
                *old.requests[1:],
            ),
        )
        request_altered = changed_record(
            manifest,
            producers=(producer, *manifest.producers[1:]),
            receipts=tuple(
                changed_record(receipt, producer_key=producer.key)
                if receipt.producer_key == old.key
                else receipt
                for receipt in manifest.receipts
            ),
        )
        candidate = copy.deepcopy(self.expanded_annotations)
        candidate["generation"] = request_altered.model_dump(mode="json")
        request_view = parse_report_storage(
            encode_report_storage(candidate, "annotations")
        )
        request_expanded = expand_report_storage(
            request_view, max_output_bytes=128 * 1024 * 1024
        )
        with self.assertRaisesRegex(ArtifactRuntimeError, "request_mismatch"):
            validate_generation(
                self.report,
                GenerationManifest.model_validate_json(
                    canonical_bytes(request_expanded["generation"])
                ),
                candidate=request_expanded,
            )

    def test_duplicate_keys_versions_fields_ids_and_reference_failures(  # noqa: PLR0915 # Keep focused importer failures together.
        self,
    ) -> None:
        """Reject malformed versions, fields, references, and table closure."""
        encoded = encode_report_storage(self.applied, "report")
        with self.assertRaisesRegex(ValueError, "duplicate_json_key"):
            parse_report_storage(
                b'{"storage_version":"a","storage_version":"b"}'
            )

        base = _envelope(encoded)
        malformed = copy.deepcopy(base)
        malformed["storage_version"] = "diffstory.report-storage.v2"
        with self.assertRaisesRegex(ValueError, "unknown_storage_version"):
            parse_report_storage(_raw(malformed))

        malformed_generation = copy.deepcopy(self.expanded_annotations)
        malformed_generation["generation"]["storage_version"] = (
            "diffstory.generation-storage.v2"
        )
        with self.assertRaisesRegex(
            ValueError, "invalid_generation_storage_version"
        ):
            encode_report_storage(malformed_generation, "annotations")

        malformed = copy.deepcopy(base)
        malformed["unknown"] = 1
        with self.assertRaisesRegex(
            ValueError, "unknown_storage_envelope_field"
        ):
            parse_report_storage(_raw(malformed))

        malformed = copy.deepcopy(base)
        malformed["tables"]["results"].append(
            copy.deepcopy(malformed["tables"]["results"][0])
        )
        with self.assertRaisesRegex(ValueError, "duplicate_storage_address"):
            parse_report_storage(_raw(malformed))

        malformed = copy.deepcopy(base)
        malformed["tables"]["results"].reverse()
        with self.assertRaisesRegex(
            ValueError, "noncanonical_storage_table_order"
        ):
            parse_report_storage(_raw(malformed))

        malformed = copy.deepcopy(base)
        malformed["root"]["reader"]["$ref"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "dangling_storage_reference"):
            parse_report_storage(_raw(malformed))

        malformed = copy.deepcopy(base)
        malformed["root"]["reader"]["kind"] = "detail"
        with self.assertRaisesRegex(
            ValueError, "wrong_kind_storage_reference"
        ):
            parse_report_storage(_raw(malformed))

        malformed = copy.deepcopy(base)
        reader_id = malformed["root"]["reader"]["$ref"]
        reader = next(
            entry["record"]
            for entry in malformed["tables"]["readers"]
            if entry["id"] == reader_id
        )
        if reader["support_provenance"]:
            provenance_id = reader["support_provenance"][0]["$ref"]
            provenance = next(
                entry["record"]
                for entry in malformed["tables"]["inherited_provenance"]
                if entry["id"] == provenance_id
            )
            provenance["result"] = {
                "$ref": provenance_id,
                "kind": "inherited_provenance",
            }
            with self.assertRaisesRegex(
                ValueError, "cyclic_storage_reference"
            ):
                parse_report_storage(_raw(malformed))
        else:
            self.fail(
                "synthetic fixture did not exercise inherited provenance"
            )

        malformed = copy.deepcopy(base)
        orphan = {"value": "not referenced"}
        orphan_id = storage_address("url", orphan)
        malformed["tables"]["urls"].append(
            {"id": orphan_id, "kind": "url", "record": orphan}
        )
        malformed["tables"]["urls"].sort(key=lambda item: item["id"])
        with self.assertRaisesRegex(ValueError, "unused_storage_record"):
            parse_report_storage(_raw(malformed))

        malformed = copy.deepcopy(base)
        url = malformed["tables"]["urls"][0]
        url["record"]["unexpected"] = False
        with self.assertRaisesRegex(ValueError, "invalid_url_record"):
            parse_report_storage(_raw(malformed))

        malformed = copy.deepcopy(base)
        url = malformed["tables"]["urls"][0]
        url["record"]["value"] = "tampered"
        with self.assertRaisesRegex(ValueError, "storage_address_mismatch"):
            parse_report_storage(_raw(malformed))

    def test_duplicate_raw_table_id_is_rejected_before_canonicality_checks(
        self,
    ) -> None:
        """Reject repeated raw table IDs before canonical ordering checks."""
        encoded = encode_report_storage(self.applied, "report")
        malformed = _envelope(encoded)
        item = malformed["tables"]["runs"][0]
        malformed["tables"]["runs"].append(copy.deepcopy(item))
        with self.assertRaisesRegex(ValueError, "duplicate_storage_address"):
            parse_report_storage(_raw(malformed))

    def test_long_shallow_wrong_kind_chain_has_controlled_failure(
        self,
    ) -> None:
        """Bound graph validation for long chains in shallow JSON input."""
        envelope = _envelope(encode_report_storage(self.applied, "report"))
        existing_ids = {
            entry["id"]
            for entries in envelope["tables"].values()
            for entry in entries
        }
        chain_ids = [f"{index:064x}" for index in range(1, 1801)]
        self.assertTrue(existing_ids.isdisjoint(chain_ids))
        span_id = _entry(envelope, "evidence_refs", "span_ref")["id"]
        chain = []
        for index, identifier in enumerate(chain_ids):
            if index + 1 < len(chain_ids):
                target_id = chain_ids[index + 1]
                target_kind = "source_slice_ref"
            else:
                target_id = span_id
                target_kind = "span_ref"
            chain.append(
                {
                    "id": identifier,
                    "kind": "source_slice_ref",
                    "record": {
                        "kind": "source_slice",
                        "source": {"$ref": target_id, "kind": target_kind},
                        "byte_start": 0,
                        "byte_end": 1,
                        "content_digest": "0" * 64,
                    },
                }
            )
        envelope["tables"]["evidence_refs"].extend(chain)
        envelope["tables"]["evidence_refs"].sort(key=lambda item: item["id"])
        with self.assertRaisesRegex(
            ValueError, "wrong_kind_storage_reference"
        ):
            parse_report_storage(_raw(envelope))

    def test_storage_address_reuse_compares_complete_record_content(
        self,
    ) -> None:
        """Reject a mocked address collision when complete records differ."""
        from diffstory.report_storage import _Builder

        builder = _Builder()
        with patch(
            "diffstory.report_storage.storage_address", return_value="0" * 64
        ):
            first = builder.put(
                "urls", "url", {"value": "https://one.invalid"}
            )
            same = builder.put("urls", "url", {"value": "https://one.invalid"})
            self.assertEqual(first, same)
            with self.assertRaisesRegex(
                ValueError, "storage_address_collision"
            ):
                builder.put("urls", "url", {"value": "https://two.invalid"})


if __name__ == "__main__":
    unittest.main()
