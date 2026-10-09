"""Offline public v2 proof, exact presentation and legacy safety boundaries."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from test_analysis import snap
from test_narrative_runtime import changed_record
from test_narrative_v2 import FixtureProvider
from test_narrative_v2 import golden_report

from diffstory.analysis import apply_annotations
from diffstory.analysis import compile_snapshot
from diffstory.analysis import validate_generated_report
from diffstory.analysis import validate_generation
from diffstory.narrative import Narrator
from diffstory.narrative_contracts import ClaimRef
from diffstory.narrative_contracts import StatementPath
from diffstory.narrative_contracts import SupportedStatement
from diffstory.narrative_hierarchy import PlanningSources
from diffstory.narrative_manifest import GenerationManifest
from diffstory.narrative_manifest import LedgerParameters
from diffstory.narrative_manifest import compiler_fact_signature
from diffstory.narrative_manifest import result_digest
from diffstory.narrative_manifest import (
    validate_generation as validate_v2_generation,
)
from diffstory.narrative_plan import RequestPackingLimits
from diffstory.narrative_plan import build_ledger_index
from diffstory.narrative_plan import canonical_bytes
from diffstory.narrative_plan import content_digest
from diffstory.narrative_runtime import ArtifactRuntimeError
from diffstory.narrative_v2 import V2Options
from diffstory.render import render
from diffstory.render import validate_report


def generate(
    report: dict,
    *,
    options: V2Options | None = None,
    provider: FixtureProvider | None = None,
) -> dict:
    """Generate through the public opt-in gate with a local authored provider."""
    with tempfile.TemporaryDirectory() as directory:
        return Narrator(
            provider or FixtureProvider(),
            architecture="v2",
            recovery_dir=directory,
            concurrency=1,
            v2_options=options,
        ).generate(report)


def replace_result(
    manifest: GenerationManifest, family: str, position: int, result: object
) -> GenerationManifest:
    """Rehash a complete result receipt so a forgery passes structural checks."""
    values = list(getattr(manifest, family))
    values[position] = result
    return changed_record(
        manifest,
        **{family: tuple(values)},
        receipts=tuple(
            changed_record(r, result_digest=result_digest(result))
            if r.item_key == result.item_key
            else r
            for r in manifest.receipts
        ),
    )


class PublicBoundaryTests(unittest.TestCase):
    """Use current compiled facts and real original production audits."""

    @classmethod
    def setUpClass(cls) -> None:
        """Share immutable JSON fixture bytes, never a trusted validation context."""
        cls.original = golden_report()
        cls.candidate = generate(cls.original)
        cls.applied = apply_annotations(cls.original, cls.candidate)

    def assert_rejected(self, manifest: GenerationManifest) -> None:
        """Exercise both persisted manifest and report production boundaries."""
        with self.assertRaises((ValueError, ArtifactRuntimeError)):
            validate_generation(
                self.original, manifest.model_dump(mode="json")
            )
        applied = copy.deepcopy(self.applied)
        applied["generation"] = manifest.model_dump(mode="json")
        with self.assertRaises((ValueError, ArtifactRuntimeError)):
            validate_report(applied)

    def test_offline_public_golden_save_reload_and_render(self) -> None:
        """No public saved boundary needs credentials, serialization or transport."""
        blocked = (
            "test_narrative_v2.FixtureProvider.require_credentials",
            "test_narrative_v2.FixtureProvider.prepare",
            "test_narrative_v2.FixtureProvider.complete_reply",
            "diffstory.narrative.CodexCLIProvider.require_credentials",
            "diffstory.narrative.CodexCLIProvider.prepare",
            "diffstory.narrative.CodexCLIProvider.complete_reply",
            "diffstory.narrative.OpenAIResponsesProvider.__init__",
            "subprocess.Popen",
            "subprocess.run",
            "shutil.which",
            "socket.socket",
            "urllib.request.urlopen",
            "pathlib.Path.read_bytes",
        )
        with ExitStack() as stack:
            for target in blocked:
                stack.enter_context(
                    patch(target, side_effect=AssertionError(target))
                )
            validate_generation(self.original, self.candidate["generation"])
            applied = apply_annotations(self.original, self.candidate)
            with tempfile.TemporaryDirectory() as directory:
                saved = Path(directory) / "report.json"
                saved.write_text(json.dumps(applied), encoding="utf-8")
                loaded = json.loads(saved.read_text(encoding="utf-8"))
            validate_generated_report(loaded)
            validate_report(loaded)
            html = render(loaded)
        self.assertIn("diffstory.presentation.v2", html)
        typed = loaded["narrative_v2"]
        self.assertEqual(typed["document"], self.candidate["document"])
        self.assertEqual(typed["sections"], self.candidate["steps"])
        self.assertEqual(
            typed["context_details"], self.candidate["context_details"]
        )
        all_details = [
            d for section in typed["sections"] for d in section["details"]
        ] + typed["context_details"]
        self.assertEqual(
            {d["unit_key"]: d for d in all_details},
            {d["unit_key"]: d for d in loaded["generation"]["details"]},
        )
        self.assertEqual(
            compiler_fact_signature(loaded),
            compiler_fact_signature(self.original),
        )
        self.assertTrue(typed["support_provenance"])
        for proof in typed["support_provenance"]:
            self.assertEqual(proof["consumption"], "inherited_statement")
            self.assertEqual(
                proof["receipt"]["item_key"],
                proof["reference"]["producer_item_key"],
            )
            producer = next(
                p
                for p in loaded["generation"]["planning"]
                if p["item_key"] == proof["reference"]["producer_item_key"]
            )
            self.assertEqual(
                content_digest(canonical_bytes(producer)),
                proof["reference"]["result_digest"],
            )

    def test_unknown_missing_and_mismatched_schemas_fail(self) -> None:
        """Never promote old v2 declarations or bypass v1 identifier validation."""
        for change in (
            lambda c: c.update(schema="diffstory.annotations.v9"),
            lambda c: c.update(schema="diffstory.annotations.v1"),
            lambda c: c["generation"].pop("request_configuration"),
            lambda c: c["generation"].pop("version"),
            lambda c: c["generation"].update(
                version="diffstory.generation.v3"
            ),
            lambda c: c["generation"]["request_configuration"].update(
                protocol="opaque"
            ),
            lambda c: c["generation"]["request_configuration"].update(
                version="unknown"
            ),
        ):
            altered = copy.deepcopy(self.candidate)
            change(altered)
            with self.assertRaises((ValueError, ArtifactRuntimeError)):
                apply_annotations(self.original, altered)
        legacy = copy.deepcopy(self.original)
        legacy["narrative_v2"] = self.applied["narrative_v2"]
        with self.assertRaises(ValueError):
            render(legacy)
        legacy = copy.deepcopy(self.original)
        legacy["generation"] = {
            "schema": "diffstory.generation.v1",
            "catalog": {"id": "f" * 64},
        }
        with self.assertRaises(ValueError):
            render(legacy)

    def test_all_presentation_fields_and_catalog_are_exact(self) -> None:
        """Valid generation cannot conceal altered applied prose, provenance or bytes."""
        mutations = (
            lambda r: r["document"].update(lead="different"),
            lambda r: r["groups"][0].update(title="different"),
            lambda r: r["groups"][0].update(number=True),
            lambda r: r["groups"][0].update(next_id=None),
            lambda r: r["groups"][0]["narrative"].update(
                provenance="verified"
            ),
            lambda r: r["groups"][0]["narrative"].update(questions=[]),
            lambda r: r["groups"].reverse(),
            lambda r: r["narrative_v2"]["document"]["lead"][0].update(
                text="different"
            ),
            lambda r: r["narrative_v2"]["sections"][0]["details"][0][
                "explanations"
            ][0].update(before="different"),
            lambda r: r["narrative_v2"]["citations"][0].update(byte_start=999),
            lambda r: r["narrative_v2"]["citations"][0].update(
                url="https://wrong.example"
            ),
            lambda r: r["narrative_v2"]["citations"][0].update(
                side="head"
                if r["narrative_v2"]["citations"][0]["side"] == "base"
                else "base"
            ),
            lambda r: r["narrative_v2"]["support_provenance"][0].update(
                consumption="direct_source"
            ),
            lambda r: r["narrative_v2"].update(extra={"id": "f" * 64}),
        )
        for index, mutate in enumerate(mutations):
            altered = copy.deepcopy(self.applied)
            mutate(altered)
            with (
                self.subTest(index=index),
                self.assertRaises((ValueError, ArtifactRuntimeError)),
            ):
                validate_report(altered)
            with (
                self.subTest(generation_boundary=index),
                self.assertRaises((ValueError, ArtifactRuntimeError)),
            ):
                validate_generation(altered, altered["generation"])
        candidate = copy.deepcopy(self.candidate)
        candidate["steps"][0]["details"][0]["questions"] = []
        with self.assertRaises((ValueError, ArtifactRuntimeError)):
            apply_annotations(self.original, candidate)

    def test_typed_ids_do_not_weaken_legacy_id_number_or_tag_rules(
        self,
    ) -> None:
        """Even a rehashed unrelated compiler field retains legacy renderer checks."""
        for field in ({"id": "f" * 64}, {"number": True}, {"tag": "invalid"}):
            report = copy.deepcopy(self.applied)
            report["extra_compiler_record"] = field
            report["generation"]["compiler_fact_signature"] = (
                compiler_fact_signature(report)
            )
            with (
                self.subTest(field=field),
                self.assertRaisesRegex(
                    ValueError, "Invalid report|Invalid diff"
                ),
            ):
                render(report)

    def test_current_source_metadata_and_compiler_facts_are_bound(
        self,
    ) -> None:
        """Classification, status, dependencies and source remain compiler facts."""
        for mutate in (
            lambda r: r["source_evidence"]["fragments"][0].update(
                text="changed"
            ),
            lambda r: r["source_evidence"]["fragments"][0].update(
                start_line=31
            ),
            lambda r: r["meta"].update(head_sha="c" * 40),
            lambda r: r["meta"].update(description="different author intent"),
            lambda r: r["stats"].update(test_runs=99),
            lambda r: r["changes"][0].update(kind="removed"),
            lambda r: r["changes"][0].update(
                basis="changed classification evidence"
            ),
            lambda r: r["groups"][0].update(
                prerequisites=[r["groups"][1]["id"]]
            ),
            lambda r: r.update(new_compiler_fact="preserved"),
        ):
            report = copy.deepcopy(self.original)
            mutate(report)
            with self.assertRaises((ValueError, ArtifactRuntimeError)):
                apply_annotations(report, self.candidate)

    def test_rehashed_statement_and_whole_source_forgeries_fail_offline(
        self,
    ) -> None:
        """Self-consistent result hashes cannot authorize unconsumed endpoints."""
        manifest = GenerationManifest.model_validate_json(
            json.dumps(self.candidate["generation"])
        )
        original = manifest.sections[0]
        statement = original.synthesis[0]
        ref = statement.basis[0]
        alterations = [
            changed_record(statement, basis=(changed_record(ref, **update),))
            for update in (
                {"producer_item_key": "0" * 64},
                {"result_digest": "0" * 64},
                {"statement_digest": "0" * 64},
                {
                    "statement_path": StatementPath(
                        field="abstraction", index=999
                    )
                },
            )
        ]
        alterations += [
            changed_record(
                statement,
                basis=(),
                claims=(),
                citations=(manifest.details[0].explanations[0].citations[0],),
            ),
            changed_record(
                statement,
                claims=(
                    ClaimRef(
                        unit_key=manifest.details[0].unit_key,
                        observation_id="0" * 64,
                    ),
                ),
            ),
        ]
        for altered in alterations:
            self.assert_rejected(
                replace_result(
                    manifest,
                    "sections",
                    0,
                    changed_record(original, synthesis=(altered,)),
                )
            )
        final = changed_record(
            manifest.sections[-1],
            transition=(
                SupportedStatement(
                    kind="editorial", text="An invented next section."
                ),
            ),
        )
        self.assert_rejected(
            replace_result(
                manifest, "sections", len(manifest.sections) - 1, final
            )
        )

    def test_rehashed_requests_peers_and_configuration_fail_offline(
        self,
    ) -> None:
        """Reconstruct original requests and their complete original peer dependencies."""
        manifest = GenerationManifest.model_validate_json(
            json.dumps(self.candidate["generation"])
        )
        old = manifest.producers[0]
        first = old.requests[0]
        for request in (
            changed_record(
                first,
                batch=changed_record(
                    first.batch, input_bound=first.batch.input_bound + 1
                ),
            ),
            changed_record(
                first,
                batch=changed_record(first.batch, request_digest="0" * 64),
            ),
        ):
            producer = changed_record(
                old, requests=(request, *old.requests[1:])
            )
            altered = changed_record(
                manifest,
                producers=(producer, *manifest.producers[1:]),
                receipts=tuple(
                    changed_record(r, producer_key=producer.key)
                    if r.producer_key == old.key
                    else r
                    for r in manifest.receipts
                ),
            )
            self.assert_rejected(altered)
        for update in (
            {"prefix_hex": "4142"},
            {
                "context_tokens": manifest.request_configuration.context_tokens
                + 1
            },
        ):
            self.assert_rejected(
                changed_record(
                    manifest,
                    request_configuration=changed_record(
                        manifest.request_configuration, **update
                    ),
                )
            )

    def test_units_obligations_source_ownership_and_peer_dependencies_fail(
        self,
    ) -> None:
        """Rehashed detail and co-batch declarations cannot replace current authority."""
        manifest = GenerationManifest.model_validate_json(
            json.dumps(self.candidate["generation"])
        )
        detail = manifest.details[0]
        for explanations in (
            detail.explanations[1:],
            (
                changed_record(
                    detail.explanations[0],
                    citations=(
                        changed_record(
                            detail.explanations[0].citations[0],
                            line_start=999,
                            line_end=999,
                        ),
                        *detail.explanations[0].citations[1:],
                    ),
                ),
                *detail.explanations[1:],
            ),
            (
                changed_record(
                    detail.explanations[0],
                    citations=(
                        manifest.details[1].explanations[0].citations[0],
                    ),
                ),
                *detail.explanations[1:],
            ),
        ):
            changed = changed_record(detail, explanations=explanations)
            altered = changed_record(
                manifest,
                details=(changed, *manifest.details[1:]),
                receipts=tuple(
                    changed_record(r, result_digest=result_digest(changed))
                    if r.item_key == detail.unit_key
                    else r
                    for r in manifest.receipts
                ),
            )
            self.assert_rejected(altered)
        old = manifest.producers[0]
        invocation = changed_record(
            old.invocation,
            dependencies=tuple(
                changed_record(d, digest="0" * 64)
                if d.kind in {"peer", "evidence"}
                else d
                for d in old.invocation.dependencies
            ),
        )
        producer = changed_record(old, invocation=invocation)
        artifact_keys = invocation.artifact_keys()
        altered = changed_record(
            manifest,
            producers=(producer, *manifest.producers[1:]),
            receipts=tuple(
                changed_record(
                    r,
                    producer_key=producer.key,
                    artifact_key=artifact_keys[r.item_key],
                )
                if r.producer_key == old.key
                else r
                for r in manifest.receipts
            ),
        )
        self.assert_rejected(altered)
        for change in (
            lambda m: m["unit_keys"].pop(),
            lambda m: m["plan"]["ordered_group_keys"].reverse(),
            lambda m: m["plan"]["production"]["section_views"][0].update(
                chapter_key="0" * 64
            ),
            lambda m: m["receipts"][0].update(artifact_key="0" * 64),
        ):
            value = copy.deepcopy(self.candidate["generation"])
            change(value)
            with self.assertRaises((ValueError, ArtifactRuntimeError)):
                validate_generation(self.original, value)

    def test_no_repeated_source_scan_per_citation_or_result(self) -> None:
        """One rebuilt index and one support forest serve the entire boundary."""
        from diffstory.narrative_hierarchy import validate_support_forest

        calls = []
        check = PlanningSources.check_current

        def count(sources: PlanningSources) -> None:
            calls.append(1)
            check(sources)

        with (
            patch(
                "diffstory.narrative_v2.build_ledger_index",
                wraps=build_ledger_index,
            ) as indexes,
            patch(
                "diffstory.narrative_v2.validate_support_forest",
                wraps=validate_support_forest,
            ) as forests,
            patch.object(PlanningSources, "check_current", count),
        ):
            validate_report(copy.deepcopy(self.applied))
        self.assertEqual(indexes.call_count, 1)
        self.assertEqual(forests.call_count, 1)
        self.assertLessEqual(len(calls), 3)
        self.assertGreater(len(self.candidate["citations"]), len(calls))

    def test_frozen_nested_mutation_is_revalidated_without_context(
        self,
    ) -> None:
        """A copied typed manifest and object.__setattr__ cannot bypass current proof."""
        manifest = GenerationManifest.model_validate_json(
            json.dumps(self.candidate["generation"])
        )
        forged = copy.deepcopy(manifest)
        object.__setattr__(
            forged.details[0].explanations[0], "before", "changed text"
        )
        with self.assertRaises(ValueError):
            validate_v2_generation(self.original, forged)
        self.assert_rejected(
            changed_record(
                manifest, plan=changed_record(manifest.plan, requests=())
            )
        )


class NondefaultBoundaryTests(unittest.TestCase):
    """Exercise persisted options, current bindings and real relationship plans."""

    def test_nondefault_options_prefix_and_capacities_survive(self) -> None:
        """Offline reconstruction uses exact declared settings rather than defaults."""
        provider = FixtureProvider()
        provider.context_tokens = 100_000
        provider.max_output_tokens = 20_000
        provider.request_prefix = b"local framed request\n"
        provider.input_overhead_bytes = 123
        options = V2Options(
            ledger=LedgerParameters(max_span_bytes=48, max_unit_spans=4),
            packing=RequestPackingLimits(
                stable_partition_count=3,
                framing_bytes=23,
                safety_tokens=777,
                repair_reserve_bytes=999,
                repair_output_tokens=123,
            ),
            card_wire_bytes=2048,
            prose_wire_bytes=4096,
            card_support_bytes=2048,
            max_chapter_groups=1,
        )
        report = golden_report()
        candidate = generate(report, provider=provider, options=options)
        manifest = GenerationManifest.model_validate_json(
            json.dumps(candidate["generation"])
        )
        self.assertEqual(manifest.request_configuration.options, options)
        self.assertEqual(
            manifest.request_configuration.prefix, provider.request_prefix
        )
        self.assertEqual(
            manifest.request_configuration.context_tokens,
            provider.context_tokens,
        )
        validate_report(apply_annotations(report, candidate))
        changed = manifest.model_dump(mode="json")
        changed["request_configuration"]["options"]["packing"][
            "framing_bytes"
        ] += 1
        with self.assertRaises((ValueError, ArtifactRuntimeError)):
            validate_generation(report, changed)

    def test_context_only_details_and_explicit_limitations_survive(
        self,
    ) -> None:
        """All units survive even when no section owns their unchanged context."""
        report = compile_snapshot(
            snap(
                {
                    "normal.py": "def f():\n    return 1\n",
                    "context.txt": "unchanged context\n",
                },
                {
                    "normal.py": "def f():\n    return 2\n",
                    "context.txt": "unchanged context\n",
                },
            )
        )
        candidate = generate(report)
        result = apply_annotations(report, candidate)
        self.assertTrue(result["narrative_v2"]["context_details"])
        self.assertEqual(
            result["narrative_v2"]["context_details"],
            candidate["context_details"],
        )
        validate_report(result)

    def test_author_terms_effects_and_uncertainty_remain_typed(self) -> None:
        """Preserve all prose families and explicit author attribution after reload."""

        def author(
            _provider: FixtureProvider, data: dict, response: dict
        ) -> dict:
            if data["stage"] != "detail":
                for output, item in zip(
                    response["items"], data["items"], strict=True
                ):
                    field = {
                        "planning": "abstraction",
                        "section": "synthesis",
                        "document": "lead",
                    }[data["stage"]]
                    proof = output[field][0]
                    proof["effect"] = (
                        "The supplied change affects normalization."
                    )
                    proof["uncertainty"] = ["No executed tests are supplied."]
                    output["terminology"] = [
                        {
                            "term": "Normalization",
                            "definition": copy.deepcopy(proof),
                        }
                    ]
                    if item["support"].get("author"):
                        output[field].append(
                            {
                                "kind": "author_motivation",
                                "text": "The author reports an intent to preserve indentation.",
                                "claims": [],
                                "citations": [],
                                "basis": [],
                                "effect": None,
                                "uncertainty": [],
                            }
                        )
            return response

        report = golden_report()
        candidate = generate(report, provider=FixtureProvider(author))
        applied = apply_annotations(report, candidate)
        validate_report(json.loads(json.dumps(applied)))
        typed = applied["narrative_v2"]
        self.assertEqual(typed["document"], candidate["document"])
        self.assertTrue(typed["document"]["terminology"])
        self.assertTrue(
            any(
                s["kind"] == "author_motivation"
                for s in typed["document"]["lead"]
            )
        )
        self.assertTrue(
            typed["sections"][0]["prose"]["synthesis"][0]["uncertainty"]
        )
        self.assertTrue(
            typed["sections"][0]["prose"]["synthesis"][0]["effect"]
        )
        manifest = GenerationManifest.model_validate_json(
            json.dumps(candidate["generation"])
        )
        section = manifest.sections[0]
        forged = changed_record(
            section.synthesis[-1], author_context_digest="0" * 64
        )
        altered = replace_result(
            manifest,
            "sections",
            0,
            changed_record(
                section, synthesis=(*section.synthesis[:-1], forged)
            ),
        )
        with self.assertRaises((ValueError, ArtifactRuntimeError)):
            validate_generation(report, altered.model_dump(mode="json"))

    def test_real_compiler_relationships_roundtrip_v2(self) -> None:
        """Actual import/call chains and SCCs retain complete public v2 plans."""
        from test_analysis import graph_snapshot

        for cyclic in (False, True):
            with self.subTest(cyclic=cyclic):
                report = compile_snapshot(graph_snapshot(72, cyclic=cyclic))
                candidate = generate(report)
                applied = apply_annotations(report, candidate)
                validate_report(json.loads(json.dumps(applied)))
                self.assertEqual(applied["edges"], report["edges"])
                self.assertEqual(applied["changes"], report["changes"])
                self.assertEqual(len(applied["groups"]), len(report["groups"]))
                plan = candidate["generation"]["plan"]
                if cyclic:
                    self.assertGreaterEqual(max(map(len, plan["cycles"])), 72)
                self.assertEqual(
                    candidate["steps"][-1]["prose"]["transition"], []
                )

    def test_exact_slices_half_crlf_utf8_shift_and_revision_reuse(  # noqa: C901, PLR0915 # Exercise current and cached source bindings end to end.
        self,
    ) -> None:
        """Resolve original byte positions across half CRLF and long physical lines."""
        import re

        line = '    value = "' + ("\u03b1" * 2030) + '\u2028\u2029"'
        line += " " * (4095 - len(line.encode()))
        text = "def source():\r\n" + line + "\r\n    return 1\r# CR\r# LF\n"
        source = snap(
            {"source.py": text},
            {
                "source.py": text.replace("\u03b1", "β").replace(
                    "return 1", "return 2"
                )
            },
            overrides={
                (side, "source.py"): {
                    "state": "supplied",
                    "coverage": "partial",
                }
                for side in ("base", "head")
            },
            fragment_options={
                (side, "source.py"): {"start_line": 31}
                for side in ("base", "head")
            },
            repository="example/project",
        )

        def observations(
            _provider: FixtureProvider, data: dict, response: dict
        ) -> dict:
            if data["stage"] == "detail":
                for output, item in zip(
                    response["items"],
                    (i for i in data["items"] if i["obligations"]),
                    strict=True,
                ):
                    output["observations"] = [
                        {
                            "text": f"Supplied record {n} retains source text.",
                            "citations": [s["handle"] for s in item["spans"]],
                        }
                        for n in range(12)
                    ]
            return response

        options = V2Options(
            ledger=LedgerParameters(max_span_bytes=4096, max_unit_spans=2),
            card_wire_bytes=1024,
            card_support_bytes=2048,
            prose_wire_bytes=4096,
        )
        with tempfile.TemporaryDirectory() as directory:
            candidates = []
            providers = []
            for revision in range(2):
                if revision:
                    for fragment in source["fragments"]:
                        fragment["start_line"] += 30
                    source["meta"].update(base_sha="c" * 40, head_sha="d" * 40)
                report = compile_snapshot(source)
                provider = FixtureProvider(observations)
                provider.context_tokens = 40_000
                provider.max_output_tokens = 15_000
                candidate = Narrator(
                    provider,
                    architecture="v2",
                    recovery_dir=Path(directory) / str(revision),
                    checkpoint_dir=Path(directory) / "cache",
                    concurrency=1,
                    v2_options=options,
                ).generate(report)
                providers.append(provider)
                candidates.append(candidate)
                applied = apply_annotations(report, candidate)
                validate_report(json.loads(json.dumps(applied)))
                fragments = {
                    f["key"]: f for f in report["source_evidence"]["fragments"]
                }
                halves = 0
                same_line_parts = set()
                slices = [
                    c
                    for c in candidate["citations"]
                    if c["consumption"] == "source_slice"
                ]
                self.assertTrue(slices)
                for entry in candidate["citations"]:
                    fragment = fragments[entry["fragment_key"]]
                    raw = fragment["text"].encode()
                    start, end = entry["byte_start"], entry["byte_end"]
                    ref = entry["reference"]
                    self.assertEqual(
                        content_digest(raw[start:end]), ref["content_digest"]
                    )
                    raw[start:end].decode("utf-8")
                    # A physical line ends only after the entire delimiter.
                    ends = [m.end() for m in re.finditer(rb"\r\n|\r|\n", raw)]
                    self.assertEqual(
                        entry["line_start"],
                        fragment["start_line"] + sum(p <= start for p in ends),
                    )
                    self.assertEqual(
                        entry["line_end"],
                        fragment["start_line"]
                        + sum(p <= end - 1 for p in ends),
                    )
                    self.assertEqual(entry["side"], fragment["side"])
                    self.assertIn(
                        source["meta"][entry["side"] + "_sha"], entry["url"]
                    )
                    for offset in (start, end):
                        halves += (
                            raw[max(0, offset - 1) : offset + 1] == b"\r\n"
                        )
                    if (
                        entry["line_start"]
                        == entry["line_end"]
                        == 32 + revision * 30
                    ):
                        same_line_parts.add((start, end))
                self.assertGreater(halves, 0)
                self.assertGreater(len(same_line_parts), 1)
                corrupted = copy.deepcopy(applied)
                corrupted["narrative_v2"]["citations"][0]["byte_end"] += 1
                with self.assertRaises((ValueError, ArtifactRuntimeError)):
                    validate_report(corrupted)
            self.assertEqual(providers[1].calls, [])
            self.assertEqual(
                candidates[0]["generation"]["producers"],
                candidates[1]["generation"]["producers"],
            )
            for old, new in zip(
                candidates[0]["citations"],
                candidates[1]["citations"],
                strict=True,
            ):
                self.assertEqual(old["key"], new["key"])
                self.assertEqual(new["line_start"], old["line_start"] + 30)
                self.assertEqual(new["line_end"], old["line_end"] + 30)
