"""Fixture-only production tests for complete batched narration and recovery."""

from __future__ import annotations

import copy
import json
import tempfile
import threading
import time
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from test_analysis import snap

from diffstory.analysis import compile_snapshot
from diffstory.narrative import CodexCLIProvider
from diffstory.narrative import Narrator
from diffstory.narrative import OpenAIResponsesProvider
from diffstory.narrative_manifest import GenerationManifest
from diffstory.narrative_manifest import LedgerParameters
from diffstory.narrative_manifest import validate_generation
from diffstory.narrative_plan import RequestPackingLimits
from diffstory.narrative_plan import build_evidence_ledger
from diffstory.narrative_plan import canonical_bytes
from diffstory.narrative_plan import reconstruct_buffer
from diffstory.narrative_runtime import ArtifactCallError
from diffstory.narrative_runtime import ArtifactRuntimeError
from diffstory.narrative_runtime import ProviderReply
from diffstory.narrative_runtime import RuntimeStageError
from diffstory.narrative_v2 import V2Options

BEFORE = 'def normalize(value):\n    value = value.strip()\n    marker = "request"\n    return value.lower()\n'
AFTER = 'def normalize(value):\n    value = value.rstrip()\n    marker = "request"\n    return value.casefold()\n'
STRIP_BEFORE = "strip() removed whitespace from both ends of the input."
STRIP_AFTER = "rstrip() removes only trailing whitespace and preserves leading whitespace."
STRIP_EFFECT = "The returned value retains leading whitespace that the base implementation removed."
CASE_BEFORE = "lower() converted cased characters to lowercase."
CASE_AFTER = (
    "casefold() applies Unicode case folding after whitespace handling."
)
CASE_EFFECT = "Unicode case folding can map characters differently from lower(), including expanding sharp s."


def golden_report(count: int = 2) -> dict:
    """Compile distinct files with two separate normalization mechanisms."""
    report = compile_snapshot(
        snap(
            {f"normal_{n}.py": BEFORE for n in range(count)},
            {f"normal_{n}.py": AFTER for n in range(count)},
        )
    )
    report["meta"]["repository"] = "example/project"
    report["meta"]["description"] = (
        "Preserve indentation and compare Unicode names consistently."
    )
    return report


def statement(text: str, support: dict | None = None) -> dict:
    """Write one supported statement using supplied handles only."""
    support = support or {"claims": [], "citations": []}
    claims = [c["handle"] for c in support["claims"][:1]]
    citations = [c["handle"] for c in support["citations"][:2]]
    basis = [c["handle"] for c in support.get("basis", [])[:1]]
    return {
        "kind": "implementation_claim"
        if claims or citations or basis
        else "editorial",
        "text": text,
        "claims": claims,
        "citations": citations,
        "basis": basis,
        "effect": None,
        "uncertainty": [],
    }


class FixtureProvider:
    """Return authored normalization prose and retain only local fixture requests."""

    request_protocol = "diffstory.fixture-json.v1"
    name = "fixture"
    model = "fixture-v2"
    destination = "fixture://local"
    context_tokens = 1_000_000
    max_output_tokens = 100_000
    input_overhead_bytes = 0
    enforceable_output_cap = True

    def __init__(self, hook: object = None) -> None:
        """Accept an optional fault injection at the transport boundary."""
        self.calls = []
        self.hook = hook

    def require_credentials(self) -> None:
        """Fixtures require no external credential or process."""

    def prepare(
        self,
        system: str,
        data: dict,
        schema_name: str,
        schema: dict,
        max_output_tokens: int,
    ) -> bytes:
        """Use actual JSON, schemas, instructions and enforceable output cap."""
        return canonical_bytes(
            {
                "system": system,
                "data": data,
                "schema_name": schema_name,
                "schema": schema,
                "max_output_tokens": max_output_tokens,
            }
        )

    def complete_reply(
        self, body: bytes, _timeout: float, *, response_byte_limit: int
    ) -> ProviderReply:
        """Make one bounded local attempt with measured response bytes."""
        request = json.loads(body)
        self.calls.append(request)
        data = request["data"]
        response = self.respond(data)
        if self.hook is not None:
            response = self.hook(self, data, response)
        raw = (
            canonical_bytes(response)
            if not isinstance(response, bytes)
            else response
        )
        usage = {
            "input_tokens": 17,
            "cached_input_tokens": 0,
            "output_tokens": 29,
        }
        if len(raw) > response_byte_limit:
            msg = "response_limit"
            raise ArtifactCallError(msg, usage=usage, response_bytes=len(raw))
        return ProviderReply(raw, usage, len(raw))

    @staticmethod
    def detail(item: dict) -> dict:
        """Explain each concrete changed region separately with exact required sides."""
        spans = {s["handle"]: s for s in item["spans"]}
        explanations = []
        for obligation in item["obligations"]:
            text = "".join(
                spans[h]["text"] for h in obligation["required_citations"]
            )
            if (
                obligation["kind"] == "structure"
                and "def normalize(value)" in text
            ):
                before = "The base defines normalize(value) with one input parameter."
                after = "The head keeps the normalize(value) function and its parameter while changing its body."
                effect = "The function name and parameter interface remain available to callers."
            elif ".strip()" in text or ".rstrip()" in text:
                before, after, effect = STRIP_BEFORE, STRIP_AFTER, STRIP_EFFECT
            elif ".lower()" in text or ".casefold()" in text:
                before, after, effect = CASE_BEFORE, CASE_AFTER, CASE_EFFECT
            else:
                before = "The base evidence is limited to the supplied source and availability facts."
                after = "The head evidence is limited to the supplied source and availability facts."
                effect = "The excerpt alone does not establish behavior beyond these source changes."
            explanations.append(
                {
                    "obligations": [obligation["handle"]],
                    "before": before,
                    "after": after,
                    "effect": effect,
                    "uncertainty": "No executed test results or caller expectations are supplied.",
                    "citations": list(obligation["required_citations"]),
                    "limitations": [
                        limit["handle"] for limit in obligation["limitations"]
                    ],
                }
            )
        head = [
            s["handle"]
            for s in item["spans"]
            if s["side"] == "head"
            and ("rstrip" in s["text"] or "casefold" in s["text"])
        ]
        return {
            "target": item["target"],
            "explanations": explanations,
            "observations": [
                {
                    "text": "The head normalization uses rstrip() and casefold().",
                    "citations": head,
                }
            ]
            if head
            else [],
            "questions": [
                {
                    "text": "Do callers require leading whitespace to remain?",
                    "citations": head[:1],
                    "obligations": [],
                }
            ]
            if head
            else [],
        }

    @classmethod
    def respond(cls, data: dict) -> dict:
        """Write complete detail, planning, section and document envelopes."""
        result = {"version": "diffstory.wire.v2", "items": []}
        if data["stage"] == "detail":
            result["acknowledgements"] = []
            for item in data["items"]:
                if item["obligations"]:
                    result["items"].append(cls.detail(item))
                else:
                    result["acknowledgements"].append(item["target"])
            return result
        for item in data["items"]:
            support = item["support"]
            prose = statement(
                "Normalization preserves leading whitespace and uses Unicode case folding.",
                support,
            )
            base = {"target": item["target"], "terminology": []}
            if data["stage"] == "planning":
                base["abstraction"] = [prose]
            elif data["stage"] == "section":
                base.update(
                    title=statement("Normalization behavior"),
                    orientation=[
                        statement(
                            "Read the whitespace rule before the case conversion."
                        )
                    ],
                    synthesis=[prose],
                    transition=[]
                    if item["final"]
                    else [
                        statement(
                            "The next section explains the neighboring change."
                        )
                    ],
                )
            else:
                base.update(
                    preamble=[
                        statement(
                            "This walkthrough separates source changes from test evidence."
                        )
                    ],
                    lead=[prose],
                    closing=[
                        statement(
                            "Check caller expectations for preserved leading whitespace."
                        )
                    ],
                )
            result["items"].append(base)
        return result


class BatchedNarratorTests(unittest.TestCase):
    """Check complete production outputs, safe reuse and approved egress."""

    def setUp(self) -> None:
        """Give each test an isolated durable recovery and artifact directory."""
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def narrator(
        self,
        provider: FixtureProvider,
        *,
        cached: bool = True,
        options: V2Options | None = None,
    ) -> Narrator:
        """Construct the public opt-in gate with explicit durable recovery."""
        return Narrator(
            provider,
            architecture="v2",
            recovery_dir=self.root / "state",
            checkpoint_dir=self.root / "cache" if cached else None,
            concurrency=1,
            v2_options=options,
        )

    def test_golden_two_groups_preserve_distinct_mechanisms_and_batch_sections(
        self,
    ) -> None:
        """One fitting evidence batch and one section batch retain all detailed prose."""
        report = golden_report()
        provider = FixtureProvider()
        narrator = self.narrator(provider)
        preview = narrator.preview(report)
        self.assertEqual(provider.calls, [])
        self.assertEqual(preview["evidence_batches"], 1)
        candidate = narrator.generate(report)
        stages = Counter(c["data"]["stage"] for c in provider.calls)
        self.assertEqual(stages["detail"], 1)
        self.assertEqual(stages["section"], 1)
        self.assertEqual(candidate["schema"], "diffstory.annotations.v2")
        manifest = GenerationManifest.model_validate_json(
            json.dumps(candidate["generation"])
        )
        self.assertEqual(len(manifest.sections), 2)
        self.assertEqual(manifest.sections[-1].transition, ())
        explanations = [e for d in manifest.details for e in d.explanations]
        self.assertEqual(
            Counter(
                e.before
                for e in explanations
                if e.before in {STRIP_BEFORE, CASE_BEFORE}
            ),
            Counter({STRIP_BEFORE: 2, CASE_BEFORE: 2}),
        )
        self.assertTrue(
            {STRIP_AFTER, CASE_AFTER} <= {e.after for e in explanations}
        )
        self.assertTrue(
            {STRIP_EFFECT, CASE_EFFECT} <= {e.effect for e in explanations}
        )
        for explanation in explanations:
            self.assertEqual(
                {c.side for c in explanation.citations}, {"base", "head"}
            )
        self.assertEqual(
            [d for step in candidate["steps"] for d in step["details"]],
            [
                d.model_dump(mode="json")
                for key in manifest.plan.ordered_group_keys
                for d in manifest.details
                if any(
                    u == d.unit_key
                    for g in manifest.plan.groups
                    if g.group.key == key
                    for u in g.group.unit_keys
                )
            ],
        )
        self.assertTrue(
            all(
                c["url"].startswith("https://github.com/example/project/blob/")
                for c in candidate["citations"]
            )
        )
        self.assertNotIn(
            report["meta"]["base_sha"], json.dumps(provider.calls)
        )
        validate_generation(
            report, manifest, provider=FixtureProvider(), candidate=candidate
        )
        with self.assertRaisesRegex(ArtifactRuntimeError, "busy"):
            narrator.generate(report)

    def test_public_dict_validation_rebuilds_its_own_production_index(
        self,
    ) -> None:
        """A public annotation dictionary must not inherit coordinator authority."""
        cases = [(golden_report(1), "diffstory.annotations.v2")]
        for report, expected_schema in cases:
            with self.subTest(schema=expected_schema):
                provider = FixtureProvider()
                narrator = self.narrator(provider)
                candidate = narrator.generate(report)
                self.assertIs(type(candidate), dict)
                self.assertEqual(candidate["schema"], expected_schema)
                manifest = GenerationManifest.model_validate_json(
                    canonical_bytes(candidate["generation"])
                )
                from diffstory.narrative_v2 import build_ledger_index

                with (
                    patch(
                        "diffstory.narrative_v2.build_evidence_ledger",
                        wraps=build_evidence_ledger,
                    ) as ledgers,
                    patch(
                        "diffstory.narrative_v2.build_ledger_index",
                        wraps=build_ledger_index,
                    ) as indexes,
                ):
                    validate_generation(report, manifest, candidate=candidate)
                self.assertEqual(ledgers.call_count, 1)
                self.assertEqual(indexes.call_count, 1)

    def test_many_fitting_groups_do_not_create_an_evidence_or_section_floor(
        self,
    ) -> None:
        """Presentation count alone does not force more provider calls."""
        provider = FixtureProvider()
        narrator = self.narrator(provider)
        candidate = narrator.generate(golden_report(8))
        stages = Counter(c["data"]["stage"] for c in provider.calls)
        self.assertEqual(stages["detail"], 1)
        self.assertEqual(stages["section"], 1)
        self.assertEqual(len(candidate["steps"]), 8)

    def test_explicit_gate_and_durable_no_cache_requirement(self) -> None:
        """Legacy stays default and v2 cannot run with ephemeral recovery only."""
        provider = FixtureProvider()
        self.assertEqual(Narrator(provider).architecture, "legacy")
        with self.assertRaisesRegex(ArtifactRuntimeError, "storage"):
            Narrator(provider, architecture="v2")
        candidate = self.narrator(provider, cached=False).generate(
            golden_report(1)
        )
        self.assertEqual(candidate["schema"], "diffstory.annotations.v2")
        with self.assertRaises((ArtifactRuntimeError, RuntimeStageError)):
            self.narrator(FixtureProvider(), cached=False).generate(
                golden_report(1)
            )

    def test_preview_rejects_changed_source_author_and_provider_before_send(
        self,
    ) -> None:
        """Approval cannot be replaced by an automatic second preview."""
        for mutation in (
            "source",
            "author",
            "model",
            "prefix",
            "capacity",
            "destination",
        ):
            with self.subTest(mutation=mutation):
                report = golden_report(1)
                provider = FixtureProvider()
                narrator = self.narrator(provider)
                narrator.preview(report)
                if mutation == "source":
                    report["source_evidence"]["fragments"][0]["text"] += (
                        "# changed\n"
                    )
                elif mutation == "author":
                    report["meta"]["description"] = (
                        "A different approved motivation."
                    )
                else:
                    field = {
                        "prefix": "request_prefix",
                        "capacity": "context_tokens",
                    }.get(mutation, mutation)
                    setattr(
                        provider,
                        field,
                        b"changed"
                        if mutation == "prefix"
                        else 2_000_000
                        if mutation == "capacity"
                        else "changed",
                    )
                with self.assertRaisesRegex(
                    ArtifactRuntimeError, "dependencies_changed"
                ):
                    narrator.generate(report)
                self.assertEqual(provider.calls, [])

    def test_unrelated_credential_value_errors_remain_visible(self) -> None:
        """Concern: a credentials ValueError can be mislabeled as changed input.

        Preconditions:
            Each case has a valid local report and an approved preview.
        Postconditions:
            The exact provider ``ValueError`` remains visible with no calls.
        Essential behavior:
            Covers public generation and explicit session-owned generation.

        Input/expected matrix:
        * Public ``generate`` -> original ``ValueError`` and zero calls.
        * ``capture_session`` plus ``generate_validated`` -> original
          ``ValueError`` and zero calls.
        """
        cases = (
            ("generate", ValueError, "fixture credentials are unavailable"),
            (
                "generate_validated",
                ValueError,
                "fixture credentials are unavailable",
            ),
        )
        for entrypoint, expected_error, expected_message in cases:
            with (
                self.subTest(entrypoint=entrypoint),
                tempfile.TemporaryDirectory() as directory,
            ):
                report = golden_report(1)
                provider = FixtureProvider()
                narrator = Narrator(
                    provider,
                    architecture="v2",
                    recovery_dir=Path(directory) / "state",
                    concurrency=1,
                )
                narrator.preview(report)
                session = None
                try:
                    if entrypoint == "generate_validated":
                        session = narrator.capture_session(report)
                    with (
                        patch.object(
                            provider,
                            "require_credentials",
                            side_effect=ValueError(expected_message),
                        ),
                        self.assertRaisesRegex(
                            expected_error, expected_message
                        ) as raised,
                    ):
                        if entrypoint == "generate":
                            narrator.generate(report)
                        else:
                            narrator.generate_validated(
                                report, session=session
                            )
                    self.assertIs(type(raised.exception), expected_error)
                    self.assertEqual(provider.calls, [])
                finally:
                    if session is not None:
                        session.close()

    def test_author_edit_reuses_source_only_details_and_invalidates_prose(
        self,
    ) -> None:
        """Historical usage remains separate while changed author context regenerates."""
        report = golden_report()
        self.narrator(FixtureProvider()).generate(report)
        report["meta"]["description"] = "Review Unicode identifiers."
        provider = FixtureProvider()
        narrator = self.narrator(provider)
        candidate = narrator.generate(report)
        self.assertNotIn(
            "detail", [c["data"]["stage"] for c in provider.calls]
        )
        self.assertIn("section", [c["data"]["stage"] for c in provider.calls])
        self.assertGreater(narrator.v2_counters["reused_artifacts"], 0)
        self.assertEqual(narrator.usage.report()["calls"], len(provider.calls))
        manifest = GenerationManifest.model_validate_json(
            json.dumps(candidate["generation"])
        )
        self.assertTrue(
            all(
                r.disposition == "reused"
                for r in manifest.receipts
                if r.result_kind == "detail"
            )
        )
        validate_generation(
            report, manifest, provider=FixtureProvider(), candidate=candidate
        )

    def test_unknown_plus_missing_repairs_only_known_gap_and_preserves_success(
        self,
    ) -> None:
        """Unknown extras stay diagnosed; the missing peer gets its one repair."""
        removed = []

        def fault(
            provider: FixtureProvider, _data: dict, response: dict
        ) -> dict:
            if len(provider.calls) == 1:
                removed.append(response["items"].pop()["target"])
                response["items"].append({"target": "unknown-extra"})
            return response

        provider = FixtureProvider(fault)
        narrator = self.narrator(provider)
        candidate = narrator.generate(golden_report())
        details = [
            c["data"] for c in provider.calls if c["data"]["stage"] == "detail"
        ]
        self.assertEqual(len(details), 2)
        self.assertTrue(details[1]["repair"])
        self.assertEqual([i["target"] for i in details[1]["items"]], removed)
        self.assertEqual(narrator.v2_counters["repairs"], 1)
        self.assertTrue(
            any(d.unknown_items == 1 for d in narrator.v2_diagnostics)
        )
        manifest = GenerationManifest.model_validate_json(
            json.dumps(candidate["generation"])
        )
        validate_generation(
            golden_report(),
            manifest,
            provider=FixtureProvider(),
            candidate=candidate,
        )

    def test_production_validator_rejects_self_consistent_unsupported_prose(
        self,
    ) -> None:
        """Typed self-consistency cannot authorize support omitted from item input."""
        report = golden_report()
        candidate = self.narrator(FixtureProvider()).generate(report)
        manifest = GenerationManifest.model_validate_json(
            json.dumps(candidate["generation"])
        )
        corrupted = copy.deepcopy(candidate)
        corrupted["steps"][0]["details"][0]["explanations"][0]["effect"] = (
            "Changed after generation."
        )
        with self.assertRaisesRegex(ArtifactRuntimeError, "coverage"):
            validate_generation(
                report,
                manifest,
                provider=FixtureProvider(),
                candidate=corrupted,
            )

    def test_source_bytes_context_acknowledgements_and_unavailable_evidence(
        self,
    ) -> None:
        """Dispatch original Unicode/CRLF/CR/LF slices and explicit unavailable facts."""
        before = "header\r\n" + "café" * 30 + "\u2028tail\rnext\n"
        after = before.replace("tail", "TAIL")
        report = compile_snapshot(
            snap(
                {"notes.txt": before, "same.txt": "unchanged\r\n"},
                {"notes.txt": after, "same.txt": "unchanged\r\n"},
            )
        )
        options = V2Options(
            ledger=LedgerParameters(max_span_bytes=9, max_unit_spans=2)
        )
        provider = FixtureProvider()
        candidate = self.narrator(provider, options=options).generate(report)
        ledger = build_evidence_ledger(report, **options.ledger.model_dump())
        sent = [
            i
            for c in provider.calls
            if c["data"]["stage"] == "detail"
            for i in c["data"]["items"]
        ]
        self.assertEqual(len(sent), len(ledger.units))
        sent_text = Counter(
            (s["side"], s["path"], s["text"]) for i in sent for s in i["spans"]
        )
        expected_text = Counter(
            (s.side, s.path, s.text) for u in ledger.units for s in u.spans
        )
        self.assertEqual(sent_text, expected_text)
        self.assertTrue(any(not i["obligations"] for i in sent))
        fragments = {
            f["key"]: f["text"].encode()
            for f in report["source_evidence"]["fragments"]
        }
        for binding in ledger.bindings:
            self.assertEqual(
                reconstruct_buffer(ledger, binding.buffer_key),
                fragments[binding.fragment_key],
            )
        manifest = GenerationManifest.model_validate_json(
            json.dumps(candidate["generation"])
        )
        self.assertEqual(len(manifest.details), len(ledger.units))
        validate_generation(
            report,
            manifest,
            provider=FixtureProvider(),
            options=options,
            candidate=candidate,
        )

    def test_hierarchy_cycles_actual_bounds_and_cross_batch_neighbors(
        self,
    ) -> None:
        """A fixed multilevel plan preserves cycles and section boundary context."""
        report = golden_report(24)
        first, second = report["groups"][:2]
        first["prerequisites"] = [second["id"]]
        second["prerequisites"] = [first["id"]]
        provider = FixtureProvider()
        provider.context_tokens = 90_000
        provider.max_output_tokens = 50_000
        options = V2Options(
            card_wire_bytes=1024,
            card_support_bytes=2048,
            prose_wire_bytes=4096,
            packing=RequestPackingLimits(stable_partition_count=1),
        )
        narrator = self.narrator(provider, options=options)
        candidate = narrator.generate(report)
        manifest = GenerationManifest.model_validate_json(
            json.dumps(candidate["generation"])
        )
        self.assertGreater(max(c.level for c in manifest.plan.cards), 1)
        self.assertEqual(len(manifest.plan.cycles), 1)
        self.assertEqual(len(manifest.plan.ordered_group_keys), 24)
        section_calls = [
            c for c in provider.calls if c["data"]["stage"] == "section"
        ]
        self.assertGreater(len(section_calls), 1)
        for call in section_calls:
            positions = [i["position"] for i in call["data"]["items"]]
            self.assertEqual(
                positions, list(range(min(positions), max(positions) + 1))
            )
            for item in call["data"]["items"]:
                self.assertEqual(
                    item["previous"] is None, item["position"] == 0
                )
                self.assertEqual(item["next"] is None, item["position"] == 23)
        for producer in manifest.producers:
            for request in producer.requests:
                self.assertLessEqual(
                    request.batch.input_bound
                    + request.batch.output_guard
                    + request.safety_tokens,
                    provider.context_tokens,
                )
                self.assertLessEqual(
                    request.batch.output_reservation,
                    provider.max_output_tokens,
                )
        validate_generation(
            report,
            manifest,
            provider=provider,
            options=options,
            candidate=candidate,
        )

    def test_unrelated_file_reuses_original_peers_before_new_packing(
        self,
    ) -> None:
        """A new file cannot merge old work into a fresh budget or request."""
        before = golden_report(2)
        old = self.narrator(FixtureProvider()).generate(before)
        provider = FixtureProvider()
        narrator = self.narrator(provider)
        candidate = narrator.generate(golden_report(3))
        detail_inputs = [
            i
            for c in provider.calls
            if c["data"]["stage"] == "detail"
            for i in c["data"]["items"]
        ]
        self.assertEqual(len(detail_inputs), 1)
        self.assertEqual(
            {s["path"] for s in detail_inputs[0]["spans"]}, {"normal_2.py"}
        )
        old_details = {d["unit_key"]: d for d in old["generation"]["details"]}
        current_details = {
            d["unit_key"]: d for d in candidate["generation"]["details"]
        }
        self.assertTrue(old_details.items() <= current_details.items())
        manifest = GenerationManifest.model_validate_json(
            json.dumps(candidate["generation"])
        )
        validate_generation(
            golden_report(3),
            manifest,
            provider=FixtureProvider(),
            candidate=candidate,
        )

    def test_shifted_lines_rebind_cached_citations_to_current_revision(
        self,
    ) -> None:
        """Content-stable artifacts retain prose and resolve current coordinates."""
        source = snap({"normal.py": BEFORE}, {"normal.py": AFTER})
        # Partial compiler fragments can shift without adding source bytes.
        for fragment in source["fragments"]:
            fragment["scope"] = "partial"
        for pair in source["file_evidence"]:
            pair["base"]["coverage"] = "partial"
            pair["head"]["coverage"] = "partial"
        report = compile_snapshot(source)
        report["meta"]["repository"] = "example/project"
        first = self.narrator(FixtureProvider()).generate(report)
        for fragment in source["fragments"]:
            fragment["start_line"] = fragment.get("start_line", 1) + 30
        source["meta"]["base_sha"] = "c" * 40
        source["meta"]["head_sha"] = "d" * 40
        current = compile_snapshot(source)
        current["meta"]["repository"] = "example/project"
        provider = FixtureProvider()
        second = self.narrator(provider).generate(current)
        self.assertEqual(provider.calls, [])
        self.assertEqual(
            first["generation"]["details"], second["generation"]["details"]
        )
        old_lines = {c["key"]: c["line_start"] for c in first["citations"]}
        for citation in second["citations"]:
            self.assertEqual(
                citation["line_start"], old_lines[citation["key"]] + 30
            )
            revision = current["meta"][citation["reference"]["side"] + "_sha"]
            self.assertIn(revision, citation["url"])

    def test_transient_request_is_identical_and_usage_is_current_only(
        self,
    ) -> None:
        """A transient failure retries exact original bytes once and keeps usage."""

        def fault(
            provider: FixtureProvider, _data: dict, response: dict
        ) -> dict:
            if len(provider.calls) == 1:
                msg = "rate_limit"
                raise ArtifactCallError(
                    msg,
                    usage={"input_tokens": 5, "output_tokens": 2},
                    response_bytes=17,
                )
            return response

        provider = FixtureProvider(fault)
        narrator = self.narrator(provider)
        narrator.generate(golden_report(1))
        self.assertEqual(provider.calls[0], provider.calls[1])
        self.assertEqual(narrator.v2_counters["retries"], 1)
        self.assertEqual(
            narrator.usage.report()["input_tokens"],
            5 + 17 * (len(provider.calls) - 1),
        )
        resumed = self.narrator(FixtureProvider())
        resumed.generate(golden_report(1))
        self.assertEqual(resumed.usage.report()["calls"], 0)

    def test_terminal_partial_failure_preserves_original_budget_on_restart(
        self,
    ) -> None:
        """Completed peers survive two missing responses without a new repair slot."""

        def omit(
            _provider: FixtureProvider, data: dict, response: dict
        ) -> dict:
            if data["stage"] == "detail":
                response["items"] = response["items"][:-1]
            return response

        provider = FixtureProvider(omit)
        narrator = self.narrator(provider)
        with self.assertRaises(RuntimeStageError):
            narrator.generate(golden_report())
        self.assertEqual(len(provider.calls), 2)
        self.assertEqual(len(provider.calls[1]["data"]["items"]), 1)
        resumed_provider = FixtureProvider()
        with self.assertRaises(RuntimeStageError):
            self.narrator(resumed_provider).generate(golden_report(3))
        self.assertEqual(resumed_provider.calls, [])

    def test_interrupted_repair_resumes_original_descriptor(self) -> None:
        """Interruption after partial success keeps the same missing peer and cap."""

        def interrupt(
            provider: FixtureProvider, data: dict, response: dict
        ) -> dict:
            if len(provider.calls) == 1:
                response["items"].pop()
            elif data["repair"]:
                raise KeyboardInterrupt
            return response

        provider = FixtureProvider(interrupt)
        with self.assertRaises((KeyboardInterrupt, RuntimeStageError)):
            self.narrator(provider).generate(golden_report())
        self.assertEqual(len(provider.calls), 2)
        resumed_provider = FixtureProvider()
        resumed = self.narrator(resumed_provider)
        candidate = resumed.generate(golden_report())
        self.assertEqual(resumed_provider.calls[0], provider.calls[1])
        self.assertEqual(resumed.v2_counters["repairs"], 0)
        self.assertEqual(resumed.v2_counters["retries"], 1)
        validate_generation(
            golden_report(),
            GenerationManifest.model_validate_json(
                json.dumps(candidate["generation"])
            ),
            provider=FixtureProvider(),
            candidate=candidate,
        )

    def test_actual_response_size_splits_by_declared_items_without_clipping(
        self,
    ) -> None:
        """A measured overflow splits the original lineage and retains all results."""

        def oversize(
            _provider: FixtureProvider, data: dict, response: dict
        ) -> dict:
            if data["stage"] == "detail" and len(data["items"]) > 1:
                msg = "response_limit"
                raise ArtifactCallError(
                    msg,
                    usage={"input_tokens": 3, "output_tokens": 4},
                    response_bytes=2_000_001,
                )
            return response

        provider = FixtureProvider(oversize)
        narrator = self.narrator(provider)
        candidate = narrator.generate(golden_report())
        detail_calls = [
            c["data"] for c in provider.calls if c["data"]["stage"] == "detail"
        ]
        self.assertEqual([len(c["items"]) for c in detail_calls], [2, 1, 1])
        self.assertEqual(narrator.v2_counters["splits"], 1)
        manifest = GenerationManifest.model_validate_json(
            json.dumps(candidate["generation"])
        )
        self.assertEqual(len(manifest.details), 2)
        validate_generation(
            golden_report(),
            manifest,
            provider=FixtureProvider(),
            candidate=candidate,
        )

    def test_auth_prohibited_tools_and_mutation_stop_further_egress(
        self,
    ) -> None:
        """Fatal provider failures and changed approved inputs stop dependent work."""
        for reason in (
            "auth",
            "prohibited_tools",
            "mutate_source",
            "mutate_model",
        ):
            with (
                self.subTest(reason=reason),
                tempfile.TemporaryDirectory() as directory,
            ):
                report = golden_report()

                def fault(
                    provider: FixtureProvider,
                    _data: dict,
                    response: dict,
                    reason: str = reason,
                    report: dict = report,
                ) -> dict:
                    if reason.startswith("mutate"):
                        if reason == "mutate_source":
                            report["meta"]["description"] += " changed"
                        else:
                            provider.model = "changed"
                        return response
                    raise ArtifactCallError(
                        reason, usage={"input_tokens": 3, "output_tokens": 1}
                    )

                provider = FixtureProvider(fault)
                narrator = Narrator(
                    provider,
                    architecture="v2",
                    recovery_dir=directory,
                    concurrency=1,
                )
                with self.assertRaises(
                    (RuntimeStageError, ArtifactRuntimeError)
                ):
                    narrator.generate(report)
                self.assertEqual(len(provider.calls), 1)

    def test_actual_output_response_and_timeout_reservations_split_fitting_input(
        self,
    ) -> None:
        """Practical output constraints split even when source input fits easily."""
        for limit in ("output", "response", "timeout"):
            with (
                self.subTest(limit=limit),
                tempfile.TemporaryDirectory() as directory,
            ):
                provider = FixtureProvider()
                packing = {"stable_partition_count": 1}
                if limit == "output":
                    provider.max_output_tokens = 4000
                elif limit == "response":
                    packing["response_byte_limit"] = 32_000
                else:
                    packing.update(
                        timeout_seconds=55, output_tokens_per_second=80
                    )
                options = V2Options(
                    card_wire_bytes=1024,
                    prose_wire_bytes=2048,
                    card_support_bytes=2048,
                    packing=RequestPackingLimits(**packing),
                )
                narrator = Narrator(
                    provider,
                    architecture="v2",
                    recovery_dir=directory,
                    v2_options=options,
                    concurrency=1,
                )
                candidate = narrator.generate(golden_report())
                detail_calls = [
                    c for c in provider.calls if c["data"]["stage"] == "detail"
                ]
                self.assertEqual(
                    [len(c["data"]["items"]) for c in detail_calls], [1, 1]
                )
                self.assertEqual(len(candidate["generation"]["details"]), 2)

    def test_unavailable_absent_and_partial_limits_survive_complete_assembly(
        self,
    ) -> None:
        """Unavailable source requires its exact limitation instead of confirmed absence."""
        for state in (
            {"state": "absent"},
            {"state": "unavailable", "reason": "read_failed"},
        ):
            with (
                self.subTest(state=state["state"]),
                tempfile.TemporaryDirectory() as directory,
            ):
                report = compile_snapshot(
                    snap(
                        {},
                        {"normal.py": AFTER},
                        overrides={("base", "normal.py"): state},
                    )
                )
                provider = FixtureProvider()
                candidate = Narrator(
                    provider, architecture="v2", recovery_dir=directory
                ).generate(report)
                details = candidate["generation"]["details"]
                limitations = [
                    limit
                    for d in details
                    for e in d["explanations"]
                    for limit in e["limitations"]
                ]
                self.assertEqual(
                    bool(limitations), state["state"] == "unavailable"
                )
                if limitations:
                    self.assertEqual(
                        {limit["code"] for limit in limitations},
                        {"unavailable"},
                    )
                units = [
                    i
                    for c in provider.calls
                    if c["data"]["stage"] == "detail"
                    for i in c["data"]["items"]
                ]
                self.assertEqual(
                    {
                        a["state"]
                        for u in units
                        for a in u["availability"]
                        if a["side"] == "base"
                    },
                    {state["state"]},
                )

    def test_oversized_card_prose_repairs_before_parent_send_and_keeps_plan_id(
        self,
    ) -> None:
        """Oversized output is an item-contract repair, never clipped or relabelled."""
        injected = []

        def oversize(
            _provider: FixtureProvider, data: dict, response: dict
        ) -> dict:
            if data["stage"] == "planning" and not injected:
                injected.append(response["items"][0]["target"])
                response["items"][0]["abstraction"][0]["text"] = (
                    "detail " * 1000
                )
            return response

        provider = FixtureProvider(oversize)
        narrator = self.narrator(provider)
        candidate = narrator.generate(golden_report())
        planning = [
            c["data"]
            for c in provider.calls
            if c["data"]["stage"] == "planning"
        ]
        self.assertEqual([i["target"] for i in planning[1]["items"]], injected)
        self.assertTrue(planning[1]["repair"])
        self.assertEqual(
            len({i["plan"] for c in planning for i in c["items"]}), 1
        )
        manifest = GenerationManifest.model_validate_json(
            json.dumps(candidate["generation"])
        )
        subjects = [
            k for batch in manifest.plan.requests for k in batch.item_keys
        ]
        self.assertEqual(
            Counter(subjects), Counter(c.id for c in manifest.plan.cards)
        )
        validate_generation(
            golden_report(),
            manifest,
            provider=FixtureProvider(),
            candidate=candidate,
        )

    def test_missing_and_duplicate_acks_repair_only_unaccounted_source(
        self,
    ) -> None:
        """Context acknowledgements are explicit source coverage records."""
        report = compile_snapshot(
            snap(
                {"normal.py": BEFORE, "context.txt": "unchanged\n"},
                {"normal.py": AFTER, "context.txt": "unchanged\n"},
            )
        )
        for mode in ("missing", "duplicate"):
            with (
                self.subTest(mode=mode),
                tempfile.TemporaryDirectory() as directory,
            ):
                target = []

                def fault(
                    provider: FixtureProvider,
                    _data: dict,
                    response: dict,
                    mode: str = mode,
                    target: list[str] = target,
                ) -> dict:
                    if len(provider.calls) == 1:
                        target.append(response["acknowledgements"][0])
                        if mode == "missing":
                            response["acknowledgements"].pop(0)
                        else:
                            response["acknowledgements"].append(target[0])
                    return response

                provider = FixtureProvider(fault)
                narrator = Narrator(
                    provider,
                    architecture="v2",
                    recovery_dir=directory,
                    concurrency=1,
                )
                candidate = narrator.generate(report)
                self.assertEqual(
                    [i["target"] for i in provider.calls[1]["data"]["items"]],
                    target,
                )
                self.assertEqual(
                    len(candidate["generation"]["details"]),
                    len(build_evidence_ledger(report).units),
                )

    def test_malformed_envelope_and_duplicate_properties_repair_whole_request(
        self,
    ) -> None:
        """A bad envelope cannot manufacture a partial success or receipt."""
        for mode in ("envelope", "duplicate"):
            with (
                self.subTest(mode=mode),
                tempfile.TemporaryDirectory() as directory,
            ):

                def fault(
                    provider: FixtureProvider,
                    _data: dict,
                    response: dict,
                    mode: str = mode,
                ) -> dict | bytes:
                    if len(provider.calls) == 1:
                        if mode == "envelope":
                            return response | {"version": "wrong"}
                        return b'{"version":"diffstory.wire.v2","items":[],"items":[],"acknowledgements":[]}'
                    return response

                provider = FixtureProvider(fault)
                narrator = Narrator(
                    provider,
                    architecture="v2",
                    recovery_dir=directory,
                    concurrency=1,
                )
                narrator.generate(golden_report())
                self.assertEqual(
                    provider.calls[0]["data"]["items"],
                    provider.calls[1]["data"]["items"],
                )
                self.assertEqual(narrator.v2_counters["repairs"], 1)

    def test_cached_invalid_citation_is_rejected_without_resetting_completed_work(
        self,
    ) -> None:
        """A valid checksum and typed result cannot authorize a stale source citation."""
        from diffstory.narrative_runtime import ArtifactStore
        from diffstory.narrative_runtime import _StoredItem
        from diffstory.narrative_storage import storage_digest

        report = golden_report()
        self.narrator(FixtureProvider()).generate(report)
        store = ArtifactStore(self.root / "cache")
        changed = False
        for path in store.directory.glob("*/*.json"):
            data = json.loads(path.read_bytes())
            if store.read(path.read_bytes()).lookup.result_kind == "detail":
                data["result"]["explanations"][0]["citations"][0][
                    "content_digest"
                ] = "0" * 64
                raw = canonical_bytes(data)
                key = storage_digest(_StoredItem.model_validate_json(raw))
                path.with_name(key + ".json").write_bytes(raw)
                path.unlink()
                changed = True
                break
        self.assertTrue(changed)
        provider = FixtureProvider()
        narrator = self.narrator(provider)
        with self.assertRaises(RuntimeStageError):
            narrator.generate(report)
        self.assertEqual(provider.calls, [])
        self.assertTrue(
            any(
                d.reason == "results_unavailable"
                for d in narrator.v2_diagnostics
            )
        )

    def test_current_terms_and_author_prose_are_retained_and_validated(
        self,
    ) -> None:
        """Supported terms and author attribution survive in the typed manifest."""

        def add_terms(
            _provider: FixtureProvider, data: dict, response: dict
        ) -> dict:
            if data["stage"] == "planning":
                for item, context in zip(
                    response["items"], data["items"], strict=True
                ):
                    item["terminology"] = [
                        {
                            "term": "case folding",
                            "definition": statement(
                                "Unicode case folding is used for normalized comparison.",
                                context["support"],
                            ),
                        }
                    ]
            if data["stage"] == "document":
                motivation = statement(
                    "The author wants indentation to remain and Unicode names to compare consistently."
                )
                motivation["kind"] = "author_motivation"
                response["items"][0]["lead"].append(motivation)
            return response

        provider = FixtureProvider(add_terms)
        candidate = self.narrator(provider).generate(golden_report())
        manifest = GenerationManifest.model_validate_json(
            json.dumps(candidate["generation"])
        )
        self.assertTrue(all(p.terminology for p in manifest.planning))
        self.assertTrue(manifest.document.lead[-1].author_context_digest)
        self.assertEqual(manifest.document.lead[-1].kind, "author_motivation")
        validate_generation(
            golden_report(),
            manifest,
            provider=FixtureProvider(),
            candidate=candidate,
        )

    def test_one_index_is_built_and_mutated_prompts_fail_approval(
        self,
    ) -> None:
        """Packing uses thin handles and prompt bytes remain part of approval."""
        from diffstory.narrative_plan import build_ledger_index
        from diffstory.narrative_v2 import PROMPTS

        provider = FixtureProvider()
        narrator = self.narrator(provider)
        report = golden_report()
        with patch(
            "diffstory.narrative_v2.build_ledger_index",
            wraps=build_ledger_index,
        ) as build:
            narrator.preview(report)
            narrator.generate(report)
        self.assertEqual(build.call_count, 1)
        other = self.narrator(FixtureProvider())
        other.preview(report)
        with (
            patch.dict(PROMPTS, detail=PROMPTS["detail"] + " Changed."),
            self.assertRaisesRegex(
                ArtifactRuntimeError, "dependencies_changed"
            ),
        ):
            other.generate(report)

    def test_reappearing_files_reuse_overlapping_original_detail_peers(
        self,
    ) -> None:
        """Different historical co-batches can reuse their complete current facts."""
        self.narrator(FixtureProvider()).generate(golden_report(2))
        other = compile_snapshot(
            snap(
                {f"normal_{n}.py": BEFORE for n in (1, 2)},
                {f"normal_{n}.py": AFTER for n in (1, 2)},
            )
        )
        other["meta"]["repository"] = "example/project"
        other["meta"]["description"] = golden_report()["meta"]["description"]
        self.narrator(FixtureProvider()).generate(other)
        provider = FixtureProvider()
        candidate = self.narrator(provider).generate(golden_report(3))
        self.assertNotIn(
            "detail", [c["data"]["stage"] for c in provider.calls]
        )
        manifest = GenerationManifest.model_validate_json(
            json.dumps(candidate["generation"])
        )
        self.assertEqual(len(manifest.details), 3)
        self.assertTrue(
            all(
                r.disposition == "reused"
                for r in manifest.receipts
                if r.result_kind == "detail"
            )
        )
        validate_generation(
            golden_report(3),
            manifest,
            provider=FixtureProvider(),
            candidate=candidate,
        )

    def test_large_group_uses_complete_multilevel_cards_with_small_neighbor(
        self,
    ) -> None:
        """A large group splits between atoms and keeps a complete structural tree."""
        before = "\n".join(
            BEFORE.replace("normalize", f"normalize_{n}") for n in range(20)
        )
        after = "\n".join(
            AFTER.replace("normalize", f"normalize_{n}") for n in range(20)
        )
        report = compile_snapshot(
            snap(
                {"large.py": before, "small.py": BEFORE},
                {"large.py": after, "small.py": AFTER},
            )
        )
        provider = FixtureProvider()
        provider.context_tokens = 40_000
        provider.max_output_tokens = 15_000
        options = V2Options(
            card_wire_bytes=1024,
            card_support_bytes=2048,
            prose_wire_bytes=4096,
            packing=RequestPackingLimits(stable_partition_count=1),
        )
        candidate = self.narrator(provider, options=options).generate(report)
        manifest = GenerationManifest.model_validate_json(
            json.dumps(candidate["generation"])
        )
        large = next(
            g for g in manifest.plan.groups if g.group.path == "large.py"
        )
        leaves = [
            c
            for c in manifest.plan.cards
            if any(a.group_key == large.group.key for a in c.atoms)
        ]
        self.assertGreater(len(leaves), 1)
        self.assertEqual(
            Counter(
                a.key for c in leaves for a in c.atoms if a.kind == "unit"
            ),
            Counter(large.group.unit_keys),
        )
        self.assertEqual(len(candidate["steps"]), len(manifest.plan.groups))
        validate_generation(
            report,
            manifest,
            provider=provider,
            options=options,
            candidate=candidate,
        )

    def test_all_known_success_with_unknown_extra_stops_without_repair(
        self,
    ) -> None:
        """Unknown output is never hidden when no known target needs a repair."""

        def fault(
            _provider: FixtureProvider, _data: dict, response: dict
        ) -> dict:
            response["items"].append({"target": "unknown-extra"})
            return response

        provider = FixtureProvider(fault)
        narrator = self.narrator(provider)
        with self.assertRaises(RuntimeStageError):
            narrator.generate(golden_report())
        self.assertEqual(len(provider.calls), 1)
        self.assertEqual(narrator.v2_counters["repairs"], 0)
        self.assertTrue(
            any(d.reason == "unknown_item" for d in narrator.v2_diagnostics)
        )

    def test_planning_prompt_edit_invalidates_only_consuming_stages(
        self,
    ) -> None:
        """Prompt dependencies do not make source-only artifacts consume prose rules."""
        from diffstory.narrative_v2 import PROMPTS

        report = golden_report()
        self.narrator(FixtureProvider()).generate(report)
        provider = FixtureProvider()
        with patch.dict(
            PROMPTS,
            planning=PROMPTS["planning"] + " Explain hierarchy boundaries.",
        ):
            narrator = self.narrator(provider)
            narrator.generate(report)
        self.assertNotIn(
            "detail", [c["data"]["stage"] for c in provider.calls]
        )
        self.assertIn("planning", [c["data"]["stage"] for c in provider.calls])

    def test_codex_prefix_actual_bytes_full_guard_and_unresolved_model_no_reuse(
        self,
    ) -> None:
        """Mock the actual CLI transport and audit its exact prefixed stdin bytes."""
        from diffstory.narrative_plan import content_digest

        provider = CodexCLIProvider(executable="fixture-codex")
        requests = []
        raw_sizes = []

        def process(request: object) -> tuple[int, bytes, bytes]:
            requests.append(request)
            self.assertTrue(request.prompt.startswith(provider.request_prefix))
            self.assertEqual(request.prompt.count(provider.request_prefix), 1)
            inner = json.loads(request.prompt[len(provider.request_prefix) :])
            response = FixtureProvider.respond(inner["data"])
            events = [
                {
                    "type": "item.completed",
                    "item": {
                        "type": "agent_message",
                        "text": json.dumps(response),
                    },
                },
                {
                    "type": "turn.completed",
                    "usage": {"input_tokens": 17, "output_tokens": 29},
                },
            ]
            raw = b"\n".join(canonical_bytes(e) for e in events)
            raw_sizes.append(len(raw))
            return 0, raw, b""

        with (
            patch.object(provider, "require_credentials"),
            patch(
                "diffstory.narrative._run_bounded_subprocess",
                side_effect=process,
            ),
        ):
            narrator = self.narrator(provider)
            candidate = narrator.generate(golden_report())
            actual_digests = {content_digest(r.prompt) for r in requests}
            manifest = GenerationManifest.model_validate_json(
                json.dumps(candidate["generation"])
            )
            for producer in manifest.producers:
                for metadata in producer.requests:
                    self.assertIn(
                        metadata.batch.request_digest, actual_digests
                    )
                    self.assertEqual(
                        metadata.batch.output_guard, provider.max_output_tokens
                    )
                    self.assertFalse(metadata.enforceable_output_cap)
            self.assertEqual(
                [
                    d.response_bytes
                    for d in narrator.v2_diagnostics
                    if d.reason == "generic"
                ],
                raw_sizes,
            )
            prior_calls = len(requests)
            with self.assertRaises(RuntimeStageError):
                self.narrator(provider).generate(golden_report())
            self.assertEqual(len(requests), prior_calls)
        self.assertFalse((self.root / "cache" / "artifacts-v3").exists())

    def test_openai_actual_wire_output_cap_and_response_metadata(self) -> None:
        """Mock HTTPS and retain its real request cap, schema and measured output."""
        requests = []

        def post(
            _api_key: str,
            body: bytes,
            _timeout: float,
            *,
            response_byte_limit: int,
        ) -> bytes:
            request = json.loads(body)
            requests.append(request)
            self.assertFalse(request["store"])
            self.assertGreater(response_byte_limit, 0)
            data = json.loads(request["input"][1]["content"])
            return canonical_bytes(
                {
                    "status": "completed",
                    "usage": {"input_tokens": 13, "output_tokens": 7},
                    "output": [
                        {
                            "type": "message",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": json.dumps(
                                        FixtureProvider.respond(data)
                                    ),
                                }
                            ],
                        }
                    ],
                }
            )

        provider = OpenAIResponsesProvider(api_key="fixture-key")
        with patch(
            "diffstory.narrative._post_openai_request", side_effect=post
        ):
            candidate = self.narrator(provider).generate(golden_report())
        manifest = GenerationManifest.model_validate_json(
            json.dumps(candidate["generation"])
        )
        self.assertEqual(manifest.current_usage.calls, len(requests))
        caps = Counter(r["max_output_tokens"] for r in requests)
        self.assertEqual(
            caps,
            Counter(
                r.batch.output_reservation
                for p in manifest.producers
                for r in p.requests
            ),
        )
        self.assertTrue(
            all(
                r.enforceable_output_cap
                and r.batch.output_guard == r.batch.output_reservation
                for p in manifest.producers
                for r in p.requests
            )
        )

    def test_auth_failure_drains_active_batch_and_stops_queued_work(
        self,
    ) -> None:
        """The narrator uses the runtime scheduler's bounded stop/drain behavior."""
        started = threading.Event()
        completed = threading.Event()
        lock = threading.Lock()
        positions = []

        def fault(
            _provider: FixtureProvider, _data: dict, response: dict
        ) -> dict:
            with lock:
                position = len(positions)
                positions.append(position)
            if position == 0:
                self.assertTrue(started.wait(2))
                msg = "auth"
                raise ArtifactCallError(msg)
            started.set()
            time.sleep(0.05)
            completed.set()
            return response

        provider = FixtureProvider(fault)
        provider.max_output_tokens = 4000
        options = V2Options(
            card_wire_bytes=1024,
            prose_wire_bytes=2048,
            card_support_bytes=2048,
        )
        narrator = Narrator(
            provider,
            architecture="v2",
            recovery_dir=self.root / "state",
            checkpoint_dir=self.root / "cache",
            concurrency=2,
            v2_options=options,
        )
        with self.assertRaises(RuntimeStageError):
            narrator.generate(golden_report(3))
        self.assertTrue(completed.is_set())
        self.assertEqual(len(provider.calls), 2)
        self.assertEqual(narrator.usage.report()["calls"], 2)
        self.assertEqual(narrator.v2_counters["repairs"], 0)


class ProductionRepairTests(unittest.TestCase):
    """Public-path closures for frozen production reductions and exact details."""

    def setUp(self) -> None:
        """Keep each reproduction in an isolated local artifact directory."""
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.options = V2Options(
            card_wire_bytes=1024,
            card_support_bytes=2048,
            prose_wire_bytes=4096,
            packing=RequestPackingLimits(stable_partition_count=1),
        )

    def run_public(
        self, report: dict, provider: FixtureProvider, *, name: str = "run"
    ) -> tuple:
        """Generate and independently rebuild the current manifest without sends."""
        from diffstory.narrative_v2 import validate_production

        provider.context_tokens = 40000
        provider.max_output_tokens = 15000
        narrator = Narrator(
            provider,
            architecture="v2",
            concurrency=1,
            v2_options=self.options,
            recovery_dir=self.root / name,
            checkpoint_dir=self.root / "cache",
        )
        candidate = narrator.generate(report)
        manifest = GenerationManifest.model_validate_json(
            canonical_bytes(candidate["generation"])
        )
        calls = len(provider.calls)
        validate_production(
            report,
            manifest,
            provider=provider,
            options=self.options,
            candidate=candidate,
        )
        self.assertEqual(len(provider.calls), calls)
        for producer in manifest.producers:
            for request in producer.requests:
                self.assertLessEqual(
                    request.batch.input_bound
                    + request.batch.output_guard
                    + request.safety_tokens,
                    40000,
                )
                self.assertLessEqual(request.batch.output_reservation, 15000)
                self.assertLessEqual(
                    request.batch.estimated_response_bytes,
                    request.response_byte_limit,
                )
                self.assertLessEqual(
                    request.batch.estimated_seconds, request.timeout_seconds
                )
        subjects = {
            c.id
            for c in (
                *manifest.plan.cards,
                *manifest.plan.production.auxiliary_cards,
            )
        }
        self.assertEqual(
            Counter(k for r in manifest.plan.requests for k in r.item_keys),
            Counter(subjects),
        )
        self.assertEqual(
            {r.binding.subject_key for r in manifest.planning}, subjects
        )
        self.assertEqual(
            {
                r.binding.plan_key
                for r in (
                    *manifest.planning,
                    *manifest.sections,
                    manifest.document,
                )
            },
            {manifest.plan.id},
        )
        self.assertEqual(
            len(manifest.receipts),
            len(manifest.details)
            + len(manifest.planning)
            + len(manifest.sections)
            + 1,
        )
        self.assertEqual(manifest.sections[-1].transition, ())
        self.assertEqual(
            [d for step in candidate["steps"] for d in step["details"]],
            [
                d.model_dump(mode="json")
                for group in manifest.plan.groups
                for unit in group.group.unit_keys
                for d in manifest.details
                if d.unit_key == unit
            ],
        )
        return narrator, candidate, manifest

    def assert_shapes_bound_actual(
        self, narrator: Narrator, manifest: GenerationManifest
    ) -> None:
        """Each sizing-only shape dominates its actual singleton, including repair."""
        from diffstory.narrative_hierarchy import planning_shapes

        inputs = narrator._v2.inputs  # noqa: SLF001 # Inspect the public gate's actual stage audit.
        stages = {
            i.subject: (stage, i)
            for stage in narrator._v2.stages  # noqa: SLF001
            for i in stage.items.values()
            if i.context is not None
        }
        body = inputs.transport.body
        captured = []

        def measured(*args: object) -> bytes:
            result = body(*args)
            captured.append(result)
            return result

        for shape in planning_shapes(manifest.plan, inputs.sources):
            subject = (
                manifest.plan.id
                if shape.stage == "document"
                else shape.subject_key
            )
            stage, item = stages[subject]
            with patch.object(inputs.transport, "body", side_effect=measured):
                inputs.check_shape(shape)
            upper = len(captured[-1])
            for repair in (False, True):
                request = stage.prepare(
                    (item.key,),
                    repair=repair,
                    reason="citation" if repair else None,
                    partition=0,
                )
                self.assertLessEqual(len(request.body), upper)

    def test_original_normal_and_47267_byte_handoff_and_warm_new_directory(
        self,
    ) -> None:
        """Accept one exact provider-valid long detail before lossless planning."""
        from diffstory.narrative_contracts import DetailResult
        from diffstory.narrative_v2 import ProductionInputs
        from diffstory.narrative_v2 import StageInputs
        from diffstory.narrative_v2 import Transport

        normal, _, _ = self.run_public(
            golden_report(1), FixtureProvider(), name="normal"
        )
        self.assertEqual(len(normal._v2.inputs.plan.production.fragments), 0)  # noqa: SLF001
        accepted = []

        def long_detail(
            _provider: FixtureProvider, data: dict, raw: dict
        ) -> dict:
            if data["stage"] == "detail":
                raw["items"][0]["explanations"][0]["before"] = (
                    "Authored long-detail handoff fixture. " * 1200
                )
                accepted.append(copy.deepcopy(raw))
            return raw

        # Use a separate cache so the reproduction truly consumes the long response.
        self.root = self.root / "long"
        provider = FixtureProvider(long_detail)
        narrator, _, manifest = self.run_public(golden_report(1), provider)
        self.assertEqual(len(accepted), 1)
        self.assertEqual(len(canonical_bytes(accepted[0])), 47267)
        self.assertEqual(
            Counter(c["data"]["stage"] for c in provider.calls)["detail"], 1
        )
        inputs = ProductionInputs(
            golden_report(1), Transport(provider, self.options)
        )
        stage = StageInputs(inputs, "detail", inputs.detail_items())
        request = stage.prepare(tuple(stage.items), repair=False, partition=0)
        envelope = inputs.wire.envelope(
            accepted[0], request.metadata.batch, {}
        )
        details = tuple(
            DetailResult.model_validate_json(canonical_bytes(r))
            for r in envelope.items
        )
        self.assertEqual(details, manifest.details)
        self.assertTrue(manifest.plan.production.fragments)
        self.assert_shapes_bound_actual(narrator, manifest)
        warm = FixtureProvider(long_detail)
        _, _, reused = self.run_public(
            golden_report(1), warm, name="new-recovery"
        )
        self.assertEqual(warm.calls, [])
        self.assertEqual(reused.details, details)
        self.assertEqual(reused.current_usage.calls, 0)
        self.assertTrue(
            all(r.disposition == "reused" for r in reused.receipts)
        )

    def test_original_64_group_fanin_and_64_member_cycle(self) -> None:
        """Large complete compiler relationships reduce before terminal requests."""
        for mode in ("fanin", "cycle"):
            with self.subTest(mode=mode):
                report = golden_report(64)
                if mode == "fanin":
                    report["groups"][-1]["prerequisites"] = [
                        g["id"] for g in report["groups"][:-1]
                    ]
                else:
                    for number, group in enumerate(report["groups"]):
                        group["prerequisites"] = [
                            report["groups"][(number + 1) % 64]["id"]
                        ]
                narrator, _, manifest = self.run_public(
                    report, FixtureProvider(), name=mode
                )
                hierarchy = manifest.plan.production
                self.assertTrue(hierarchy.auxiliary_cards)
                if mode == "fanin":
                    target = next(
                        g.group.key
                        for g in manifest.plan.groups
                        if len(g.group.prerequisites) == 63
                    )
                    view = next(
                        v
                        for v in hierarchy.section_views
                        if v.group_key == target
                    )
                    self.assertEqual(
                        sum(
                            s.relationship == "prerequisite"
                            for s in view.sources
                        ),
                        63,
                    )
                    self.assertIsNotNone(view.context_root)
                else:
                    self.assertEqual(len(manifest.plan.cycles), 1)
                    self.assertEqual(len(manifest.plan.cycles[0]), 64)
                    self.assertEqual(
                        len(hierarchy.cycle_roots[0].group_keys), 64
                    )
                self.assert_shapes_bound_actual(narrator, manifest)

    def test_unicode_records_author_heading_and_source_slice_provenance(
        self,
    ) -> None:
        """Preserve mixed physical endings and all fields through metadata reductions."""
        from diffstory.narrative_hierarchy import fragment_material
        from diffstory.narrative_hierarchy import record_key
        from diffstory.narrative_hierarchy import record_material

        text = ('é😀\r\nCR\rLF\nLS\u2028PS\u2029"\\  ' * 700) + "terminal  "
        report = golden_report(1)
        report["meta"]["title"] = "Authored title 😀 " * 2200
        report["meta"]["repository"] = "example/project/" * 1800
        report["meta"]["description"] = "Author motivation\r\n" * 2200
        report["groups"][0]["theme"] = "Compiler heading " * 2500
        accepted = []

        def expanded(
            _provider: FixtureProvider, data: dict, raw: dict
        ) -> dict:
            if data["stage"] == "detail":
                item = raw["items"][0]
                item["explanations"][0]["before"] = text
                item["observations"][0]["text"] = (
                    "Observation β\r\n\u2028  " * 700
                )
                item["observations"].extend(
                    {
                        "text": f"Additional observation {n}.",
                        "citations": list(
                            item["observations"][0]["citations"]
                        ),
                    }
                    for n in range(32)
                )
                accepted.append(copy.deepcopy(raw))
            elif data["stage"] == "planning":
                for output, item in zip(
                    raw["items"], data["items"], strict=True
                ):
                    if item["support"]["author"] and not any(
                        item["support"][k]
                        for k in ("claims", "citations", "basis")
                    ):
                        output["abstraction"][0]["kind"] = "author_motivation"
                        output["abstraction"][0]["text"] = (
                            "The author reports a motivation for the change."
                        )
            return raw

        narrator, candidate, manifest = self.run_public(
            report, FixtureProvider(expanded)
        )
        self.assertEqual(len(accepted), 1)
        self.assertEqual(manifest.details[0].explanations[0].before, text)
        self.assertEqual(len(manifest.details[0].observations), 33)
        h = manifest.plan.production
        self.assertEqual(
            {f.owner.kind for f in h.fragments},
            {"source", "detail", "author", "heading"},
        )
        inputs = narrator._v2.inputs  # noqa: SLF001
        fields = {}
        for f in h.fragments:
            fields.setdefault((record_key(f.owner), f.path), []).append(f)
        for (key, path), fragments in fields.items():
            actual = b"".join(
                fragment_material(inputs.sources, f) for f in fragments
            )
            expected = next(
                f.data
                for f in record_material(inputs.sources, key).fields
                if f.path == path
            )
            self.assertEqual(actual, expected)
        self.assertTrue(candidate["support_provenance"])
        slices = [
            c
            for c in candidate["citations"]
            if c.get("consumption") == "source_slice"
        ]
        self.assertTrue(slices)
        source_fragments = {
            f["key"]: f["text"].encode()
            for f in report["source_evidence"]["fragments"]
        }
        from diffstory.narrative_plan import content_digest

        for entry in slices:
            raw = source_fragments[entry["fragment_key"]][
                entry["byte_start"] : entry["byte_end"]
            ]
            self.assertEqual(
                content_digest(raw), entry["reference"]["content_digest"]
            )
        self.assert_shapes_bound_actual(narrator, manifest)

    def test_additional_json_layer_bounds_escaping_and_reduces_budgets(
        self,
    ) -> None:
        """Provider framing and an extra JSON string layer enter every size check."""

        class LayeredFixture(FixtureProvider):
            request_json_layers = 1
            request_prefix = b"local fixture framing\n"
            input_overhead_bytes = 113

            def prepare(self, *args: object) -> bytes:
                inner = super().prepare(*args)
                return canonical_bytes({"payload": inner.decode("utf-8")})

            def complete_reply(
                self, body: bytes, timeout: float, *, response_byte_limit: int
            ) -> ProviderReply:
                return super().complete_reply(
                    json.loads(body)["payload"].encode(),
                    timeout,
                    response_byte_limit=response_byte_limit,
                )

        def escaping(
            _provider: FixtureProvider, data: dict, raw: dict
        ) -> dict:
            if data["stage"] == "detail":
                raw["items"][0]["explanations"][0]["before"] = (
                    '😀"\\\r\n\u2028\u2029  ' * 1200
                )
            elif data["stage"] == "planning":
                for output, item in zip(
                    raw["items"], data["items"], strict=True
                ):
                    if item["max_support_bytes"] < 1024:
                        support = item["support"]
                        compact = {
                            "claims": [],
                            "citations": support["citations"][:1],
                            "basis": support["basis"][:1],
                        }
                        output["abstraction"] = [
                            statement(
                                "A supplied input supports this abstraction.",
                                compact,
                            )
                        ]
                    # Fill the wire budget with escaping-heavy complete text.
                    # The next request must remain below its frozen shape bound.
                    text = "Supported "
                    while True:
                        candidate = text + '\\"😀\u2028'
                        output["abstraction"][0]["text"] = candidate
                        if (
                            len(canonical_bytes(output))
                            > item["max_result_bytes"]
                        ):
                            output["abstraction"][0]["text"] = text
                            break
                        text = candidate
            return raw

        narrator, _, manifest = self.run_public(
            golden_report(1), LayeredFixture(escaping)
        )
        self.assertTrue(manifest.plan.production.fragments)
        self.assertLess(
            manifest.plan.production.card_budget.support_bytes,
            self.options.card_support_bytes,
        )
        self.assertGreaterEqual(
            manifest.plan.production.card_budget.support_bytes, 516
        )
        self.assert_shapes_bound_actual(narrator, manifest)

    def test_large_source_and_citation_collections_rebind_30_lines_without_calls(
        self,
    ) -> None:
        """Many physical source records keep exact slices and historical producers."""
        before = (
            "def source():\r\n"
            + "".join(
                f'    v{n} = "' + ("\u03b1" * 900) + '"\r\n' for n in range(12)
            )
            + "    return 1\r\n"
        )
        after = before.replace("\u03b1", "β").replace("return 1", "return 2")
        source = snap(
            {"source.py": before},
            {"source.py": after},
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
        )
        self.options = self.options.model_copy(
            update={
                "ledger": LedgerParameters(
                    max_span_bytes=4096, max_unit_spans=2
                )
            }
        )

        def observations(
            _provider: FixtureProvider, data: dict, raw: dict
        ) -> dict:
            if data["stage"] == "detail":
                for output, item in zip(
                    raw["items"],
                    (i for i in data["items"] if i["obligations"]),
                    strict=True,
                ):
                    output["observations"] = [
                        {
                            "text": f"Supplied record {n} retains its source text.",
                            "citations": [s["handle"] for s in item["spans"]],
                        }
                        for n in range(12)
                    ]
            return raw

        report = compile_snapshot(source)
        narrator, candidate, manifest = self.run_public(
            report, FixtureProvider(observations)
        )
        self.assertGreater(len(manifest.details), 10)
        self.assertTrue(manifest.plan.production.fragments)
        self.assertTrue(
            any(
                f.owner.kind == "source" and f.path == "text"
                for f in manifest.plan.production.fragments
            )
        )
        source_slices = [
            c
            for c in candidate["citations"]
            if c.get("consumption") == "source_slice"
        ]
        self.assertTrue(source_slices)
        for entry in source_slices:
            self.assertGreaterEqual(entry["line_start"], 31)
            self.assertLessEqual(entry["line_end"], 44)
        for fragment in source["fragments"]:
            fragment["start_line"] += 30
        source["meta"]["base_sha"] = "c" * 40
        source["meta"]["head_sha"] = "d" * 40
        current = compile_snapshot(source)
        provider = FixtureProvider(observations)
        _, rebound, reused = self.run_public(current, provider, name="shifted")
        self.assertEqual(provider.calls, [])
        self.assertEqual(manifest.details, reused.details)
        self.assertEqual(manifest.plan, reused.plan)
        self.assertEqual(
            [
                (c["key"], c["byte_start"], c["byte_end"])
                for c in candidate["citations"]
            ],
            [
                (c["key"], c["byte_start"], c["byte_end"])
                for c in rebound["citations"]
            ],
        )
        for old, new in zip(
            candidate["citations"], rebound["citations"], strict=True
        ):
            self.assertEqual(new["line_start"], old["line_start"] + 30)
            self.assertEqual(new["line_end"], old["line_end"] + 30)
        self.assert_shapes_bound_actual(narrator, manifest)

    def test_large_prose_batch_reserves_complete_response_envelope(
        self,
    ) -> None:
        """Many tiny valid outputs still need all JSON array separators reserved."""
        from diffstory.narrative_plan import semantic_digest
        from diffstory.narrative_v2 import InputItem
        from diffstory.narrative_v2 import ProductionInputs
        from diffstory.narrative_v2 import StageInputs
        from diffstory.narrative_v2 import Transport

        inputs = ProductionInputs(
            golden_report(1), Transport(FixtureProvider(), self.options)
        )
        outputs, items = [], []
        for number in range(300):
            key = semantic_digest("local envelope capacity fixture", number)
            target = inputs.wire.aliases.add("i", key)
            result = {
                "target": target,
                "abstraction": [statement("x")],
                "terminology": [],
            }
            outputs.append(result)
            items.append(
                InputItem(
                    key,
                    key,
                    (),
                    canonical_bytes({"target": target}),
                    len(canonical_bytes(result)),
                )
            )
        stage = StageInputs(inputs, "planning", tuple(items))
        request = stage.prepare(tuple(stage.items), repair=False, partition=0)
        response = canonical_bytes(
            {"version": "diffstory.wire.v2", "items": outputs}
        )
        # One token per serialized byte is the declared conservative allowance.
        # The older fixed 64-byte envelope margin missed these 299 commas,
        # even after adding the ordinary 256-token repair reserve.
        self.assertLessEqual(
            len(response), request.metadata.batch.output_reservation
        )


if __name__ == "__main__":
    unittest.main()


class FramedFixture(FixtureProvider):
    """Use the same adapter class with any declared closed fixture framing."""

    def __init__(self, layers: int = 0, hook: object = None) -> None:
        """Select framing before the transport captures its configuration."""
        super().__init__(hook)
        self.request_json_layers = layers

    def prepare(self, *args: object) -> bytes:
        """Wrap exact fixture bytes with the declared number of JSON layers."""
        raw = super().prepare(*args)
        for _ in range(self.request_json_layers):
            raw = canonical_bytes({"payload": raw.decode()})
        return raw

    def complete_reply(
        self, body: bytes, timeout: float, *, response_byte_limit: int
    ) -> ProviderReply:
        """Unwrap only the declared framing before returning fixture prose."""
        for _ in range(self.request_json_layers):
            body = json.loads(body)["payload"].encode()
        return super().complete_reply(
            body, timeout, response_byte_limit=response_byte_limit
        )


class CodecReplayTests(unittest.TestCase):
    """Public narration must distinguish codecs without resetting legacy budgets."""

    def setUp(self) -> None:
        """Keep every issuance and replay isolated on local storage."""
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.options = V2Options(
            packing=RequestPackingLimits(stable_partition_count=1)
        )

    def narrator(
        self,
        provider: FixtureProvider,
        *,
        state: str = "state",
        cached: bool = True,
    ) -> Narrator:
        """Construct the actual public gated narrator."""
        return Narrator(
            provider,
            architecture="v2",
            concurrency=1,
            recovery_dir=self.root / state,
            checkpoint_dir=self.root / "cache" if cached else None,
            v2_options=self.options,
        )

    def legacy(
        self, provider: FixtureProvider, report: dict, *, cached: bool = True
    ) -> tuple:
        """Issue real requests under the f9a62a6 identity, before any record exists.

        The old transport hashed this exact legacy payload for every framing.
        Only this issuance fixture selects that old formula. Runtime preparation,
        reservation, sends, artifact commits and producer runs remain actual.
        No issued record, receipt, invocation or request is relabelled.
        """
        from diffstory.narrative_plan import semantic_digest
        from diffstory.narrative_runtime import ArtifactRuntime
        from diffstory.narrative_v2 import ProductionInputs
        from diffstory.narrative_v2 import StageInputs
        from diffstory.narrative_v2 import Transport

        transport = Transport(provider, self.options)
        transport.config_digest = semantic_digest(
            "diffstory.transport.v2",
            transport.descriptor.legacy_configuration(),
        )
        inputs = ProductionInputs(report, transport)
        stage = StageInputs(inputs, "detail", inputs.detail_items())
        runtime = ArtifactRuntime(
            inputs.run,
            recovery_dir=self.root / "state",
            artifact_dir=self.root / "cache" if cached else None,
        )
        directory = self.root / "state" / "producer-runs-v2"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{inputs.run.digest}.json").write_text(
            inputs.run.model_dump_json()
        )
        invocations = tuple(
            stage.invocation(b) for b in stage.pack(tuple(stage.items))
        )
        return inputs, stage, runtime, invocations

    def issue(
        self, provider: FixtureProvider, report: dict, *, cached: bool = True
    ) -> tuple:
        """Commit actual successful legacy detail requests through the runtime."""
        inputs, stage, runtime, invocations = self.legacy(
            provider, report, cached=cached
        )
        artifacts = []
        for invocation in invocations:
            artifacts.extend(
                runtime.execute_batch(
                    invocation, stage.registry, stage.callbacks()
                ).artifacts
            )
        return inputs, stage, runtime, tuple(artifacts)

    def offline(self, report: dict, candidate: dict) -> GenerationManifest:
        """Validate, apply, reload and render with provider operations disabled."""
        from diffstory.analysis import apply_annotations
        from diffstory.render import render

        manifest = GenerationManifest.model_validate_json(
            json.dumps(candidate["generation"])
        )
        with (
            patch.object(
                FramedFixture, "prepare", side_effect=AssertionError("prepare")
            ),
            patch.object(
                FramedFixture,
                "require_credentials",
                side_effect=AssertionError("credentials"),
            ),
            patch.object(
                FramedFixture,
                "complete_reply",
                side_effect=AssertionError("send"),
            ),
            patch(
                "diffstory.narrative.OpenAIResponsesProvider.__init__",
                side_effect=AssertionError("provider"),
            ),
            patch("shutil.which", side_effect=AssertionError("executable")),
            patch("subprocess.Popen", side_effect=AssertionError("process")),
            patch("socket.socket", side_effect=AssertionError("network")),
        ):
            validate_generation(report, manifest, candidate=candidate)
            applied = apply_annotations(report, candidate)
            loaded = json.loads(json.dumps(applied))
            html = render(loaded)
            self.assertIn("Normalization", html)
        return manifest

    def test_codec_changes_with_fresh_and_retained_recovery(self) -> None:
        """New framing receives independent work under fresh and retained recovery."""
        for before, after in ((0, 1), (1, 2)):
            for retained in (False, True):
                with self.subTest(
                    before=before, after=after, retained=retained
                ):
                    # Independent directories for every public transition.
                    self.root = (
                        Path(self.temp.name) / f"{before}-{after}-{retained}"
                    )
                    report = golden_report(1)
                    first = self.narrator(FramedFixture(before)).generate(
                        report
                    )
                    provider = FramedFixture(after)
                    second = self.narrator(
                        provider, state="state" if retained else "new-state"
                    ).generate(report)
                    self.assertTrue(
                        any(
                            c["data"]["stage"] == "detail"
                            for c in provider.calls
                        )
                    )
                    left = GenerationManifest.model_validate_json(
                        json.dumps(first["generation"])
                    )
                    right = self.offline(report, second)
                    self.assertNotEqual(
                        left.request_configuration.configuration_digest,
                        right.request_configuration.configuration_digest,
                    )
                    self.assertEqual(left.details, right.details)
                    self.assertTrue(
                        {p.key for p in left.producers}.isdisjoint(
                            p.key for p in right.producers
                        )
                    )
                    warm = FramedFixture(after)
                    self.narrator(warm, state="warm-state").generate(report)
                    self.assertEqual(warm.calls, [])

    def test_genuine_legacy_layer_one_reuses_exact_originals_and_mixed_manifest(
        self,
    ) -> None:
        """Mixed original and new producers retain exact old receipts and requests."""
        from diffstory.narrative_manifest import build_receipts

        report = golden_report(1)
        inputs, _stage, runtime, artifacts = self.issue(
            FramedFixture(1), report
        )
        before = tuple(
            runtime.recovery.load(a.invocation.semantic_key) for a in artifacts
        )
        producers, receipts = build_receipts(
            artifacts,
            lambda _digest: inputs.run,
            frozenset(a.record.semantic_key for a in artifacts),
        )
        provider = FramedFixture(1)
        candidate = self.narrator(provider).generate(golden_report(2))
        manifest = self.offline(golden_report(2), candidate)
        self.assertTrue(all(p in manifest.producers for p in producers))
        self.assertTrue(all(r in manifest.receipts for r in receipts))
        self.assertEqual(
            tuple(
                runtime.recovery.load(a.invocation.semantic_key)
                for a in artifacts
            ),
            before,
        )
        old_keys = {a.lookup.item_key for a in artifacts}
        self.assertTrue(old_keys < set(manifest.unit_keys))
        # Only the new source unit receives a new detail send.
        detail_calls = [
            c for c in provider.calls if c["data"]["stage"] == "detail"
        ]
        self.assertEqual(len(detail_calls), 1)
        self.assertEqual(
            len(detail_calls[0]["data"]["items"]),
            len(set(manifest.unit_keys) - old_keys),
        )
        warm = FramedFixture(1)
        self.narrator(warm).generate(golden_report(2))
        self.assertEqual(warm.calls, [])

    def test_legacy_success_same_report_has_zero_detail_sends(self) -> None:
        """An unchanged legacy framing reuses every detail without a send."""
        report = golden_report(1)
        _inputs, _stage, _runtime, artifacts = self.issue(
            FramedFixture(1), report
        )
        provider = FramedFixture(1)
        manifest = self.offline(
            report, self.narrator(provider).generate(report)
        )
        self.assertFalse(
            any(c["data"]["stage"] == "detail" for c in provider.calls)
        )
        self.assertEqual(
            {
                r.artifact_key
                for r in manifest.receipts
                if r.result_kind == "detail"
            },
            {a.record.semantic_key for a in artifacts},
        )

    def test_reverse_legacy_layer_one_collision_keeps_origin_and_uses_closed_zero(
        self,
    ) -> None:
        """Proved historical collisions move only affected fresh work to C(0)."""
        report = golden_report(1)
        _inputs, stage, runtime, artifacts = self.issue(
            FramedFixture(1), report
        )
        original = runtime.recovery.load(artifacts[0].invocation.semantic_key)
        provider = FramedFixture(0)
        manifest = self.offline(
            report, self.narrator(provider).generate(report)
        )
        descriptor = manifest.request_configuration
        configs = {
            d.digest
            for p in manifest.producers
            if p.invocation.batch.stage == "detail"
            for d in p.invocation.dependencies
            if d.kind == "configuration"
        }
        self.assertEqual(configs, {descriptor.identity_digest("codec")})
        self.assertEqual(runtime.recovery.load(original.key), original)
        self.assertNotEqual(
            stage.static,
            next(
                p.invocation.lookups[0].static_dependencies
                for p in manifest.producers
                if p.invocation.batch.stage == "detail"
            ),
        )
        warm = FramedFixture(0)
        self.narrator(warm).generate(report)
        self.assertEqual(warm.calls, [])

    def test_legacy_cacheless_exhaustion_and_missing_results_never_get_new_allowance(
        self,
    ) -> None:
        """Completed or exhausted legacy work retains its original durable limit."""
        for mode in ("exhausted", "missing"):
            with self.subTest(mode=mode):
                self.root = Path(self.temp.name) / mode
                report = golden_report(1)
                provider = FramedFixture(
                    1,
                    (lambda _p, _d, _r: b"{") if mode == "exhausted" else None,
                )
                _inputs, stage, runtime, invocations = self.legacy(
                    provider, report, cached=False
                )
                if mode == "exhausted":
                    with self.assertRaises(ArtifactRuntimeError):
                        runtime.execute_batch(
                            invocations[0], stage.registry, stage.callbacks()
                        )
                else:
                    runtime.execute_batch(
                        invocations[0], stage.registry, stage.callbacks()
                    )
                old = runtime.recovery.load(invocations[0].semantic_key)
                fresh = FramedFixture(1)
                with self.assertRaises(RuntimeStageError) as caught:
                    self.narrator(fresh, cached=False).generate(report)
                self.assertEqual(fresh.calls, [])
                self.assertEqual(runtime.recovery.load(old.key), old)
                if mode == "missing":
                    self.assertEqual(
                        caught.exception.diagnostic.reason,
                        "results_unavailable",
                    )

    def test_legacy_partial_and_interrupted_transient_keep_original_remaining_limits(
        self,
    ) -> None:
        """Replay consumes only the original remaining reservations."""
        for mode in ("initial", "partial", "transient"):
            with self.subTest(mode=mode):
                self.root = Path(self.temp.name) / mode
                report = golden_report(2)

                def fault(
                    provider: FixtureProvider,
                    _data: dict,
                    response: dict,
                    mode: str = mode,
                ) -> dict:
                    if mode == "partial" and len(provider.calls) == 1:
                        response["items"].pop()
                        return response
                    if mode == "transient" and len(provider.calls) == 1:
                        msg = "transport"
                        raise ArtifactCallError(msg)
                    raise KeyboardInterrupt

                provider = FramedFixture(1, fault)
                _inputs, stage, runtime, invocations = self.legacy(
                    provider, report
                )
                with self.assertRaises(KeyboardInterrupt):
                    runtime.execute_batch(
                        invocations[0], stage.registry, stage.callbacks()
                    )
                old = runtime.recovery.load(invocations[0].semantic_key)
                resumed = FramedFixture(1)
                if mode == "transient":
                    with self.assertRaises(RuntimeStageError):
                        self.narrator(resumed).generate(report)
                    self.assertEqual(resumed.calls, [])
                    self.assertEqual(
                        runtime.recovery.load(old.key).consumed_attempts,
                        old.consumed_attempts,
                    )
                    continue
                manifest = self.offline(
                    report, self.narrator(resumed).generate(report)
                )
                new = runtime.recovery.load(old.key)
                self.assertEqual(new.invocation, old.invocation)
                self.assertEqual(new.policy, old.policy)
                self.assertEqual(
                    new.consumed_attempts, old.consumed_attempts + 1
                )
                detail = [
                    c for c in resumed.calls if c["data"]["stage"] == "detail"
                ]
                self.assertEqual(len(detail), 1)
                if mode != "transient":
                    self.assertEqual(detail[0], provider.calls[-1])
                self.assertEqual(
                    {
                        r.item_key
                        for r in manifest.receipts
                        if r.result_kind == "detail"
                    },
                    set(manifest.unit_keys),
                )

    def test_both_legacy_candidate_orders_and_internal_recovery_are_audited(
        self,
    ) -> None:
        """Another framing cannot hide a later compatible record in any cache path."""
        from diffstory.narrative_runtime import ArtifactRuntime
        from diffstory.narrative_v2 import CodecReplay
        from diffstory.narrative_v2 import ProductionInputs
        from diffstory.narrative_v2 import StageInputs
        from diffstory.narrative_v2 import Transport

        report = golden_report(1)
        self.root = Path(self.temp.name) / "zero"
        _i, _s, _r, other = self.issue(FramedFixture(0), report)
        self.root = Path(self.temp.name) / "one"
        inputs, stage, runtime, current = self.issue(FramedFixture(1), report)
        for artifact in other:
            runtime.artifacts.commit(artifact)
        live = ProductionInputs(
            report, Transport(FramedFixture(1), self.options)
        )
        replay = CodecReplay(StageInputs(live, "detail", live.detail_items()))
        view = replay.legacy

        def audit(artifact: object) -> bool:
            return (
                replay.select(
                    artifact.invocation, (artifact.production_request,)
                )
                is not None
            )

        native_sorted = sorted
        for reverse in (False, True):

            def ordered(
                values: object,
                *args: object,
                reverse: bool = reverse,
                **kwargs: object,
            ) -> list:
                result = native_sorted(values, *args, **kwargs)
                return (
                    list(reversed(result))
                    if reverse and result and isinstance(result[0], Path)
                    else result
                )

            with patch(
                "diffstory.narrative_runtime.sorted",
                side_effect=ordered,
                create=True,
            ):
                for artifact in current:
                    found = runtime.artifacts.lookup(
                        artifact.lookup,
                        view.registry,
                        view.validate,
                        audit_candidate=audit,
                    )
                    self.assertEqual(found, artifact)
                for supplied in ((), (*other, *current), (*current, *other)):
                    state = (
                        self.root
                        / f"internal-{reverse}-{len(supplied)}-{bool(supplied and supplied[0] == current[0])}"
                    )
                    fresh = ArtifactRuntime(
                        inputs.run,
                        recovery_dir=state,
                        artifact_dir=self.root / "cache",
                    )
                    outcome = fresh.execute_batch(
                        current[0].invocation,
                        stage.registry,
                        stage.callbacks(),
                        available=supplied,
                        audit_candidate=audit,
                    )
                    self.assertEqual(outcome.artifacts, current)
                    self.assertEqual(fresh.usage.report()["calls"], 0)
                for recovery in (f"public-{reverse}", "state"):
                    provider = FramedFixture(1)
                    candidate = self.narrator(
                        provider, state=recovery
                    ).generate(report)
                    self.offline(report, candidate)
                    self.assertFalse(
                        any(
                            c["data"]["stage"] == "detail"
                            for c in provider.calls
                        )
                    )

    def test_damaged_legacy_requests_fail_closed_before_any_new_send(  # noqa: C901 # Independent corruption dimensions.
        self,
    ) -> None:
        """Rehashed metadata and attempt corruption cannot prove another codec."""
        from diffstory.narrative_plan import content_digest

        for mutation in (
            "request",
            "bound",
            "attempt",
            "policy",
            "peer",
            "context",
            "run",
        ):
            with self.subTest(mutation=mutation):
                self.root = Path(self.temp.name) / mutation
                report = golden_report(1)
                _inputs, _stage, runtime, artifacts = self.issue(
                    FramedFixture(1), report
                )
                original = artifacts[0]
                path = (
                    runtime.recovery.directory
                    / f"{original.invocation.semantic_key}.json"
                )
                saved = json.loads(path.read_text())
                if mutation == "request":
                    saved["state"]["invocation"]["batch"]["request_digest"] = (
                        "a" * 64
                    )
                elif mutation == "bound":
                    saved["state"]["invocation"]["batch"]["input_bound"] += 1
                elif mutation == "attempt":
                    saved["state"]["nodes"][0]["attempts"][0]["prepared"][
                        "overhead_bytes"
                    ] += 1
                elif mutation == "policy":
                    saved["state"]["policy"]["total_attempts"] = 120
                elif mutation == "peer":
                    saved["state"]["invocation"]["batch"]["item_keys"] = []
                elif mutation == "context":
                    for dependency in saved["state"]["invocation"][
                        "dependencies"
                    ]:
                        if dependency["kind"] == "context":
                            dependency["digest"] = "a" * 64
                else:
                    run = (
                        self.root
                        / "state"
                        / "producer-runs-v2"
                        / f"{original.original_run_digest}.json"
                    )
                    value = json.loads(run.read_text())
                    value["model"] = "wrong-model"
                    run.write_text(json.dumps(value))
                if mutation != "run":
                    saved["digest"] = content_digest(
                        canonical_bytes(saved["state"])
                    )
                    path.write_text(json.dumps(saved))
                provider = FramedFixture(0)
                with self.assertRaises(
                    (ArtifactRuntimeError, RuntimeStageError)
                ):
                    self.narrator(provider).generate(report)
                self.assertEqual(provider.calls, [])

    def test_memoization_binds_complete_requests_and_current_payload(
        self,
    ) -> None:
        """An exact audit can be reused; a rehashed bound or payload cannot."""
        from dataclasses import replace
        from types import MappingProxyType

        from diffstory.narrative_runtime import PreparedMetadata
        from diffstory.narrative_v2 import CodecReplay
        from diffstory.narrative_v2 import ProductionInputs
        from diffstory.narrative_v2 import StageInputs
        from diffstory.narrative_v2 import Transport

        report = golden_report(1)
        _i, _s, _r, artifacts = self.issue(FramedFixture(1), report)
        inputs = ProductionInputs(
            report, Transport(FramedFixture(1), self.options)
        )
        replay = CodecReplay(
            StageInputs(inputs, "detail", inputs.detail_items())
        )
        artifact = artifacts[0]
        with (
            patch.object(
                replay.legacy, "reconstruct", wraps=replay.legacy.reconstruct
            ) as reconstruct,
            patch.object(
                replay.stage,
                "identity_view",
                side_effect=AssertionError("registry rebuilt per candidate"),
            ),
        ):
            for _ in range(3):
                replay.required(
                    artifact.invocation, (artifact.production_request,)
                )
            self.assertEqual(reconstruct.call_count, 1)
            data = artifact.production_request.model_dump()
            data["safety_tokens"] += 1
            altered = PreparedMetadata.model_validate(data)
            with self.assertRaisesRegex(
                ArtifactRuntimeError, "request_mismatch"
            ):
                replay.required(artifact.invocation, (altered,))
        original_limits = replay.legacy.transport.limits
        replay.legacy.transport.limits = RequestPackingLimits(
            **(
                original_limits.model_dump()
                | {"safety_tokens": original_limits.safety_tokens + 1}
            )
        )
        with self.assertRaises(ArtifactRuntimeError):
            replay.required(
                artifact.invocation, (artifact.production_request,)
            )
        replay.legacy.transport.limits = original_limits
        key = artifact.lookup.item_key
        items = dict(replay.legacy.items)
        items[key] = replace(
            items[key],
            payload=items[key].payload.replace(b"rstrip", b"lstrip"),
        )
        replay.legacy.items = MappingProxyType(items)
        with self.assertRaises(ArtifactRuntimeError):
            replay.required(
                artifact.invocation, (artifact.production_request,)
            )

    def test_zero_protocol_or_layer_mutation_after_preview_stops_before_dispatch(
        self,
    ) -> None:
        """Approval includes zero framing and the full known protocol tuple."""
        report = golden_report(1)
        for name, value in (
            ("request_json_layers", 1),
            ("request_protocol", "unknown"),
        ):
            provider = FramedFixture(0)
            narrator = self.narrator(provider)
            narrator.preview(report)
            setattr(provider, name, value)
            with self.assertRaisesRegex(
                ArtifactRuntimeError, "dependencies_changed"
            ):
                narrator.generate(report)
            self.assertEqual(provider.calls, [])

    def test_reverse_collision_retains_compatible_zero_originals_in_one_stage(
        self,
    ) -> None:
        """Mixed L histories retain compatible peers and assign C(0) only to misses."""
        import shutil

        from diffstory.narrative_manifest import build_receipts

        self.root = Path(self.temp.name) / "zero"
        inputs, _stage, zero_runtime, zeros = self.issue(
            FramedFixture(0), golden_report(1)
        )
        zero_state = zero_runtime.recovery.load(
            zeros[0].invocation.semantic_key
        )
        self.root = Path(self.temp.name) / "one"
        _inputs, _stage, one_runtime, ones = self.issue(
            FramedFixture(1), golden_report(2)
        )
        one_state = one_runtime.recovery.load(ones[0].invocation.semantic_key)
        for artifact in zeros:
            one_runtime.artifacts.commit(artifact)
        for suffix in (".claim", ".json"):
            shutil.copyfile(
                zero_runtime.recovery.directory / f"{zero_state.key}{suffix}",
                one_runtime.recovery.directory / f"{zero_state.key}{suffix}",
            )
        directory = self.root / "state" / "producer-runs-v2"
        (directory / f"{inputs.run.digest}.json").write_text(
            inputs.run.model_dump_json()
        )
        providers, receipts = build_receipts(
            zeros,
            lambda _digest: inputs.run,
            frozenset(a.record.semantic_key for a in zeros),
        )
        provider = FramedFixture(0)
        candidate = self.narrator(provider).generate(golden_report(2))
        manifest = self.offline(golden_report(2), candidate)
        self.assertTrue(all(p in manifest.producers for p in providers))
        self.assertTrue(all(r in manifest.receipts for r in receipts))
        self.assertEqual(one_runtime.recovery.load(zero_state.key), zero_state)
        self.assertEqual(one_runtime.recovery.load(one_state.key), one_state)
        calls = [c for c in provider.calls if c["data"]["stage"] == "detail"]
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(calls[0]["data"]["items"]), 1)
        warm = FramedFixture(0)
        self.narrator(warm).generate(golden_report(2))
        self.assertEqual(warm.calls, [])

    def test_damaged_bound_codec_and_production_metadata_are_not_independence(
        self,
    ) -> None:
        """Both closed identities and legacy cache-only metadata require byte proof."""
        from diffstory.narrative_plan import content_digest
        from diffstory.narrative_runtime import StoredArtifact

        report = golden_report(1)
        for source in ("bound-history", "legacy-artifact"):
            with self.subTest(source=source):
                self.root = Path(self.temp.name) / source
                if source == "bound-history":
                    self.narrator(FramedFixture(1)).generate(report)
                    for path in (self.root / "state" / "recovery-v2").glob(
                        "*.json"
                    ):
                        value = json.loads(path.read_text())
                        if (
                            value["state"]["invocation"]["batch"]["stage"]
                            == "detail"
                        ):
                            value["state"]["nodes"][0]["attempts"][0][
                                "prepared"
                            ]["safety_tokens"] += 1
                            value["digest"] = content_digest(
                                canonical_bytes(value["state"])
                            )
                            path.write_text(json.dumps(value))
                else:
                    _i, _s, runtime, artifacts = self.issue(
                        FramedFixture(1), report
                    )
                    for path in runtime.artifacts.directory.rglob("*.json"):
                        path.unlink()
                    for artifact in artifacts:
                        data = artifact.model_dump()
                        data["production_request"]["overhead_bytes"] += 1
                        runtime.artifacts.commit(
                            StoredArtifact.model_validate(data)
                        )
                provider = FramedFixture(2)
                with self.assertRaises(
                    (ArtifactRuntimeError, RuntimeStageError)
                ):
                    self.narrator(
                        provider,
                        state="fresh"
                        if source == "legacy-artifact"
                        else "state",
                    ).generate(report)
                self.assertEqual(provider.calls, [])

    def test_legacy_completed_transient_audits_both_requests_without_detail_sends(
        self,
    ) -> None:
        """Original transient bytes and spent reservations survive successful reuse."""
        report = golden_report(1)

        def fault(
            provider: FixtureProvider, _data: dict, response: dict
        ) -> dict:
            if len(provider.calls) == 1:
                msg = "rate_limit"
                raise ArtifactCallError(msg)
            return response

        provider = FramedFixture(1, fault)
        _i, _s, runtime, artifacts = self.issue(provider, report)
        old = runtime.recovery.load(artifacts[0].invocation.semantic_key)
        self.assertEqual(old.consumed_attempts, 2)
        self.assertEqual(
            old.nodes[0].attempts[0].prepared,
            old.nodes[0].attempts[1].prepared,
        )
        fresh = FramedFixture(1)
        candidate = self.narrator(fresh).generate(report)
        self.offline(report, candidate)
        self.assertFalse(
            any(c["data"]["stage"] == "detail" for c in fresh.calls)
        )
        self.assertEqual(runtime.recovery.load(old.key), old)

    def test_same_legacy_key_retains_complete_zero_results_without_touching_other_history(
        self,
    ) -> None:
        """Exact complete zero originals survive a proved layer-one L state collision."""
        from diffstory.narrative_manifest import build_receipts

        report = golden_report(2)
        self.root = Path(self.temp.name) / "zero"
        inputs, _stage, _runtime, zeros = self.issue(FramedFixture(0), report)
        self.root = Path(self.temp.name) / "one"
        _inputs, _stage, runtime, ones = self.issue(FramedFixture(1), report)
        old = runtime.recovery.load(ones[0].invocation.semantic_key)
        self.assertEqual(old.key, zeros[0].invocation.semantic_key)
        self.assertNotEqual(old.invocation, zeros[0].invocation)
        for artifact in zeros:
            runtime.artifacts.commit(artifact)
        producers, receipts = build_receipts(
            zeros,
            lambda _digest: inputs.run,
            frozenset(a.record.semantic_key for a in zeros),
        )
        provider = FramedFixture(0)
        candidate = self.narrator(provider).generate(report)
        manifest = self.offline(report, candidate)
        self.assertTrue(all(p in manifest.producers for p in producers))
        self.assertTrue(all(r in manifest.receipts for r in receipts))
        self.assertFalse(
            any(c["data"]["stage"] == "detail" for c in provider.calls)
        )
        self.assertEqual(runtime.recovery.load(old.key), old)
        warm = FramedFixture(0)
        self.narrator(warm).generate(report)
        self.assertEqual(warm.calls, [])
        # Losing one member of the complete original must not grant a new L
        # or C(0) attempt while the other codec owns the durable key.
        missing = zeros[-1]
        for path in (runtime.artifacts.directory / missing.lookup.key).glob(
            "*.json"
        ):
            value = runtime.artifacts.read(path.read_bytes())
            if (
                value.production_request.batch.request_digest
                == missing.production_request.batch.request_digest
            ):
                path.unlink()
        failed = FramedFixture(0)
        with self.assertRaisesRegex(
            ArtifactRuntimeError, "results_unavailable"
        ):
            self.narrator(failed).generate(report)
        self.assertEqual(failed.calls, [])
        self.assertEqual(runtime.recovery.load(old.key), old)
