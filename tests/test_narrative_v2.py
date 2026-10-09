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
    return {
        "kind": "implementation_claim" if claims or citations else "editorial",
        "text": text,
        "claims": claims,
        "citations": citations,
        "effect": None,
        "uncertainty": [],
    }


class FixtureProvider:
    """Return authored normalization prose and retain only local fixture requests."""

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
        from diffstory.narrative_plan import content_digest

        report = golden_report()
        self.narrator(FixtureProvider()).generate(report)
        changed = False
        for path in (self.root / "cache" / "artifacts-v2").glob("*/*.json"):
            data = json.loads(path.read_bytes())
            if data["lookup"]["result_kind"] == "detail":
                data["record"]["result"]["explanations"][0]["citations"][0][
                    "content_digest"
                ] = "0" * 64
                raw = canonical_bytes(data)
                path.with_name(content_digest(raw) + ".json").write_bytes(raw)
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
        self.assertFalse((self.root / "cache" / "artifacts-v2").exists())

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


if __name__ == "__main__":
    unittest.main()
