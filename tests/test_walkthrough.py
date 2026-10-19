"""Exercise the one-request walkthrough through public generation boundaries."""

import copy
import json
import threading
import time
import unittest
from pathlib import Path

from test_analysis import snap

from diffstory.analysis import compile_snapshot
from diffstory.compact_packet import build_packet
from diffstory.compact_packet import canonical_bytes
from diffstory.compact_packet import packet_identity
from diffstory.render import render
from diffstory.walkthrough import WalkthroughNarrator
from diffstory.walkthrough import apply_walkthrough_annotations
from diffstory.walkthrough import validate_story
from diffstory.walkthrough import validate_walkthrough
from diffstory.walkthrough import walkthrough_annotations


def fixture_story(packet: dict) -> dict:
    """Return synthetic prose for structural tests without claiming source fidelity."""
    passage_ids = ["passage"]
    return {
        "preamble": (
            "This tool compares two saved versions of code and links its "
            "explanations to the original text.\n\nA definition is a named unit, "
            "such as a function. A comparison shows what changed between "
            "the two versions."
        ),
        "introduction": "Inspect the supplied changes and their source.",
        "closing": "The source remains available for review.",
        "quick_summary": {
            "before": "The earlier version has the supplied prior behavior.",
            "after": "The later version contains the supplied change.",
            "passage_ids": passage_ids,
        },
        "impact": None,
        "route": [{"kind": "mechanism", "passage_ids": passage_ids}],
        "author_reported_checks": [],
        "review_checks": [],
        "sections": [
            {
                "id": "concept",
                "title": "Related changes",
                "summary": "Synthetic structural fixture.",
                "passages": [
                    {
                        "id": "passage",
                        "text": "Review these related edits together.",
                        "change_ids": [c["id"] for c in packet["changes"]],
                        "region_ids": [r["id"] for r in packet["regions"]],
                        "citations": [
                            ref
                            for r in packet["regions"]
                            for ref in r["citations"]
                        ],
                    }
                ],
            }
        ],
    }


class FixtureProvider:
    """Count requests to a deterministic offline structured-output provider.

    Attributes:
        calls: Completed synthetic calls, initially zero.
        delay: Simulated provider seconds; no network or subprocess is used.
        timeout: Last remaining deadline received by the adapter.
    """

    name = "fixture"
    model = "structural-fixture-v1"
    destination = "fixture://offline"
    context_tokens = 3_000_000
    max_output_tokens = 128_000
    input_overhead_bytes = 0

    def __init__(self, delay: float = 0) -> None:
        """Start one independent synthetic provider with optional call delay."""
        self.calls = 0
        self.delay = delay
        self.timeout = None
        self.lock = threading.Lock()
        self.active = self.max_active = 0
        self.written_prose = []
        self.editor_request = None
        self.drop_editor_passage = False
        self.part_barrier = None
        self.editor_mutation = None
        self.part_text_override = None
        self.include_review_refs = False

    def prepare(
        self,
        system: str,
        data: dict,
        schema_name: str,
        schema: dict,
        max_output_tokens: int,
    ) -> bytes:
        """Serialize a complete local request with all framing and schema."""
        return canonical_bytes(
            {
                "system": system,
                "data": data,
                "schema_name": schema_name,
                "schema": schema,
                "max_output_tokens": max_output_tokens,
            }
        )

    def complete(
        self, body: bytes, timeout: float | None
    ) -> tuple[dict, dict]:
        """Return deterministic source-linked prose and known usage once."""
        with self.lock:
            self.calls += 1
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.timeout = timeout
        try:
            request = json.loads(body)
            if (
                self.part_barrier is not None
                and request["schema_name"] == "walkthrough_part"
            ):
                self.part_barrier.wait(timeout=5)
            time.sleep(self.delay)
            data = request["data"]
            if request["schema_name"] == "walkthrough_editor":
                self.editor_request = body
                ids = [
                    passage["id"]
                    for writer in data["writers"]
                    for section in writer["sections"]
                    for passage in section["passages"]
                ]
                available_test_ids = list(
                    dict.fromkeys(
                        test_id
                        for writer in data["writers"]
                        for section in writer["sections"]
                        for passage in section["passages"]
                        for test_id in passage["test_ids"]
                    )
                )
                if self.drop_editor_passage:
                    ids.pop()
                if self.editor_mutation == "duplicate":
                    ids[-1] = ids[0]
                elif self.editor_mutation == "unknown":
                    ids[-1] = "p-unknown"
                impact = (
                    {
                        "text": "The edited source changes the reviewed behavior.",
                        "passage_ids": ids[:2],
                    }
                    if self.include_review_refs
                    else None
                )
                result = {
                    "preamble": "This system compares code versions. A change is an edit between them.",
                    "introduction": "Read the supplied related mechanisms.",
                    "closing": "Check the cited source.",
                    "quick_summary": {
                        "before": "The earlier behavior follows the old mechanism.",
                        "after": "The edited mechanism explains the supplied change.",
                        "passage_ids": ids[:2] or ids,
                    },
                    "impact": impact,
                    "route": [
                        {
                            "kind": "mechanism",
                            "passage_ids": ids or ["p-missing"],
                        }
                    ],
                    "author_reported_checks": [],
                    "review_checks": (
                        [
                            {
                                "text": "Confirm the remaining producer cases.",
                                "passage_ids": ids[:2],
                                "test_ids": available_test_ids[:4],
                            }
                        ]
                        if self.include_review_refs
                        else []
                    ),
                    "sections": [
                        {
                            "title": "Connected mechanisms",
                            "summary": "Editorial reading connection.",
                            "groups": [
                                {
                                    "passage_ids": list(reversed(ids)),
                                    "text": "One edited explanation connects the supplied mechanisms.",
                                }
                            ],
                        }
                    ],
                }
            else:
                result = fixture_story(data)
                if request["schema_name"] == "walkthrough_part":
                    result = {
                        "orientation": "This subsystem compares saved code versions and links changes to their source.",
                        "sections": result["sections"],
                    }
                    passage = result["sections"][0]["passages"][0]
                    passage["text"] = self.part_text_override or (
                        "Detailed explanation for "
                        + ", ".join(passage["change_ids"])
                    )
                    with self.lock:
                        self.written_prose.append(passage["text"])
            return result, {
                "input_tokens": 100,
                "cached_input_tokens": 0,
                "output_tokens": 50,
            }
        finally:
            with self.lock:
                self.active -= 1


class ResponsesFixtureProvider(FixtureProvider):
    """Use the Responses API wire nesting with the deterministic local fixture."""

    name = "openai"

    def prepare(
        self,
        system: str,
        data: dict,
        schema_name: str,
        schema: dict,
        max_output_tokens: int,
    ) -> bytes:
        """Serialize the same nested user JSON string used by Responses."""
        request = {
            "model": self.model,
            "store": False,
            "max_output_tokens": max_output_tokens,
            "input": [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": json.dumps(
                        data, ensure_ascii=False, separators=(",", ":")
                    ),
                },
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "strict": True,
                    "schema": schema,
                }
            },
        }
        return json.dumps(
            request, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")

    def complete(
        self, body: bytes, timeout: float | None
    ) -> tuple[dict, dict]:
        """Decode Responses framing before using the shared fixture behavior."""
        request = json.loads(body)
        adapted = canonical_bytes(
            {
                "system": request["input"][0]["content"],
                "data": json.loads(request["input"][1]["content"]),
                "schema_name": request["text"]["format"]["name"],
                "schema": request["text"]["format"]["schema"],
            }
        )
        return super().complete(adapted, timeout)


class WalkthroughTests(unittest.TestCase):
    """Protect source links, exact coverage, and complete-state boundaries."""

    def test_public_annotation_application_rejects_retired_report_markers(
        self,
    ) -> None:
        """Unsupported compiler markers must not be projected into a new report."""
        compiler_report = compile_snapshot(snap({}, {"a.py": "x = 1\n"}))
        generated = WalkthroughNarrator(FixtureProvider()).generate(
            compiler_report
        )
        annotations = walkthrough_annotations(generated)

        applied = apply_walkthrough_annotations(compiler_report, annotations)
        self.assertEqual(applied["schema"], "diffstory.report.v2")
        for invalid_report in (None, [], {"schema": "diffstory.report.v0"}):
            with (
                self.subTest(invalid_report=invalid_report),
                self.assertRaises(ValueError),
            ):
                build_packet(invalid_report)

        markers = [
            {"storage_version": "diffstory.report-storage.v1"},
            {"artifact_kind": "stored-report"},
            {"root": {}, "tables": {}},
            {"generation": {"version": "diffstory.generation.v1"}},
            {"narrative_v2": {"schema": "diffstory.presentation.v2"}},
        ]
        for marker in markers:
            with self.subTest(marker=marker):
                old_report = copy.deepcopy(compiler_report)
                old_report.update(marker)
                with self.assertRaises(ValueError):
                    apply_walkthrough_annotations(old_report, annotations)

    def test_demo_generates_and_reloads_in_one_request(self) -> None:
        """The bundled multi-file demo must not trigger per-group writing calls."""
        cases = [
            Path(__file__).resolve().parents[1] / "examples/demo.report.json"
        ]
        for path in cases:
            with self.subTest(fixture=path.name):
                provider = FixtureProvider()
                narrator = WalkthroughNarrator(provider)
                report = narrator.generate(json.loads(path.read_text()))
                validate_walkthrough(json.loads(canonical_bytes(report)))
                self.assertEqual(provider.calls, 1)
                self.assertEqual(report["generation"]["state"], "complete")
                self.assertEqual(
                    report["generation"]["usage"]["input_tokens"], 100
                )
                self.assertGreater(provider.timeout, 0)
                self.assertLessEqual(provider.timeout, 240)

    def test_missing_changes_regions_and_invalid_citations_fail(self) -> None:
        """Plausible prose cannot hide omitted edits or cite nonexistent source."""
        packet = build_packet(
            compile_snapshot(snap({}, {"a.py": "x = 1\n", "b.py": "y = 2\n"}))
        )
        cases = ["change", "region", "revision", "range", "unsupported"]
        for mutation in cases:
            with self.subTest(mutation=mutation):
                story = fixture_story(packet)
                passage = story["sections"][0]["passages"][0]
                if mutation == "change":
                    passage["change_ids"].pop()
                elif mutation == "region":
                    passage["region_ids"].pop()
                elif mutation == "revision":
                    passage["citations"][0]["revision"] = "a" * 40
                elif mutation == "range":
                    passage["citations"][0]["end"] = 100
                else:
                    passage["citations"] = passage["citations"][-1:]
                with self.assertRaises(ValueError):
                    validate_story(packet, story)

    def test_preamble_keeps_separate_prose_and_sketch_allowances(self) -> None:
        """Reject blank/oversized orientation without losing valid prose or sketches."""
        packet = build_packet(compile_snapshot(snap({}, {"a.py": "x = 1\n"})))
        cases = [
            (
                "This tool compares saved code.\n\nA definition is a named unit.",
                True,
            ),
            ("a" * 4_000, True),
            ("a" * 4_000 + "\n```text\n" + "b" * 2_000 + "\n```", True),
            ("a" * 3_998 + "\n\n```text\n" + "b" * 2_000 + "\n```", True),
            ("", False),
            (" \n\t", False),
            ("a" * 4_001, False),
            ("Overview\n\n```text\n" + "b" * 4_000 + "\n```", False),
        ]
        for preamble, accepted in cases:
            with self.subTest(length=len(preamble), accepted=accepted):
                story = fixture_story(packet)
                story["preamble"] = preamble
                if accepted:
                    self.assertEqual(
                        validate_story(packet, story)["preamble"], preamble
                    )
                else:
                    with self.assertRaises(ValueError):
                        validate_story(packet, story)

    def test_only_current_generation_and_story_contract_are_accepted(
        self,
    ) -> None:
        """Archived markers and stories without current editorial fields fail."""
        report = WalkthroughNarrator(FixtureProvider()).generate(
            compile_snapshot(snap({}, {"a.py": "x = 1\n"}))
        )
        self.assertEqual(
            report["generation"]["version"],
            "diffstory.generation.walkthrough.v3",
        )
        self.assertEqual(
            report["story"]["quick_summary"]["before"],
            "The earlier version has the supplied prior behavior.",
        )
        for version in (
            "diffstory.generation.walkthrough.v1",
            "diffstory.generation.walkthrough.v2",
        ):
            with self.subTest(version=version):
                saved = copy.deepcopy(report)
                saved["generation"]["version"] = version
                if version.endswith("v1"):
                    saved["story"].pop("preamble")
                for field in (
                    "quick_summary",
                    "impact",
                    "route",
                    "author_reported_checks",
                    "review_checks",
                ):
                    saved["story"].pop(field)
                with self.assertRaises(ValueError):
                    validate_walkthrough(saved)
                with self.assertRaises(ValueError):
                    render(saved)

        current = copy.deepcopy(report)
        current["story"].pop("quick_summary")
        with self.assertRaises(ValueError):
            validate_walkthrough(current)

    def test_editorial_checks_are_exact_and_passage_grounded(self) -> None:
        """Reports must not invent author quotes, references, or test relevance."""
        before = "def test_existing():\n    assert True\n"
        report = compile_snapshot(
            snap(
                {"tests/test_existing.py": before},
                {"tests/test_existing.py": before, "app.py": "value = 1\n"},
                description="Author reports: local checks passed.",
            )
        )
        packet = build_packet(report)
        story = fixture_story(packet)
        story["author_reported_checks"] = [
            "Author reports: local checks passed."
        ]
        story["review_checks"] = [
            {
                "text": "Confirm the reported test list matches CI.",
                "passage_ids": ["passage"],
                "test_ids": [],
            }
        ]
        validate_story(packet, story)

        mutations = []
        fabricated_author = copy.deepcopy(story)
        fabricated_author["author_reported_checks"] = [
            "The tests passed on every platform."
        ]
        mutations.append(fabricated_author)
        dangling_review = copy.deepcopy(story)
        dangling_review["review_checks"][0]["passage_ids"] = ["missing"]
        mutations.append(dangling_review)
        unrelated_test = copy.deepcopy(story)
        unrelated_test["review_checks"][0]["test_ids"] = [
            packet["analysis"]["tests"][0]["id"]
        ]
        mutations.append(unrelated_test)
        for candidate in mutations:
            with (
                self.subTest(candidate=candidate["review_checks"]),
                self.assertRaises(ValueError),
            ):
                validate_story(packet, candidate)

        test_before = "def test_value():\n    assert value() == 1\n"
        test_packet = build_packet(
            compile_snapshot(
                snap(
                    {"tests/test_value.py": test_before},
                    {
                        "tests/test_value.py": test_before.replace(
                            "== 1", "== 2"
                        )
                    },
                )
            )
        )
        related = fixture_story(test_packet)
        related["review_checks"] = [
            {
                "text": "Confirm this changed assertion matches the intended result.",
                "passage_ids": ["passage"],
                "test_ids": [test_packet["analysis"]["tests"][0]["id"]],
            }
        ]
        validate_story(test_packet, related)

    def test_v3_allows_honest_absence_and_rejects_dangling_summary_routes(
        self,
    ) -> None:
        """Unknown impact and review targets stay absent instead of implying proof."""
        packet = build_packet(compile_snapshot(snap({}, {"a.py": "x = 1\n"})))
        story = fixture_story(packet)
        story["impact"] = None
        story["author_reported_checks"] = []
        self.assertEqual(validate_story(packet, story)["impact"], None)
        mutations = []
        dangling = copy.deepcopy(story)
        dangling["quick_summary"]["passage_ids"] = ["missing"]
        mutations.append(dangling)
        missing_mechanism = copy.deepcopy(story)
        missing_mechanism["route"] = [
            {"kind": "tests", "passage_ids": ["passage"]}
        ]
        mutations.append(missing_mechanism)
        blank_summary = copy.deepcopy(story)
        blank_summary["quick_summary"]["before"] = "  "
        mutations.append(blank_summary)
        for candidate in mutations:
            with (
                self.subTest(candidate=candidate["quick_summary"]),
                self.assertRaises(ValueError),
            ):
                validate_story(packet, candidate)

    def test_annotation_limit_rejects_full_story_without_trimming(
        self,
    ) -> None:
        """A story over one MiB must fail instead of losing passage prose."""
        packet = build_packet(compile_snapshot(snap({}, {"a.py": "x = 1\n"})))
        story = fixture_story(packet)
        original = story["sections"][0]["passages"][0]
        story["sections"][0]["passages"] = [
            {
                **copy.deepcopy(original),
                "id": "passage" if index == 0 else f"passage-{index}",
                "text": "x" * 12_000,
            }
            for index in range(90)
        ]
        with self.assertRaisesRegex(ValueError, "1 MiB"):
            validate_story(packet, story)

    def test_oversized_request_is_rejected_without_source_transfer(
        self,
    ) -> None:
        """Local preflight must reject a request before any provider call."""
        report = compile_snapshot(snap({}, {"a.py": "x = 1\n"}))
        for capacity in [1_000, 128_000]:
            with self.subTest(capacity=capacity):
                provider = FixtureProvider()
                provider.context_tokens = capacity
                with self.assertRaises(ValueError):
                    WalkthroughNarrator(provider).generate(report)
                self.assertEqual(provider.calls, 0)

    def test_unchanged_context_does_not_trigger_a_model_request(self) -> None:
        """An unchanged comparison must stop before sending context to a model."""
        source = "def stable():\n    return 1\n"
        report = compile_snapshot(snap({"a.py": source}, {"a.py": source}))
        provider = FixtureProvider()
        with self.assertRaisesRegex(ValueError, "No supplied changes"):
            WalkthroughNarrator(provider).generate(report)
        self.assertEqual(provider.calls, 0)

    def test_late_provider_result_never_becomes_complete(self) -> None:
        """A provider ignoring its timeout cannot produce a complete late report."""
        report = compile_snapshot(snap({}, {"a.py": "x = 1\n"}))
        delay, deadline = (0.35, 0.2)
        provider = FixtureProvider(delay)
        with self.assertRaises(TimeoutError):
            WalkthroughNarrator(provider, deadline_seconds=deadline).generate(
                report
            )
        self.assertEqual(provider.calls, 1)

    def test_saved_report_does_not_trust_rebound_evidence(self) -> None:
        """Changing metadata or completion state must invalidate saved narration."""
        report = compile_snapshot(snap({}, {"a.py": "x = 1\n"}))
        generated = WalkthroughNarrator(FixtureProvider()).generate(report)
        for field, value in [
            ("state", "failed"),
            ("evidence_digest", "0" * 64),
            ("head_sha", "c" * 40),
        ]:
            with self.subTest(field=field):
                changed = copy.deepcopy(generated)
                changed["generation"][field] = value
                with self.assertRaises(ValueError):
                    validate_walkthrough(changed)

    def test_saved_report_rejects_markup_in_numeric_header_before_rendering(
        self,
    ) -> None:
        """A forged PR number must not become active markup in the header."""
        report = compile_snapshot(snap({}, {"a.py": "x = 1\n"}))
        generated = WalkthroughNarrator(FixtureProvider()).generate(report)
        generated["packet"]["meta"]["number"] = (
            '<img src=x onerror="alert(1)">'
        )
        # This is a shape/safety check, independent of evidence-digest matching.
        generated["generation"]["evidence_digest"] = packet_identity(
            generated["packet"]
        )
        with self.assertRaises(ValueError):
            render(generated)


if __name__ == "__main__":
    unittest.main()
