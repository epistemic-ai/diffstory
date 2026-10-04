"""Mocked provider tests for opt-in, bounded, revision-locked narration."""

from __future__ import annotations

import copy
import io
import itertools
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock
from unittest.mock import patch
from urllib.error import HTTPError

from diffstory.analysis import apply_annotations
from diffstory.analysis import compile_snapshot
from diffstory.analysis import validate_generated_report
from diffstory.cli import main
from diffstory.models import MAX_PREAMBLE_CHARS
from diffstory.models import MAX_PREAMBLE_SKETCH_CHARS
from diffstory.models import MAX_PREAMBLE_TOTAL_CHARS
from diffstory.narrative import ASD_STYLE_INSTRUCTION
from diffstory.narrative import CODEX_DEFAULT_CONTEXT_TOKENS
from diffstory.narrative import CODEX_MODEL_CAPACITIES
from diffstory.narrative import DOCUMENT_OUTPUT_RESERVE
from diffstory.narrative import MAX_RESPONSE_BYTES
from diffstory.narrative import OPENAI_MODEL
from diffstory.narrative import OPENAI_URL
from diffstory.narrative import ORDER_OUTPUT_RESERVE
from diffstory.narrative import PROCESS_CLEANUP_TIMEOUT_SECONDS
from diffstory.narrative import PROVIDER_CALL_TIMEOUT_SECONDS
from diffstory.narrative import CodexCLIProvider
from diffstory.narrative import Narrator
from diffstory.narrative import OpenAIResponsesProvider
from diffstory.narrative import ProviderResponseError
from diffstory.narrative import _ProcessRequest
from diffstory.narrative import _run_bounded_subprocess
from diffstory.render import render

if TYPE_CHECKING:
    from collections.abc import Sequence


def snapshot(
    head_source: str = "def calculate(value):\n    return value + 1\n",
) -> dict:
    """
    Build a two-revision snapshot for a value-transforming function.

    Args:
        head_source: Optional head-side source text.

    Returns:
        A bounded ``diffstory.snapshot.v1`` mapping.

    """
    return {
        "schema": "diffstory.snapshot.v1",
        "meta": {
            "base_sha": "a" * 40,
            "head_sha": "b" * 40,
            "scope": "changed files",
            "changed_files": 1,
            "title": "A sample change",
        },
        "fragments": [
            {
                "path": "module.py",
                "side": "base",
                "text": "def calculate(value):\n    return value\n",
                "start_line": 1,
                "scope": "full",
            },
            {
                "path": "module.py",
                "side": "head",
                "text": head_source,
                "start_line": 1,
                "scope": "full",
            },
        ],
        "file_evidence": [
            {
                "base": {
                    "path": "module.py",
                    "state": "supplied",
                    "coverage": "full",
                },
                "head": {
                    "path": "module.py",
                    "state": "supplied",
                    "coverage": "full",
                },
            },
        ],
        "warnings": [],
    }


class FakeProvider:
    """Provide deterministic structured responses without external calls."""

    name = "openai"
    model = OPENAI_MODEL
    destination = OPENAI_URL
    context_tokens = 1_050_000
    max_output_tokens = 128_000

    def __init__(self, *, mode: str = "valid") -> None:
        """
        Configure the fake provider response mode and request log.

        Args:
            mode: ``valid`` for schema-conforming responses or ``malformed`` for an
                intentionally incomplete leaf response.

        """
        self.mode = mode
        self.calls = []
        self.timeouts = []

    def require_credentials(self) -> None:
        """
        Satisfy credential preflight without external authentication.

        Returns:
            ``None``; the fake provider never requires credentials.

        """
        return

    def prepare(
        self,
        system: str,
        data: dict,
        schema_name: str,
        schema: dict,
        max_output_tokens: int,
    ) -> bytes:
        """
        Serialize fake provider instructions and request data.

        Args:
            system: Provider instructions.
            data: Evidence and context payload.
            schema_name: Structured-output schema name.
            schema: JSON output schema.
            max_output_tokens: Per-call output reservation.

        Returns:
            Encoded JSON request bytes.

        """
        return json.dumps(
            {
                "system": system,
                "data": data,
                "name": schema_name,
                "schema": schema,
                "max_output_tokens": max_output_tokens,
            },
            separators=(",", ":"),
        ).encode()

    def complete(
        self,
        body: bytes,
        timeout: float | None,
    ) -> tuple[dict, dict]:
        """
        Record one fake request and return its configured structured response.

        Args:
            body: Encoded fake request.
            timeout: Unused provider timeout.

        Returns:
            Response mapping and measured fake token usage.

        """
        request = json.loads(body)
        self.timeouts.append(timeout)
        self.calls.append(request)
        if self.mode == "malformed":
            response = {
                "summary": "A summary.",
                "questions": [],
                "passages": [],
            }
            return response, {"input_tokens": 100, "output_tokens": 10}
        return self._valid_response(request), {
            "input_tokens": 120,
            "output_tokens": 60,
        }

    def _valid_response(self, request: dict) -> dict:
        """Create the valid fake response for one narration request kind.

        Args:
            request: Serialized request and its task-specific data.

        Returns:
            Deterministic response matching the request schema.

        Raises:
            AssertionError: If the request uses an unexpected schema name.
        """
        response_builders = {
            "diffstory_chunk": self._chunk_response,
            "diffstory_summary": self._summary_response,
            "diffstory_story_order": self._order_response,
            "diffstory_step": self._step_response,
            "diffstory_document": self._document_response,
        }
        builder = response_builders.get(request["name"])
        if builder is None:
            raise AssertionError(request["name"])
        return builder(request["data"])

    @staticmethod
    def _chunk_response(data: dict) -> dict:
        """Return source-bound passages for every changed ID in a chunk.

        Args:
            data: Chunk evidence and group context.

        Returns:
            Summary, unresolved questions, and a passage per changed ID.
        """
        seen = {
            piece["change_id"]: piece.get("source") for piece in data["pieces"]
        }
        passages = []
        for change_id, source in seen.items():
            passage = {
                "text": "This source shows how the value is transformed.",
                "change_ids": [change_id],
                "view": "definition",
                "focus": None,
            }
            if source and source["partial"]:
                passage["focus"] = {
                    "start": source["start"],
                    "end": source["end"],
                }
            passages.append(passage)
        return {
            "summary": "The function transforms the supplied value.",
            "questions": [],
            "passages": passages,
        }

    @staticmethod
    def _summary_response(_data: dict) -> dict:
        """Return the deterministic fake hierarchical summary.

        Args:
            _data: Summary scope and observations, unused by the fixed fixture.

        Returns:
            A one-field summary response.
        """
        return {"summary": "The change updates the value transformation."}

    @staticmethod
    def _order_response(data: dict) -> dict:
        """Return groups in their supplied deterministic order.

        Args:
            data: Story-order request containing candidate groups.

        Returns:
            The same group IDs in supplied order.
        """
        return {"group_ids": [group["id"] for group in data["groups"]]}

    @staticmethod
    def _step_response(_data: dict) -> dict:
        """Return a deterministic story step matching the step schema.

        Args:
            _data: Group context, unused by the fixed fixture.

        Returns:
            A valid narrative-step response.
        """
        return {
            "title": "Transform the value",
            "intent": "Update the value transformation.",
            "why_now": "Read this after its listed prerequisites.",
            "takeaway": "The function returns an adjusted value.",
            "invariants": [
                "The return value is the result of the expression."
            ],
            "questions": ["Are callers prepared for the changed result?"],
            "transition": "Next, inspect the following step.",
        }

    @staticmethod
    def _document_response(_data: dict) -> dict:
        """Return deterministic preamble, opening, and closing prose.

        Args:
            _data: Whole-change context, unused by the fixed fixture.

        Returns:
            A document response with preamble, lead, and closing prose.
        """
        return {
            "preamble": (
                "This change adjusts a value transformation. The section "
                "explains the change. Its source follows the explanation."
            ),
            "lead": "This change adjusts a value transformation.",
            "closing": "The source shows the adjusted return path.",
        }


class NarrativeTests(unittest.TestCase):
    """Verify narration providers, source packing, preflight, and coverage."""

    def test_generation_and_report_roundtrip_keep_provenance(self) -> None:
        """Retain revision binding and evidence when narrations are rendered."""
        source_snapshot = snapshot()
        source_snapshot["meta"]["changed_files"] = 2
        source_snapshot["fragments"].extend(
            [
                {
                    "path": "other.py",
                    "side": "base",
                    "text": "def normalize(value):\n    return value\n",
                    "start_line": 1,
                    "scope": "full",
                },
                {
                    "path": "other.py",
                    "side": "head",
                    "text": "def normalize(value):\n    return value.strip()\n",
                    "start_line": 1,
                    "scope": "full",
                },
            ],
        )
        source_snapshot["file_evidence"].append(
            {
                "base": {
                    "path": "other.py",
                    "state": "supplied",
                    "coverage": "full",
                },
                "head": {
                    "path": "other.py",
                    "state": "supplied",
                    "coverage": "full",
                },
            },
        )
        report = compile_snapshot(source_snapshot)
        provider = FakeProvider()
        narrator = Narrator(provider)
        preview = narrator.preview(report)
        annotations = narrator.generate(report)
        self.assertEqual(preview["calls"], len(provider.calls))
        prose_requests = [
            request
            for request in provider.calls
            if request["name"]
            in {
                "diffstory_chunk",
                "diffstory_summary",
                "diffstory_step",
                "diffstory_document",
            }
        ]
        self.assertEqual(
            {request["name"] for request in prose_requests},
            {
                "diffstory_chunk",
                "diffstory_summary",
                "diffstory_step",
                "diffstory_document",
            },
        )
        self.assertTrue(
            all(
                ASD_STYLE_INSTRUCTION in request["system"]
                for request in prose_requests
            )
        )
        document_request = next(
            request
            for request in prose_requests
            if request["name"] == "diffstory_document"
        )
        self.assertIn("preamble", document_request["schema"]["required"])
        self.assertIn("preamble", annotations["document"])
        generated = apply_annotations(report, annotations)
        self.assertEqual(generated["generation"]["verification"], "unverified")
        self.assertEqual(
            generated["generation"]["base_sha"], report["meta"]["base_sha"]
        )
        self.assertIn(
            "Model-generated narration · unverified.", render(generated)
        )
        self.assertTrue(
            all(
                group["narrative"]["provenance"]
                == "model-generated · unverified"
                for group in generated["groups"]
            )
        )
        saved = json.loads(json.dumps(generated))
        self.assertIn("Model-generated narration · unverified.", render(saved))

    def test_generated_preamble_is_required(self) -> None:
        """Reject generated annotations and saved reports that omit the overview."""
        source_report = compile_snapshot(snapshot())
        candidate = Narrator(FakeProvider()).generate(source_report)
        generated = apply_annotations(source_report, candidate)
        for name in ("annotations", "saved report"):
            with self.subTest(boundary=name):
                value = copy.deepcopy(
                    candidate if name == "annotations" else generated
                )
                value["document"].pop("preamble")
                with self.assertRaisesRegex(ValueError, "preamble"):
                    if name == "annotations":
                        apply_annotations(source_report, value)
                    else:
                        render(value)

    def test_generated_manifest_requires_current_fields(self) -> None:
        """Reject obsolete usage and limits records at annotation import."""
        source_report = compile_snapshot(snapshot())
        candidate = Narrator(FakeProvider()).generate(source_report)
        cases = [
            (
                "obsolete usage",
                "usage",
                {
                    "input_tokens": 1,
                    "output_tokens": 1,
                    "calls": 1,
                    "elapsed_seconds": 1.0,
                },
            ),
            ("obsolete limits", "limits", {}),
        ]
        for name, field, value in cases:
            with self.subTest(case=name):
                invalid = copy.deepcopy(candidate)
                invalid["generation"][field] = value
                with self.assertRaises(ValueError):
                    apply_annotations(source_report, invalid)

    def test_generated_usage_subtotals_fit_totals(self) -> None:
        """Reject impossible token and call totals while accepting exact bounds."""
        source_report = compile_snapshot(snapshot())
        candidate = Narrator(FakeProvider()).generate(source_report)
        usage = {
            "input_tokens": 10,
            "cached_input_tokens": 10,
            "output_tokens": 5,
            "calls": 2,
            "unreported_calls": 2,
        }
        cases = [
            ("exact bounds", {}, True),
            ("cached above input", {"cached_input_tokens": 11}, False),
            ("unreported above calls", {"unreported_calls": 3}, False),
            ("negative count", {"output_tokens": -1}, False),
            ("boolean count", {"calls": True}, False),
        ]
        for name, changes, valid in cases:
            with self.subTest(case=name):
                annotated = copy.deepcopy(candidate)
                annotated["generation"]["usage"] = {**usage, **changes}
                if valid:
                    apply_annotations(source_report, annotated)
                else:
                    with self.assertRaises(ValueError):
                        apply_annotations(source_report, annotated)

    def test_asd_style_guidance_keeps_natural_prose_rhythm(self) -> None:
        """Use ASD principles as a strong guide without making prose robotic."""
        self.assertIn("strong guide", ASD_STYLE_INSTRUCTION)
        self.assertIn("roughly 80 to 90 percent", ASD_STYLE_INSTRUCTION)
        self.assertIn("Do not claim formal compliance", ASD_STYLE_INSTRUCTION)
        self.assertIn("natural rhythm", ASD_STYLE_INSTRUCTION)

    def test_new_document_response_requires_a_nonempty_preamble(self) -> None:
        """Require a preamble in new provider output and reject an empty value."""
        valid = FakeProvider._document_response({})
        Narrator._validate_document_response(valid)

        missing = copy.deepcopy(valid)
        missing.pop("preamble")
        empty = copy.deepcopy(valid)
        empty["preamble"] = " \n"
        for response in (missing, empty):
            with (
                self.subTest(response=response),
                self.assertRaisesRegex(ValueError, "preamble"),
            ):
                Narrator._validate_document_response(response)

    def test_document_prompt_supports_a_flexible_conceptual_overview(
        self,
    ) -> None:
        """Describe an evidence-scaled, conceptual preamble in the document contract."""
        provider = FakeProvider()
        Narrator(provider).generate(compile_snapshot(snapshot()))
        request = next(
            call
            for call in provider.calls
            if call["name"] == "diffstory_document"
        )
        prompt = request["system"]
        preamble_schema = request["schema"]["properties"]["preamble"]
        self.assertIn("before any code tour", prompt)
        self.assertIn("about 200 to 450 words", prompt)
        self.assertIn("Do not pad", prompt)
        self.assertIn("fenced plain-text blocks", prompt)
        self.assertIn("blank line", prompt)
        self.assertIn("do not use Mermaid", prompt)
        self.assertEqual(preamble_schema["type"], "string")
        self.assertEqual(
            preamble_schema["maxLength"], MAX_PREAMBLE_TOTAL_CHARS
        )
        self.assertIn("about 200 to 450 words", preamble_schema["description"])
        self.assertIn("multi-part change", preamble_schema["description"])

    def test_document_output_reserve_covers_a_page_sized_preamble(
        self,
    ) -> None:
        """Reserve response space for a page of prose plus a conceptual sketch."""
        provider = FakeProvider()
        Narrator(provider).generate(compile_snapshot(snapshot()))
        request = next(
            call
            for call in provider.calls
            if call["name"] == "diffstory_document"
        )
        self.assertEqual(request["max_output_tokens"], DOCUMENT_OUTPUT_RESERVE)
        self.assertGreaterEqual(request["max_output_tokens"], 3_200)

    def test_document_preamble_enforces_its_character_bound(self) -> None:
        """Accept a preamble at its bound and reject text above the bound."""
        valid = FakeProvider._document_response({})
        valid["preamble"] = "x" * MAX_PREAMBLE_CHARS
        Narrator._validate_document_response(valid)

        too_long = copy.deepcopy(valid)
        too_long["preamble"] += "x"
        with self.assertRaisesRegex(ValueError, "preamble"):
            Narrator._validate_document_response(too_long)

    def test_provider_preamble_keeps_separate_sketch_allowance(self) -> None:
        """A provider sketch must fit its own allowance and preserve prose space."""
        fence = "\n```text\n{}\n```"
        cases = [
            (
                "prose and sketch",
                "x" * MAX_PREAMBLE_CHARS + fence.format("diagram"),
                True,
            ),
            (
                "oversized sketch",
                "x" + fence.format("y" * MAX_PREAMBLE_SKETCH_CHARS),
                False,
            ),
        ]
        for name, preamble, accepted in cases:
            with self.subTest(case=name):
                document = FakeProvider._document_response({})
                document["preamble"] = preamble
                if accepted:
                    Narrator._validate_document_response(document)
                else:
                    with self.assertRaisesRegex(ValueError, "preamble"):
                        Narrator._validate_document_response(document)

    def test_generated_report_rechecks_preamble_bound(self) -> None:
        """Reject an oversized saved preamble after provider validation."""
        source_report = compile_snapshot(snapshot())
        candidate = Narrator(FakeProvider()).generate(source_report)
        generated = apply_annotations(source_report, candidate)
        for name, preamble, valid in (
            ("prose at limit", "x" * MAX_PREAMBLE_CHARS, True),
            ("prose over limit", "x" * (MAX_PREAMBLE_CHARS + 1), False),
            (
                "prose and sketch",
                "x" * MAX_PREAMBLE_CHARS + "\n```text\ndiagram\n```",
                True,
            ),
            (
                "sketch over limit",
                "x\n```text\n" + "y" * MAX_PREAMBLE_SKETCH_CHARS + "\n```",
                False,
            ),
        ):
            with self.subTest(case=name):
                saved = copy.deepcopy(generated)
                saved["document"]["preamble"] = preamble
                if valid:
                    validate_generated_report(saved)
                    render(saved)
                else:
                    with self.assertRaisesRegex(ValueError, "preamble"):
                        validate_generated_report(saved)
                    with self.assertRaises(ValueError):
                        render(saved)

    def test_large_single_group_is_split_into_stable_bounded_slices(
        self,
    ) -> None:
        """Split a large group into stable bounded slices that retain complete source and coverage."""
        source = (
            "def calculate(value):\n"
            + "".join(f"    value = value + {n}\n" for n in range(1600))
            + "    return value\n"
        )
        report = compile_snapshot(snapshot(source))
        narrator = Narrator(FakeProvider())
        chunks = narrator._chunks(report)
        self.assertEqual(len(chunks), len({chunk.id for chunk in chunks}))
        pieces = [piece for chunk in chunks for piece in chunk.pieces]
        self.assertGreater(len(pieces), 1)
        self.assertTrue(
            all(
                len(piece["source"]["source"].encode()) <= 12_000
                for piece in pieces
                if piece.get("source")
            )
        )
        ranges = sorted(
            (piece["source"]["start"], piece["source"]["end"])
            for piece in pieces
            if piece.get("source")
        )
        self.assertEqual(ranges[0][0], 1)
        self.assertEqual(ranges[-1][1], 1602)
        for left, right in itertools.pairwise(ranges):
            self.assertEqual(left[1] + 1, right[0])
        annotations = narrator.generate(report)
        self.assertEqual(
            len(annotations["generation"]["chunk_coverage"]), len(chunks)
        )
        generated = apply_annotations(report, annotations)
        self.assertEqual(
            generated["generation"]["covered_changes"],
            [report["changes"][0]["id"]],
        )

    def test_oversized_single_line_is_split_without_losing_source(
        self,
    ) -> None:
        """Split an oversized source line without losing or reordering any source text."""
        source = "def calculate(): return '" + ("x" * 13_000) + "'\n"
        report = compile_snapshot(snapshot(source))
        provider = FakeProvider()
        Narrator(provider).generate(report)
        head_fragments = [
            piece["source"]["source"]
            for call in provider.calls
            if call["name"] == "diffstory_chunk"
            for piece in call["data"]["pieces"]
            if piece.get("source") and piece["source"].get("side") == "head"
        ]
        self.assertEqual(
            "".join(head_fragments), report["changes"][0]["after"]["source"]
        )

    def test_long_line_slices_have_unique_piece_and_chunk_ids(self) -> None:
        """
        Keep long same-line source slices distinct through annotation application.

        Returns:
            ``None``; assertions verify stable unique evidence IDs and coverage.

        """
        report = compile_snapshot(snapshot())
        report["changes"][0]["after"]["source"] = "x" * 2_900_000
        narrator = Narrator(FakeProvider())

        chunks = narrator._chunks(report)
        piece_ids = [piece["id"] for chunk in chunks for piece in chunk.pieces]
        self.assertGreater(len(chunks), 1)
        self.assertEqual(len(piece_ids), len(set(piece_ids)))
        self.assertEqual(len(chunks), len({chunk.id for chunk in chunks}))

        annotations = narrator.generate(report)
        self.assertEqual(
            len(annotations["generation"]["chunk_coverage"]), len(chunks)
        )
        apply_annotations(report, annotations)

    def test_large_required_citation_sets_split_leaf_chunks(self) -> None:
        """
        Scale output reservations and split chunks to cover many changes.

        Returns:
            ``None``; assertions verify all changes fit within model output limits.

        """
        source = (
            "\n".join(f"value_{index} = {index}" for index in range(1_000))
            + "\n"
        )
        report = compile_snapshot(snapshot(source))
        provider = FakeProvider()
        narrator = Narrator(provider)

        self.assertGreaterEqual(len(report["changes"]), 1_000)
        chunks = narrator._chunks(report)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(
            all(
                narrator._leaf_output(chunk.change_ids)
                <= provider.max_output_tokens
                for chunk in chunks
            )
        )
        covered = {
            change_id for chunk in chunks for change_id in chunk.change_ids
        }
        self.assertEqual(
            covered, {change["id"] for change in report["changes"]}
        )

        many_group_ids = [
            f"group-{index:04d}-" + "a" * 32 for index in range(1_000)
        ]
        self.assertGreater(
            narrator._order_output(many_group_ids), ORDER_OUTPUT_RESERVE
        )
        self.assertLessEqual(
            narrator._order_output(many_group_ids), provider.max_output_tokens
        )
        annotations = narrator.generate(report)
        apply_annotations(report, annotations)

    def test_preview_has_no_run_token_or_call_ceilings(self) -> None:
        """Preview the request plan without imposing arbitrary token or call ceilings."""
        report = compile_snapshot(snapshot())
        preview = Narrator(FakeProvider()).preview(report)
        self.assertGreater(preview["calls"], 0)
        self.assertNotIn("reserved_input_tokens", preview)
        self.assertNotIn("max_calls", preview)

    def test_request_capacity_tracks_model_context(self) -> None:
        """Derive input capacity from the provider context and output reservation."""
        provider = FakeProvider()
        narrator = Narrator(provider)
        self.assertEqual(
            narrator.capacity.input_upper_bound,
            provider.context_tokens - provider.max_output_tokens,
        )

    def test_provider_calls_use_finite_per_request_timeout(self) -> None:
        """Pass a finite timeout to every provider call without run ceilings."""
        provider = FakeProvider()
        Narrator(provider).generate(compile_snapshot(snapshot()))
        self.assertTrue(provider.timeouts)
        self.assertEqual(
            set(provider.timeouts), {PROVIDER_CALL_TIMEOUT_SECONDS}
        )

    def test_malformed_provider_response_does_not_create_annotation(
        self,
    ) -> None:
        """Reject a provider response without valid source-bound passages."""
        report = compile_snapshot(snapshot())
        provider = FakeProvider(mode="malformed")
        narrator = Narrator(provider)
        with self.assertRaisesRegex(ValueError, "no source-bound passages"):
            narrator.generate(report)
        self.assertEqual(len(provider.calls), 1)

    def test_generated_annotations_reject_wrong_revision(self) -> None:
        """Reject generated annotations tied to a different head revision."""
        report = compile_snapshot(snapshot())
        annotations = Narrator(FakeProvider()).generate(report)
        annotations["head_sha"] = "c" * 40
        with self.assertRaisesRegex(ValueError, "different head revision"):
            apply_annotations(report, annotations)

    def test_generated_annotations_require_all_groups_and_passages(
        self,
    ) -> None:
        """Require complete group and passage coverage for generated annotations."""
        report = compile_snapshot(snapshot())
        annotations = Narrator(FakeProvider()).generate(report)
        annotations["steps"][0]["passages"] = []
        with self.assertRaisesRegex(
            ValueError, "missing source-bound passages"
        ):
            apply_annotations(report, annotations)

    def test_generated_report_rejects_stale_coverage_on_render(self) -> None:
        """Revalidate generated coverage when rendering a saved report."""
        report = compile_snapshot(snapshot())
        generated = apply_annotations(
            report, Narrator(FakeProvider()).generate(report)
        )
        generated["generation"]["completed_groups"] = []
        with self.assertRaisesRegex(ValueError, "every report group"):
            render(generated)

    def test_aggregate_snapshot_source_limit_is_enforced(self) -> None:
        """Enforce the aggregate source-byte limit when compiling a snapshot."""
        from unittest.mock import patch

        with (
            patch("diffstory.models.MAX_SNAPSHOT_SOURCE_BYTES", 4),
            self.assertRaisesRegex(ValueError, "aggregate limit"),
        ):
            compile_snapshot(snapshot())

    def test_cli_without_narrate_never_calls_provider(self) -> None:
        """Avoid provider calls and annotation output unless narration is requested."""
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "walkthrough.html"
            snap = snapshot()
            with (
                patch("diffstory.cli.from_github", return_value=snap),
                patch.object(
                    OpenAIResponsesProvider,
                    "complete",
                    side_effect=AssertionError("unexpected model request"),
                ),
            ):
                result = main(["github", "owner/repo#1", "--out", str(out)])
            self.assertEqual(result, 0)
            self.assertTrue(out.exists())
            self.assertFalse(out.with_suffix(".annotations.json").exists())

    def test_cli_narration_requires_confirmation_before_provider_call(
        self,
    ) -> None:
        """Require transfer confirmation before the first provider request."""
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "walkthrough.html"
            provider = FakeProvider()
            with (
                patch("diffstory.cli.from_github", return_value=snapshot()),
                patch(
                    "diffstory.cli.OpenAIResponsesProvider",
                    return_value=provider,
                ),
                patch("diffstory.cli.sys.stdin.isatty", return_value=False),
            ):
                result = main(
                    ["github", "owner/repo#1", "--narrate", "--out", str(out)]
                )
            self.assertEqual(result, 2)
            self.assertEqual(provider.calls, [])
            self.assertFalse(out.exists())

    def test_cli_opt_in_writes_candidate_and_generated_report(self) -> None:
        """Write candidate annotations and the generated report after explicit opt-in."""
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "walkthrough.html"
            provider = FakeProvider()
            with (
                patch("diffstory.cli.from_github", return_value=snapshot()),
                patch(
                    "diffstory.cli.OpenAIResponsesProvider",
                    return_value=provider,
                ),
            ):
                result = main(
                    [
                        "github",
                        "owner/repo#1",
                        "--narrate",
                        "--yes",
                        "--out",
                        str(out),
                    ]
                )
            candidate = out.with_suffix(".annotations.json")
            report_path = out.with_suffix(".report.json")
            self.assertEqual(result, 0)
            self.assertTrue(candidate.exists())
            self.assertTrue(out.exists())
            report = json.loads(report_path.read_text())
            annotations = json.loads(candidate.read_text())
            self.assertEqual(
                report["generation"]["schema"], "diffstory.generation.v1"
            )
            self.assertEqual(
                annotations["generation"]["head_sha"],
                report["meta"]["head_sha"],
            )
            self.assertNotIn("api_key", candidate.read_text())
            self.assertTrue(provider.calls)
            imported = Path(directory) / "candidate-import.html"
            self.assertEqual(
                main(
                    [
                        "render",
                        str(report_path),
                        "--annotations",
                        str(candidate),
                        "--out",
                        str(imported),
                    ]
                ),
                0,
            )
            self.assertIn(
                "Model-generated narration · unverified.", imported.read_text()
            )

    def test_cli_provider_failure_does_not_write_a_partial_report(
        self,
    ) -> None:
        """Leave report, HTML, and annotation outputs unwritten when generation fails."""
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "walkthrough.html"
            provider = FakeProvider(mode="malformed")
            error_output = io.StringIO()
            with (
                patch("diffstory.cli.from_github", return_value=snapshot()),
                patch(
                    "diffstory.cli.OpenAIResponsesProvider",
                    return_value=provider,
                ),
                redirect_stderr(error_output),
            ):
                result = main(
                    [
                        "github",
                        "owner/repo#1",
                        "--narrate",
                        "--yes",
                        "--out",
                        str(out),
                    ]
                )
            self.assertEqual(result, 2)
            self.assertIn("source-bound passages", error_output.getvalue())
            self.assertFalse(out.exists())
            self.assertFalse(out.with_suffix(".report.json").exists())
            self.assertFalse(out.with_suffix(".annotations.json").exists())

    def test_codex_model_capacity_is_not_inherited_from_openai(self) -> None:
        """Use Codex model metadata and reject overrides without known capacity."""
        default_provider = CodexCLIProvider()
        self.assertEqual(
            default_provider.context_tokens, CODEX_DEFAULT_CONTEXT_TOKENS
        )
        for model, (
            context_tokens,
            output_tokens,
        ) in CODEX_MODEL_CAPACITIES.items():
            with self.subTest(model=model):
                provider = CodexCLIProvider(model=model)
                self.assertEqual(provider.context_tokens, context_tokens)
                self.assertEqual(provider.max_output_tokens, output_tokens)
        with self.assertRaisesRegex(ValueError, "no known context capacity"):
            CodexCLIProvider(model="unlisted-codex-model")

    def test_codex_cli_uses_isolated_flags_scrubs_keys_and_parses_usage(
        self,
    ) -> None:
        """Verify Codex flags, temporary cwd, key removal, JSONL, and usage."""
        schema = {
            "type": "object",
            "properties": {"summary": {"type": "string"}},
            "required": ["summary"],
            "additionalProperties": False,
        }
        provider = CodexCLIProvider(model="gpt-6.1-sol")
        body = provider.prepare(
            "Summarize evidence.", {"source": "value"}, "summary", schema, 100
        )
        events = [
            {
                "type": "item.completed",
                "item": {"type": "agent_message", "text": '{"summary":"ok"}'},
            },
            {
                "type": "turn.completed",
                "usage": {
                    "input_tokens": 8,
                    "cached_input_tokens": 2,
                    "output_tokens": 3,
                },
            },
        ]
        process = MagicMock()
        process.stdin = MagicMock()
        process.stdin.write.return_value = len(
            provider._PROMPT.encode()
        ) + len(body)
        process.stdout = io.BytesIO(
            "\n".join(json.dumps(event) for event in events).encode()
        )
        process.wait.return_value = 0
        process.returncode = 0
        launch = {}

        def start_process(
            command: Sequence[str],
            **kwargs: object,
        ) -> MagicMock:
            """
            Capture launch details while the provider temp directory exists.

            Args:
                command: Codex CLI argument vector.
                **kwargs: Subprocess options supplied by the provider.

            Returns:
                A mocked process with prepared JSONL output.

            """
            launch["command"] = command
            launch["cwd"] = kwargs["cwd"]
            launch["environment"] = kwargs["env"].copy()
            schema_path = Path(command[command.index("--output-schema") + 1])
            launch["schema_path"] = str(schema_path)
            launch["schema"] = json.loads(schema_path.read_text())
            launch["cwd_exists"] = Path(kwargs["cwd"]).is_dir()
            launch["options"] = kwargs
            return process

        with (
            patch.object(provider, "require_credentials"),
            patch(
                "diffstory.narrative.subprocess.Popen",
                side_effect=start_process,
            ),
            patch.dict(
                "os.environ",
                {
                    "OPENAI_API_KEY": "parent-api-key",
                    "CODEX_API_KEY": "parent-codex-key",
                },
            ),
        ):
            result, usage = provider.complete(body, 7)

        self.assertEqual(result, {"summary": "ok"})
        self.assertEqual(
            usage,
            {"input_tokens": 8, "cached_input_tokens": 2, "output_tokens": 3},
        )
        self._assert_codex_launch(launch, schema)
        self._assert_codex_prompt(process, provider, body)
        process.wait.assert_called_once_with(timeout=7)
        process.stdin.close.assert_called_once()

    def _assert_codex_launch(self, launch: dict, schema: dict) -> None:
        """Assert the Codex process has its bounded, isolated launch settings.

        Args:
            launch: Captured command, cwd, environment, and process options.
            schema: Expected output schema written to the temporary directory.
        """
        command = launch["command"]
        options = launch["options"]
        self.assertEqual(
            command[:4], ["codex", "exec", "--model", "gpt-6.1-sol"]
        )
        self.assertTrue(
            all(
                option in command
                for option in (
                    "--ephemeral",
                    "--ignore-user-config",
                    "--ignore-rules",
                    "--json",
                    "--skip-git-repo-check",
                )
            ),
        )
        self.assertEqual(command[command.index("--sandbox") + 1], "read-only")
        self.assertEqual(
            [
                command[index + 1]
                for index, argument in enumerate(command[:-1])
                if argument == "--disable"
            ],
            [
                "shell_tool",
                "code_mode_host",
                "apps",
                "plugins",
                "browser_use",
                "browser_use_external",
                "computer_use",
                "tool_call_mcp_elicitation",
            ],
        )
        self.assertEqual(
            (options["stdin"], options["stdout"], options["stderr"]),
            (subprocess.PIPE, subprocess.PIPE, subprocess.DEVNULL),
        )
        if os.name == "nt":
            self.assertTrue(
                options["creationflags"] & subprocess.CREATE_NEW_PROCESS_GROUP
            )
        else:
            self.assertTrue(options["start_new_session"])
        self.assertTrue(launch["cwd_exists"])
        self.assertTrue(
            Path(launch["cwd"]).name.startswith("diffstory-codex-")
        )
        self.assertEqual(command[command.index("--cd") + 1], launch["cwd"])
        self.assertEqual(
            launch["schema_path"],
            command[command.index("--output-schema") + 1],
        )
        self.assertEqual(launch["schema"], schema)
        self.assertNotIn("OPENAI_API_KEY", launch["environment"])
        self.assertNotIn("CODEX_API_KEY", launch["environment"])

    def _assert_codex_prompt(
        self,
        process: MagicMock,
        provider: CodexCLIProvider,
        body: bytes,
    ) -> None:
        """Assert the request uses the fixed prompt prefix and prepared body.

        Args:
            process: Mock process whose stdin received the request.
            provider: Codex provider supplying the request prefix.
            body: Serialized provider-neutral request bytes.
        """
        prompt = bytes(process.stdin.write.call_args.args[0])
        self.assertTrue(prompt.startswith(provider._PROMPT.encode()))
        self.assertTrue(prompt.endswith(body))

    def test_codex_cli_timeout_is_passed_to_process_and_reported(self) -> None:
        """Terminate and report a timed-out Codex subprocess request."""
        provider = CodexCLIProvider()
        body = provider.prepare(
            "system", {}, "summary", {"type": "object"}, 100
        )
        process = MagicMock()
        process.stdin = MagicMock()
        process.stdin.write.return_value = len(
            provider._PROMPT.encode()
        ) + len(body)
        process.stdout = io.BytesIO(b"")
        process.wait.side_effect = [
            subprocess.TimeoutExpired(["codex"], 4),
            -9,
        ]
        with (
            patch.object(provider, "require_credentials"),
            patch(
                "diffstory.narrative.subprocess.Popen", return_value=process
            ),
            patch(
                "diffstory.narrative._terminate_process_tree"
            ) as terminate_tree,
            self.assertRaisesRegex(ProviderResponseError, "timed out"),
        ):
            provider.complete(body, 4)
        process.wait.assert_any_call(timeout=4)
        process.wait.assert_any_call(timeout=PROCESS_CLEANUP_TIMEOUT_SECONDS)
        terminate_tree.assert_called_once_with(process)

    def test_codex_cli_stdout_is_terminated_at_the_response_limit(
        self,
    ) -> None:
        """Kill a streaming child as soon as stdout exceeds the configured cap."""
        source = (
            "import sys, time; "
            f"sys.stdout.write('x' * {MAX_RESPONSE_BYTES + 1}); "
            "sys.stdout.flush(); time.sleep(60)"
        )
        with (
            tempfile.TemporaryDirectory() as directory,
            self.assertRaisesRegex(
                ProviderResponseError,
                f"configured {MAX_RESPONSE_BYTES:,}-byte limit",
            ),
        ):
            _run_bounded_subprocess(
                _ProcessRequest(
                    command=[sys.executable, "-c", source],
                    prompt=b"",
                    cwd=directory,
                    environment=os.environ.copy(),
                    timeout=5,
                    output_limit=MAX_RESPONSE_BYTES,
                ),
            )

    def test_codex_cli_timeout_terminates_descendants_holding_stdout(
        self,
    ) -> None:
        """
        Terminate descendant processes so inherited stdout cannot hang cleanup.

        Returns:
            ``None``; the subprocess raises a timeout promptly after tree cleanup.

        """
        source = (
            "import subprocess, sys, time; "
            "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)']); "
            "time.sleep(30)"
        )
        with tempfile.TemporaryDirectory() as directory:
            started = time.monotonic()
            with self.assertRaisesRegex(ProviderResponseError, "timed out"):
                _run_bounded_subprocess(
                    _ProcessRequest(
                        command=[sys.executable, "-c", source],
                        prompt=b"",
                        cwd=directory,
                        environment=os.environ.copy(),
                        timeout=0.1,
                        output_limit=MAX_RESPONSE_BYTES,
                    ),
                )
            self.assertLess(
                time.monotonic() - started,
                PROCESS_CLEANUP_TIMEOUT_SECONDS,
            )

    def test_codex_jsonl_rejects_tool_calls_and_malformed_events(self) -> None:
        """Reject Codex output that invokes tools or violates JSONL structure."""
        tool_events = [
            {"type": "item.completed", "item": {"type": "command_execution"}},
            {
                "type": "item.completed",
                "item": {"type": "agent_message", "text": '{"summary":"ok"}'},
            },
        ]
        with self.assertRaisesRegex(ProviderResponseError, "attempted tool"):
            CodexCLIProvider._parse_response(
                "\n".join(json.dumps(event) for event in tool_events).encode()
            )
        with self.assertRaisesRegex(ProviderResponseError, "malformed JSONL"):
            CodexCLIProvider._parse_response(b"not-json\n")

    def test_codex_nonzero_exit_is_sanitized(self) -> None:
        """Report a safe error when Codex exits unsuccessfully."""
        provider = CodexCLIProvider()
        body = provider.prepare(
            "system", {}, "summary", {"type": "object"}, 100
        )
        process = MagicMock()
        process.stdin = MagicMock()
        process.stdin.write.return_value = len(
            provider._PROMPT.encode()
        ) + len(body)
        process.stdout = io.BytesIO(b"")
        process.wait.return_value = 1
        process.returncode = 1
        with (
            patch.object(provider, "require_credentials"),
            patch(
                "diffstory.narrative.subprocess.Popen", return_value=process
            ),
            self.assertRaisesRegex(ProviderResponseError, "request failed"),
        ):
            provider.complete(body, 7)

    def test_openai_adapter_uses_structured_output_and_one_attempt(
        self,
    ) -> None:
        """Send one structured-output request and preserve sanitized measured usage."""
        response = MagicMock()
        response.read.return_value = json.dumps(
            {
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": '{"summary":"ok"}',
                            }
                        ],
                    }
                ],
                "usage": {
                    "input_tokens": 8,
                    "input_tokens_details": {"cached_tokens": 2},
                    "output_tokens": 3,
                },
            },
        ).encode()
        opener = MagicMock()
        opener.open.return_value.__enter__.return_value = response
        with patch("diffstory.narrative.build_opener", return_value=opener):
            provider = OpenAIResponsesProvider(api_key="test-secret")
            body = provider.prepare(
                "Return JSON.",
                {"source": "safe"},
                "summary",
                {"type": "object"},
                100,
            )
            result, usage = provider.complete(body, 5)
        self.assertEqual(provider.model, "gpt-6.1-sol")
        request = opener.open.call_args.args[0]
        self.assertEqual(opener.open.call_count, 1)
        self.assertEqual(opener.open.call_args.kwargs["timeout"], 5)
        self.assertEqual(
            request.get_header("Authorization"), "Bearer test-secret"
        )
        self.assertEqual(result, {"summary": "ok"})
        self.assertEqual(
            usage,
            {"input_tokens": 8, "cached_input_tokens": 2, "output_tokens": 3},
        )
        self.assertEqual(json.loads(body)["store"], False)

    def test_openai_http_error_is_sanitized(self) -> None:
        """Hide response bodies and API keys from provider HTTP error messages."""
        sensitive_value = "provider-error-secret"
        error = HTTPError(
            OPENAI_URL,
            403,
            "Forbidden",
            {},
            io.BytesIO(sensitive_value.encode()),
        )
        opener = MagicMock()
        opener.open.side_effect = error
        with patch("diffstory.narrative.build_opener", return_value=opener):
            provider = OpenAIResponsesProvider(api_key="test-secret")
            with self.assertRaises(ValueError) as raised:
                provider.complete(b"{}", 1)
        self.assertTrue(error.fp.closed)
        self.assertNotIn(sensitive_value, str(raised.exception))
        self.assertNotIn("test-secret", str(raised.exception))

    def test_openai_invalid_api_key_header_is_sanitized_before_network_access(
        self,
    ) -> None:
        """
        Reject control characters in API keys without exposing their value.

        Returns:
            ``None``; the API key is rejected before constructing a network opener.

        """
        credential_value = "review-test-secret\n"
        provider = OpenAIResponsesProvider(api_key=credential_value)
        with (
            patch("diffstory.narrative.build_opener") as build_opener,
            self.assertRaises(ValueError) as raised,
        ):
            provider.complete(b"{}", 1)
        self.assertIn("invalid header characters", str(raised.exception))
        self.assertNotIn("review-test-secret", str(raised.exception))
        build_opener.assert_not_called()


if __name__ == "__main__":
    unittest.main()
