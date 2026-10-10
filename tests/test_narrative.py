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
import threading
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
from diffstory.narrative import MAX_SOURCE_SLICE_BYTES
from diffstory.narrative import MAX_STDERR_BYTES
from diffstory.narrative import OPENAI_MODEL
from diffstory.narrative import OPENAI_URL
from diffstory.narrative import ORDER_OUTPUT_RESERVE
from diffstory.narrative import PROCESS_CLEANUP_TIMEOUT_SECONDS
from diffstory.narrative import PROVIDER_CALL_TIMEOUT_SECONDS
from diffstory.narrative import CodexCLIProvider
from diffstory.narrative import Narrator
from diffstory.narrative import OpenAIResponsesProvider
from diffstory.narrative import ProviderResponseError
from diffstory.narrative import _classify_codex_stderr
from diffstory.narrative import _decode_codex_events
from diffstory.narrative import _NarrativeCall
from diffstory.narrative import _ProcessRequest
from diffstory.narrative import _run_bounded_subprocess
from diffstory.narrative import _split_source
from diffstory.narrative_runtime import CallSpec
from diffstory.narrative_runtime import ProgressEvent
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


def two_group_report() -> dict:
    """Build two independent groups to exercise ordered parallel stages."""
    source = snapshot()
    source["meta"]["changed_files"] = 2
    source["fragments"].extend(
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
    source["file_evidence"].append(
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
    return compile_snapshot(source)


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


class ConcurrentProvider(FakeProvider):
    """Track overlap while keeping deterministic fake responses."""

    def __init__(self) -> None:
        """Set counters for active and overlapping provider calls."""
        super().__init__()
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()

    def complete(
        self, body: bytes, timeout: float | None
    ) -> tuple[dict, dict]:
        """Delay a deterministic request while measuring active calls."""
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(0.03)
            return super().complete(body, timeout)
        finally:
            with self.lock:
                self.active -= 1


class FailOnOrderProvider(FakeProvider):
    """Fail one semantic stage with measured, categorized usage."""

    def complete(
        self, body: bytes, timeout: float | None
    ) -> tuple[dict, dict]:
        """Return normal results until the story-order request fails."""
        request = json.loads(body)
        if request["name"] == "diffstory_story_order":
            self.timeouts.append(timeout)
            self.calls.append(request)
            message = "private provider body"
            usage = {
                "input_tokens": 12,
                "cached_input_tokens": 3,
                "output_tokens": 0,
            }
            raise ProviderResponseError(
                message,
                usage,
                category="rate_limit",
            )
        return super().complete(body, timeout)


class NarrativeTests(unittest.TestCase):
    """Verify narration providers, source packing, preflight, and coverage."""

    def test_report_signature_ignores_only_capture_time(self) -> None:
        """Keep source and all other report metadata in the run identity."""
        report = compile_snapshot(snapshot())
        report["meta"]["captured_at"] = "2026-01-01T12:00:00Z"
        refreshed = copy.deepcopy(report)
        refreshed["meta"]["captured_at"] = "2026-01-02T12:00:00Z"
        self.assertEqual(
            Narrator._report_signature(report),
            Narrator._report_signature(refreshed),
        )

        refreshed["changes"][0]["label"] += " changed"
        self.assertNotEqual(
            Narrator._report_signature(report),
            Narrator._report_signature(refreshed),
        )

    def test_codex_prompt_prefix_is_in_checkpoint_request_digest(self) -> None:
        """Include Codex's fixed framing in the request that names a cache."""
        provider = CodexCLIProvider(model=OPENAI_MODEL)
        narrator = Narrator(provider)
        narrator._start_runtime(compile_snapshot(snapshot()))
        call = _NarrativeCall(
            task_id="prefix-check",
            schema_name="diffstory_summary",
            system="Summarize the supplied evidence.",
            data={"observations": []},
            response_schema={"type": "object"},
            output_reservation=100,
            validation_version="summary-v1",
        )
        with patch.object(
            narrator.runtime, "invoke", return_value={}
        ) as invoke:
            narrator._invoke(call, lambda response: response)

        actual = invoke.call_args.args[0]
        body = provider.prepare(
            call.system,
            call.data,
            call.schema_name,
            call.response_schema,
            call.output_reservation,
        )
        expected = CallSpec.from_request(
            task_id=call.task_id,
            schema_name=call.schema_name,
            request=provider.request_prefix + body,
            schema=call.response_schema,
            output_reservation=call.output_reservation,
            validation_version=call.validation_version,
        )
        body_only = CallSpec.from_request(
            task_id=call.task_id,
            schema_name=call.schema_name,
            request=body,
            schema=call.response_schema,
            output_reservation=call.output_reservation,
            validation_version=call.validation_version,
        )
        self.assertEqual(actual.request_digest, expected.request_digest)
        self.assertNotEqual(actual.request_digest, body_only.request_digest)

    def test_parallel_stages_preserve_story_order_and_progress_safety(
        self,
    ) -> None:
        """Run independent work together and retain input order in results."""
        report = two_group_report()
        provider = ConcurrentProvider()
        events: list[ProgressEvent] = []
        callback_threads = []
        coordinator = threading.get_ident()

        def record_progress(event: ProgressEvent) -> None:
            events.append(event)
            callback_threads.append(threading.get_ident())

        annotations = Narrator(
            provider,
            concurrency=3,
            on_progress=record_progress,
        ).generate(report)

        self.assertGreaterEqual(provider.max_active, 2)
        self.assertEqual(
            [step["group_id"] for step in annotations["steps"]],
            [group["id"] for group in report["groups"]],
        )
        self.assertTrue(events)
        self.assertEqual(set(callback_threads), {coordinator})
        self.assertTrue(
            all(isinstance(event, ProgressEvent) for event in events)
        )
        self.assertTrue(any(event.stage == "chunks" for event in events))

    def test_interrupted_generation_resumes_cached_work_and_usage(
        self,
    ) -> None:
        """Resume after a provider failure without reusing historical usage."""
        report = compile_snapshot(snapshot())
        report["meta"]["captured_at"] = "fetch-one"
        with tempfile.TemporaryDirectory() as directory:
            first_provider = FailOnOrderProvider()
            first_narrator = Narrator(
                first_provider,
                checkpoint_dir=directory,
            )
            self.assertEqual(first_narrator.reused_checkpoints, 0)
            with self.assertRaises(ProviderResponseError) as raised:
                first_narrator.generate(report)

            self.assertEqual(raised.exception.category, "rate_limit")
            self.assertNotIn("private provider body", str(raised.exception))
            self.assertEqual(
                first_narrator.usage.report(),
                {
                    "input_tokens": 132,
                    "cached_input_tokens": 3,
                    "output_tokens": 60,
                    "calls": 2,
                    "unreported_calls": 0,
                },
            )

            resumed_report = copy.deepcopy(report)
            resumed_report["meta"]["captured_at"] = "fetch-two"
            resumed_provider = FakeProvider()
            resumed_narrator = Narrator(
                resumed_provider,
                checkpoint_dir=directory,
            )
            resumed_narrator.generate(resumed_report)

        self.assertCountEqual(
            [call["name"] for call in resumed_provider.calls],
            [
                "diffstory_story_order",
                "diffstory_step",
                "diffstory_document",
            ],
        )
        self.assertEqual(resumed_narrator.reused_checkpoints, 1)
        self.assertEqual(resumed_narrator.usage.report()["calls"], 3)

    def test_completed_generation_reuses_all_validated_checkpoints(
        self,
    ) -> None:
        """Expose completed-run cache reuse without importing old usage."""
        report = compile_snapshot(snapshot())
        with tempfile.TemporaryDirectory() as directory:
            first_provider = FakeProvider()
            first_narrator = Narrator(
                first_provider,
                checkpoint_dir=directory,
            )
            first = first_narrator.generate(report)

            resumed_provider = FakeProvider()
            resumed_narrator = Narrator(
                resumed_provider,
                checkpoint_dir=directory,
            )
            self.assertEqual(resumed_narrator.reused_checkpoints, 0)
            resumed = resumed_narrator.generate(report)

        self.assertEqual(len(first_provider.calls), 4)
        self.assertEqual(resumed_provider.calls, [])
        self.assertEqual(first["document"], resumed["document"])
        self.assertEqual(first["steps"], resumed["steps"])
        self.assertEqual(resumed_narrator.reused_checkpoints, 4)
        self.assertEqual(resumed_narrator.usage.report()["calls"], 0)

    def test_invalid_cached_focus_is_rejected_and_regenerated(self) -> None:
        """Recheck cached focus against source before accepting a chunk."""
        report = compile_snapshot(snapshot())
        with tempfile.TemporaryDirectory() as directory:
            Narrator(FakeProvider(), checkpoint_dir=directory).generate(report)
            changed_checkpoint = False
            for path in Path(directory).glob("*/*.json"):
                saved = json.loads(path.read_text(encoding="utf-8"))
                result = saved.get("result")
                if not isinstance(result, dict) or "passages" not in result:
                    continue
                result["passages"][0]["focus"] = {
                    "start": 9000,
                    "end": 9000,
                }
                path.write_text(json.dumps(saved), encoding="utf-8")
                changed_checkpoint = True
                break
            self.assertTrue(changed_checkpoint)

            provider = FakeProvider()
            resumed = Narrator(provider, checkpoint_dir=directory)
            resumed.generate(report)

        self.assertEqual(
            [call["name"] for call in provider.calls], ["diffstory_chunk"]
        )
        self.assertEqual(resumed.reused_checkpoints, 3)
        self.assertEqual(resumed.usage.report()["calls"], 1)

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
        with self.assertRaises(ProviderResponseError) as raised:
            narrator.generate(report)
        self.assertEqual(raised.exception.category, "validation_error")
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
                    [
                        "github",
                        "owner/repo#1",
                        "--narrate",
                        "--architecture",
                        "legacy",
                        "--out",
                        str(out),
                    ]
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
                        "--architecture",
                        "legacy",
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
                        "--architecture",
                        "legacy",
                        "--yes",
                        "--out",
                        str(out),
                    ]
                )
            self.assertEqual(result, 2)
            self.assertIn("validation_error", error_output.getvalue())
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
                "multi_agent",
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
            (subprocess.PIPE, subprocess.PIPE, subprocess.PIPE),
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
        """Bound stderr, classify its category, and hide its raw contents."""
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
        private_stderr = b"unauthorized private-error-secret " + b"x" * (
            MAX_STDERR_BYTES * 4
        )
        process.stderr = io.BytesIO(private_stderr)
        process.wait.return_value = 1
        process.returncode = 1
        with (
            patch.object(provider, "require_credentials"),
            patch(
                "diffstory.narrative.subprocess.Popen", return_value=process
            ),
            patch(
                "diffstory.narrative._classify_codex_stderr",
                wraps=_classify_codex_stderr,
            ) as classify_stderr,
            self.assertRaises(ProviderResponseError) as raised,
        ):
            provider.complete(body, 7)
        self.assertEqual(raised.exception.category, "authentication")
        self.assertNotIn("private-error-secret", str(raised.exception))
        self.assertLessEqual(
            len(classify_stderr.call_args.args[0]),
            MAX_STDERR_BYTES,
        )

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


class PhysicalSourceNarrativeTests(unittest.TestCase):
    """Source slicing and JSONL retain Unicode characters within a line."""

    def test_source_slices_preserve_unicode_long_lines_and_eols(self) -> None:
        """Keep exact bytes and physical line numbers, even across byte splits."""
        for eol in ("\n", "\r\n", "\r"):
            for text, lines in (
                ("", 1),
                (eol, 1),
                ("a\u2028b\u2029c\x85d\ve\ff" + eol + "end" + eol, 2),
                ("é\u2028" * 9000 + eol + "end", 2),
            ):
                with self.subTest(eol=eol, length=len(text)):
                    parts = _split_source({"source": text, "start": 17})
                    self.assertEqual(
                        "".join(p["source"] for p in parts).encode(),
                        text.encode(),
                    )
                    self.assertEqual(
                        max(p["end"] for p in parts), 17 + lines - 1
                    )
                    self.assertTrue(
                        all(
                            len(p["source"].encode()) <= MAX_SOURCE_SLICE_BYTES
                            for p in parts
                        )
                    )

    def test_unicode_focus_accepts_physical_line_and_rejects_phantom_line(
        self,
    ) -> None:
        """A definition slice can focus its last actual line only."""
        report = compile_snapshot(
            snapshot(
                "def calculate(value):\n    marker = 'a\u2028b'\n    return value + 1\n"
            )
        )
        narrator = Narrator(FakeProvider())
        chunk = narrator._chunks(report)[0]
        ranges = Narrator._source_ranges(chunk)
        change_id = chunk.change_ids[0]
        focus = {"start": 3, "end": 3}
        self.assertEqual(
            Narrator._passage_focus(
                focus, "definition", [change_id], ranges, chunk.id
            ),
            focus,
        )
        with self.assertRaisesRegex(ValueError, "outside the supplied"):
            Narrator._passage_focus(
                {"start": 4, "end": 4},
                "definition",
                [change_id],
                ranges,
                chunk.id,
            )

    def test_codex_jsonl_unicode_raw_and_structured_agent_messages(
        self,
    ) -> None:
        """Legal JSON string contents survive raw and escaped event encodings."""
        text = "a\u2028b\u2029c\x85d\ve\ff"
        for ascii_only in (False, True):
            for eol in ("\n", "\r\n", "\r"):
                event = {
                    "type": "item.completed",
                    "item": {"type": "agent_message", "text": text},
                }
                raw = (
                    json.dumps(event, ensure_ascii=ascii_only) + eol
                ).encode()
                self.assertEqual(_decode_codex_events(raw), (event,))
                event["item"]["text"] = json.dumps(
                    {"summary": text}, ensure_ascii=ascii_only
                )
                raw = (
                    json.dumps(event, ensure_ascii=ascii_only) + eol
                ).encode()
                self.assertEqual(
                    CodexCLIProvider._parse_response(raw)[0], {"summary": text}
                )
        for raw in (
            b'{"text":"a\nb"}\n',
            b'{"text":"a\rb"}\n',
            b'{"text":"a\x0bb"}\n',
        ):
            with self.assertRaises(ProviderResponseError):
                _decode_codex_events(raw)


class ProviderWireMetadataTests(unittest.TestCase):
    """Keep raw parser reasons, usage, and measured transport sizes."""

    def test_openai_duplicate_output_preserves_usage_and_raw_size(
        self,
    ) -> None:
        """A valid provider envelope cannot hide duplicate structured fields."""
        provider = OpenAIResponsesProvider(api_key="fixture-key")
        usage = {"input_tokens": 12, "output_tokens": 6}
        raw = json.dumps(
            {
                "status": "completed",
                "usage": usage,
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": '{"items":[],"items":["private"]}',
                            }
                        ],
                    }
                ],
            }
        ).encode()
        with (
            patch(
                "diffstory.narrative._post_openai_request", return_value=raw
            ),
            self.assertRaises(ProviderResponseError) as caught,
        ):
            provider.complete_reply(b"{}", 10)
        self.assertEqual(caught.exception.reason, "malformed_json")
        self.assertEqual(caught.exception.response_bytes, len(raw))
        self.assertEqual(caught.exception.usage["output_tokens"], 6)
        self.assertNotIn("private", str(caught.exception))

    def test_openai_outer_duplicate_fails_before_dict_conversion(self) -> None:
        """Ambiguous provider envelopes have no trusted usage or partial data."""
        provider = OpenAIResponsesProvider(api_key="fixture-key")
        raw = b'{"status":"completed","status":"private"}'
        with (
            patch(
                "diffstory.narrative._post_openai_request", return_value=raw
            ),
            self.assertRaises(ProviderResponseError) as caught,
        ):
            provider.complete_reply(b"{}", 10)
        self.assertEqual(caught.exception.reason, "malformed_json")
        self.assertEqual(caught.exception.response_bytes, len(raw))
        self.assertIsNone(caught.exception.usage)

    def test_codex_duplicate_physical_event_or_final_json(self) -> None:
        """Distinguish malformed JSONL from malformed final JSON."""
        usage = {"input_tokens": 12, "output_tokens": 6}
        completed = (
            json.dumps({"type": "turn.completed", "usage": usage}).encode()
            + b"\n"
        )
        event = json.dumps(
            {
                "type": "item.completed",
                "item": {
                    "type": "agent_message",
                    "text": '{"items":[],"items":["private"]}',
                },
            }
        ).encode()
        for raw, reason in (
            (
                completed + b'{"type":"item.completed","type":"private"}',
                "malformed_jsonl",
            ),
            (event + b"\n" + completed, "malformed_json"),
            (completed, "missing_final"),
        ):
            with (
                self.subTest(reason=reason),
                self.assertRaises(ProviderResponseError) as caught,
            ):
                CodexCLIProvider._parse_response(raw)
            self.assertEqual(caught.exception.reason, reason)
            self.assertEqual(caught.exception.response_bytes, len(raw))
            self.assertEqual(caught.exception.usage["output_tokens"], 6)
            self.assertNotIn("private", str(caught.exception))

    def test_codex_prohibited_type_names_remain_private(self) -> None:
        """An arbitrary tool type cannot enter diagnostics or discard usage."""
        raw = b'{"type":"item.completed","item":{"type":"private_secret_call"}}\n{"type":"turn.completed","usage":{"input_tokens":12,"output_tokens":6}}'
        with self.assertRaises(ProviderResponseError) as caught:
            CodexCLIProvider._parse_response(raw)
        self.assertEqual(caught.exception.reason, "prohibited_tools")
        self.assertEqual(caught.exception.response_bytes, len(raw))
        self.assertEqual(caught.exception.usage["output_tokens"], 6)
        self.assertNotIn("private_secret", str(caught.exception))

    def test_codex_nonzero_exit_keeps_recognized_usage_and_size(self) -> None:
        """Failed CLI authentication retains safe physical-output metadata."""
        provider = CodexCLIProvider(executable="fixture-codex")
        raw = b'{"type":"turn.completed","usage":{"input_tokens":12,"output_tokens":6}}\n'
        with (
            patch.object(provider, "require_credentials"),
            patch(
                "diffstory.narrative._run_bounded_subprocess",
                return_value=(1, raw, b"authentication private-secret"),
            ),
            self.assertRaises(ProviderResponseError) as caught,
        ):
            provider.complete_reply(b'{"schema":{}}', 10)
        self.assertEqual(caught.exception.reason, "auth")
        self.assertEqual(caught.exception.response_bytes, len(raw))
        self.assertEqual(caught.exception.usage["output_tokens"], 6)
        self.assertNotIn("private-secret", str(caught.exception))

    def test_codex_nonobject_and_invalid_event_type_keep_safe_metadata(
        self,
    ) -> None:
        """Malformed physical records cannot lose previously recognized usage."""
        prefix = b'{"type":"turn.completed","usage":{"input_tokens":12,"output_tokens":6}}\n'
        for suffix in (
            b"[]",
            b'{"type":[]}',
            b'{"type":"item.completed","item":[]}',
            b"\xff",
            b'{"type":"turn.completed","usage":null}\n[]',
        ):
            raw = prefix + suffix
            with (
                self.subTest(suffix=suffix),
                self.assertRaises(ProviderResponseError) as caught,
            ):
                CodexCLIProvider._parse_response(raw)
            self.assertEqual(caught.exception.reason, "malformed_jsonl")
            self.assertEqual(caught.exception.response_bytes, len(raw))
            self.assertEqual(caught.exception.usage["output_tokens"], 6)

    def test_bounded_process_overflow_reports_specific_reason_usage_and_bytes(
        self,
    ) -> None:
        """The real subprocess collector reports overflow without exposing stdout."""
        raw_usage = '{"type":"turn.completed","usage":{"input_tokens":12,"output_tokens":6}}\n'
        command = [
            sys.executable,
            "-c",
            f"import sys; sys.stdout.write({raw_usage!r}); sys.stdout.write('x' * 10000); sys.stdout.flush()",
        ]
        with (
            tempfile.TemporaryDirectory() as directory,
            self.assertRaises(ProviderResponseError) as caught,
        ):
            _run_bounded_subprocess(
                _ProcessRequest(
                    command=command,
                    prompt=b"",
                    cwd=directory,
                    environment={},
                    timeout=10,
                    output_limit=200,
                )
            )
        self.assertEqual(caught.exception.reason, "response_limit")
        self.assertEqual(caught.exception.response_bytes, 201)
        self.assertEqual(caught.exception.usage["output_tokens"], 6)


if __name__ == "__main__":
    unittest.main()
