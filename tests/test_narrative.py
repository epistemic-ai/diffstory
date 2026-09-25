"""Mocked provider tests for opt-in, bounded, revision-locked narration."""
import json
from contextlib import redirect_stderr
import io
import tempfile
import unittest
from pathlib import Path
from urllib.error import HTTPError
from unittest.mock import MagicMock, patch

from diffstory.analysis import apply_annotations, compile_snapshot
from diffstory.budget import BudgetLimits
from diffstory.cli import main
from diffstory.narrative import Narrator, OPENAI_MODEL, OPENAI_URL, OpenAIResponsesProvider
from diffstory.render import render


def snapshot(head_source="def calculate(value):\n    return value + 1\n"):
    return {
        "schema": "diffstory.snapshot.v1",
        "meta": {"base_sha": "a" * 40, "head_sha": "b" * 40, "scope": "changed files", "changed_files": 1,
                 "title": "A sample change"},
        "fragments": [
            {"path": "module.py", "side": "base", "text": "def calculate(value):\n    return value\n", "start_line": 1, "scope": "full"},
            {"path": "module.py", "side": "head", "text": head_source, "start_line": 1, "scope": "full"},
        ],
        "warnings": [],
    }


class FakeProvider:
    name = "openai"
    model = OPENAI_MODEL
    destination = OPENAI_URL
    context_tokens = 1_050_000
    max_output_tokens = 128_000

    def __init__(self, *, mode="valid"):
        self.mode = mode
        self.calls = []

    def require_credentials(self):
        return None

    def prepare(self, system, data, schema_name, schema, max_output_tokens):
        return json.dumps({"system": system, "data": data, "name": schema_name, "schema": schema,
                           "max_output_tokens": max_output_tokens}, separators=(",", ":")).encode()

    def complete(self, body, timeout):
        request = json.loads(body)
        data = request["data"]
        self.calls.append(request)
        if self.mode == "malformed":
            return {"summary": "A summary.", "questions": [], "passages": []}, {"input_tokens": 100, "output_tokens": 10}
        if request["name"] == "diffstory_chunk":
            seen = {}
            for piece in data["pieces"]:
                src = piece.get("source")
                if src:
                    seen[piece["change_id"]] = src
                else:
                    seen[piece["change_id"]] = None
            passages = []
            for change_id, src in seen.items():
                passage = {"text": "This source shows how the value is transformed.",
                           "change_ids": [change_id], "view": "definition", "focus": None}
                if src and src["partial"]:
                    passage["focus"] = {"start": src["start"], "end": src["end"]}
                passages.append(passage)
            response = {"summary": "The function transforms the supplied value.", "questions": [], "passages": passages}
        elif request["name"] == "diffstory_summary":
            response = {"summary": "The change updates the value transformation."}
        elif request["name"] == "diffstory_step":
            group = data["group"]
            response = {"title": "Transform the value", "intent": "Update the value transformation.",
                        "why_now": "Read this after its listed prerequisites.", "takeaway": "The function returns an adjusted value.",
                        "invariants": ["The return value is the result of the expression."],
                        "questions": ["Are callers prepared for the changed result?"],
                        "transition": "Next, inspect the following step."}
        elif request["name"] == "diffstory_document":
            response = {"lead": "This change adjusts a value transformation.", "closing": "The source shows the adjusted return path."}
        else:
            raise AssertionError(request["name"])
        return response, {"input_tokens": 120, "output_tokens": 60}


def limits(**updates):
    values = {"context_tokens": 100_000, "request_input_tokens": 20_000, "request_output_tokens": 3_000,
              "total_input_tokens": 100_000, "total_output_tokens": 20_000, "calls": 40, "seconds": 30}
    values.update(updates)
    return BudgetLimits(**values)


class NarrativeTests(unittest.TestCase):
    def test_generation_and_report_roundtrip_keep_provenance(self):
        report = compile_snapshot(snapshot())
        provider = FakeProvider()
        narrator = Narrator(provider, limits=limits())
        preview = narrator.preview(report)
        self.assertEqual(preview["calls"], len(report["groups"]) + preview["chunks"] + 1)
        annotations = narrator.generate(report)
        generated = apply_annotations(report, annotations)
        self.assertEqual(generated["generation"]["verification"], "unverified")
        self.assertEqual(generated["generation"]["base_sha"], report["meta"]["base_sha"])
        self.assertIn("Model-generated narration · unverified.", render(generated))
        self.assertIn("Model generated · unverified", render(generated))
        saved = json.loads(json.dumps(generated))
        self.assertIn("Model-generated narration · unverified.", render(saved))

    def test_large_single_group_is_split_into_stable_bounded_slices(self):
        source = "def calculate(value):\n" + "".join(f"    value = value + {n}\n" for n in range(1600)) + "    return value\n"
        report = compile_snapshot(snapshot(source))
        narrator = Narrator(FakeProvider(), limits=limits(request_input_tokens=20_000))
        chunks = narrator._chunks(report)
        self.assertGreater(len(chunks), 1)
        self.assertEqual(len(chunks), len({chunk.id for chunk in chunks}))
        pieces = [piece for chunk in chunks for piece in chunk.pieces]
        self.assertTrue(all(len(piece["source"]["source"].encode()) <= 12_000 for piece in pieces if piece.get("source")))
        ranges = sorted((piece["source"]["start"], piece["source"]["end"]) for piece in pieces if piece.get("source"))
        self.assertEqual(ranges[0][0], 1)
        self.assertEqual(ranges[-1][1], 1602)
        for left, right in zip(ranges, ranges[1:]):
            self.assertEqual(left[1] + 1, right[0])
        annotations = narrator.generate(report)
        self.assertEqual(len(annotations["generation"]["chunk_coverage"]), len(chunks))
        generated = apply_annotations(report, annotations)
        self.assertEqual(generated["generation"]["covered_changes"], [report["changes"][0]["id"]])

    def test_oversized_single_line_fails_before_provider_calls(self):
        source = "def calculate(): return '" + ("x" * 13_000) + "'\n"
        report = compile_snapshot(snapshot(source))
        provider = FakeProvider()
        with self.assertRaisesRegex(ValueError, "source line.*no provider request was sent"):
            Narrator(provider, limits=limits()).generate(report)
        self.assertEqual(provider.calls, [])

    def test_over_budget_preflight_makes_zero_provider_calls(self):
        report = compile_snapshot(snapshot())
        provider = FakeProvider()
        narrator = Narrator(provider, limits=limits(total_output_tokens=100))
        with self.assertRaisesRegex(ValueError, "output tokens"):
            narrator.generate(report)
        self.assertEqual(provider.calls, [])

    def test_model_specific_limits_are_enforced(self):
        provider = FakeProvider()
        with self.assertRaisesRegex(ValueError, "context budget exceeds"):
            Narrator(provider, limits=limits(context_tokens=provider.context_tokens + 1))
        with self.assertRaisesRegex(ValueError, "output budget exceeds"):
            Narrator(provider, limits=limits(context_tokens=200_000, request_input_tokens=1000,
                                             request_output_tokens=provider.max_output_tokens + 1))

    def test_malformed_provider_response_does_not_create_annotation(self):
        report = compile_snapshot(snapshot())
        provider = FakeProvider(mode="malformed")
        narrator = Narrator(provider, limits=limits())
        with self.assertRaisesRegex(ValueError, "no source-bound passages"):
            narrator.generate(report)
        self.assertEqual(len(provider.calls), 1)

    def test_generated_annotations_reject_wrong_revision(self):
        report = compile_snapshot(snapshot())
        annotations = Narrator(FakeProvider(), limits=limits()).generate(report)
        annotations["head_sha"] = "c" * 40
        with self.assertRaisesRegex(ValueError, "different head revision"):
            apply_annotations(report, annotations)

    def test_generated_annotations_require_all_groups_and_passages(self):
        report = compile_snapshot(snapshot())
        annotations = Narrator(FakeProvider(), limits=limits()).generate(report)
        annotations["steps"][0]["passages"] = []
        with self.assertRaisesRegex(ValueError, "missing source-bound passages"):
            apply_annotations(report, annotations)

    def test_generated_report_rejects_stale_coverage_on_render(self):
        report = compile_snapshot(snapshot())
        generated = apply_annotations(report, Narrator(FakeProvider(), limits=limits()).generate(report))
        generated["generation"]["completed_groups"] = []
        with self.assertRaisesRegex(ValueError, "every report group"):
            render(generated)

    def test_aggregate_snapshot_source_limit_is_enforced(self):
        from unittest.mock import patch
        with patch("diffstory.analysis.MAX_SNAPSHOT_SOURCE_BYTES", 4):
            with self.assertRaisesRegex(ValueError, "aggregate limit"):
                compile_snapshot(snapshot())

    def test_cli_without_narrate_never_calls_provider(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "walkthrough.html"
            snap = snapshot()
            with patch("diffstory.cli.from_github", return_value=snap), \
                 patch.object(OpenAIResponsesProvider, "complete", side_effect=AssertionError("unexpected model request")):
                result = main(["github", "owner/repo#1", "--out", str(out)])
            self.assertEqual(result, 0)
            self.assertTrue(out.exists())
            self.assertFalse(out.with_suffix(".annotations.json").exists())

    def test_cli_narration_requires_confirmation_before_provider_call(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "walkthrough.html"
            provider = FakeProvider()
            with patch("diffstory.cli.from_github", return_value=snapshot()), \
                 patch("diffstory.cli.OpenAIResponsesProvider", return_value=provider), \
                 patch("diffstory.cli.sys.stdin.isatty", return_value=False):
                result = main(["github", "owner/repo#1", "--narrate", "--out", str(out)])
            self.assertEqual(result, 2)
            self.assertEqual(provider.calls, [])
            self.assertFalse(out.exists())

    def test_cli_opt_in_writes_candidate_and_generated_report(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "walkthrough.html"
            provider = FakeProvider()
            with patch("diffstory.cli.from_github", return_value=snapshot()), \
                 patch("diffstory.cli.OpenAIResponsesProvider", return_value=provider):
                result = main(["github", "owner/repo#1", "--narrate", "--yes", "--out", str(out)])
            candidate = out.with_suffix(".annotations.json")
            report_path = out.with_suffix(".report.json")
            self.assertEqual(result, 0)
            self.assertTrue(candidate.exists())
            self.assertTrue(out.exists())
            report = json.loads(report_path.read_text())
            annotations = json.loads(candidate.read_text())
            self.assertEqual(report["generation"]["schema"], "diffstory.generation.v1")
            self.assertEqual(annotations["generation"]["head_sha"], report["meta"]["head_sha"])
            self.assertNotIn("api_key", candidate.read_text())
            self.assertTrue(provider.calls)
            imported = Path(directory) / "candidate-import.html"
            self.assertEqual(main(["render", str(report_path), "--annotations", str(candidate), "--out", str(imported)]), 0)
            self.assertIn("Model-generated narration · unverified.", imported.read_text())

    def test_cli_provider_failure_does_not_write_a_partial_report(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "walkthrough.html"
            provider = FakeProvider(mode="malformed")
            error_output = io.StringIO()
            with patch("diffstory.cli.from_github", return_value=snapshot()), \
                 patch("diffstory.cli.OpenAIResponsesProvider", return_value=provider), \
                 redirect_stderr(error_output):
                result = main(["github", "owner/repo#1", "--narrate", "--yes", "--out", str(out)])
            self.assertEqual(result, 2)
            self.assertIn("source-bound passages", error_output.getvalue())
            self.assertFalse(out.exists())
            self.assertFalse(out.with_suffix(".report.json").exists())
            self.assertFalse(out.with_suffix(".annotations.json").exists())

    def test_openai_adapter_uses_structured_output_and_one_attempt(self):
        class Response:
            def __enter__(self): return self
            def __exit__(self, *_): return False
            def read(self, _limit):
                return json.dumps({"status": "completed", "output": [{"type": "message", "content": [
                    {"type": "output_text", "text": '{"summary":"ok"}'}]}],
                    "usage": {"input_tokens": 8, "output_tokens": 3}}).encode()
        opener = MagicMock()
        opener.open.return_value = Response()
        with patch("diffstory.narrative.build_opener", return_value=opener):
            provider = OpenAIResponsesProvider(api_key="test-secret")
            body = provider.prepare("Return JSON.", {"source": "safe"}, "summary",
                                    {"type": "object"}, 100)
            result, usage = provider.complete(body, 5)
        request = opener.open.call_args.args[0]
        self.assertEqual(opener.open.call_count, 1)
        self.assertEqual(opener.open.call_args.kwargs["timeout"], 5)
        self.assertEqual(request.get_header("Authorization"), "Bearer test-secret")
        self.assertEqual(result, {"summary": "ok"})
        self.assertEqual(usage, {"input_tokens": 8, "output_tokens": 3})
        self.assertEqual(json.loads(body)["store"], False)

    def test_openai_http_error_is_sanitized(self):
        secret = "provider-error-secret"
        error = HTTPError(OPENAI_URL, 403, "Forbidden", {}, io.BytesIO(secret.encode()))
        opener = MagicMock()
        opener.open.side_effect = error
        with patch("diffstory.narrative.build_opener", return_value=opener):
            provider = OpenAIResponsesProvider(api_key="test-secret")
            with self.assertRaises(ValueError) as raised:
                provider.complete(b"{}", 1)
        self.assertTrue(error.fp.closed)
        self.assertNotIn(secret, str(raised.exception))
        self.assertNotIn("test-secret", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
