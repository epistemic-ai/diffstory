"""Mocked provider tests for opt-in, bounded, revision-locked narration."""
import io
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from urllib.error import HTTPError
from unittest.mock import MagicMock, patch

from diffstory.analysis import apply_annotations, compile_snapshot
from diffstory.cli import main
from diffstory.narrative import (
    CODEX_DEFAULT_CONTEXT_TOKENS,
    CODEX_MODEL_CAPACITIES,
    MAX_RESPONSE_BYTES,
    LEAF_OUTPUT_RESERVE,
    ORDER_OUTPUT_RESERVE,
    PROCESS_CLEANUP_TIMEOUT_SECONDS,
    PROVIDER_CALL_TIMEOUT_SECONDS,
    CodexCLIProvider,
    Narrator,
    OPENAI_MODEL,
    OPENAI_URL,
    OpenAIResponsesProvider,
    ProviderResponseError,
    _run_bounded_subprocess,
)
from diffstory.render import render


def snapshot(head_source="def calculate(value):\n    return value + 1\n"):
    """Build a two-revision snapshot for a value-transforming function.

    Args:
        head_source: Optional head-side source text.

    Returns:
        A bounded ``diffstory.snapshot.v1`` mapping.
    """
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
        """Configure the fake provider response mode and request log.

        Args:
            mode: ``valid`` for schema-conforming responses or ``malformed`` for an
                intentionally incomplete leaf response.
        """
        self.mode = mode
        self.calls = []
        self.timeouts = []

    def require_credentials(self):
        """Satisfy credential preflight without external authentication.

        Returns:
            ``None``; the fake provider never requires credentials.
        """
        return None

    def prepare(self, system, data, schema_name, schema, max_output_tokens):
        """Serialize fake provider instructions and request data.

        Args:
            system: Provider instructions.
            data: Evidence and context payload.
            schema_name: Structured-output schema name.
            schema: JSON output schema.
            max_output_tokens: Per-call output reservation.

        Returns:
            Encoded JSON request bytes.
        """
        return json.dumps({"system": system, "data": data, "name": schema_name, "schema": schema,
                           "max_output_tokens": max_output_tokens}, separators=(",", ":")).encode()

    def complete(self, body, timeout):
        """Record one fake request and return its configured structured response.

        Args:
            body: Encoded fake request.
            timeout: Unused provider timeout.

        Returns:
            Response mapping and measured fake token usage.
        """
        request = json.loads(body)
        self.timeouts.append(timeout)
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
        elif request["name"] == "diffstory_story_order":
            response = {"group_ids": [group["id"] for group in data["groups"]]}
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


class NarrativeTests(unittest.TestCase):
    def test_generation_and_report_roundtrip_keep_provenance(self):
        """Keep revision binding, unverified labels, and evidence when generated
        annotations are applied and rendered.
        """
        report = compile_snapshot(snapshot())
        provider = FakeProvider()
        narrator = Narrator(provider)
        preview = narrator.preview(report)
        self.assertEqual(preview["calls"], len(report["groups"]) + preview["chunks"] + 2)
        annotations = narrator.generate(report)
        generated = apply_annotations(report, annotations)
        self.assertEqual(generated["generation"]["verification"], "unverified")
        self.assertEqual(generated["generation"]["base_sha"], report["meta"]["base_sha"])
        self.assertIn("Model-generated narration · unverified.", render(generated))
        self.assertTrue(
            all(
                group["narrative"]["provenance"] == "model-generated · unverified"
                for group in generated["groups"]
            )
        )
        saved = json.loads(json.dumps(generated))
        self.assertIn("Model-generated narration · unverified.", render(saved))

    def test_large_single_group_is_split_into_stable_bounded_slices(self):
        """Split a large group into stable bounded slices that retain complete source and coverage.
        """
        source = "def calculate(value):\n" + "".join(f"    value = value + {n}\n" for n in range(1600)) + "    return value\n"
        report = compile_snapshot(snapshot(source))
        narrator = Narrator(FakeProvider())
        chunks = narrator._chunks(report)
        self.assertEqual(len(chunks), len({chunk.id for chunk in chunks}))
        pieces = [piece for chunk in chunks for piece in chunk.pieces]
        self.assertGreater(len(pieces), 1)
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

    def test_oversized_single_line_is_split_without_losing_source(self):
        """Split an oversized source line without losing or reordering any source text.
        """
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
        self.assertEqual("".join(head_fragments), report["changes"][0]["after"]["source"])

    def test_long_line_slices_have_unique_piece_and_chunk_ids(self):
        """Keep long same-line source slices distinct through annotation application.

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

    def test_large_required_citation_sets_split_leaf_chunks(self):
        """Scale output reservations and split chunks to cover many changes.

        Returns:
            ``None``; assertions verify all changes fit within model output limits.
        """
        source = "\n".join(
            f"value_{index} = {index}" for index in range(1_000)
        ) + "\n"
        report = compile_snapshot(snapshot(source))
        provider = FakeProvider()
        narrator = Narrator(provider)

        self.assertGreaterEqual(len(report["changes"]), 1_000)
        chunks = narrator._chunks(report)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(
            all(
                narrator._leaf_output(chunk.change_ids) <= provider.max_output_tokens
                for chunk in chunks
            )
        )
        covered = {change_id for chunk in chunks for change_id in chunk.change_ids}
        self.assertEqual(covered, {change["id"] for change in report["changes"]})

        many_group_ids = [f"group-{index:04d}-" + "a" * 32 for index in range(1_000)]
        self.assertGreater(narrator._order_output(many_group_ids), ORDER_OUTPUT_RESERVE)
        self.assertLessEqual(
            narrator._order_output(many_group_ids), provider.max_output_tokens
        )
        annotations = narrator.generate(report)
        apply_annotations(report, annotations)

    def test_preview_has_no_run_token_or_call_ceilings(self):
        """Preview the request plan without imposing arbitrary token or call ceilings.
        """
        report = compile_snapshot(snapshot())
        preview = Narrator(FakeProvider()).preview(report)
        self.assertGreater(preview["calls"], 0)
        self.assertNotIn("reserved_input_tokens", preview)
        self.assertNotIn("max_calls", preview)

    def test_request_capacity_tracks_model_context(self):
        """Derive input capacity from the provider context and output reservation.
        """
        provider = FakeProvider()
        narrator = Narrator(provider)
        self.assertEqual(
            narrator.capacity.input_upper_bound,
            provider.context_tokens - provider.max_output_tokens,
        )

    def test_provider_calls_use_finite_per_request_timeout(self):
        """Pass a finite timeout to every provider call without run ceilings.
        """
        provider = FakeProvider()
        Narrator(provider).generate(compile_snapshot(snapshot()))
        self.assertTrue(provider.timeouts)
        self.assertEqual(
            set(provider.timeouts), {PROVIDER_CALL_TIMEOUT_SECONDS}
        )

    def test_malformed_provider_response_does_not_create_annotation(self):
        """Reject a provider response without valid source-bound passages.
        """
        report = compile_snapshot(snapshot())
        provider = FakeProvider(mode="malformed")
        narrator = Narrator(provider)
        with self.assertRaisesRegex(ValueError, "no source-bound passages"):
            narrator.generate(report)
        self.assertEqual(len(provider.calls), 1)

    def test_generated_annotations_reject_wrong_revision(self):
        """Reject generated annotations tied to a different head revision.
        """
        report = compile_snapshot(snapshot())
        annotations = Narrator(FakeProvider()).generate(report)
        annotations["head_sha"] = "c" * 40
        with self.assertRaisesRegex(ValueError, "different head revision"):
            apply_annotations(report, annotations)

    def test_generated_annotations_require_all_groups_and_passages(self):
        """Require complete group and passage coverage for generated annotations.
        """
        report = compile_snapshot(snapshot())
        annotations = Narrator(FakeProvider()).generate(report)
        annotations["steps"][0]["passages"] = []
        with self.assertRaisesRegex(ValueError, "missing source-bound passages"):
            apply_annotations(report, annotations)

    def test_generated_report_rejects_stale_coverage_on_render(self):
        """Revalidate generated coverage when rendering a saved report.
        """
        report = compile_snapshot(snapshot())
        generated = apply_annotations(report, Narrator(FakeProvider()).generate(report))
        generated["generation"]["completed_groups"] = []
        with self.assertRaisesRegex(ValueError, "every report group"):
            render(generated)

    def test_aggregate_snapshot_source_limit_is_enforced(self):
        """Enforce the aggregate source-byte limit when compiling a snapshot.
        """
        from unittest.mock import patch
        with patch("diffstory.analysis.MAX_SNAPSHOT_SOURCE_BYTES", 4):
            with self.assertRaisesRegex(ValueError, "aggregate limit"):
                compile_snapshot(snapshot())

    def test_cli_without_narrate_never_calls_provider(self):
        """Avoid provider calls and annotation output unless narration is requested.
        """
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
        """Require transfer confirmation before the first provider request.
        """
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
        """Write candidate annotations and the generated report after explicit opt-in.
        """
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
        """Leave report, HTML, and annotation outputs unwritten when generation fails.
        """
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

    def test_codex_model_capacity_is_not_inherited_from_openai(self):
        """Use Codex model metadata and reject overrides without known capacity.
        """
        default_provider = CodexCLIProvider()
        self.assertEqual(default_provider.context_tokens, CODEX_DEFAULT_CONTEXT_TOKENS)
        for model, (context_tokens, output_tokens) in CODEX_MODEL_CAPACITIES.items():
            with self.subTest(model=model):
                provider = CodexCLIProvider(model=model)
                self.assertEqual(provider.context_tokens, context_tokens)
                self.assertEqual(provider.max_output_tokens, output_tokens)
        with self.assertRaisesRegex(ValueError, "no known context capacity"):
            CodexCLIProvider(model="unlisted-codex-model")

    def test_codex_cli_uses_isolated_flags_scrubs_keys_and_parses_usage(self):
        """Verify Codex flags, temporary cwd, key removal, JSONL, and usage.
        """
        schema = {
            "type": "object",
            "properties": {"summary": {"type": "string"}},
            "required": ["summary"],
            "additionalProperties": False,
        }
        provider = CodexCLIProvider(model="gpt-6.1-sol")
        body = provider.prepare("Summarize evidence.", {"source": "value"}, "summary", schema, 100)
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
        process.stdin.write.return_value = len(provider._PROMPT.encode()) + len(body)
        process.stdout = io.BytesIO(
            "\n".join(json.dumps(event) for event in events).encode()
        )
        process.wait.return_value = 0
        process.returncode = 0
        launch = {}

        def start_process(command, **kwargs):
            """Capture launch details while the provider temp directory exists.

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

        with patch.object(provider, "require_credentials"), \
             patch("diffstory.narrative.subprocess.Popen", side_effect=start_process), \
             patch.dict("os.environ", {
                 "OPENAI_API_KEY": "parent-api-key",
                 "CODEX_API_KEY": "parent-codex-key",
             }):
            result, usage = provider.complete(body, 7)

        command = launch["command"]
        options = launch["options"]
        self.assertEqual(result, {"summary": "ok"})
        self.assertEqual(
            usage,
            {"input_tokens": 8, "cached_input_tokens": 2, "output_tokens": 3},
        )
        self.assertEqual(command[:4], ["codex", "exec", "--model", "gpt-6.1-sol"])
        self.assertIn("--ephemeral", command)
        self.assertIn("--ignore-user-config", command)
        self.assertIn("--ignore-rules", command)
        self.assertIn("--json", command)
        self.assertIn("--skip-git-repo-check", command)
        self.assertEqual(command[command.index("--sandbox") + 1], "read-only")
        self.assertEqual(
            [command[index + 1] for index, arg in enumerate(command[:-1]) if arg == "--disable"],
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
        self.assertEqual(options["stdin"], subprocess.PIPE)
        self.assertEqual(options["stdout"], subprocess.PIPE)
        self.assertEqual(options["stderr"], subprocess.DEVNULL)
        if os.name == "nt":
            self.assertTrue(
                options["creationflags"] & subprocess.CREATE_NEW_PROCESS_GROUP
            )
        else:
            self.assertTrue(options["start_new_session"])
        self.assertTrue(launch["cwd_exists"])
        self.assertTrue(Path(launch["cwd"]).name.startswith("diffstory-codex-"))
        self.assertEqual(command[command.index("--cd") + 1], launch["cwd"])
        self.assertEqual(launch["schema_path"], command[command.index("--output-schema") + 1])
        self.assertEqual(launch["schema"], schema)
        self.assertNotIn("OPENAI_API_KEY", launch["environment"])
        self.assertNotIn("CODEX_API_KEY", launch["environment"])
        prompt = bytes(process.stdin.write.call_args.args[0])
        self.assertTrue(prompt.startswith(provider._PROMPT.encode()))
        self.assertTrue(prompt.endswith(body))
        process.wait.assert_called_once_with(timeout=7)
        process.stdin.close.assert_called_once()

    def test_codex_cli_timeout_is_passed_to_process_and_reported(self):
        """Terminate and report a timed-out Codex subprocess request.
        """
        provider = CodexCLIProvider()
        body = provider.prepare("system", {}, "summary", {"type": "object"}, 100)
        process = MagicMock()
        process.stdin = MagicMock()
        process.stdin.write.return_value = len(provider._PROMPT.encode()) + len(body)
        process.stdout = io.BytesIO(b"")
        process.wait.side_effect = [subprocess.TimeoutExpired(["codex"], 4), -9]
        with patch.object(provider, "require_credentials"), \
             patch("diffstory.narrative.subprocess.Popen", return_value=process), \
             patch("diffstory.narrative._terminate_process_tree") as terminate_tree:
            with self.assertRaisesRegex(ProviderResponseError, "timed out"):
                provider.complete(body, 4)
        process.wait.assert_any_call(timeout=4)
        process.wait.assert_any_call(timeout=PROCESS_CLEANUP_TIMEOUT_SECONDS)
        terminate_tree.assert_called_once_with(process)

    def test_codex_cli_stdout_is_terminated_at_the_response_limit(self):
        """Kill a streaming child as soon as stdout exceeds the configured cap.
        """
        source = (
            "import sys, time; "
            f"sys.stdout.write('x' * {MAX_RESPONSE_BYTES + 1}); "
            "sys.stdout.flush(); time.sleep(60)"
        )
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                ProviderResponseError,
                f"configured {MAX_RESPONSE_BYTES:,}-byte limit",
            ):
                _run_bounded_subprocess(
                    [sys.executable, "-c", source],
                    b"",
                    cwd=directory,
                    environment=os.environ.copy(),
                    timeout=5,
                    output_limit=MAX_RESPONSE_BYTES,
                )

    def test_codex_cli_timeout_terminates_descendants_holding_stdout(self):
        """Terminate descendant processes so inherited stdout cannot hang cleanup.

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
                    [sys.executable, "-c", source],
                    b"",
                    cwd=directory,
                    environment=os.environ.copy(),
                    timeout=0.1,
                    output_limit=MAX_RESPONSE_BYTES,
                )
            self.assertLess(
                time.monotonic() - started,
                PROCESS_CLEANUP_TIMEOUT_SECONDS,
            )

    def test_codex_jsonl_rejects_tool_calls_and_malformed_events(self):
        """Reject Codex output that invokes tools or violates JSONL structure.
        """
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

    def test_codex_nonzero_exit_is_sanitized(self):
        """Report a safe error when Codex exits unsuccessfully.
        """
        provider = CodexCLIProvider()
        body = provider.prepare("system", {}, "summary", {"type": "object"}, 100)
        process = MagicMock()
        process.stdin = MagicMock()
        process.stdin.write.return_value = len(provider._PROMPT.encode()) + len(body)
        process.stdout = io.BytesIO(b"")
        process.wait.return_value = 1
        process.returncode = 1
        with patch.object(provider, "require_credentials"), \
             patch("diffstory.narrative.subprocess.Popen", return_value=process):
            with self.assertRaisesRegex(ProviderResponseError, "request failed"):
                provider.complete(body, 7)

    def test_openai_adapter_uses_structured_output_and_one_attempt(self):
        """Send one structured-output request and preserve sanitized measured usage.
        """
        class Response:
            """Provide deterministic HTTP response bytes to the adapter."""

            def __enter__(self):
                """Return this fake response for use in a ``with`` block.

                Returns:
                    This response object.
                """
                return self

            def __exit__(self, *_):
                """Allow exceptions from the adapter to propagate.

                Args:
                    *_: Exception details supplied by the context manager.

                Returns:
                    ``False`` so an active exception is not suppressed.
                """
                return False
            def read(self, _limit):
                """Return the prepared response bytes.

                Args:
                    _limit: Maximum requested number of bytes; accepted for the HTTP response
                        interface.

                Returns:
                    The prepared HTTP response body as JSON bytes.
                """
                return json.dumps({"status": "completed", "output": [{"type": "message", "content": [
                    {"type": "output_text", "text": '{"summary":"ok"}'}]}],
                    "usage": {
                        "input_tokens": 8,
                        "input_tokens_details": {"cached_tokens": 2},
                        "output_tokens": 3,
                    }}).encode()
        opener = MagicMock()
        opener.open.return_value = Response()
        with patch("diffstory.narrative.build_opener", return_value=opener):
            provider = OpenAIResponsesProvider(api_key="test-secret")
            body = provider.prepare("Return JSON.", {"source": "safe"}, "summary",
                                    {"type": "object"}, 100)
            result, usage = provider.complete(body, 5)
        self.assertEqual(provider.model, "gpt-6.1-sol")
        request = opener.open.call_args.args[0]
        self.assertEqual(opener.open.call_count, 1)
        self.assertEqual(opener.open.call_args.kwargs["timeout"], 5)
        self.assertEqual(request.get_header("Authorization"), "Bearer test-secret")
        self.assertEqual(result, {"summary": "ok"})
        self.assertEqual(
            usage,
            {"input_tokens": 8, "cached_input_tokens": 2, "output_tokens": 3},
        )
        self.assertEqual(json.loads(body)["store"], False)

    def test_openai_http_error_is_sanitized(self):
        """Hide response bodies and API keys from provider HTTP error messages.
        """
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

    def test_openai_invalid_api_key_header_is_sanitized_before_network_access(self):
        """Reject control characters in API keys without exposing their value.

        Returns:
            ``None``; the API key is rejected before constructing a network opener.
        """
        secret = "review-test-secret\n"
        provider = OpenAIResponsesProvider(api_key=secret)
        with patch("diffstory.narrative.build_opener") as build_opener:
            with self.assertRaises(ValueError) as raised:
                provider.complete(b"{}", 1)
        self.assertIn("invalid header characters", str(raised.exception))
        self.assertNotIn("review-test-secret", str(raised.exception))
        build_opener.assert_not_called()


if __name__ == "__main__":
    unittest.main()
