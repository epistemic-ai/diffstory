"""Exercise provider request and response boundaries without live services."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch
from urllib.error import HTTPError

from diffstory.providers import MAX_STDERR_BYTES
from diffstory.providers import OPENAI_URL
from diffstory.providers import CodexCLIProvider
from diffstory.providers import OpenAIResponsesProvider
from diffstory.providers import ProviderResponseError
from diffstory.providers import _classify_codex_stderr
from diffstory.providers import _ProcessRequest
from diffstory.providers import _RejectRedirects
from diffstory.providers import _run_bounded_subprocess


def _openai_response(
    output_text: str,
    usage: object = None,
    *,
    status: str = "completed",
    incomplete_reason: str | None = None,
) -> bytes:
    """Build one local Responses API fixture.

    Args:
        output_text: Structured JSON text returned by the fixture.
        usage: Raw token usage to expose to the adapter.
        status: Provider completion status.
        incomplete_reason: Optional status detail for incomplete output.

    Returns:
        UTF-8 JSON response bytes.
    """
    response = {
        "status": status,
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": output_text}],
            }
        ],
        "usage": usage,
    }
    if incomplete_reason is not None:
        response["incomplete_details"] = {"reason": incomplete_reason}
    return json.dumps(response).encode("utf-8")


def _codex_events(*events: dict) -> bytes:
    """Encode ordered local Codex events as JSONL.

    Args:
        *events: Event objects to serialize in output order.

    Returns:
        UTF-8 JSONL bytes.
    """
    return "\n".join(json.dumps(event) for event in events).encode("utf-8")


class OpenAIProviderTests(unittest.TestCase):
    """Keep the Responses API adapter strict, bounded, and credential safe."""

    def test_prepared_request_preserves_provider_contract_and_source(
        self,
    ) -> None:
        """The encoded request keeps its provider contract and source text."""
        provider = OpenAIResponsesProvider(api_key="fixture-key")
        schema = {
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
            "additionalProperties": False,
        }
        body = provider.prepare("system text", {"x": "é"}, "schema", schema, 7)
        request = json.loads(body)

        self.assertEqual(request["model"], "gpt-6.1-sol")
        self.assertIs(type(request["store"]), bool)
        self.assertFalse(request["store"])
        self.assertEqual(request["max_output_tokens"], 7)
        self.assertEqual(
            request["input"],
            [
                {"role": "system", "content": "system text"},
                {"role": "user", "content": '{"x":"é"}'},
            ],
        )
        self.assertEqual(
            request["text"],
            {
                "format": {
                    "type": "json_schema",
                    "name": "schema",
                    "strict": True,
                    "schema": schema,
                }
            },
        )

    def test_completed_response_keeps_usage_and_rejects_redirects(
        self,
    ) -> None:
        """A redirect could forward credentials to an unapproved endpoint."""
        usage = {
            "input_tokens": 8,
            "input_tokens_details": {"cached_tokens": 2},
            "output_tokens": 3,
        }
        raw = _openai_response('{"accepted":true}', usage)
        opener = MagicMock()
        opener.open.return_value.__enter__.return_value.read.return_value = raw
        provider = OpenAIResponsesProvider(api_key="fixture-secret")
        body = provider.prepare(
            "Return JSON.", {"source": "fixture"}, "fixture", {}, 100
        )
        with patch(
            "diffstory.providers.build_opener", return_value=opener
        ) as build:
            reply = provider.complete_reply(body, 5)

        self.assertEqual(reply.response, {"accepted": True})
        self.assertEqual(
            reply.usage,
            {
                "input_tokens": 8,
                "cached_input_tokens": 2,
                "output_tokens": 3,
            },
        )
        self.assertEqual(reply.response_bytes, len(raw))
        request = opener.open.call_args.args[0]
        self.assertEqual(request.data, body)
        self.assertEqual(
            request.get_header("Authorization"), "Bearer fixture-secret"
        )
        self.assertEqual(opener.open.call_args.kwargs["timeout"], 5)
        self.assertEqual(opener.open.call_count, 1)
        redirect_handler = build.call_args.args[0]
        self.assertIsInstance(redirect_handler, _RejectRedirects)
        with self.assertRaisesRegex(ValueError, "redirect rejected"):
            redirect_handler.redirect_request(
                None, None, 302, "Found", {}, "https://example.invalid"
            )

    def test_malformed_truncated_and_incomplete_output_is_rejected(
        self,
    ) -> None:
        """Partial or ambiguous provider output could become a false result."""
        usage = {
            "input_tokens": 12,
            "cached_input_tokens": 0,
            "output_tokens": 6,
        }
        cases = [
            (
                b'{"status":"completed","status":"private"}',
                "malformed_json",
                None,
            ),
            (b'{"status":"completed"', "malformed_json", None),
            (
                _openai_response('{"items":[],"items":["private"]}', usage),
                "malformed_json",
                usage,
            ),
            (
                _openai_response(
                    "{}",
                    usage,
                    status="incomplete",
                    incomplete_reason="max_output_tokens",
                ),
                "output_limit",
                usage,
            ),
        ]
        provider = OpenAIResponsesProvider(api_key="fixture-key")
        for raw, reason, expected_usage in cases:
            with (
                self.subTest(reason=reason, raw=raw),
                patch(
                    "diffstory.providers._post_openai_request",
                    return_value=raw,
                ),
                self.assertRaises(ProviderResponseError) as caught,
            ):
                provider.complete_reply(b"{}", 5)
            self.assertEqual(caught.exception.reason, reason)
            self.assertEqual(caught.exception.usage, expected_usage)
            self.assertEqual(caught.exception.response_bytes, len(raw))
            self.assertNotIn("private", str(caught.exception))

    def test_invalid_usage_is_not_counted(self) -> None:
        """Invalid token totals could corrupt run-wide usage accounting."""
        cases = [
            None,
            {"input_tokens": True, "output_tokens": 1},
            {"input_tokens": -1, "output_tokens": 1},
            {
                "input_tokens": 8,
                "input_tokens_details": {"cached_tokens": 9},
                "output_tokens": 1,
            },
        ]
        provider = OpenAIResponsesProvider(api_key="fixture-key")
        for usage in cases:
            raw = _openai_response('{"ok":true}', usage)
            with (
                self.subTest(usage=usage),
                patch(
                    "diffstory.providers._post_openai_request",
                    return_value=raw,
                ),
            ):
                reply = provider.complete_reply(b"{}", 5)
            self.assertEqual(reply.response, {"ok": True})
            self.assertIsNone(reply.usage)

    def test_http_error_hides_response_and_credential(self) -> None:
        """Provider diagnostics could leak an API key or private error body."""
        private_body_text = "private-provider-error"
        api_key = "fixture-api-key"
        error = HTTPError(
            OPENAI_URL,
            403,
            "Forbidden",
            {},
            io.BytesIO(private_body_text.encode()),
        )
        opener = MagicMock()
        opener.open.side_effect = error
        provider = OpenAIResponsesProvider(api_key=api_key)
        with (
            patch("diffstory.providers.build_opener", return_value=opener),
            self.assertRaises(ProviderResponseError) as caught,
        ):
            provider.complete(b"{}", 1)
        self.assertTrue(error.fp.closed)
        self.assertNotIn(private_body_text, str(caught.exception))
        self.assertNotIn(api_key, str(caught.exception))

    def test_response_byte_limit_stops_openai_body_collection(self) -> None:
        """An oversized HTTP body could exceed the configured memory bound."""
        raw = b"x" * 9
        opener = MagicMock()
        opener.open.return_value.__enter__.return_value.read.return_value = raw
        provider = OpenAIResponsesProvider(api_key="fixture-key")
        with (
            patch("diffstory.providers.build_opener", return_value=opener),
            self.assertRaises(ProviderResponseError) as caught,
        ):
            provider.complete_reply(b"{}", 1, response_byte_limit=8)
        self.assertEqual(caught.exception.reason, "response_limit")
        self.assertEqual(caught.exception.response_bytes, 9)
        self.assertEqual(
            opener.open.return_value.__enter__.return_value.read.call_args.args,
            (9,),
        )

    def test_invalid_api_key_is_rejected_before_network_access(self) -> None:
        """A malformed credential could enter headers or diagnostics."""
        invalid_header_value = "fixture-api-key\n"
        provider = OpenAIResponsesProvider(api_key=invalid_header_value)
        with (
            patch("diffstory.providers.build_opener") as build,
            self.assertRaisesRegex(
                ValueError, "invalid header characters"
            ) as caught,
        ):
            provider.complete(b"{}", 1)
        self.assertNotIn("fixture-api-key", str(caught.exception))
        build.assert_not_called()


class CodexProviderTests(unittest.TestCase):
    """Keep Codex CLI requests read-only and bound at the process boundary."""

    def test_unknown_model_without_capacity_is_rejected(self) -> None:
        """Unknown model capacity is rejected before planning a request."""
        with self.assertRaisesRegex(ValueError, "no known context capacity"):
            CodexCLIProvider(model="unlisted-model")

    def test_request_is_exact_read_only_and_credential_isolated(self) -> None:
        """A CLI launch could gain tools or inherit direct API credentials."""
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
        expected_body = json.dumps(
            {
                "system": "Summarize evidence.",
                "data": {"source": "value"},
                "schema_name": "summary",
                "schema": schema,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        self.assertEqual(body, expected_body)
        raw = _codex_events(
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
        )
        process = MagicMock()
        process.stdin = MagicMock()
        process.stdin.write.return_value = len(provider.request_prefix) + len(
            body
        )
        process.stdin.flush.return_value = None
        process.stdin.close.return_value = None
        process.stdout = io.BytesIO(raw)
        process.stderr = io.BytesIO(b"")
        process.wait.return_value = 0
        process.returncode = 0
        launch = {}

        def start_process(command: list[str], **kwargs: object) -> MagicMock:
            """Capture temporary launch settings before cleanup."""
            launch["command"] = command
            launch["environment"] = kwargs["env"].copy()
            launch["options"] = kwargs
            launch["cwd"] = kwargs["cwd"]
            schema_path = Path(command[command.index("--output-schema") + 1])
            launch["schema"] = json.loads(schema_path.read_text())
            return process

        with (
            patch.object(provider, "require_credentials"),
            patch(
                "diffstory.providers.subprocess.Popen",
                side_effect=start_process,
            ),
            patch.dict(
                os.environ,
                {
                    "OPENAI_API_KEY": "parent-openai-secret",
                    "CODEX_API_KEY": "parent-codex-secret",
                },
            ),
        ):
            reply = provider.complete_reply(body, 7)

        command = launch["command"]
        disabled_tools = [
            command[index + 1]
            for index, argument in enumerate(command[:-1])
            if argument == "--disable"
        ]
        self.assertEqual(reply.response, {"summary": "ok"})
        self.assertEqual(reply.response_bytes, len(raw))
        self.assertEqual(
            reply.usage,
            {
                "input_tokens": 8,
                "cached_input_tokens": 2,
                "output_tokens": 3,
            },
        )
        self.assertEqual(launch["schema"], schema)
        self.assertTrue(
            Path(launch["cwd"]).name.startswith("diffstory-codex-")
        )
        self.assertNotIn("OPENAI_API_KEY", launch["environment"])
        self.assertNotIn("CODEX_API_KEY", launch["environment"])
        self.assertIn("--ephemeral", command)
        self.assertIn("--ignore-user-config", command)
        self.assertIn("--ignore-rules", command)
        self.assertEqual(command[command.index("--sandbox") + 1], "read-only")
        self.assertEqual(
            disabled_tools,
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
        options = launch["options"]
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
        self.assertEqual(
            bytes(process.stdin.write.call_args.args[0]),
            provider.request_prefix + body,
        )
        process.stdin.close.assert_called_once()

    def test_invalid_transcripts_keep_only_safe_measured_usage(self) -> None:
        """Malformed, incomplete, or tool-using CLI output could be accepted."""
        usage = {
            "input_tokens": 12,
            "cached_input_tokens": 0,
            "output_tokens": 6,
        }
        completed = _codex_events({"type": "turn.completed", "usage": usage})
        cases = [
            (
                completed + b'\n{"type":"item.completed","type":"private"}',
                "malformed_jsonl",
            ),
            (
                _codex_events(
                    {
                        "type": "item.completed",
                        "item": {
                            "type": "agent_message",
                            "text": '{"items":[],"items":["private"]}',
                        },
                    },
                    {"type": "turn.completed", "usage": usage},
                ),
                "malformed_json",
            ),
            (completed, "missing_final"),
            (
                _codex_events(
                    {
                        "type": "item.completed",
                        "item": {"type": "private_secret_call"},
                    },
                    {"type": "turn.completed", "usage": usage},
                ),
                "prohibited_tools",
            ),
        ]
        provider = CodexCLIProvider(executable="fixture-codex")
        for raw, reason in cases:
            with (
                self.subTest(reason=reason),
                patch.object(provider, "require_credentials"),
                patch(
                    "diffstory.providers._run_bounded_subprocess",
                    return_value=(0, raw, b""),
                ),
                self.assertRaises(ProviderResponseError) as caught,
            ):
                provider.complete_reply(b'{"schema":{}}', 5)
            self.assertEqual(caught.exception.reason, reason)
            self.assertEqual(caught.exception.usage, usage)
            self.assertEqual(caught.exception.response_bytes, len(raw))
            self.assertNotIn("private", str(caught.exception))

    def test_invalid_usage_is_not_counted(self) -> None:
        """Invalid CLI token totals could inflate or understate run usage."""
        cases = [
            {"input_tokens": True, "output_tokens": 1},
            {"input_tokens": -1, "output_tokens": 1},
            {
                "input_tokens": 8,
                "cached_input_tokens": 9,
                "output_tokens": 1,
            },
        ]
        provider = CodexCLIProvider(executable="fixture-codex")
        for usage in cases:
            raw = _codex_events(
                {
                    "type": "item.completed",
                    "item": {"type": "agent_message", "text": '{"ok":true}'},
                },
                {"type": "turn.completed", "usage": usage},
            )
            with (
                self.subTest(usage=usage),
                patch.object(provider, "require_credentials"),
                patch(
                    "diffstory.providers._run_bounded_subprocess",
                    return_value=(0, raw, b""),
                ),
            ):
                reply = provider.complete_reply(b'{"schema":{}}', 5)
            self.assertEqual(reply.response, {"ok": True})
            self.assertIsNone(reply.usage)

    def test_failed_cli_sanitizes_bounded_stderr_and_keeps_usage(self) -> None:
        """A failed child could leak stderr or discard valid measured usage."""
        usage = {
            "input_tokens": 12,
            "cached_input_tokens": 0,
            "output_tokens": 6,
        }
        raw = _codex_events({"type": "turn.completed", "usage": usage})
        private_stderr = b"unauthorized private-secret " + b"x" * (
            MAX_STDERR_BYTES * 4
        )
        process = MagicMock()
        process.stdin = MagicMock()
        process.stdin.write.return_value = len(
            CodexCLIProvider.request_prefix
        ) + len(b'{"schema":{}}')
        process.stdout = io.BytesIO(raw)
        process.stderr = io.BytesIO(private_stderr)
        process.wait.return_value = 1
        process.returncode = 1
        provider = CodexCLIProvider(executable="fixture-codex")
        with (
            patch.object(provider, "require_credentials"),
            patch(
                "diffstory.providers.subprocess.Popen", return_value=process
            ),
            patch(
                "diffstory.providers._classify_codex_stderr",
                wraps=_classify_codex_stderr,
            ) as classify,
            self.assertRaises(ProviderResponseError) as caught,
        ):
            provider.complete_reply(b'{"schema":{}}', 5)
        self.assertEqual(caught.exception.category, "authentication")
        self.assertEqual(caught.exception.usage, usage)
        self.assertNotIn("private-secret", str(caught.exception))
        self.assertLessEqual(len(classify.call_args.args[0]), MAX_STDERR_BYTES)

    def test_stdout_overflow_terminates_child_and_retains_usage(self) -> None:
        """A child could stream unbounded output or survive the response cap."""
        usage_line = (
            _codex_events(
                {
                    "type": "turn.completed",
                    "usage": {"input_tokens": 12, "output_tokens": 6},
                }
            )
            + b"\n"
        )
        source = (
            "import sys,time; "
            f"sys.stdout.buffer.write({usage_line!r}); "
            "sys.stdout.buffer.write(b'x' * 10000); "
            "sys.stdout.flush(); time.sleep(60)"
        )
        with (
            tempfile.TemporaryDirectory() as directory,
            self.assertRaises(ProviderResponseError) as caught,
        ):
            _run_bounded_subprocess(
                _ProcessRequest(
                    command=[sys.executable, "-c", source],
                    prompt=b"",
                    cwd=directory,
                    environment=os.environ.copy(),
                    timeout=5,
                    output_limit=200,
                )
            )
        self.assertEqual(caught.exception.reason, "response_limit")
        self.assertEqual(caught.exception.response_bytes, 201)
        self.assertEqual(caught.exception.usage["output_tokens"], 6)


if __name__ == "__main__":
    unittest.main()
