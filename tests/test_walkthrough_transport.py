"""Check real provider subprocess termination without external model calls."""

from __future__ import annotations

import contextlib
import json
import os
import signal
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from diffstory.providers import CodexCLIProvider
from diffstory.providers import OpenAIResponsesProvider
from diffstory.walkthrough_runtime import RunLimitError
from diffstory.walkthrough_runtime import RunLimits
from diffstory.walkthrough_runtime import WalkthroughRuntime


class WalkthroughTransportTests(unittest.TestCase):
    """Keep adapter startup, stalled I/O, and cleanup inside the run allowance."""

    @staticmethod
    def _provider(root: Path, source: str) -> CodexCLIProvider:
        """Use a local executable in place of Codex; never call a model."""
        executable = root / "fixture-codex"
        executable.write_text(f"#!{sys.executable}\n{source}")
        executable.chmod(0o700)
        return CodexCLIProvider(
            model="gpt-6.1-sol", executable=str(executable)
        )

    @staticmethod
    def _body(provider: CodexCLIProvider) -> bytes:
        """Prepare the same framing used for a real structured request."""
        return provider.prepare("Fixture", {}, "fixture", {}, 100)

    def test_real_transport_retains_output_and_measured_usage(self) -> None:
        """Successful supervision must preserve the actual adapter result."""
        events = [
            {
                "type": "item.completed",
                "item": {"type": "agent_message", "text": '{"ok":true}'},
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
        with tempfile.TemporaryDirectory() as directory:
            provider = self._provider(
                Path(directory),
                "import sys\nsys.stdin.buffer.read()\n"
                f"print({chr(10).join(json.dumps(e) for e in events)!r})\n",
            )
            with WalkthroughRuntime(provider, "fixture") as runtime:
                value = runtime.call(self._body(provider), lambda v: v)
                usage = runtime.finish()
            self.assertEqual(value, {"ok": True})
            self.assertEqual(
                usage,
                {
                    "input_tokens": 8,
                    "cached_input_tokens": 2,
                    "output_tokens": 3,
                    "calls": 1,
                    "unreported_calls": 0,
                },
            )

    def test_openai_supervisor_uses_a_local_responses_fixture(self) -> None:
        """The production OpenAI adapter crosses supervision without a live call."""
        fixture = {
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": '{"accepted":true}',
                        }
                    ],
                }
            ],
            "usage": {
                "input_tokens": 8,
                "input_tokens_details": {"cached_tokens": 2},
                "output_tokens": 3,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request_path = root / "request.json"
            response_bytes = json.dumps(fixture).encode("utf-8")
            fixture_source = (
                "import json\n"
                "from pathlib import Path\n"
                "from diffstory import providers\n"
                f"REQUEST_PATH = {str(request_path)!r}\n"
                f"RESPONSE_BYTES = {response_bytes!r}\n"
                "def fixture_post(api_key, body, timeout, *, response_byte_limit):\n"
                "    request = {'api_key': api_key, 'body': json.loads(body),\n"
                "               'timeout': timeout, 'response_byte_limit': response_byte_limit}\n"
                "    Path(REQUEST_PATH).write_text(json.dumps(request), encoding='utf-8')\n"
                "    return RESPONSE_BYTES\n"
                "providers._post_openai_request = fixture_post\n"
            )
            (root / "sitecustomize.py").write_text(
                fixture_source,
                encoding="utf-8",
            )
            existing_pythonpath = os.environ.get("PYTHONPATH")
            fixture_pythonpath = str(root)
            if existing_pythonpath:
                fixture_pythonpath += os.pathsep + existing_pythonpath
            provider = OpenAIResponsesProvider(api_key="fixture-secret")
            body = provider.prepare(
                "Return accepted JSON.",
                {"source": "local fixture"},
                "fixture",
                {"type": "object"},
                100,
            )
            with (
                patch.dict(os.environ, {"PYTHONPATH": fixture_pythonpath}),
                WalkthroughRuntime(
                    provider,
                    "openai-fixture",
                    limits=RunLimits(seconds=10),
                ) as runtime,
            ):
                result = runtime.call(body, lambda value: value)
                usage = runtime.finish()
            received = json.loads(request_path.read_text(encoding="utf-8"))

        self.assertEqual(result, {"accepted": True})
        self.assertEqual(usage["input_tokens"], 8)
        self.assertEqual(usage["cached_input_tokens"], 2)
        self.assertEqual(usage["output_tokens"], 3)
        self.assertEqual(usage["calls"], 1)
        self.assertEqual(received["api_key"], "fixture-secret")
        self.assertGreater(received["timeout"], 0)
        self.assertFalse(received["body"]["store"])
        self.assertEqual(received["body"]["model"], provider.model)

    @unittest.skipIf(os.name == "nt", "POSIX executable and process signals")
    def test_stalled_pipe_cleanup_does_not_extend_run_or_leave_work_running(
        self,
    ) -> None:
        """An exited CLI's descendant can hold stdout open past socket timeouts."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            heartbeat = root / "heartbeat"
            pid_file = root / "descendant.pid"
            descendant = (
                "import os,time\n"
                "from pathlib import Path\n"
                f"Path({str(pid_file)!r}).write_text(str(os.getpid()))\n"
                "while True:\n"
                f"    with open({str(heartbeat)!r}, 'ab') as stream:\n"
                "        stream.write(b'.')\n"
                "    time.sleep(.02)\n"
            )
            source = (
                "import subprocess,sys\n"
                "sys.stdin.buffer.read()\n"
                f"subprocess.Popen([sys.executable, '-c', {descendant!r}])\n"
            )
            provider = self._provider(root, source)
            started = time.monotonic()
            try:
                with (
                    self.assertRaises(RunLimitError),
                    WalkthroughRuntime(
                        provider, "stalled", limits=RunLimits(seconds=3)
                    ) as runtime,
                ):
                    runtime.call(self._body(provider), lambda v: v)
                self.assertLess(time.monotonic() - started, 3.6)
                # Ensure the fixture actually reached the blocked-pipe case.
                self.assertTrue(pid_file.exists())
                stopped_size = heartbeat.stat().st_size
                time.sleep(0.1)
                self.assertEqual(heartbeat.stat().st_size, stopped_size)
                self.assertNotEqual(runtime.record["state"], "complete")
            finally:
                if pid_file.exists():
                    with contextlib.suppress(ProcessLookupError):
                        os.kill(int(pid_file.read_text()), signal.SIGKILL)


if __name__ == "__main__":
    unittest.main()
