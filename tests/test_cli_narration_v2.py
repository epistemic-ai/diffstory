"""Offline public CLI checks for explicit v2 narration and durable recovery."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import tempfile
import unittest
from contextlib import ExitStack
from contextlib import redirect_stderr
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from test_analysis import snap
from test_narrative_v2 import AFTER
from test_narrative_v2 import BEFORE
from test_narrative_v2 import FixtureProvider

from diffstory.analysis import compile_snapshot
from diffstory.cli import MAX_INPUT_BYTES
from diffstory.cli import MAX_REPORT_BYTES
from diffstory.cli import _build_parser
from diffstory.cli import _v2_progress_callback
from diffstory.cli import main
from diffstory.cli import read_json
from diffstory.models import MAX_SNAPSHOT_SOURCE_BYTES
from diffstory.narrative import Narrator
from diffstory.narrative_manifest import LedgerParameters
from diffstory.narrative_runtime import ArtifactCallError
from diffstory.narrative_runtime import ArtifactRuntimeError
from diffstory.narrative_runtime import ProgressEvent
from diffstory.narrative_v2 import V2Options

SENTINEL = "SYNTHETIC_PRIVATE_RESPONSE_API_KEY_SENTINEL"


def source_snapshot() -> dict:
    """Supply authored public normalization examples, never private PR source."""
    return snap(
        {f"normal_{n}.py": BEFORE for n in range(2)},
        {f"normal_{n}.py": AFTER for n in range(2)},
        repository="example/project",
        description="Preserve indentation and compare Unicode names consistently.",
    )


def state_hashes(directory: Path) -> dict[str, str]:
    """Compare every persisted byte across refused restarts."""
    return {
        str(p.relative_to(directory)): hashlib.sha256(
            p.read_bytes()
        ).hexdigest()
        for p in directory.rglob("*")
        if p.is_file()
    }


class CliV2Tests(unittest.TestCase):
    """Exercise the actual narrator, validator, persistence and renderer locally."""

    def setUp(self) -> None:
        """Use isolated output/state and prohibit network and provider processes."""
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.out = self.root / "walkthrough.html"
        self.state = self.out.with_suffix(".narration-state")
        self.cache = self.out.with_suffix(".narration-cache")
        self.snapshot = source_snapshot()
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        for target in (
            "socket.socket",
            "urllib.request.urlopen",
            "subprocess.Popen",
            "subprocess.run",
        ):
            self.stack.enter_context(
                patch(
                    target,
                    side_effect=AssertionError("External I/O forbidden"),
                )
            )

    def command(self, *options: str) -> list[str]:
        """Build the supported GitHub command with explicit architecture."""
        return [
            "github",
            "owner/repo#1",
            "--narrate",
            "--yes",
            "--architecture",
            "v2",
            "--concurrency",
            "1",
            "--out",
            str(self.out),
            *options,
        ]

    def run_cli(
        self,
        command: list[str],
        provider: FixtureProvider | None = None,
    ) -> tuple[int, str, str]:
        """Capture user-visible output while substituting only source and transport."""
        with (
            patch(
                "diffstory.cli.from_github",
                return_value=copy.deepcopy(self.snapshot),
            ) as ingest,
            patch(
                "diffstory.cli.OpenAIResponsesProvider",
                return_value=provider or FixtureProvider(),
            ),
            patch(
                "diffstory.cli.CodexCLIProvider",
                return_value=provider or FixtureProvider(),
            ),
            redirect_stdout(io.StringIO()) as output,
            redirect_stderr(io.StringIO()) as errors,
        ):
            code = main(command)
            self.last_ingest = ingest.call_args
        self.assertNotIn(SENTINEL, output.getvalue() + errors.getvalue())
        return code, output.getvalue(), errors.getvalue()

    def test_nondefault_options_and_paths_survive_offline_reload(self) -> None:
        """Honor CLI overrides and decode persisted nondefault ledger parameters."""
        options = V2Options(
            ledger=LedgerParameters(max_span_bytes=256, max_unit_spans=4),
            max_chapter_groups=1,
        )

        class ConfiguredNarrator(Narrator):
            """Select supported nondefault library options for this public path."""

            def __init__(self, *args: object, **kwargs: object) -> None:
                """Keep CLI configuration and add the frozen production options."""
                super().__init__(*args, v2_options=options, **kwargs)

        annotations = self.root / "candidate.json"
        cache = self.root / "selected-cache"
        state = self.root / "selected-state"
        command = self.command(
            "--concurrency",
            "2",
            "--state-dir",
            str(state),
            "--cache-dir",
            str(cache),
            "--annotations-out",
            str(annotations),
            "--max-source-bytes",
            "10000",
            "--max-report-bytes",
            "1000000",
        )
        command[1] = "https://github.com/owner/repo/pull/1"
        with patch("diffstory.cli.Narrator", ConfiguredNarrator):
            code, _, errors = self.run_cli(command)
        self.assertEqual(code, 0, errors)
        self.assertEqual(self.last_ingest.args, (command[1],))
        self.assertEqual(self.last_ingest.kwargs["max_source_bytes"], 10000)
        self.assertTrue(annotations.exists())
        self.assertTrue(state_hashes(state))
        self.assertTrue(state_hashes(cache))
        self.assertFalse(self.state.exists())
        candidate = read_json(str(annotations))
        self.assertEqual(
            candidate["generation"]["ledger_parameters"]["max_span_bytes"], 256
        )
        code, _, errors = self.run_cli(
            [
                "render",
                str(self.out.with_suffix(".report.json")),
                "--out",
                str(self.root / "configured.html"),
            ]
        )
        self.assertEqual(code, 0, errors)

    def test_corrupt_recovery_is_not_replaced_or_bypassed(self) -> None:
        """A persisted parse failure never acquires a new attempt allowance."""
        self.assertEqual(self.run_cli(self.command())[0], 0)
        lineage = next(self.state.glob("recovery-v2/*.json"))
        lineage.write_text('{"' + SENTINEL)
        before = state_hashes(self.state)
        provider = FixtureProvider()
        code, output, errors = self.run_cli(self.command(), provider)
        self.assertEqual(code, 2)
        self.assertIn("state_corrupt", errors)
        self.assertIn("actual attempts 0", output)
        self.assertEqual(provider.calls, [])
        self.assertEqual(state_hashes(self.state), before)

    def test_generate_preview_warm_reuse_and_offline_roundtrip(self) -> None:
        """Count the exact base preflight plan and reload all public saved formats."""
        provider = FixtureProvider()
        report = compile_snapshot(self.snapshot)
        expected = Narrator(
            provider, architecture="v2", recovery_dir=self.root / "preview"
        ).preview(report)
        self.assertEqual(provider.calls, [])
        code, output, errors = self.run_cli(self.command(), provider)
        self.assertEqual(code, 0, errors)
        self.assertIn(
            f"Logical evidence units: {expected['units']} · presentation groups: {expected['groups']}",
            output,
        )
        self.assertIn(
            f"Base detail request plan: {expected['evidence_batches']} packed batches · {expected['calls']} initial sends before reuse/recovery/repairs",
            output,
        )
        self.assertEqual(
            sum(c["data"]["stage"] == "detail" for c in provider.calls),
            expected["calls"],
        )
        self.assertIn("unknown until validated outputs", output)
        self.assertIn("provider batch lineages 1/1", output)
        self.assertIn("targeted repairs 0 · splits 0", output)
        self.assertIn(f"actual attempts {len(provider.calls)}", output)
        self.assertIn("unique reused results: 0", output)
        self.assertIn("Original producers for reused results: 0", output)
        self.assertTrue(state_hashes(self.state))
        self.assertTrue(state_hashes(self.cache))
        annotations = read_json(str(self.out.with_suffix(".annotations.json")))
        saved = read_json(str(self.out.with_suffix(".report.json")))
        self.assertEqual(annotations["schema"], "diffstory.annotations.v2")
        self.assertEqual(
            saved["generation"]["version"], "diffstory.generation.v2"
        )
        self.assertEqual(
            saved["generation"]["current_usage"]["calls"], len(provider.calls)
        )
        self.assertIn("rstrip() removes only trailing", self.out.read_text())
        warm = FixtureProvider()
        code, warm_output, errors = self.run_cli(self.command(), warm)
        self.assertEqual(code, 0, errors)
        self.assertEqual(warm.calls, [])
        self.assertIn(
            "initial sends before reuse/recovery/repairs", warm_output
        )
        self.assertIn("actual attempts 0", warm_output)
        self.assertIn("input 0 (cached 0) · output 0", warm_output)
        warm_saved = read_json(str(self.out.with_suffix(".report.json")))
        receipts = warm_saved["generation"]["receipts"]
        self.assertIn(f"unique reused results: {len(receipts)}", warm_output)
        self.assertEqual(warm_saved["generation"]["current_usage"]["calls"], 0)
        compiler = self.root / "compiler.json"
        compiler.write_text(json.dumps(report))
        with (
            patch.object(
                FixtureProvider,
                "require_credentials",
                side_effect=AssertionError,
            ),
            patch.object(
                FixtureProvider, "prepare", side_effect=AssertionError
            ),
            patch.object(
                FixtureProvider, "complete_reply", side_effect=AssertionError
            ),
        ):
            for source, options in (
                (self.out.with_suffix(".report.json"), []),
                (
                    compiler,
                    [
                        "--annotations",
                        str(self.out.with_suffix(".annotations.json")),
                    ],
                ),
            ):
                reloaded = self.root / f"reload-{len(options)}.html"
                code, _, errors = self.run_cli(
                    ["render", str(source), "--out", str(reloaded), *options]
                )
                self.assertEqual(code, 0, errors)
                self.assertIn(
                    "rstrip() removes only trailing", reloaded.read_text()
                )

    def test_no_cache_and_unresolved_model_keep_state_and_refuse_missing_results(
        self,
    ) -> None:
        """No payload storage is not permission to reset a completed lineage."""
        for suffix, options in (
            ("nocache", ["--no-cache"]),
            ("default", ["--provider", "codex"]),
        ):
            with self.subTest(mode=suffix):
                state = self.root / suffix
                provider = FixtureProvider()
                if suffix == "default":
                    provider.model = "Codex CLI default"
                command = self.command("--state-dir", str(state), *options)
                code, output, errors = self.run_cli(command, provider)
                self.assertEqual(code, 0, errors)
                self.assertIn(
                    "Payload cache disabled. Attempt budgets persist", output
                )
                before = state_hashes(state)
                self.assertTrue(before)
                self.assertFalse(self.cache.exists())
                restarted = FixtureProvider()
                if suffix == "default":
                    restarted.model = "Codex CLI default"
                code, _, errors = self.run_cli(command, restarted)
                self.assertEqual(code, 2)
                self.assertIn("results_unavailable", errors)
                self.assertEqual(restarted.calls, [])
                self.assertEqual(state_hashes(state), before)

    def test_default_state_exists_without_cache_and_explicit_fresh_never_deletes(
        self,
    ) -> None:
        """A deliberate fresh directory is the only CLI action that starts new budgets."""
        command = self.command("--no-cache")
        self.assertEqual(self.run_cli(command)[0], 0)
        before = state_hashes(self.state)
        self.assertTrue(before)
        fresh = self.root / "intentional.narration-state"
        fresh_command = [*command, "--fresh-state", str(fresh)]
        code, output, errors = self.run_cli(fresh_command)
        self.assertEqual(code, 0, errors)
        self.assertIn("Explicit fresh state", output)
        self.assertEqual(state_hashes(self.state), before)
        fresh_before = state_hashes(fresh)
        provider = FixtureProvider()
        code, _, errors = self.run_cli(fresh_command, provider)
        self.assertEqual(code, 2)
        self.assertIn("requires a new directory", errors)
        self.assertEqual(provider.calls, [])
        self.assertEqual(state_hashes(fresh), fresh_before)
        self.assertEqual(state_hashes(self.state), before)

    def test_auth_and_prohibited_tools_stop_and_spent_budget_persists(
        self,
    ) -> None:
        """Stop reasons do not retry on this run or a restart with the same state."""
        for reason in ("auth", "prohibited_tools"):
            with self.subTest(reason=reason):

                def fail(
                    _p: object, _d: dict, _r: dict, reason: str = reason
                ) -> dict:
                    raise ArtifactCallError(
                        reason,
                        usage={"input_tokens": 5, "output_tokens": 7},
                        response_bytes=43,
                    )

                provider = FixtureProvider(fail)
                state = self.root / reason
                command = self.command("--state-dir", str(state), "--no-cache")
                code, output, errors = self.run_cli(command, provider)
                self.assertEqual(code, 2)
                self.assertEqual(len(provider.calls), 1)
                self.assertIn("input 5 (cached 0) · output 7", output)
                self.assertIn(f"reason={reason}", errors)
                self.assertIn("response_bytes=43", errors)
                self.assertIn("attempt=1", errors)
                before = state_hashes(state)
                resumed = FixtureProvider()
                self.assertEqual(self.run_cli(command, resumed)[0], 2)
                self.assertEqual(resumed.calls, [])
                self.assertEqual(state_hashes(state), before)
                self.assertFalse(self.out.exists())

    def test_partial_failure_exhaustion_and_restart_do_not_reset(self) -> None:
        """Retain successful peers and the consumed repair budget after failure."""

        def omit(_p: object, data: dict, response: dict) -> dict:
            if data["stage"] == "detail":
                response["items"] = response["items"][:-1]
            return response

        provider = FixtureProvider(omit)
        code, output, errors = self.run_cli(self.command(), provider)
        self.assertEqual(code, 2)
        self.assertEqual(len(provider.calls), 2)
        self.assertEqual(len(provider.calls[1]["data"]["items"]), 1)
        self.assertIn("targeted repairs 1", output)
        self.assertIn("missing_items=1", errors)
        self.assertIn("valid_items=1", errors)
        self.assertFalse(self.out.exists())
        before = state_hashes(self.state)
        restarted = FixtureProvider()
        code, output, errors = self.run_cli(self.command(), restarted)
        self.assertEqual(code, 2)
        self.assertEqual(restarted.calls, [])
        self.assertIn("actual attempts 0", output)
        self.assertIn("missing_item", errors)
        self.assertEqual(state_hashes(self.state), before)

    def test_interrupt_drains_usage_then_resumes_original_repair(self) -> None:
        """Exit 130 retains the missing peer, original request and spent attempt."""

        def interrupt(
            provider: FixtureProvider, data: dict, response: dict
        ) -> dict:
            if len(provider.calls) == 1:
                response["items"].pop()
            elif data["repair"]:
                raise KeyboardInterrupt
            return response

        first = FixtureProvider(interrupt)
        code, output, errors = self.run_cli(self.command(), first)
        self.assertEqual(code, 130, errors)
        self.assertIn("diffstory: interrupted", errors)
        self.assertIn("actual attempts 2", output)
        self.assertIn("Provider usage unavailable for 1 of 2 calls", output)
        self.assertFalse(self.out.exists())
        second = FixtureProvider()
        code, output, errors = self.run_cli(self.command(), second)
        self.assertEqual(code, 0, errors)
        self.assertEqual(second.calls[0], first.calls[1])
        self.assertIn("retries 1 · targeted repairs 0", output)
        self.assertTrue(self.out.exists())

    def test_split_and_targeted_repair_counters_are_not_unit_counts(
        self,
    ) -> None:
        """A split or invalid peer changes attempts while all logical results survive."""
        for mode in ("split", "wrong_citation", "unknown_item"):
            with self.subTest(mode=mode):

                def fault(
                    provider: FixtureProvider,
                    _data: dict,
                    response: dict,
                    mode: str = mode,
                ) -> dict:
                    if len(provider.calls) == 1:
                        if mode == "split":
                            reason = "response_limit"
                            raise ArtifactCallError(
                                reason, response_bytes=2_000_001
                            )
                        if mode == "wrong_citation":
                            response["items"][-1]["explanations"][0][
                                "citations"
                            ] = [SENTINEL]
                        else:
                            response["items"].pop()
                            response["items"].append({"target": SENTINEL})
                    return response

                provider = FixtureProvider(fault)
                code, output, errors = self.run_cli(
                    self.command(
                        "--state-dir", str(self.root / mode), "--no-cache"
                    ),
                    provider,
                )
                self.assertEqual(code, 0, errors)
                self.assertIn("detail results: 2", output)
                self.assertIn(
                    "splits 1" if mode == "split" else "targeted repairs 1",
                    output,
                )
                if mode != "split":
                    self.assertEqual(
                        len(provider.calls[1]["data"]["items"]), 1
                    )
                self.assertIn(
                    "unknown_items=1" if mode == "unknown_item" else "reason=",
                    errors,
                )

    def test_approval_mutation_is_rejected_before_any_send(self) -> None:
        """Use the actual frozen report, options and provider prepared by preview."""
        for field in ("report", "provider", "options"):
            with self.subTest(field=field):
                provider = FixtureProvider()
                observed = []

                class ObservedNarrator(Narrator):
                    def __init__(
                        self, *args: object, **kwargs: object
                    ) -> None:
                        super().__init__(*args, **kwargs)
                        observed.append(self)  # noqa: B023 # Constructed and consumed within this subtest.

                    def preview(self, report: dict) -> dict:
                        self.report_for_test = report
                        return super().preview(report)

                def change(
                    *,
                    observed: list = observed,
                    field: str = field,
                    provider: FixtureProvider = provider,
                    **_kwargs: object,
                ) -> None:
                    narrator = observed[0]
                    if field == "report":
                        narrator.report_for_test["meta"]["description"] = (
                            SENTINEL
                        )
                    elif field == "provider":
                        provider.destination = SENTINEL
                    else:
                        narrator._v2.options = V2Options(card_wire_bytes=5000)  # noqa: SLF001

                with (
                    patch("diffstory.cli.Narrator", ObservedNarrator),
                    patch(
                        "diffstory.cli._confirm_source_transfer",
                        side_effect=change,
                    ),
                ):
                    code, _, errors = self.run_cli(self.command(), provider)
                self.assertEqual(code, 2)
                self.assertIn("dependencies_changed", errors)
                self.assertEqual(provider.calls, [])

    def test_persisted_report_and_annotation_errors_redact_unknown_keys_and_values(
        self,
    ) -> None:
        """Reject source-controlled schema names and locations without echoing them."""
        self.assertEqual(self.run_cli(self.command())[0], 0)
        saved = read_json(str(self.out.with_suffix(".report.json")))
        annotation = read_json(str(self.out.with_suffix(".annotations.json")))
        compiler = self.root / "compiler.json"
        compiler.write_text(json.dumps(compile_snapshot(self.snapshot)))
        for kind in ("report", "annotations"):
            for mutation in (
                "unknown_key",
                "known_value",
                "schema",
                "wrong_citation",
            ):
                with self.subTest(kind=kind, mutation=mutation):
                    bad = copy.deepcopy(
                        saved if kind == "report" else annotation
                    )
                    detail = bad["generation"]["details"][0]
                    if mutation == "unknown_key":
                        detail[SENTINEL] = True
                    elif mutation == "known_value":
                        detail["explanations"][0]["before"] = {
                            SENTINEL: SENTINEL
                        }
                    elif mutation == "schema":
                        bad["generation"]["version"] = SENTINEL
                    else:
                        citation = detail["explanations"][0]["citations"][0]
                        citation["side"] = (
                            "head" if citation["side"] == "base" else "base"
                        )
                    path = self.root / "bad.json"
                    path.write_text(json.dumps(bad))
                    output = self.root / "bad.html"
                    command = [
                        "render",
                        str(path if kind == "report" else compiler),
                        "--out",
                        str(output),
                    ]
                    if kind == "annotations":
                        command += ["--annotations", str(path)]
                    code, _, errors = self.run_cli(command)
                    self.assertEqual(code, 2)
                    self.assertIn("schema", errors)
                    self.assertFalse(output.exists())

    def test_parser_provider_preflight_and_storage_errors_are_safe(
        self,
    ) -> None:
        """Cover errors before runtime creation as well as provider parser failures."""
        for error in (
            ValueError(SENTINEL),
            OSError(SENTINEL),
            RuntimeError(SENTINEL),
        ):
            with (
                self.subTest(error=type(error).__name__),
                patch.object(
                    FixtureProvider, "require_credentials", side_effect=error
                ),
            ):
                code, _, errors = self.run_cli(self.command())
                self.assertEqual(code, 2)
                self.assertIn("Check the API key", errors)
        for reason in (
            "malformed_json",
            "malformed_jsonl",
            "missing_final",
            "state_corrupt",
            "request_mismatch",
            "dependencies_changed",
        ):
            with (
                self.subTest(reason=reason),
                patch.object(
                    Narrator,
                    "preview",
                    side_effect=ArtifactRuntimeError(reason),
                ),
            ):
                code, _, errors = self.run_cli(self.command())
                self.assertEqual(code, 2)
                self.assertIn(reason, errors)
        self.state.write_text(SENTINEL)
        provider = FixtureProvider()
        code, _, errors = self.run_cli(self.command(), provider)
        self.assertEqual(code, 2)
        self.assertIn("storage", errors)
        self.assertEqual(provider.calls, [])
        self.assertEqual(self.state.read_text(), SENTINEL)
        self.assertEqual(
            list(self.root.glob("*.narration-state")), [self.state]
        )

    def test_malformed_provider_json_never_reaches_stderr(self) -> None:
        """Real wire parsing fails safely and records bounded attempts."""
        provider = FixtureProvider(
            lambda _p, _d, _r: b'{"' + SENTINEL.encode()
        )
        code, output, errors = self.run_cli(self.command(), provider)
        self.assertEqual(code, 2)
        self.assertEqual(len(provider.calls), 2)
        self.assertIn("malformed_json", errors)
        self.assertIn("actual attempts 2", output)
        self.assertFalse(self.out.exists())

    def test_generic_provider_failure_keeps_safe_diagnostic_bounds(
        self,
    ) -> None:
        """Unknown transport exceptions retain telemetry without their text."""

        def fail(_provider: object, _data: dict, _response: dict) -> dict:
            raise RuntimeError(SENTINEL)

        provider = FixtureProvider(fail)
        code, output, errors = self.run_cli(self.command(), provider)
        self.assertEqual(code, 2)
        self.assertEqual(len(provider.calls), 2)
        self.assertIn("reason=generic", errors)
        self.assertIn("task_digest=", errors)
        self.assertIn("input_bound=", errors)
        self.assertIn("missing_items=2", errors)
        self.assertIn("actual attempts 2", output)
        self.assertFalse(self.out.exists())

    def test_confirmation_url_and_source_limit_are_preserved(self) -> None:
        """Only the existing source-transfer confirmation authorizes sends."""
        for pr in ("owner/repo#1", "https://github.com/owner/repo/pull/1"):
            command = self.command("--max-source-bytes", "10000")
            command[1] = pr
            command.remove("--yes")
            provider = FixtureProvider()
            with patch("diffstory.cli.sys.stdin.isatty", return_value=False):
                code, _, errors = self.run_cli(command, provider)
            self.assertEqual(code, 2)
            self.assertIn("Rerun with --yes", errors)
            self.assertEqual(provider.calls, [])
            args = _build_parser().parse_args(command)
            self.assertEqual(args.pr, pr)
            self.assertEqual(args.max_source_bytes, 10000)
        self.assertEqual(
            _build_parser()
            .parse_args(["github", "a/b#1", "--out", "x.html"])
            .architecture,
            "legacy",
        )
        self.assertEqual(MAX_SNAPSHOT_SOURCE_BYTES, 64_000_000)

    def test_progress_throttle_preserves_stage_boundaries_and_batch_labels(
        self,
    ) -> None:
        """Progress completion counts are lineage jobs, never evidence units."""
        narrator = Narrator(
            FixtureProvider(), architecture="v2", recovery_dir=self.state
        )
        events = [
            ProgressEvent(
                event=name,
                stage="detail",
                task_id=None,
                active_calls=1,
                completed=1,
                planned=3,
                reused_checkpoints=77,
                provider_calls=0,
                elapsed_seconds=1.2,
            )
            for name in (
                "stage_started",
                "heartbeat",
                "call_started",
                "heartbeat",
                "stage_completed",
                "stage_started",
            )
        ]
        with (
            patch(
                "diffstory.cli.time.monotonic",
                side_effect=[0, 1, 4.9, 5, 5.1, 5.2],
            ),
            redirect_stdout(io.StringIO()) as output,
        ):
            callback = _v2_progress_callback(lambda: narrator)
            for event in events:
                callback(event)
        lines = output.getvalue().splitlines()
        self.assertEqual(len(lines), 4)
        self.assertTrue(
            all("provider batch lineages 1/3" in line for line in lines)
        )
        self.assertNotIn("77", output.getvalue())
        self.assertIn("reused artifacts (lookup hits) 0", output.getvalue())

    def test_bounded_reload_above_old_limit_and_output_override(self) -> None:
        """Public render reads a large authored file while snapshots keep their limit."""
        path = self.root / "padded.report.json"
        report = compile_snapshot(self.snapshot)
        raw = json.dumps(report).encode()
        size = 109_957_391
        with path.open("wb") as stream:
            stream.write(raw)
            remaining = size - len(raw)
            while remaining:
                chunk = min(remaining, 1_000_000)
                stream.write(b" " * chunk)
                remaining -= chunk
        self.assertGreater(size, MAX_INPUT_BYTES)
        self.assertLess(size, MAX_REPORT_BYTES)
        code, _, errors = self.run_cli(
            ["render", str(path), "--out", str(self.out)]
        )
        self.assertEqual(code, 0, errors)
        with self.assertRaisesRegex(ValueError, "byte limit"):
            read_json(str(path))
        code, _, errors = self.run_cli(
            [
                "render",
                str(path),
                "--out",
                str(self.root / "reject.html"),
                "--max-report-bytes",
                str(size - 1),
            ]
        )
        self.assertEqual(code, 2)
        self.assertIn("byte limit", errors)
        self.assertFalse((self.root / "reject.html").exists())
        small = self.root / "small.json"
        small.write_bytes(raw)
        limit = len(raw) + 1
        code, _, errors = self.run_cli(
            [
                "render",
                str(small),
                "--out",
                str(self.root / "too-small.html"),
                "--max-report-bytes",
                str(limit),
            ]
        )
        self.assertEqual(code, 2)
        self.assertIn("Output JSON needs", errors)
        self.assertFalse((self.root / "too-small.html").exists())
        code, _, errors = self.run_cli(
            [
                "render",
                str(small),
                "--out",
                str(self.root / "override.html"),
                "--max-report-bytes",
                "1000000",
            ]
        )
        self.assertEqual(code, 0, errors)
        for content in (
            '{"value": NaN}',
            '{"value": 1e9999}',
            "[]",
            '{"' + SENTINEL,
        ):
            small.write_text(content)
            code, _, errors = self.run_cli(
                [
                    "render",
                    str(small),
                    "--out",
                    str(self.root / "invalid.html"),
                ]
            )
            self.assertEqual(code, 2)
            self.assertFalse((self.root / "invalid.html").exists())


if __name__ == "__main__":
    unittest.main()
