"""Exercise actual shared limits, persistence, and cache failure boundaries."""

from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import ClassVar
from unittest.mock import patch

from diffstory.compact_packet import canonical_bytes
from diffstory.compact_packet import digest
from diffstory.providers import ProviderResponseError
from diffstory.walkthrough_runtime import RunLimitError
from diffstory.walkthrough_runtime import RunLimits
from diffstory.walkthrough_runtime import WalkthroughRuntime


def accept(value: dict) -> dict:
    """Reject semantically invalid fixture output instead of accepting any JSON."""
    if value != {"accepted": True}:
        raise ValueError
    return value


class ProviderFailureError(ValueError):
    """Carry known failed-call usage and untrusted provider diagnostic text."""

    usage: ClassVar[dict] = {
        "input_tokens": 11,
        "cached_input_tokens": 0,
        "output_tokens": 3,
    }


class RuntimeProvider:
    """Count real fixture calls and active threads under configurable failures.

    Attributes:
        calls: Actual complete invocations, including failed ones.
        active: Calls currently inside the provider.
        max_active: Highest measured concurrent call count.
        failure: None, first-per-request failure, always-fail, or interrupt mode.
        barrier: Optional initial-wave synchronization for a concurrency test.
        delay: Real provider seconds used to test late results.
    """

    name = "fixture"
    model = "runtime-v1"
    destination = "fixture://offline-runtime"
    context_tokens = 1_000_000
    max_output_tokens = 128_000
    input_overhead_bytes = 0

    def __init__(self, failure: str | None = None, delay: float = 0) -> None:
        """Create fresh measured call counters without external side effects."""
        self.calls = self.active = self.max_active = 0
        self.failure = failure
        self.delay = delay
        self.barrier = None
        self.counts = {}
        self.lock = threading.Lock()

    def complete(self, body: bytes, timeout: float) -> tuple[dict, dict]:
        """Return valid output or a configured real failure for this invocation."""
        if timeout <= 0:
            raise TimeoutError
        with self.lock:
            self.calls += 1
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.counts[body] = self.counts.get(body, 0) + 1
            attempt = self.counts[body]
        try:
            if self.barrier is not None and attempt == 1:
                self.barrier.wait(timeout=5)
            time.sleep(self.delay)
            if self.failure == "always" or (
                self.failure == "first" and attempt == 1
            ):
                msg = "untrusted secret diagnostic"
                raise ProviderFailureError(msg)
            if self.failure == "interrupt":
                raise KeyboardInterrupt
            return {"accepted": True}, {
                "input_tokens": 17,
                "cached_input_tokens": 0,
                "output_tokens": 5,
            }
        finally:
            with self.lock:
                self.active -= 1


class WalkthroughRuntimeTests(unittest.TestCase):
    """Prevent silent overspending, deadline extension, and invalid cache reuse."""

    def test_concurrent_retries_share_one_budget(self) -> None:
        """Parallel failures must not each acquire an independent retry allowance."""
        maximum = 6
        with tempfile.TemporaryDirectory() as directory:
            provider = RuntimeProvider("first")
            provider.barrier = threading.Barrier(4)
            root = Path(directory)
            with (
                self.assertRaises(RunLimitError),
                WalkthroughRuntime(
                    provider,
                    "same-input",
                    state_dir=root,
                    limits=RunLimits(requests=maximum),
                ) as runtime,
            ):
                runtime.parallel(
                    [
                        (canonical_bytes({"request": i}), accept)
                        for i in range(4)
                    ]
                )
            saved = json.loads((root / "run.json").read_text())
            self.assertEqual(provider.calls, maximum)
            self.assertEqual(sum(saved["attempts"].values()), maximum)
            self.assertEqual(provider.max_active, 4)
            self.assertEqual(saved["state"], "failed")

    def test_oversized_plan_is_rejected_before_calls(self) -> None:
        """A plan already over its call ceiling must never transfer source."""
        maximum, planned = (2, 3)
        provider = RuntimeProvider()
        with (
            self.assertRaises(RunLimitError),
            WalkthroughRuntime(
                provider, "input", limits=RunLimits(requests=maximum)
            ) as runtime,
        ):
            runtime.parallel(
                [
                    (canonical_bytes({"request": i}), accept)
                    for i in range(planned)
                ]
            )
        self.assertEqual(provider.calls, 0)

    def test_separate_callers_share_the_same_concurrency_ceiling(self) -> None:
        """Concurrent dispatchers must not each receive four provider slots."""
        provider = RuntimeProvider(delay=0.05)
        provider.barrier = threading.Barrier(4)
        with WalkthroughRuntime(provider, "input") as runtime:
            with ThreadPoolExecutor(max_workers=8) as callers:
                futures = [
                    callers.submit(runtime.call, str(i).encode(), accept)
                    for i in range(8)
                ]
                for future in futures:
                    self.assertEqual(future.result(), {"accepted": True})
            usage = runtime.finish()
        self.assertEqual(provider.calls, 8)
        self.assertEqual(provider.max_active, 4)
        self.assertEqual(usage["calls"], 8)

    def test_authentication_and_prohibited_tool_failures_are_not_retried(
        self,
    ) -> None:
        """A retry must not repeat requests after credentials or tool policy fail."""
        for category, reason in [
            ("authentication", "auth"),
            ("invalid_response", "prohibited_tools"),
        ]:
            with self.subTest(category=category, reason=reason):
                provider = RuntimeProvider()
                error = ProviderResponseError(
                    "Fixture failure", category=category, reason=reason
                )
                with (
                    patch.object(
                        provider, "complete", side_effect=error
                    ) as send,
                    self.assertRaises(RunLimitError),
                    WalkthroughRuntime(provider, "input") as runtime,
                ):
                    runtime.call(b"request", accept)
                self.assertEqual(send.call_count, 1)
                self.assertEqual(runtime.report()["calls"], 1)

    def test_failed_call_tokens_remain_measured_without_leaking_diagnostics(
        self,
    ) -> None:
        """Provider failures must retain reported usage and hide untrusted messages."""
        expected_calls = 2
        with tempfile.TemporaryDirectory() as directory:
            provider = RuntimeProvider("always")
            with (
                self.assertRaises(RunLimitError) as caught,
                WalkthroughRuntime(
                    provider, "input", state_dir=Path(directory)
                ) as runtime,
            ):
                runtime.call(b"request", accept)
            self.assertNotIn("secret", str(caught.exception))
            saved = json.loads((Path(directory) / "run.json").read_text())
            self.assertEqual(saved["usage"]["input_tokens"], 22)
            self.assertEqual(saved["usage"]["output_tokens"], 6)
            self.assertEqual(provider.calls, expected_calls)

    def test_resume_retains_an_interrupted_attempt(self) -> None:
        """Interrupting after dispatch must not grant a fresh per-request budget."""
        expected_calls = 2
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            provider = RuntimeProvider("interrupt")
            with (
                self.assertRaises(KeyboardInterrupt),
                WalkthroughRuntime(
                    provider, "input", state_dir=root
                ) as runtime,
            ):
                runtime.call(b"request", accept)
            provider.failure = None
            with WalkthroughRuntime(
                provider, "input", state_dir=root
            ) as runtime:
                runtime.call(b"request", accept)
                usage = runtime.finish()
            self.assertEqual(provider.calls, expected_calls)
            self.assertEqual(usage["calls"], expected_calls)
            self.assertEqual(usage["unreported_calls"], 1)

    def test_resume_does_not_reset_exhausted_retries(self) -> None:
        """Repeated CLI invocations must not create unlimited attempts for one key."""
        invocations, expected_calls = (2, 2)
        with tempfile.TemporaryDirectory() as directory:
            provider = RuntimeProvider("always")
            for _ in range(invocations):
                with (
                    self.assertRaises(RunLimitError),
                    WalkthroughRuntime(
                        provider, "input", state_dir=Path(directory)
                    ) as runtime,
                ):
                    runtime.call(b"request", accept)
            self.assertEqual(provider.calls, expected_calls)

    def test_old_run_state_version_is_rejected_explicitly(self) -> None:
        """An old accounting layout must not be resumed with guessed cache charges."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            provider = RuntimeProvider()
            with WalkthroughRuntime(
                provider, "input", state_dir=root
            ) as runtime:
                runtime.finish()
            path = root / "run.json"
            saved = json.loads(path.read_text())
            saved["version"] = "diffstory.walkthrough-run.v1"
            path.write_bytes(canonical_bytes(saved))

            with (
                self.assertRaisesRegex(
                    RunLimitError, "version is incompatible"
                ),
                WalkthroughRuntime(provider, "input", state_dir=root),
            ):
                self.fail(
                    "An old run record must require a new state directory"
                )
            self.assertEqual(provider.calls, 0)

    def test_cache_binds_evidence_prompt_schema_and_model(self) -> None:
        """Any changed request binding must miss the previous validated result."""
        with tempfile.TemporaryDirectory() as directory:
            provider = RuntimeProvider()
            cache = Path(directory)
            base = {"evidence": "one", "prompt": "one", "schema": "one"}
            cases = [
                (base, "runtime-v1", 1),
                (base, "runtime-v1", 1),
                ({**base, "evidence": "two"}, "runtime-v1", 2),
                ({**base, "prompt": "two"}, "runtime-v1", 3),
                ({**base, "schema": "two"}, "runtime-v1", 4),
                (base, "runtime-v2", 5),
            ]
            for request, model, expected_calls in cases:
                with self.subTest(request=request, model=model):
                    provider.model = model
                    with WalkthroughRuntime(
                        provider, "input", cache_dir=cache
                    ) as runtime:
                        runtime.call(canonical_bytes(request), accept)
                        runtime.finish()
                    self.assertEqual(provider.calls, expected_calls)

            provider.request_prefix = b"Changed adapter-level prompt"
            with WalkthroughRuntime(
                provider, "input", cache_dir=cache
            ) as runtime:
                runtime.call(canonical_bytes(base), accept)
                runtime.finish()
            self.assertEqual(provider.calls, 6)

    def test_modified_cache_is_revalidated_without_an_extra_call(self) -> None:
        """A valid JSON/checksum pair must still pass the semantic validator."""
        invalid = {"accepted": False}
        with tempfile.TemporaryDirectory() as directory:
            provider = RuntimeProvider()
            cache = Path(directory)
            with WalkthroughRuntime(
                provider, "input", cache_dir=cache
            ) as runtime:
                runtime.call(b"request", accept)
            path = next(cache.glob("*.json"))
            entry = json.loads(path.read_text())
            entry.update(result=invalid, digest=digest(invalid))
            path.write_bytes(canonical_bytes(entry))
            with (
                self.assertRaises(RunLimitError),
                WalkthroughRuntime(
                    provider, "input", cache_dir=cache
                ) as runtime,
            ):
                runtime.call(b"request", accept)
            self.assertEqual(provider.calls, 1)

    def test_preparation_time_can_exhaust_the_deadline(self) -> None:
        """Local preparation must consume the same budget as provider work."""
        prepared = 241
        provider = RuntimeProvider()
        with (
            self.assertRaises(RunLimitError),
            WalkthroughRuntime(
                provider, "input", preparation_seconds=prepared
            ) as runtime,
        ):
            runtime.call(b"request", accept)
        self.assertEqual(provider.calls, 0)

    def test_late_result_is_rejected_and_deadline_is_not_extended_on_resume(
        self,
    ) -> None:
        """A late result or resumed invocation must not restart the run clock."""
        seconds, delay = (0.5, 0.6)
        with tempfile.TemporaryDirectory() as directory:
            provider = RuntimeProvider(delay=delay)
            root = Path(directory)
            with (
                self.assertRaises(RunLimitError),
                WalkthroughRuntime(
                    provider,
                    "input",
                    state_dir=root,
                    limits=RunLimits(seconds=seconds),
                ) as runtime,
            ):
                runtime.call(b"request", accept)
            with (
                self.assertRaises(RunLimitError),
                WalkthroughRuntime(
                    provider,
                    "input",
                    state_dir=root,
                    limits=RunLimits(seconds=seconds),
                ),
            ):
                self.fail("Expired runs must fail before entering the body")
            self.assertEqual(provider.calls, 1)

    def test_annotation_limit_is_shared_between_successful_requests(
        self,
    ) -> None:
        """Individually small responses must not exceed the run-wide data ceiling."""
        limit = 20
        provider = RuntimeProvider()
        with WalkthroughRuntime(
            provider, "input", limits=RunLimits(annotation_bytes=limit)
        ) as runtime:
            runtime.call(b"first", accept)
            with self.assertRaises(RunLimitError):
                runtime.call(b"second", accept)
            with self.assertRaises(RunLimitError):
                runtime.finish()
        self.assertEqual(provider.calls, 2)

    def test_cached_and_mixed_results_share_the_aggregate_limit(self) -> None:
        """Cached results cannot bypass the shared response-byte ceiling."""
        cases = [
            ("two cached results", 2, 0),
            ("cached then fresh", 1, 1),
        ]
        for name, seeded_results, expected_new_calls in cases:
            with (
                self.subTest(case=name),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                cache = root / "cache"
                state = root / "state"
                provider = RuntimeProvider()
                requests = [b"first", b"second"]
                with WalkthroughRuntime(
                    provider, "seed", cache_dir=cache
                ) as runtime:
                    for body in requests[:seeded_results]:
                        runtime.call(body, accept)

                with (
                    self.assertRaisesRegex(
                        RunLimitError, "shared 1 MiB run limit"
                    ),
                    WalkthroughRuntime(
                        provider,
                        "limited",
                        cache_dir=cache,
                        state_dir=state,
                        limits=RunLimits(annotation_bytes=20),
                    ) as runtime,
                ):
                    runtime.call(requests[0], accept)
                    runtime.call(requests[1], accept)

                saved = json.loads((state / "run.json").read_text())
                self.assertEqual(saved["annotation_bytes"], 17)
                self.assertEqual(
                    provider.calls, seeded_results + expected_new_calls
                )

    def test_cached_completion_resumes_without_a_second_charge(self) -> None:
        """A cache-only completion keeps its byte charge and needs no attempt."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache, state = root / "cache", root / "state"
            provider = RuntimeProvider()
            with WalkthroughRuntime(
                provider, "seed", cache_dir=cache
            ) as runtime:
                runtime.call(b"request", accept)

            for _ in range(2):
                with WalkthroughRuntime(
                    provider, "input", cache_dir=cache, state_dir=state
                ) as runtime:
                    runtime.call(b"request", accept)
                    runtime.finish()
                saved = json.loads((state / "run.json").read_text())
                self.assertEqual(saved["annotation_bytes"], 17)
                self.assertEqual(saved["attempts"], {})
                self.assertEqual(provider.calls, 1)

    def test_failed_validation_bytes_and_attempts_survive_resume(self) -> None:
        """Rejected response bytes and attempts remain after an accepted cache resume."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache, state = root / "cache", root / "state"
            provider = RuntimeProvider()
            validations = 0

            def reject_once(value: dict) -> dict:
                nonlocal validations
                validations += 1
                if validations == 1:
                    raise ValueError
                return accept(value)

            with WalkthroughRuntime(
                provider, "input", cache_dir=cache, state_dir=state
            ) as runtime:
                runtime.call(b"request", reject_once)
                runtime.finish()
            saved = json.loads((state / "run.json").read_text())
            self.assertEqual(saved["annotation_bytes"], 34)
            self.assertEqual(saved["usage"]["calls"], 2)
            self.assertEqual(sum(saved["attempts"].values()), 2)

            with WalkthroughRuntime(
                provider, "input", cache_dir=cache, state_dir=state
            ) as runtime:
                runtime.call(b"request", accept)
                runtime.finish()
            resumed = json.loads((state / "run.json").read_text())
            self.assertEqual(resumed["annotation_bytes"], 34)
            self.assertEqual(resumed["usage"]["calls"], 2)
            self.assertEqual(provider.calls, 2)

    def test_changed_accepted_cached_result_is_rejected(self) -> None:
        """A valid replacement cache entry cannot change an accepted run result."""

        def accept_variant(value: dict) -> dict:
            if value.get("accepted") is not True or set(value) - {
                "accepted",
                "variant",
            }:
                raise ValueError
            return value

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache, state = root / "cache", root / "state"
            provider = RuntimeProvider()
            with WalkthroughRuntime(
                provider, "seed", cache_dir=cache
            ) as runtime:
                runtime.call(b"request", accept)
            with WalkthroughRuntime(
                provider, "input", cache_dir=cache, state_dir=state
            ) as runtime:
                runtime.call(b"request", accept_variant)
                runtime.finish()

            cache_path = next(cache.glob("*.json"))
            entry = json.loads(cache_path.read_text())
            replacement = {"accepted": True, "variant": 2}
            entry.update(result=replacement, digest=digest(replacement))
            cache_path.write_bytes(canonical_bytes(entry))

            with (
                self.assertRaisesRegex(
                    RunLimitError, "changed after this run accepted"
                ),
                WalkthroughRuntime(
                    provider, "input", cache_dir=cache, state_dir=state
                ) as runtime,
            ):
                runtime.call(b"request", accept_variant)
            self.assertEqual(provider.calls, 1)
            saved = json.loads((state / "run.json").read_text())
            self.assertEqual(saved["annotation_bytes"], 17)

    def test_concurrent_run_directory_is_locked(self) -> None:
        """Two processes or runtimes must not independently spend one run record."""
        identity = "input"
        with tempfile.TemporaryDirectory() as directory:
            provider = RuntimeProvider()
            root = Path(directory)
            with (
                WalkthroughRuntime(provider, identity, state_dir=root),
                self.assertRaises(OSError),
                WalkthroughRuntime(provider, identity, state_dir=root),
            ):
                self.fail("A second owner must not acquire the run")
            self.assertEqual(provider.calls, 0)


if __name__ == "__main__":
    unittest.main()
