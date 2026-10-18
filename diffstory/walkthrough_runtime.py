"""One run-wide request budget, deadline, and request-keyed result cache."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .compact_packet import canonical_bytes
from .compact_packet import digest
from .models import GenerationUsage
from .usage import RunUsage

if TYPE_CHECKING:
    from collections.abc import Callable

    from .narrative import NarrativeProvider

RUN_VERSION = "diffstory.walkthrough-run.v2"
CACHE_VERSION = "diffstory.walkthrough-cache.v1"
ATTEMPTS_PER_REQUEST = 2
MAX_RECORD_BYTES = 65_536
MAX_SECONDS = 240.0
MAX_REQUESTS = 12
REQUEST_KEY_LENGTH = 64


class RunLimitError(ValueError):
    """Expose only application-owned diagnostics for a failed bounded run."""


class RunDeadlineError(RunLimitError, TimeoutError):
    """Report a run deadline through both the limit and timeout contracts."""


@dataclass(frozen=True)
class RunLimits:
    """Set hard ceilings shared by preparation, writers, retries, and editing.

    Attributes:
        requests: Total provider attempts including failed calls, at most 12.
        concurrency: Simultaneous calls, at most four.
        seconds: Narration deadline including local preparation, at most 240.
        annotation_bytes: Cumulative canonical JSON bytes of decoded model
            responses, at most 1 MiB. Fresh responses count per attempt,
            including rejected results. A newly accepted cached result counts
            once; exact accepted-result reuse after resume is not charged again.
    """

    requests: int = MAX_REQUESTS
    concurrency: int = 4
    seconds: float = MAX_SECONDS
    annotation_bytes: int = 1_048_576

    def __post_init__(self) -> None:
        """Reject invalid limits rather than silently broadening a run budget."""
        for value, maximum in (
            (self.requests, MAX_REQUESTS),
            (self.concurrency, 4),
            (self.annotation_bytes, 1_048_576),
        ):
            if type(value) is not int or not 1 <= value <= maximum:
                msg = "Narration limits exceed the supported run ceilings"
                raise RunLimitError(msg)
        if (
            type(self.seconds) not in {int, float}
            or not math.isfinite(self.seconds)
            or not 0 < self.seconds <= MAX_SECONDS
        ):
            msg = "Narration deadline must be greater than zero and at most 240 seconds"
            raise RunLimitError(msg)


def provider_identity(provider: NarrativeProvider) -> dict:
    """Bind prepared requests to the exact adapter destination and model."""
    return {
        "provider": provider.name,
        "model": provider.model,
        "destination": provider.destination,
        "context_tokens": provider.context_tokens,
        "max_output_tokens": provider.max_output_tokens,
        "input_overhead_bytes": provider.input_overhead_bytes,
        "request_prefix_sha256": hashlib.sha256(
            getattr(provider, "request_prefix", b"")
        ).hexdigest(),
    }


def request_key(body: bytes, identity: dict) -> str:
    """Hash exact request bytes, schema/prompt included, and provider identity."""
    return digest(
        {
            "request_sha256": hashlib.sha256(body).hexdigest(),
            "provider": identity,
        }
    )


def _atomic_json(path: Path, value: dict) -> None:
    """Durably replace one small record, leaving an old record intact on failure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(canonical_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _expect_record(condition: bool) -> None:  # noqa: FBT001 # Internal guard for parsed record invariants.
    """Reject an invalid parsed record without exposing any stored values."""
    if not condition:
        raise ValueError


def _read_record(path: Path, limit: int) -> dict:
    """Read one bounded JSON object even if the file grows after inspection."""
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    _expect_record(len(raw) <= limit)
    value = json.loads(raw)
    _expect_record(type(value) is dict)
    return value


class WalkthroughRuntime:
    """Execute bounded requests while retaining spent attempts across resumes.

    Attributes:
        provider: Adapter that must honor the remaining timeout supplied to it.
        limits: Immutable run-wide ceilings; retries share these same ceilings.
        state_dir: Optional durable run directory. None means an ephemeral run.
        cache_dir: Optional request-keyed cache with source-free model results.
        record: Small run identity, attempt counts, usage, and completion state.
        cache_hits: Validated results reused during this invocation.

    Use as a context manager. A process lock prevents concurrent writers to the
    same run record. A resume retains its original wall deadline and attempts;
    starting a new budget requires a different run directory.
    """

    def __init__(  # noqa: PLR0913 # Keep each explicit run control visible to callers.
        self,
        provider: NarrativeProvider,
        identity: str,
        *,
        limits: RunLimits | None = None,
        state_dir: Path | None = None,
        cache_dir: Path | None = None,
        preparation_seconds: float = 0,
    ) -> None:
        """Capture one approved plan and its already-spent preparation time."""
        self.provider = provider
        self.provider_binding = provider_identity(provider)
        self.limits = limits or RunLimits()
        self.state_dir, self.cache_dir = state_dir, cache_dir
        self.lock = threading.RLock()
        self._slots = threading.BoundedSemaphore(self.limits.concurrency)
        self.cache_hits = 0
        self.new_calls = 0
        self._abort = False
        self._file = None
        self.deadline = (
            time.monotonic() + self.limits.seconds - preparation_seconds
        )
        self.record = {
            "version": RUN_VERSION,
            "identity": identity,
            "limits": asdict(self.limits),
            "state": "running",
            "deadline_at": time.time()
            + self.limits.seconds
            - preparation_seconds,
            "attempts": {},
            "completed": {},
            "annotation_bytes": 0,
            "timings": {"provider_seconds": 0.0, "validation_seconds": 0.0},
            "usage": {
                "input_tokens": 0,
                "cached_input_tokens": 0,
                "output_tokens": 0,
                "calls": 0,
                "unreported_calls": 0,
            },
        }

    def __enter__(self) -> WalkthroughRuntime:  # noqa: PYI034 # Retain Python 3.10 support.
        """Lock and load exact run state without resetting failed/spent attempts."""
        if self.state_dir is None:
            return self
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._file = (self.state_dir / "run.lock").open("a+b")
        try:
            if os.name == "nt":
                import msvcrt

                self._file.write(b"0")
                self._file.flush()
                self._file.seek(0)
                msvcrt.locking(self._file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            path = self.state_dir / "run.json"
            if path.exists():
                saved = _read_record(path, MAX_RECORD_BYTES)
                self._validate_record(saved)
                self.record = saved
                self.deadline = min(
                    self.deadline,
                    time.monotonic() + saved["deadline_at"] - time.time(),
                )
            self.remaining()
            self.record["state"] = "running"
            self._save()
        except BaseException:
            self._file.close()
            self._file = None
            raise
        return self

    def __exit__(
        self, error_type: object, _error: object, _traceback: object
    ) -> None:
        """Record failure, release the actual process lock, and retain all state."""
        try:
            if error_type is not None or self.record["state"] != "complete":
                self.record["state"] = (
                    "failed" if error_type is not None else "incomplete"
                )
                self._save()
        finally:
            if self._file is not None:
                self._file.close()
                self._file = None

    def _validate_record(self, saved: dict) -> None:
        """Reject corrupted or rebound budgets before any provider transfer."""
        if saved.get("version") != RUN_VERSION:
            msg = "Narration run state version is incompatible; use a new state directory"
            raise RunLimitError(msg)
        try:
            _expect_record(set(saved) == set(self.record))
            if (
                saved["identity"] != self.record["identity"]
                or saved["limits"] != self.record["limits"]
            ):
                msg = "Run inputs or limits changed; select a new state directory"
                raise RunLimitError(msg)  # noqa: TRY301 # Keep the distinct input-change diagnostic.
            counts = saved["attempts"]
            _expect_record(
                type(counts) is dict
                and all(
                    type(n) is int and 1 <= n <= ATTEMPTS_PER_REQUEST
                    for n in counts.values()
                )
            )
            _expect_record(
                sum(counts.values()) <= self.limits.requests
                and all(
                    isinstance(k, str)
                    and len(k) == REQUEST_KEY_LENGTH
                    and all(c in "0123456789abcdef" for c in k)
                    for k in counts
                )
            )
            usage = GenerationUsage.model_validate(saved["usage"])
            _expect_record(usage.calls == sum(counts.values()))
            _expect_record(
                type(saved["deadline_at"]) in {int, float}
                and math.isfinite(saved["deadline_at"])
            )
            _expect_record(
                type(saved["annotation_bytes"]) is int
                and 0
                <= saved["annotation_bytes"]
                <= self.limits.annotation_bytes
            )
            completed = saved["completed"]
            _expect_record(
                type(completed) is dict
                and all(
                    isinstance(key, str)
                    and len(key) == REQUEST_KEY_LENGTH
                    and all(c in "0123456789abcdef" for c in key)
                    and isinstance(value, str)
                    and len(value) == REQUEST_KEY_LENGTH
                    and all(c in "0123456789abcdef" for c in value)
                    for key, value in completed.items()
                )
            )
            _expect_record(
                saved["state"]
                in {"running", "failed", "incomplete", "complete"}
            )
            _expect_record(
                type(saved["timings"]) is dict
                and set(saved["timings"])
                == {"provider_seconds", "validation_seconds"}
            )
            _expect_record(
                all(
                    type(value) in {int, float}
                    and math.isfinite(value)
                    and value >= 0
                    for value in saved["timings"].values()
                )
            )
        except RunLimitError:
            raise
        except (ValueError, TypeError, KeyError):
            msg = (
                "Narration run state is corrupt; spent attempts were not reset"
            )
            raise RunLimitError(msg) from None

    def _save(self) -> None:
        """Persist the small source-free run record while its lock is held."""
        if self.state_dir is not None:
            _atomic_json(self.state_dir / "run.json", self.record)

    def remaining(self) -> float:
        """Return remaining total runtime or reject an expired run immediately."""
        value = min(
            self.deadline - time.monotonic(),
            self.record["deadline_at"] - time.time(),
        )
        if value <= 0:
            msg = "Narration exceeded its four-minute run deadline"
            raise RunDeadlineError(msg)
        return value

    def _reserve(self, key: str) -> float:
        """Durably charge a shared attempt before releasing a provider call."""
        with self.lock:
            remaining = self.remaining()
            if self._abort:
                msg = "Narration stopped after a failed request"
                raise RunLimitError(msg)
            if provider_identity(self.provider) != self.provider_binding:
                msg = (
                    "Provider configuration changed after request preparation"
                )
                raise RunLimitError(msg)
            attempts = self.record["attempts"]
            if (
                sum(attempts.values()) >= self.limits.requests
                or attempts.get(key, 0) >= ATTEMPTS_PER_REQUEST
            ):
                self._abort = True
                msg = "Narration request budget exhausted; spent attempts are retained"
                raise RunLimitError(msg)
            attempts[key] = attempts.get(key, 0) + 1
            self.record["usage"]["calls"] += 1
            self.record["usage"]["unreported_calls"] += 1
            self._save()
            return min(remaining, self.remaining())

    def _usage(self, usage: dict | None) -> None:
        """Keep measured tokens from failed as well as successful provider calls."""
        measured = RunUsage()
        measured.record(usage)
        values = measured.report()
        with self.lock:
            for field in (
                "input_tokens",
                "cached_input_tokens",
                "output_tokens",
            ):
                self.record["usage"][field] += values[field]
            self.record["usage"]["unreported_calls"] -= (
                1 - values["unreported_calls"]
            )
            self._save()

    def _cached(
        self, key: str, validate: Callable[[dict], dict]
    ) -> dict | None:
        """Revalidate and account for an exact request-keyed cached result."""
        if (
            self.cache_dir is None
            or not (self.cache_dir / f"{key}.json").exists()
        ):
            return None
        path = self.cache_dir / f"{key}.json"
        try:
            entry = _read_record(path, self.limits.annotation_bytes + 1_024)
            _expect_record(
                set(entry) == {"version", "key", "result", "digest"}
                and entry["version"] == CACHE_VERSION
                and entry["key"] == key
                and digest(entry["result"]) == entry["digest"]
            )
            _expect_record(
                len(canonical_bytes(entry["result"]))
                <= self.limits.annotation_bytes
            )
            result_digest = digest(entry["result"])
            result_bytes = len(canonical_bytes(entry["result"]))
            result = self._validate(entry["result"], validate)
            self.remaining()
        except (ValueError, TypeError, KeyError):
            msg = "Cached narration failed validation; no attempt budget was reset"
            raise RunLimitError(msg) from None
        with self.lock:
            accepted_digest = self.record["completed"].get(key)
            if (
                accepted_digest is not None
                and accepted_digest != result_digest
            ):
                msg = "Cached narration changed after this run accepted its result"
                raise RunLimitError(msg)
            if accepted_digest is None:
                if (
                    self.record["annotation_bytes"] + result_bytes
                    > self.limits.annotation_bytes
                ):
                    msg = "Model annotation data exceeded the shared 1 MiB run limit"
                    raise RunLimitError(msg)
                self.record["annotation_bytes"] += result_bytes
                self.record["completed"][key] = result_digest
                self._save()
            self.cache_hits += 1
        return result

    def call(self, body: bytes, validate: Callable[[dict], dict]) -> dict:
        """Reuse or execute one fixed request with at most one bounded retry.

        Args:
            body: Complete provider request, capacity-checked by the planner.
            validate: Boundary validator returning detached accepted data.

        Returns:
            Validated output. Failed calls, retries, and resumed calls all spend
            the same durable run-wide budget; no split or retry tree is built.

        Raises:
            RunLimitError: If any request, runtime, state, or output limit fails.
        """
        acquired = False
        try:
            acquired = self._slots.acquire(timeout=self.remaining())
            if not acquired:
                msg = "Narration deadline expired waiting for a provider slot"
                raise RunDeadlineError(msg)  # noqa: TRY301 # Keep the slot timeout within the run abort guard.
            return self._call(body, validate)
        except BaseException:
            with self.lock:
                self._abort = True
            raise
        finally:
            if acquired:
                self._slots.release()

    def _call(self, body: bytes, validate: Callable[[dict], dict]) -> dict:
        """Perform the fixed cache/retry sequence inside the run's failure guard."""
        key = request_key(body, self.provider_binding)
        cached = self._cached(key, validate)
        if cached is not None:
            return cached
        for _ in range(ATTEMPTS_PER_REQUEST):
            timeout = self._reserve(key)
            try:
                response, usage = self._send(body, timeout)
            except Exception as error:  # noqa: BLE001 # Provider failures must not expose untrusted payloads.
                self._usage(getattr(error, "usage", None))
                if getattr(error, "category", None) in {
                    "auth",
                    "authentication",
                    "prohibited_tools",
                    "cancelled",
                } or getattr(error, "reason", None) in {
                    "auth",
                    "prohibited_tools",
                    "cancelled",
                }:
                    break
                continue
            self._usage(usage)
            self.remaining()
            raw = canonical_bytes(response)
            with self.lock:
                if (
                    self.record["annotation_bytes"] + len(raw)
                    > self.limits.annotation_bytes
                ):
                    msg = "Model annotation data exceeded the shared 1 MiB run limit"
                    raise RunLimitError(msg)
                self.record["annotation_bytes"] += len(raw)
                self._save()
            try:
                result = self._validate(response, validate)
            except ValueError:
                continue
            self.remaining()
            with self.lock:
                self.record["completed"][key] = digest(response)
                self._save()
                if self.cache_dir is not None:
                    _atomic_json(
                        self.cache_dir / f"{key}.json",
                        {
                            "version": CACHE_VERSION,
                            "key": key,
                            "result": response,
                            "digest": digest(response),
                        },
                    )
                self.remaining()
            return result
        with self.lock:
            self._abort = True
        msg = "Narration request failed after its bounded attempts; results are incomplete"
        raise RunLimitError(msg)

    def _send(self, body: bytes, timeout: float) -> tuple[dict, dict | None]:
        """Measure provider adapter time for success, failure, and interruption."""
        with self.lock:
            self.new_calls += 1
        started = time.monotonic()
        try:
            if self.provider.name in {"openai", "codex"}:
                from .bounded_provider import complete_bounded

                return complete_bounded(
                    self.provider, body, min(timeout, self.remaining())
                )
            return self.provider.complete(body, min(timeout, self.remaining()))
        finally:
            elapsed = time.monotonic() - started
            with self.lock:
                self.record["timings"]["provider_seconds"] += elapsed
                self._save()

    def _validate(self, value: dict, validate: Callable[[dict], dict]) -> dict:
        """Measure schema/source validation separately from provider work."""
        started = time.monotonic()
        try:
            return validate(value)
        finally:
            elapsed = time.monotonic() - started
            with self.lock:
                self.record["timings"]["validation_seconds"] += elapsed
                self._save()

    def report(self) -> dict:
        """Return measured cumulative run usage, including earlier attempts."""
        with self.lock:
            return dict(self.record["usage"])

    def parallel(
        self, requests: list[tuple[bytes, Callable[[dict], dict]]]
    ) -> list[dict]:
        """Run prepared writers with a shared budget and bounded concurrency."""
        if len(requests) > self.limits.requests:
            msg = "Narration plan exceeds the maximum request count before source transfer"
            raise RunLimitError(msg)
        results = [None] * len(requests)
        with ThreadPoolExecutor(
            max_workers=self.limits.concurrency
        ) as executor:
            futures = {
                executor.submit(self.call, body, validate): index
                for index, (body, validate) in enumerate(requests)
            }
            try:
                for future in as_completed(futures, timeout=self.remaining()):
                    results[futures[future]] = future.result()
            except BaseException:
                with self.lock:
                    self._abort = True
                for future in futures:
                    future.cancel()
                raise
        return results

    def finish(self) -> dict:
        """Mark completion only after caller assembly and validation succeeded."""
        with self.lock:
            self.remaining()
            if self._abort:
                msg = "Failed narration cannot be marked complete"
                raise RunLimitError(msg)
            self.record["state"] = "complete"
            self._save()
            return dict(self.record["usage"])
