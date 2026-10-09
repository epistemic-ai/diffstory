"""Bounded, resumable execution for validated narration units."""

from __future__ import annotations

import hashlib
import json
import os
import queue
import re
import tempfile
import threading
import time
from collections.abc import Sequence
from concurrent.futures import FIRST_COMPLETED
from concurrent.futures import Future
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait
from contextlib import contextmanager
from contextlib import suppress
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING
from typing import Literal

from pydantic import ConfigDict
from pydantic import Field
from pydantic import PrivateAttr
from pydantic import ValidationError
from pydantic import field_validator
from pydantic import model_validator

from .models import StrictModel
from .narrative_contracts import ArtifactDependency
from .narrative_contracts import ArtifactRecord
from .narrative_contracts import BatchResult
from .narrative_contracts import BatchSpec
from .narrative_contracts import BatchStage
from .narrative_contracts import Contract
from .narrative_contracts import DetailResult
from .narrative_contracts import Digest
from .narrative_contracts import ItemFailure
from .narrative_contracts import NarrativeResult
from .narrative_contracts import Natural
from .narrative_contracts import Positive
from .narrative_contracts import ResultKind
from .narrative_contracts import (
    StoryPlan,  # noqa: F401 # Preserve the M2 module export.
)
from .narrative_contracts import result_key
from .narrative_contracts import result_key_field
from .narrative_contracts import result_model
from .narrative_contracts import result_stage
from .usage import RunUsage
from .usage import estimate_input

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterator
    from collections.abc import Mapping

    from .narrative_plan import PreparedRequest
    from .narrative_plan import RequestPackingLimits
    from .usage import ModelCapacity

try:
    import fcntl
except (
    ImportError
):  # Legacy v1 remains usable on platforms without POSIX locks.
    fcntl = None

RUNTIME_CONTRACT_VERSION = "1"
CHECKPOINT_FORMAT_VERSION = 1
DEFAULT_CONCURRENCY = 4
_ASCII_CONTROL_CHARACTER_MAX = 31
_ASCII_DELETE_CHARACTER = 127
_MAX_REQUEST_DIGEST_CHARS = 256
_PAIR_LENGTH = 2
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_HEX_DIGEST = re.compile(r"^[a-f0-9]{64}$")
_FAILURE_ACTIONS = {
    "authentication": "Check provider credentials, then rerun the stage.",
    "rate_limit": "Wait before you rerun the stage.",
    "transport": "Check provider connectivity, then rerun the stage.",
    "timeout": "Check provider responsiveness, then rerun the stage.",
    "invalid_response": "Check the provider response against its schema.",
    "provider_error": "Check the provider setup and request, then rerun.",
    "validation_error": "Correct the response or validator, then rerun.",
    "checkpoint_error": "Check checkpoint storage and permissions, then rerun.",
    "callback_error": "Correct the progress callback, then rerun.",
    "task_error": "Correct the failed task, then rerun.",
    "cancelled": "Rerun the stage after stopping its remaining work.",
}


def _canonical_digest(value: object) -> str:
    """Hash JSON-compatible data with a stable key order."""
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _valid_identifier(value: str) -> str:
    """Accept a short identifier that is safe to show in progress and errors."""
    if not _SAFE_ID.fullmatch(value):
        msg = "Identifier must use only letters, digits, dot, colon, or dash"
        raise ValueError(msg)
    return value


class RuntimeRecord(StrictModel):
    """Use immutable, strict Pydantic records at runtime boundaries."""

    model_config = ConfigDict(
        strict=True,
        extra="forbid",
        hide_input_in_errors=True,
        frozen=True,
    )


class RunIdentity(RuntimeRecord):
    """Name the exact report and provider configuration for a run.

    Attributes:
        base_revision: Pinned base revision.
        head_revision: Pinned head revision.
        report_signature: Digest of the full deterministic report input.
        provider_destination: Provider endpoint or execution destination.
        model: Exact provider model name.
        runtime_contract_version: Version for cache-invalidating runtime rules.
    """

    base_revision: str = Field(min_length=1, max_length=256)
    head_revision: str = Field(min_length=1, max_length=256)
    report_signature: str = Field(min_length=1, max_length=256)
    provider_destination: str = Field(min_length=1, max_length=512)
    model: str = Field(min_length=1, max_length=256)
    runtime_contract_version: str = RUNTIME_CONTRACT_VERSION

    @field_validator(
        "base_revision",
        "head_revision",
        "report_signature",
        "provider_destination",
        "model",
        "runtime_contract_version",
    )
    @classmethod
    def no_control_characters(cls, value: str) -> str:
        """Reject controls that can corrupt identity serialization or output."""
        if any(
            ord(character) <= _ASCII_CONTROL_CHARACTER_MAX
            or ord(character) == _ASCII_DELETE_CHARACTER
            for character in value
        ):
            msg = "Run identity fields cannot contain control characters"
            raise ValueError(msg)
        return value

    @property
    def digest(self) -> str:
        """Return a digest without writing identity values into cache paths."""
        return _canonical_digest(self.model_dump(mode="json"))


class CallSpec(RuntimeRecord):
    """Describe one semantic result and the inputs that make it reusable.

    Attributes:
        task_id: Short safe label for progress and failure messages.
        schema_name: Structured response schema name.
        request_digest: Digest of the prepared request, schema, and reservation.
        validation_version: Version of the semantic validator for this result.
    """

    task_id: str
    schema_name: str
    request_digest: str = Field(min_length=1, max_length=256)
    validation_version: str = Field(min_length=1, max_length=128)

    @field_validator("task_id", "schema_name", "validation_version")
    @classmethod
    def safe_names(cls, value: str) -> str:
        """Require identifiers that are safe to report to a user."""
        return _valid_identifier(value)

    @field_validator("request_digest")
    @classmethod
    def safe_request_digest(cls, value: str) -> str:
        """Reject controls and whitespace from request digest labels."""
        if (
            not value
            or len(value) > _MAX_REQUEST_DIGEST_CHARS
            or any(character.isspace() for character in value)
        ):
            msg = "Request digest must be a nonempty, bounded token"
            raise ValueError(msg)
        return value

    @property
    def cache_key(self) -> str:
        """Hash every field that can change the semantic result."""
        return _canonical_digest(self.model_dump(mode="json"))

    @classmethod
    # Request identity needs every member of the prepared-call contract.
    def from_request(  # noqa: PLR0913
        cls,
        *,
        task_id: str,
        schema_name: str,
        request: bytes,
        schema: dict[str, object],
        output_reservation: int,
        validation_version: str,
    ) -> CallSpec:
        """Build a request digest that includes schema and output reservation.

        Args:
            task_id: Short safe label for this semantic unit.
            schema_name: Structured response schema name.
            request: Exact prepared provider request bytes.
            schema: JSON schema sent with the provider request.
            output_reservation: Reserved output-token count.
            validation_version: Semantic validator version.

        Returns:
            A call specification whose digest changes with any request input.

        Raises:
            TypeError: If request or schema has the wrong Python type.
            ValueError: If the reservation is not a positive integer.
        """
        if type(request) is not bytes or type(schema) is not dict:
            msg = "Request bytes and a JSON schema are required"
            raise TypeError(msg)
        if type(output_reservation) is not int or output_reservation <= 0:
            msg = "Output reservation must be a positive integer"
            raise ValueError(msg)
        digest = _canonical_digest(
            {
                "request_sha256": hashlib.sha256(request).hexdigest(),
                "schema": schema,
                "output_reservation": output_reservation,
            },
        )
        return cls(
            task_id=task_id,
            schema_name=schema_name,
            request_digest=digest,
            validation_version=validation_version,
        )


class CheckpointRecord(RuntimeRecord):
    """Store one validated result without provider-call history.

    Attributes:
        format_version: Version of the on-disk checkpoint record.
        identity_digest: Digest of the complete run identity.
        task_id: Safe semantic-unit label.
        schema_name: Structured response schema name.
        request_digest: Digest of the exact prepared request contract.
        validation_version: Version of the semantic validator.
        result: Validated JSON-compatible dictionary.
    """

    format_version: Literal[CHECKPOINT_FORMAT_VERSION] = (
        CHECKPOINT_FORMAT_VERSION
    )
    identity_digest: str
    task_id: str
    schema_name: str
    request_digest: str
    validation_version: str
    result: dict[str, object]

    @field_validator("identity_digest")
    @classmethod
    def identity_digest_is_sha256(cls, value: str) -> str:
        """Require the full identity digest before using it as a directory."""
        if not _HEX_DIGEST.fullmatch(value):
            msg = "Checkpoint identity digest is invalid"
            raise ValueError(msg)
        return value

    @field_validator("task_id", "schema_name", "validation_version")
    @classmethod
    def checkpoint_names_are_safe(cls, value: str) -> str:
        """Validate safe labels before progress or cache use."""
        return _valid_identifier(value)

    @property
    def cache_key(self) -> str:
        """Return the filename key for this semantic result."""
        return _canonical_digest(
            {
                "task_id": self.task_id,
                "schema_name": self.schema_name,
                "request_digest": self.request_digest,
                "validation_version": self.validation_version,
            },
        )


class ProgressEvent(RuntimeRecord):
    """Report safe run progress without provider bodies or exception text.

    Attributes:
        event: Stage, call, reuse, or heartbeat event.
        stage: Current stage name.
        task_id: Safe task label when the event belongs to one task.
        active_calls: Provider calls currently running.
        completed: Stage jobs that have finished.
        planned: Total jobs in the stage.
        reused_checkpoints: Validated results reused during this process.
        provider_calls: Provider attempts made during this process.
        elapsed_seconds: Monotonic time since runtime creation.
        outcome: Call outcome when this is a call-completion event.
    """

    event: Literal[
        "stage_started",
        "stage_completed",
        "call_started",
        "call_completed",
        "checkpoint_reused",
        "heartbeat",
    ]
    stage: str
    task_id: str | None = None
    active_calls: int = Field(ge=0)
    completed: int = Field(ge=0)
    planned: int = Field(ge=0)
    reused_checkpoints: int = Field(ge=0)
    provider_calls: int = Field(ge=0)
    elapsed_seconds: float = Field(ge=0)
    outcome: Literal["succeeded", "failed"] | None = None
    task_digest: Digest | None = None
    stage_position: int | None = Field(default=None, ge=0)
    diagnostic: SafeDiagnostic | None = None

    @field_validator("stage")
    @classmethod
    def safe_stage(cls, value: str) -> str:
        """Reject values that could leak long or multiline request data."""
        return _valid_identifier(value)

    @field_validator("task_id")
    @classmethod
    def safe_task(cls, value: str | None) -> str | None:
        """Validate optional task labels before they reach a progress sink."""
        return None if value is None else _valid_identifier(value)


class RuntimeStageError(RuntimeError):
    """Sanitized stage error with safe identifiers and a resume action."""

    def __init__(
        self,
        stage: str,
        task_id: str,
        category: str,
        *,
        diagnostic: SafeDiagnostic | None = None,
        stage_position: int | None = None,
    ) -> None:
        """Create an error without provider or validation input text."""
        if not isinstance(stage, str) or not _SAFE_ID.fullmatch(stage):
            stage = "stage"
        if not isinstance(task_id, str) or not _SAFE_ID.fullmatch(task_id):
            task_id = "task"
        if not isinstance(category, str) or category not in _FAILURE_ACTIONS:
            category = "task_error"
        self.stage = stage
        self.task_id = task_id
        self.category = category
        self.diagnostic = diagnostic
        self.task_digest = task_id if _HEX_DIGEST.fullmatch(task_id) else None
        self.stage_position = stage_position
        self.action = _FAILURE_ACTIONS[category]
        super().__init__(
            f"Stage {stage!r} failed for task {task_id!r} ({category}). "
            f"{self.action}",
        )


class CheckpointError(RuntimeError):
    """Report a safe checkpoint write failure without exposing file paths."""


class RuntimeValidationError(ValueError):
    """Report a failed semantic validator without exposing its input."""


class _StageCancelledError(RuntimeError):
    """Stop a running task before it starts a new provider attempt."""


class RuntimeCallError(RuntimeError):
    """Provider failure with an allowlisted category and optional usage."""

    def __init__(
        self,
        schema_name: str,
        category: str,
        usage: dict[str, int] | None = None,
    ) -> None:
        """Create a categorized error without provider response text."""
        if not isinstance(category, str) or category not in _FAILURE_ACTIONS:
            category = "provider_error"
        if not isinstance(schema_name, str) or not _SAFE_ID.fullmatch(
            schema_name
        ):
            schema_name = "provider"
        self.schema_name = schema_name
        self.category = category
        self.usage = usage if type(usage) is dict else None
        super().__init__(
            f"Provider call for {schema_name!r} failed ({category})"
        )


class CheckpointStore:
    """Read and atomically replace validated JSON checkpoints."""

    def __init__(self, directory: str | Path) -> None:
        """Set the private checkpoint root.

        Args:
            directory: Parent directory for hash-namespaced run checkpoints.
        """
        self._directory = Path(directory)
        self._lock = threading.RLock()

    def load(
        self,
        identity: RunIdentity,
        spec: CallSpec,
    ) -> CheckpointRecord | None:
        """Load one valid record for an exact run and semantic identity.

        Invalid, incomplete, obsolete, or mismatched files are cache misses.
        Temporary files are ignored because only ``*.json`` records are read.

        Args:
            identity: Complete run identity for the namespace.
            spec: Required semantic-unit specification to load.

        Returns:
            One matching record or a cache miss.
        """
        if not isinstance(identity, RunIdentity) or not isinstance(
            spec, CallSpec
        ):
            msg = "Checkpoint load requires RunIdentity and CallSpec records"
            raise TypeError(msg)
        namespace = self._directory / identity.digest
        path = namespace / f"{spec.cache_key}.json"
        record = self._read(path, identity)
        if record is None or not self._matches(record, identity, spec):
            return None
        return record

    def commit(self, record: CheckpointRecord) -> None:
        """Write one complete record with a same-directory atomic replace.

        Args:
            record: A validated result with no historical call metadata.

        Raises:
            CheckpointError: If the atomic write cannot complete.
        """
        if not isinstance(record, CheckpointRecord):
            msg = "Checkpoint commit requires a CheckpointRecord"
            raise TypeError(msg)
        directory = self._directory / record.identity_digest
        target = directory / f"{record.cache_key}.json"
        temporary_path: Path | None = None
        try:
            directory.mkdir(mode=0o700, parents=True, exist_ok=True)
            encoded = record.model_dump_json().encode("utf-8")
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=".checkpoint-",
                suffix=".tmp",
                dir=directory,
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                temporary_path.chmod(0o600)
                temporary.write(encoded)
                temporary.flush()
                os.fsync(temporary.fileno())
            with self._lock:
                temporary_path.replace(target)
                temporary_path = None
        except (OSError, ValueError, TypeError):
            msg = "Checkpoint write failed; check the checkpoint directory and rerun"
            raise CheckpointError(msg) from None
        finally:
            if temporary_path is not None:
                with suppress(OSError):
                    temporary_path.unlink()

    @staticmethod
    def _matches(
        record: CheckpointRecord,
        identity: RunIdentity,
        spec: CallSpec,
    ) -> bool:
        """Require every namespace and semantic key field to match."""
        return (
            record.identity_digest == identity.digest
            and record.task_id == spec.task_id
            and record.schema_name == spec.schema_name
            and record.request_digest == spec.request_digest
            and record.validation_version == spec.validation_version
        )

    @staticmethod
    def _read(path: Path, identity: RunIdentity) -> CheckpointRecord | None:
        """Read one valid JSON record and treat all file errors as cache misses."""
        try:
            serialized = path.read_bytes()
            value = json.loads(serialized)
        except (OSError, ValidationError, ValueError, TypeError):
            return None
        if type(value) is not dict:
            return None
        version = value.get("format_version")
        if type(version) is not int or version != CHECKPOINT_FORMAT_VERSION:
            return None
        try:
            record = CheckpointRecord.model_validate_json(serialized)
        except (ValidationError, ValueError, TypeError):
            return None
        if record.identity_digest != identity.digest:
            return None
        return record


class GenerationRuntime:
    """Run independent generation jobs with safe progress and checkpoints.

    Args:
        identity: Full report, revision, provider, and model identity.
        concurrency: Maximum number of active jobs. Defaults to four.
        checkpoint_dir: Optional persistent checkpoint root.
        on_progress: Optional callback invoked by the coordinating thread.
        usage: Thread-safe provider usage accumulator. A new one is created by
            default.
    """

    def __init__(
        self,
        identity: RunIdentity,
        concurrency: int = DEFAULT_CONCURRENCY,
        checkpoint_dir: str | Path | None = None,
        on_progress: Callable[[ProgressEvent], None] | None = None,
        usage: RunUsage | None = None,
    ) -> None:
        """Create a runtime and its optional checkpoint store.

        Raises:
            ValueError: If concurrency is not positive or an unresolved model
                is used with persistent checkpoints.
            TypeError: If the progress callback is not callable.
        """
        if not isinstance(identity, RunIdentity):
            msg = "GenerationRuntime requires a RunIdentity record"
            raise TypeError(msg)
        if type(concurrency) is not int or concurrency < 1:
            msg = "Concurrency must be a positive integer"
            raise ValueError(msg)
        if on_progress is not None and not callable(on_progress):
            msg = "Progress callback must be callable"
            raise TypeError(msg)
        if (
            checkpoint_dir is not None
            and identity.model.strip().casefold()
            in {
                "default",
                "codex cli default",
            }
        ):
            msg = "Persistent checkpoints require an explicit provider model"
            raise ValueError(msg)

        self.identity = identity
        self.concurrency = concurrency
        self.usage = usage if usage is not None else RunUsage()
        self.on_progress = on_progress
        self._store = (
            CheckpointStore(checkpoint_dir)
            if checkpoint_dir is not None
            else None
        )
        self._lock = threading.RLock()
        self._stage_guard = threading.Lock()
        self._events: queue.SimpleQueue[ProgressEvent] = queue.SimpleQueue()
        self._local = threading.local()
        self._started_at = time.monotonic()
        self._active_calls = 0
        self._reused_checkpoints = 0
        self._stage_coordinator: int | None = None
        self._stage_cancel: threading.Event | None = None
        self._stage_name = "runtime"
        self._stage_completed = 0
        self._stage_planned = 0
        self._first_failure: tuple[str, str] | None = None
        self._first_diagnostic: SafeDiagnostic | None = None
        self._first_position: int | None = None
        self._callback_failed = False

    @property
    def reused_checkpoints(self) -> int:
        """Return the current-process count of semantically revalidated hits."""
        with self._lock:
            return self._reused_checkpoints

    def run_stage(
        self,
        stage: str,
        jobs: Sequence[Callable[[], object]],
    ) -> list[object]:
        """Run a bounded stage and return results in input order.

        The runtime submits no more than ``concurrency`` callables at a time.
        On failure, it stops new submissions, cancels work that has not
        started, drains active jobs, then raises a safe stage error.

        Args:
            stage: Short stage name for progress and safe errors.
            jobs: Callables in the required result order.

        Returns:
            Results in the order of ``jobs``.

        Raises:
            RuntimeStageError: If a job fails or progress callback fails.
            TypeError: If jobs is not a callable sequence.
        """
        stage = _valid_identifier(stage)
        if not isinstance(jobs, Sequence) or any(
            not callable(job) for job in jobs
        ):
            msg = "Stage jobs must be a sequence of callables"
            raise TypeError(msg)
        parsed = list(jobs)
        if not self._stage_guard.acquire(blocking=False):
            msg = "A generation stage is already active"
            raise RuntimeError(msg)

        self._begin_stage(stage, len(parsed))
        executor = ThreadPoolExecutor(max_workers=self.concurrency)
        try:
            self._emit("stage_started")
            results = self._execute_stage(executor, parsed)
            self._raise_stage_failure(stage)
            self._emit("stage_completed")
            executor.shutdown(wait=True, cancel_futures=True)
            self._drain_events()
            self._raise_callback_failure(stage)
        except BaseException:
            self._cancel_stage()
            self._shutdown_after_interrupt(executor)
            self._discard_events()
            raise
        else:
            return results
        finally:
            try:
                self._finish_stage()
            finally:
                self._stage_guard.release()

    def _cancel_stage(self) -> None:
        """Stop every active task before the coordinator drains the executor."""
        with self._lock:
            if self._stage_cancel is not None:
                self._stage_cancel.set()

    @staticmethod
    def _shutdown_after_interrupt(executor: ThreadPoolExecutor) -> None:
        """Stop submissions without joining workers on the signal path."""
        while True:
            try:
                executor.shutdown(wait=False, cancel_futures=True)
            except KeyboardInterrupt:
                continue
            except BaseException:  # noqa: BLE001
                return
            return

    def _discard_events(self) -> None:
        """Drop queued events after a coordinator exception without callbacks."""
        while not self._events.empty():
            self._events.get_nowait()

    def _begin_stage(self, stage: str, planned: int) -> None:
        """Set coordinator state before any stage event or job starts."""
        self._local.diagnostic = None
        with self._lock:
            self._stage_coordinator = threading.get_ident()
            self._stage_cancel = threading.Event()
            self._stage_name = stage
            self._stage_completed = 0
            self._stage_planned = planned
            self._first_failure = None
            self._first_diagnostic = None
            self._first_position = None
            self._callback_failed = False

    def _finish_stage(self) -> None:
        """Clear transient stage state after every active job has drained."""
        with self._lock:
            self._stage_coordinator = None
            self._stage_cancel = None
            self._stage_name = "runtime"
            self._stage_planned = 0
            self._stage_completed = 0
            self._first_failure = None
            self._first_diagnostic = None
            self._first_position = None
            self._callback_failed = False

    def _execute_stage(
        self,
        executor: ThreadPoolExecutor,
        jobs: list[Callable[[], object]],
    ) -> list[object]:
        """Submit bounded work, drain completions, and retain ordered results."""
        results: list[object] = [None] * len(jobs)
        pending: dict[Future[object], tuple[int, str]] = {}
        try:
            next_position = self._fill_stage_slots(executor, pending, jobs, 0)
            while pending:
                self._drain_events()
                done, _ = wait(
                    tuple(pending),
                    timeout=0.25,
                    return_when=FIRST_COMPLETED,
                )
                if not done:
                    self._emit("heartbeat")
                    continue
                self._collect_stage_results(done, pending, results)
                if self._stop_requested():
                    self._cancel_queued(pending)
                else:
                    next_position = self._fill_stage_slots(
                        executor,
                        pending,
                        jobs,
                        next_position,
                    )
            self._drain_events()
        except BaseException:
            self._cancel_stage()
            self._drain_pending_futures(pending)
            raise
        return results

    def _drain_pending_futures(
        self,
        pending: dict[Future[object], tuple[int, str]],
    ) -> None:
        """Cancel queued tasks and wait for active futures without thread joins."""
        while True:
            try:
                for future in tuple(pending):
                    if future.cancel() or future.done():
                        pending.pop(future)
                        with self._lock:
                            self._stage_completed += 1
                if not pending:
                    return
                wait(
                    tuple(pending),
                    timeout=0.05,
                    return_when=FIRST_COMPLETED,
                )
            except KeyboardInterrupt:  # noqa: PERF203
                continue

    def _fill_stage_slots(
        self,
        executor: ThreadPoolExecutor,
        pending: dict[Future[object], tuple[int, str]],
        jobs: list[Callable[[], object]],
        next_position: int,
    ) -> int:
        """Submit work only while capacity remains and the stage can continue."""
        while (
            next_position < len(jobs)
            and len(pending) < self.concurrency
            and not self._stop_requested()
        ):
            self._submit(executor, pending, jobs, next_position)
            next_position += 1
        return next_position

    def _collect_stage_results(
        self,
        done: set[Future[object]],
        pending: dict[Future[object], tuple[int, str]],
        results: list[object],
    ) -> None:
        """Collect completed futures after workers record any safe failure."""
        for future in sorted(done, key=lambda item: pending[item][0]):
            position, _task_id = pending.pop(future)
            with suppress(BaseException):
                results[position] = future.result()
            with self._lock:
                self._stage_completed += 1
            self._drain_events()

    def _cancel_queued(
        self,
        pending: dict[Future[object], tuple[int, str]],
    ) -> None:
        """Cancel futures that have not started after a stage failure."""
        for future in tuple(pending):
            if future.cancel():
                pending.pop(future)
                with self._lock:
                    self._stage_completed += 1

    def _raise_stage_failure(self, stage: str) -> None:
        """Raise only safe stage, task, and failure-category details."""
        with self._lock:
            failure = self._first_failure
            callback_failed = self._callback_failed
            stopped = self._stop_requested_locked()
        if callback_failed:
            raise RuntimeStageError(stage, "progress", "callback_error")
        if failure is not None:
            task_id, category = failure
            raise RuntimeStageError(
                stage,
                task_id,
                category,
                diagnostic=self._first_diagnostic,
                stage_position=self._first_position,
            )
        if stopped:
            raise RuntimeStageError(stage, "stage", "cancelled")

    def _raise_callback_failure(self, stage: str) -> None:
        """Raise a safe error when the progress callback rejects an event."""
        with self._lock:
            failed = self._callback_failed
        if failed:
            raise RuntimeStageError(stage, "progress", "callback_error")

    def invoke(
        self,
        spec: CallSpec,
        produce: Callable[[], dict[str, object]],
        validate: Callable[[dict[str, object]], dict[str, object]],
    ) -> dict[str, object]:
        """Produce, validate, and optionally checkpoint one semantic result.

        A cache read runs the current validator again. Invalid cached data is
        treated as a miss. A fresh result is committed only after validation.
        The producer should use ``call`` for each provider attempt.

        Args:
            spec: Stable schema, request, task, and validator identity.
            produce: Function that creates a response dictionary.
            validate: Function that returns its validated dictionary.

        Returns:
            The validated semantic result.

        Raises:
            TypeError: If producer or validator is not callable.
        """
        if not isinstance(spec, CallSpec):
            msg = "GenerationRuntime.invoke requires a CallSpec record"
            raise TypeError(msg)
        if not callable(produce) or not callable(validate):
            msg = "Producer and validator must be callable"
            raise TypeError(msg)
        self._check_cancelled()

        if self._store is not None:
            cached = self._store.load(self.identity, spec)
            if isinstance(cached, CheckpointRecord):
                try:
                    result = self._validated(cached.result, validate)
                except RuntimeValidationError:
                    pass
                else:
                    with self._lock:
                        self._reused_checkpoints += 1
                    self._emit("checkpoint_reused", task_id=spec.task_id)
                    return result

        previous_task = getattr(self._local, "task_id", None)
        self._local.task_id = spec.task_id
        try:
            candidate = produce()
        finally:
            self._local.task_id = previous_task
        result = self._validated(candidate, validate)
        if self._store is not None:
            record = CheckpointRecord(
                identity_digest=self.identity.digest,
                task_id=spec.task_id,
                schema_name=spec.schema_name,
                request_digest=spec.request_digest,
                validation_version=spec.validation_version,
                result=self._json_value(result),
            )
            self._store.commit(record)
        return result

    def call(
        self,
        schema_name: str,
        send: Callable[[], tuple[dict[str, object], dict[str, int] | None]],
    ) -> dict[str, object]:
        """Attempt one provider call and account for success or failure.

        ``send`` must return a response dictionary and optional usage mapping.
        A provider exception may expose a safe ``usage`` mapping. This method
        counts the attempt before semantic validation, including failed and
        malformed responses. A checkpoint reuse does not call this method.

        Args:
            schema_name: Short response schema name.
            send: Synchronous provider operation.

        Returns:
            The provider response dictionary.

        Raises:
            RuntimeCallError: If a provider fails or returns an invalid result.
            _StageCancelledError: If a stage failed before this attempt starts.
        """
        schema_name = _valid_identifier(schema_name)
        if not callable(send):
            msg = "Provider send operation must be callable"
            raise TypeError(msg)
        task_id = getattr(self._local, "task_id", schema_name)
        with self._lock:
            cancellation = getattr(self._local, "cancellation", None)
            if cancellation is None:
                cancellation = self._stage_cancel
            if cancellation is not None and cancellation.is_set():
                raise _StageCancelledError
            self._active_calls += 1
        self._emit("call_started", task_id=task_id)

        status: Literal["succeeded", "failed"] = "failed"
        try:
            try:
                sent = send()
            except BaseException as error:
                raw_usage = getattr(error, "usage", None)
                self.usage.record(
                    raw_usage if type(raw_usage) is dict else None
                )
                if isinstance(error, Exception):
                    category = self._provider_failure_category(error)
                    usage = raw_usage if type(raw_usage) is dict else None
                    raise RuntimeCallError(
                        schema_name, category, usage
                    ) from None
                raise

            if type(sent) is not tuple or len(sent) != _PAIR_LENGTH:
                self.usage.record(None)
                raise RuntimeCallError(schema_name, "invalid_response")
            response, raw_usage = sent
            self.usage.record(raw_usage if type(raw_usage) is dict else None)
            if type(response) is not dict or (
                raw_usage is not None and type(raw_usage) is not dict
            ):
                raise RuntimeCallError(schema_name, "invalid_response")
            status = "succeeded"
            return response
        finally:
            with self._lock:
                self._active_calls -= 1
            self._emit("call_completed", task_id=task_id, outcome=status)

    @staticmethod
    def _provider_failure_category(error: BaseException) -> str:
        """Map provider failures to fixed categories without using error text."""
        if isinstance(error, RuntimeCallError):
            return error.category
        if isinstance(error, TimeoutError):
            return "timeout"
        if isinstance(error, (ConnectionError, OSError)):
            return "transport"
        if isinstance(error, (ValidationError, ValueError)):
            return "invalid_response"
        return "provider_error"

    def _submit(
        self,
        executor: ThreadPoolExecutor,
        pending: dict[Future[object], tuple[int, str]],
        jobs: list[Callable[[], object]],
        position: int,
    ) -> None:
        """Submit one task with its failure metadata attached."""
        job = jobs[position]
        task_id = (
            job.task_digest
            if isinstance(job, SemanticJob)
            else f"task-{position}"
        )
        # A worker takes the same lock before running. Register its future before
        # it can send, including when SIGINT arrives during executor submission.
        with self._lock:
            try:
                future = executor.submit(self._run_job, task_id, job, position)
                pending[future] = (position, task_id)
            except BaseException:
                self._cancel_stage()
                raise

    def _run_job(
        self, task_id: str, job: Callable[[], object], position: int
    ) -> object:
        """Run one task and report its first failure without its message."""
        with self._lock:
            cancellation = self._stage_cancel
        self._local.cancellation = cancellation
        self._local.stage_position = position
        self._local.task_digest = (
            task_id if _HEX_DIGEST.fullmatch(task_id) else None
        )
        self._local.diagnostic = None
        try:
            self._check_cancelled()
            return job()
        except _StageCancelledError:
            with self._lock:
                if self._first_failure is None and not self._callback_failed:
                    self._first_failure = (task_id, "cancelled")
            raise
        except BaseException as error:
            self._record_stage_failure(
                task_id,
                self._stage_error_category(error),
            )
            raise
        finally:
            self._local.cancellation = None
            self._local.stage_position = None
            self._local.task_digest = None
            self._local.diagnostic = None

    @staticmethod
    def _stage_error_category(error: BaseException) -> str:
        """Return one allowlisted category without reading exception text."""
        if isinstance(error, RuntimeCallError):
            return error.category
        if isinstance(error, ArtifactRuntimeError):
            return {
                "auth": "authentication",
                "storage": "checkpoint_error",
                "schema": "validation_error",
                "citation": "validation_error",
                "coverage": "validation_error",
            }.get(error.reason, "task_error")
        category_rules: tuple[tuple[type[BaseException], str], ...] = (
            (CheckpointError, "checkpoint_error"),
            (RuntimeValidationError, "validation_error"),
            (TimeoutError, "timeout"),
            (ConnectionError, "transport"),
            (OSError, "transport"),
            (ValidationError, "validation_error"),
            (ValueError, "validation_error"),
        )
        for error_type, category in category_rules:
            if isinstance(error, error_type):
                return category
        return "task_error"

    def _record_stage_failure(self, task_id: str, category: str) -> None:
        """Stop new submissions after the first task failure."""
        with self._lock:
            if self._first_failure is None:
                self._first_diagnostic = getattr(
                    self._local, "diagnostic", None
                )
                if self._first_diagnostic is not None:
                    task_id = self._first_diagnostic.task_digest
                self._first_failure = (task_id, category)
                self._first_position = getattr(
                    self._local, "stage_position", None
                )
                if self._stage_cancel is not None:
                    self._stage_cancel.set()

    def _validated(
        self,
        candidate: dict[str, object],
        validate: Callable[[dict[str, object]], dict[str, object]],
    ) -> dict[str, object]:
        """Run a semantic validator without exposing invalid response text."""
        if type(candidate) is not dict:
            raise RuntimeValidationError
        try:
            normalized = validate(candidate)
        except Exception:  # noqa: BLE001
            raise RuntimeValidationError from None
        if type(normalized) is not dict:
            raise RuntimeValidationError
        return normalized

    @staticmethod
    def _json_value(value: dict[str, object]) -> dict[str, object]:
        """Make an isolated JSON dictionary copy for persistent storage."""
        if type(value) is not dict:
            raise CheckpointError
        try:
            encoded = json.dumps(
                value,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            )
            result = json.loads(encoded)
        except (TypeError, ValueError):
            raise CheckpointError from None
        if type(result) is not dict:
            raise CheckpointError
        return result

    def _check_cancelled(self) -> None:
        """Prevent a stage task from starting another provider attempt."""
        cancellation = getattr(self._local, "cancellation", None)
        if cancellation is None:
            with self._lock:
                cancellation = self._stage_cancel
        if cancellation is not None and cancellation.is_set():
            raise _StageCancelledError

    def _stop_requested(self) -> bool:
        """Return whether a task or progress callback has stopped the stage."""
        with self._lock:
            return self._stop_requested_locked()

    def _stop_requested_locked(self) -> bool:
        """Read stage stop state while holding the runtime lock."""
        return self._callback_failed or (
            self._stage_cancel is not None and self._stage_cancel.is_set()
        )

    def _emit(
        self,
        event_name: Literal[
            "stage_started",
            "stage_completed",
            "call_started",
            "call_completed",
            "checkpoint_reused",
            "heartbeat",
        ],
        *,
        task_id: str | None = None,
        outcome: Literal["succeeded", "failed"] | None = None,
    ) -> None:
        """Deliver or queue a safe progress event for the coordinator."""
        if self.on_progress is None:
            return
        with self._lock:
            event = ProgressEvent(
                event=event_name,
                stage=self._stage_name,
                task_id=task_id,
                active_calls=self._active_calls,
                completed=self._stage_completed,
                planned=self._stage_planned,
                reused_checkpoints=self._reused_checkpoints,
                provider_calls=self.usage.report()["calls"],
                elapsed_seconds=max(0.0, time.monotonic() - self._started_at),
                outcome=outcome,
                task_digest=getattr(self._local, "task_digest", None),
                stage_position=getattr(self._local, "stage_position", None),
                diagnostic=getattr(self._local, "diagnostic", None),
            )
            coordinator = self._stage_coordinator
        if coordinator is None or coordinator == threading.get_ident():
            self._deliver(event)
        else:
            self._events.put(event)

    def _drain_events(self) -> None:
        """Deliver queued worker events from the stage's coordinating thread."""
        while True:
            try:
                event = self._events.get_nowait()
            except queue.Empty:
                return
            self._deliver(event)

    def _deliver(self, event: ProgressEvent) -> None:
        """Call the optional sink and stop new work if it raises."""
        if self.on_progress is None:
            return
        try:
            self.on_progress(event)
        except Exception:  # noqa: BLE001
            with self._lock:
                self._callback_failed = True
                if self._stage_cancel is not None:
                    self._stage_cancel.set()


# V2 is additive. These records never reinterpret CHECKPOINT_FORMAT_VERSION.
ARTIFACT_FORMAT_VERSION = 2
_STATIC_KINDS = frozenset(
    {"provider", "model", "configuration", "prompt", "schema", "validator"}
)
_CONTEXT_KINDS = frozenset({"context", "order", "terminology"})
_TRANSIENT_REASONS = frozenset({"timeout", "transport", "rate_limit"})
_SIZE_REASONS = frozenset(
    {"input_limit", "output_limit", "response_limit", "timeout_limit"}
)
_STOP_REASONS = frozenset({"auth", "prohibited_tools"})
Reason = Literal[
    "generic",
    "malformed_json",
    "malformed_jsonl",
    "malformed_envelope",
    "missing_final",
    "prohibited_tools",
    "response_limit",
    "input_limit",
    "output_limit",
    "timeout_limit",
    "schema",
    "citation",
    "coverage",
    "timeout",
    "auth",
    "transport",
    "rate_limit",
    "missing_item",
    "duplicate_item",
    "unknown_item",
    "cancelled",
    "storage",
    "state_corrupt",
    "policy_conflict",
    "pending_lineage",
    "attempt_limit",
    "split_limit",
    "request_mismatch",
    "dependencies_changed",
    "results_unavailable",
    "busy",
]
_REASONS = frozenset(Reason.__args__)
Category = Literal[
    "provider", "validation", "capacity", "recovery", "storage", "cancelled"
]


def _require_v2(condition: bool, reason: str = "schema") -> None:  # noqa: FBT001
    if not condition:
        raise ArtifactRuntimeError(reason) from None


def _reason(value: object) -> Reason:
    return value if type(value) is str and value in _REASONS else "generic"


class ArtifactRuntimeError(RuntimeError):
    """Expose fixed codes only; callers must not pass exception text through."""

    def __init__(self, reason: str = "generic") -> None:
        """Map unknown reasons to one generic code."""
        self.reason = _reason(reason)
        self.diagnostic: SafeDiagnostic | None = None
        super().__init__(f"Narration runtime failed ({self.reason})")


class ArtifactCallError(ArtifactRuntimeError):
    """Adapter failure with safe reason, reported usage and response length."""

    def __init__(
        self,
        reason: str,
        *,
        usage: dict[str, int] | None = None,
        response_bytes: int | None = None,
    ) -> None:
        """Retain numeric usage only; never retain a provider error body."""
        super().__init__(reason)
        self.usage = _safe_usage(usage)
        self.response_bytes = (
            response_bytes
            if type(response_bytes) is int and response_bytes >= 0
            else None
        )


def _safe_usage(value: object) -> dict[str, int] | None:
    if type(value) is not dict:
        return None
    names = ("input_tokens", "output_tokens", "cached_input_tokens")
    if not all(type(value.get(name)) is int for name in names[:2]):
        return None
    result = {name: value[name] for name in names if name in value}
    if any(type(v) is not int or v < 0 for v in result.values()):
        return None
    if result.get("cached_input_tokens", 0) > result["input_tokens"]:
        return None
    return result


def _dependencies(
    values: tuple[ArtifactDependency, ...],
) -> tuple[ArtifactDependency, ...]:
    result = tuple(sorted(values, key=lambda d: (d.kind, d.key)))
    _require_v2(len({(d.kind, d.key) for d in result}) == len(result))
    return result


@dataclass(frozen=True, slots=True, eq=False)
class _DependencyIndex:
    """Bind a validated map to one registry and its exact dependency tuple."""

    registry: DependencyRegistry
    dependencies: tuple[ArtifactDependency, ...] = field(init=False)
    values: Mapping[tuple[str, str], str] = field(init=False)

    def __post_init__(self) -> None:
        """Derive the map from current immutable records, never supplied values."""
        dependencies = self.registry.dependencies
        _require_v2(type(dependencies) is tuple)
        _require_v2(all(type(d) is ArtifactDependency for d in dependencies))
        validated = _dependencies(
            tuple(ArtifactDependency.model_validate(d) for d in dependencies)
        )
        object.__setattr__(self, "dependencies", dependencies)
        object.__setattr__(
            self,
            "values",
            MappingProxyType({(d.kind, d.key): d.digest for d in validated}),
        )


class DependencyRegistry(Contract):
    """Current semantic inputs, built once from the caller's ledger and config.

    Values must exclude revisions, absolute source positions and legacy IDs.
    Explicit context/order/terminology dependencies must also describe absence.
    The caller owns the meaning and completeness of each input digest.
    Copies and unvalidated construction cannot reuse the private index. Call
    model_validate on such records before using them as a current registry.
    """

    dependencies: tuple[ArtifactDependency, ...]
    _index: _DependencyIndex | None = PrivateAttr(default=None)

    @field_validator("dependencies")
    @classmethod
    def canonical(
        cls, value: tuple[ArtifactDependency, ...]
    ) -> tuple[ArtifactDependency, ...]:
        """Normalize dependency order and reject conflicting names."""
        return _dependencies(value)

    @model_validator(mode="after")
    def bind_index(self) -> DependencyRegistry:
        """Build the shared index only after strict field validation succeeds."""
        self._index = _DependencyIndex(self)
        return self

    def contains(self, required: tuple[ArtifactDependency, ...]) -> bool:
        """Check original peers in O(required), rejecting an unbound index."""
        index = getattr(self, "_index", None)
        if (
            type(index) is not _DependencyIndex
            or index.registry is not self
            or index.dependencies is not self.dependencies
        ):
            return False
        return all(
            index.values.get((d.kind, d.key)) == d.digest for d in required
        )


def static_artifact_dependencies(
    identity: RunIdentity,
    *,
    configuration_digest: str,
    prompt_digest: str,
    schema_digest: str,
    validator_digest: str,
) -> tuple[ArtifactDependency, ...]:
    """Build static identity dimensions without retaining raw provider config.

    Hash complete canonical config/prompt/schema/validator contracts beforehand.
    Provider/model digests use this helper's canonical JSON string convention;
    ArtifactRuntime verifies them against its exact RunIdentity.
    """
    values = {
        "provider": _canonical_digest(identity.provider_destination),
        "model": _canonical_digest(identity.model),
        "configuration": configuration_digest,
        "prompt": prompt_digest,
        "schema": schema_digest,
        "validator": validator_digest,
    }
    return _dependencies(
        tuple(
            ArtifactDependency(
                kind=kind, key=_canonical_digest(kind), digest=value
            )
            for kind, value in values.items()
        )
    )


class ArtifactLookup(Contract):
    """Find an item before packing misses; no new batch boundary enters its key.

    static_dependencies must contain provider/model/configuration/prompt/schema/
    validator. Context and every original peer are checked from the stored full
    descriptor after lookup. validator_version must be a short version label.
    """

    item_key: Digest
    result_kind: ResultKind = "detail"
    static_dependencies: tuple[ArtifactDependency, ...]
    validator_version: str

    @field_validator("validator_version")
    @classmethod
    def safe_version(cls, value: str) -> str:
        """Keep metadata free of arbitrary validator messages."""
        return _valid_identifier(value)

    @field_validator("static_dependencies")
    @classmethod
    def static_contract(
        cls, value: tuple[ArtifactDependency, ...]
    ) -> tuple[ArtifactDependency, ...]:
        """Require all static contract dimensions and no dynamic inputs."""
        value = _dependencies(value)
        _require_v2(
            {d.kind for d in value} == _STATIC_KINDS
            and len(value) == len(_STATIC_KINDS)
        )
        return value

    @property
    def key(self) -> str:
        """Return the content-safe index directory name."""
        return _canonical_digest(self.model_dump(mode="json"))


class ArtifactInvocation(Contract):
    """Original batch identity and complete consumed inputs, without bodies.

    ``batch`` is the actual original BatchSpec, never a preview. Each original
    item has both evidence and peer dependencies with matching digests. All
    static dependencies and explicit context/order/terminology dependencies are
    mandatory. Repair/split nodes retain this descriptor unchanged.
    """

    version: Literal["diffstory.invocation.v2"] = "diffstory.invocation.v2"
    batch: BatchSpec
    lookups: tuple[ArtifactLookup, ...]
    dependencies: tuple[ArtifactDependency, ...]

    @field_validator("dependencies")
    @classmethod
    def canonical(
        cls, value: tuple[ArtifactDependency, ...]
    ) -> tuple[ArtifactDependency, ...]:
        """Normalize dependency order before deriving semantic keys."""
        return _dependencies(value)

    @model_validator(mode="after")
    def complete(self) -> ArtifactInvocation:
        """Reject incomplete peers, duplicate keys and mismatched contracts."""
        _require_v2(
            tuple(item.item_key for item in self.lookups)
            == self.batch.item_keys
        )
        _require_v2(
            all(
                result_stage(item.result_kind) == self.batch.stage
                for item in self.lookups
            )
        )
        required = _STATIC_KINDS | _CONTEXT_KINDS | {"evidence", "peer"}
        _require_v2({d.kind for d in self.dependencies} == required)
        evidence = {
            d.key: d.digest for d in self.dependencies if d.kind == "evidence"
        }
        peers = {
            d.key: d.digest for d in self.dependencies if d.kind == "peer"
        }
        _require_v2(
            evidence == peers and set(peers) == set(self.batch.item_keys)
        )
        static = tuple(d for d in self.dependencies if d.kind in _STATIC_KINDS)
        _require_v2(
            all(item.static_dependencies == static for item in self.lookups)
        )
        return self

    @property
    def semantic_key(self) -> str:
        """Identify original semantic work independently of runs and requests."""
        return _canonical_digest(
            {
                "version": self.version,
                "stage": self.batch.stage,
                "items": [
                    item.model_dump(mode="json") for item in self.lookups
                ],
                "groups": self.batch.group_keys,
                "dependencies": [
                    d.model_dump(mode="json") for d in self.dependencies
                ],
            }
        )

    def artifact_key(self, lookup: ArtifactLookup) -> str:
        """Bind one result to its whole original consumed context."""
        _require_v2(lookup in self.lookups)
        return _canonical_digest(
            {"lookup": lookup.key, "invocation": self.semantic_key}
        )

    def artifact_keys(self) -> dict[str, str]:
        """Derive all item keys after hashing the shared invocation once."""
        invocation = self.semantic_key
        return {
            lookup.item_key: _canonical_digest(
                {"lookup": lookup.key, "invocation": invocation}
            )
            for lookup in self.lookups
        }


class PreparedMetadata(Contract):
    """Exact serialized request bounds, safe to persist before a send.

    input_bound must include estimate_input(body), framing/schema/repair reserves.
    output_guard is the full provider maximum unless enforceable_output_cap is
    true. The adapter must enforce response_byte_limit and timeout_seconds.
    """

    batch: BatchSpec
    context_tokens: Positive
    max_output_tokens: Positive
    enforceable_output_cap: bool = False
    overhead_bytes: Natural = 0
    safety_tokens: Natural = 0
    response_byte_limit: Positive
    timeout_seconds: Positive

    def check(self, body: bytes) -> None:
        """Recheck actual bytes, all reservations and practical limits."""
        batch = self.batch
        _require_v2(
            hashlib.sha256(body).hexdigest() == batch.request_digest,
            "request_mismatch",
        )
        _require_v2(
            batch.input_bound
            >= estimate_input(body, overhead_bytes=self.overhead_bytes),
            "input_limit",
        )
        _require_v2(
            batch.input_bound + batch.output_guard + self.safety_tokens
            <= self.context_tokens,
            "input_limit",
        )
        _require_v2(
            batch.output_reservation <= self.max_output_tokens, "output_limit"
        )
        minimum = (
            batch.output_reservation
            if self.enforceable_output_cap
            else self.max_output_tokens
        )
        _require_v2(batch.output_guard >= minimum, "output_limit")
        _require_v2(
            batch.estimated_response_bytes <= self.response_byte_limit,
            "response_limit",
        )
        _require_v2(
            batch.estimated_seconds <= self.timeout_seconds, "timeout_limit"
        )


class PreparedArtifactRequest(Contract):
    """Ephemeral exact body and typed bounds; the body is never persisted."""

    body: bytes = Field(exclude=True, repr=False)
    metadata: PreparedMetadata


def bind_prepared_request(
    request: PreparedRequest,
    capacity: ModelCapacity,
    limits: RequestPackingLimits,
    *,
    enforceable_output_cap: bool = False,
) -> PreparedArtifactRequest:
    """Bind actual public planner output to checked runtime send metadata.

    Use the same capacity, limits, and cap enforcement as prepare_request_items.
    Every initial, repair, split, or resumed node uses that public preparer and
    this adapter; callers need not repeat private reservation arithmetic.
    """
    metadata = PreparedMetadata(
        batch=request.batch,
        context_tokens=capacity.context_tokens,
        max_output_tokens=capacity.max_output_tokens,
        enforceable_output_cap=enforceable_output_cap,
        overhead_bytes=limits.framing_bytes
        + limits.schema_reserve_bytes
        + limits.repair_reserve_bytes,
        safety_tokens=limits.safety_tokens,
        response_byte_limit=limits.response_byte_limit,
        timeout_seconds=limits.timeout_seconds,
    )
    metadata.check(request.body)
    return PreparedArtifactRequest(body=request.body, metadata=metadata)


class StoredArtifact(Contract):
    """Private typed payload with the full original batch and audit identity."""

    format_version: Literal[2] = ARTIFACT_FORMAT_VERSION
    original_run_digest: Digest
    invocation: ArtifactInvocation
    lookup: ArtifactLookup
    record: ArtifactRecord
    production_request: PreparedMetadata

    @model_validator(mode="after")
    def bindings(self) -> StoredArtifact:
        """Check complete descriptor, payload, key and request relationships."""
        record, lookup, invocation = self.record, self.lookup, self.invocation
        _require_v2(record.semantic_key == invocation.artifact_key(lookup))
        _require_v2(record.dependencies == invocation.dependencies)
        _require_v2(record.validator_version == lookup.validator_version)
        _require_v2(
            record.original_request_digest == invocation.batch.request_digest
        )
        _require_v2(lookup.item_key in self.production_request.batch.item_keys)
        _require_v2(
            set(self.production_request.batch.item_keys)
            <= set(invocation.batch.item_keys)
        )
        _require_v2(
            self.production_request.batch.stage == invocation.batch.stage
        )
        _validate_result_key(lookup, record.result)
        return self


def _validate_result_key(
    lookup: ArtifactLookup, result: NarrativeResult
) -> None:
    _require_v2(
        type(result) is result_model(lookup.result_kind)
        and result_key(result) == lookup.item_key
    )


def _atomic_record(path: Path, record: Contract) -> None:
    """Replace one private, fsynced record without damaging its prior value."""
    temporary_path = None
    try:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=".v2-", dir=path.parent, delete=False
        ) as stream:
            temporary_path = Path(stream.name)
            temporary_path.chmod(0o600)
            stream.write(record.model_dump_json().encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
        temporary_path.replace(path)
        temporary_path = None
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except (OSError, ValueError, TypeError):
        msg = "storage"
        raise ArtifactRuntimeError(msg) from None
    finally:
        if temporary_path is not None:
            with suppress(OSError):
                temporary_path.unlink()


class RunManifest(Contract):
    """Exact run/config audit binding, without credentials or revision text."""

    version: Literal["diffstory.run.v2"] = "diffstory.run.v2"
    identity_digest: Digest
    configuration_digest: Digest

    @property
    def key(self) -> str:
        """Keep each exact run/config manifest in its own namespace."""
        return _canonical_digest(self.model_dump(mode="json"))


class ArtifactStore:
    """Global typed artifact indices; malformed or stale candidates are misses.

    Each index is a directory of immutable content-digest records. There is no
    shared mutable index whose updates can lose another process's artifacts.
    Only enabled artifact caches persist narration. Validators return None and
    must use the current ledger/index for coverage and citation checks.
    """

    def __init__(self, directory: str | Path) -> None:
        """Select the optional private artifact cache."""
        self.directory = Path(directory) / "artifacts-v2"

    def commit(self, artifact: StoredArtifact) -> None:
        """Write a fully typed artifact under its lookup/content digests."""
        artifact = StoredArtifact.model_validate(artifact)
        try:
            self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        except OSError:
            reason = "storage"
            raise ArtifactRuntimeError(reason) from None
        digest = _canonical_digest(artifact.model_dump(mode="json"))
        _atomic_record(
            self.directory / artifact.lookup.key / f"{digest}.json", artifact
        )

    def lookup(
        self,
        lookup: ArtifactLookup,
        registry: DependencyRegistry,
        validate: Callable[[ArtifactLookup, NarrativeResult], None],
    ) -> StoredArtifact | None:
        """Revalidate original dependencies and current result before reuse."""
        for path in sorted((self.directory / lookup.key).glob("*.json")):
            try:
                candidate = StoredArtifact.model_validate_json(
                    path.read_bytes()
                )
                if path.stem != _canonical_digest(
                    candidate.model_dump(mode="json")
                ):
                    continue
                if candidate.lookup != lookup or not registry.contains(
                    candidate.invocation.dependencies
                ):
                    continue
                _require_v2(validate(lookup, candidate.record.result) is None)
            except Exception:  # noqa: BLE001, S112 # Cache failures are silent misses.
                continue
            return candidate
        return None

    def import_typed(
        self,
        serialized: bytes,
        *,
        original_run: RunIdentity,
        invocation: ArtifactInvocation,
        registry: DependencyRegistry,
        validate: Callable[[ArtifactLookup, NarrativeResult], None],
    ) -> bool:
        """Import only already-v2 records; ordinary v1 prose is always a miss.

        Both the exact original run and request descriptor must match. There is
        no conversion, version relabeling, or copying prose into detail fields.
        """
        try:
            candidate = StoredArtifact.model_validate_json(serialized)
            if (
                candidate.original_run_digest != original_run.digest
                or candidate.invocation != invocation
            ):
                return False
            if not registry.contains(candidate.invocation.dependencies):
                return False
            _require_v2(
                validate(candidate.lookup, candidate.record.result) is None
            )
        except Exception:  # noqa: BLE001
            return False
        self.commit(candidate)
        return True


class RetryPolicy(Contract):
    """Durable per-original-batch reliability limits, not a call target.

    Each node allows one initial send, one transient retry and one targeted
    repair. Effective total is min(total_attempts, 3*(2*N-1)); effective depth is
    min(max_split_depth, N-1), where N is the original item count.
    """

    transient_retries: Literal[1] = 1
    targeted_repairs: Literal[1] = 1
    total_attempts: Positive = 12
    max_split_depth: Natural = 8


DEFAULT_RETRY_POLICY = RetryPolicy()
_MIN_SPLIT_CHILDREN = 2
_MAX_LOCK_SECONDS = 30


Purpose = Literal["initial", "transient", "repair"]
Outcome = Literal["reserved", "succeeded", "failed", "cancelled"]


class AttemptRecord(Contract):
    """One durable reservation; no body, response, usage history or error text."""

    number: Positive
    purpose: Purpose
    item_keys: tuple[Digest, ...] = Field(min_length=1)
    prepared: PreparedMetadata
    outcome: Outcome = "reserved"
    reason: Reason = "generic"


class RecoveryNode(Contract):
    """Metadata needed to rebuild one original, repair or split request."""

    key: Digest
    parent_key: Digest | None
    depth: Natural
    item_keys: tuple[Digest, ...] = Field(min_length=1)
    missing_keys: tuple[Digest, ...]
    children: tuple[Digest, ...] = ()
    attempts: tuple[AttemptRecord, ...] = ()
    status: Literal["pending", "active", "split", "succeeded", "failed"] = (
        "pending"
    )
    reason: Reason = "generic"

    @property
    def transient_count(self) -> int:
        """Return consumed transient reservations, including cancelled sends."""
        return sum(a.purpose == "transient" for a in self.attempts)

    @property
    def repair_count(self) -> int:
        """Return consumed repair reservations, including cancelled sends."""
        return sum(a.purpose == "repair" for a in self.attempts)


class LineageState(Contract):
    """Metadata-only shared budget; any malformed state fails closed."""

    version: Literal["diffstory.recovery.v2"] = "diffstory.recovery.v2"
    invocation: ArtifactInvocation
    policy: RetryPolicy
    nodes: tuple[RecoveryNode, ...]
    consumed_attempts: Natural = 0

    @property
    def key(self) -> str:
        """Return original semantic identity, independent of the exact run."""
        return self.invocation.semantic_key

    @property
    def attempt_cap(self) -> int:
        """Bound attempts by both policy and the maximum item-boundary tree."""
        return min(
            self.policy.total_attempts,
            3 * (2 * len(self.invocation.lookups) - 1),
        )

    @property
    def depth_cap(self) -> int:
        """No lineage can split further than its original item count allows."""
        return min(
            self.policy.max_split_depth, len(self.invocation.lookups) - 1
        )

    @model_validator(mode="after")
    def consistent(self) -> LineageState:
        """Check budget totals, exact partitions and every parent/attempt link."""
        nodes = {node.key: node for node in self.nodes}
        _require_v2(
            bool(nodes) and len(nodes) == len(self.nodes), "state_corrupt"
        )
        root = self.nodes[0]
        _require_v2(
            root.key == self.key
            and root.parent_key is None
            and root.depth == 0,
            "state_corrupt",
        )
        _require_v2(
            root.item_keys == self.invocation.batch.item_keys, "state_corrupt"
        )
        numbers = []
        reachable = {root.key}
        for node in self.nodes:
            _require_v2(
                node.key in reachable and node.depth <= self.depth_cap,
                "state_corrupt",
            )
            _require_v2(
                len(set(node.item_keys)) == len(node.item_keys),
                "state_corrupt",
            )
            _require_v2(
                node.missing_keys
                == tuple(
                    k for k in node.item_keys if k in set(node.missing_keys)
                ),
                "state_corrupt",
            )
            _require_v2(
                node.transient_count <= 1 and node.repair_count <= 1,
                "state_corrupt",
            )
            _require_v2(
                sum(a.purpose == "initial" for a in node.attempts) <= 1,
                "state_corrupt",
            )
            for attempt in node.attempts:
                numbers.append(attempt.number)
                _require_v2(
                    attempt.item_keys == attempt.prepared.batch.item_keys,
                    "state_corrupt",
                )
                _require_v2(
                    set(attempt.item_keys) <= set(node.item_keys),
                    "state_corrupt",
                )
                _require_v2(
                    len(set(attempt.item_keys)) == len(attempt.item_keys),
                    "state_corrupt",
                )
                _require_v2(
                    attempt.prepared.batch.stage
                    == self.invocation.batch.stage,
                    "state_corrupt",
                )
            if node.children:
                _require_v2(
                    node.status == "split"
                    and len(node.children) >= _MIN_SPLIT_CHILDREN,
                    "state_corrupt",
                )
                child_items = []
                for position, key in enumerate(node.children):
                    _require_v2(key in nodes, "state_corrupt")
                    child = nodes[key]
                    _require_v2(
                        child.parent_key == node.key
                        and child.depth == node.depth + 1,
                        "state_corrupt",
                    )
                    _require_v2(
                        key == _child_key(node.key, position, child.item_keys),
                        "state_corrupt",
                    )
                    _require_v2(
                        len(child.item_keys) < len(node.missing_keys),
                        "state_corrupt",
                    )
                    child_items.extend(child.item_keys)
                    reachable.add(key)
                _require_v2(
                    tuple(child_items) == node.missing_keys, "state_corrupt"
                )
            else:
                _require_v2(node.status != "split", "state_corrupt")
            if node.status == "succeeded":
                _require_v2(not node.missing_keys, "state_corrupt")
        _require_v2(len(reachable) == len(nodes), "state_corrupt")
        _require_v2(
            sorted(numbers) == list(range(1, self.consumed_attempts + 1)),
            "state_corrupt",
        )
        _require_v2(
            self.consumed_attempts <= self.attempt_cap, "state_corrupt"
        )
        return self


def _child_key(parent: str, position: int, keys: tuple[str, ...]) -> str:
    return _canonical_digest(
        {"parent": parent, "position": position, "items": keys}
    )


class RecoveryFile(Contract):
    """Check accidental disk corruption before interpreting durable state."""

    state: LineageState
    digest: Digest

    @model_validator(mode="after")
    def intact(self) -> RecoveryFile:
        """Never accept changed counters as a new budget."""
        _require_v2(
            self.digest
            == _canonical_digest(self.state.model_dump(mode="json")),
            "state_corrupt",
        )
        return self


class RecoveryClaim(Contract):
    """Permanent root claim: a missing state file must not reset its budget."""

    version: Literal["diffstory.recovery-claim.v2"] = (
        "diffstory.recovery-claim.v2"
    )
    key: Digest


def _replace_record(record: Contract, **changes: object) -> Contract:
    """Validate replacements; Pydantic model_copy(update=...) bypasses checks."""
    return type(record).model_validate({**record.model_dump(), **changes})


class RecoveryStore:
    """Durable metadata, with bounded OS locks across threads and processes.

    The directory lock protects discovery/create/read-modify-write. A separate
    lineage lease covers request preparation and execution, so a second process
    cannot treat an active reservation as an interrupted send. OS locks release
    on process exit; lock files are never deleted or cleared as stale.
    """

    def __init__(
        self, directory: str | Path, *, lock_timeout: float = 5.0
    ) -> None:
        """Require a real state directory, also when artifact cache is disabled."""
        _require_v2(
            type(lock_timeout) in {float, int}
            and 0 < lock_timeout <= _MAX_LOCK_SECONDS,
            "busy",
        )
        self.directory = Path(directory) / "recovery-v2"
        self.lock_timeout = lock_timeout

    @contextmanager
    def lease(self, key: str = "store") -> Iterator[None]:
        """Acquire a bounded flock; never delete another owner's lock file."""
        _require_v2(fcntl is not None, "storage")
        _require_v2(
            key == "store" or bool(_HEX_DIGEST.fullmatch(key)), "state_corrupt"
        )
        try:
            self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
            fd = os.open(
                self.directory / f"{key}.lock",
                os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW,
                0o600,
            )
        except OSError:
            msg = "storage"
            raise ArtifactRuntimeError(msg) from None
        deadline = time.monotonic() + self.lock_timeout
        try:
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        msg = "busy"
                        raise ArtifactRuntimeError(msg) from None
                    time.sleep(0.01)
                except OSError:
                    reason = "storage"
                    raise ArtifactRuntimeError(reason) from None
            yield
        finally:
            os.close(fd)

    def _read(self, path: Path) -> LineageState:
        try:
            claim = RecoveryClaim.model_validate_json(
                path.with_suffix(".claim").read_bytes()
            )
            _require_v2(claim.key == path.stem, "state_corrupt")
            saved = RecoveryFile.model_validate_json(path.read_bytes())
            state = saved.state
            _require_v2(path.stem == state.key, "state_corrupt")
        except Exception:  # noqa: BLE001 # Never reset corrupt durable counters.
            msg = "state_corrupt"
            raise ArtifactRuntimeError(msg) from None
        return state

    def _states(self) -> tuple[LineageState, ...]:
        claims = {p.stem for p in self.directory.glob("*.claim")}
        states = {p.stem for p in self.directory.glob("*.json")}
        _require_v2(claims == states, "state_corrupt")
        return tuple(
            self._read(path) for path in sorted(self.directory.glob("*.json"))
        )

    def discover(
        self, registry: DependencyRegistry
    ) -> tuple[LineageState, ...]:
        """Read current original lineages before packing any cache misses.

        Includes exhausted/completed lineages: those still own their budgets.
        A successful no-cache result cannot be recovered from metadata; a caller
        must supply revalidated in-memory results or report results_unavailable.
        """
        with self.lease():
            return tuple(
                s
                for s in self._states()
                if registry.contains(s.invocation.dependencies)
            )

    def ensure(
        self,
        invocation: ArtifactInvocation,
        policy: RetryPolicy,
        registry: DependencyRegistry,
    ) -> LineageState:
        """Find the original budget, or atomically claim previously unseen work."""
        _require_v2(
            registry.contains(invocation.dependencies), "dependencies_changed"
        )
        with self.lease():
            states = self._states()
            for state in states:
                if state.key == invocation.semantic_key:
                    _require_v2(state.policy == policy, "policy_conflict")
                    _require_v2(
                        state.invocation == invocation, "request_mismatch"
                    )
                    return state
            requested = {lookup.key for lookup in invocation.lookups}
            for state in states:
                if registry.contains(state.invocation.dependencies):
                    _require_v2(
                        not requested.intersection(
                            lookup.key for lookup in state.invocation.lookups
                        ),
                        "pending_lineage",
                    )
            node = RecoveryNode(
                key=invocation.semantic_key,
                parent_key=None,
                depth=0,
                item_keys=invocation.batch.item_keys,
                missing_keys=invocation.batch.item_keys,
            )
            state = LineageState(
                invocation=invocation, policy=policy, nodes=(node,)
            )
            _atomic_record(
                self.directory / f"{state.key}.claim",
                RecoveryClaim(key=state.key),
            )
            self._write(state)
            return state

    def load(self, key: str) -> LineageState:
        """Load one original lineage or fail closed if it was removed/corrupted."""
        _require_v2(bool(_HEX_DIGEST.fullmatch(key)), "state_corrupt")
        with self.lease():
            return self._read(self.directory / f"{key}.json")

    def _write(self, state: LineageState) -> None:
        state = LineageState.model_validate(state)
        saved = RecoveryFile(
            state=state,
            digest=_canonical_digest(state.model_dump(mode="json")),
        )
        _atomic_record(self.directory / f"{state.key}.json", saved)

    def update_node(self, key: str, node: RecoveryNode) -> LineageState:
        """Commit an outcome under the directory lock; caller holds root lease."""
        with self.lease():
            state = self._read(self.directory / f"{key}.json")
            old = next((n for n in state.nodes if n.key == node.key), None)
            _require_v2(
                old is not None and len(old.attempts) == len(node.attempts),
                "state_corrupt",
            )
            _require_v2(
                (
                    old.key,
                    old.parent_key,
                    old.depth,
                    old.item_keys,
                    old.children,
                )
                == (
                    node.key,
                    node.parent_key,
                    node.depth,
                    node.item_keys,
                    node.children,
                ),
                "state_corrupt",
            )
            for previous, current in zip(
                old.attempts, node.attempts, strict=True
            ):
                _require_v2(
                    (
                        previous.number,
                        previous.purpose,
                        previous.item_keys,
                        previous.prepared,
                    )
                    == (
                        current.number,
                        current.purpose,
                        current.item_keys,
                        current.prepared,
                    ),
                    "state_corrupt",
                )
            nodes = tuple(
                node if n.key == node.key else n for n in state.nodes
            )
            state = _replace_record(state, nodes=nodes)
            self._write(state)
            return state

    def reserve(
        self,
        key: str,
        node_key: str,
        purpose: Purpose,
        prepared: PreparedMetadata,
    ) -> LineageState:
        """Consume allowance durably before any provider send, under OS lock."""
        with self.lease():
            state = self._read(self.directory / f"{key}.json")
            _require_v2(
                state.consumed_attempts < state.attempt_cap, "attempt_limit"
            )
            node = next((n for n in state.nodes if n.key == node_key), None)
            _require_v2(
                node is not None and node.status in {"pending", "active"},
                "state_corrupt",
            )
            _require_v2(
                bool(node.missing_keys)
                and prepared.batch.item_keys == node.missing_keys,
                "request_mismatch",
            )
            _require_v2(
                purpose in {"initial", "transient", "repair"}, "state_corrupt"
            )
            _require_v2(
                not any(a.purpose == purpose for a in node.attempts),
                "attempt_limit",
            )
            if not node.attempts:
                _require_v2(purpose == "initial", "state_corrupt")
            elif purpose == "transient":
                _require_v2(
                    prepared == node.attempts[-1].prepared, "request_mismatch"
                )
            attempt = AttemptRecord(
                number=state.consumed_attempts + 1,
                purpose=purpose,
                item_keys=node.missing_keys,
                prepared=prepared,
            )
            node = _replace_record(
                node, attempts=(*node.attempts, attempt), status="active"
            )
            state = _replace_record(
                state,
                nodes=tuple(
                    node if n.key == node_key else n for n in state.nodes
                ),
                consumed_attempts=attempt.number,
            )
            self._write(state)
            return state

    def split(
        self, key: str, node_key: str, partitions: tuple[tuple[str, ...], ...]
    ) -> LineageState:
        """Commit a strict ordered partition of missing declared targets only."""
        with self.lease():
            state = self._read(self.directory / f"{key}.json")
            node = next((n for n in state.nodes if n.key == node_key), None)
            _require_v2(
                node is not None and not node.children, "state_corrupt"
            )
            _require_v2(
                node.depth < state.depth_cap and len(node.missing_keys) > 1,
                "split_limit",
            )
            _require_v2(
                type(partitions) is tuple
                and len(partitions) >= _MIN_SPLIT_CHILDREN,
                "split_limit",
            )
            _require_v2(
                all(
                    type(part) is tuple
                    and 0 < len(part) < len(node.missing_keys)
                    for part in partitions
                ),
                "split_limit",
            )
            _require_v2(
                tuple(k for part in partitions for k in part)
                == node.missing_keys,
                "split_limit",
            )
            children = tuple(
                RecoveryNode(
                    key=_child_key(node.key, i, part),
                    parent_key=node.key,
                    depth=node.depth + 1,
                    item_keys=part,
                    missing_keys=part,
                )
                for i, part in enumerate(partitions)
            )
            node = _replace_record(
                node,
                children=tuple(child.key for child in children),
                status="split",
            )
            state = _replace_record(
                state,
                nodes=tuple(
                    node if n.key == node_key else n for n in state.nodes
                )
                + children,
            )
            self._write(state)
            return state


class SafeDiagnostic(Contract):
    """Allowlisted telemetry; no request, source, response or exception text."""

    category: Category
    reason: Reason
    task_digest: Digest
    stage: BatchStage
    stage_position: Natural | None = None
    attempt: Natural | None
    input_bound: Natural | None = None
    output_reservation: Positive | None = None
    output_guard: Positive | None = None
    response_bytes: Natural | None = None
    usage_available: bool = False
    missing_items: Natural = 0
    invalid_items: Natural = 0
    valid_items: Natural = 0
    unknown_items: Natural = 0
    retries: Natural = 0
    repairs: Natural = 0
    splits: Natural = 0
    reused_artifacts: Natural = 0


@dataclass(frozen=True)
class SemanticJob:
    """Optional stage wrapper retaining semantic identity and input position."""

    task_digest: str
    run: Callable[[], object]

    def __post_init__(self) -> None:
        """Reject unsafe identity labels before scheduling."""
        _require_v2(bool(_HEX_DIGEST.fullmatch(self.task_digest)))
        _require_v2(callable(self.run))

    def __call__(self) -> object:
        """Run the caller-owned job through the legacy scheduler."""
        return self.run()


@dataclass(frozen=True)
class ProviderReply:
    """Ephemeral adapter output; response may be raw JSON or adapter wire data.

    response_bytes is measured by the adapter, not estimated from parsed JSON.
    Unknown lengths remain None. usage is sanitized before accounting.
    """

    response: object
    usage: dict[str, int] | None = None
    response_bytes: int | None = None


@dataclass(frozen=True)
class WireEnvelope:
    """Untrusted items after strict overall shape/version/binding validation.

    This envelope is never persisted. A custom adapter can expand compact keys
    before returning it. It must reject invalid framing and bind batch_key
    locally to the supplied BatchSpec, without asking a model to echo a digest
    of the same serialized request. Items are parsed independently afterward.
    """

    batch_key: str
    items: tuple[object, ...]
    failures: tuple[ItemFailure, ...] = ()


def bind_wire_envelope(raw: object, batch: BatchSpec) -> WireEnvelope:
    """Validate the default v2 JSON envelope, permitting an omitted batch key.

    A bad envelope yields no partial records. Malformed items inside an otherwise
    valid items array are left for per-item validation. Failures are envelope
    metadata and must already satisfy ItemFailure before any item is accepted.
    """
    if isinstance(raw, (bytes, str)):
        try:
            raw = json.loads(raw, object_pairs_hook=_unique_wire_object)
        except (ValueError, UnicodeError):
            msg = "malformed_json"
            raise ArtifactRuntimeError(msg) from None
    _require_v2(type(raw) is dict, "malformed_envelope")
    _require_v2(
        set(raw) <= {"version", "batch_key", "items", "failures"},
        "malformed_envelope",
    )
    _require_v2(
        raw.get("version") == "diffstory.batch-result.v2"
        and type(raw.get("items")) is list,
        "malformed_envelope",
    )
    _require_v2(
        "batch_key" not in raw or raw["batch_key"] == batch.id,
        "malformed_envelope",
    )
    failures = raw.get("failures", [])
    _require_v2(type(failures) is list, "malformed_envelope")
    try:
        parsed = tuple(
            ItemFailure.model_validate_json(json.dumps(failure))
            for failure in failures
        )
    except (ValueError, TypeError):
        msg = "malformed_envelope"
        raise ArtifactRuntimeError(msg) from None
    return WireEnvelope(batch.id, tuple(raw["items"]), parsed)


def _unique_wire_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    _require_v2(
        len({key for key, _value in pairs}) == len(pairs), "malformed_json"
    )
    return dict(pairs)


@dataclass(frozen=True)
class BatchCallbacks:
    """Caller-owned reconstruction, transport and current-ledger validation.

    prepare(invocation, node, purpose) reconstructs node.missing_keys from the
    current ledger and returns actual body bytes plus PreparedMetadata. For a
    transient retry it must reproduce the previous request exactly. For repair,
    include only declared missing targets, necessary original evidence/context,
    and safe reason codes. Never copy arbitrary invalid response/error bodies.
    send(prepared) performs exactly one provider attempt (no hidden retries).
    validate(lookup, result) returns None or raises; share one LedgerIndex here.
    envelope(raw, batch) rejects bad framing before returning independent items.
    split(node) returns ordered, disjoint, nonempty partitions of missing_keys.
    check(prepared) optionally enforces additional adapter-specific capacity.
    """

    prepare: Callable[
        [ArtifactInvocation, RecoveryNode, Purpose], PreparedArtifactRequest
    ]
    send: Callable[[PreparedArtifactRequest], ProviderReply]
    validate: Callable[[ArtifactLookup, NarrativeResult], None]
    envelope: Callable[[object, BatchSpec], WireEnvelope] = bind_wire_envelope
    split: Callable[[RecoveryNode], tuple[tuple[str, ...], ...]] | None = None
    check: Callable[[PreparedArtifactRequest], None] | None = None

    def __post_init__(self) -> None:
        """Reject incomplete adapters before reserving any provider work."""
        _require_v2(
            all(
                callable(callback)
                for callback in (
                    self.prepare,
                    self.send,
                    self.validate,
                    self.envelope,
                )
            )
        )
        _require_v2(self.split is None or callable(self.split))
        _require_v2(self.check is None or callable(self.check))


class ArtifactBatchOutcome(Contract):
    """Typed validated results, with no provider response or historical usage."""

    batch_key: Digest
    artifacts: tuple[StoredArtifact, ...]

    @model_validator(mode="after")
    def unique_results(self) -> ArtifactBatchOutcome:
        """Never expose duplicate accepted results to assembly."""
        _require_v2(
            len({a.lookup.item_key for a in self.artifacts})
            == len(self.artifacts)
        )
        return self

    def detail_batch(self) -> BatchResult:
        """Assemble the existing typed BatchResult after independent validation."""
        _require_v2(
            all(
                isinstance(a.record.result, DetailResult)
                for a in self.artifacts
            )
        )
        return BatchResult(
            batch_key=self.batch_key,
            items=tuple(a.record.result for a in self.artifacts),
            failures=(),
        )


class _ItemInspection(Contract):
    """Typed results and safe counts from one well-formed wire envelope."""

    results: tuple[NarrativeResult, ...]
    missing_keys: tuple[Digest, ...]
    invalid_keys: tuple[Digest, ...]
    unknown_count: Natural
    reason: Reason


def _inspect_items(  # noqa: C901, PLR0912 # Item parsing follows complete envelope grouping.
    envelope: WireEnvelope,
    invocation: ArtifactInvocation,
    batch: BatchSpec,
    validate: Callable[[ArtifactLookup, NarrativeResult], None],
    accept: Callable[[NarrativeResult], None],
) -> _ItemInspection:
    _require_v2(
        isinstance(envelope, WireEnvelope) and envelope.batch_key == batch.id,
        "malformed_envelope",
    )
    _require_v2(
        type(envelope.items) is tuple and type(envelope.failures) is tuple,
        "malformed_envelope",
    )
    _require_v2(
        all(isinstance(failure, ItemFailure) for failure in envelope.failures),
        "malformed_envelope",
    )
    lookups = {
        item.item_key: item
        for item in invocation.lookups
        if item.item_key in batch.item_keys
    }
    grouped: dict[str, list[object]] = {key: [] for key in batch.item_keys}
    unknown = 0
    fields = {
        result_key_field(lookup.result_kind) for lookup in invocation.lookups
    }
    for raw in envelope.items:
        present = (
            [raw[field] for field in fields if field in raw]
            if type(raw) is dict
            else []
        )
        key = present[0] if len(present) == 1 else None
        if type(key) is not str or key not in grouped:
            unknown += 1
        else:
            grouped[key].append(raw)
    for failure in envelope.failures:
        if failure.item_key in grouped:
            grouped[failure.item_key].append(failure)
        else:
            unknown += 1
    valid = []
    missing = []
    invalid = []
    reason: Reason = "unknown_item" if unknown else "generic"
    for key, items in grouped.items():
        if not items:
            missing.append(key)
            if reason == "generic":
                reason = "missing_item"
            continue
        if len(items) != 1:
            invalid.append(key)
            reason = "duplicate_item"
            continue
        if isinstance(items[0], ItemFailure):
            invalid.append(key)
            reason = {
                "invalid_contract": "schema",
                "invalid_citation": "citation",
                "incomplete_coverage": "coverage",
                "invalid_envelope": "malformed_envelope",
                "hierarchy_cannot_reduce": "input_limit",
            }.get(items[0].reason, _reason(items[0].reason))
            continue
        try:
            lookup = lookups[key]
            model = result_model(lookup.result_kind)
            result = model.model_validate_json(
                json.dumps(items[0], allow_nan=False)
            )
            _validate_result_key(lookup, result)
            _require_v2(validate(lookup, result) is None)
        except Exception as error:  # noqa: BLE001 # Arbitrary callback messages remain private.
            invalid.append(key)
            reason = (
                error.reason
                if isinstance(error, ArtifactRuntimeError)
                else "schema"
            )
        else:
            accept(result)
            valid.append(result)
    return _ItemInspection(
        results=tuple(valid),
        missing_keys=tuple(missing),
        invalid_keys=tuple(invalid),
        unknown_count=unknown,
        reason="unknown_item" if unknown else reason,
    )


class ArtifactRuntime(GenerationRuntime):
    """V2 execution with durable original-lineage budgets and optional artifacts.

    Public integration sequence::

        runtime = ArtifactRuntime(identity, recovery_dir=state_dir,
                                  artifact_dir=cache_dir)
        hits = runtime.prefetch(lookups, registry, callbacks.validate)
        originals = runtime.pending_lineages(registry)  # BEFORE packing misses
        for state in originals:
            runtime.execute_batch(state.invocation, registry, callbacks,
                                  policy=state.policy, available=hits)
        # Exclude every original lineage's keys, then pack genuinely new misses.
        outcome = runtime.execute_batch(invocation, registry, callbacks)
        details = outcome.detail_batch()

    recovery_dir is mandatory unless inferred from checkpoint_dir. artifact_dir
    None disables payload persistence, including on unresolved-model runs. The
    v1 GenerationRuntime constructor and invoke remain unchanged. For no-cache
    restart, completed results must come from revalidated caller memory; metadata
    alone cannot reconstruct prose and returns results_unavailable safely.

    pending_lineages returns current original descriptors, even exhausted ones.
    Never repack their missing keys into a new root. execute_batch also enforces
    this claim under an OS lock. Metadata does not expire or reset automatically.
    """

    def __init__(  # noqa: PLR0913 # Explicit independent storage and scheduling inputs.
        self,
        identity: RunIdentity,
        *,
        recovery_dir: str | Path | None = None,
        checkpoint_dir: str | Path | None = None,
        artifact_dir: str | Path | None = None,
        concurrency: int = DEFAULT_CONCURRENCY,
        on_progress: Callable[[ProgressEvent], None] | None = None,
        usage: RunUsage | None = None,
    ) -> None:
        """Require durable recovery independently of optional artifact reuse."""
        state_dir = (
            recovery_dir if recovery_dir is not None else checkpoint_dir
        )
        _require_v2(state_dir is not None, "storage")
        super().__init__(
            identity,
            concurrency=concurrency,
            on_progress=on_progress,
            usage=usage,
        )
        self.recovery = RecoveryStore(state_dir)
        resolved = identity.model.strip().casefold() not in {
            "default",
            "codex cli default",
        }
        self.artifacts = (
            ArtifactStore(artifact_dir)
            if artifact_dir is not None and resolved
            else None
        )
        self._diagnostics: list[SafeDiagnostic] = []
        self._reused_artifacts = 0
        self._retries = 0
        self._repairs = 0
        self._splits = 0

    @property
    def diagnostics(self) -> tuple[SafeDiagnostic, ...]:
        """Return immutable safe diagnostics from current execution only."""
        with self._lock:
            return tuple(self._diagnostics)

    @property
    def reused_artifacts(self) -> int:
        """Keep v2 artifact reuse separate from legacy checkpoint reuse."""
        with self._lock:
            return self._reused_artifacts

    @property
    def counters(self) -> dict[str, int]:
        """Return current attempts/retries/repairs/splits/reuse, never history."""
        with self._lock:
            return {
                "attempts": self.usage.report()["calls"],
                "retries": self._retries,
                "repairs": self._repairs,
                "splits": self._splits,
                "reused_artifacts": self._reused_artifacts,
            }

    def prefetch(
        self,
        lookups: tuple[ArtifactLookup, ...],
        registry: DependencyRegistry,
        validate: Callable[[ArtifactLookup, NarrativeResult], None],
    ) -> tuple[StoredArtifact, ...]:
        """Read reusable original artifacts before deciding new packing boundaries."""
        _require_v2(len({item.key for item in lookups}) == len(lookups))
        if self.artifacts is None:
            return ()
        found = []
        for lookup in lookups:
            self._check_identity(lookup)
            artifact = self.artifacts.lookup(lookup, registry, validate)
            if artifact is not None:
                found.append(artifact)
        with self._lock:
            self._reused_artifacts += len(found)
        return tuple(found)

    def _check_identity(self, lookup: ArtifactLookup) -> None:
        values = {d.kind: d.digest for d in lookup.static_dependencies}
        _require_v2(
            values["provider"]
            == _canonical_digest(self.identity.provider_destination)
            and values["model"] == _canonical_digest(self.identity.model),
            "dependencies_changed",
        )

    def pending_lineages(
        self, registry: DependencyRegistry
    ) -> tuple[LineageState, ...]:
        """Discover original budgets with current dependencies before repacking."""
        return self.recovery.discover(registry)

    def _manifest(self, invocation: ArtifactInvocation) -> None:
        static = [
            d.model_dump(mode="json")
            for d in invocation.dependencies
            if d.kind in _STATIC_KINDS
        ]
        manifest = RunManifest(
            identity_digest=self.identity.digest,
            configuration_digest=_canonical_digest(static),
        )
        _atomic_record(
            self.recovery.directory.parent
            / "runs-v2"
            / f"{manifest.key}.json",
            manifest,
        )

    def execute_batch(
        self,
        invocation: ArtifactInvocation,
        registry: DependencyRegistry,
        callbacks: BatchCallbacks,
        *,
        policy: RetryPolicy = DEFAULT_RETRY_POLICY,
        available: tuple[StoredArtifact, ...] = (),
    ) -> ArtifactBatchOutcome:
        """Resume an original lineage or execute new work without resetting caps.

        Returns only when all declared items validate. On failure, already valid
        artifacts remain individually committed and diagnostics describe safe
        counts. Caller exceptions and untrusted content never reach error text.
        The prepare/check callbacks run before EVERY provider send. Holding the
        original lease also makes post-crash reserved attempts distinguishable
        from another process's active send.
        """
        invocation = ArtifactInvocation.model_validate(invocation)
        registry = DependencyRegistry.model_validate(registry)
        policy = RetryPolicy.model_validate(policy)
        _require_v2(isinstance(callbacks, BatchCallbacks))
        self._local.diagnostic = None
        try:
            for lookup in invocation.lookups:
                self._check_identity(lookup)
            self._check_cancelled()
            _require_v2(
                registry.contains(invocation.dependencies),
                "dependencies_changed",
            )
            self._manifest(invocation)
            with self.recovery.lease(invocation.semantic_key):
                state = self.recovery.ensure(invocation, policy, registry)
                results = self._available(
                    invocation, registry, callbacks, available
                )
                if len(results) == len(invocation.lookups):
                    return self._outcome(invocation, results)
                self._check_lost_results(state, results)
                for node in state.nodes:
                    if not node.children:
                        self._execute_node(
                            state.key, node.key, registry, callbacks, results
                        )
                state = self.recovery.load(state.key)
                _require_v2(
                    len(results) == len(invocation.lookups),
                    "results_unavailable",
                )
                return self._outcome(invocation, results)
        except ArtifactRuntimeError as error:
            self._attach_diagnostic(invocation, error)
            raise

    def _attach_diagnostic(
        self, invocation: ArtifactInvocation, error: ArtifactRuntimeError
    ) -> None:
        diagnostic = getattr(self._local, "diagnostic", None)
        if (
            diagnostic is None
            or diagnostic.task_digest != invocation.semantic_key
            or diagnostic.reason != error.reason
        ):
            category: Category = "recovery"
            if error.reason == "storage":
                category = "storage"
            attempts = None
            if error.reason != "state_corrupt":
                with suppress(ArtifactRuntimeError):
                    attempts = self.recovery.load(
                        invocation.semantic_key
                    ).consumed_attempts
            diagnostic = SafeDiagnostic(
                category=category,
                reason=error.reason,
                task_digest=invocation.semantic_key,
                stage=invocation.batch.stage,
                stage_position=getattr(self._local, "stage_position", None),
                attempt=attempts,
                input_bound=invocation.batch.input_bound,
                output_reservation=invocation.batch.output_reservation,
                output_guard=invocation.batch.output_guard,
                missing_items=len(invocation.lookups),
                retries=self.counters["retries"],
                repairs=self.counters["repairs"],
                splits=self.counters["splits"],
                reused_artifacts=self.reused_artifacts,
            )
            with self._lock:
                self._diagnostics.append(diagnostic)
            self._local.diagnostic = diagnostic
        error.diagnostic = diagnostic

    def _available(
        self,
        invocation: ArtifactInvocation,
        registry: DependencyRegistry,
        callbacks: BatchCallbacks,
        available: tuple[StoredArtifact, ...],
    ) -> dict[str, StoredArtifact]:
        results = {}
        desired = {lookup.key: lookup for lookup in invocation.lookups}
        for candidate in available:
            artifact = StoredArtifact.model_validate(candidate)
            if artifact.lookup.key in desired and registry.contains(
                artifact.invocation.dependencies
            ):
                try:
                    _require_v2(
                        callbacks.validate(
                            artifact.lookup, artifact.record.result
                        )
                        is None
                    )
                except Exception:  # noqa: BLE001, S112 # Invalid cached items are misses.
                    continue
                results[artifact.lookup.item_key] = artifact
        missing = tuple(
            lookup
            for lookup in invocation.lookups
            if lookup.item_key not in results
        )
        for artifact in self.prefetch(missing, registry, callbacks.validate):
            results[artifact.lookup.item_key] = artifact
        return results

    @staticmethod
    def _outcome(
        invocation: ArtifactInvocation, results: dict[str, StoredArtifact]
    ) -> ArtifactBatchOutcome:
        return ArtifactBatchOutcome(
            batch_key=invocation.batch.id,
            artifacts=tuple(
                results[key] for key in invocation.batch.item_keys
            ),
        )

    @staticmethod
    def _check_lost_results(
        state: LineageState, results: dict[str, StoredArtifact]
    ) -> None:
        for node in state.nodes:
            completed = set(node.item_keys) - set(node.missing_keys)
            _require_v2(completed <= results.keys(), "results_unavailable")

    def _execute_node(  # noqa: C901, PLR0912, PLR0915 # One durable state-machine transition loop.
        self,
        key: str,
        node_key: str,
        registry: DependencyRegistry,
        callbacks: BatchCallbacks,
        results: dict[str, StoredArtifact],
    ) -> None:
        while True:
            state = self.recovery.load(key)
            node = next(n for n in state.nodes if n.key == node_key)
            if node.children:
                for child in node.children:
                    self._execute_node(
                        key, child, registry, callbacks, results
                    )
                return
            remaining = tuple(k for k in node.missing_keys if k not in results)
            if remaining != node.missing_keys:
                node = _replace_record(node, missing_keys=remaining)
                self.recovery.update_node(key, node)
            if not remaining:
                if node.status != "succeeded":
                    node = _replace_record(node, status="succeeded")
                    self.recovery.update_node(key, node)
                return
            _require_v2(node.status != "failed", node.reason)
            if node.attempts and node.attempts[-1].outcome == "reserved":
                last = _replace_record(
                    node.attempts[-1], outcome="cancelled", reason="cancelled"
                )
                node = _replace_record(
                    node,
                    attempts=(*node.attempts[:-1], last),
                    reason="cancelled",
                )
                state = self.recovery.update_node(key, node)
            action = self._next_action(node)
            if action == "split":
                self._split_node(state, node, callbacks)
                continue
            if action is None:
                self._fail_node(state, node, node.reason)
            self._check_cancelled()
            try:
                prepared = callbacks.prepare(state.invocation, node, action)
                _require_v2(
                    isinstance(prepared, PreparedArtifactRequest),
                    "request_mismatch",
                )
                prepared.metadata.check(prepared.body)
                if callbacks.check is not None:
                    callbacks.check(prepared)
                self._check_request(state, node, action, prepared.metadata)
            except Exception as error:  # noqa: BLE001
                reason = (
                    error.reason
                    if isinstance(error, ArtifactRuntimeError)
                    else "request_mismatch"
                )
                node = _replace_record(node, reason=reason)
                state = self.recovery.update_node(key, node)
                self._diagnose(state, node, reason)
                if reason in _SIZE_REASONS and callbacks.split is not None:
                    self._split_node(state, node, callbacks)
                    continue
                self._fail_node(state, node, reason)
            _require_v2(
                registry.contains(state.invocation.dependencies),
                "dependencies_changed",
            )
            self._check_cancelled()
            state = self.recovery.reserve(
                key, node.key, action, prepared.metadata
            )
            node = next(n for n in state.nodes if n.key == node_key)
            inspection, reason, response_size, usage_available = self._attempt(
                state, node, prepared, callbacks, results
            )
            missing = tuple(k for k in node.missing_keys if k not in results)
            succeeded = not missing and reason == "generic"
            last = _replace_record(
                node.attempts[-1],
                outcome="succeeded" if succeeded else "failed",
                reason=reason,
            )
            node = _replace_record(
                node,
                attempts=(*node.attempts[:-1], last),
                missing_keys=missing,
                status="succeeded" if succeeded else "active",
                reason=reason,
            )
            state = self.recovery.update_node(key, node)
            self._diagnose(
                state,
                node,
                reason,
                prepared.metadata,
                inspection,
                response_size,
                usage_available,
            )
            if succeeded:
                return
            # Unknown extras remain failed diagnostics. Known gaps can use only
            # the original node's remaining recovery allowance.
            if reason in _STOP_REASONS | {"storage"} or not missing:
                self._fail_node(state, node, reason)

    @staticmethod
    def _next_action(node: RecoveryNode) -> Purpose | Literal["split"] | None:
        if not node.attempts:
            return "initial"
        if node.reason in _STOP_REASONS:
            return None
        if node.reason in _TRANSIENT_REASONS | {"cancelled"}:
            if node.missing_keys != node.attempts[-1].item_keys:
                return "repair" if not node.repair_count else None
            return "transient" if not node.transient_count else None
        if node.reason in _SIZE_REASONS:
            return "split"
        return "repair" if not node.repair_count else None

    @staticmethod
    def _check_request(
        state: LineageState,
        node: RecoveryNode,
        purpose: Purpose,
        metadata: PreparedMetadata,
    ) -> None:
        batch = metadata.batch
        _require_v2(batch.item_keys == node.missing_keys, "request_mismatch")
        _require_v2(
            batch.stage == state.invocation.batch.stage
            and set(batch.group_keys)
            <= set(state.invocation.batch.group_keys),
            "request_mismatch",
        )
        if (
            purpose == "initial"
            and node.parent_key is None
            and node.missing_keys == node.item_keys
        ):
            _require_v2(batch == state.invocation.batch, "request_mismatch")
        if purpose == "transient":
            _require_v2(
                metadata == node.attempts[-1].prepared, "request_mismatch"
            )

    def _split_node(
        self,
        state: LineageState,
        node: RecoveryNode,
        callbacks: BatchCallbacks,
    ) -> None:
        if callbacks.split is None:
            self._fail_node(state, node, node.reason)
        try:
            partitions = callbacks.split(node)
        except Exception:  # noqa: BLE001
            self._fail_node(state, node, "split_limit")
        self.recovery.split(state.key, node.key, partitions)
        with self._lock:
            self._splits += 1

    def _fail_node(
        self, state: LineageState, node: RecoveryNode, reason: str
    ) -> None:
        node = _replace_record(node, status="failed", reason=_reason(reason))
        self.recovery.update_node(state.key, node)
        raise ArtifactRuntimeError(reason) from None

    def _commit_item(
        self,
        state: LineageState,
        metadata: PreparedMetadata,
        result: NarrativeResult,
        results: dict[str, StoredArtifact],
    ) -> None:
        item_key = result_key(result)
        lookup = next(
            item
            for item in state.invocation.lookups
            if item.item_key == item_key
        )
        record = ArtifactRecord(
            semantic_key=state.invocation.artifact_key(lookup),
            dependencies=state.invocation.dependencies,
            original_request_digest=state.invocation.batch.request_digest,
            validator_version=lookup.validator_version,
            result=result,
        )
        artifact = StoredArtifact(
            original_run_digest=self.identity.digest,
            invocation=state.invocation,
            lookup=lookup,
            record=record,
            production_request=metadata,
        )
        if self.artifacts is not None:
            self.artifacts.commit(artifact)
        results[item_key] = artifact

    def _attempt(
        self,
        state: LineageState,
        node: RecoveryNode,
        prepared: PreparedArtifactRequest,
        callbacks: BatchCallbacks,
        results: dict[str, StoredArtifact],
    ) -> tuple[_ItemInspection | None, Reason, int | None, bool]:
        usage = None
        response_size = None
        reason: Reason = "generic"
        inspection = None
        with self._lock:
            self._active_calls += 1
            self._retries += node.attempts[-1].purpose == "transient"
            self._repairs += node.attempts[-1].purpose == "repair"
        try:
            # The reservation is already durable. Count failed/cancelled attempts
            # even when transport returns no usage or is interrupted.
            try:
                self._emit("call_started", task_id=state.key)
                self._check_cancelled()
                reply = callbacks.send(prepared)
                _require_v2(
                    isinstance(reply, ProviderReply), "malformed_envelope"
                )
                usage = _safe_usage(reply.usage)
                response_size = (
                    reply.response_bytes
                    if type(reply.response_bytes) is int
                    and reply.response_bytes >= 0
                    else None
                )
                if response_size is None and type(reply.response) is bytes:
                    response_size = len(reply.response)
                if response_size is not None:
                    _require_v2(
                        response_size <= prepared.metadata.response_byte_limit,
                        "response_limit",
                    )
                envelope = callbacks.envelope(
                    reply.response, prepared.metadata.batch
                )
                inspection = _inspect_items(
                    envelope,
                    state.invocation,
                    prepared.metadata.batch,
                    callbacks.validate,
                    lambda result: self._commit_item(
                        state, prepared.metadata, result, results
                    ),
                )
                reason = inspection.reason
            except Exception as error:  # noqa: BLE001
                reason = self._attempt_reason(error)
                if isinstance(error, (ArtifactCallError, RuntimeCallError)):
                    usage = _safe_usage(error.usage)
                if isinstance(error, ArtifactCallError):
                    response_size = error.response_bytes
            except BaseException:
                last = _replace_record(
                    node.attempts[-1], outcome="cancelled", reason="cancelled"
                )
                node = _replace_record(
                    node,
                    attempts=(*node.attempts[:-1], last),
                    reason="cancelled",
                )
                self.recovery.update_node(state.key, node)
                self._diagnose(state, node, "cancelled", prepared.metadata)
                raise
        finally:
            with self._lock:
                self.usage.record(usage)
                self._active_calls -= 1
        return inspection, reason, response_size, usage is not None

    @staticmethod
    def _attempt_reason(error: Exception) -> Reason:
        if isinstance(error, ArtifactRuntimeError):
            return error.reason
        if isinstance(error, RuntimeCallError):
            return {
                "authentication": "auth",
                "timeout": "timeout",
                "transport": "transport",
                "rate_limit": "rate_limit",
            }.get(error.category, "generic")
        if isinstance(error, TimeoutError):
            return "timeout"
        if isinstance(error, OSError):
            return "transport"
        if isinstance(error, _StageCancelledError):
            return "cancelled"
        return "generic"

    def _diagnose(  # noqa: PLR0913 # One safe telemetry record.
        self,
        state: LineageState,
        node: RecoveryNode,
        reason: str,
        metadata: PreparedMetadata | None = None,
        inspection: _ItemInspection | None = None,
        response_size: int | None = None,
        usage_available: bool = False,  # noqa: FBT001, FBT002
    ) -> None:
        category: Category = "validation"
        if reason in _SIZE_REASONS:
            category = "capacity"
        elif reason in _TRANSIENT_REASONS | _STOP_REASONS:
            category = "provider"
        elif reason == "cancelled":
            category = "cancelled"
        batch = metadata.batch if metadata is not None else None
        with self._lock:
            event = SafeDiagnostic(
                category=category,
                reason=_reason(reason),
                task_digest=state.key,
                stage=state.invocation.batch.stage,
                stage_position=getattr(self._local, "stage_position", None),
                attempt=state.consumed_attempts,
                input_bound=batch.input_bound if batch else None,
                output_reservation=batch.output_reservation if batch else None,
                output_guard=batch.output_guard if batch else None,
                response_bytes=response_size,
                usage_available=usage_available,
                missing_items=len(inspection.missing_keys)
                if inspection
                else len(node.missing_keys),
                invalid_items=len(inspection.invalid_keys)
                if inspection
                else 0,
                valid_items=len(inspection.results) if inspection else 0,
                unknown_items=inspection.unknown_count if inspection else 0,
                retries=self._retries,
                repairs=self._repairs,
                splits=self._splits,
                reused_artifacts=self._reused_artifacts,
            )
            self._diagnostics.append(event)
        self._local.diagnostic = event
        self._emit(
            "call_completed",
            task_id=state.key,
            outcome="succeeded" if reason == "generic" else "failed",
        )


# Resolve the additive diagnostic annotation after its standalone definition.
ProgressEvent.model_rebuild()
