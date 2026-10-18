"""Read-only CLI. Run from the extracted folder with python -m diffstory."""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import tempfile
import time
from contextlib import ExitStack
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING
from typing import get_args

from pydantic import ValidationError

from . import __version__
from .analysis import SCHEMA
from .analysis import apply_annotations
from .analysis import compile_snapshot
from .analysis import evidence_packet
from .ingest import from_git
from .ingest import from_github
from .models import MAX_SNAPSHOT_SOURCE_BYTES
from .narrative import CODEX_MODEL_CAPACITIES
from .narrative import OPENAI_MODEL
from .narrative import CodexCLIProvider
from .narrative import NarrativeProvider
from .narrative import Narrator
from .narrative import OpenAIResponsesProvider
from .narrative_runtime import ArtifactRuntimeError
from .narrative_runtime import CheckpointError
from .narrative_runtime import ProgressEvent
from .narrative_runtime import Reason
from .narrative_runtime import RuntimeCallError
from .narrative_runtime import RuntimeStageError
from .narrative_runtime import SafeDiagnostic
from .render import iter_render_validated
from .render import render

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterable
    from collections.abc import Sequence
    from typing import BinaryIO

MAX_INPUT_BYTES = 80_000_000
MAX_REPORT_BYTES = 256_000_000
MAX_HTML_BYTES = 256_000_000
PROGRESS_INTERVAL_SECONDS = 5.0
_JSON_ASCII_MAX = 0x7F


class CliError(ValueError):
    """Carry only CLI-authored messages, never untrusted exception text."""


def _reject_json_constant(_value: str) -> None:
    """Reject non-JSON numeric constants without echoing input."""
    msg = "JSON must contain finite numbers"
    raise CliError(msg)


def _finite_json_float(value: str) -> float:
    """Reject numeric overflow without printing the untrusted numeric token."""
    result = float(value)
    if not math.isfinite(result):
        _reject_json_constant(value)
    return result


def _positive_integer(value: str) -> int:
    """Parse a positive integer command-line value.

    Args:
        value: One command-line token supplied for an integer option.

    Returns:
        The integer represented by ``value``, which is greater than zero.

    Raises:
        argparse.ArgumentTypeError: If ``value`` is not a positive integer.

    """
    try:
        parsed = int(value)
    except ValueError as error:
        msg = "must be a positive integer"
        raise argparse.ArgumentTypeError(msg) from error
    if parsed < 1:
        msg = "must be a positive integer"
        raise argparse.ArgumentTypeError(msg)
    return parsed


def read_json(path: str, *, max_bytes: int = MAX_INPUT_BYTES) -> dict:
    """
    Read one bounded UTF-8 JSON object from disk.

    Args:
        path: Input file path.
        max_bytes: Maximum encoded file size, independent of source limits.

    Returns:
        The decoded JSON object.

    Raises:
        OSError: If the file cannot be read.
        ValueError: If the file is too large or malformed.
        TypeError: If the JSON value is not an object.
        UnicodeError: If the file is not valid UTF-8.

    """
    raw = _read_bounded_bytes(path, max_bytes=max_bytes)
    return _decode_json_object(raw)


def _read_bounded_bytes(path: str, *, max_bytes: int) -> bytes:
    """Read exact input bytes and enforce the configured limit after stat.

    Args:
        path: Input file path.
        max_bytes: Positive maximum number of bytes to read.

    Returns:
        The exact input bytes when the file fits the configured limit.

    Raises:
        OSError: If the input file cannot be inspected or read.
        CliError: If the limit is invalid or the file exceeds it.

    """
    input_path = Path(path)
    if type(max_bytes) is not int or max_bytes < 1:
        msg = "JSON byte limit must be a positive integer"
        raise CliError(msg)
    if input_path.stat().st_size > max_bytes:
        msg = f"Input JSON exceeds the configured {max_bytes} byte limit"
        raise CliError(msg)
    # A bounded read also covers a file that grows after stat().
    with input_path.open("rb") as stream:
        raw = stream.read(max_bytes + 1)
    if len(raw) > max_bytes:
        msg = f"Input JSON exceeds the configured {max_bytes} byte limit"
        raise CliError(msg)

    return raw


def _decode_json_object(raw: bytes) -> dict:
    """Decode one strict UTF-8 JSON object without exposing input values.

    Args:
        raw: Bounded UTF-8 JSON bytes.

    Returns:
        The decoded JSON object.

    Raises:
        CliError: If the decoded JSON value is not an object.
        json.JSONDecodeError: If the bytes are malformed JSON.
        UnicodeError: If the bytes are not valid UTF-8.

    """
    value = json.loads(
        raw.decode("utf-8"),
        parse_constant=_reject_json_constant,
        parse_float=_finite_json_float,
    )
    if not isinstance(value, dict):
        msg = "JSON must be an object"
        raise CliError(msg)
    return value


def _looks_like_shared_storage(raw: bytes) -> bool:  # noqa: C901, PLR0911 # Keep the top-level discriminator scan closed and local.
    """Recognize a top-level report-storage version claim without building a graph.

    Args:
        raw: Bounded JSON bytes read from a report or annotation input.

    Returns:
        ``True`` when a top-level ``storage_version`` string uses the
        report-storage version namespace. Other metadata fields and nested
        values do not select the shared import path.

    Essential behavior:
        Scans only top-level member boundaries. It skips nested JSON values as
        bytes, so detection does not parse or retain a second report graph.
        Strict parsing and canonical envelope checks remain the importer's job.

    Preconditions:
        ``raw`` contains the bounded bytes read for one JSON input.
    Postconditions:
        The result is only a routing hint. It never validates or authorizes an
        envelope.
    """
    index = _skip_json_whitespace(raw, 0)
    if index >= len(raw) or raw[index] != ord("{"):
        return False

    index = _skip_json_whitespace(raw, index + 1)
    while index < len(raw) and raw[index] != ord("}"):
        if raw[index] != ord('"'):
            return False
        key_end = _json_string_end(raw, index)
        if key_end is None:
            return False
        try:
            key = json.loads(raw[index:key_end].decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError):
            return False
        index = _skip_json_whitespace(raw, key_end)
        if index >= len(raw) or raw[index] != ord(":"):
            return False
        value_start = _skip_json_whitespace(raw, index + 1)
        value_end = _skip_json_value(raw, value_start)
        if value_end is None:
            return False
        if key == "storage_version" and _json_string_has_prefix(
            raw, value_start, value_end, "diffstory.report-storage."
        ):
            return True

        index = _skip_json_whitespace(raw, value_end)
        if index >= len(raw):
            return False
        if raw[index] == ord(","):
            index = _skip_json_whitespace(raw, index + 1)
            continue
        return False
    return False


def _skip_json_whitespace(raw: bytes, index: int) -> int:
    """Advance over JSON whitespace in a bounded byte string.

    Args:
        raw: JSON bytes being scanned.
        index: Nonnegative byte offset at or before the end of ``raw``.

    Returns:
        The first non-whitespace offset, or ``len(raw)``.

    Preconditions:
        ``raw`` is bytes and ``index`` is a valid nonnegative offset.
    Postconditions:
        The input is unchanged and only the four JSON whitespace bytes are
        skipped.
    Essential behavior:
        Keeps the top-level scanner independent of decoded report strings.
    """
    while index < len(raw) and raw[index] in {
        ord(" "),
        ord("\t"),
        ord("\r"),
        ord("\n"),
    }:
        index += 1
    return index


def _json_string_end(raw: bytes, start: int) -> int | None:
    """Return the byte after a JSON string token, or ``None`` if it is cut off.

    Args:
        raw: JSON bytes being scanned.
        start: Byte offset of the opening quote.

    Returns:
        The offset after the closing quote, or ``None`` for an incomplete
        string token.

    Preconditions:
        ``raw`` is bytes and ``start`` is a nonnegative byte offset.
    Postconditions:
        No string value is decoded or retained.
    Essential behavior:
        Skips escaped bytes so quotes inside JSON strings do not change the
        surrounding top-level scan.
    """
    if start >= len(raw) or raw[start] != ord('"'):
        return None
    index = start + 1
    while index < len(raw):
        byte = raw[index]
        if byte == ord("\\"):
            index += 2
            continue
        if byte == ord('"'):
            return index + 1
        index += 1
    return None


def _skip_json_value(raw: bytes, start: int) -> int | None:  # noqa: C901, PLR0911 # Skip containers without building nested values.
    """Skip one JSON value without materializing strings, arrays, or objects.

    Args:
        raw: JSON bytes being scanned.
        start: Byte offset of the first value byte.

    Returns:
        The offset after one string, scalar, array, or object token, or
        ``None`` if a container or string is incomplete or mismatched.

    Preconditions:
        ``raw`` is bytes and ``start`` is a nonnegative byte offset.
    Postconditions:
        Nested JSON values are not parsed into Python containers.
    Essential behavior:
        Tracks only matching container delimiters and string escapes. Full
        syntax and type checks remain with the existing JSON and storage
        validators.
    """
    if start >= len(raw):
        return None
    first = raw[start]
    if first == ord('"'):
        return _json_string_end(raw, start)
    if first in {ord("{"), ord("[")}:
        closers = [ord("}") if first == ord("{") else ord("]")]
        index = start + 1
        while index < len(raw):
            byte = raw[index]
            if byte == ord('"'):
                string_end = _json_string_end(raw, index)
                if string_end is None:
                    return None
                index = string_end
                continue
            if byte in {ord("{"), ord("[")}:
                closers.append(ord("}") if byte == ord("{") else ord("]"))
            elif byte in {ord("}"), ord("]")}:
                if not closers or byte != closers[-1]:
                    return None
                closers.pop()
                if not closers:
                    return index + 1
            index += 1
        return None

    index = start
    while index < len(raw) and raw[index] not in {
        ord(","),
        ord("}"),
        ord(" "),
        ord("\t"),
        ord("\r"),
        ord("\n"),
    }:
        index += 1
    return index if index > start else None


def _json_string_has_prefix(  # noqa: C901, PLR0911, PLR0912 # Decode only a short reserved string prefix.
    raw: bytes, start: int, end: int, expected: str
) -> bool:
    """Compare a JSON string's decoded prefix without decoding its full value.

    Args:
        raw: JSON bytes being scanned.
        start: Opening quote offset for the candidate value.
        end: Offset immediately after its closing quote.
        expected: ASCII prefix reserved for the storage discriminator.

    Returns:
        ``True`` when the decoded string begins with ``expected``.

    Preconditions:
        ``raw`` contains the bounded input bytes and ``expected`` contains
        only ASCII characters.
    Postconditions:
        At most the expected prefix is decoded; the remaining string is not
        copied or normalized.
    Essential behavior:
        Supports ordinary JSON escapes for prefix characters while leaving
        complete string validation to the existing JSON parser.
    """
    if start >= end or raw[start] != ord('"') or raw[end - 1] != ord('"'):
        return False
    index = start + 1
    for expected_char in expected:
        if index >= end - 1:
            return False
        byte = raw[index]
        if byte == ord("\\"):
            index += 1
            if index >= end - 1:
                return False
            escape = raw[index]
            simple_escapes = {
                ord('"'): '"',
                ord("\\"): "\\",
                ord("/"): "/",
                ord("b"): "\b",
                ord("f"): "\f",
                ord("n"): "\n",
                ord("r"): "\r",
                ord("t"): "\t",
            }
            if escape == ord("u"):
                digits_end = index + 5
                if digits_end > end - 1:
                    return False
                try:
                    decoded_char = chr(
                        int(raw[index + 1 : digits_end].decode("ascii"), 16)
                    )
                except (UnicodeDecodeError, ValueError):
                    return False
                index = digits_end
            else:
                decoded_char = simple_escapes.get(escape)
                if decoded_char is None:
                    return False
                index += 1
        else:
            if byte > _JSON_ASCII_MAX:
                return False
            decoded_char = chr(byte)
            index += 1
        if decoded_char != expected_char:
            return False
    return True


def _add_narration_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Add source and narration options to a Git or GitHub command.

    Args:
        parser: Argument parser for the Git or GitHub source command.

    Side Effects:
        Adds source-limit, narration, provider, model, confirmation,
        annotation, concurrency, and mutually exclusive cache options directly
        to ``parser``.

    """
    parser.add_argument(
        "--max-source-bytes",
        type=int,
        default=MAX_SNAPSHOT_SOURCE_BYTES,
        help="Aggregate source limit across both revisions (default: 64000000; may be lowered)",
    )
    parser.add_argument(
        "--narrate",
        action="store_true",
        help="Generate model-written source explanations (sends PR source to the selected provider)",
    )
    parser.add_argument(
        "--provider", choices=["openai", "codex"], default="openai"
    )
    parser.add_argument(
        "--model",
        help=(
            "Optional model override (OpenAI: gpt-6.1-sol; Codex: "
            "gpt-6-astra, gpt-6.1-sol, gpt-6-luna, or gpt-5.3-codex)"
        ),
    )
    parser.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        help="Environment variable holding the OpenAI API key (OpenAI provider only)",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirm source transfer without an interactive prompt",
    )
    parser.add_argument(
        "--annotations-out",
        help="Candidate annotations path (default: <out>.annotations.json)",
    )
    parser.add_argument(
        "--concurrency",
        type=_positive_integer,
        default=4,
        help="Maximum concurrent narration tasks (default: 4)",
    )
    parser.add_argument(
        "--architecture",
        choices=["legacy", "v2", "walkthrough"],
        default="v2",
        help="Narration architecture (default: v2; legacy remains supported)",
    )
    state_group = parser.add_mutually_exclusive_group()
    state_group.add_argument(
        "--state-dir",
        type=Path,
        help="Durable v2 recovery directory (default: <out>.narration-state)",
    )
    state_group.add_argument(
        "--fresh-state",
        type=Path,
        metavar="NEW_DIRECTORY",
        help=(
            "Intentionally start new v2 attempt budgets in a new directory; "
            "keeps old state and may reuse validated cached results"
        ),
    )
    cache_group = parser.add_mutually_exclusive_group()
    cache_group.add_argument(
        "--cache-dir",
        type=Path,
        help=(
            "Directory for persistent narration checkpoints (default: output "
            "path with .narration-cache suffix)"
        ),
    )
    cache_group.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable payload caching; v2 still persists recovery and attempt budgets",
    )


def _build_parser() -> argparse.ArgumentParser:
    """
    Build the command-line parser for all supported Diffstory commands.

    Returns:
        A configured parser with required subcommands and their options.

    """
    parser = argparse.ArgumentParser(
        prog="diffstory",
        description=(
            "Compile a semantic code walkthrough. Repository code is never executed."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    git_parser = subparsers.add_parser(
        "git",
        help="Analyze committed revisions from a local Git checkout",
    )
    git_parser.add_argument("--repo", default=".")
    git_parser.add_argument("--base", required=True)
    git_parser.add_argument("--head", default="HEAD")
    git_parser.add_argument(
        "--two-dot",
        action="store_true",
        help="Compare endpoints instead of merge-base to head",
    )
    git_parser.add_argument("--max-files", type=int, default=500)

    github_parser = subparsers.add_parser(
        "github",
        help="Read a GitHub PR (uses gh auth or GITHUB_TOKEN)",
    )
    github_parser.add_argument("pr", help="owner/repo#123 or GitHub PR URL")
    github_parser.add_argument(
        "--token-env",
        default="GITHUB_TOKEN",
        help="environment variable for GitHub auth; falls back to signed-in gh CLI",
    )
    github_parser.add_argument("--max-files", type=int, default=500)

    for source_parser in (git_parser, github_parser):
        _add_narration_arguments(source_parser)

    snapshot_parser = subparsers.add_parser(
        "snapshot",
        help="Compile a saved diffstory.snapshot.v1 JSON document",
    )
    snapshot_parser.add_argument("input")
    _add_narration_arguments(snapshot_parser)

    render_parser = subparsers.add_parser(
        "render",
        help="Render an existing report, optionally with authored/model annotations",
    )
    render_parser.add_argument("input")

    evidence_parser = subparsers.add_parser(
        "evidence",
        help="Export evidence for a human or model; makes no network call",
    )
    evidence_parser.add_argument("input")
    evidence_parser.add_argument("--out", required=True)

    for report_parser in (
        git_parser,
        github_parser,
        snapshot_parser,
        render_parser,
        evidence_parser,
    ):
        report_parser.add_argument(
            "--max-report-bytes",
            type=_positive_integer,
            default=MAX_REPORT_BYTES,
            help=(
                "Report/annotation JSON read and write limit in bytes "
                "(default: 256000000); does not raise the source limit"
            ),
        )

    for command_parser in (
        git_parser,
        github_parser,
        snapshot_parser,
        render_parser,
    ):
        command_parser.add_argument(
            "--report-storage",
            choices=["expanded", "shared"],
            default="expanded",
            help=(
                "Report JSON storage (default: expanded; shared requires "
                "a complete v2 proof)"
            ),
        )
        command_parser.add_argument(
            "--max-html-bytes",
            type=_positive_integer,
            default=MAX_HTML_BYTES,
            help=(
                "Whole HTML output limit in UTF-8 bytes, including assets "
                "(default: 256000000)"
            ),
        )

    for command_parser in (
        git_parser,
        github_parser,
        snapshot_parser,
        render_parser,
    ):
        command_parser.add_argument(
            "--out", required=True, help="Output .html file"
        )
        command_parser.add_argument(
            "--annotations",
            help="Optional revision-bound narrative JSON",
        )
        command_parser.add_argument(
            "--save-snapshot",
            help="Save reusable input source JSON",
        )

    return parser


def _load_snapshot(args: argparse.Namespace) -> dict | None:
    """
    Load source data for a repository command or saved snapshot.

    Args:
        args: Parsed command-line arguments.

    Returns:
        A source snapshot for ``git``, ``github``, or ``snapshot``; ``None``
        for commands that consume report JSON instead.

    Raises:
        ValueError: If repository ingestion or snapshot decoding fails.
        OSError: If the saved snapshot cannot be read.

    """
    if args.command == "git":
        return from_git(
            args.repo,
            args.base,
            args.head,
            two_dot=args.two_dot,
            max_files=args.max_files,
            max_source_bytes=args.max_source_bytes,
        )
    if args.command == "github":
        return from_github(
            args.pr,
            token_env=args.token_env,
            max_files=args.max_files,
            max_source_bytes=args.max_source_bytes,
        )
    if args.command == "snapshot":
        return read_json(args.input)
    return None


def _print_narration_preview(
    provider: NarrativeProvider,
    report: dict,
    preview: dict,
) -> None:
    """
    Print the evidence-transfer destination and planned narration work.

    Args:
        provider: Selected narration provider and model.
        report: Compiled report whose source may be sent.
        preview: Local narration plan returned by ``Narrator.preview``.

    Side Effects:
        Writes a transfer summary to standard output; makes no provider call.

    """
    meta = report["meta"]
    base_sha = meta.get("base_sha", "")[:12]
    head_sha = meta.get("head_sha", "")[:12]
    source_scope = meta.get("scope", "supplied snapshot")
    changed_files = meta.get("changed_files", 0)
    source_bytes = meta.get("source_bytes", 0)

    print("Narration source-transfer preview")
    print(f"Provider: {provider.name} · model: {provider.model}")
    print(f"Destination: {provider.destination}")
    print(
        f"Source scope: {source_scope} · {changed_files} changed files · "
        f"{source_bytes} source bytes · {base_sha} → {head_sha}"
    )
    if preview.get("architecture") == "v2":
        print("Architecture: v2")
        print(
            f"Logical evidence units: {preview['units']} · "
            f"presentation groups: {preview['groups']}"
        )
        print(
            f"Base detail request plan: {preview['evidence_batches']} packed batches · "
            f"{preview['calls']} initial sends before reuse/recovery/repairs"
        )
        print(
            "Planning, section and document request counts: unknown until "
            "validated outputs. Actual attempts and reuse are reported during generation."
        )
        return
    print("Architecture: legacy")
    conditional_calls = preview.get("conditional_calls", 0)
    repair_note = (
        f" · up to {conditional_calls} citation repairs if needed"
        if conditional_calls
        else ""
    )
    print(
        f"Evidence chunks: {preview['chunks']} · planned calls: {preview['calls']}"
        f"{repair_note}"
    )


def _print_token_usage(usage: dict) -> None:
    """
    Print provider-reported token totals and any calls lacking usage data.

    Args:
        usage: Usage mapping returned by ``RunUsage.report``.

    Side Effects:
        Writes token totals to standard output.

    """
    print(
        "Provider-reported tokens: "
        f"input {usage['input_tokens']:,} "
        f"(cached {usage['cached_input_tokens']:,}) · "
        f"output {usage['output_tokens']:,}"
    )
    if usage["unreported_calls"]:
        print(
            f"Provider usage unavailable for "
            f"{usage['unreported_calls']:,} of {usage['calls']:,} calls"
        )


def _print_narration_summary(
    narrator: Narrator,
    annotations: dict | None = None,
    *,
    indexed_counts: tuple[int, int, int, int] | None = None,
) -> None:
    """Print provider usage and validated checkpoint reuse for this run.

    Args:
        narrator: Narrator whose generation has completed or failed.
        annotations: Successfully validated candidate, if available.
        indexed_counts: Receipt, detail, unique reused result, and reused
            producer counts read from a checked session payload.

    Side Effects:
        Writes current-run token totals and checkpoint reuse to standard output.

    """
    if getattr(narrator, "architecture", "legacy") == "v2":
        print("Current run usage (historical producer usage excluded):")
        _print_token_usage(narrator.usage.report())
        print(_v2_counter_text(narrator))
        if annotations is not None:
            manifest = annotations["generation"]
            receipts = manifest["receipts"]
            reused = [r for r in receipts if r["disposition"] == "reused"]
            print(
                f"Validated logical results: {len(receipts)} · "
                f"detail results: {len(manifest['details'])} · "
                f"unique reused results: {len({r['item_key'] for r in reused})}"
            )
            print(
                "Original producers for reused results: "
                f"{len({r['producer_key'] for r in reused})}; "
                "retained for audit, not charged to current usage."
            )
        elif indexed_counts is not None:
            (
                receipt_count,
                detail_count,
                unique_reused_count,
                reused_producer_count,
            ) = indexed_counts
            print(
                f"Validated logical results: {receipt_count} · "
                f"detail results: {detail_count} · "
                f"unique reused results: {unique_reused_count}"
            )
            print(
                "Original producers for reused results: "
                f"{reused_producer_count}; retained for audit, "
                "not charged to current usage."
            )
        for diagnostic in narrator.v2_diagnostics:
            if (
                diagnostic.reason != "generic"
                or diagnostic.missing_items
                or diagnostic.invalid_items
                or diagnostic.unknown_items
            ):
                print(_diagnostic_text(diagnostic), file=sys.stderr)
        return
    _print_token_usage(narrator.usage.report())
    print(f"Reused narration checkpoints: {narrator.reused_checkpoints}")


def _v2_counter_text(narrator: Narrator) -> str:
    """Label current attempts and lookup hits separately from logical results."""
    counters = narrator.v2_counters
    return (
        f"actual attempts {counters['attempts']} · retries {counters['retries']} · "
        f"targeted repairs {counters['repairs']} · splits {counters['splits']} · "
        f"reused artifacts (lookup hits) {counters['reused_artifacts']}"
    )


def _diagnostic_text(diagnostic: SafeDiagnostic) -> str:
    """Revalidate closed telemetry before printing any field, including labels."""
    try:
        safe = SafeDiagnostic.model_validate(diagnostic.model_dump())
    except (AttributeError, ValueError, TypeError):
        return "Narration diagnostic: schema"
    return "Narration diagnostic: " + " · ".join(
        f"{key}={value}"
        for key, value in safe.model_dump().items()
        if key != "category" and value is not None
    )


def _v2_progress_callback(
    narrator: Callable[[], Narrator],
) -> Callable[[ProgressEvent], None]:
    """Print batch lineage jobs and current counters, throttled at five seconds."""
    last_output_at: float | None = None

    def report(event: ProgressEvent) -> None:
        nonlocal last_output_at
        now = time.monotonic()
        if (
            event.event not in {"stage_started", "stage_completed"}
            and last_output_at is not None
            and now - last_output_at < PROGRESS_INTERVAL_SECONDS
        ):
            return
        last_output_at = now
        stage = (
            event.stage
            if event.stage in {"detail", "planning", "section", "document"}
            else "stage"
        )
        print(
            f"Narration progress: {stage} · provider batch lineages "
            f"{event.completed}/{event.planned} · active {event.active_calls} · "
            f"{_v2_counter_text(narrator())} · elapsed {event.elapsed_seconds:.0f}s",
            flush=True,
        )

    return report


def _progress_callback() -> Callable[[ProgressEvent], None]:
    """Create a progress printer throttled between narration stage boundaries.

    Returns:
        A callback that reports stage and job counts, active calls, total calls,
        reused checkpoints, and elapsed time.

    Side Effects:
        The returned callback writes progress to standard output at most once
        every five seconds, except that it always writes stage boundaries.

    """
    last_output_at: float | None = None

    def report(event: ProgressEvent) -> None:
        nonlocal last_output_at
        now = time.monotonic()
        stage_boundary = event.event in {"stage_started", "stage_completed"}
        if (
            not stage_boundary
            and last_output_at is not None
            and now - last_output_at < PROGRESS_INTERVAL_SECONDS
        ):
            return
        last_output_at = now
        print(
            f"Narration progress: {event.stage} · "
            f"tasks {event.completed}/{event.planned} · "
            f"active {event.active_calls} · calls {event.provider_calls} · "
            f"reused {event.reused_checkpoints} · "
            f"elapsed {event.elapsed_seconds:.0f}s",
            flush=True,
        )

    return report


def _checkpoint_directory(args: argparse.Namespace) -> Path | None:
    """Select the checkpoint directory for the chosen narration mode.

    Args:
        args: Parsed source and narration options.

    Returns:
        The requested or default checkpoint path, or ``None`` when caching is
        disabled.

    Raises:
        ValueError: If Codex uses its unresolved default model with an explicit
            checkpoint directory.

    Side Effects:
        Prints a notice when Codex's unresolved default model disables caching.

    """
    if args.provider == "codex" and args.model is None:
        if args.cache_dir is not None:
            msg = (
                "--cache-dir requires --model when using the Codex CLI default"
            )
            raise CliError(msg)
        if not args.no_cache:
            print(
                "Narration cache disabled: Codex uses its unresolved default "
                "model; pass --model to enable caching."
            )
        return None
    if args.no_cache:
        return None
    if args.cache_dir is not None:
        return args.cache_dir
    return Path(args.out).with_suffix(".narration-cache")


def _recovery_directory(args: argparse.Namespace) -> Path:
    """Select durable state without any automatic reset or fallback."""
    if args.fresh_state is not None:
        return args.fresh_state
    if args.state_dir is not None:
        return args.state_dir
    return Path(args.out).with_suffix(".narration-state")


def _confirm_source_transfer(
    *,
    assume_yes: bool,
    provider: NarrativeProvider,
) -> None:
    """
    Require consent before source evidence is transferred to a provider.

    Args:
        assume_yes: Whether the caller explicitly supplied ``--yes``.
        provider: Provider whose destination is presented for confirmation.

    Raises:
        ValueError: If interactive confirmation is unavailable or declined.

    Side Effects:
        May prompt on standard input. Returns without prompting when
        ``assume_yes`` is true.

    """
    if assume_yes:
        return
    if not sys.stdin.isatty():
        msg = "Source was not sent. Rerun with --yes only after approving the displayed transfer"
        raise CliError(msg)

    try:
        provider_name = getattr(provider, "consent_name", provider.name)
        answer = (
            input(
                f"Send source evidence to {provider_name} and generate narration? [y/N] "
            )
            .strip()
            .lower()
        )
    except EOFError:
        msg = "Source was not sent because confirmation was not available"
        raise CliError(msg) from None
    if answer not in {"y", "yes"}:
        msg = "Narration cancelled; no model request was made"
        raise CliError(msg)


def _generate_walkthrough(
    args: argparse.Namespace, report: dict
) -> tuple[dict, dict]:
    """Preview the complete bounded source plan, confirm it, and write the story.

    Args:
        args: Parsed source, provider, consent, and output options.
        report: Compiled pinned source to narrate.

    Returns:
        Flat annotations and the complete flat saved report.

    Raises:
        ValueError: If consent, capacity, or output validation fails.
        TimeoutError: If narration exceeds its four-minute deadline.
    """
    from .walkthrough import WalkthroughNarrator
    from .walkthrough import walkthrough_annotations

    provider = _create_provider(args)
    state = _recovery_directory(args)
    if args.fresh_state is not None and state.exists():
        msg = "--fresh-state requires a new directory; existing budgets are retained"
        raise CliError(msg)
    narrator = WalkthroughNarrator(
        provider,
        concurrency=args.concurrency,
        state_dir=state,
        cache_dir=_checkpoint_directory(args),
    )
    plan = narrator.prepare(report)
    print(f"Provider: {provider.name} · model: {provider.model}")
    print(f"Destination: {provider.destination}")
    print(
        f"Walkthrough: {len(plan.writers)} writing + {int(not plan.single)} editorial requests · {len(plan.packet['regions'])} changed regions · {sum(len(writer.body) for writer in plan.writers):,} prepared writing bytes"
    )
    print(
        "Limits: 12 total attempts · 4 concurrent calls · 240 seconds · 1 MiB model annotations"
    )
    _confirm_source_transfer(assume_yes=args.yes, provider=provider)
    generated = narrator.generate_prepared()
    _print_token_usage(narrator.usage.report())
    print(f"Cached results reused: {narrator.cache_hits}")
    return walkthrough_annotations(generated), generated


def _generate_narration(
    args: argparse.Namespace,
    report: dict,
) -> tuple[dict, dict]:
    """
    Generate validated narration annotations after preview and consent.

    Args:
        args: Parsed source and provider options.
        report: Deterministic report and source evidence to narrate.

    Returns:
        Candidate annotations and the report with those annotations applied.

    Raises:
        ValueError: If credentials, model capacity, consent, provider output,
            or provenance validation fails.
        Exception: Re-raises failures after printing drained current usage;
            ``main`` formats only safe codes and CLI-authored guidance.

    Side Effects:
        Prints a local transfer preview, throttled progress, and usage summary.
        After consent, may send source evidence to the selected provider. Does
        not write output files.

    """
    checkpoint_dir = _checkpoint_directory(args)
    provider = _create_provider(args)
    is_v2 = args.architecture == "v2"
    options = {}
    if is_v2:
        options = {
            "architecture": "v2",
            "recovery_dir": _recovery_directory(args),
        }
    narrator = Narrator(
        provider,
        concurrency=args.concurrency,
        checkpoint_dir=checkpoint_dir,
        on_progress=_v2_progress_callback(lambda: narrator)
        if is_v2
        else _progress_callback(),
        **options,
    )
    annotations = None
    try:
        preview = narrator.preview(report)
        _print_narration_preview(provider, report, preview)
        if is_v2:
            print(f"Durable recovery: {_recovery_directory(args)}")
            if checkpoint_dir is None:
                print(
                    "Payload cache disabled. Attempt budgets persist, but unavailable "
                    "result payloads cannot be recovered; state is never reset automatically."
                )
            if args.fresh_state is not None:
                print(
                    "Explicit fresh state: new attempt budgets; old state is retained."
                )
        _confirm_source_transfer(assume_yes=args.yes, provider=provider)
        if is_v2 and args.fresh_state is not None:
            try:
                args.fresh_state.mkdir(
                    mode=0o700, parents=True, exist_ok=False
                )
            except FileExistsError:
                msg = "--fresh-state requires a new directory; existing state was retained"
                raise CliError(msg) from None
        annotations = narrator.generate(report)
        annotated_report = apply_annotations(report, annotations)
    finally:
        _print_narration_summary(narrator, annotations)
    return annotations, annotated_report


def _generate_narration_v2(
    args: argparse.Namespace,
    report: dict,
    lifetime: ExitStack,
) -> tuple[object, object, object]:
    """Generate v2 annotations and keep their checked session handles live.

    Args:
        args: Parsed repository and v2 narration options.
        report: Compiled report approved for narration.
        lifetime: Main-command ``ExitStack`` that owns the session until every
            requested report writer completes.

    Returns:
        The live session, generation capability, and applied report
        capability. The session is already registered with ``lifetime`` and
        closes when the command exits.

    Preconditions:
        ``args.architecture`` is ``"v2"`` and source transfer was confirmed.
    Postconditions:
        Generation has run through ``generate_validated`` and ``session.apply``
        exactly once. The caller can stream either output format without
        dictionary application or rendering adapters.
    Essential behavior:
        Reuses the exact preview capture and keeps the same session open across
        apply, report/annotation serialization, and HTML rendering.

    Raises:
        CliError: If source transfer is declined or a fresh state directory
            already exists.
        ValueError: If narration validation fails.
    """
    checkpoint_dir = _checkpoint_directory(args)
    provider = _create_provider(args)
    narrator = Narrator(
        provider,
        concurrency=args.concurrency,
        checkpoint_dir=checkpoint_dir,
        on_progress=_v2_progress_callback(lambda: narrator),
        architecture="v2",
        recovery_dir=_recovery_directory(args),
    )
    generation = None
    report_handle = None
    indexed_counts = None
    try:
        preview = narrator.preview(report)
        _print_narration_preview(provider, report, preview)
        print(f"Durable recovery: {_recovery_directory(args)}")
        if checkpoint_dir is None:
            print(
                "Payload cache disabled. Attempt budgets persist, but unavailable "
                "result payloads cannot be recovered; state is never reset automatically."
            )
        if args.fresh_state is not None:
            print(
                "Explicit fresh state: new attempt budgets; old state is retained."
            )
        _confirm_source_transfer(assume_yes=args.yes, provider=provider)
        if args.fresh_state is not None:
            try:
                args.fresh_state.mkdir(
                    mode=0o700, parents=True, exist_ok=False
                )
            except FileExistsError:
                msg = "--fresh-state requires a new directory; existing state was retained"
                raise CliError(msg) from None

        session = lifetime.enter_context(narrator.capture_session(report))
        generation = narrator.generate_validated(report, session=session)
        report_handle = session.apply(generation)
        application = session._checked_report_data(  # noqa: SLF001 # Read only counts from the exact session-owned report capability.
            report_handle
        )
        receipt_rows = [
            application.graph.record("receipts", reference.storage_id)
            for reference in application.records.receipts
        ]
        reused_rows = [
            receipt
            for receipt in receipt_rows
            if receipt["disposition"] == "reused"
        ]
        indexed_counts = (
            len(application.records.receipts),
            len(application.records.details),
            len({receipt["item_key"] for receipt in reused_rows}),
            len({receipt["producer_key"] for receipt in reused_rows}),
        )
    finally:
        _print_narration_summary(
            narrator,
            indexed_counts=indexed_counts,
        )

    return session, generation, report_handle


def _create_provider(args: argparse.Namespace) -> NarrativeProvider:
    """Check known model choices and credentials without printing adapter errors."""
    if args.provider == "openai":
        if args.model not in {None, OPENAI_MODEL}:
            msg = f"OpenAI narration currently supports only {OPENAI_MODEL}"
            raise CliError(msg)
    elif args.model is not None and args.model not in CODEX_MODEL_CAPACITIES:
        msg = "Codex model has no known capacity; choose a supported --model from --help"
        raise CliError(msg)
    try:
        provider = (
            OpenAIResponsesProvider(token_env=args.api_key_env)
            if args.provider == "openai"
            else CodexCLIProvider(model=args.model)
        )
        provider.require_credentials()
    except Exception:  # noqa: BLE001 # Adapter setup can contain credential text.
        msg = (
            "Provider setup failed (auth). Check the API key in the environment "
            "selected by --api-key-env and its API access."
            if args.provider == "openai"
            else "Provider setup failed (auth). Check that Codex is available and sign in with `codex login`."
        )
        raise CliError(msg) from None
    return provider


def _write_outputs(  # noqa: C901, PLR0912 # Keep one concrete staging and promotion boundary.
    args: argparse.Namespace,
    report: object,
    snapshot: dict | None,
    annotations: object | None,
    *,
    session: object | None = None,
) -> None:
    """Stage bounded outputs and promote them after every writer completes.

    Args:
        args: Parsed output and optional snapshot paths.
        report: Expanded report mapping or session-issued report capability.
        snapshot: Source snapshot to save when requested, or ``None``.
        annotations: Expanded candidate or session-issued generation
            capability to save, or ``None``.
        session: The live session that issued any supplied capabilities.

    Raises:
        CliError: If output paths conflict, a storage mode lacks a complete
            v2 handle, or an output exceeds its configured byte limit.
        OSError: If a stage file cannot be created, written, or promoted.
        ValueError: If report validation or a session output check fails.

    Side Effects:
        Creates parent directories and sibling stage files. Promotes requested
        files only after every serializer and renderer finishes successfully.
        Prints output paths, report counts, and warnings after promotion.

    Promotion Limit:
        Staging protects all existing destinations from validation, encoding,
        size, and write failures. Filesystem promotion can still fail partway
        through a set of different directories, after an earlier destination
        was already replaced.

    """
    output_path = Path(args.out)
    report_path = output_path.with_suffix(".report.json")
    candidate_path = None
    destinations = [output_path, report_path]
    if annotations is not None:
        candidate_path = (
            Path(args.annotations_out)
            if args.annotations_out
            else output_path.with_suffix(".annotations.json")
        )
        destinations.append(candidate_path)
    snapshot_path = (
        Path(args.save_snapshot)
        if args.save_snapshot and snapshot is not None
        else None
    )
    if snapshot_path is not None:
        destinations.append(snapshot_path)
    _check_output_paths(destinations)

    report_view = report
    staged: list[tuple[Path, Path]] = []
    try:
        if session is not None:
            application = session._checked_report_data(  # noqa: SLF001 # Require the exact session-issued report before writing trusted outputs.
                report
            )
            report_view = application.base_report
            _stage_output(
                report_path,
                lambda stream: _write_session_stream(
                    stream,
                    session.iter_report_json(
                        report,
                        report_storage=args.report_storage,
                        max_output_bytes=args.max_report_bytes,
                    ),
                    output_name="Report JSON",
                    option_name="--max-report-bytes",
                    limit=args.max_report_bytes,
                ),
                staged,
            )
            if annotations is not None:
                _stage_output(
                    candidate_path,
                    lambda stream: _write_session_stream(
                        stream,
                        session.iter_annotations_json(
                            annotations,
                            report_storage=args.report_storage,
                            max_output_bytes=args.max_report_bytes,
                        ),
                        output_name="Annotation JSON",
                        option_name="--max-report-bytes",
                        limit=args.max_report_bytes,
                    ),
                    staged,
                )
            _stage_output(
                output_path,
                lambda stream: _write_session_stream(
                    stream,
                    iter_render_validated(
                        session,
                        report,
                        report_storage=args.report_storage,
                        max_output_bytes=args.max_html_bytes,
                    ),
                    output_name="HTML",
                    option_name="--max-html-bytes",
                    limit=args.max_html_bytes,
                ),
                staged,
            )
        else:
            if args.report_storage == "shared":
                msg = (
                    "--report-storage shared requires a complete v2 report "
                    "proof; use expanded storage for legacy or manual data"
                )
                raise CliError(msg)
            if type(report) is not dict:
                msg = "Expanded compatibility output requires a report object"
                raise CliError(msg)
            _stage_output(
                report_path,
                lambda stream: _write_compat_json(
                    stream,
                    report,
                    max_bytes=args.max_report_bytes,
                    value_name="Report",
                ),
                staged,
            )
            if annotations is not None:
                if type(annotations) is not dict:
                    msg = "Expanded annotation output requires annotation data"
                    raise CliError(msg)
                _stage_output(
                    candidate_path,
                    lambda stream: _write_compat_json(
                        stream,
                        annotations,
                        max_bytes=args.max_report_bytes,
                        value_name="Annotation",
                    ),
                    staged,
                )
            _stage_output(
                output_path,
                lambda stream: _write_compat_html(
                    stream,
                    report,
                    max_bytes=args.max_html_bytes,
                ),
                staged,
            )

        if snapshot_path is not None:
            _stage_output(
                snapshot_path,
                lambda stream: _write_json_value(
                    stream,
                    snapshot,
                    indent=2,
                ),
                staged,
            )

        for temporary_path, destination in staged:
            temporary_path.replace(destination)
    finally:
        for temporary_path, _destination in staged:
            with suppress(OSError):
                temporary_path.unlink(missing_ok=True)

    print(f"HTML: {output_path}\nData: {report_path}")
    if candidate_path is not None:
        print(f"Candidate annotations: {candidate_path}")
    _print_report_summary(report_view)


def _print_report_summary(report: dict) -> None:
    """Print counts and warning totals from a validated saved report."""
    if report.get("schema") == "diffstory.report.v2":
        packet = report["packet"]
        print(
            f"{len(report['story']['sections'])} reading sections · {len(packet['changes'])} change units · {len(packet['regions'])} changed regions"
        )
        if packet["warnings"]:
            print(
                f"{len(packet['warnings'])} scope/parse warning(s); see report evidence panel"
            )
        return
    stats = report["stats"]
    print(
        f"{stats['groups']} reading steps · {stats['units']} change units · "
        f"{stats['identical_ast_moves']} identical-AST moves"
    )
    if report.get("warnings"):
        print(
            f"{len(report['warnings'])} scope/parse warning(s); "
            "see report evidence panel"
        )


def _check_output_paths(destinations: Iterable[Path]) -> None:
    """Reject outputs that resolve to the same destination before staging."""
    normalized = [path.resolve(strict=False) for path in destinations]
    if len(normalized) != len(set(normalized)):
        msg = "HTML, report, annotation, and snapshot outputs must use different paths"
        raise CliError(msg)


def _stage_output(
    destination: Path,
    writer: Callable[[BinaryIO], None],
    staged: list[tuple[Path, Path]],
) -> None:
    """Write one output to a temporary sibling without touching its destination."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        delete=False,
    ) as stream:
        temporary_path = Path(stream.name)
        staged.append((temporary_path, destination))
        writer(stream)


def _write_json_value(
    stream: BinaryIO,
    value: object,
    *,
    indent: int | None = None,
    max_bytes: int | None = None,
    value_name: str = "JSON",
) -> int:
    """Stream compact or compatibility-pretty UTF-8 JSON into a stage file."""
    encoder = json.JSONEncoder(
        ensure_ascii=False,
        allow_nan=False,
        indent=indent,
        separators=None if indent is not None else (",", ":"),
    )
    total = 0
    for part in encoder.iterencode(value):
        chunk = part.encode("utf-8")
        if max_bytes is not None and total + len(chunk) > max_bytes:
            msg = (
                f"{value_name} JSON exceeds --max-report-bytes {max_bytes}; "
                "raise --max-report-bytes to save and reload this output"
            )
            raise CliError(msg)
        stream.write(chunk)
        total += len(chunk)
    return total


def _write_compat_json(
    stream: BinaryIO,
    value: dict,
    *,
    max_bytes: int,
    value_name: str,
) -> None:
    """Write compact compatibility JSON with the exact configured byte cap."""
    _write_json_value(
        stream,
        value,
        max_bytes=max_bytes,
        value_name=value_name,
    )


def _write_session_stream(
    stream: BinaryIO,
    chunks: Iterable[bytes],
    *,
    output_name: str,
    option_name: str,
    limit: int,
) -> None:
    """Write a session stream and translate only its closed size failures."""
    error_codes = {
        "Report JSON": "report_output_too_large",
        "Annotation JSON": "annotations_output_too_large",
        "HTML": "render_output_too_large",
    }
    try:
        for chunk in chunks:
            stream.write(chunk)
    except ValueError as error:
        if str(error) != error_codes[output_name]:
            raise
        msg = (
            f"{output_name} exceeds {option_name} {limit}; raise "
            f"{option_name} to save this complete output"
        )
        raise CliError(msg) from None


def _write_compat_html(
    stream: BinaryIO,
    report: dict,
    *,
    max_bytes: int,
) -> None:
    """Render compatibility data and check its complete UTF-8 HTML size."""
    html = render(report)
    output = html.encode("utf-8")
    if len(output) > max_bytes:
        msg = (
            f"HTML output exceeds --max-html-bytes {max_bytes}; raise "
            "--max-html-bytes to save the packaged page and report"
        )
        raise CliError(msg)
    stream.write(output)


def _import_shared_storage(
    raw: bytes,
    *,
    compiler_report: object | None,
    max_input_bytes: int,
) -> tuple[object, object]:
    """Independently prove exact shared bytes in a new report session.

    Args:
        raw: Exact bounded shared report or annotation bytes from disk.
        compiler_report: Matching expanded compiler report for annotation
            storage, or ``None`` for a report export with its own source base.
        max_input_bytes: Exact configured cap for the bytes passed to import.

    Returns:
        The newly created live session and its complete generation capability.

    Preconditions:
        ``raw`` came from the bounded input reader or the public shared codec.
    Postconditions:
        A complete independent proof exists only in the returned new session.
    Essential behavior:
        Sends the original bytes to ``ReportSession.import_indexed``. The CLI
        does not expand the envelope or trust parsed counts before proof.

    Raises:
        CliError: If an annotation export has no matching compiler report.
        ValueError: If storage structure, closure, or proof validation fails.
    """
    from .narrative_session import ReportSession

    try:
        return ReportSession.import_indexed(
            raw,
            compiler_report=compiler_report,
            max_input_bytes=max_input_bytes,
        )
    except ValueError as error:
        if str(error) == "compiler_report_required":
            msg = (
                "Shared annotation input requires its matching compiler report"
            )
            raise CliError(msg) from None
        if str(error) == "compiler_report_not_used_for_report":
            msg = "Shared report input already contains its compiler report"
            raise CliError(msg) from None
        raise


def _encode_shared_storage(
    value: dict,
    *,
    artifact_kind: str,
    max_input_bytes: int,
) -> bytes:
    """Encode expanded v2 compatibility input before a fresh shared import."""
    from .report_storage import encode_report_storage

    raw = encode_report_storage(value, artifact_kind)
    if len(raw) > max_input_bytes:
        msg = (
            f"Shared {artifact_kind} JSON exceeds --max-report-bytes "
            f"{max_input_bytes}; raise --max-report-bytes to import and save it"
        )
        raise CliError(msg)
    return raw


def _has_complete_v2_generation(report: object) -> bool:
    """Return whether a report visibly carries v2 generation and presentation."""
    return (
        type(report) is dict
        and type(report.get("generation")) is dict
        and report["generation"].get("version") == "diffstory.generation.v2"
        and type(report.get("narrative_v2")) is dict
    )


def _has_v2_annotations(annotations: object) -> bool:
    """Return whether candidate data uses the complete v2 annotations schema."""
    return (
        type(annotations) is dict
        and annotations.get("schema") == "diffstory.annotations.v2"
        and type(annotations.get("generation")) is dict
        and annotations["generation"].get("version")
        == "diffstory.generation.v2"
    )


def _write_evidence_packet(
    stream: BinaryIO,
    packet: object,
    *,
    max_bytes: int,
) -> None:
    """Write one immutable or mutable evidence packet under the JSON byte cap.

    Args:
        stream: Binary sibling stage file that receives the packet.
        packet: Primitive JSON data, including session-owned mappings and
            arrays from an indexed report.
        max_bytes: Exact positive byte limit for the encoded UTF-8 packet.

    Preconditions:
        ``packet`` contains only values supported by the report-storage
        primitive JSON iterator; ``max_bytes`` is positive.
    Postconditions:
        The complete canonical packet is written when it fits. A size failure
        leaves promotion to the caller and does not alter its destination.
    Essential behavior:
        Serializes owned mappings and arrays directly in bounded chunks. It
        does not recursively copy an immutable compiler report into a second
        mutable object graph.
    """
    from .report_storage import _iter_bounded_json_chunks
    from .report_storage import _iter_logical_record_json

    total = 0
    chunks = _iter_bounded_json_chunks(_iter_logical_record_json(packet))
    for chunk in chunks:
        if total + len(chunk) > max_bytes:
            msg = (
                f"Evidence JSON exceeds --max-report-bytes {max_bytes}; "
                "raise --max-report-bytes to save this complete packet"
            )
            raise CliError(msg)
        stream.write(chunk)
        total += len(chunk)


def _export_evidence(args: argparse.Namespace) -> int:
    """
    Write a compact evidence packet from an existing report JSON file.

    Args:
        args: Parsed evidence input and required output path.

    Returns:
        Zero after the evidence file is written.

    Raises:
        ValueError: If the input does not contain a Diffstory report.
        OSError: If input or output files cannot be accessed.

    """
    raw = _read_bounded_bytes(
        args.input,
        max_bytes=args.max_report_bytes,
    )
    evidence_path = Path(args.out)
    staged: list[tuple[Path, Path]] = []
    try:
        if _looks_like_shared_storage(raw):
            session, generation = _import_shared_storage(
                raw,
                compiler_report=None,
                max_input_bytes=args.max_report_bytes,
            )
            with session:
                report_handle = session.apply(generation)
                application = session._checked_report_data(  # noqa: SLF001 # Read the exact checked shared report base for its compiler-only evidence packet.
                    report_handle
                )
                packet = evidence_packet(application.base_report)
                _stage_output(
                    evidence_path,
                    lambda stream: _write_evidence_packet(
                        stream,
                        packet,
                        max_bytes=args.max_report_bytes,
                    ),
                    staged,
                )
                # The live session checks that the captured report remains
                # current after serialization and before the staged file is
                # promoted. No second import or proof is performed.
                session._checked_report_data(report_handle)  # noqa: SLF001
                for temporary_path, destination in staged:
                    temporary_path.replace(destination)
        else:
            report = _decode_json_object(raw)
            if report.get("schema") != SCHEMA:
                msg = "Evidence input must be report JSON"
                raise CliError(msg)
            packet = evidence_packet(report)
            _stage_output(
                evidence_path,
                lambda stream: _write_evidence_packet(
                    stream,
                    packet,
                    max_bytes=args.max_report_bytes,
                ),
                staged,
            )
            for temporary_path, destination in staged:
                temporary_path.replace(destination)
    finally:
        for temporary_path, _destination in staged:
            with suppress(OSError):
                temporary_path.unlink(missing_ok=True)
    print(f"Evidence written: {args.out}")
    return 0


def _validate_command_options(args: argparse.Namespace) -> None:
    """
    Reject incompatible narration modes before loading their inputs.

    Args:
        args: Parsed command-line options.

    Raises:
        ValueError: If repository narration and authored annotations are both
            requested for the same command.
    """
    uses_repository_source = args.command in {"git", "github", "snapshot"}
    if uses_repository_source and args.narrate and args.annotations:
        msg = "Use --narrate or --annotations, not both"
        raise CliError(msg)
    if (
        uses_repository_source
        and (args.state_dir is not None or args.fresh_state is not None)
        and (
            not args.narrate or args.architecture not in {"v2", "walkthrough"}
        )
    ):
        msg = "--state-dir and --fresh-state require --narrate --architecture v2 or walkthrough"
        raise CliError(msg)
    if (
        uses_repository_source
        and args.narrate
        and args.architecture == "legacy"
        and getattr(args, "report_storage", "expanded") == "shared"
    ):
        msg = (
            "--report-storage shared requires --architecture v2; legacy "
            "narration has no complete v2 report proof"
        )
        raise CliError(msg)


def _safe_error_message(error: Exception) -> str:  # noqa: C901, PLR0911 # Closed error mapping at the CLI boundary.
    """Translate failures using fixed reasons; never format arbitrary exceptions."""
    from .walkthrough import WalkthroughError
    from .walkthrough_runtime import RunLimitError

    if isinstance(error, (WalkthroughError, RunLimitError)):
        return str(error)
    if isinstance(error, CliError):
        return str(error)
    if isinstance(error, ArtifactRuntimeError):
        reason = (
            error.reason if error.reason in get_args(Reason) else "generic"
        )
        guidance = {
            "auth": "Check provider credentials and access; spent attempts remain recorded.",
            "prohibited_tools": "Provider tool activity was rejected; inspect provider configuration before retrying.",
            "results_unavailable": "Recovery budgets remain, but validated result payloads are unavailable. Restore the matching payload cache or explicitly start --fresh-state NEW_DIRECTORY.",
            "attempt_limit": "The lineage attempt budget is exhausted. Existing state is retained; --fresh-state NEW_DIRECTORY intentionally starts a new budget.",
            "missing_item": "Required results remain missing after bounded recovery. Spent attempts persist; no state is reset automatically.",
            "split_limit": "The split limit is exhausted; existing state is retained.",
            "state_corrupt": "Check recovery/cache integrity; no state was reset or replaced automatically.",
            "storage": "Check recovery/cache permissions and available disk space; no fallback directory was selected.",
            "dependencies_changed": "Approved input or provider configuration changed; rerun to preview and approve the current input.",
        }.get(
            reason,
            "Check the input and provider configuration; existing recovery state is retained.",
        )
        return f"Narration failed ({reason}). {guidance}"
    if isinstance(error, RuntimeStageError):
        if error.diagnostic is not None:
            safe = ArtifactRuntimeError(error.diagnostic.reason)
            return _safe_error_message(safe)
        # Legacy labels are generated locally. Do not accept arbitrary identifiers.
        stage = (
            error.stage
            if error.stage
            in {
                "chunks",
                "group-summaries",
                "global-summary",
                "story-order",
                "story-content",
            }
            else "stage"
        )
        task = (
            error.task_id
            if re.fullmatch(r"task-[0-9]+", error.task_id)
            else "task"
        )
        category = _legacy_category(error.category)
        return f"Stage {stage!r} failed for task {task!r} ({category}). Check provider setup and checkpoint state."
    if isinstance(error, RuntimeCallError):
        return f"Provider call failed ({_legacy_category(error.category)}). Check provider setup."
    if isinstance(error, CheckpointError):
        return "Checkpoint storage failed; check permissions and available disk space."
    if isinstance(error, json.JSONDecodeError):
        return "Invalid JSON (malformed_json); check the saved input."
    if isinstance(
        error,
        (ValidationError, TypeError, KeyError, ValueError, RecursionError),
    ):
        return "Input or narration validation failed (schema); check the supported format, types, coverage and citations."
    if isinstance(error, OSError):
        return "Input/output failed (storage); check file access, permissions and available disk space."
    if isinstance(error, subprocess.TimeoutExpired):
        return "Operation timed out (timeout); check provider responsiveness."
    return "Narration or input processing failed (generic); check configuration and retained recovery state."


def _legacy_category(value: str) -> str:
    """Keep known legacy failure categories without arbitrary diagnostic text."""
    return (
        value
        if value
        in {
            "authentication",
            "rate_limit",
            "transport",
            "timeout",
            "invalid_response",
            "provider_error",
            "validation_error",
            "checkpoint_error",
            "callback_error",
            "task_error",
            "cancelled",
        }
        else "task_error"
    )


def _load_cli_report(
    args: argparse.Namespace,
    lifetime: ExitStack,
) -> tuple[
    dict | None, object | None, object | None, object | None, object | None
]:
    """Load source or report input, independently importing shared bytes.

    Args:
        args: Parsed output command options.
        lifetime: Command-owned stack that closes imported sessions on exit.

    Returns:
        The optional source snapshot, expanded report or ``None``, and optional
        live session, generation capability, and applied report capability.

    Preconditions:
        Command option validation has completed.
    Postconditions:
        Shared storage, when present, was read as bounded bytes and passed
        directly to a new independent import session.
    Essential behavior:
        Keeps snapshot source limits separate from report storage limits.
    """
    snapshot = _load_snapshot(args)
    if snapshot is not None:
        return snapshot, compile_snapshot(snapshot), None, None, None

    report_bytes = _read_bounded_bytes(
        args.input,
        max_bytes=args.max_report_bytes,
    )
    if not _looks_like_shared_storage(report_bytes):
        return None, _decode_json_object(report_bytes), None, None, None

    session, generation = _import_shared_storage(
        report_bytes,
        compiler_report=None,
        max_input_bytes=args.max_report_bytes,
    )
    session = lifetime.enter_context(session)
    return None, None, session, generation, session.apply(generation)


def _apply_cli_annotations(
    args: argparse.Namespace,
    lifetime: ExitStack,
    report: object | None,
    current: tuple[object | None, object | None, object | None],
) -> tuple[object | None, object | None, object | None, object | None]:
    """Validate optional annotation bytes against their matching base report.

    Args:
        args: Parsed output command options.
        lifetime: Command-owned stack for any newly imported session.
        report: Current expanded report, or ``None`` when it has a handle.
        current: Current session, generation capability, and report
            capability, in that order.

    Returns:
        The updated report, session, generation capability, and report
        capability. Expanded manual compatibility output remains data-only.

    Preconditions:
        The command is not also generating narration, and its source report is
        either an expanded compiler object or a session-owned shared report.
    Postconditions:
        Shared annotations have an independently proved handle in a new scope.
        Expanded annotations either use the strict dictionary boundary or are
        encoded and independently imported for an explicit shared export.
    Essential behavior:
        No parsed storage envelope or caller-set flag grants report authority.
    """
    if not args.annotations:
        session, generation, report_handle = current
        return report, session, generation, report_handle

    session, generation, report_handle = current
    annotation_bytes = _read_bounded_bytes(
        args.annotations,
        max_bytes=args.max_report_bytes,
    )
    compiler_report = (
        session.captured_report()
        if session is not None and report_handle is not None
        else report
    )
    if _looks_like_shared_storage(annotation_bytes):
        session, generation = _import_shared_storage(
            annotation_bytes,
            compiler_report=compiler_report,
            max_input_bytes=args.max_report_bytes,
        )
        session = lifetime.enter_context(session)
        return None, session, generation, session.apply(generation)

    annotations = _decode_json_object(annotation_bytes)
    if annotations.get("schema") == "diffstory.annotations.walkthrough.v1":
        from .walkthrough import apply_walkthrough_annotations

        if (
            args.report_storage != "expanded"
            or type(compiler_report) is not dict
        ):
            msg = "Walkthrough annotations require a compiler report and flat/expanded output"
            raise CliError(msg)
        return (
            apply_walkthrough_annotations(compiler_report, annotations),
            None,
            None,
            None,
        )
    if args.report_storage == "shared":
        if not _has_v2_annotations(annotations):
            msg = (
                "--report-storage shared requires complete "
                "diffstory.annotations.v2 proof data; use expanded storage "
                "for manual or legacy annotations"
            )
            raise CliError(msg)
        if type(compiler_report) is not dict:
            msg = "Shared annotations require a matching compiler report"
            raise CliError(msg)
        raw = _encode_shared_storage(
            annotations,
            artifact_kind="annotations",
            max_input_bytes=args.max_report_bytes,
        )
        session, generation = _import_shared_storage(
            raw,
            compiler_report=compiler_report,
            max_input_bytes=args.max_report_bytes,
        )
        session = lifetime.enter_context(session)
        return None, session, generation, session.apply(generation)

    if type(compiler_report) is not dict:
        msg = "Annotations require a compiler report object"
        raise CliError(msg)
    return apply_annotations(compiler_report, annotations), None, None, None


def _ensure_shared_report_handle(
    args: argparse.Namespace,
    lifetime: ExitStack,
    report: object | None,
    current: tuple[object | None, object | None, object | None],
) -> tuple[object | None, object | None, object | None, object | None]:
    """Require a real v2 report capability before shared report serialization.

    Args:
        args: Parsed output command options.
        lifetime: Command-owned stack for a compatibility import session.
        report: Expanded input report, if no session handle exists.
        current: Current session, generation capability, and report
            capability, in that order.

    Returns:
        The report, session, generation, and report handle, with a new indexed
        handle when an expanded v2 report needed a shared compatibility import.

    Preconditions:
        The user selected ``--report-storage shared`` or an indexed report
        handle already exists.
    Postconditions:
        Manual, legacy, and compiler-only inputs cannot be relabeled as shared.
    Essential behavior:
        An expanded v2 report is encoded by the existing strict codec and then
        independently proved from the resulting storage bytes.
    """
    session, generation, report_handle = current
    if args.report_storage != "shared" or report_handle is not None:
        return report, session, generation, report_handle
    if not _has_complete_v2_generation(report):
        msg = (
            "--report-storage shared requires a complete v2 generation proof; "
            "use expanded storage for this input"
        )
        raise CliError(msg)

    raw = _encode_shared_storage(
        report,
        artifact_kind="report",
        max_input_bytes=args.max_report_bytes,
    )
    session, generation = _import_shared_storage(
        raw,
        compiler_report=None,
        max_input_bytes=args.max_report_bytes,
    )
    session = lifetime.enter_context(session)
    return None, session, generation, session.apply(generation)


def _run_output_command(
    args: argparse.Namespace,
    lifetime: ExitStack,
) -> None:
    """Run one report-producing command inside its session lifetime.

    Args:
        args: Parsed source, narration, and output options.
        lifetime: Stack that owns fresh and imported v2 sessions until every
            requested writer finishes.

    Preconditions:
        The command is a Git, GitHub, snapshot, or render output command.
    Postconditions:
        Requested report, annotation, snapshot, and HTML files have passed all
        validation and output caps before staged paths are promoted.
    Essential behavior:
        Fresh v2 narration holds one session through apply and trusted writers;
        legacy and manual dictionary paths retain independent public behavior.
    """
    snapshot, report, session, generation, report_handle = _load_cli_report(
        args,
        lifetime,
    )
    annotations_output = None
    if args.command in {"git", "github", "snapshot"} and args.narrate:
        if args.architecture == "walkthrough":
            annotations_output, report = _generate_walkthrough(args, report)
        elif args.architecture == "v2":
            session, generation, report_handle = _generate_narration_v2(
                args, report, lifetime
            )
            annotations_output = generation
            report = None
        else:
            annotations_output, report = _generate_narration(args, report)
            session = None
            generation = None
            report_handle = None
    else:
        report, session, generation, report_handle = _apply_cli_annotations(
            args,
            lifetime,
            report,
            (session, generation, report_handle),
        )

    report, session, generation, report_handle = _ensure_shared_report_handle(
        args,
        lifetime,
        report,
        (session, generation, report_handle),
    )
    _write_outputs(
        args,
        report_handle if report_handle is not None else report,
        snapshot,
        annotations_output,
        session=session,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """
    Run the requested Diffstory command and translate expected failures.

    Args:
        argv: Optional argument sequence; defaults to process command-line
            arguments.

    Returns:
        Zero on success, two when a handled input, provider, runtime, or I/O
        error prevents completion, or 130 when interrupted.

    Side Effects:
        May read source/report files, contact GitHub or a narration provider
        when explicitly requested, write outputs, and print status to standard
        output or errors to standard error.

    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        _validate_command_options(args)
        if args.command == "evidence":
            return _export_evidence(args)

        with ExitStack() as lifetime:
            _run_output_command(args, lifetime)
    except Exception as error:  # noqa: BLE001 # The CLI must never expose untrusted exception text.
        if (
            isinstance(error, ArtifactRuntimeError)
            and error.reason == "cancelled"
        ) or (
            isinstance(error, RuntimeStageError)
            and (
                error.category == "cancelled"
                or (
                    isinstance(error.diagnostic, SafeDiagnostic)
                    and error.diagnostic.reason == "cancelled"
                )
            )
        ):
            print("diffstory: interrupted", file=sys.stderr)
            return 130
        print(f"diffstory: {_safe_error_message(error)}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("diffstory: interrupted", file=sys.stderr)
        return 130
    else:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
