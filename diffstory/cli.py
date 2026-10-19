"""Read-only CLI. Run from the extracted folder with python -m diffstory."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING

from . import __version__
from .analysis import SCHEMA
from .analysis import apply_annotations
from .analysis import compile_snapshot
from .analysis import evidence_packet
from .ingest import from_git
from .ingest import from_github
from .models import MAX_SNAPSHOT_SOURCE_BYTES
from .providers import CODEX_MODEL_CAPACITIES
from .providers import OPENAI_MODEL
from .providers import CodexCLIProvider
from .providers import NarrativeProvider
from .providers import OpenAIResponsesProvider
from .render import _render_prevalidated
from .render import validate_report

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterable
    from collections.abc import Sequence
    from typing import BinaryIO

MAX_INPUT_BYTES = 80_000_000
MAX_REPORT_BYTES = 256_000_000
MAX_HTML_BYTES = 256_000_000


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
    state_group = parser.add_mutually_exclusive_group()
    state_group.add_argument(
        "--state-dir",
        type=Path,
        help="Durable attempt budget directory (default: <out>.narration-state)",
    )
    state_group.add_argument(
        "--fresh-state",
        type=Path,
        metavar="NEW_DIRECTORY",
        help=(
            "Intentionally start new attempt budgets in a new directory; "
            "keeps old state and may reuse validated cached results"
        ),
    )
    cache_group = parser.add_mutually_exclusive_group()
    cache_group.add_argument(
        "--cache-dir",
        type=Path,
        help=(
            "Directory for persistent walkthrough results (default: output "
            "path with .narration-cache suffix)"
        ),
    )
    cache_group.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable cached results; attempt budgets remain durable",
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
        help="Render an existing report, optionally with annotations",
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
        A flat annotation mapping and the complete flat saved report. Their
        story and generation values are the same objects as in the validated
        report.

    Raises:
        ValueError: If consent, capacity, or output validation fails.
        TimeoutError: If narration exceeds its four-minute deadline.

    Preconditions:
        ``report`` is a compiler report that passed the loader's validation;
        the caller has authorized source transfer or will confirm it here.
    Postconditions:
        The returned report passed the narrator's complete validation pass.
        The returned annotation contains no source packet or report history.
    """
    from .walkthrough import WalkthroughNarrator

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
    return {
        "schema": "diffstory.annotations.walkthrough.v1",
        "story": generated["story"],
        "generation": generated["generation"],
    }, generated


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


def _write_outputs(
    args: argparse.Namespace,
    report: dict,
    snapshot: dict | None,
    annotations: dict | None,
) -> None:
    """Stage bounded outputs and promote them after every writer completes.

    Args:
        args: Parsed output paths and report/HTML byte limits.
        report: Compiler or flat report for the HTML and adjacent JSON file.
        snapshot: Optional source snapshot to save beside the report.
        annotations: Optional flat generated annotation export.

    Returns:
        ``None`` after all staged files have been promoted.

    Raises:
        ValueError: If output paths conflict or a configured byte cap is
            exceeded.
        OSError: If a stage file cannot be written or promoted.

    Preconditions:
        ``report`` has passed the applicable validation boundary: either
        ``validate_report`` in this CLI call path or the narrator's final
        ``validate_walkthrough`` pass. Any supplied snapshot and annotation
        mapping come from the same source operation.
    Postconditions:
        Report, annotation, HTML, and requested snapshot files are promoted
        only after every writer completes. Failed staging leaves no partial
        newly written output.
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

    staged: list[tuple[Path, Path]] = []
    try:
        _stage_output(
            report_path,
            lambda stream: _write_json_value(
                stream,
                report,
                max_bytes=args.max_report_bytes,
                value_name="Report",
            ),
            staged,
        )
        if annotations is not None:
            _stage_output(
                candidate_path,
                lambda stream: _write_json_value(
                    stream,
                    annotations,
                    max_bytes=args.max_report_bytes,
                    value_name="Annotation",
                ),
                staged,
            )
        _stage_output(
            output_path,
            lambda stream: _write_html(
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
    _print_report_summary(report)


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


def _write_html(
    stream: BinaryIO,
    report: dict,
    *,
    max_bytes: int,
) -> None:
    """Render a just-validated report and check its complete UTF-8 HTML size.

    Args:
        stream: Staged binary output stream.
        report: Report validated earlier in this synchronous CLI operation.
        max_bytes: Maximum complete HTML output size in encoded UTF-8 bytes.

    Returns:
        ``None`` after writing the rendered page.

    Raises:
        CliError: If the complete rendered HTML exceeds ``max_bytes``.
        OSError: If packaged assets cannot be read or the stream cannot write.
        TypeError: If report data cannot be serialized as JSON.

    Preconditions:
        This exact report passed ``validate_report`` or was returned by the
        immediately preceding ``generate_prepared`` call, and has not changed.
    Postconditions:
        The complete page is written only when its UTF-8 byte size is within
        the configured cap.
    """
    html = _render_prevalidated(report)
    output = html.encode("utf-8")
    if len(output) > max_bytes:
        msg = (
            f"HTML output exceeds --max-html-bytes {max_bytes}; raise "
            "--max-html-bytes to save the packaged page and report"
        )
        raise CliError(msg)
    stream.write(output)


def _export_evidence(args: argparse.Namespace) -> int:
    """Write a bounded source packet from a compiler or current flat report.

    Args:
        args: Parsed input path, output path, and report byte limit.

    Returns:
        Zero after the evidence packet has been staged and promoted.

    Raises:
        ValueError: If the input report or packet fails validation or size
            limits.
        OSError: If input, staging, or promotion fails.

    Preconditions:
        The report is untrusted input. This function validates it before
        exporting any source packet.
    Postconditions:
        The output contains only the validated source packet, subject to the
        configured JSON byte limit. No provider call is made.
    """
    raw = _read_bounded_bytes(args.input, max_bytes=args.max_report_bytes)
    report = _decode_json_object(raw)
    validate_report(report)
    packet = (
        report["packet"]
        if report.get("schema") == "diffstory.report.v2"
        else evidence_packet(report)
    )
    evidence_path = Path(args.out)
    staged: list[tuple[Path, Path]] = []
    try:
        _stage_output(
            evidence_path,
            lambda stream: _write_json_value(
                stream,
                packet,
                max_bytes=args.max_report_bytes,
                value_name="Evidence",
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
    """Reject conflicting source-generation and annotation options."""
    uses_repository_source = args.command in {"git", "github", "snapshot"}
    if uses_repository_source and args.narrate and args.annotations:
        msg = "Use --narrate or --annotations, not both"
        raise CliError(msg)
    if (
        uses_repository_source
        and not args.narrate
        and (args.state_dir is not None or args.fresh_state is not None)
    ):
        msg = "--state-dir and --fresh-state require --narrate"
        raise CliError(msg)


def _safe_error_message(error: Exception) -> str:
    """Translate failures with fixed CLI text and safe current-run limits."""
    from .walkthrough import WalkthroughError
    from .walkthrough_runtime import RunLimitError

    if isinstance(error, (CliError, WalkthroughError, RunLimitError)):
        return str(error)
    if isinstance(error, json.JSONDecodeError):
        return "Invalid JSON; check the saved input."
    if isinstance(error, subprocess.TimeoutExpired):
        return "Operation timed out; check GitHub or provider responsiveness."
    if isinstance(error, OSError):
        return "Input/output failed; check file access, permissions and available disk space."
    if isinstance(error, (TypeError, KeyError, ValueError, RecursionError)):
        return "Input or narration validation failed; check the supported format, types, coverage and citations."
    return "Narration or input processing failed; check the input and provider configuration."


def _load_cli_report(args: argparse.Namespace) -> tuple[dict | None, dict]:
    """Load compiler source or one bounded, validated report artifact.

    Args:
        args: Parsed source or report input and byte limit.

    Returns:
        ``(snapshot, report)`` for source commands, or ``(None, report)`` for
        a saved report. The returned report has passed ``validate_report``.

    Raises:
        ValueError: If input decoding, size, or report validation fails.

    Postconditions:
        No loaded report is returned before the independent report boundary
        accepts it. A compiled snapshot also passes that boundary here.
    """
    snapshot = _load_snapshot(args)
    if snapshot is not None:
        report = compile_snapshot(snapshot)
        validate_report(report)
        return snapshot, report

    report_bytes = _read_bounded_bytes(
        args.input,
        max_bytes=args.max_report_bytes,
    )
    report = _decode_json_object(report_bytes)
    validate_report(report)
    return None, report


def _apply_cli_annotations(args: argparse.Namespace, report: dict) -> dict:
    """Apply annotations and validate the resulting compiler report.

    Args:
        args: Parsed optional annotation path and report byte limit.
        report: Validated compiler report or current flat report.

    Returns:
        The unchanged input when no annotation file is configured; otherwise,
        a newly assembled report that passed its applicable validation.

    Raises:
        ValueError: If annotation decoding, source binding, or validation
            fails.

    Preconditions:
        ``report`` passed ``validate_report`` earlier in this CLI call path.
    Postconditions:
        A manually annotated compiler report passes ``validate_report``.
        Walkthrough annotation application returns its fully validated flat
        report. No input report is mutated.
    """
    if not args.annotations:
        return report
    annotation_bytes = _read_bounded_bytes(
        args.annotations,
        max_bytes=args.max_report_bytes,
    )
    annotations = _decode_json_object(annotation_bytes)
    if annotations.get("schema") == "diffstory.annotations.walkthrough.v1":
        from .walkthrough import apply_walkthrough_annotations

        if report.get("schema") != SCHEMA:
            msg = "Walkthrough annotations require a compiler report"
            raise CliError(msg)
        return apply_walkthrough_annotations(report, annotations)
    result = apply_annotations(report, annotations)
    validate_report(result)
    return result


def _run_output_command(args: argparse.Namespace) -> None:
    """Run one source or render command and stage all requested outputs.

    Args:
        args: Parsed command, input, annotations, consent, and output options.

    Returns:
        ``None`` after output promotion.

    Raises:
        ValueError: If source, report, annotation, provider, or output checks
            fail.
        TimeoutError: If a bounded narration run exceeds its deadline.

    Postconditions:
        HTML uses the private renderer only after this call path has validated
        the exact report, or after generation returned its validated report.
        Public ``render`` continues to validate each independent library call.
    """
    snapshot, report = _load_cli_report(args)
    annotations_output = None
    if args.command in {"git", "github", "snapshot"} and args.narrate:
        annotations_output, report = _generate_walkthrough(args, report)
    else:
        report = _apply_cli_annotations(args, report)
    _write_outputs(args, report, snapshot, annotations_output)


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
        _run_output_command(args)
    except KeyboardInterrupt:
        print("diffstory: interrupted", file=sys.stderr)
        return 130
    except Exception as error:  # noqa: BLE001 # The CLI must never expose untrusted exception text.
        print(f"diffstory: {_safe_error_message(error)}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
