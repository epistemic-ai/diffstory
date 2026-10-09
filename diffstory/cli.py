"""Read-only CLI. Run from the extracted folder with python -m diffstory."""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import time
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
from .render import render

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence

MAX_INPUT_BYTES = 80_000_000
MAX_REPORT_BYTES = 256_000_000
PROGRESS_INTERVAL_SECONDS = 5.0


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
    parser.add_argument(
        "--architecture",
        choices=["legacy", "v2"],
        default="legacy",
        help="Narration architecture (default: legacy; v2 activation is gated)",
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
) -> None:
    """Print provider usage and validated checkpoint reuse for this run.

    Args:
        narrator: Narrator whose generation has completed or failed.
        annotations: Successfully validated candidate, if available.

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


def _generate_narration(
    args: argparse.Namespace,
    report: dict,
) -> tuple[dict, dict, Path]:
    """
    Generate validated narration annotations after preview and consent.

    Args:
        args: Parsed source and provider options.
        report: Deterministic report and source evidence to narrate.

    Returns:
        Candidate annotations, the report with those annotations applied, and
        the path where the candidate annotations should be saved.

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
    if args.annotations_out:
        candidate_path = Path(args.annotations_out)
    else:
        candidate_path = Path(args.out).with_suffix(".annotations.json")
    return annotations, annotated_report, candidate_path


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
    candidate_path: Path | None,
) -> None:
    """
    Render and write requested HTML, report, snapshot, and annotations.

    Args:
        args: Parsed output and optional snapshot paths.
        report: Report to render and serialize.
        snapshot: Source snapshot to save when requested, or ``None``.
        annotations: Candidate annotations to save, or ``None``.
        candidate_path: Annotation output path, or ``None``.

    Raises:
        OSError: If an output directory or file cannot be created or written.
        ValueError: If report validation or JSON rendering fails.

    Side Effects:
        Creates parent directories and writes requested output files; prints
        output paths, report counts, warnings, and usage to standard output.

    """
    report_json = _output_json(report, args.max_report_bytes)
    annotation_json = (
        _output_json(annotations, args.max_report_bytes)
        if annotations is not None
        else None
    )
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    html = render(report)

    if annotations is not None:
        candidate_path.parent.mkdir(parents=True, exist_ok=True)
        candidate_path.write_text(
            annotation_json,
            encoding="utf-8",
        )

    report_path = output_path.with_suffix(".report.json")
    output_path.write_text(html, encoding="utf-8")
    report_path.write_text(
        report_json,
        encoding="utf-8",
    )

    if args.save_snapshot and snapshot is not None:
        Path(args.save_snapshot).write_text(
            json.dumps(snapshot, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(f"HTML: {output_path}\nData: {report_path}")
    if candidate_path:
        print(f"Candidate annotations: {candidate_path}")
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


def _output_json(value: dict, max_bytes: int) -> str:
    """Bound persisted report JSON so the same limit supports its reload."""
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    size = len(text.encode("utf-8"))
    if size > max_bytes:
        msg = (
            f"Output JSON needs {size} bytes, above --max-report-bytes {max_bytes}; "
            f"use --max-report-bytes {size} for generation and reload if intended"
        )
        raise CliError(msg)
    return text


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
    report = read_json(args.input, max_bytes=args.max_report_bytes)
    if report.get("schema") != SCHEMA:
        msg = "Evidence input must be report JSON"
        raise CliError(msg)
    evidence_path = Path(args.out)
    evidence_path.write_text(
        json.dumps(evidence_packet(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
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
    uses_repository_source = args.command in {"git", "github"}
    if uses_repository_source and args.narrate and args.annotations:
        msg = "Use --narrate or --annotations, not both"
        raise CliError(msg)
    if (
        uses_repository_source
        and (args.state_dir is not None or args.fresh_state is not None)
        and (not args.narrate or args.architecture != "v2")
    ):
        msg = (
            "--state-dir and --fresh-state require --narrate --architecture v2"
        )
        raise CliError(msg)


def _safe_error_message(error: Exception) -> str:  # noqa: C901, PLR0911 # Closed error mapping at the CLI boundary.
    """Translate failures using fixed reasons; never format arbitrary exceptions."""
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

        snapshot = _load_snapshot(args)
        if snapshot is not None:
            report = compile_snapshot(snapshot)
        else:
            report = read_json(args.input, max_bytes=args.max_report_bytes)

        uses_repository_source = args.command in {"git", "github"}

        annotations = None
        candidate_path = None
        if uses_repository_source and args.narrate:
            annotations, report, candidate_path = _generate_narration(
                args, report
            )

        if args.annotations:
            authored_annotations = read_json(
                args.annotations, max_bytes=args.max_report_bytes
            )
            report = apply_annotations(report, authored_annotations)

        _write_outputs(args, report, snapshot, annotations, candidate_path)
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
