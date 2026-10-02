"""Read-only CLI. Run from the extracted folder with python -m diffstory."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from . import __version__
from .analysis import MAX_SNAPSHOT_SOURCE_BYTES
from .analysis import SCHEMA
from .analysis import apply_annotations
from .analysis import compile_snapshot
from .analysis import evidence_packet
from .ingest import from_git
from .ingest import from_github
from .narrative import OPENAI_MODEL
from .narrative import CodexCLIProvider
from .narrative import NarrativeProvider
from .narrative import Narrator
from .narrative import OpenAIResponsesProvider
from .render import render

if TYPE_CHECKING:
    from collections.abc import Sequence

MAX_INPUT_BYTES = 80_000_000


def read_json(path: str) -> dict:
    """
    Read one bounded UTF-8 JSON object from disk.

    Args:
        path: Input file path.

    Returns:
        The decoded JSON object.

    Raises:
        OSError: If the file cannot be read.
        ValueError: If the file is too large or malformed.
        TypeError: If the JSON value is not an object.
        UnicodeError: If the file is not valid UTF-8.

    """
    input_path = Path(path)
    if input_path.stat().st_size > MAX_INPUT_BYTES:
        msg = "Input JSON exceeds 80 MB"
        raise ValueError(msg)

    value = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        msg = "JSON must be an object"
        raise TypeError(msg)
    return value


def _add_narration_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Add source-limit and opt-in narration flags to a source subcommand.

    Args:
        parser: Argument parser for a Git or GitHub source command.

    Side Effects:
        Adds narration, provider, model, confirmation, and annotation options
        directly to ``parser``.

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
        raise ValueError(msg)

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
        raise ValueError(msg) from None
    if answer not in {"y", "yes"}:
        msg = "Narration cancelled; no model request was made"
        raise ValueError(msg)


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
        Exception: Re-raises provider/generation failures after printing
            accumulated usage. Provider-specific errors are intentionally not
            hidden here.

    Side Effects:
        Prints a local transfer preview and, after consent, may send source
        evidence to the selected provider. Does not write output files.

    """
    if args.provider == "openai":
        if args.model not in {None, OPENAI_MODEL}:
            msg = f"OpenAI narration currently supports only {OPENAI_MODEL}"
            raise ValueError(msg)
        provider: NarrativeProvider = OpenAIResponsesProvider(
            token_env=args.api_key_env
        )
    else:
        provider = CodexCLIProvider(model=args.model)
    provider.require_credentials()

    narrator = Narrator(provider)
    preview = narrator.preview(report)
    _print_narration_preview(provider, report, preview)
    _confirm_source_transfer(assume_yes=args.yes, provider=provider)

    try:
        annotations = narrator.generate(report)
        annotated_report = apply_annotations(report, annotations)
    except Exception:
        _print_token_usage(narrator.usage.report())
        raise
    if args.annotations_out:
        candidate_path = Path(args.annotations_out)
    else:
        candidate_path = Path(args.out).with_suffix(".annotations.json")
    return annotations, annotated_report, candidate_path


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
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    html = render(report)

    if annotations is not None:
        candidate_path.parent.mkdir(parents=True, exist_ok=True)
        candidate_path.write_text(
            json.dumps(annotations, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    report_path = output_path.with_suffix(".report.json")
    output_path.write_text(html, encoding="utf-8")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
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
    if annotations is not None:
        _print_token_usage(annotations["generation"]["usage"])


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
    report = read_json(args.input)
    if report.get("schema") != SCHEMA:
        msg = "Evidence input must be report JSON"
        raise ValueError(msg)
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
        raise ValueError(msg)


def main(argv: Sequence[str] | None = None) -> int:
    """
    Run the requested Diffstory command and translate expected failures.

    Args:
        argv: Optional argument sequence; defaults to process command-line
            arguments.

    Returns:
        Zero on success or two when a handled input, provider, or I/O error
        prevents completion.

    Side Effects:
        May read source/report files, contact GitHub or a narration provider
        when explicitly requested, write outputs, and print status to standard
        output or errors to standard error.

    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "evidence":
            return _export_evidence(args)

        snapshot = _load_snapshot(args)
        if snapshot is not None:
            report = compile_snapshot(snapshot)
        else:
            report = read_json(args.input)

        _validate_command_options(args)
        uses_repository_source = args.command in {"git", "github"}

        annotations = None
        candidate_path = None
        if uses_repository_source and args.narrate:
            annotations, report, candidate_path = _generate_narration(
                args, report
            )

        if args.annotations:
            authored_annotations = read_json(args.annotations)
            report = apply_annotations(report, authored_annotations)

        _write_outputs(args, report, snapshot, annotations, candidate_path)
    except (
        ValueError,
        KeyError,
        TypeError,
        OSError,
        RecursionError,
        subprocess.TimeoutExpired,
    ) as error:
        print(f"diffstory: {error}", file=sys.stderr)
        return 2
    else:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
