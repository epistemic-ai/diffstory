"""Read-only CLI. Run from the extracted folder with python -m diffstory."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from . import __version__
from .analysis import (
    MAX_SNAPSHOT_SOURCE_BYTES,
    SCHEMA,
    apply_annotations,
    compile_snapshot,
    evidence_packet,
)
from .ingest import from_git, from_github
from .narrative import (
    OPENAI_MODEL,
    CodexCLIProvider,
    NarrativeProvider,
    Narrator,
    OpenAIResponsesProvider,
)
from .render import render


def read_json(path: str) -> dict:
    input_path = Path(path)
    if input_path.stat().st_size > 80_000_000:
        raise ValueError("Input JSON exceeds 80 MB")

    value = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON must be an object")
    return value


def _add_narration_arguments(parser: argparse.ArgumentParser) -> None:
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
        help="Optional model override (OpenAI: gpt-6-astra; Codex: CLI model ID)",
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

    for command_parser in (git_parser, github_parser, snapshot_parser, render_parser):
        command_parser.add_argument("--out", required=True, help="Output .html file")
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
    assume_yes: bool,
    provider: NarrativeProvider,
) -> None:
    if assume_yes:
        return
    if not sys.stdin.isatty():
        raise ValueError(
            "Source was not sent. Rerun with --yes only after approving the displayed transfer"
        )

    try:
        provider_name = getattr(provider, "consent_name", provider.name)
        answer = input(
            f"Send source evidence to {provider_name} and generate narration? [y/N] "
        ).strip().lower()
    except EOFError:
        raise ValueError(
            "Source was not sent because confirmation was not available"
        ) from None
    if answer not in {"y", "yes"}:
        raise ValueError("Narration cancelled; no model request was made")


def _generate_narration(
    args: argparse.Namespace,
    report: dict,
) -> tuple[dict, dict, Path]:
    if args.provider == "openai":
        if args.model not in {None, OPENAI_MODEL}:
            raise ValueError(f"OpenAI narration currently supports only {OPENAI_MODEL}")
        provider: NarrativeProvider = OpenAIResponsesProvider(
            token_env=args.api_key_env
        )
    else:
        provider = CodexCLIProvider(model=args.model)
    provider.require_credentials()

    narrator = Narrator(provider)
    preview = narrator.preview(report)
    _print_narration_preview(provider, report, preview)
    _confirm_source_transfer(args.yes, provider)

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
    report = read_json(args.input)
    if report.get("schema") != SCHEMA:
        raise ValueError("Evidence input must be report JSON")
    evidence_path = Path(args.out)
    evidence_path.write_text(
        json.dumps(evidence_packet(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Evidence written: {args.out}")
    return 0


def main(argv=None) -> int:
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

        uses_repository_source = args.command in {"git", "github"}
        if uses_repository_source and args.narrate and args.annotations:
            raise ValueError("Use --narrate or --annotations, not both")

        annotations = None
        candidate_path = None
        if uses_repository_source and args.narrate:
            annotations, report, candidate_path = _generate_narration(args, report)

        if args.annotations:
            authored_annotations = read_json(args.annotations)
            report = apply_annotations(report, authored_annotations)

        _write_outputs(args, report, snapshot, annotations, candidate_path)
        return 0
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


if __name__ == "__main__":
    raise SystemExit(main())
