"""Read-only CLI. Run from the extracted folder with python -m diffstory."""
from __future__ import annotations
import argparse
import json
import sys
import subprocess
from pathlib import Path
from .analysis import (MAX_SNAPSHOT_SOURCE_BYTES, compile_snapshot, apply_annotations,
                       evidence_packet, SCHEMA)
from .budget import BudgetLimits
from .ingest import from_git, from_github
from .narrative import MODEL_CONTEXT_TOKENS, OPENAI_MODEL, Narrator, OpenAIResponsesProvider
from .render import render
from . import __version__


def read_json(path: str) -> dict:
    p = Path(path)
    if p.stat().st_size > 80_000_000: raise ValueError("Input JSON exceeds 80 MB")
    obj = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(obj, dict): raise ValueError("JSON must be an object")
    return obj


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="diffstory", description="Compile a semantic code walkthrough. Repository code is never executed.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    g = sub.add_parser("git", help="Analyze committed revisions from a local Git checkout")
    g.add_argument("--repo", default="."); g.add_argument("--base", required=True); g.add_argument("--head", default="HEAD")
    g.add_argument("--two-dot", action="store_true", help="Compare endpoints instead of merge-base to head")
    g.add_argument("--max-files", type=int, default=500)
    gh_help = "Read a GitHub PR (GITHUB_TOKEN for private repositories)"
    gh = sub.add_parser("github", help=gh_help)
    gh.add_argument("pr", help="owner/repo#123 or GitHub PR URL"); gh.add_argument("--token-env", default="GITHUB_TOKEN"); gh.add_argument("--max-files", type=int, default=500)
    for p in (g, gh):
        p.add_argument("--max-source-bytes", type=int, default=MAX_SNAPSHOT_SOURCE_BYTES,
                       help="Aggregate source limit across both revisions (default: 64000000; may be lowered)")
        p.add_argument("--narrate", action="store_true", help="Generate model-written source explanations (sends PR source to OpenAI)")
        p.add_argument("--provider", choices=["openai"], default="openai")
        p.add_argument("--model", choices=[OPENAI_MODEL], default=OPENAI_MODEL)
        p.add_argument("--api-key-env", default="OPENAI_API_KEY", help="Environment variable holding the OpenAI API key")
        p.add_argument("--yes", action="store_true", help="Confirm source transfer without an interactive prompt")
        p.add_argument("--max-request-input-tokens", type=int, default=40_000)
        p.add_argument("--max-request-output-tokens", type=int, default=4_000)
        p.add_argument("--max-input-tokens", type=int, default=256_000, help="Run-wide reserved input-token limit")
        p.add_argument("--max-output-tokens", type=int, default=40_000, help="Run-wide reserved output-token limit")
        p.add_argument("--max-provider-calls", type=int, default=64)
        p.add_argument("--narration-timeout", type=float, default=900,
                       help="Run-wide narration deadline in seconds")
        p.add_argument("--annotations-out", help="Candidate annotations path (default: <out>.annotations.json)")
    sn = sub.add_parser("snapshot", help="Compile a saved diffstory.snapshot.v1 JSON document")
    sn.add_argument("input")
    rr = sub.add_parser("render", help="Render an existing report, optionally with authored/model annotations")
    rr.add_argument("input")
    ex = sub.add_parser("evidence", help="Export evidence for a human or model; makes no network call")
    ex.add_argument("input"); ex.add_argument("--out", required=True)
    for p in (g, gh, sn, rr):
        p.add_argument("--out", required=True, help="Output .html file")
        p.add_argument("--annotations", help="Optional revision-bound narrative JSON")
        p.add_argument("--save-snapshot", help="Save reusable input source JSON")
    args = parser.parse_args(argv)
    try:
        if args.command == "evidence":
            report = read_json(args.input)
            if report.get("schema") != SCHEMA: raise ValueError("Evidence input must be report JSON")
            Path(args.out).write_text(json.dumps(evidence_packet(report), ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Evidence written: {args.out}"); return 0
        snapshot = None
        if args.command == "git": snapshot = from_git(args.repo, args.base, args.head, two_dot=args.two_dot,
                                                       max_files=args.max_files, max_source_bytes=args.max_source_bytes)
        elif args.command == "github": snapshot = from_github(args.pr, token_env=args.token_env,
                                                               max_files=args.max_files, max_source_bytes=args.max_source_bytes)
        elif args.command == "snapshot": snapshot = read_json(args.input)
        report = compile_snapshot(snapshot) if snapshot is not None else read_json(args.input)
        if args.command in {"git", "github"} and args.narrate and args.annotations:
            raise ValueError("Use --narrate or --annotations, not both")
        annotations = None
        candidate_path = None
        if args.command in {"git", "github"} and args.narrate:
            provider = OpenAIResponsesProvider(token_env=args.api_key_env)
            provider.require_credentials()
            limits = BudgetLimits(
                context_tokens=MODEL_CONTEXT_TOKENS,
                request_input_tokens=args.max_request_input_tokens,
                request_output_tokens=args.max_request_output_tokens,
                total_input_tokens=args.max_input_tokens,
                total_output_tokens=args.max_output_tokens,
                calls=args.max_provider_calls,
                seconds=args.narration_timeout,
            )
            narrator = Narrator(provider, limits=limits)
            preview = narrator.preview(report)
            print("Narration source-transfer preview")
            print(f"Provider: {provider.name} · model: {provider.model}")
            print(f"Destination: {provider.destination}")
            print(f"Source scope: {report['meta'].get('scope', 'supplied snapshot')} · "
                  f"{report['meta'].get('changed_files', 0)} changed files · "
                  f"{report['meta'].get('source_bytes', 0)} source bytes · "
                  f"{report['meta'].get('base_sha', '')[:12]} → {report['meta'].get('head_sha', '')[:12]}")
            print(f"Evidence chunks: {preview['chunks']} · planned calls: {preview['calls']} · "
                  f"reserved input: {preview['reserved_input_tokens']} tokens · "
                  f"reserved output: {preview['reserved_output_tokens']} tokens")
            print(f"Run ceilings: {limits.total_input_tokens} input tokens · {limits.total_output_tokens} output tokens · "
                  f"{limits.calls} calls · {limits.seconds:g} seconds")
            if not args.yes:
                if not sys.stdin.isatty():
                    raise ValueError("Source was not sent. Rerun with --yes only after approving the displayed transfer")
                try:
                    answer = input("Send source evidence to OpenAI and generate narration? [y/N] ").strip().lower()
                except EOFError:
                    raise ValueError("Source was not sent because confirmation was not available") from None
                if answer not in {"y", "yes"}:
                    raise ValueError("Narration cancelled; no model request was made")
            annotations = narrator.generate(report)
            report = apply_annotations(report, annotations)
            candidate_path = Path(args.annotations_out) if args.annotations_out else Path(args.out).with_suffix(".annotations.json")
        if args.annotations: report = apply_annotations(report, read_json(args.annotations))
        out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
        html = render(report)
        if annotations is not None:
            candidate_path.parent.mkdir(parents=True, exist_ok=True)
            candidate_path.write_text(json.dumps(annotations, ensure_ascii=False, indent=2), encoding="utf-8")
        out.write_text(html, encoding="utf-8")
        out.with_suffix(".report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        if args.save_snapshot and snapshot is not None:
            Path(args.save_snapshot).write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"HTML: {out}\nData: {out.with_suffix('.report.json')}")
        if candidate_path: print(f"Candidate annotations: {candidate_path}")
        print(f"{report['stats']['groups']} reading steps · {report['stats']['units']} change units · {report['stats']['identical_ast_moves']} identical-AST moves")
        if report.get("warnings"): print(f"{len(report['warnings'])} scope/parse warning(s); see report evidence panel")
        return 0
    except (ValueError, KeyError, TypeError, OSError, RecursionError, subprocess.TimeoutExpired) as e:
        print(f"diffstory: {e}", file=sys.stderr); return 2

if __name__ == "__main__": raise SystemExit(main())
