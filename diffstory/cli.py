"""Read-only CLI. Run from the extracted folder with python -m diffstory."""
from __future__ import annotations
import argparse
import json
import sys
import subprocess
from pathlib import Path
from .analysis import compile_snapshot, apply_annotations, evidence_packet, SCHEMA
from .ingest import from_git, from_github
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
    gh = sub.add_parser("github", help="Read a GitHub PR (GITHUB_TOKEN for private repositories)")
    gh.add_argument("pr", help="owner/repo#123 or GitHub PR URL"); gh.add_argument("--token-env", default="GITHUB_TOKEN"); gh.add_argument("--max-files", type=int, default=500)
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
        if args.command == "git": snapshot = from_git(args.repo, args.base, args.head, two_dot=args.two_dot, max_files=args.max_files)
        elif args.command == "github": snapshot = from_github(args.pr, token_env=args.token_env, max_files=args.max_files)
        elif args.command == "snapshot": snapshot = read_json(args.input)
        report = compile_snapshot(snapshot) if snapshot is not None else read_json(args.input)
        if args.annotations: report = apply_annotations(report, read_json(args.annotations))
        out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(render(report), encoding="utf-8")
        out.with_suffix(".report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        if args.save_snapshot and snapshot is not None:
            Path(args.save_snapshot).write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"HTML: {out}\nData: {out.with_suffix('.report.json')}")
        print(f"{report['stats']['groups']} reading steps · {report['stats']['units']} change units · {report['stats']['identical_ast_moves']} identical-AST moves")
        if report.get("warnings"): print(f"{len(report['warnings'])} scope/parse warning(s); see report evidence panel")
        return 0
    except (ValueError, KeyError, TypeError, OSError, RecursionError, subprocess.TimeoutExpired) as e:
        print(f"diffstory: {e}", file=sys.stderr); return 2

if __name__ == "__main__": raise SystemExit(main())
