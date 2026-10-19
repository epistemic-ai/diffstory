"""Measure the flat walkthrough with an offline provider unless --live is set.

Examples:
    python scripts/benchmark_walkthrough.py --fixture demo
    python scripts/benchmark_walkthrough.py --fixture complex --snapshot /private/input.json
    python scripts/benchmark_walkthrough.py --fixture synthetic

Live generation additionally requires --live --yes. Private source and generated
artifacts stay in the selected output directory, ignored by default. Metrics
contain fixed labels, hashes, revisions, counts, timings, and sizes, never source.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path

STARTED = time.monotonic()
ROOT = Path(__file__).resolve().parents[1]
MAX_SYNTHETIC_FILES = 128
MAX_SYNTHETIC_LINES = 4_000
sys.path.insert(0, str(ROOT))

from diffstory.analysis import compile_snapshot  # noqa: E402
from diffstory.compact_packet import canonical_bytes  # noqa: E402
from diffstory.compact_packet import digest  # noqa: E402
from diffstory.render import render  # noqa: E402
from diffstory.walkthrough import WalkthroughError  # noqa: E402
from diffstory.walkthrough import WalkthroughNarrator  # noqa: E402
from diffstory.walkthrough_runtime import RunLimitError  # noqa: E402


def synthetic_snapshot(files: int = 32, lines: int = 1_200) -> dict:
    """Build a reproducible changeset larger than the private 1.7 MB fixture.

    Args:
        files: Number of complete changed modules, from 1 through 128.
        lines: Assignments per function, from 1 through 4,000.

    Returns:
        A pinned synthetic snapshot with complete before/after definitions.

    Raises:
        ValueError: If counts exceed the bounded local fixture contract.
    """
    if (
        not 1 <= files <= MAX_SYNTHETIC_FILES
        or not 1 <= lines <= MAX_SYNTHETIC_LINES
    ):
        msg = "Synthetic counts exceed the fixture bounds"
        raise ValueError(msg)
    before = (
        "def process(value):\n"
        + "".join(
            f"    item_{i} = value + {i}  # supplied context\n"
            for i in range(lines)
        )
        + f"    return item_{lines - 1}\n"
    )
    after = before.replace(
        f"return item_{lines - 1}", f"return item_{lines - 1} + 1"
    )
    paths = [f"subsystem_{i:03d}/process.py" for i in range(files)]
    return {
        "schema": "diffstory.snapshot.v1",
        "meta": {
            "base_sha": hashlib.sha256(before.encode()).hexdigest(),
            "head_sha": hashlib.sha256(after.encode()).hexdigest(),
            "repository": "synthetic/benchmark",
            "title": "Synthetic mechanism change",
            "scope": "changed files",
            "input": "synthetic example",
        },
        "fragments": [
            {"side": side, "path": path, "text": text}
            for side, text in (("base", before), ("head", after))
            for path in paths
        ],
        "file_evidence": [
            {
                side: {"path": path, "state": "supplied", "coverage": "full"}
                for side in ("base", "head")
            }
            for path in paths
        ],
        "patches": [
            {
                "path": path,
                "origin": "git",
                "patch": (
                    f"@@ -{lines + 2} +{lines + 2} @@\n"
                    f"-    return item_{lines - 1}\n"
                    f"+    return item_{lines - 1} + 1\n"
                ),
            }
            for path in paths
        ],
    }


class OfflineProvider:
    """Generate deterministic structural prose without network or subprocesses.

    Attributes:
        name: Synthetic destination, never a supported live provider.
        model: Explicit structural-fixture identity.
        context_tokens: Capacity used by the configured Codex model in this repo.
        max_output_tokens: Structural fixture output bound.
        input_overhead_bytes: No external framing is added.
    """

    name = "offline-fixture"
    model = "structural-walkthrough-v1"
    destination = "fixture://local-only"
    context_tokens = 1_050_000
    max_output_tokens = 128_000
    input_overhead_bytes = 8_192

    def prepare(
        self,
        system: str,
        data: dict,
        schema_name: str,
        schema: dict,
        max_output_tokens: int,
    ) -> bytes:
        """Serialize a complete synthetic request, including schema and framing."""
        return canonical_bytes(
            {
                "system": system,
                "data": data,
                "schema_name": schema_name,
                "schema": schema,
                "max_output_tokens": max_output_tokens,
            }
        )

    def complete(
        self, body: bytes, timeout: float | None
    ) -> tuple[dict, None]:
        """Return source-linked fixture prose without claiming semantic quality."""
        if timeout is not None and timeout <= 0:
            raise TimeoutError
        request = json.loads(body)
        packet = request["data"]
        if request["schema_name"] == "walkthrough_editor":
            passage_ids = [
                passage["id"]
                for writer in packet["writers"]
                for section in writer["sections"]
                for passage in section["passages"]
            ]
            opening_refs = passage_ids[:1]
            return {
                "preamble": "This is a synthetic source-comparison benchmark. Its prose checks structure only.",
                "introduction": "Synthetic structural benchmark; source fidelity is not measured.",
                "closing": "Review the supplied source before drawing conclusions.",
                "quick_summary": {
                    "before": "The earlier structural fixture contains its supplied changes.",
                    "after": "The edited structural fixture connects the supplied explanations.",
                    "passage_ids": opening_refs,
                },
                "impact": None,
                "route": [{"kind": "mechanism", "passage_ids": opening_refs}],
                "author_reported_checks": [],
                "review_checks": [],
                "sections": [
                    {
                        "title": "Connected changes",
                        "summary": "Local editorial ordering fixture.",
                        "groups": [
                            {
                                "passage_ids": passage_ids,
                                "text": "These supplied changed regions remain accessible together.",
                            }
                        ],
                    }
                ],
            }, None
        story = {
            "preamble": "This is a synthetic source-comparison benchmark. Its prose checks structure only.",
            "introduction": "Synthetic structural benchmark. Review the supplied source for meaning.",
            "closing": "This fixture does not measure narration quality or provider latency.",
            "quick_summary": {
                "before": "The earlier structural fixture supplies the prior evidence.",
                "after": "The later structural fixture accounts for every changed region.",
                "passage_ids": ["all-regions"],
            },
            "impact": None,
            "route": [{"kind": "mechanism", "passage_ids": ["all-regions"]}],
            "author_reported_checks": [],
            "review_checks": [],
            "sections": [
                {
                    "id": "all-changes",
                    "title": "Supplied changes",
                    "summary": "Local accounting fixture.",
                    "passages": [
                        {
                            "id": "all-regions",
                            "text": "These supplied changed regions remain accessible together.",
                            "change_ids": [c["id"] for c in packet["changes"]],
                            "region_ids": [r["id"] for r in packet["regions"]],
                            "citations": [
                                ref
                                for r in packet["regions"]
                                for ref in r["citations"]
                            ],
                        }
                    ],
                }
            ],
        }
        return (
            {
                "orientation": "Synthetic code comparison; model prose quality is not measured.",
                "sections": story["sections"],
            }
            if request["schema_name"] == "walkthrough_part"
            else story
        ), None


def code_identity() -> dict:
    """Bind results to committed and uncommitted implementation bytes."""
    files = sorted(
        [  # Resolve the installed Git.
            *ROOT.joinpath("diffstory").rglob("*.py"),
            *ROOT.joinpath("diffstory/assets").glob("*"),
            Path(__file__).resolve(),
        ]
    )
    hashes = {
        str(path.relative_to(ROOT)): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in files
        if path.is_file()
    }
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],  # noqa: S607 # Resolve the user's installed Git.
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return {"head": result.stdout.strip(), "files_sha256": digest(hashes)}


def _read_fixture(args: argparse.Namespace) -> tuple[dict, str]:
    """Read the chosen snapshot without publishing its source or private path."""
    if args.fixture == "synthetic":
        snapshot = synthetic_snapshot()
        return snapshot, digest(snapshot)
    path = (
        ROOT / "examples/demo.snapshot.json"
        if args.fixture == "demo"
        else args.snapshot
    )
    if path is None:
        msg = (
            "Complex fixture requires --snapshot with a private snapshot path"
        )
        raise ValueError(msg)
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def _private_output(path: Path) -> None:
    """Require private artifacts to remain outside the repository's public set."""
    try:
        relative = path.resolve().relative_to(ROOT)
    except ValueError:
        return
    result = subprocess.run(  # noqa: S603 # Fixed read-only Git query; the path is a separate argument.
        [  # noqa: S607 # Resolve the installed Git.
            "git",
            "check-ignore",
            "--quiet",
            "--",
            str(relative / ".privacy-probe"),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        msg = "Benchmark artifacts require an ignored directory or a path outside the repository"
        raise ValueError(msg)


def run(args: argparse.Namespace) -> dict:
    """Measure each local stage and write complete artifacts only on success.

    Args:
        args: Validated fixture, explicit live mode, provider, and output options.

    Returns:
        Source-free metrics bound to exact code and fixture revisions.

    Raises:
        ValueError: If live transfer is unconfirmed or narration fails.
        OSError: If input/output is unavailable.
    """
    if args.live and not args.yes:
        msg = "Live benchmark sends source; select --live --yes explicitly"
        raise ValueError(msg)
    _private_output(args.output_dir)
    started = time.monotonic()
    snapshot, fixture_sha = _read_fixture(args)
    ingestion = time.monotonic() - started
    started = time.monotonic()
    compiler_report = compile_snapshot(snapshot)
    compilation = time.monotonic() - started
    provider = OfflineProvider()
    if args.live:
        from diffstory.providers import CodexCLIProvider
        from diffstory.providers import OpenAIResponsesProvider

        if (
            args.provider == "openai"
            and args.model != OpenAIResponsesProvider.model
        ):
            msg = "The OpenAI benchmark requires the adapter's supported model"
            raise ValueError(msg)
        provider = (
            CodexCLIProvider(model=args.model)
            if args.provider == "codex"
            else OpenAIResponsesProvider()
        )
        provider.require_credentials()
    narrator = WalkthroughNarrator(
        provider,
        state_dir=args.output_dir / "run",
        cache_dir=args.output_dir / "cache",
    )
    report = narrator.generate(compiler_report)
    started = time.monotonic()
    html = render(report).encode()
    rendering = time.monotonic() - started
    started = time.monotonic()
    report_bytes = canonical_bytes(report)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "walkthrough.report.json").write_bytes(report_bytes)
    (args.output_dir / "walkthrough.html").write_bytes(html)
    output_seconds = time.monotonic() - started
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    packet = report["packet"]
    metrics = {
        "version": "diffstory.walkthrough-benchmark.v1",
        "state": "complete",
        "mode": "live" if args.live else "offline",
        "fixture": args.fixture,
        "semantic_quality": "requires source-fidelity review"
        if args.live
        else "not_measured; structural fixture",
        "code": code_identity(),
        "fixture_sha256": fixture_sha,
        "base_sha": packet["meta"]["base_sha"],
        "head_sha": packet["meta"]["head_sha"],
        "source_bytes": sum(
            len(s["text"].encode()) for s in packet["sources"]
        ),
        "changes": len(packet["changes"]),
        "file_evidence_records": len(packet["file_evidence"]),
        "regions": len(packet["regions"]),
        "provider": provider.name,
        "model": provider.model,
        "usage": report["generation"]["usage"],
        "new_requests": report["generation"]["new_requests"],
        "cache_hits": report["generation"]["cache_hits"],
        "writing_requests": report["generation"]["writing_requests"],
        "editorial_requests": report["generation"]["editorial_requests"],
        "timings": {
            "ingestion_seconds": ingestion,
            "ingestion_kind": "saved_snapshot",
            "compilation_seconds": compilation,
            **narrator.timings,
            "rendering_seconds": rendering,
            "output_seconds": output_seconds,
            "end_to_end_seconds": time.monotonic() - STARTED,
        },
        "peak_memory_bytes": rss if sys.platform == "darwin" else rss * 1_024,
        "report_bytes": len(report_bytes),
        "html_bytes": len(html),
        "annotation_bytes": len(canonical_bytes(report["story"])),
        "cache_bytes": sum(
            path.stat().st_size
            for path in (args.output_dir / "cache").glob("*.json")
        ),
        "run_record_bytes": (args.output_dir / "run/run.json").stat().st_size,
    }
    (args.output_dir / "metrics.json").write_bytes(canonical_bytes(metrics))
    return metrics


def main() -> int:
    """Run one reproducible fixture; source transfer requires explicit live flags."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture", choices=["demo", "complex", "synthetic"], default="demo"
    )
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument(
        "--provider", choices=["codex", "openai"], default="codex"
    )
    parser.add_argument("--model", default="gpt-6.1-sol")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    args = parser.parse_args()
    if args.output_dir is None:
        parent = ROOT / ".walkthrough-benchmark.narration-state"
        parent.mkdir(parents=True, exist_ok=True)
        args.output_dir = Path(
            tempfile.mkdtemp(prefix=f"{args.fixture}-", dir=parent)
        )
    try:
        metrics = run(args)
    except (OSError, ValueError, TimeoutError) as error:
        diagnostic = {"state": "failed", "error_type": type(error).__name__}
        if isinstance(error, (WalkthroughError, RunLimitError)):
            diagnostic["reason"] = str(error)
        print(json.dumps(diagnostic), file=sys.stderr)  # noqa: T201 # Source-free command diagnostic.
        return 2
    print(json.dumps(metrics, indent=2))  # noqa: T201 # Metrics are the command's public output.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
