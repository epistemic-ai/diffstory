"""Check real parallel dispatch, editorial fidelity, and recovery boundaries."""

from __future__ import annotations

import json
import tempfile
import threading
import unittest
from collections import Counter
from pathlib import Path

from test_analysis import snap
from test_walkthrough import FixtureProvider
from test_walkthrough import ResponsesFixtureProvider
from test_walkthrough import fixture_story

from diffstory.analysis import compile_snapshot
from diffstory.compact_packet import build_packet
from diffstory.compact_packet import canonical_bytes
from diffstory.compact_packet import digest
from diffstory.walkthrough import WalkthroughNarrator
from diffstory.walkthrough import validate_walkthrough
from diffstory.walkthrough_plan import assemble_story
from diffstory.walkthrough_runtime import RunLimitError

SENTINEL = "SOURCE_ONLY_NOT_FOR_EDITOR"


def complex_report() -> dict:
    """Provide four source-coherent modules large enough to require part writing."""
    before = (
        f"def process(value):\n    # {SENTINEL} "
        + "x" * 12_000
        + "\n    return value\n"
    )
    after = before.replace("return value", "return value + 1")
    return compile_snapshot(
        snap(
            {f"m{i}.py": before for i in range(4)},
            {f"m{i}.py": after for i in range(4)},
        )
    )


def complex_report_with_static_test() -> dict:
    """Add one changed test whose source range overlaps a writer citation."""
    base_test = (
        "def test_process():\n"
        "    # " + "x" * 12_000 + "\n"
        "    assert process() == 1\n"
    )
    before = {"tests/test_process.py": base_test}
    after = {"tests/test_process.py": base_test.replace("== 1", "== 2")}
    for index in range(1, 4):
        code = (
            f"def process_{index}(value):\n    # {SENTINEL} "
            + "x" * 12_000
            + "\n    return value\n"
        )
        before[f"m{index}.py"] = code
        after[f"m{index}.py"] = code.replace(
            "return value", "return value + 1"
        )
    return compile_snapshot(snap(before, after))


class ParallelWalkthroughTests(unittest.TestCase):
    """Protect bounded whole-story editing and retained source evidence."""

    def test_parallel_editor_consolidates_prose_and_preserves_evidence(
        self,
    ) -> None:
        """The editor must revise related prose and retain its exact evidence."""
        concurrent = 4
        provider = FixtureProvider()
        provider.part_barrier = threading.Barrier(concurrent)
        provider.include_review_refs = True
        report = WalkthroughNarrator(provider).generate(complex_report())
        validate_walkthrough(report)
        self.assertEqual(provider.calls, 5)
        self.assertEqual(provider.max_active, concurrent)
        packet = build_packet(complex_report())
        texts = [
            p["text"]
            for s in report["story"]["sections"]
            for p in s["passages"]
        ]
        self.assertEqual(len(texts), 1)
        self.assertEqual(
            texts[0],
            "One edited explanation connects the supplied mechanisms.",
        )
        self.assertNotEqual(Counter(texts), Counter(provider.written_prose))
        self.assertNotIn(SENTINEL.encode(), provider.editor_request)
        for prose in provider.written_prose:
            self.assertIn(prose.encode(), provider.editor_request)
        passage = report["story"]["sections"][0]["passages"][0]
        self.assertEqual(
            set(passage["change_ids"]),
            {item["id"] for item in packet["changes"]},
        )
        self.assertEqual(
            set(passage["region_ids"]),
            {item["id"] for item in packet["regions"]},
        )
        expected_citations = {
            (ref["source_id"], ref["revision"], ref["start"], ref["end"])
            for region in packet["regions"]
            for ref in region["citations"]
        }
        actual_citations = {
            (ref["source_id"], ref["revision"], ref["start"], ref["end"])
            for ref in passage["citations"]
        }
        self.assertEqual(actual_citations, expected_citations)
        final_id = passage["id"]
        self.assertEqual(
            report["story"]["quick_summary"]["passage_ids"], [final_id]
        )
        self.assertEqual(
            report["story"]["route"][0]["passage_ids"], [final_id]
        )
        self.assertEqual(report["story"]["impact"]["passage_ids"], [final_id])
        self.assertEqual(
            report["story"]["review_checks"][0]["passage_ids"], [final_id]
        )

    def test_omitted_editorial_passage_never_becomes_complete(self) -> None:
        """An editor cannot silently discard a previously validated explanation."""
        missing = True
        provider = FixtureProvider()
        provider.drop_editor_passage = missing
        with self.assertRaises(RunLimitError):
            WalkthroughNarrator(provider).generate(complex_report())
        self.assertEqual(provider.calls, 6)

    def test_duplicate_and_unknown_editorial_ids_never_become_complete(
        self,
    ) -> None:
        """Every original passage must occur once, even after prose merging."""
        for mutation in ("duplicate", "unknown"):
            with self.subTest(mutation=mutation):
                provider = FixtureProvider()
                provider.editor_mutation = mutation
                with self.assertRaises(RunLimitError):
                    WalkthroughNarrator(provider).generate(complex_report())

    def test_merged_citation_limit_rejects_instead_of_trimming_evidence(
        self,
    ) -> None:
        """A group over 256 exact citations stays incomplete and untrimmed."""
        passage_count = 257
        passages = {
            f"w0-p{index}": {
                "id": f"w0-p{index}",
                "change_ids": [f"change-{index}"],
                "region_ids": [],
                "citations": [
                    {
                        "source_id": f"source-{index}",
                        "revision": "b" * 40,
                        "start": 1,
                        "end": 1,
                    }
                ],
            }
            for index in range(passage_count)
        }
        ids = list(passages)
        editorial = {
            "preamble": "A project orientation.",
            "introduction": "A short lead.",
            "closing": "Review the cited evidence.",
            "quick_summary": {
                "before": "The earlier fixture has one state.",
                "after": "The later fixture has another state.",
                "passage_ids": [ids[0]],
            },
            "impact": None,
            "route": [{"kind": "mechanism", "passage_ids": [ids[0]]}],
            "author_reported_checks": [],
            "review_checks": [],
            "sections": [
                {
                    "title": "One concept",
                    "summary": "Merge every supplied source group.",
                    "groups": [{"passage_ids": ids, "text": "Edited prose."}],
                }
            ],
        }
        with self.assertRaisesRegex(ValueError, "256-citation limit"):
            assemble_story({}, passages, editorial)

    def test_editorial_envelope_is_preflighted_before_writing(self) -> None:
        """A too-small final request capacity must fail before any source transfer."""
        capacity = 200_000
        provider = FixtureProvider()
        provider.context_tokens = capacity
        with self.assertRaises(ValueError):
            WalkthroughNarrator(provider).generate(complex_report())
        self.assertEqual(provider.calls, 0)

    def test_final_review_check_links_cited_static_test_ids(self) -> None:
        """Review links must use test IDs supplied for cited writer passages."""
        report = complex_report_with_static_test()
        packet = build_packet(report)
        expected_test_id = packet["analysis"]["tests"][0]["id"]
        provider = FixtureProvider()
        provider.include_review_refs = True
        generated = WalkthroughNarrator(provider).generate(report)
        self.assertEqual(
            generated["story"]["review_checks"][0]["test_ids"],
            [expected_test_id],
        )

    def test_resume_revalidates_cached_editorial_selection_without_new_calls(
        self,
    ) -> None:
        """Cached editorial output must reload as raw selection data and retain usage."""
        expected_calls = 5
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            provider = FixtureProvider()
            compiler = complex_report()
            first = WalkthroughNarrator(
                provider, state_dir=root / "run", cache_dir=root / "cache"
            ).generate(compiler)
            compiler["meta"]["captured_at"] = (
                "new capture of the same pinned source"
            )
            second = WalkthroughNarrator(
                provider, state_dir=root / "run", cache_dir=root / "cache"
            ).generate(compiler)
            self.assertEqual(provider.calls, expected_calls)
            self.assertEqual(first["story"], second["story"])
            self.assertEqual(
                second["generation"]["usage"]["calls"], expected_calls
            )
            self.assertEqual(second["generation"]["new_requests"], 0)
            self.assertEqual(
                second["generation"]["cache_hits"], expected_calls
            )

    def test_responses_escaping_counts_toward_writer_projection_limit(
        self,
    ) -> None:
        """JSON escaping cannot hide an oversized source-free editor projection."""
        from diffstory.walkthrough_plan import validate_part

        packet = build_packet(compile_snapshot(snap({}, {"a.py": "x = 1\n"})))
        story = fixture_story(packet)
        part = {
            "orientation": "A source-grounded fixture orientation.",
            "sections": [
                {
                    "id": "s",
                    "title": "Escaping case",
                    "summary": "A source-linked passage with escaped characters.",
                    "passages": story["sections"][0]["passages"],
                }
            ],
        }
        part["sections"][0]["passages"][0]["text"] = '"\\' * 4_000
        provider = ResponsesFixtureProvider()
        with self.assertRaisesRegex(ValueError, "24 KiB"):
            validate_part(
                packet,
                part,
                packet=packet,
                writer_index=0,
                provider=provider,
            )

    def test_cached_writer_projection_is_revalidated_before_editor_call(
        self,
    ) -> None:
        """A forged cache entry cannot bypass the complete projection bound."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            provider = ResponsesFixtureProvider()
            report = complex_report()
            WalkthroughNarrator(
                provider, state_dir=root / "run", cache_dir=root / "cache"
            ).generate(report)
            writer_entry = next(
                path
                for path in (root / "cache").glob("*.json")
                if "orientation" in path.read_text()
            )
            entry = json.loads(writer_entry.read_text())
            entry["result"]["sections"][0]["passages"][0]["text"] = (
                '"\\' * 4_000
            )
            entry["digest"] = digest(entry["result"])
            writer_entry.write_bytes(canonical_bytes(entry))

            with self.assertRaisesRegex(RunLimitError, "Cached narration"):
                WalkthroughNarrator(
                    provider,
                    state_dir=root / "run",
                    cache_dir=root / "cache",
                ).generate(report)
            self.assertEqual(provider.calls, 5)


if __name__ == "__main__":
    unittest.main()
