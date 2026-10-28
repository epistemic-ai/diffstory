"""Check whole-hunk partitioning and model-aware narration preflight."""

from __future__ import annotations

import copy
import unittest
from collections import Counter
from unittest.mock import patch

from test_analysis import snap
from test_walkthrough import FixtureProvider
from test_walkthrough import fixture_story
from tiktoken import Encoding

from diffstory.analysis import compile_snapshot
from diffstory.compact_packet import build_packet
from diffstory.compact_packet import packet_identity
from diffstory.compact_packet import source_index
from diffstory.render import render
from diffstory.walkthrough import WalkthroughNarrator
from diffstory.walkthrough import validate_walkthrough
from diffstory.walkthrough_plan import check_capacity
from diffstory.walkthrough_plan import validate_part


def large_module(functions: int = 4) -> tuple[str, str]:
    """Return separated changed functions whose complete file exceeds a writer.

    Args:
        functions: Positive number of independently changed function hunks.

    Returns:
        Earlier and later source with each long comment unchanged.
    """
    before = "".join(
        f"def f{index}():\n    # "
        + "x" * 80_000
        + "\n    return 1\n"
        + "\n" * 10
        for index in range(functions)
    )
    return before, before.replace("return 1", "return 2")


class WalkthroughPlanTests(unittest.TestCase):
    """Preserve complete change coverage while respecting each input budget."""

    def test_oversized_file_splits_without_changing_saved_packet(self) -> None:
        """File subdivision must retain every region, source, and original patch."""
        before, after = large_module()
        cases = [False, True]
        for include_neighbor in cases:
            with self.subTest(include_neighbor=include_neighbor):
                old, new = {"large.py": before}, {"large.py": after}
                if include_neighbor:
                    old["small.py"] = "value = 1\n"
                    new["small.py"] = "value = 2\n"
                compiled = compile_snapshot(snap(old, new))
                packet = build_packet(compiled)
                original = copy.deepcopy(packet)
                provider = FixtureProvider()
                provider.context_tokens = 400_000
                narrator = WalkthroughNarrator(provider)
                plan = narrator.prepare(compiled)
                self.assertFalse(plan.single)
                self.assertGreater(len(plan.writers), 1)
                self.assertLessEqual(len(plan.writers), 8)
                assigned = Counter(
                    region["id"]
                    for writer in plan.writers
                    for region in writer.evidence["regions"]
                )
                self.assertEqual(
                    assigned,
                    Counter(region["id"] for region in packet["regions"]),
                )
                for writer in plan.writers:
                    check_capacity(provider, writer.body)
                generated = narrator.generate_prepared()
                validate_walkthrough(generated)
                self.assertEqual(generated["packet"], original)
                self.assertEqual(
                    plan.evidence_digest, packet_identity(original)
                )
                self.assertIn("<!doctype html", render(generated).lower())

    def test_large_function_can_split_between_intact_hunks(self) -> None:
        """A complete definition must not reconnect otherwise fitting hunks."""
        before = (
            "def f():\n    first = 1\n"
            + "".join(
                f"    value_{index} = {index}  # " + "x" * 200 + "\n"
                for index in range(1_000)
            )
            + "    return first\n"
        )
        after = before.replace("first = 1", "first = 2").replace(
            "return first", "return first + 1"
        )
        compiled = compile_snapshot(
            snap({"large.py": before}, {"large.py": after})
        )
        provider = FixtureProvider()
        provider.context_tokens = 400_000
        plan = WalkthroughNarrator(provider).prepare(compiled)
        self.assertEqual(len(plan.writers), 2)
        original_diffs = {diff["id"]: diff for diff in plan.packet["diffs"]}
        for writer in plan.writers:
            for change in writer.evidence["changes"]:
                self.assertIn(
                    "partial", change["definition_coverage"].values()
                )
            for diff in writer.evidence["diffs"]:
                for hunk in diff["hunks"]:
                    self.assertEqual(
                        {
                            key: value
                            for key, value in hunk.items()
                            if key != "original_index"
                        },
                        original_diffs[diff["id"]]["hunks"][
                            hunk["original_index"]
                        ],
                    )

    def test_unsent_lines_cannot_support_a_partitioned_passage(self) -> None:
        """A citation must stay within its writer slice even if full source exists."""
        before, after = large_module()
        compiled = compile_snapshot(
            snap({"large.py": before}, {"large.py": after})
        )
        provider = FixtureProvider()
        provider.context_tokens = 400_000
        plan = WalkthroughNarrator(provider).prepare(compiled)
        writer = plan.writers[0]
        story = fixture_story(writer.evidence)
        part = {
            "orientation": "Fixture context.",
            "sections": story["sections"],
        }
        citation = part["sections"][0]["passages"][0]["citations"][0]
        full = source_index(plan.packet)
        source = full[citation["source_id"]]
        last_line = source["start_line"] + len(full.physical[source["id"]]) - 1
        citation["start"] = citation["end"] = last_line
        with self.assertRaisesRegex(ValueError, "outside supplied source"):
            validate_part(
                writer.evidence,
                part,
                packet=plan.packet,
                writer_index=0,
                provider=provider,
            )

    def test_indivisible_units_report_size_without_sending_source(
        self,
    ) -> None:
        """Whole hunks and source-only ranges must fail with usable diagnostics."""
        text = "x" * 400_000
        cases = [
            (
                {},
                {"huge.py": "# " + text + "\nvalue = 1\n"},
                "huge.py",
                "hunk 0",
            ),
            (
                {"huge.txt": text + "\n"},
                {"huge.txt": text + "\r\n"},
                "huge.txt",
                "change-",
            ),
        ]
        for before, after, path, region in cases:
            with self.subTest(path=path):
                provider = FixtureProvider()
                provider.context_tokens = 400_000
                snapshot = snap(before, after)
                if region == "change-":
                    snapshot["patches"][0]["patch"] = None
                compiled = compile_snapshot(snapshot)
                with self.assertRaisesRegex(
                    ValueError,
                    r"complete unit.*estimated input tokens.*available",
                ) as raised:
                    WalkthroughNarrator(provider).generate(compiled)
                self.assertIn(path, str(raised.exception))
                self.assertIn(region, str(raised.exception))
                self.assertIn("unit was kept intact", str(raised.exception))
                self.assertIn(
                    "Review without narration", str(raised.exception)
                )
                self.assertEqual(provider.calls, 0)

    def test_underestimated_unit_still_reaches_whole_hunk_subdivision(
        self,
    ) -> None:
        """An optimistic packing hint must not bypass oversized-unit recovery."""
        before, after = large_module()
        compiled = compile_snapshot(
            snap({"large.py": before}, {"large.py": after})
        )
        cases = [1]
        for isolated_record_tokens in cases:
            with (
                self.subTest(isolated_record_tokens=isolated_record_tokens),
                patch(
                    "diffstory.walkthrough_plan._input_size",
                    return_value=isolated_record_tokens,
                ),
            ):
                provider = FixtureProvider()
                provider.context_tokens = 400_000
                plan = WalkthroughNarrator(provider).prepare(compiled)
                self.assertGreater(len(plan.writers), 1)
                self.assertTrue(
                    all(
                        writer.evidence["version"]
                        == "diffstory.writing-evidence.v2"
                        for writer in plan.writers
                    )
                )
                for writer in plan.writers:
                    check_capacity(provider, writer.body)
                self.assertEqual(provider.calls, 0)

    def test_underestimated_batches_are_repacked_before_the_final_guard(
        self,
    ) -> None:
        """Individually fitting units must not produce oversized combined batches."""
        before, after = {}, {}
        for index in range(10):
            source = (
                f"def f{index}():\n    # " + "x" * 30_000 + "\n    return 1\n"
            )
            before[f"m{index}.py"] = source
            after[f"m{index}.py"] = source.replace("return 1", "return 2")
        compiled = compile_snapshot(snap(before, after))
        cases = [1]
        for isolated_record_tokens in cases:
            with (
                self.subTest(isolated_record_tokens=isolated_record_tokens),
                patch(
                    "diffstory.walkthrough_plan._input_size",
                    return_value=isolated_record_tokens,
                ),
            ):
                provider = FixtureProvider()
                provider.context_tokens = 400_000
                plan = WalkthroughNarrator(provider).prepare(compiled)
                self.assertGreater(len(plan.writers), 4)
                self.assertLessEqual(len(plan.writers), 8)
                for writer in plan.writers:
                    self.assertEqual(
                        writer.evidence["version"],
                        "diffstory.writing-evidence.v1",
                    )
                    check_capacity(provider, writer.body)
                self.assertEqual(provider.calls, 0)

    def test_subdivision_cannot_exceed_the_writer_limit(self) -> None:
        """Source subdivision must not silently widen the existing call budget."""
        cases = [9]
        for functions in cases:
            with self.subTest(functions=functions):
                before, after = large_module(functions)
                provider = FixtureProvider()
                provider.context_tokens = 400_000
                compiled = compile_snapshot(
                    snap({"large.py": before}, {"large.py": after})
                )
                with self.assertRaisesRegex(ValueError, "eight-writer"):
                    WalkthroughNarrator(provider).generate(compiled)
                self.assertEqual(provider.calls, 0)

    def test_known_tokens_can_admit_a_request_rejected_by_byte_sizing(
        self,
    ) -> None:
        """Compressible requests must use local model tokens at the final guard."""
        ranks = {bytes([value]): value for value in range(256)}
        for power in range(1, 17):
            ranks[b"x" * (2**power)] = len(ranks)
        encoding = Encoding(
            name="fixture-packing",
            pat_str=r"[\s\S]+",
            mergeable_ranks=ranks,
            special_tokens={},
        )
        source = "# " + "x" * 300_000 + "\nvalue = 1\n"
        compiled = compile_snapshot(snap({}, {"large.py": source}))
        cases = [(None, False), (encoding, True)]
        for counter, fits in cases:
            with (
                self.subTest(tokenized=fits),
                patch("diffstory.usage._model_encoder", return_value=counter),
            ):
                provider = FixtureProvider()
                provider.context_tokens = 400_000
                narrator = WalkthroughNarrator(provider)
                if fits:
                    plan = narrator.prepare(compiled)
                    self.assertTrue(plan.single)
                    self.assertGreater(len(plan.writers[0].body), 400_000)
                    check_capacity(provider, plan.writers[0].body)
                else:
                    with self.assertRaisesRegex(ValueError, "complete unit"):
                        narrator.prepare(compiled)


if __name__ == "__main__":
    unittest.main()
