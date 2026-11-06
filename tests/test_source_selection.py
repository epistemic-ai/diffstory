"""Check explicit test-file selection without weakening source evidence."""

from __future__ import annotations

import copy
import unittest

from test_analysis import snap

from diffstory.analysis import compile_snapshot
from diffstory.source_selection import is_test_path
from diffstory.source_selection import without_test_files


class SourceSelectionTests(unittest.TestCase):
    """Keep recognized test paths out while preserving production evidence."""

    def test_path_conventions_distinguish_tests_from_production(self) -> None:
        """Common test names must not exclude similarly named application code."""
        cases = [
            ("tests/helpers.py", True),
            ("test/fixture.go", True),
            ("src/__tests__/feature.ts", True),
            ("pkg/conftest.py", True),
            ("pkg/test_value.py", True),
            ("pkg/value_test.py", True),
            ("pkg/value_test.go", True),
            ("pkg/value.test.tsx", True),
            ("pkg/value.spec.js", True),
            ("pkg/value_spec.rb", True),
            ("contest.py", False),
            ("pkg/testing.py", False),
            ("specs/openapi.yaml", False),
            ("src/latest.ts", False),
        ]
        for path, expected in cases:
            with self.subTest(path=path):
                self.assertEqual(is_test_path(path), expected)

    def test_selection_preserves_input_and_reports_omitted_files(self) -> None:
        """Filtering must remove all test evidence without mutating the capture."""
        source = "def f():\n    return 1\n"
        cases = [False, True]
        for tests_only in cases:
            with self.subTest(tests_only=tests_only):
                paths = {"tests/test_value.py": source}
                if not tests_only:
                    paths["app.py"] = source
                original = snap({}, paths)
                unchanged = copy.deepcopy(original)
                selected = without_test_files(original)
                self.assertEqual(original, unchanged)
                self.assertEqual(selected["meta"]["excluded_test_files"], 1)
                self.assertEqual(
                    selected["meta"]["changed_files"], int(not tests_only)
                )
                self.assertEqual(
                    selected["meta"]["scope"], "selected excerpts"
                )
                self.assertIn("--include-tests", selected["warnings"][-1])
                for field in ("fragments", "patches"):
                    self.assertEqual(
                        {record["path"] for record in selected[field]},
                        set() if tests_only else {"app.py"},
                    )
                report = compile_snapshot(selected)
                self.assertEqual(
                    report["stats"]["supplied_paths"], int(not tests_only)
                )
                self.assertEqual(without_test_files(selected), selected)

    def test_cross_boundary_rename_retains_both_source_sides(self) -> None:
        """A production/test rename must retain the complete linked comparison."""
        cases = [("tests/old.py", "app.py"), ("app.py", "tests/new.py")]
        for before, after in cases:
            with self.subTest(before=before, after=after):
                source = "def f():\n    return 1\n"
                snapshot = snap({before: source}, {after: source})
                snapshot["file_evidence"] = [
                    {
                        "base": {
                            "path": before,
                            "state": "supplied",
                            "coverage": "full",
                        },
                        "head": {
                            "path": after,
                            "state": "supplied",
                            "coverage": "full",
                        },
                    }
                ]
                snapshot["patches"] = [
                    {"path": after, "origin": "git", "patch": None}
                ]
                for fragment in snapshot["fragments"]:
                    fragment["region"] = after
                selected = without_test_files(snapshot)
                self.assertEqual(selected, snapshot)
                self.assertEqual(
                    compile_snapshot(selected)["stats"]["identical_ast_moves"],
                    1,
                )

    def test_optional_metadata_is_created_when_tests_are_excluded(
        self,
    ) -> None:
        """A valid snapshot without producer metadata must remain compilable."""
        cases = [False, True]
        for tests_only in cases:
            with self.subTest(tests_only=tests_only):
                paths = {"tests/test_value.py": "value = 1\n"}
                if not tests_only:
                    paths["app.py"] = "value = 2\n"
                snapshot = snap({}, paths)
                snapshot.pop("meta")
                selected = without_test_files(snapshot)
                self.assertNotIn("meta", snapshot)
                self.assertEqual(selected["meta"]["excluded_test_files"], 1)
                self.assertEqual(
                    compile_snapshot(selected)["stats"]["supplied_paths"],
                    int(not tests_only),
                )

    def test_unmapped_production_source_cannot_be_hidden_by_filtering(
        self,
    ) -> None:
        """Unclaimed production fragments must still fail evidence validation."""
        snapshot = snap({}, {"tests/test_value.py": "value = 1\n"})
        snapshot["fragments"].append(
            {
                "path": "app.py",
                "side": "head",
                "text": "value = 2\n",
                "start_line": 1,
                "scope": "full",
            }
        )
        with self.assertRaises(ValueError):
            compile_snapshot(without_test_files(snapshot))


if __name__ == "__main__":
    unittest.main()
