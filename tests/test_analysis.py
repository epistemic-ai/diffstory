"""Deterministic semantic analysis and annotation regression tests."""

import copy
import tempfile
import unittest
from pathlib import Path

from diffstory.analysis import apply_annotations
from diffstory.analysis import compile_snapshot
from diffstory.analysis import evidence_packet
from diffstory.analysis import ordered_components
from diffstory.ingest import parse_pr
from diffstory.render import render


def snap(
    before: dict[str, str],
    after: dict[str, str],
    **meta: object,
) -> dict:
    """
    Build a snapshot from base and head path-to-source mappings.

    Args:
        before: Base-side source mapping.
        after: Head-side source mapping.
        **meta: Metadata fields merged into the snapshot.

    Returns:
        A complete ``diffstory.snapshot.v1`` mapping.

    """
    return {
        "schema": "diffstory.snapshot.v1",
        "meta": {
            "head_sha": "b" * 40,
            "base_sha": "a" * 40,
            "scope": "changed files",
            **meta,
        },
        "fragments": [
            {
                "path": path,
                "side": side,
                "text": src,
                "start_line": 1,
                "scope": "full",
            }
            for side, items in [("base", before), ("head", after)]
            for path, src in items.items()
        ],
    }


def symbols(r: dict) -> list[dict]:
    """
    Return semantic changes from a compiled report.

    Args:
        r: Compiled report mapping.

    Returns:
        Changes other than raw wiring and text context.

    """
    return [c for c in r["changes"] if c["kind"] not in {"wiring", "text"}]


class CompilerTests(unittest.TestCase):
    """Check deterministic change classification and report assembly."""

    def test_exact_move(self) -> None:
        """Recognize an unchanged declaration moved to a new path as an identical-AST move."""
        r = compile_snapshot(
            snap(
                {"a.py": "def f(x):\n    return x + 1\n"},
                {"b.py": "def f(x):\n    return x + 1\n"},
            )
        )
        self.assertEqual([c["kind"] for c in symbols(r)], ["moved"])
        self.assertEqual(r["stats"]["identical_ast_moves"], 1)

    def test_rename(self) -> None:
        """Recognize a declaration moved and renamed as a moved-rename candidate."""
        r = compile_snapshot(
            snap(
                {"a.py": "def old(x):\n    return x\n"},
                {"b.py": "def new(x):\n    return x\n"},
            )
        )
        self.assertEqual(symbols(r)[0]["kind"], "moved_renamed")

    def test_literal_change_never_move_identical(self) -> None:
        """Keep literal changes from being classified as identical-AST moves."""
        r = compile_snapshot(
            snap(
                {"a.py": "def f(x):\n    return x + 1\n"},
                {"b.py": "def f(x):\n    return x + 2\n"},
            )
        )
        self.assertEqual(symbols(r)[0]["kind"], "moved_modified")

    def test_string_literal_whitespace_is_not_normalized(self) -> None:
        """Preserve string-literal whitespace as part of structural identity."""
        r = compile_snapshot(
            snap(
                {"a.py": 'def f():\n    return "a b"\n'},
                {"b.py": 'def f():\n    return "a  b"\n'},
            )
        )
        self.assertEqual(r["stats"]["identical_ast_moves"], 0)
        self.assertEqual(symbols(r)[0]["kind"], "moved_modified")

    def test_internal_rename_not_erased(self) -> None:
        """Preserve internal identifier changes when comparing declarations."""
        r = compile_snapshot(
            snap(
                {"a.py": "def f(x):\n    return service(x)\n"},
                {"b.py": "def f(x):\n    return unsafe_service(x)\n"},
            )
        )
        self.assertNotEqual(symbols(r)[0]["kind"], "moved")

    def test_decorator_preserved(self) -> None:
        """Keep decorator changes visible in classification and source evidence."""
        r = compile_snapshot(
            snap(
                {"a.py": "@cache\ndef f(x):\n    return x\n"},
                {"b.py": "@async_cache\ndef f(x):\n    return x\n"},
            )
        )
        self.assertEqual(symbols(r)[0]["kind"], "moved_modified")
        self.assertIn("@cache", symbols(r)[0]["before"]["source"])

    def test_signature_default_change(self) -> None:
        """Report a changed default value as a signature change."""
        r = compile_snapshot(
            snap(
                {"a.py": "def f(x=1):\n    return x\n"},
                {"a.py": "def f(x=2):\n    return x\n"},
            )
        )
        self.assertTrue(symbols(r)[0]["signature_changed"])

    def test_comments_are_source_only(self) -> None:
        """Classify comment-only edits as source changes without AST changes."""
        r = compile_snapshot(
            snap(
                {"a.py": "def f():\n    # old\n    return 1\n"},
                {"a.py": "def f():\n    # new\n    return 1\n"},
            )
        )
        self.assertEqual(symbols(r)[0]["kind"], "source_only")

    def test_docstring_difference_not_identical(self) -> None:
        """Keep docstring changes visible and distinct from identical AST moves."""
        r = compile_snapshot(
            snap(
                {"a.py": 'def f():\n    """old"""\n    return 1\n'},
                {"b.py": 'def f():\n    """new"""\n    return 1\n'},
            )
        )
        self.assertEqual(r["stats"]["identical_ast_moves"], 0)
        self.assertTrue(symbols(r)[0]["docstring_changed"])

    def test_ambiguous_equal_bodies_not_paired(self) -> None:
        """Leave ambiguous identical bodies unmatched instead of guessing correspondence."""
        r = compile_snapshot(
            snap(
                {"a.py": "def a(x):\n return x\ndef b(x):\n return x\n"},
                {"c.py": "def c(x):\n return x\ndef d(x):\n return x\n"},
            )
        )
        self.assertEqual(len(symbols(r)), 4)
        self.assertEqual(r["stats"]["identical_ast_moves"], 0)

    def test_no_source_execution(self) -> None:
        """Parse untrusted source without executing its contents."""
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "bad"
            compile_snapshot(
                snap({}, {"evil.py": f'open({str(p)!r}, "w").write("oops")\n'})
            )
            self.assertFalse(p.exists())

    def test_parse_error_is_visible_not_dropped(self) -> None:
        """Preserve raw evidence and warnings when Python parsing fails."""
        r = compile_snapshot(snap({}, {"a.py": "def broken("}))
        self.assertTrue(any("AST unavailable" in w for w in r["warnings"]))
        self.assertTrue(r["raw_files"])
        self.assertTrue(r["changes"])

    def test_text_file_preserved(self) -> None:
        """Retain non-Python text changes as report evidence."""
        r = compile_snapshot(
            snap({"BUILD.bazel": "x = 1"}, {"BUILD.bazel": "x = 2"})
        )
        self.assertEqual(r["changes"][0]["kind"], "text")
        self.assertIn("x = 2", r["changes"][0]["after"]["source"])

    def test_import_wiring(self) -> None:
        """Surface import-only edits as wiring changes."""
        r = compile_snapshot(
            snap(
                {"caller.py": "from a import f\n"},
                {"caller.py": "from b import f\n"},
            )
        )
        self.assertTrue(any(c["kind"] == "wiring" for c in r["changes"]))

    def test_alias_resolves(self) -> None:
        """Resolve a uniquely aliased import to its referenced symbol."""
        r = compile_snapshot(
            snap(
                {},
                {
                    "lib.py": "def f(x):\n return x\n",
                    "caller.py": "from lib import f as renamed\ndef run():\n return renamed(1)\n",
                },
            )
        )
        self.assertTrue(any(e["type"] == "calls" for e in r["symbol_edges"]))

    def test_parameter_shadow_does_not_resolve_import(self) -> None:
        """Do not resolve an imported name shadowed by a parameter."""
        r = compile_snapshot(
            snap(
                {},
                {
                    "lib.py": "def f(x):\n return x\n",
                    "caller.py": "from lib import f\ndef run(f):\n return f(1)\n",
                },
            )
        )
        self.assertFalse(any(e["type"] == "calls" for e in r["symbol_edges"]))

    def test_local_shadow_does_not_resolve_import(self) -> None:
        """Do not resolve an imported name shadowed by a local assignment."""
        r = compile_snapshot(
            snap(
                {},
                {
                    "lib.py": "def f(x):\n return x\n",
                    "caller.py": "from lib import f\ndef run():\n f = print\n return f(1)\n",
                },
            )
        )
        self.assertFalse(any(e["type"] == "calls" for e in r["symbol_edges"]))

    def test_test_reference_not_pass(self) -> None:
        """Treat static test references as evidence of test intent, not executed test runs."""
        r = compile_snapshot(
            snap(
                {},
                {
                    "lib.py": "def f(x):\n return x\n",
                    "tests/test_lib.py": "from lib import f\ndef test_f():\n assert f(1) == 1\n",
                },
            )
        )
        self.assertEqual(r["stats"]["test_runs"], 0)
        links = [x for g in r["groups"] for x in g["test_links"]]
        self.assertTrue(links)
        self.assertTrue(
            all(x["status"] == "referenced, not run" for x in links)
        )

    def test_parametrize_reference_is_detected_without_call_claim(
        self,
    ) -> None:
        """Detect parametrized references without claiming the test called the symbol."""
        r = compile_snapshot(
            snap(
                {},
                {
                    "lib.py": "def f(x):\n return x\n",
                    "tests/test_lib.py": 'from lib import f\nimport pytest\n@pytest.mark.parametrize("builder", [f])\ndef test_f(builder):\n assert builder(1)==1\n',
                },
            )
        )
        self.assertTrue(
            any(e["type"] == "test_references" for e in r["symbol_edges"])
        )
        self.assertFalse(
            any(e["type"] == "test_calls" for e in r["symbol_edges"])
        )

    def test_prerequisite_order(self) -> None:
        """Place prerequisite groups before dependent groups."""
        r = compile_snapshot(
            snap(
                {},
                {
                    "z.py": "def f(x):\n return x\n",
                    "a.py": "from z import f\ndef run():\n return f(1)\n",
                },
            )
        )
        order = {g["id"]: i for i, g in enumerate(r["groups"])}
        self.assertTrue(
            all(order[e["from"]] < order[e["to"]] for e in r["edges"])
        )

    def test_cycles_condensed(self) -> None:
        """Keep prerequisite cycles explicit and order downstream nodes after them."""
        order, cycles = ordered_components(
            ["a", "b", "c"], {"a": {"b"}, "b": {"a"}, "c": {"b"}}, lambda x: x
        )
        self.assertEqual(set(cycles[0]), {"a", "b"})
        self.assertEqual(order[-1], "c")

    def test_excerpt_not_assumed_added(self) -> None:
        """Label unmatched excerpt definitions as observed rather than proven additions."""
        r = compile_snapshot(
            snap(
                {},
                {"a.py": "def f():\n return 1\n"},
                scope="selected excerpts",
            )
        )
        self.assertEqual(symbols(r)[0]["kind"], "observed_head")

    def test_overlapping_fragments_rejected(self) -> None:
        """Reject overlapping source fragments for the same path and side."""
        s = snap({}, {"a.py": "def f():\n return 1\n"})
        s["fragments"].append(dict(s["fragments"][0]))
        with self.assertRaisesRegex(ValueError, "Overlapping"):
            compile_snapshot(s)

    def test_original_line_offsets(self) -> None:
        """Preserve original fragment line offsets in symbols and diff hunks."""
        s = snap(
            {"a.py": "def f():\n return 1\n"},
            {"a.py": "def f():\n return 2\n"},
        )
        for f in s["fragments"]:
            f["start_line"] = 100
        r = compile_snapshot(s)
        c = symbols(r)[0]
        self.assertEqual(c["after"]["start"], 100)
        self.assertTrue(
            any(
                row["new"] == 101
                for h in c["hunks"]
                for row in h["rows"]
                if row["tag"] == "add"
            )
        )

    def test_safe_html_embedding(self) -> None:
        """Escape report data for script embedding and prohibit network connections."""
        r = compile_snapshot(
            snap({}, {"bad.py": 'x = "</script><script>alert(1)</script>"\n'})
        )
        html = render(r)
        self.assertNotIn("</script><script>alert(1)</script>", html)
        self.assertIn("\\u003c/script\\u003e", html)
        self.assertIn("connect-src 'none'", html)

    def test_malformed_report_identifier_rejected(self) -> None:
        """Reject malformed report identifiers before rendering HTML."""
        r = compile_snapshot(snap({}, {"a.py": "x=1"}))
        r["groups"][0]["id"] = '"><img src=x onerror=alert(1)>'
        with self.assertRaisesRegex(ValueError, "identifier"):
            render(r)

    def test_malformed_report_number_rejected(self) -> None:
        """Reject non-numeric report statistics before rendering HTML."""
        r = compile_snapshot(snap({}, {"a.py": "x=1"}))
        r["stats"]["units"] = "<img src=x>"
        with self.assertRaisesRegex(ValueError, "statistic"):
            render(r)

    def test_malformed_report_diff_tag_rejected(self) -> None:
        """Reject unsupported diff-row tags before rendering HTML."""
        r = compile_snapshot(snap({}, {"a.py": "x=1"}))
        r["changes"][0]["hunks"][0]["rows"][0]["tag"] = "evil"
        with self.assertRaisesRegex(ValueError, "diff row tag"):
            render(r)

    def test_annotations_wrong_revision_rejected(self) -> None:
        """Reject annotations bound to a different report revision."""
        r = compile_snapshot(snap({}, {"a.py": "x=1"}))
        with self.assertRaisesRegex(ValueError, "different head"):
            apply_annotations(
                r,
                {
                    "schema": "diffstory.annotations.v1",
                    "base_sha": "a" * 40,
                    "head_sha": "wrong",
                    "steps": [],
                },
            )

    def test_annotations_cross_group_evidence_rejected(self) -> None:
        """Reject evidence IDs owned by another group."""
        r = compile_snapshot(snap({}, {"a.py": "x=1"}))
        g = r["groups"][0]
        a = {
            "schema": "diffstory.annotations.v1",
            "base_sha": "a" * 40,
            "head_sha": "b" * 40,
            "steps": [
                {"group_id": g["id"], "evidence_change_ids": ["invented"]}
            ],
        }
        with self.assertRaisesRegex(ValueError, "outside its group"):
            apply_annotations(r, a)

    def test_annotations_cannot_override_facts(self) -> None:
        """Allow narrative labels while preserving compiler facts and classifications."""
        r = compile_snapshot(snap({}, {"a.py": "x=1"}))
        g = r["groups"][0]
        a = {
            "schema": "diffstory.annotations.v1",
            "base_sha": "a" * 40,
            "head_sha": "b" * 40,
            "steps": [
                {
                    "group_id": g["id"],
                    "evidence_change_ids": g["change_ids"],
                    "title": "A better title",
                    "kind": "moved",
                    "test_runs": 100,
                }
            ],
        }
        out = apply_annotations(r, a)
        self.assertEqual(out["stats"]["test_runs"], 0)
        self.assertEqual(out["changes"], r["changes"])
        self.assertEqual(out["groups"][0]["title"], "A better title")

    def test_deterministic_compile(self) -> None:
        """Produce byte-for-byte equal reports from the same snapshot."""
        s = snap(
            {"a.py": "def a(x):\n return x\n"},
            {"b.py": "def b(x):\n return x\n"},
        )
        self.assertEqual(
            compile_snapshot(s), compile_snapshot(copy.deepcopy(s))
        )

    def test_empty_change_compiles(self) -> None:
        """Compile identical source revisions as an empty change report."""
        r = compile_snapshot(snap({"a.py": "x=1"}, {"a.py": "x=1"}))
        self.assertEqual(r["stats"]["units"], 0)
        self.assertEqual(r["groups"], [])

    def test_github_url_parser(self) -> None:
        """Accept supported pull-request references and reject unrelated hosts."""
        self.assertEqual(
            parse_pr("example/catalog#42"), ("example/catalog", 42)
        )
        self.assertEqual(
            parse_pr("https://github.com/example/catalog/pull/42/changes"),
            ("example/catalog", 42),
        )
        with self.assertRaises(ValueError):
            parse_pr("https://evil.example/repo/pull/1")

    def test_evidence_packet(self) -> None:
        """Export deterministic review instructions and source evidence without a network call."""
        r = compile_snapshot(snap({}, {"a.py": "x=1"}))
        e = evidence_packet(r)
        self.assertEqual(e["schema"], "diffstory.narrative-request.v1")
        self.assertIn("untrusted", e["instructions"])


if __name__ == "__main__":
    unittest.main()
