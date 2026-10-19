"""Deterministic semantic analysis and annotation regression tests."""

import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from difflib import unified_diff
from itertools import permutations
from itertools import product
from pathlib import Path

from diffstory.analysis import apply_annotations
from diffstory.analysis import compile_snapshot
from diffstory.analysis import evidence_packet
from diffstory.analysis import make_hunks
from diffstory.analysis import ordered_components
from diffstory.ingest import parse_pr
from diffstory.render import render
from diffstory.source_lines import split_source_lines


def snap(
    before: dict[str, str],
    after: dict[str, str],
    *,
    overrides: dict[tuple[str, str], dict] | None = None,
    fragment_options: dict[tuple[str, str], dict] | None = None,
    warnings: list[str] | None = None,
    **meta: object,
) -> dict:
    """Build explicit evidence from complete base and head source mappings.

    Args:
        before: Base-side source by repository path.
        after: Head-side source by repository path.
        overrides: Optional complete evidence side records keyed by side/path.
        fragment_options: Optional fields such as range and region by side/path.
        warnings: Producer warning strings.
        **meta: Snapshot metadata fields that override defaults.

    Returns:
        A v1 snapshot with supplied and absent file evidence.
    """
    overrides = overrides or {}
    fragment_options = fragment_options or {}
    paths = sorted(before.keys() | after.keys())
    records = []
    for path in paths:
        record = {}
        for side, sources in (("base", before), ("head", after)):
            value = (
                {"path": path, "state": "supplied", "coverage": "full"}
                if path in sources
                else {"path": path, "state": "absent"}
            )
            value = dict(overrides.get((side, path), value))
            value.setdefault("path", path)
            record[side] = value
        records.append(record)
    fragments = []
    for side, sources in (("base", before), ("head", after)):
        for path, text in sources.items():
            fragment = {"path": path, "side": side, "text": text}
            fragment.update(fragment_options.get((side, path), {}))
            fragments.append(fragment)
    return {
        "schema": "diffstory.snapshot.v1",
        "meta": {
            "base_sha": "a" * 40,
            "head_sha": "b" * 40,
            "scope": "changed files",
            **meta,
        },
        "fragments": fragments,
        "file_evidence": records,
        # Synthetic test input only. Production must capture the supplied patch.
        "patches": [
            {
                "path": path,
                "origin": "github",
                "patch": "\n".join(
                    row.rstrip("\r\n")
                    for row in unified_diff(
                        split_source_lines(
                            before.get(path, ""), keepends=True
                        ),
                        split_source_lines(after.get(path, ""), keepends=True),
                        lineterm="",
                    )
                )
                if all(
                    side["state"] == "absent" or side.get("coverage") == "full"
                    for side in record.values()
                )
                else None,
            }
            for path, record in zip(paths, records, strict=True)
        ],
        "warnings": warnings or [],
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


def graph_snapshot(size: int, *, cyclic: bool) -> dict:
    """Build full modules whose functions call the next module in a chain or ring."""
    sources = {}
    for index in range(size):
        following = (index + 1) % size
        sources[f"m_{index:04d}.py"] = (
            f"from m_{following:04d} import f_{following}\n"
            f"def f_{index}():\n    return f_{following}()\n"
            if cyclic or index + 1 < size
            else f"def f_{index}():\n    return 1\n"
        )
    return snap({}, sources)


class GraphOrderingTests(unittest.TestCase):
    """Check SCC membership, deterministic audits, and compiler priority rules."""

    def test_all_three_node_graphs_match_reachability(self) -> None:
        """Check every directed graph, including self edges and nested cycles."""
        nodes = ("a", "b", "c")
        possible_edges = tuple(product(nodes, repeat=2))
        for mask in range(1 << len(possible_edges)):
            with self.subTest(mask=mask):
                prereqs = {node: set() for node in nodes}
                for bit, (node, prerequisite) in enumerate(possible_edges):
                    if mask & (1 << bit):
                        prereqs[node].add(prerequisite)
                reachable = {node: {node, *prereqs[node]} for node in nodes}
                for intermediate in nodes:
                    for node in nodes:
                        if intermediate in reachable[node]:
                            reachable[node].update(reachable[intermediate])
                components = {
                    tuple(
                        other
                        for other in nodes
                        if other in reachable[node]
                        and node in reachable[other]
                    )
                    for node in nodes
                }
                order, cycles = ordered_components(nodes, prereqs, str)
                self.assertEqual(Counter(order), Counter(nodes))
                self.assertEqual(
                    cycles,
                    tuple(sorted(c for c in components if len(c) > 1)),
                )
                position = {node: i for i, node in enumerate(order)}
                for component in components:
                    positions = [position[node] for node in component]
                    self.assertEqual(
                        max(positions) - min(positions) + 1, len(component)
                    )
                    for node in component:
                        for prerequisite in prereqs[node] - set(component):
                            self.assertLess(
                                position[prerequisite], position[node]
                            )

    def test_ready_components_and_cycle_members_use_caller_priority(
        self,
    ) -> None:
        """A newly available component can outrank one already in the queue."""
        priorities = {"a": 9, "b": 1, "c": 8, "d": 0, "e": 4, "f": 3}
        order, cycles = ordered_components(
            reversed(priorities),
            {"b": {"c"}, "c": {"b"}, "d": {"b"}, "f": {"e"}},
            priorities.__getitem__,
        )
        self.assertEqual(order, ("b", "c", "d", "e", "f", "a"))
        self.assertEqual(cycles, (("b", "c"),))

    def test_shuffled_nodes_and_adjacencies_keep_equal_priority_order(
        self,
    ) -> None:
        """Equal priorities and multiple cycles do not inherit traversal order."""
        graph = {
            "a": {"b", "c"},
            "b": {"a", "c"},
            "c": set(),
            "d": {"e"},
            "e": {"d"},
        }
        expected = (("c", "a", "b", "d", "e"), (("a", "b"), ("d", "e")))
        for nodes in permutations(graph):
            for reverse in (False, True):
                # dict_keys is set-like but lets this test vary adjacency order.
                prereqs = {
                    node: dict.fromkeys(
                        sorted(graph[node], reverse=reverse)
                    ).keys()
                    for node in nodes
                }
                self.assertEqual(
                    ordered_components(iter(nodes), prereqs, lambda _: 0),
                    expected,
                )

    def test_empty_duplicate_and_reachable_nodes(self) -> None:
        """Retain prerequisite closure and omit singleton self loops from cycles."""
        self.assertEqual(ordered_components([], {}, str), ((), ()))
        self.assertEqual(
            ordered_components(
                iter(("b", "b")), {"b": {"a"}, "a": {"a", "c"}}, str
            ),
            (("c", "a", "b"), ()),
        )

    def test_public_report_and_graph_audits_are_hash_seed_independent(
        self,
    ) -> None:
        """Subprocess hash seeds cannot change report order or SCC audit arrays."""
        probe = """
import hashlib
import json
from diffstory.analysis import compile_snapshot, ordered_components
from test_analysis import snap

nodes = set('abcdefg')
prereqs = {node: set() for node in nodes}
for node, prerequisite in {('a', 'b'), ('b', 'a'), ('c', 'd'), ('d', 'c'),
                            ('e', 'a'), ('e', 'c'), ('g', 'f')}:
    prereqs[node].add(prerequisite)
sources = {
    f'{node}.py': ''.join(f'from {p} import f_{p}\\n' for p in sorted(prereqs[node]))
    + f'def f_{node}():\\n    return '
    + (' + '.join(f'f_{p}()' for p in sorted(prereqs[node])) or '1') + '\\n'
    for node in sorted(nodes)
}
payload = {
    'report': compile_snapshot(snap({}, sources)),
    'graph': ordered_components(nodes, prereqs, lambda _: 0),
}
print(hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest())
"""
        digests = []
        for seed in ("1", "7", "101", "999"):
            result = subprocess.run(
                [sys.executable, "-c", probe],
                cwd=Path(__file__).resolve().parents[1],
                env={
                    **os.environ,
                    "PYTHONPATH": str(Path(__file__).resolve().parent),
                    "PYTHONHASHSEED": seed,
                },
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            )
            digests.append(result.stdout.strip())
        self.assertRegex(digests[0], r"^[0-9a-f]{64}$")
        self.assertEqual(len(set(digests)), 1)


class CompilerGraphTests(unittest.TestCase):
    """Exercise deep graphs through the compiler, annotation, and reader APIs."""

    def check_compiled_graph(self, *, cyclic: bool) -> None:
        """Require exact source changes, graph membership, edges, and ordering."""
        recursion_limit = sys.getrecursionlimit()
        size = max(1050, recursion_limit + 50)
        report = compile_snapshot(graph_snapshot(size, cyclic=cyclic))
        edge_count = size if cyclic else size - 1
        expected_paths = {f"m_{index:04d}.py" for index in range(size)}
        groups = report["groups"]
        by_path = {group["path"]: group for group in groups}
        self.assertEqual(len(groups), size)
        self.assertEqual(by_path.keys(), expected_paths)
        changes = report["changes"]
        self.assertEqual(
            Counter(change["kind"] for change in changes),
            {"added": size, "wiring": edge_count},
        )
        functions = {
            change["after"]["name"]: change["after"]["id"]
            for change in changes
            if change["kind"] == "added"
        }
        self.assertEqual(functions.keys(), {f"f_{i}" for i in range(size)})
        members = {path: [] for path in expected_paths}
        for change in changes:
            members[change["after"]["path"]].append(change["id"])
        self.assertEqual(
            Counter(cid for group in groups for cid in group["change_ids"]),
            Counter(c["id"] for c in changes),
        )
        for path, group in by_path.items():
            self.assertEqual(set(group["change_ids"]), set(members[path]))
        expected_symbol_edges = set()
        expected_group_edges = set()
        for index in range(size):
            group = by_path[f"m_{index:04d}.py"]
            if not cyclic and index == size - 1:
                self.assertEqual(group["prerequisites"], [])
                continue
            following = (index + 1) % size
            prerequisite = by_path[f"m_{following:04d}.py"]["id"]
            self.assertEqual(group["prerequisites"], [prerequisite])
            expected_symbol_edges.add(
                (functions[f"f_{index}"], functions[f"f_{following}"], "calls")
            )
            expected_group_edges.add((prerequisite, group["id"], "calls"))
        for field, expected in (
            ("symbol_edges", expected_symbol_edges),
            ("edges", expected_group_edges),
        ):
            self.assertEqual(len(report[field]), edge_count)
            self.assertEqual(
                {(e["from"], e["to"], e["type"]) for e in report[field]},
                expected,
            )
        self.assertEqual(
            [group["path"] for group in groups],
            sorted(expected_paths, reverse=not cyclic),
        )
        self.assertEqual(
            report["cycles"],
            (tuple(sorted(g["id"] for g in groups)),) if cyclic else (),
        )
        self.assertEqual(report["tests"], [])
        self.assertEqual(report["stats"]["test_runs"], 0)
        annotated = apply_annotations(
            report,
            {
                "schema": "diffstory.annotations.v1",
                "base_sha": report["meta"]["base_sha"],
                "head_sha": report["meta"]["head_sha"],
                "steps": [
                    {
                        "group_id": group["id"],
                        "evidence_change_ids": group["change_ids"],
                        "passages": [
                            {
                                "text": "Follow this function and its import.",
                                "change_ids": group["change_ids"],
                                "view": "definition",
                            }
                        ],
                    }
                    for group in groups
                ],
            },
        )
        for field in (
            "changes",
            "edges",
            "symbol_edges",
            "cycles",
            "source_evidence",
            "tests",
        ):
            self.assertEqual(annotated[field], report[field])
        for index, group in enumerate(annotated["groups"]):
            self.assertEqual(group["number"], index + 1)
            self.assertEqual(
                group["next_id"],
                groups[index + 1]["id"] if index + 1 < size else None,
            )
        self.assertIn(
            "Follow this function and its import.",
            render(json.loads(json.dumps(annotated))),
        )
        self.assertEqual(sys.getrecursionlimit(), recursion_limit)

    def test_public_chain_above_recursion_limit(self) -> None:
        """Compile, annotate, and render a chain longer than the call stack."""
        self.check_compiled_graph(cyclic=False)

    def test_public_cycle_above_recursion_limit(self) -> None:
        """Retain the entire large function SCC through public consumers."""
        self.check_compiled_graph(cyclic=True)


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
                overrides={
                    ("base", "a.py"): {
                        "state": "unavailable",
                        "reason": "not_supplied",
                    },
                    ("head", "a.py"): {
                        "state": "supplied",
                        "coverage": "partial",
                    },
                },
                scope="selected excerpts",
            )
        )
        self.assertEqual(symbols(r)[0]["kind"], "observed_head")

    def test_unrelated_input_preserves_kind(self) -> None:
        """A limit or warning in another file must not change a full-file result."""
        rows = [
            ("addition", False, "added"),
            ("removal", True, "removed"),
        ]
        unrelated = [
            ("none", {}, []),
            (
                "markdown",
                {"README.md": ("old text\n", "new text\n")},
                [],
            ),
            (
                "build file",
                {"BUILD.bazel": ("old rule\n", "new rule\n")},
                [],
            ),
            (
                "parse failure",
                {"broken.py": ("def broken(:\n", "def broken(:\n")},
                [],
            ),
            ("incoming warning", {}, ["Unrelated source was skipped."]),
        ]
        for direction, is_removal, expected in rows:
            for extra_name, extra_files, warning_list in unrelated:
                with self.subTest(direction=direction, unrelated=extra_name):
                    base = "def keep():\n    return 1\n"
                    head = base
                    if is_removal:
                        base += "\ndef target():\n    return 2\n"
                    else:
                        head += "\ndef target():\n    return 2\n"
                    before = {"config.py": base}
                    after = {"config.py": head}
                    for path, (old_text, new_text) in extra_files.items():
                        before[path] = old_text
                        after[path] = new_text
                    report = compile_snapshot(
                        snap(before, after, warnings=warning_list),
                    )
                    target = next(
                        change
                        for change in symbols(report)
                        if (change.get("after") or change.get("before"))[
                            "name"
                        ]
                        == "target"
                    )
                    self.assertEqual(target["kind"], expected)

    def test_relevant_limits_leave_kind_observed(self) -> None:
        """Missing, partial, or unparsed counterpart evidence must stay observed."""
        own_text = "def keep():\n    return 1\n\ndef target():\n    return 2\n"
        other_text = "def keep():\n    return 1\n"
        cases = [
            (
                "missing",
                None,
                {"state": "unavailable", "reason": "not_supplied"},
                {},
            ),
            (
                "unavailable",
                None,
                {"state": "unavailable", "reason": "size_limit"},
                {},
            ),
            (
                "partial",
                other_text,
                {"state": "supplied", "coverage": "partial"},
                {},
            ),
            (
                "line 20",
                other_text,
                {"state": "supplied", "coverage": "partial"},
                {"start_line": 20},
            ),
            (
                "parse failure",
                "def broken(:\n",
                {"state": "supplied", "coverage": "full"},
                {},
            ),
        ]
        for expected, own_side, other_side in (
            ("observed_head", "head", "base"),
            ("observed_base", "base", "head"),
        ):
            for name, counterpart_text, evidence, options in cases:
                with self.subTest(case=name, expected=expected):
                    sources = {own_side: {"x.py": own_text}, other_side: {}}
                    if counterpart_text is not None:
                        sources[other_side] = {"x.py": counterpart_text}
                    snapshot = snap(
                        sources["base"],
                        sources["head"],
                        overrides={(other_side, "x.py"): evidence},
                        fragment_options={(other_side, "x.py"): options},
                    )
                    report = compile_snapshot(snapshot)
                    self.assertEqual(symbols(report)[0]["kind"], expected)

    def test_empty_python_is_full_parsed(self) -> None:
        """Empty Python files must support confirmed additions and removals."""
        rows = [
            (
                {"empty.py": ""},
                {"empty.py": "def f():\n    return 1\n"},
                "added",
            ),
            (
                {"empty.py": "def f():\n    return 1\n"},
                {"empty.py": ""},
                "removed",
            ),
        ]
        for before, after, expected in rows:
            with self.subTest(expected=expected):
                self.assertEqual(
                    symbols(compile_snapshot(snap(before, after)))[0]["kind"],
                    expected,
                )

    def test_explicit_absence_confirms_kind(self) -> None:
        """Explicit absent file sides must confirm whole-file additions and removals."""
        source = "def f():\n    return 1\n"
        rows = [
            ({}, {"new.py": source}, "added"),
            ({"old.py": source}, {}, "removed"),
        ]
        for before, after, expected in rows:
            with self.subTest(expected=expected):
                change = symbols(compile_snapshot(snap(before, after)))[0]
                self.assertEqual(change["kind"], expected)
                self.assertIn("confirmed absent", change["basis"])

    def test_file_evidence_is_required(self) -> None:
        """Reject missing evidence instead of guessing file pairs or source states."""
        for name, value in (("omitted", None), ("empty", [])):
            with self.subTest(case=name):
                snapshot = snap({}, {"x.py": "def f():\n    return 1\n"})
                if value is None:
                    snapshot.pop("file_evidence")
                else:
                    snapshot["file_evidence"] = value
                with self.assertRaisesRegex(ValueError, "[Ff]ile.evidence"):
                    compile_snapshot(snapshot)

    def test_repeated_declaration_blocks_confirmation(self) -> None:
        """A remaining same-name declaration blocks unmatched confirmation after matching."""
        one = "def f():\n    return 1\n"
        duplicate = one + "\n" + one
        rows = [
            ({"x.py": duplicate}, {"x.py": one}, "observed_base"),
            ({"x.py": one}, {"x.py": duplicate}, "observed_head"),
        ]
        for before, after, expected in rows:
            with self.subTest(expected=expected):
                report = compile_snapshot(snap(before, after))
                change = next(
                    item
                    for item in symbols(report)
                    if item["kind"] == expected
                )
                self.assertEqual(change["kind"], expected)
                self.assertIn("same name and type remains", change["basis"])

    def test_partial_parse_failure_keeps_valid_symbols(self) -> None:
        """One failed excerpt must not discard symbols from another excerpt."""
        snapshot = {
            "patches": [{"path": "x.py", "origin": "github", "patch": None}],
            "schema": "diffstory.snapshot.v1",
            "meta": {"scope": "selected excerpts", "changed_files": 1},
            "fragments": [
                {
                    "path": "x.py",
                    "side": "head",
                    "start_line": 1,
                    "scope": "complete",
                    "region": "part-a",
                    "text": "def valid():\n    return 1\n",
                },
                {
                    "path": "x.py",
                    "side": "head",
                    "start_line": 10,
                    "scope": "complete",
                    "region": "part-b",
                    "text": "def broken(:\n",
                },
            ],
            "file_evidence": [
                {
                    "base": {"path": "x.py", "state": "absent"},
                    "head": {
                        "path": "x.py",
                        "state": "supplied",
                        "coverage": "partial",
                    },
                },
            ],
        }
        report = compile_snapshot(snapshot)
        change = next(
            item
            for item in symbols(report)
            if (item.get("after") or {}).get("name") == "valid"
        )
        self.assertEqual(change["kind"], "observed_head")
        self.assertIn(
            "Only part of the head file is supplied", change["basis"]
        )
        self.assertIn(
            "head Python source could not be parsed", change["basis"]
        )
        self.assertTrue(
            any("AST unavailable" in warning for warning in report["warnings"])
        )

    def test_evidence_shape_is_strict(self) -> None:
        """Malformed v1 file evidence must fail instead of being repaired."""
        valid = snap({}, {"x.py": "def f():\n    return 1\n"})
        bad_values = [
            ("not a list", {"base": valid["file_evidence"][0]["base"]}),
            ("record fields", [{**valid["file_evidence"][0], "extra": 1}]),
            (
                "side object",
                [{"base": [], "head": valid["file_evidence"][0]["head"]}],
            ),
            (
                "side fields",
                [
                    {
                        "base": {
                            **valid["file_evidence"][0]["base"],
                            "reason": "not_supplied",
                        },
                        "head": valid["file_evidence"][0]["head"],
                    }
                ],
            ),
            (
                "unknown state",
                [
                    {
                        "base": {"path": "x.py", "state": "maybe"},
                        "head": valid["file_evidence"][0]["head"],
                    }
                ],
            ),
            (
                "unknown coverage",
                [
                    {
                        "base": valid["file_evidence"][0]["base"],
                        "head": {
                            "path": "x.py",
                            "state": "supplied",
                            "coverage": "some",
                        },
                    }
                ],
            ),
            (
                "missing reason",
                [
                    {
                        "base": {"path": "x.py", "state": "unavailable"},
                        "head": valid["file_evidence"][0]["head"],
                    }
                ],
            ),
            (
                "unknown reason",
                [
                    {
                        "base": {
                            "path": "x.py",
                            "state": "unavailable",
                            "reason": "weird",
                        },
                        "head": valid["file_evidence"][0]["head"],
                    }
                ],
            ),
        ]
        for name, invalid in bad_values:
            with self.subTest(case=name):
                snapshot = copy.deepcopy(valid)
                snapshot["file_evidence"] = invalid
                with self.assertRaises(ValueError):
                    compile_snapshot(snapshot)

    def test_evidence_paths_are_relative(self) -> None:
        """Evidence paths must not escape the repository or name its root."""
        valid = snap({}, {"x.py": "def f():\n    return 1\n"})
        paths = [
            "",
            "/x.py",
            "\\root.py",
            "a\0b.py",
            "a/./b.py",
            "a/../b.py",
            "C:/x.py",
        ]
        for path in paths:
            with self.subTest(path=path):
                snapshot = copy.deepcopy(valid)
                snapshot["file_evidence"][0]["base"]["path"] = path
                with self.assertRaisesRegex(ValueError, "repository-relative"):
                    compile_snapshot(snapshot)
        self.assertGreater(compile_snapshot(valid)["stats"]["units"], 0)

    def test_validation_errors_do_not_echo_source(self) -> None:
        """Malformed snapshot errors must not expose source or producer metadata."""
        marker = "private fixture content must stay out of diagnostics"
        cases = [
            ("region", {"region": None}),
            ("line number", {"start_line": True}),
            ("revision", {"side": "other"}),
        ]
        for name, fields in cases:
            with self.subTest(case=name):
                snapshot = snap({}, {"x.py": marker}, description=marker)
                snapshot["fragments"][0].update(fields)
                with self.assertRaises(ValueError) as caught:
                    compile_snapshot(snapshot)
                self.assertNotIn(marker, str(caught.exception))

    def test_evidence_side_is_unique(self) -> None:
        """One side/path pair cannot belong to more than one evidence record."""
        snapshot = snap({}, {"x.py": "def f():\n    return 1\n"})
        snapshot["file_evidence"].append(
            {
                "base": {
                    "path": "x.py",
                    "state": "unavailable",
                    "reason": "not_supplied",
                },
                "head": {
                    "path": "other.py",
                    "state": "unavailable",
                    "reason": "not_supplied",
                },
            },
        )
        with self.assertRaisesRegex(ValueError, "must be unique"):
            compile_snapshot(snapshot)

    def test_absence_record_is_valid(self) -> None:
        """Both absent is invalid, while two unavailable sides remain unresolved."""
        snapshot = snap(
            {"a.py": "value = 1\n"},
            {"a.py": "value = 2\n"},
        )
        accepted = copy.deepcopy(snapshot)
        accepted["file_evidence"].append(
            {
                "base": {
                    "path": "not-read.py",
                    "state": "unavailable",
                    "reason": "read_failed",
                },
                "head": {
                    "path": "not-read.py",
                    "state": "unavailable",
                    "reason": "read_failed",
                },
            },
        )
        accepted["patches"].append(
            {"path": "not-read.py", "origin": "github", "patch": None}
        )
        rejected = copy.deepcopy(snapshot)
        rejected["file_evidence"].append(
            {
                "base": {"path": "gone.py", "state": "absent"},
                "head": {"path": "gone.py", "state": "absent"},
            },
        )
        rows = [
            ("two unavailable", accepted, True),
            ("two absent", rejected, False),
        ]
        for name, candidate, valid in rows:
            with self.subTest(case=name):
                if valid:
                    compile_snapshot(candidate)
                else:
                    with self.assertRaisesRegex(ValueError, "both sides"):
                        compile_snapshot(candidate)

    def test_empty_snapshot_rule_is_preserved(self) -> None:
        """Evidence records alone must not replace the declared empty comparison rule."""
        empty = {
            "patches": [],
            "schema": "diffstory.snapshot.v1",
            "meta": {"changed_files": 0},
            "fragments": [],
            "file_evidence": [],
        }
        rows = [
            ("declared empty", empty, True),
            (
                "undeclared empty",
                {**empty, "meta": {"changed_files": 1}},
                False,
            ),
        ]
        for name, snapshot, valid in rows:
            with self.subTest(case=name):
                if valid:
                    compile_snapshot(snapshot)
                else:
                    with self.assertRaisesRegex(
                        ValueError, "no source fragments"
                    ):
                        compile_snapshot(snapshot)

    def test_evidence_covers_fragments(self) -> None:
        """Every fragment must belong to one supplied side and empty source counts."""
        valid = snap({"x.py": ""}, {"x.py": ""})
        uncovered = copy.deepcopy(valid)
        uncovered["fragments"].append(
            {"path": "extra.py", "side": "head", "text": ""},
        )
        absent_with_source = copy.deepcopy(valid)
        absent_with_source["file_evidence"][0]["base"] = {
            "path": "x.py",
            "state": "absent",
        }
        no_supplied_fragment = snap({}, {"x.py": ""})
        no_supplied_fragment["file_evidence"][0]["base"] = {
            "path": "x.py",
            "state": "supplied",
            "coverage": "full",
        }
        rows = [
            ("uncovered fragment", uncovered, False),
            ("absent side has source", absent_with_source, False),
            ("supplied side has no fragment", no_supplied_fragment, False),
            ("empty supplied source", valid, True),
        ]
        for name, snapshot, accepted in rows:
            with self.subTest(case=name):
                if accepted:
                    compile_snapshot(snapshot)
                else:
                    with self.assertRaises(ValueError):
                        compile_snapshot(snapshot)

    def test_full_coverage_claim_is_valid(self) -> None:
        """Full claims require one line-one fragment and permit documented scopes."""
        source = "def f():\n    return 1\n"
        rows = [
            ("multiple fragments", "full", 1, 2, False),
            ("non-one start", "full", 2, None, False),
            ("boolean start", "full", True, None, False),
            ("excerpt scope", "selected excerpts", 1, None, False),
            ("omitted scope", None, 1, None, True),
            ("full scope", "full", 1, None, True),
            ("complete scope", "complete", 1, None, True),
        ]
        for name, scope, start, second_start, accepted in rows:
            with self.subTest(case=name):
                snapshot = snap(
                    {"x.py": source},
                    {"x.py": source},
                    fragment_options={
                        ("base", "x.py"): {
                            "start_line": start,
                            **({} if scope is None else {"scope": scope}),
                        },
                        ("head", "x.py"): {
                            "start_line": 1,
                            **({} if scope is None else {"scope": scope}),
                        },
                    },
                )
                if second_start is not None:
                    snapshot["fragments"].append(
                        {
                            "path": "x.py",
                            "side": "head",
                            "text": "def g():\n    return 2\n",
                            "start_line": second_start,
                            "scope": "complete",
                            "region": "part-b",
                        },
                    )
                if accepted:
                    compile_snapshot(snapshot)
                else:
                    with self.assertRaises(ValueError):
                        compile_snapshot(snapshot)

    def test_partial_ranges_do_not_overlap(self) -> None:
        """Adjacent excerpt ranges remain partial while overlapping ranges fail."""
        rows = [("adjacent", 3, True), ("overlap", 2, False)]
        for name, second_start, accepted in rows:
            with self.subTest(case=name):
                snapshot = {
                    "patches": [
                        {"path": "x.py", "origin": "github", "patch": None}
                    ],
                    "schema": "diffstory.snapshot.v1",
                    "meta": {"changed_files": 1},
                    "fragments": [
                        {
                            "path": "x.py",
                            "side": "base",
                            "text": "def f():\n    return 1\n",
                            "start_line": 1,
                            "region": "one",
                        },
                        {
                            "path": "x.py",
                            "side": "base",
                            "text": "def g():\n    return 2\n",
                            "start_line": second_start,
                            "region": "two",
                        },
                    ],
                    "file_evidence": [
                        {
                            "base": {
                                "path": "x.py",
                                "state": "supplied",
                                "coverage": "partial",
                            },
                            "head": {
                                "path": "x.py",
                                "state": "unavailable",
                                "reason": "not_supplied",
                            },
                        },
                    ],
                }
                if accepted:
                    compile_snapshot(snapshot)
                else:
                    with self.assertRaisesRegex(ValueError, "Overlapping"):
                        compile_snapshot(snapshot)

    def test_empty_excerpts_do_not_reserve_lines(self) -> None:
        """Empty text must not block source at the same offset; a blank line must."""
        cases = [
            ("empty first", "", False, True),
            ("empty last", "", True, True),
            ("blank line first", "\n", False, False),
            ("blank line last", "\n", True, False),
        ]
        for name, text, reverse, accepted in cases:
            with self.subTest(case=name):
                snapshot = snap(
                    {"x.py": "def target():\n    return 1\n"},
                    {},
                    overrides={
                        ("base", "x.py"): {
                            "path": "x.py",
                            "state": "supplied",
                            "coverage": "partial",
                        },
                    },
                    fragment_options={("base", "x.py"): {"region": "source"}},
                )
                empty = {
                    "path": "x.py",
                    "side": "base",
                    "start_line": 1,
                    "text": text,
                    "region": "empty",
                }
                snapshot["fragments"].insert(0, empty)
                if reverse:
                    snapshot["fragments"].reverse()
                if accepted:
                    result = compile_snapshot(snapshot)
                    self.assertEqual(
                        symbols(result)[0]["before"]["name"], "target"
                    )
                else:
                    with self.assertRaisesRegex(ValueError, "Overlapping"):
                        compile_snapshot(snapshot)

    def test_region_side_is_unique(self) -> None:
        """A raw region cannot overwrite another fragment on the same side."""
        snapshot = snap(
            {"x.py": "def f():\n    return 1\n"},
            {},
            overrides={
                ("base", "x.py"): {
                    "path": "x.py",
                    "state": "supplied",
                    "coverage": "partial",
                },
            },
            fragment_options={("base", "x.py"): {"region": "same"}},
        )
        snapshot["fragments"].append(
            {
                "path": "x.py",
                "side": "base",
                "start_line": 10,
                "region": "same",
                "text": "value = 2\n",
            },
        )
        with self.assertRaisesRegex(ValueError, "Duplicate raw region"):
            compile_snapshot(snapshot)

    def test_region_links_stay_within_record(self) -> None:
        """Explicit raw regions must agree with their file record and side pair."""
        mismatched = snap(
            {"x.py": "value = 1\n"},
            {"x.py": "value = 2\n"},
            fragment_options={
                ("base", "x.py"): {"region": "base-region"},
                ("head", "x.py"): {"region": "head-region"},
            },
        )
        cross_record = snap(
            {"old.py": "value = 1\n"},
            {"new.py": "value = 2\n"},
            fragment_options={
                ("base", "old.py"): {"region": "shared"},
                ("head", "new.py"): {"region": "shared"},
            },
        )
        rows = [
            ("full sides disagree", mismatched, "share a region"),
            (
                "region crosses records",
                cross_record,
                "different file evidence records",
            ),
        ]
        for name, snapshot, reason in rows:
            with (
                self.subTest(case=name),
                self.assertRaisesRegex(ValueError, reason),
            ):
                compile_snapshot(snapshot)

    def test_raw_rows_survive_classification(self) -> None:
        """Confirmed additions and removals must retain their original raw diff rows."""
        snapshot = snap(
            {"config.py": "def removed():\n    return 1\n"},
            {"config.py": "added = 2\n"},
        )
        report = compile_snapshot(snapshot)
        self.assertEqual(
            {item["kind"] for item in symbols(report)},
            {"added", "removed"},
        )
        raw = next(
            item for item in report["raw_files"] if item["path"] == "config.py"
        )
        tags = {row["tag"] for hunk in raw["hunks"] for row in hunk["rows"]}
        self.assertEqual(tags, {"add", "delete"})

    def test_partial_regions_keep_source_without_inventing_a_patch(
        self,
    ) -> None:
        """Partial source stays available even when no original patch was supplied."""
        snapshot = {
            "patches": [{"path": "x.py", "origin": "github", "patch": None}],
            "schema": "diffstory.snapshot.v1",
            "meta": {"changed_files": 1, "scope": "selected excerpts"},
            "fragments": [
                {
                    "path": "x.py",
                    "side": "base",
                    "start_line": 1,
                    "region": "first",
                    "text": "value = 1\n",
                },
                {
                    "path": "x.py",
                    "side": "head",
                    "start_line": 1,
                    "region": "first",
                    "text": "value = 2\n",
                },
                {
                    "path": "x.py",
                    "side": "base",
                    "start_line": 10,
                    "region": "second",
                    "text": "other = 3\n",
                },
                {
                    "path": "x.py",
                    "side": "head",
                    "start_line": 10,
                    "region": "second",
                    "text": "other = 4\n",
                },
            ],
            "file_evidence": [
                {
                    "base": {
                        "path": "x.py",
                        "state": "supplied",
                        "coverage": "partial",
                    },
                    "head": {
                        "path": "x.py",
                        "state": "supplied",
                        "coverage": "partial",
                    },
                },
            ],
        }
        report = compile_snapshot(snapshot)
        self.assertEqual(
            {file["region"] for file in report["raw_files"]},
            {"first", "second"},
        )
        self.assertEqual(
            sum(
                row["tag"] != "context"
                for file in report["raw_files"]
                for hunk in file["hunks"]
                for row in hunk["rows"]
            ),
            0,
        )
        self.assertEqual(
            [
                fragment["text"]
                for fragment in report["source_evidence"]["fragments"]
            ],
            [fragment["text"] for fragment in snapshot["fragments"]],
        )

    def test_basis_reasons_are_deterministic(self) -> None:
        """Basis text must list base then head blockers and both linked paths."""
        snapshot = {
            "patches": [{"path": "new.py", "origin": "github", "patch": None}],
            "schema": "diffstory.snapshot.v1",
            "meta": {"changed_files": 1},
            "fragments": [
                {
                    "path": "old.py",
                    "side": "base",
                    "start_line": 1,
                    "region": "rename",
                    "text": "def f():\n    return 1\n",
                },
            ],
            "file_evidence": [
                {
                    "base": {
                        "path": "old.py",
                        "state": "supplied",
                        "coverage": "partial",
                    },
                    "head": {
                        "path": "new.py",
                        "state": "unavailable",
                        "reason": "size_limit",
                    },
                },
            ],
        }
        first = symbols(compile_snapshot(snapshot))[0]["basis"]
        second = symbols(compile_snapshot(snapshot))[0]["basis"]
        self.assertEqual(first, second)
        self.assertLess(
            first.index("part of the base"),
            first.index("head source is unavailable"),
        )
        self.assertIn("Base path: old.py. Head path: new.py.", first)

    def test_evidence_packet_includes_notes(self) -> None:
        """Evidence packets must retain informational notes from current reports."""
        cases = [
            ({"README.md": "Read this.\n"}, True),
            ({"x.py": "value = 1\n"}, False),
        ]
        for sources, has_notes in cases:
            with self.subTest(paths=list(sources)):
                report = compile_snapshot(snap({}, sources))
                packet = evidence_packet(report)
                self.assertEqual(packet["notes"], report["notes"])
                self.assertEqual(bool(packet["notes"]), has_notes)

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
        for side in s["file_evidence"][0].values():
            side["coverage"] = "partial"
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


class SourceEvidenceTests(unittest.TestCase):
    """Check original source bindings, validated states, and raw text ownership."""

    def test_projection_preserves_original_buffers_and_public_bindings(
        self,
    ) -> None:
        """Reconstruct partial UTF-8 excerpts without inventing full-file bytes."""
        snapshot = snap(
            {"x.py": '# café\r\ndef f():\r\n    return "é"\r\n'},
            {"x.py": '# changé\r\ndef f():\r\n    return "λ"'},
            overrides={
                (side, "x.py"): {"state": "supplied", "coverage": "partial"}
                for side in ("base", "head")
            },
            fragment_options={
                (side, "x.py"): {
                    "start_line": 51,
                    "region": "excerpt",
                    "scope": "selected excerpts",
                    "producer_secret": "must not escape",
                    "id": "spoofed",
                    "_raw_region": "spoofed",
                    "parse_status": "spoofed",
                    "content_digest": "spoofed",
                }
                for side in ("base", "head")
            },
        )
        original = copy.deepcopy(snapshot)
        report = compile_snapshot(snapshot)
        self.assertEqual(snapshot, original)
        projection = report["source_evidence"]
        self.assertEqual(
            set(projection),
            {"schema", "file_evidence", "fragments"},
        )
        self.assertEqual(projection["schema"], "diffstory.source-evidence.v1")
        self.assertEqual(json.loads(json.dumps(projection)), projection)
        self.assertNotIn("must not escape", json.dumps(projection))
        fragments = {item["key"]: item for item in projection["fragments"]}
        self.assertEqual(len(fragments), 2)
        for supplied, fragment in zip(
            snapshot["fragments"],
            projection["fragments"],
            strict=True,
        ):
            self.assertEqual(
                set(fragment),
                {
                    "key",
                    "side",
                    "path",
                    "start_line",
                    "region",
                    "scope",
                    "text",
                    "content_digest",
                    "parse_status",
                },
            )
            self.assertRegex(fragment["key"], r"^[0-9a-f]{16}$")
            raw = supplied["text"].encode("utf-8")
            self.assertEqual(fragment["text"].encode("utf-8"), raw)
            self.assertEqual(
                fragment["content_digest"], hashlib.sha256(raw).hexdigest()
            )
            self.assertEqual(fragment["parse_status"], "ok")
            self.assertEqual(fragment["region"], "excerpt")
            self.assertEqual(fragment["scope"], "selected excerpts")
            self.assertEqual(fragment["start_line"], 51)
            # These offsets are relative to this fragment, even at file line 51.
            encoded_lines = [
                line.encode("utf-8")
                for line in fragment["text"].splitlines(keepends=True)
            ]
            offset = 0
            recovered = []
            for line in encoded_lines:
                stop = offset + len(line)
                recovered.append(raw[offset:stop])
                offset = stop
            self.assertEqual(b"".join(recovered), raw)
            self.assertEqual(offset, len(raw))
        for change in report["changes"]:
            for side, name in (("base", "before"), ("head", "after")):
                record = change[name]
                self.assertIsNotNone(record)
                fragment = fragments[record["fragment_key"]]
                self.assertEqual(fragment["side"], side)
                self.assertEqual(fragment["path"], record["path"])
                start = record["start"] - fragment["start_line"]
                stop = record["end"] - fragment["start_line"] + 1
                recovered = "".join(
                    fragment["text"].splitlines(keepends=True)[start:stop]
                )
                self.assertEqual(
                    recovered.splitlines(), record["source"].splitlines()
                )
                self.assertNotIn("fragment_id", record)
                self.assertNotIn("fingerprint", record)
        for state in projection["file_evidence"][0].values():
            self.assertEqual(
                set(state),
                {
                    "path",
                    "state",
                    "coverage",
                    "fragment_keys",
                    "parse_statuses",
                    "full_source",
                    "full_and_parsed",
                },
            )
            self.assertEqual(state["coverage"], "partial")
            self.assertEqual(state["parse_statuses"], ["ok"])
            self.assertFalse(state["full_source"])
            self.assertFalse(state["full_and_parsed"])
            self.assertEqual(len(state["fragment_keys"]), 1)
            self.assertIn(state["fragment_keys"][0], fragments)
        render(report)

    def test_empty_absent_and_unavailable_are_distinct(self) -> None:
        """Keep empty supplied buffers even when no lines or changes exist."""
        for state in (
            {"state": "supplied", "coverage": "full"},
            {"state": "absent"},
            {"state": "unavailable", "reason": "read_failed"},
        ):
            with self.subTest(state=state["state"]):
                snapshot = snap(
                    {"empty.py": ""} if state["state"] == "supplied" else {},
                    {"empty.py": ""},
                    overrides={("base", "empty.py"): state},
                )
                report = compile_snapshot(snapshot)
                evidence = report["source_evidence"]
                pair = evidence["file_evidence"][0]
                self.assertEqual(pair["base"]["state"], state["state"])
                self.assertEqual(pair["head"]["state"], "supplied")
                self.assertTrue(pair["head"]["full_and_parsed"])
                self.assertEqual(pair["head"]["parse_statuses"], ["ok"])
                self.assertEqual(report["changes"], [])
                self.assertEqual(report["raw_files"], [])
                if state["state"] != "supplied":
                    self.assertEqual(pair["base"]["fragment_keys"], [])
                    self.assertEqual(pair["base"]["parse_statuses"], [])
                    self.assertFalse(pair["base"]["full_source"])
                    self.assertFalse(pair["base"]["full_and_parsed"])
                if state["state"] == "unavailable":
                    self.assertEqual(pair["base"]["reason"], "read_failed")
                for fragment in evidence["fragments"]:
                    self.assertEqual(fragment["text"], "")
                    self.assertEqual(
                        fragment["content_digest"],
                        hashlib.sha256(b"").hexdigest(),
                    )
                    self.assertIsNone(fragment["scope"])

    def test_parse_states_are_compiler_owned(self) -> None:
        """Expose actual parse results despite false producer metadata or warnings."""
        snapshot = snap(
            {"x.py": "def broken(:\n", "notes.txt": "old\r\n"},
            {"x.py": "value = 1\n", "notes.txt": "new\r\n"},
            fragment_options={
                ("base", "x.py"): {
                    "parse_status": "ok",
                    "full_and_parsed": True,
                },
                ("head", "x.py"): {"parse_status": "failed"},
            },
            warnings=["All base definitions are confirmed absent."],
        )
        snapshot["file_evidence"].append(
            {
                side: {
                    "path": "unread.py",
                    "state": "unavailable",
                    "reason": "size_limit",
                }
                for side in ("base", "head")
            }
        )
        snapshot["patches"].append(
            {"path": "unread.py", "origin": "github", "patch": None}
        )
        projection = compile_snapshot(snapshot)["source_evidence"]
        pairs = {
            item["base"]["path"]: item for item in projection["file_evidence"]
        }
        self.assertEqual(pairs["x.py"]["base"]["parse_statuses"], ["failed"])
        self.assertFalse(pairs["x.py"]["base"]["full_and_parsed"])
        self.assertTrue(pairs["x.py"]["base"]["full_source"])
        self.assertTrue(pairs["x.py"]["head"]["full_and_parsed"])
        self.assertEqual(
            pairs["notes.txt"]["head"]["parse_statuses"], ["text_only"]
        )
        self.assertFalse(pairs["notes.txt"]["head"]["full_and_parsed"])
        self.assertEqual(pairs["unread.py"]["head"]["fragment_keys"], [])
        self.assertEqual(pairs["unread.py"]["head"]["reason"], "size_limit")
        self.assertNotIn("warnings", projection)

    def test_projection_rejects_invalid_source_claims(self) -> None:
        """Producer projection fields cannot bypass source-state validation."""
        valid = snap({"x.py": "value = 1\n"}, {"x.py": "value = 2\n"})
        candidates = []
        for fields in (
            {"text": "\ud800"},
            {"start_line": True},
            {"start_line": 20},
            {"region": "unpaired-full-file"},
        ):
            candidate = copy.deepcopy(valid)
            candidate["fragments"][0].update(fields)
            candidates.append(candidate)
        candidate = copy.deepcopy(valid)
        candidate["file_evidence"][0]["base"] = {
            "path": "x.py",
            "state": "unavailable",
            "reason": "not_supplied",
        }
        candidates.append(candidate)
        for index, candidate in enumerate(candidates):
            with self.subTest(case=index):
                candidate["source_evidence"] = {
                    "schema": "spoofed",
                    "fragments": [],
                }
                with self.assertRaises(ValueError):
                    compile_snapshot(candidate)

    def test_projection_keeps_separate_regions_and_renamed_paths(self) -> None:
        """Keep fragment order, partial offsets, and cross-path file pairing."""
        snapshot = {
            "patches": [{"path": "new.py", "origin": "github", "patch": None}],
            "schema": "diffstory.snapshot.v1",
            "fragments": [
                {
                    "side": "head",
                    "path": "new.py",
                    "start_line": 40,
                    "region": "later",
                    "text": "value = 2\r\n",
                },
                {
                    "side": "base",
                    "path": "old.py",
                    "start_line": 10,
                    "region": "first",
                    "text": "def broken(:\r\n",
                },
                {
                    "side": "head",
                    "path": "new.py",
                    "start_line": 20,
                    "region": "first",
                    "text": "\r\n",
                },
                {
                    "side": "base",
                    "path": "old.py",
                    "start_line": 30,
                    "region": "later",
                    "text": "value = 1\r\n",
                },
            ],
            "file_evidence": [
                {
                    "base": {
                        "path": "old.py",
                        "state": "supplied",
                        "coverage": "partial",
                    },
                    "head": {
                        "path": "new.py",
                        "state": "supplied",
                        "coverage": "partial",
                    },
                }
            ],
        }
        projection = compile_snapshot(snapshot)["source_evidence"]
        fragments = projection["fragments"]
        self.assertEqual(
            [item["text"] for item in fragments],
            [item["text"] for item in snapshot["fragments"]],
        )
        self.assertEqual(
            [item["start_line"] for item in fragments], [40, 10, 20, 30]
        )
        self.assertEqual(
            [item["region"] for item in fragments],
            ["later", "first", "first", "later"],
        )
        pair = projection["file_evidence"][0]
        self.assertEqual(pair["base"]["path"], "old.py")
        self.assertEqual(pair["head"]["path"], "new.py")
        self.assertEqual(
            pair["base"]["fragment_keys"],
            [fragments[1]["key"], fragments[3]["key"]],
        )
        self.assertEqual(pair["base"]["parse_statuses"], ["failed", "ok"])
        self.assertEqual(
            pair["head"]["fragment_keys"],
            [fragments[0]["key"], fragments[2]["key"]],
        )
        self.assertEqual(pair["head"]["parse_statuses"], ["ok", "ok"])
        self.assertFalse(pair["head"]["full_and_parsed"])

    def test_raw_formatting_changes_get_group_ownership(self) -> None:
        """Retain blank-only and newline-only changes as bound text records."""
        cases = [
            (
                "line endings",
                "def f():\r\n    return 1\r\n",
                "def f():\n    return 1\n",
            ),
            (
                "final newline",
                "def f():\n    return 1",
                "def f():\n    return 1\n",
            ),
            ("empty to blank", "", "\n"),
            ("blank to empty", "\r\n", ""),
            ("blank whitespace", " \n", "\t\r\n"),
            (
                "trailing blank",
                "def f():\n    return 1\n",
                "def f():\n    return 1\n\n",
            ),
        ]
        for name, before, after in cases:
            with self.subTest(case=name):
                snapshot = snap({"x.py": before}, {"x.py": after})
                report = compile_snapshot(snapshot)
                self.assertEqual(report, compile_snapshot(snapshot))
                self.assertEqual(
                    [item["kind"] for item in report["changes"]], ["text"]
                )
                change = report["changes"][0]
                self.assertEqual(
                    report["groups"][0]["change_ids"], [change["id"]]
                )
                self.assertEqual(
                    change["raw_id"], report["raw_files"][0]["id"]
                )
                self.assertIn("terminal newline bytes", change["basis"])
                self.assertIn(
                    "No behavior change is inferred", change["basis"]
                )
                fragments = {
                    item["key"]: item
                    for item in report["source_evidence"]["fragments"]
                }
                for side in ("before", "after"):
                    record = change[side]
                    if record is None:
                        continue
                    fragment = fragments[record["fragment_key"]]
                    lines = fragment["text"].splitlines(keepends=True)
                    self.assertEqual(
                        record["source"],
                        "".join(lines[record["start"] - 1 : record["end"]]),
                    )
                render(report)

    def test_existing_structural_changes_own_their_raw_bytes(self) -> None:
        """Line endings inside edited declarations must not duplicate AST changes."""
        original = compile_snapshot(
            snap(
                {"x.py": "def f():\n    return 1\n"},
                {"x.py": "def f():\n    return 2\n"},
            )
        )
        changed = compile_snapshot(
            snap(
                {"x.py": "def f():\r\n    return 1\r\n"},
                {"x.py": "def f():\n    return 2"},
            )
        )
        self.assertEqual(len(changed["changes"]), 1)
        self.assertEqual(changed["changes"][0]["kind"], "modified")
        original.pop("source_evidence")
        changed.pop("source_evidence")
        self.assertEqual(original["meta"].pop("source_bytes"), 44)
        self.assertEqual(changed["meta"].pop("source_bytes"), 45)
        for field in ("supplied_additions", "supplied_deletions"):
            self.assertEqual(original["stats"].pop(field), 1)
            self.assertEqual(changed["stats"].pop(field), 2)
        self.assertNotEqual(
            original.pop("raw_files"), changed.pop("raw_files")
        )
        self.assertEqual(original, changed)

    def test_blank_changes_outside_edited_symbols_remain_owned(self) -> None:
        """A substantive edit must not hide a separate changed blank region."""
        report = compile_snapshot(
            snap(
                {"x.py": "def f():\n    return 1\n"},
                {"x.py": "\ndef f():\n    return 2\n\n"},
            )
        )
        self.assertEqual(
            [item["kind"] for item in report["changes"]],
            ["modified", "text", "text"],
        )
        text_changes = report["changes"][1:]
        self.assertEqual(
            [item["after"]["start"] for item in text_changes], [1, 4]
        )
        self.assertEqual(
            [item["after"]["end"] for item in text_changes], [1, 4]
        )
        self.assertEqual(
            [item["after"]["source"] for item in text_changes], ["\n", "\n"]
        )
        self.assertEqual(len({item["id"] for item in report["changes"]}), 3)
        owned = [
            key for group in report["groups"] for key in group["change_ids"]
        ]
        self.assertCountEqual(
            owned, [item["id"] for item in report["changes"]]
        )
        self.assertEqual(
            report["stats"]["by_kind"], {"modified": 1, "text": 2}
        )

    def test_unpaired_blank_region_does_not_claim_absence(self) -> None:
        """Keep unavailable and supplied-partial counterpart states explicit."""
        for direction in ("base", "head"):
            other = "head" if direction == "base" else "base"
            with self.subTest(side=direction):
                snapshot = snap(
                    {"x.py": "\r\n"} if direction == "base" else {},
                    {"x.py": "\r\n"} if direction == "head" else {},
                    overrides={
                        (other, "x.py"): {
                            "state": "unavailable",
                            "reason": "not_supplied",
                        }
                    },
                )
                report = compile_snapshot(snapshot)
                change = report["changes"][0]
                self.assertEqual(change["kind"], "text")
                self.assertIn(
                    "does not establish file addition or removal",
                    change["basis"],
                )
                self.assertNotIn("confirmed absent", change["basis"])
                self.assertEqual(
                    report["source_evidence"]["file_evidence"][0][other][
                        "state"
                    ],
                    "unavailable",
                )
        snapshot = snap(
            {"x.py": "\r\n"},
            {"x.py": "\n"},
            overrides={
                (side, "x.py"): {"state": "supplied", "coverage": "partial"}
                for side in ("base", "head")
            },
            fragment_options={
                ("base", "x.py"): {"region": "first", "start_line": 10},
                ("head", "x.py"): {"region": "second", "start_line": 20},
            },
        )
        report = compile_snapshot(snapshot)
        self.assertEqual(len(report["changes"]), 2)
        for change in report["changes"]:
            self.assertIn(
                "does not establish file addition or removal", change["basis"]
            )
        self.assertEqual(
            {
                item["region"]
                for item in report["source_evidence"]["fragments"]
            },
            {"first", "second"},
        )

    def test_unchanged_raw_source_has_no_changes(self) -> None:
        """Store unchanged and empty source without adding narrative units."""
        for text in ("", "\r\n", "def f():\r\n    return 'é'\r\n"):
            with self.subTest(text=text):
                report = compile_snapshot(snap({"x.py": text}, {"x.py": text}))
                self.assertEqual(report["changes"], [])
                self.assertEqual(report["groups"], [])
                self.assertEqual(report["raw_files"], [])
                fragments = report["source_evidence"]["fragments"]
                self.assertEqual(
                    [item["text"] for item in fragments], [text, text]
                )
                self.assertEqual(
                    fragments[0]["content_digest"],
                    fragments[1]["content_digest"],
                )
                self.assertNotEqual(fragments[0]["key"], fragments[1]["key"])


class PhysicalSourceAnalysisTests(unittest.TestCase):
    """Use physical source coordinates for extraction, excerpts, and display."""

    def test_unicode_string_keeps_modified_function_and_return(self) -> None:
        """Legal string characters must not remove the final function line."""
        for separator in ("\u2028", "\u2029", "\x85", "\v", "\f"):
            for eol in ("\n", "\r\n", "\r"):
                with self.subTest(separator=separator, eol=eol):
                    before = eol.join(
                        (
                            "def f():",
                            f"    value = 'a{separator}b'",
                            "    return 1",
                            "",
                        )
                    )
                    after = before.replace("return 1", "return 2")
                    report = compile_snapshot(
                        snap(
                            {"example.py": before},
                            {"example.py": after},
                            repository="example/project",
                        )
                    )
                    change = next(
                        c for c in report["changes"] if c["kind"] == "modified"
                    )
                    self.assertEqual(change["after"]["name"], "f")
                    self.assertEqual(change["after"]["end"], 3)
                    self.assertIn("return 2", change["after"]["source"])
                    self.assertIn(separator, change["after"]["source"])
                    self.assertTrue(change["after"]["url"].endswith("#L1-L3"))
                    rows = [r for h in change["hunks"] for r in h["rows"]]
                    self.assertIn(
                        3, [r["new"] for r in rows if "return 2" in r["text"]]
                    )

    def test_partial_adjacent_and_overlapping_unicode_excerpts(self) -> None:
        """Only CR and LF reserve lines in partial source fragments."""
        for eol in ("\n", "\r\n", "\r"):
            source = snap(
                {"x.py": "value = 'a\u2028b'" + eol},
                {},
                overrides={
                    ("base", "x.py"): {
                        "state": "supplied",
                        "coverage": "partial",
                    },
                },
                fragment_options={
                    ("base", "x.py"): {"start_line": 20, "region": "one"}
                },
            )
            source["fragments"].append(
                {
                    "path": "x.py",
                    "side": "base",
                    "text": "next_value = 2" + eol,
                    "start_line": 21,
                    "region": "two",
                }
            )
            compile_snapshot(source)
            source["fragments"][-1]["start_line"] = 20
            with self.assertRaisesRegex(ValueError, "Overlapping"):
                compile_snapshot(source)

    def test_text_hunks_context_and_fragment_url_use_physical_lines(
        self,
    ) -> None:
        """Non-Python source display and raw-file links keep exact line bounds."""
        before = "a\u2028b\nold\n"
        after = before.replace("old", "new")
        report = compile_snapshot(
            snap(
                {"x.txt": before},
                {"x.txt": after},
                repository="example/project",
            )
        )
        for change in report["changes"]:
            if change["after"]:
                self.assertLessEqual(change["after"]["end"], 2)
                self.assertTrue(change["after"]["url"].endswith("-L2"))
        self.assertTrue(report["raw_files"][0]["after_url"].endswith("#L1-L2"))
        hunks = make_hunks(before, after)
        rows = [r for h in hunks for r in h["rows"]]
        self.assertIn(2, [r["new"] for r in rows if r["text"] == "new"])


if __name__ == "__main__":
    unittest.main()
