"""Build a reproducible, entirely synthetic walkthrough. No repository/network access."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from diffstory.analysis import compile_snapshot, apply_annotations
from diffstory.render import render

LABEL = '''def normalize_label(value: str) -> str:
    """Collapse incidental whitespace without changing case or punctuation."""
    return " ".join(value.split())
'''
TAGS = '''def split_tags(value: str) -> list[str]:
    """Preserve source order while dropping empty, whitespace-only tags."""
    tags = []
    for part in value.split(","):
        if part.strip():
            tags.append(part.strip().lower())
    return tags
'''
MERGE = '''def merge_records(records):
    """Keep the first occurrence of each identifier, in encounter order."""
    seen = set()
    for record in records:
        key = record["id"]
        if key in seen:
            continue
        seen.add(key)
        yield record
'''
QUERY_BASE = '''def construct_query(term: str, limit: int = 25):
    """Select a page of catalog items using bound parameters.

    The caller provides a database connection. This helper only produces
    a statement and its parameters; it never executes a query itself.
    LIKE wildcard semantics are intentionally left to the database.
    """
    statement = (
        "SELECT id, label, tags, views, clicks FROM items "
        "WHERE label LIKE ? ORDER BY id LIMIT ?"
    )
    return statement, (f"%{term}%", limit)
'''
QUERY_HEAD = '''def construct_query(term: str, limit: int = 25):
    """Select a bounded page of catalog items using bound parameters.

    Query construction is independent of connection management and row
    interpretation. It returns two values: SQL and the parameter tuple.
    The caller passes both directly to connection.execute().

    Contract:
        * Only integer page sizes from 1 through 100 are accepted.
        * Booleans are rejected even though bool is an int subclass.
        * Search terms remain parameters, never SQL fragments.
        * ORDER BY id keeps the page ordering deterministic.
        * The default page size remains 25.

    LIKE wildcards inside a search term are not escaped by this function.
    The caller should not interpret the search as a literal substring
    match when the input contains SQL wildcard characters.

    This extraction intentionally introduces a page-size restriction.
    That is a behavioral edit, not an unchanged move. Callers that used
    zero, a negative value, or more than 100 now receive ValueError.
    """
    if type(limit) is not int or not 1 <= limit <= 100:
        raise ValueError("limit must be an integer between 1 and 100")

    statement = (
        "SELECT id, label, tags, views, clicks FROM items "
        "WHERE label LIKE ? ORDER BY id LIMIT ?"
    )
    parameters = (f"%{term}%", limit)
    return statement, parameters
'''
PARSE_BASE = '''def parse_record(row):
    """Turn a database row into a display record."""
    result = {
        "id": str(row["id"]),
        "label": normalize_label(row["label"]),
        "tags": split_tags(row.get("tags", "")),
        "click_rate": None,
    }
    if row.get("views"):
        result["click_rate"] = round(row.get("clicks", 0) / row["views"], 4)
    return result
'''
METRICS = '''def populate_metrics(row, result):
    """Add a click rate only when views are present and nonzero."""
    if row.get("views"):
        result["click_rate"] = round(row.get("clicks", 0) / row["views"], 4)
'''
PARSE_HEAD = '''def parse_record(row):
    """Turn a database row into a display record.

    Output fields:
        id: stable string representation of the source identifier.
        label: whitespace-normalized title with original capitalization.
        tags: lower-case tags in their original encounter order.
        click_rate: clicks divided by views, rounded to four places.

    Absent or zero views leave click_rate as None. Missing required
    identifiers and labels still raise KeyError; we do not silently
    invent identifiers or descriptions in a normalization step.

    The metrics calculation is now shared, but its rounding precision
    and missing-value rule are not intended to change. Test those rules
    at the public parse_record boundary, not just on the helper.
    """
    result = {
        "id": str(row["id"]),
        "label": normalize_label(row["label"]),
        "tags": parse_tags(row.get("tags", "")),
        "click_rate": None,
    }
    populate_metrics(row, result)
    return result
'''
RUN_BASE = '''def get_records(connection, term: str, limit: int = 25):
    statement, parameters = construct_query(term, limit)
    rows = connection.execute(statement, parameters)
    parsed = [parse_record(row) for row in rows]
    return list(merge_records(parsed))
'''
RUN_HEAD = '''def get_records(connection, term: str, limit: int = 25):
    """Build, execute, interpret, then deduplicate a single catalog page."""
    statement, parameters = construct_query(term, limit)
    rows = connection.execute(statement, parameters)
    # Materialize rows before merging, as before this extraction.
    records = [parse_record(row) for row in rows]
    return list(merge_records(records))
'''
TEST_QUERY = '''from catalog.query import construct_query


def test_query_default():
    statement, parameters = construct_query("notebook")
    assert "ORDER BY id LIMIT ?" in statement
    assert parameters == ("%notebook%", 25)


def test_term_stays_a_parameter():
    term = "paper' OR 1=1 --"
    statement, parameters = construct_query(term)
    assert term not in statement
    assert parameters[0] == f"%{term}%"


def test_page_size_boundaries():
    for limit in (1, 100):
        assert construct_query("pen", limit)[1][1] == limit
    for limit in (0, -1, 101, True, 2.5):
        try:
            construct_query("pen", limit)
        except ValueError:
            continue
        raise AssertionError(f"unexpectedly accepted {limit!r}")
'''
TEST_PARSE = '''from catalog.parsing import parse_record, parse_tags
from catalog.merge import merge_records


def test_record_preserves_display_contract():
    assert parse_record({"id": 7, "label": "  Blue   notebook ", "tags": "Office,, PAPER ", "views": 3, "clicks": 1}) == {
        "id": "7", "label": "Blue notebook", "tags": ["office", "paper"], "click_rate": 0.3333,
    }


def test_missing_and_zero_views():
    for row in ({"id": 7, "label": "Pen"}, {"id": 7, "label": "Pen", "views": 0}):
        assert parse_record(row)["click_rate"] is None


def test_tag_order_and_duplicates():
    assert parse_tags("paper, office, PAPER") == ["paper", "office", "paper"]


def test_first_record_wins():
    first = {"id": "7", "label": "First"}
    later = {"id": "7", "label": "Later"}
    assert list(merge_records([first, later])) == [first]
'''


def demo_snapshot():
    before = {
        "catalog/search.py": '\n\n'.join([LABEL, TAGS, MERGE, QUERY_BASE, PARSE_BASE, RUN_BASE]),
        "tools/export.py": 'from catalog.search import split_tags\n\n\ndef export_tags(value):\n    return ";".join(split_tags(value))\n',
    }
    after = {
        "catalog/labels.py": LABEL,
        "catalog/query.py": QUERY_HEAD,
        "catalog/merge.py": MERGE,
        "catalog/parsing.py": 'from catalog.labels import normalize_label\n\n\n' + TAGS.replace('def split_tags(', 'def parse_tags(') + '\n\n' + METRICS + '\n\n' + PARSE_HEAD,
        "catalog/search.py": 'from catalog.query import construct_query\nfrom catalog.parsing import parse_record\nfrom catalog.merge import merge_records\n\n\n' + RUN_HEAD,
        "tools/export.py": 'from catalog.parsing import parse_tags\n\n\ndef export_tags(value):\n    return ";".join(parse_tags(value))\n',
        "tests/test_query.py": TEST_QUERY,
        "tests/test_parsing.py": TEST_PARSE,
    }
    digest = lambda source: hashlib.sha1(json.dumps(source, sort_keys=True).encode()).hexdigest()
    return {
        "schema": "diffstory.snapshot.v1",
        "meta": {
            "title": "One search module. Three clear responsibilities.",
            "repository": "Synthetic catalog",
            "base_sha": digest(before), "head_sha": digest(after),
            "scope": "changed files", "input": "synthetic example",
            "comparison": "original synthetic before/after sources",
            "changed_files": len(before.keys() | after.keys()),
            "description": "An original catalog-search example for this open-source project. No private source or real customer data. The query page-size validation is an intentional behavior change.",
            "validation": "Test source is provided for explanation. Diffstory does not execute or assert results for those tests.",
        },
        "fragments": [{"path": path, "side": side, "text": code, "start_line": 1, "scope": "full"}
                      for side, files in (("base", before), ("head", after)) for path, code in sorted(files.items())],
        "warnings": [],
    }


EXPLANATIONS = {
    'construct_query': ('A move can also change the contract', 'The query builder moves out of the search module, but this is **not an unchanged extraction**. It now rejects page sizes outside 1–100, non-integers, and booleans. The SQL still uses bound parameters. Read the actual delta before treating the new module as a mechanical move.', 'diff'),
    'normalize_label': ('Give labels one small, predictable rule', 'One of the smallest dependencies is the label rule. `normalize_label` collapses whitespace and preserves case. The declaration moves unchanged to `catalog/labels.py`. Showing it once makes the rule easier to inspect than a deletion followed by an identical addition.', 'definition'),
    'parse_tags': ('Rename the helper, not its behavior', '`split_tags` becomes `parse_tags` in the parsing module. Empty fields disappear, tags become lowercase, and source order—including duplicates—survives. Only the declaration name changes; internal names and literal values still participate in the structural comparison.', 'definition'),
    'populate_metrics': ('Extract a rule with observable edge cases', 'The click-rate calculation is now a shared helper. A missing or zero view count leaves the existing result unchanged. A nonzero view count produces a ratio rounded to four decimal places. This helper is newly declared; its calculation was previously inline, so “new definition” does not mean “new behavior.”', 'definition'),
    'parse_record': ('Assemble the same public record', 'The parsing boundary brings those rules together. It still returns the same four fields, but delegates label normalization, tag parsing, and metrics. Compare the helper calls as well as the output keys: an unchanged shape is not sufficient evidence of unchanged values.', 'diff'),
    'merge_records': ('Preserve which duplicate wins', 'This generator moves intact into `catalog/merge.py`. Its rule is first-record-wins, not last-record-wins. Encounter order therefore remains part of the API even though the function now lives in a different module.', 'definition'),
    'get_records': ('Return to the caller with the pieces understood', 'Now the entry point is readable as a sequence: construct a query, execute it, interpret each row, and merge duplicates. It still materializes the parsed rows before merging. The tighter page-size rule reaches users through this call; it must not be hidden under the refactor label.', 'definition'),
    'export_tags': ('Follow the boundary beyond the main caller', 'The export utility must follow the renamed parser. Its separator is still a semicolon, and tag normalization comes from the same implementation as the catalog path. This is the kind of small consumer update that can be missed when review stops at the new modules.', 'diff'),
}


def demo_annotations(report):
    by_id = {c['id']: c for c in report['changes']}
    steps = []
    for group in report['groups']:
        passages = []
        title = None
        for cid in group['change_ids']:
            c = by_id[cid]; sym = c.get('after') or c.get('before'); name = sym['name']
            if name in EXPLANATIONS:
                heading, text, view = EXPLANATIONS[name]
                title = title or heading
            elif name == 'module imports / context':
                title = title or 'Point imports at the defining modules'
                text = 'These imports bind the names used below to their defining modules. This is wiring, not a new runtime rule; still check the import paths and module initialization order.'
                view = 'diff'
            elif group['theme'] == 'tests':
                title = 'Check the contract at its boundary' if 'query' in group['path'] else 'Make the preserved rules executable'
                text = {
                    'test_query_default': 'The default case locks down the 25-row page size and the shape of the parameter tuple. This test calls the extracted query module directly.',
                    'test_term_stays_a_parameter': 'A quote in the search term must remain data rather than changing the SQL statement. Inspect both the SQL and the bound value; checking only that the call succeeds would miss the contract.',
                    'test_page_size_boundaries': 'These are the boundary cases for the intentional behavior change. Accept 1 and 100; reject 0, 101, booleans, and other non-integer values. This is test source, not a recorded passing run.',
                    'test_record_preserves_display_contract': 'A representative row checks the complete display record: identifier, normalized label, ordered tags, and the rounded click rate. Delegation should not change those values.',
                    'test_missing_and_zero_views': 'Both absent and zero view counts must keep the rate unset. The explicit cases protect against introducing a division-by-zero path during extraction.',
                    'test_tag_order_and_duplicates': 'Case normalization and empty-tag removal must not turn the list into a set. The assertion preserves duplicates and source order.',
                    'test_first_record_wins': 'Two records share an identifier. The earlier record must win, so this assertion checks values as well as deduplication.',
                }.get(name, f'`{name}` states an expectation about the public boundary. This is test source, not an executed result.')
                view = 'definition'
            elif group['theme'] == 'wiring':
                title = 'Point imports at the defining modules'
                text = 'These imports connect the implementation to the definitions already introduced. Check that the new modules expose every requested name and do not introduce a circular import.'
                view = 'diff'
            else:
                title = title or 'Keep the supporting changes in view'
                text = 'This supporting change remains part of the same reading. Its source is preserved here rather than being dropped from the review because it is not a primary abstraction.'
                view = 'diff'
            passages.append({'text': text, 'change_ids': [cid], 'view': view})
        if group['theme'] == 'tests': title = 'Check the contract at its boundary' if 'query' in group['path'] else 'Make the preserved rules executable'
        if group['theme'] == 'result parsing': title = 'Build one parsing boundary'
        if group['theme'] == 'orchestration': title = 'Return to the search entry point'
        steps.append({
            'group_id': group['id'], 'title': title,
            'evidence_change_ids': list(group['change_ids']),
            'passages': passages,
            'transition': 'Follow the next dependency without leaving this document.' if group['next_id'] else 'The source trail now reaches the tests as well as the callers.',
        })
    return {'schema': 'diffstory.annotations.v1', 'base_sha': report['meta']['base_sha'], 'head_sha': report['meta']['head_sha'],
        'document': {
            'lead': 'A module extraction is easiest to understand as a sequence of ideas, not a list of files. Follow the small rules, the query boundary, and the callers—with each explanation next to its source.',
            'closing': 'The search entry point now composes focused modules. Several declarations moved without structural changes; the query limit contract deliberately changed. Those are different review tasks. The source and tests stay together so the distinction remains visible.'},
        'steps': steps}


def build():
    snapshot = demo_snapshot()
    report = compile_snapshot(snapshot)
    annotations = demo_annotations(report)
    report = apply_annotations(report, annotations)
    return snapshot, annotations, report


def main():
    snapshot, annotations, report = build()
    out = ROOT / 'examples'
    for suffix, obj in [('snapshot', snapshot), ('annotations', annotations), ('report', report)]:
        (out / f'demo.{suffix}.json').write_text(json.dumps(obj, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    (out / 'demo.html').write_text(render(report), encoding='utf-8')
    print(f'Synthetic example: {report["stats"]["groups"]} sections, {report["stats"]["units"]} changes.')


if __name__ == '__main__':
    main()
