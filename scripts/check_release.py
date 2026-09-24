#!/usr/bin/env python3
"""Check public release inputs and optionally seal/verify an exact-file manifest.

This is a deterministic guardrail, not a comprehensive secret detector or an
independent licensing audit. Review examples and screenshots before publication.
No file contents or candidate secret values are printed on errors.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = 'PUBLICATION.json'
SCHEMA = 'diffstory.publication.v1'
TOP_FILES = {
    'README.md', 'LICENSE', 'NOTICE', 'CONTRIBUTING.md', 'SECURITY.md',
    'CODE_OF_CONDUCT.md', 'CHANGELOG.md', 'pyproject.toml', 'MANIFEST.in',
    '.gitignore', '.gitattributes', '.editorconfig',
}
TREES = {'diffstory', 'tests', 'examples', 'docs', 'scripts', '.github'}
SKIP_DIRS = {'.git', '.venv', 'venv', '__pycache__', '.pytest_cache', 'build', 'dist'}
SKIP_SUFFIXES = {'.pyc', '.pyo'}
TEXT_SUFFIXES = {'.py', '.js', '.css', '.html', '.json', '.md', '.yml', '.yaml', '.svg', '.sh', '.toml', '.in'}
PUBLIC_DATA = {'examples/demo.html', 'examples/demo.report.json',
               'examples/demo.snapshot.json', 'examples/demo.annotations.json'}
# Deliberately assembled: the source for this guard must not match its own rules.
PRIVATE_MARKERS = tuple(value.lower() for value in (
    'epistemic-ai/' + 'foundation', 'pr' + '20781', 'eai_' + 'query',
    'CF' + 'TR2', 'Clin' + 'Var', 'ba7ce6c20e0e699b7034d' + '1105604758e74552da2',
    '8a56c02f3c49bf4e1dfb2a9' + 'cec3d5207ca594181',
))
SECRET_PATTERNS = (
    re.compile(r'\bgh[pousr]_[A-Za-z0-9]{30,}\b'),
    re.compile(r'\bgithub_pat_[A-Za-z0-9_]{30,}\b'),
    re.compile(r'\bAKIA[0-9A-Z]{16}\b'),
    re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
    re.compile(r'https://[^\s"<>]+[?&]token=[A-Za-z0-9_-]{16,}'),
)


def publication_files(root: Path) -> list[Path]:
    """Return the exact candidate set, refusing unknown inputs and symlinks."""
    files: list[Path] = []

    def walk(directory: Path):
        for path in sorted(directory.iterdir()):
            rel = path.relative_to(root).as_posix()
            if path.is_symlink():
                raise ValueError(f'Symlinks are not public release inputs: {rel}')
            if path.is_dir():
                if path.name in SKIP_DIRS or path.name.endswith('.egg-info'):
                    continue
                if directory == root and path.name not in TREES:
                    raise ValueError(f'Unrecognized release directory: {rel}')
                walk(path)
                continue
            if path.name == MANIFEST and directory == root:
                continue
            if path.suffix in SKIP_SUFFIXES or path.name == '.DS_Store':
                continue
            if path.name.startswith('.env') or path.suffix.lower() in {'.pem', '.key', '.p12', '.pfx'}:
                raise ValueError(f'Credential-like file refused: {rel}')
            if directory == root and path.name not in TOP_FILES:
                raise ValueError(f'Unrecognized release file: {rel}')
            if directory != root and path.suffix not in TEXT_SUFFIXES:
                if not (path.suffix == '.png' and rel.startswith('docs/reader/')):
                    raise ValueError(f'Unsupported public file type: {rel}')
            if any(rel.endswith(suffix) for suffix in ('.report.json', '.snapshot.json', '.annotations.json', '.review.json')) and rel not in PUBLIC_DATA:
                raise ValueError(f'Unreviewed report data refused: {rel}')
            if path.suffix == '.html' and rel not in PUBLIC_DATA | {'diffstory/assets/app.html'}:
                raise ValueError(f'Unreviewed rendered HTML refused: {rel}')
            files.append(path)

    walk(root)
    return sorted(files, key=lambda p: p.relative_to(root).as_posix())


def inspect_files(root: Path) -> list[Path]:
    files = publication_files(root)
    for path in files:
        if path.stat().st_size > 12_000_000:
            raise ValueError(f'Unexpectedly large release file: {path.relative_to(root)}')
        if path.suffix == '.png':
            if not path.read_bytes().startswith(b'\x89PNG\r\n\x1a\n'):
                raise ValueError(f'Invalid PNG signature: {path.relative_to(root)}')
            continue
        text = path.read_text(encoding='utf-8')
        if any(marker in text.lower() for marker in PRIVATE_MARKERS):
            raise ValueError(f'Private prototype marker found in {path.relative_to(root)}')
        if any(pattern.search(text) for pattern in SECRET_PATTERNS):
            raise ValueError(f'Possible secret found in {path.relative_to(root)}')
    for name in ('demo.snapshot.json', 'demo.report.json'):
        path = root / 'examples' / name
        if not path.is_file():
            raise ValueError(f'Missing public synthetic example: {path.relative_to(root)}')
        data = json.loads(path.read_text())
        meta = data.get('meta', {})
        if meta.get('input') != 'synthetic example' or meta.get('number') or meta.get('url'):
            raise ValueError(f'Public demo must be explicitly synthetic: {path.relative_to(root)}')
    return files


def entries(root: Path, files: list[Path]) -> list[dict[str, str]]:
    return [{'path': path.relative_to(root).as_posix(),
             'sha256': hashlib.sha256(path.read_bytes()).hexdigest()} for path in files]


def check(root: Path = ROOT, *, verify: bool = False, write: bool = False) -> int:
    root = root.resolve()
    files = inspect_files(root)
    current = entries(root, files)
    if write:
        meta = {'schema': SCHEMA, 'repository': 'epistemic-ai/diffstory',
                'visibility': 'public', 'files': current}
        (root / MANIFEST).write_text(json.dumps(meta, indent=2) + '\n', encoding='utf-8')
    if verify:
        recorded = json.loads((root / MANIFEST).read_text(encoding='utf-8'))
        if (recorded.get('schema') != SCHEMA or recorded.get('repository') != 'epistemic-ai/diffstory'
                or recorded.get('visibility') != 'public' or recorded.get('files') != current):
            raise ValueError('Publication manifest does not match the exact current file set and hashes')
    return len(files)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--manifest', action='store_true', help='Verify sealed paths and hashes')
    action.add_argument('--write-manifest', action='store_true', help='Seal reviewed public inputs')
    args = parser.parse_args(argv)
    try:
        count = check(args.root, verify=args.manifest, write=args.write_manifest)
    except (OSError, ValueError, UnicodeError) as exc:
        print(f'Release check failed: {exc}', file=sys.stderr)
        return 1
    print(f'Release check passed: {count} public files' + ('; exact manifest verified.' if args.manifest else '.'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
