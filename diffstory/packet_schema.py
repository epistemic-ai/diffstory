"""Strict saved-packet shapes; source and reference invariants live in the builder."""

from __future__ import annotations

from typing import Annotated
from typing import Literal

from pydantic import ConfigDict
from pydantic import Field
from pydantic import JsonValue

from .models import FileEvidence
from .models import FileSide
from .models import ParseStatus
from .models import RevisionSide
from .models import StrictModel

Identifier = Annotated[str, Field(pattern=r"^[0-9a-f]{16}$")]
Revision = Annotated[str, Field(pattern=r"^[0-9a-f]{7,64}$")]
Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Line = Annotated[int, Field(ge=1)]
Count = Annotated[int, Field(ge=0)]


class PacketMeta(StrictModel):
    """Validate reader-visible metadata while retaining producer JSON context."""

    model_config = ConfigDict(extra="allow")
    __pydantic_extra__: dict[str, JsonValue] = Field(init=False)
    base_sha: Revision
    head_sha: Revision
    source_bytes: Count
    number: Count | None = None
    changed_files: Count | None = None
    additions: Count | None = None
    deletions: Count | None = None
    title: str | None = None
    repository: str | None = None
    head_repository: str | None = None
    url: str | None = None
    scope: str | None = None
    input: str | None = None
    description: str | None = None
    validation: str | None = None
    captured_at: str | None = None


class PacketSource(FileSide):
    """Store one exact compiler buffer at a pinned original position."""

    id: Identifier
    side: RevisionSide
    revision: Revision
    start_line: Line
    region: str = Field(min_length=1)
    scope: str | None
    parse_status: ParseStatus
    text: str
    digest: Digest


class SourceRange(StrictModel):
    """Address inclusive original source lines without embedding source text."""

    source_id: Identifier
    revision: Revision
    start: Line
    end: Line


class DefinitionRange(SourceRange):
    """Retain a compiler definition's display name and optional source URL."""

    name: str
    url: str | None


class PacketChange(StrictModel):
    """Describe a structural change with before and after source references."""

    id: Identifier
    kind: str = Field(pattern=r"^[a-z_]+$")
    label: str
    basis: str
    before: DefinitionRange | None
    after: DefinitionRange | None
    comparison_hunks: list[DiffHunk] | None = None


class DiffRow(StrictModel):
    """Reference one original diff row, with no duplicated row text."""

    tag: Literal["add", "delete", "context"]
    old: Line | None
    new: Line | None


class DiffHunk(StrictModel):
    """Keep the original diff coordinates and complete supplied row inventory."""

    old_start: Count
    new_start: Count
    old_count: Count
    new_count: Count
    rows: list[DiffRow]


class PacketDiff(FileSide):
    """Bind a supplied before/after diff to its sole source buffers."""

    id: Identifier
    old_path: str | None
    region: str
    scope: str | None
    before_url: str | None
    after_url: str | None
    base_source: Identifier | None
    head_source: Identifier | None
    hunks: list[DiffHunk]
    diff_origin: Literal["git", "github"]
    patch_status: Literal["captured", "unavailable", "unbound"]
    original_patch: str | None


class ChangedRegion(StrictModel):
    """Account for a changed hunk or a source edit hidden by line normalization."""

    id: str
    diff_id: Identifier | None
    hunk: Count | None
    citations: list[SourceRange]
    change_ids: list[Identifier]


class TestLink(StrictModel):
    """Record a static test relationship without claiming test execution."""

    test_id: Identifier
    symbol_id: Identifier
    relationship: str
    url: str | None
    status: Literal["referenced, not run"]


class PacketGroup(StrictModel):
    """Supply compiler grouping and dependency hints to the writer."""

    id: Identifier
    title: str
    path: str
    change_ids: list[Identifier]
    prerequisites: list[Identifier]
    theme: str
    test_links: list[TestLink]


class Dependency(StrictModel):
    """Represent a compiler relationship between group IDs."""

    from_id: Identifier = Field(alias="from")
    to: Identifier
    type: str
    url: str | None
    label: str


class SymbolEdge(StrictModel):
    """Retain a static symbol relationship and its original source location."""

    from_id: Identifier = Field(alias="from")
    to: Identifier
    type: str
    path: str
    line: Line
    url: str | None
    evidence: str


class StaticTest(StrictModel):
    """Refer to a test definition in the shared source store."""

    id: Identifier
    source: DefinitionRange


class UnresolvedReference(StrictModel):
    """Keep an unresolved static binding and its compiler explanation."""

    from_id: Identifier = Field(alias="from")
    reference: str
    reason: str


class StaticAnalysis(StrictModel):
    """Retain source analysis as data independent of model prose."""

    tests: list[StaticTest]
    symbol_edges: list[SymbolEdge]
    cycles: list[list[Identifier]]
    unresolved: list[UnresolvedReference]
    method: dict[str, str]


class Packet(StrictModel):
    """Accept only the flat evidence fields supported by the saved reader."""

    version: Literal["diffstory.packet.v1"]
    meta: PacketMeta
    sources: list[PacketSource]
    file_evidence: list[FileEvidence]
    changes: list[PacketChange]
    diffs: list[PacketDiff]
    regions: list[ChangedRegion]
    groups: list[PacketGroup]
    dependencies: list[Dependency]
    analysis: StaticAnalysis
    warnings: list[str]
    notes: list[str]
