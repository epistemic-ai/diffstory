"""Validated source and narration contracts shared by the compiler and reader."""

from __future__ import annotations

import re
from typing import Annotated
from typing import Literal

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import ValidationError
from pydantic import field_validator
from pydantic import model_validator

MAX_SNAPSHOT_SOURCE_BYTES = 64_000_000
MAX_NARRATIVE_TEXT_CHARS = 6_000
MAX_PREAMBLE_CHARS = 4_000
MAX_PREAMBLE_SKETCH_CHARS = 4_000
MAX_PREAMBLE_TOTAL_CHARS = MAX_PREAMBLE_CHARS + MAX_PREAMBLE_SKETCH_CHARS

# Use the same complete fenced-text blocks that the reader recognizes.
_PREAMBLE_SKETCH = re.compile(
    r"(^|\n)```text[ \t]*\n.*?\n```(?=\n|$)", re.DOTALL
)

RevisionSide = Literal["base", "head"]
ParseStatus = Literal["ok", "text_only", "failed"]
AvailabilityReason = Literal[
    "not_supplied",
    "size_limit",
    "binary_or_non_utf8",
    "unsupported_object",
    "invalid_content_size",
    "unsupported_encoding",
    "read_failed",
]


class StrictModel(BaseModel):
    """Require exact field types and reject unknown fields at data boundaries."""

    model_config = ConfigDict(
        strict=True, extra="forbid", hide_input_in_errors=True
    )


class FileSide(StrictModel):
    """Identify a file at one pinned revision.

    Attributes:
        path: Nonempty repository-relative path, without dot or parent parts.
    """

    path: str

    @field_validator("path")
    @classmethod
    def relative_path(cls, path: str) -> str:
        """Accept a repository path without resolving or reading the filesystem.

        Args:
            path: Strict string from a source-side record.

        Returns:
            The original path when it is relative and has no unsafe parts.

        Raises:
            ValueError: If the path is empty, absolute, or contains NUL or dots.
        """
        unsafe = (
            not path
            or "\0" in path
            or path.startswith(("/", "\\"))
            or re.match(r"^[A-Za-z]:", path)
            or any(part in {".", ".."} for part in path.split("/"))
        )
        if unsafe:
            msg = "File evidence path must be a nonempty repository-relative path"
            raise ValueError(msg)
        return path


class AbsentFile(FileSide):
    """State that a pinned revision has no file at the recorded path.

    Attributes:
        state: Always ``absent``; source and availability fields are forbidden.
    """

    state: Literal["absent"] = "absent"


class UnavailableFile(FileSide):
    """State that source could not be supplied, without claiming file absence.

    Attributes:
        state: Always ``unavailable``.
        reason: Supported code describing why source is missing or skipped.
    """

    state: Literal["unavailable"] = "unavailable"
    reason: AvailabilityReason


class SuppliedFile(FileSide):
    """State that source is supplied with explicit coverage.

    Attributes:
        state: Always ``supplied``.
        coverage: ``full`` for the whole file or ``partial`` for excerpts.
    """

    state: Literal["supplied"] = "supplied"
    coverage: Literal["full", "partial"]


SourceSide = Annotated[
    AbsentFile | UnavailableFile | SuppliedFile, Field(discriminator="state")
]


class FileEvidence(StrictModel):
    """Link the base and head paths for one changed file.

    Attributes:
        base: Source state at the effective base revision.
        head: Source state at the head revision; both sides cannot be absent.
    """

    base: SourceSide
    head: SourceSide

    @model_validator(mode="after")
    def existing_side(self) -> FileEvidence:
        """Require at least one side to represent an existing or unread file.

        Returns:
            This record when its two sides are not both absent.

        Raises:
            ValueError: If the record claims absence on both revisions.
        """
        if isinstance(self.base, AbsentFile) and isinstance(
            self.head, AbsentFile
        ):
            msg = "A file evidence record cannot be absent on both sides"
            raise ValueError(msg)
        return self


class SourceFragment(StrictModel):
    """Validate source text and its position while preserving producer metadata.

    Attributes:
        path: Source path; legacy fragments retain their original path rules.
        side: ``base`` or ``head``.
        text: UTF-8 source text, including an empty file.
        start_line: Positive original line number; booleans are rejected.
        scope: Producer's coverage label, or ``None`` when omitted.
        region: Nonempty label joining before/after excerpts, when supplied.
    """

    model_config = ConfigDict(extra="allow")

    path: str
    side: RevisionSide
    text: str
    start_line: int = Field(default=1, ge=1)
    scope: str | None = None
    region: str | None = None

    @field_validator("region")
    @classmethod
    def nonempty_region(cls, value: str | None) -> str:
        """Reject an empty supplied region while allowing an omitted field.

        Args:
            value: Region from an explicitly supplied field.

        Returns:
            The nonempty region string.

        Raises:
            ValueError: If the supplied region is empty or null.
        """
        if not value:
            msg = "Fragment region must be a nonempty string"
            raise ValueError(msg)
        return value


class SnapshotInput(StrictModel):
    """Validate the snapshot envelope before parsing or indexing source.

    Attributes:
        schema_version: Fixed v1 schema, read from the JSON ``schema`` field.
        meta: Producer metadata with measured source bytes added after validation.
        fragments: Source records in their original order.
        warnings: Producer warnings; their text never controls classification.
        file_evidence: Explicit file pairs, or ``None`` for legacy snapshots.
    """

    model_config = ConfigDict(extra="ignore")

    schema_version: Literal["diffstory.snapshot.v1"] = Field(alias="schema")
    meta: dict = Field(default_factory=dict)
    fragments: list[SourceFragment] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    file_evidence: list[FileEvidence] | None = None

    @field_validator("file_evidence", mode="before")
    @classmethod
    def evidence_list(cls, value: object) -> object:
        """Require a list when explicit file evidence is present.

        Args:
            value: Explicit field value; this check does not run on omission.

        Returns:
            The input list for model validation.

        Raises:
            ValueError: If the field is not a list, including explicit null.
        """
        if not isinstance(value, list):
            msg = "Snapshot file_evidence must be a list"
            raise ValueError(msg)
        return value

    @field_validator("file_evidence")
    @classmethod
    def unique_paths(cls, records: list[FileEvidence]) -> list[FileEvidence]:
        """Keep each revision/path in exactly one comparison record.

        Args:
            records: Validated file comparisons.

        Returns:
            The records without changing their order.

        Raises:
            ValueError: If a side/path pair repeats.
        """
        seen = set()
        for record in records:
            for side_name in ("base", "head"):
                key = (side_name, getattr(record, side_name).path)
                if key in seen:
                    msg = "File evidence side path must be unique"
                    raise ValueError(msg)
                seen.add(key)
        return records

    @model_validator(mode="after")
    def bounded_source(self) -> SnapshotInput:
        """Measure UTF-8 bytes and reject incomplete empty comparisons.

        Returns:
            This snapshot with measured source bytes in its metadata.

        Raises:
            ValueError: If text cannot encode as UTF-8, source exceeds the cap,
                or an empty snapshot is not declared as an empty comparison.
        """
        try:
            source_bytes = sum(
                len(item.text.encode("utf-8")) for item in self.fragments
            )
        except UnicodeEncodeError as error:
            msg = "Snapshot fragment text must be valid UTF-8"
            raise ValueError(msg) from error
        if source_bytes > MAX_SNAPSHOT_SOURCE_BYTES:
            msg = f"Snapshot source exceeds the {MAX_SNAPSHOT_SOURCE_BYTES} byte aggregate limit"
            raise ValueError(msg)
        if not self.fragments and self.meta.get("changed_files") != 0:
            msg = "Snapshot contains no source fragments"
            raise ValueError(msg)
        self.meta["source_bytes"] = source_bytes
        return self


class DocumentNarrative(StrictModel):
    """Validate optional authored document prose and retain field omission.

    Attributes:
        preamble: Nonblank overview with at most 4,000 prose characters and
            a separate 4,000-character allowance for complete text sketches.
        lead: Opening prose of at most 6,000 characters, when supplied.
        closing: Closing prose of at most 6,000 characters, when supplied.
    """

    preamble: str | None = Field(
        default=None, max_length=MAX_PREAMBLE_TOTAL_CHARS
    )
    lead: str | None = Field(default=None, max_length=MAX_NARRATIVE_TEXT_CHARS)
    closing: str | None = Field(
        default=None, max_length=MAX_NARRATIVE_TEXT_CHARS
    )

    @field_validator("preamble", "lead", "closing", mode="before")
    @classmethod
    def text_fields(cls, value: object) -> str:
        """Reject null and non-text values for explicitly supplied prose.

        Args:
            value: Explicit prose value; omitted fields retain their defaults.

        Returns:
            The original string without trimming reader-visible text.

        Raises:
            ValueError: If the supplied value is not a string.
        """
        if not isinstance(value, str):
            msg = "Document narrative fields must be strings"
            raise ValueError(msg)
        return value

    @field_validator("preamble")
    @classmethod
    def bounded_preamble(cls, value: str) -> str:
        """Keep prose and complete fenced sketches within separate allowances.

        Args:
            value: Validated preamble string.

        Returns:
            The original nonblank text without changing its line endings.

        Raises:
            ValueError: If the preamble is blank, prose exceeds 4,000 characters,
                or all sketches together exceed 4,000 characters. Sketch counts
                include fences. Incomplete fences count as prose.
        """
        if not value.strip():
            msg = "Invalid document preamble"
            raise ValueError(msg)
        content = value.replace("\r\n", "\n").replace("\r", "\n")
        sketch_chars = sum(
            match.end() - match.start()
            for match in _PREAMBLE_SKETCH.finditer(content)
        )
        prose_chars = len(content) - sketch_chars
        if prose_chars > MAX_PREAMBLE_CHARS:
            msg = f"Document preamble prose exceeds {MAX_PREAMBLE_CHARS} characters"
            raise ValueError(msg)
        if sketch_chars > MAX_PREAMBLE_SKETCH_CHARS:
            msg = f"Document preamble sketches exceed {MAX_PREAMBLE_SKETCH_CHARS} characters"
            raise ValueError(msg)
        return value


class GeneratedDocument(DocumentNarrative):
    """Require opening and closing prose in saved generated narration.

    Attributes:
        lead: Required nonblank opening, at most 6,000 characters.
        closing: Required nonblank closing, at most 6,000 characters.
    """

    lead: str = Field(max_length=MAX_NARRATIVE_TEXT_CHARS)
    closing: str = Field(max_length=MAX_NARRATIVE_TEXT_CHARS)

    @field_validator("lead", "closing")
    @classmethod
    def nonblank_prose(cls, value: str) -> str:
        """Require generated prose to contain text without changing its spacing.

        Args:
            value: Validated opening or closing string.

        Returns:
            The original nonblank text.

        Raises:
            ValueError: If the field is blank.
        """
        if not value.strip():
            msg = "Generated narration requires a nonempty document opening and closing"
            raise ValueError(msg)
        return value


class ProviderDocument(GeneratedDocument):
    """Require a preamble in newly generated narration.

    Attributes:
        preamble: Required overview with separate prose and sketch allowances.
    """

    preamble: str = Field(
        max_length=MAX_PREAMBLE_TOTAL_CHARS,
        description=(
            "Introduce the whole change before a code tour. Use about 200 to 450 words "
            "when the evidence supports that length; use less for a small change. "
            "Explain the conceptual areas and reading path without real source names. "
            "For a multi-part change, include one useful conceptual sketch when "
            "supported. Use a fenced text block; a decision flow chart has a label "
            "and two branches: ├─ condition → outcome, then └─ condition → outcome. "
            "Keep other sketches as text. Allow at most 4,000 characters of prose "
            "and a separate 4,000 characters for all complete sketches, including "
            "their fences. Keep the total within 8,000 characters. "
            "Separate prose paragraphs with a blank line."
        ),
    )


def validate_document(
    value: object, *, generated: bool = False
) -> DocumentNarrative:
    """Apply the same document contract at annotation and rendering boundaries.

    Args:
        value: Untrusted document mapping.
        generated: Whether opening and closing fields are required.

    Returns:
        Validated prose with omitted optional fields retained as omission.

    Raises:
        ValueError: If document fields violate their type, length, or text rules.
    """
    contract = GeneratedDocument if generated else DocumentNarrative
    try:
        return contract.model_validate(value)
    except ValidationError as error:
        msg = f"Invalid document narrative: {error}"
        raise ValueError(msg) from error
