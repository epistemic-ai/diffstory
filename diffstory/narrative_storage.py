"""Lossless storage codecs, independent of runtime imports and cache policy.

The invocation codec stores static dependencies once. Callers supply the
expanded model class when decoding, so its existing validators remain the
authority. The same codec can be used at other persistence boundaries.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING
from typing import Literal
from typing import TypeVar

from pydantic import model_validator

from .narrative_contracts import ArtifactDependency
from .narrative_contracts import BatchSpec
from .narrative_contracts import Contract
from .narrative_contracts import Digest
from .narrative_contracts import ResultKind

if TYPE_CHECKING:
    from .narrative_runtime import ArtifactInvocation

_Invocation = TypeVar("_Invocation", bound=Contract)
_STATIC_KINDS = frozenset(
    {"provider", "model", "configuration", "prompt", "schema", "validator"}
)


class StoredLookup(Contract):
    """Ordered item fields whose static dependencies come from the invocation."""

    item_key: Digest
    result_kind: ResultKind
    validator_version: str


class StoredInvocation(Contract):
    """Versioned full invocation content with one static dependency set."""

    version: Literal["diffstory.invocation-storage.v1"] = (
        "diffstory.invocation-storage.v1"
    )
    batch: BatchSpec
    lookups: tuple[StoredLookup, ...]
    static_dependencies: tuple[ArtifactDependency, ...]
    dynamic_dependencies: tuple[ArtifactDependency, ...]

    @model_validator(mode="after")
    def dependency_partition(self) -> StoredInvocation:
        """Forbid alternate or duplicate placements of the static set."""
        if (
            len(self.static_dependencies) != len(_STATIC_KINDS)
            or {d.kind for d in self.static_dependencies} != _STATIC_KINDS
            or any(d.kind in _STATIC_KINDS for d in self.dynamic_dependencies)
        ):
            msg = "Invalid stored dependency partition"
            raise ValueError(msg)
        return self


def encode_invocation(invocation: ArtifactInvocation) -> StoredInvocation:
    """Encode a validated expanded invocation without changing any identity."""
    invocation = type(invocation).model_validate(invocation)
    return StoredInvocation(
        batch=invocation.batch,
        lookups=tuple(
            StoredLookup(
                item_key=item.item_key,
                result_kind=item.result_kind,
                validator_version=item.validator_version,
            )
            for item in invocation.lookups
        ),
        static_dependencies=tuple(
            d for d in invocation.dependencies if d.kind in _STATIC_KINDS
        ),
        dynamic_dependencies=tuple(
            d for d in invocation.dependencies if d.kind not in _STATIC_KINDS
        ),
    )


def decode_invocation(
    stored: StoredInvocation, invocation_type: type[_Invocation]
) -> _Invocation:
    """Expand through the supplied invocation model's complete validation."""
    stored = StoredInvocation.model_validate(stored)
    return invocation_type.model_validate(
        {
            "version": "diffstory.invocation.v2",
            "batch": stored.batch,
            "lookups": tuple(
                {
                    **item.model_dump(),
                    "static_dependencies": stored.static_dependencies,
                }
                for item in stored.lookups
            ),
            "dependencies": (
                *stored.static_dependencies,
                *stored.dynamic_dependencies,
            ),
        }
    )


def storage_digest(record: Contract) -> str:
    """Address full canonical content, separated by storage domain and version.

    Each record carries a distinct required version. No semantic key or supplied
    batch ID substitutes for any part of its exact content.
    """
    value = {
        "domain": "diffstory.private-storage.v3",
        "record": record.model_dump(mode="json"),
    }
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
