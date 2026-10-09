"""Frozen request declarations and pure codecs for saved narration audits.

Audit identities are opaque strings. No codec resolves executables, imports a
stored type, constructs a provider, reads credentials, or performs transport.
"""

from __future__ import annotations

import json
from typing import Literal

from pydantic import Field
from pydantic import field_validator
from pydantic import model_validator

from .narrative_contracts import Contract
from .narrative_contracts import Digest
from .narrative_contracts import Natural
from .narrative_contracts import Positive
from .narrative_plan import RequestPackingLimits
from .narrative_plan import canonical_bytes
from .narrative_plan import content_digest
from .narrative_plan import semantic_digest


class LedgerParameters(Contract):
    """Declare the exact ledger partition parameters for reconstruction."""

    max_span_bytes: int = Field(ge=4)
    max_unit_spans: int = Field(ge=2)


class V2Options(Contract):
    """Explicit production limits; all fields enter configuration identity."""

    ledger: LedgerParameters = LedgerParameters(
        max_span_bytes=4096, max_unit_spans=16
    )
    packing: RequestPackingLimits = RequestPackingLimits()
    card_wire_bytes: int = Field(default=4096, ge=512)
    prose_wire_bytes: int = Field(default=8192, ge=512)
    card_support_bytes: int = Field(default=16384, ge=512)
    max_chapter_groups: int = Field(default=32, ge=1)


class StageIdentity(Contract):
    """Bind current prompts, schema and validator without storing instructions."""

    stage: Literal["detail", "planning", "section", "document"]
    prompt_digest: Digest
    schema_digest: Digest
    validator: str = Field(min_length=1, max_length=256)


class AdapterIdentity(Contract):
    """Retain legacy configuration identity as inert audit data only."""

    python_type: str = Field(min_length=1, max_length=1024)
    executable: str | None = Field(max_length=4096)
    requested_model: str | None = Field(max_length=256)


class RequestConfiguration(Contract):
    """Declare a known byte protocol, complete options and provider capacities."""

    version: Literal["diffstory.request-configuration.v1"]
    protocol: Literal[
        "diffstory.codex-json.v1",
        "diffstory.responses-json.v1",
        "diffstory.fixture-json.v1",
    ]
    name: str = Field(min_length=1, max_length=256)
    model: str = Field(min_length=1, max_length=256)
    destination: str = Field(min_length=1, max_length=2048)
    context_tokens: Positive
    max_output_tokens: Positive
    enforceable_output_cap: bool
    input_overhead_bytes: Natural
    prefix_hex: str = Field(pattern=r"^(?:[0-9a-f]{2})*$", max_length=131072)
    # Declarative fixture framing: zero to two canonical JSON string wrappers.
    # This is not an arbitrary serializer or a mutable options dictionary.
    fixture_json_layers: int = Field(ge=0, le=2)
    adapter: AdapterIdentity
    options: V2Options
    stages: tuple[StageIdentity, ...] = Field(min_length=4, max_length=4)

    @field_validator("options", mode="before")
    @classmethod
    def complete_options(cls, value: object) -> object:
        """Require every persisted option explicitly; construction defaults are local."""
        if isinstance(value, dict):
            complete = set(value) == set(V2Options.model_fields)
            for name, contract in (
                ("ledger", LedgerParameters),
                ("packing", RequestPackingLimits),
            ):
                nested = value.get(name)
                complete = (
                    complete
                    and isinstance(nested, dict)
                    and set(nested) == set(contract.model_fields)
                )
            if not complete:
                msg = "incomplete_request_options"
                raise ValueError(msg)
        return value

    @model_validator(mode="after")
    def coherent(self) -> RequestConfiguration:
        """Reject impossible or incomplete declarations before reconstructing bytes."""
        valid = (
            tuple(s.stage for s in self.stages)
            == ("detail", "planning", "section", "document")
            and self.max_output_tokens < self.context_tokens
        )
        if self.protocol == "diffstory.responses-json.v1":
            valid = valid and (
                self.destination == "https://api.openai.com/v1/responses"
                and self.name == "openai"
                and self.enforceable_output_cap
                and self.fixture_json_layers == 0
                and self.adapter.executable is None
                and self.adapter.requested_model is None
            )
        elif self.protocol == "diffstory.codex-json.v1":
            valid = valid and (
                self.name == "codex"
                and not self.enforceable_output_cap
                and self.fixture_json_layers == 0
                and bool(self.adapter.executable)
                and (
                    self.adapter.requested_model is None
                    or self.adapter.requested_model == self.model
                )
            )
        else:
            valid = valid and (
                self.destination.startswith("fixture://")
                and self.adapter.executable is None
            )
        if not valid:
            msg = "invalid_request_configuration"
            raise ValueError(msg)
        return self

    @property
    def prefix(self) -> bytes:
        """Decode exact bounded request framing, including non-UTF8 bytes."""
        return bytes.fromhex(self.prefix_hex)

    def legacy_configuration(self) -> dict:
        """Recreate the unchanged original configuration dependency identity."""
        return {
            "type": self.adapter.python_type,
            "name": self.name,
            "model": self.model,
            "destination": self.destination,
            "context_tokens": self.context_tokens,
            "max_output_tokens": self.max_output_tokens,
            "input_overhead_bytes": self.input_overhead_bytes,
            "prefix": content_digest(self.prefix),
            "requested_model": self.adapter.requested_model,
            "executable": self.adapter.executable,
            "enforceable_output_cap": self.enforceable_output_cap,
            "options": self.options.model_dump(mode="json"),
        }

    @property
    def configuration_digest(self) -> str:
        """Keep ordinary identities; bind any additional fixture framing."""
        return self.identity_digest(
            "codec" if self.fixture_json_layers else "legacy"
        )

    def identity_digest(self, form: Literal["legacy", "codec"]) -> str:
        """Derive only the two closed identity forms used for audited replay."""
        if form not in {"legacy", "codec"}:
            msg = "invalid_configuration_identity"
            raise ValueError(msg)
        value = self.legacy_configuration()
        if form == "codec":
            value["codec"] = {
                "protocol": self.protocol,
                "fixture_json_layers": self.fixture_json_layers,
            }
        return semantic_digest("diffstory.transport.v2", value)


def serialize_request(  # noqa: PLR0913 # Exact adapter serialization signature.
    configuration: RequestConfiguration,
    system: str,
    data: dict,
    schema_name: str,
    schema: dict,
    output_tokens: int,
) -> bytes:
    """Recreate exact adapter bytes, including schemas, output caps and prefix."""
    configuration = RequestConfiguration.model_validate(configuration)
    if not 0 < output_tokens <= configuration.max_output_tokens:
        msg = "invalid_request_output"
        raise ValueError(msg)
    if configuration.protocol == "diffstory.responses-json.v1":
        body = {
            "model": configuration.model,
            "store": False,
            "max_output_tokens": output_tokens,
            "input": [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": json.dumps(
                        data, ensure_ascii=False, separators=(",", ":")
                    ),
                },
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "strict": True,
                    "schema": schema,
                }
            },
        }
    else:
        body = {
            "system": system,
            "data": data,
            "schema_name": schema_name,
            "schema": schema,
        }
        if configuration.protocol == "diffstory.fixture-json.v1":
            body["max_output_tokens"] = output_tokens
            raw = canonical_bytes(body)
            for _ in range(configuration.fixture_json_layers):
                raw = canonical_bytes({"payload": raw.decode("utf-8")})
            return configuration.prefix + raw
    return configuration.prefix + json.dumps(
        body, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
