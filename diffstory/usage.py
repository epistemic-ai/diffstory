"""Model-context packing and provider-reported narration usage."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:
    from tiktoken import Encoding

INPUT_FRAMING_MARGIN = 1024


@dataclass(frozen=True)
class ModelCapacity:
    """Keep each serialized request within the selected model's context window."""

    context_tokens: int
    max_output_tokens: int

    def __post_init__(self) -> None:
        """
        Require positive capacities with output smaller than the context.

        Raises:
            ValueError: If either capacity is not a positive integer or the
                output capacity leaves no input capacity.

        """
        if type(self.context_tokens) is not int or self.context_tokens <= 0:
            msg = "Model context must be a positive integer"
            raise ValueError(msg)
        if (
            type(self.max_output_tokens) is not int
            or self.max_output_tokens <= 0
        ):
            msg = "Model output capacity must be a positive integer"
            raise ValueError(msg)
        if self.max_output_tokens >= self.context_tokens:
            msg = "Model output capacity must be smaller than its context"
            raise ValueError(msg)

    @property
    def input_upper_bound(self) -> int:
        """
        Return context tokens remaining after reserving maximum output.

        Returns:
            Input token capacity available to serialized request content.

        """
        return self.context_tokens - self.max_output_tokens

    def output_guard(
        self,
        output_reservation: int | None = None,
        *,
        enforceable_output_cap: bool = False,
    ) -> int:
        """Reserve maximum output unless the adapter can enforce a smaller cap.

        Args:
            output_reservation: Requested output tokens, or the model maximum.
            enforceable_output_cap: Whether the adapter enforces this request.

        Returns:
            Output tokens that must be held out of the context window.

        Raises:
            ValueError: If the reservation or enforcement flag is invalid.
        """
        requested = (
            self.max_output_tokens
            if output_reservation is None
            else output_reservation
        )
        if (
            type(requested) is not int
            or not 0 < requested <= self.max_output_tokens
        ):
            msg = "Output reservation must fit the model output capacity"
            raise ValueError(msg)
        if type(enforceable_output_cap) is not bool:
            msg = "Output cap enforcement must be a boolean"
            raise ValueError(msg)
        return requested if enforceable_output_cap else self.max_output_tokens

    def input_bound_for(
        self,
        output_reservation: int | None = None,
        *,
        enforceable_output_cap: bool = False,
    ) -> int:
        """Return input capacity after the adapter's enforceable output guard."""
        return self.context_tokens - self.output_guard(
            output_reservation, enforceable_output_cap=enforceable_output_cap
        )


@lru_cache(maxsize=16)
def _model_encoder(model: str) -> Encoding | None:
    """Load a locally supported model encoding or retain the byte fallback.

    Args:
        model: Provider model identifier used by ``tiktoken``'s model registry.

    Returns:
        The locally available encoding, or ``None`` for an unknown model or an
        unavailable public vocabulary file. Failures are cached so repeated
        requests do not repeatedly try to load missing encodings.
    """
    try:
        import tiktoken

        return tiktoken.encoding_for_model(model)
    except Exception:  # noqa: BLE001 # Unknown models and unavailable vocabularies use bytes.
        return None


def count_input_tokens(
    request_body: bytes, *, model: str | None = None
) -> int:
    """Count body tokens locally when the selected model has a known encoding.

    The result counts serialized request text only. It does not include message
    framing, provider overhead, or output tokens. Token counts are estimates of
    body text, not provider-reported usage.

    Args:
        request_body: UTF-8 serialized provider request.
        model: Model ID supported by the installed ``tiktoken`` registry; an
            unknown or omitted model uses the UTF-8 byte length.

    Returns:
        The ordinary-text token count, or UTF-8 byte count when the model is
        unknown, the vocabulary cannot load, or the request is not valid UTF-8.

    Raises:
        TypeError: If ``request_body`` is not bytes.
    """
    if not isinstance(request_body, bytes):
        msg = "The serialized provider request must be bytes"
        raise TypeError(msg)
    try:
        text = request_body.decode("utf-8")
    except UnicodeDecodeError:
        return len(request_body)
    if model is None:
        return len(request_body)
    encoder = _model_encoder(model)
    if encoder is None:
        return len(request_body)
    try:
        return len(encoder.encode_ordinary(text))
    except Exception:  # noqa: BLE001 # Tokenizer failures retain the safe byte estimate.
        return len(request_body)


def estimate_input(
    request_body: bytes,
    *,
    overhead_bytes: int = 0,
    model: str | None = None,
) -> int:
    """
    Estimate a conservative input-token bound from serialized request size.

    Uses a locally available model tokenizer for UTF-8 bodies and otherwise
    treats each byte as a token. It adds the existing framing margin and is
    intended for request packing, not as provider-reported usage.

    Args:
        request_body: Serialized provider request.
        overhead_bytes: Additional provider-specific framing bytes to reserve.
        model: Model ID used for local tokenization when recognized.

    Returns:
        The estimated body tokens plus fixed framing and caller-provided margins.

    Raises:
        TypeError: If ``request_body`` is not bytes.
        ValueError: If ``overhead_bytes`` is not a non-negative integer.

    """
    if not isinstance(request_body, bytes):
        msg = "The serialized provider request must be bytes"
        raise TypeError(msg)
    if type(overhead_bytes) is not int or overhead_bytes < 0:
        msg = "Input overhead must be a non-negative integer"
        raise ValueError(msg)
    return (
        count_input_tokens(request_body, model=model)
        + INPUT_FRAMING_MARGIN
        + overhead_bytes
    )


class RunUsage:
    """Accumulate token counts reported by providers without enforcing budgets."""

    def __init__(self) -> None:
        """Initialize zeroed token totals and call counters."""
        self._lock = threading.Lock()
        self._input_tokens = 0
        self._cached_input_tokens = 0
        self._output_tokens = 0
        self._calls = 0
        self._unreported_calls = 0

    def record(self, usage: dict[str, Any] | None) -> None:
        """
        Count one provider call and accumulate valid reported token usage.

        Invalid or absent usage still counts as a call and increments the
        unreported-call count; it does not change token totals.

        Args:
            usage: Provider token counts, or ``None`` when the call reported
                no usage. Valid mappings contain non-negative integer input
                and output counts and cached input no greater than input.

        """
        with self._lock:
            self._calls += 1
            if usage is None:
                self._unreported_calls += 1
                return

            input_tokens = usage.get("input_tokens")
            cached_input_tokens = usage.get("cached_input_tokens", 0)
            output_tokens = usage.get("output_tokens")
            if (
                type(input_tokens) is not int
                or input_tokens < 0
                or type(cached_input_tokens) is not int
                or not 0 <= cached_input_tokens <= input_tokens
                or type(output_tokens) is not int
                or output_tokens < 0
            ):
                self._unreported_calls += 1
                return

            self._input_tokens += input_tokens
            self._cached_input_tokens += cached_input_tokens
            self._output_tokens += output_tokens

    def report(self) -> dict[str, int]:
        """
        Return a snapshot of accumulated token totals and call counts.

        Returns:
            A new mapping with input, cached input, output, total call, and
            unreported call counts. Mutating it does not change this accumulator.

        """
        with self._lock:
            return {
                "input_tokens": self._input_tokens,
                "cached_input_tokens": self._cached_input_tokens,
                "output_tokens": self._output_tokens,
                "calls": self._calls,
                "unreported_calls": self._unreported_calls,
            }
