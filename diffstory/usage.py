"""Model-context packing and provider-reported narration usage."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


INPUT_FRAMING_MARGIN = 1024


@dataclass(frozen=True)
class ModelCapacity:
    """Keep each serialized request within the selected model's context window."""

    context_tokens: int
    max_output_tokens: int

    def __post_init__(self) -> None:
        """Require positive capacities with output smaller than the context.

        Raises:
            ValueError: If either capacity is not a positive integer or the
                output capacity leaves no input capacity.
        """
        if type(self.context_tokens) is not int or self.context_tokens <= 0:
            raise ValueError("Model context must be a positive integer")
        if type(self.max_output_tokens) is not int or self.max_output_tokens <= 0:
            raise ValueError("Model output capacity must be a positive integer")
        if self.max_output_tokens >= self.context_tokens:
            raise ValueError("Model output capacity must be smaller than its context")

    @property
    def input_upper_bound(self) -> int:
        """Return context tokens remaining after reserving maximum output.

        Returns:
            Input token capacity available to serialized request content.
        """
        return self.context_tokens - self.max_output_tokens


def estimate_input(request_body: bytes, *, overhead_bytes: int = 0) -> int:
    """Estimate a conservative input-token bound from serialized request size.

    The estimate treats each UTF-8 byte as a token and adds framing margin; it
    is intended for request packing, not as provider-reported usage.

    Args:
        request_body: Serialized provider request.
        overhead_bytes: Additional provider-specific framing bytes to reserve.

    Returns:
        The byte-based upper bound, including fixed and caller-provided margins.

    Raises:
        TypeError: If ``request_body`` is not bytes.
        ValueError: If ``overhead_bytes`` is not a non-negative integer.
    """
    if not isinstance(request_body, bytes):
        raise TypeError("The serialized provider request must be bytes")
    if type(overhead_bytes) is not int or overhead_bytes < 0:
        raise ValueError("Input overhead must be a non-negative integer")
    return len(request_body) + INPUT_FRAMING_MARGIN + overhead_bytes


class RunUsage:
    """Accumulate token counts reported by providers without enforcing budgets."""

    def __init__(self) -> None:
        """Initialize zeroed token totals and call counters."""
        self._input_tokens = 0
        self._cached_input_tokens = 0
        self._output_tokens = 0
        self._calls = 0
        self._unreported_calls = 0

    def record(self, usage: dict[str, Any] | None) -> None:
        """Count one provider call and accumulate valid reported token usage.

        Invalid or absent usage still counts as a call and increments the
        unreported-call count; it does not change token totals.

        Args:
            usage: Provider token counts, or ``None`` when the call reported
                no usage. Valid mappings contain non-negative integer input
                and output counts and cached input no greater than input.
        """
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
        """Return a snapshot of accumulated token totals and call counts.

        Returns:
            A new mapping with input, cached input, output, total call, and
            unreported call counts. Mutating it does not change this accumulator.
        """
        return {
            "input_tokens": self._input_tokens,
            "cached_input_tokens": self._cached_input_tokens,
            "output_tokens": self._output_tokens,
            "calls": self._calls,
            "unreported_calls": self._unreported_calls,
        }
