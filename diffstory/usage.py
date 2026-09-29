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
        if type(self.context_tokens) is not int or self.context_tokens <= 0:
            raise ValueError("Model context must be a positive integer")
        if type(self.max_output_tokens) is not int or self.max_output_tokens <= 0:
            raise ValueError("Model output capacity must be a positive integer")
        if self.max_output_tokens >= self.context_tokens:
            raise ValueError("Model output capacity must be smaller than its context")

    @property
    def input_upper_bound(self) -> int:
        return self.context_tokens - self.max_output_tokens


def estimate_input(request_body: bytes, *, overhead_bytes: int = 0) -> int:
    """Bound request tokens by serialized bytes plus provider framing overhead."""
    if not isinstance(request_body, bytes):
        raise TypeError("The serialized provider request must be bytes")
    if type(overhead_bytes) is not int or overhead_bytes < 0:
        raise ValueError("Input overhead must be a non-negative integer")
    return len(request_body) + INPUT_FRAMING_MARGIN + overhead_bytes


class RunUsage:
    """Accumulate token counts reported by providers without enforcing budgets."""

    def __init__(self) -> None:
        self._input_tokens = 0
        self._cached_input_tokens = 0
        self._output_tokens = 0
        self._calls = 0
        self._unreported_calls = 0

    def record(self, usage: dict[str, Any] | None) -> None:
        """Record one attempted provider call and any usage it reported."""
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
        """Return measured totals and the count of calls without usage data."""
        return {
            "input_tokens": self._input_tokens,
            "cached_input_tokens": self._cached_input_tokens,
            "output_tokens": self._output_tokens,
            "calls": self._calls,
            "unreported_calls": self._unreported_calls,
        }
