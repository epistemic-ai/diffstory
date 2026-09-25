"""Conservative request sizing and a run-wide model-call budget.

The byte count is deliberately an upper bound, not a tokenizer estimate: a
token cannot encode more UTF-8 bytes than the payload contains. Framing and
provider-specific overhead cover prompt structure that is not in the JSON
body. Usage reported by a provider replaces the reservation when it is known.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any


INPUT_FRAMING_MARGIN = 1024


@dataclass(frozen=True)
class BudgetLimits:
    context_tokens: int = 1_050_000
    request_input_tokens: int = 40_000
    request_output_tokens: int = 4_000
    total_input_tokens: int = 256_000
    total_output_tokens: int = 40_000
    calls: int = 64
    seconds: float = 900.0

    def __post_init__(self) -> None:
        for name in (
            "context_tokens",
            "request_input_tokens",
            "request_output_tokens",
            "total_input_tokens",
            "total_output_tokens",
            "calls",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.request_input_tokens + self.request_output_tokens > self.context_tokens:
            raise ValueError("Per-request token limits exceed the model context")
        if (
            isinstance(self.seconds, bool)
            or not isinstance(self.seconds, (int, float))
            or not math.isfinite(self.seconds)
            or self.seconds <= 0
        ):
            raise ValueError("seconds must be positive")


@dataclass(frozen=True)
class Reservation:
    id: int
    input_tokens: int
    output_tokens: int
    timeout_seconds: float


class RunBudget:
    """Authorize every provider attempt before dispatch and reconcile usage."""

    def __init__(self, limits: BudgetLimits, *, clock=time.monotonic) -> None:
        self.limits = limits
        self._clock = clock
        self._started = clock()
        self._next_id = 1
        self._reservations: dict[int, Reservation] = {}
        self._input_tokens = 0
        self._output_tokens = 0
        self._calls = 0

    def remaining_seconds(self) -> float:
        elapsed = self._clock() - self._started
        remaining = self.limits.seconds - elapsed
        if remaining <= 0:
            raise ValueError("Narration time budget exhausted")
        return remaining

    def estimate_input(
        self,
        request_body: bytes,
        *,
        overhead_bytes: int = 0,
    ) -> int:
        if not isinstance(request_body, bytes):
            raise TypeError("The serialized provider request must be bytes")
        if type(overhead_bytes) is not int or overhead_bytes < 0:
            raise ValueError("Input overhead must be a non-negative integer")
        return len(request_body) + INPUT_FRAMING_MARGIN + overhead_bytes

    def authorize(
        self,
        request_body: bytes,
        max_output_tokens: int,
        *,
        input_overhead_bytes: int = 0,
    ) -> Reservation:
        """Reserve the complete input bound and maximum output before a call.

        Each retry or repair is a new invocation and therefore needs a new
        reservation. Failed preflight leaves call count and reservations intact.
        """
        if type(max_output_tokens) is not int or max_output_tokens <= 0:
            raise ValueError("Maximum output tokens must be a positive integer")
        request_input_tokens = self.estimate_input(
            request_body,
            overhead_bytes=input_overhead_bytes,
        )
        if request_input_tokens > self.limits.request_input_tokens:
            raise ValueError(
                "Request input exceeds the per-request budget "
                f"({request_input_tokens} > "
                f"{self.limits.request_input_tokens} conservative tokens)"
            )
        request_output_tokens = max_output_tokens
        if request_output_tokens > self.limits.request_output_tokens:
            raise ValueError("Requested output exceeds the per-request output budget")
        if request_input_tokens + request_output_tokens > self.limits.context_tokens:
            raise ValueError("Request exceeds the selected model's context limit")
        if self._calls >= self.limits.calls:
            raise ValueError("Narration call budget exhausted")
        if self._input_tokens + request_input_tokens > self.limits.total_input_tokens:
            raise ValueError("Narration input-token budget exhausted")
        if self._output_tokens + request_output_tokens > self.limits.total_output_tokens:
            raise ValueError("Narration output-token budget exhausted")
        timeout_seconds = self.remaining_seconds()
        reservation = Reservation(
            self._next_id,
            request_input_tokens,
            request_output_tokens,
            timeout_seconds,
        )
        self._next_id += 1
        self._reservations[reservation.id] = reservation
        self._input_tokens += request_input_tokens
        self._output_tokens += request_output_tokens
        self._calls += 1
        return reservation

    def settle(self, reservation: Reservation, usage: dict[str, Any] | None) -> None:
        """Replace a reservation with actual usage, or retain it if unknown."""
        current = self._reservations.get(reservation.id)
        if current != reservation:
            raise ValueError("Unknown or already settled request reservation")
        if usage is None:
            del self._reservations[reservation.id]
            return
        input_tokens = usage.get("input_tokens")
        output_tokens = usage.get("output_tokens")
        if (
            type(input_tokens) is not int
            or input_tokens < 0
            or type(output_tokens) is not int
            or output_tokens < 0
        ):
            raise ValueError("Provider usage must contain non-negative input and output token counts")
        del self._reservations[reservation.id]
        self._input_tokens += input_tokens - reservation.input_tokens
        self._output_tokens += output_tokens - reservation.output_tokens
        if input_tokens > reservation.input_tokens or output_tokens > reservation.output_tokens:
            raise ValueError("Provider-reported usage exceeded the authorized request reservation")

    def cancel_before_dispatch(self, reservation: Reservation) -> None:
        """Release a reservation only when no provider request was dispatched."""
        current = self._reservations.pop(reservation.id, None)
        if current != reservation:
            raise ValueError("Unknown or already settled request reservation")
        self._input_tokens -= reservation.input_tokens
        self._output_tokens -= reservation.output_tokens
        self._calls -= 1

    def usage(self) -> dict[str, int | float]:
        elapsed = self._clock() - self._started
        return {
            "input_tokens": self._input_tokens,
            "output_tokens": self._output_tokens,
            "calls": self._calls,
            "elapsed_seconds": round(min(self.limits.seconds, elapsed), 3),
        }
