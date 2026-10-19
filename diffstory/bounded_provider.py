"""Run network adapters in a killable process under the narration deadline.

Socket timeouts bound individual I/O operations. This supervisor also bounds
slow response streams, DNS waits, and adapter cleanup. Credentials and request
bytes travel over stdin; neither is added to argv or written to a state file.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

from .compact_packet import canonical_bytes
from .providers import MAX_RESPONSE_BYTES
from .providers import CodexCLIProvider
from .providers import OpenAIResponsesProvider
from .providers import ProviderResponseError
from .providers import _ProcessRequest
from .providers import _run_bounded_subprocess

if TYPE_CHECKING:
    from .providers import NarrativeProvider


def complete_bounded(
    provider: NarrativeProvider, body: bytes, timeout: float
) -> tuple[dict, dict | None]:
    """Supervise supported external adapters and retain measured reply usage.

    In-process fixture adapters must honor their timeout contract. Production
    adapters run in a separate process group, so a stalled transport cannot
    retain a worker or continue a Codex request after the run deadline.
    """
    if type(provider) not in {CodexCLIProvider, OpenAIResponsesProvider}:
        return provider.complete(body, timeout)
    deadline = time.monotonic() + timeout
    provider.require_credentials()
    if type(provider) is CodexCLIProvider:
        config = {
            "provider": "codex",
            "model": provider._requested_model,  # noqa: SLF001 # Private transport handoff, never persisted.
            "executable": provider._executable,  # noqa: SLF001 # Preserve the exact approved adapter executable.
        }
    else:
        config = {
            "provider": "openai",
            "api_key": provider._api_key,  # noqa: SLF001 # Credentials travel only over the private stdin pipe.
        }
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError
    # Leave process termination and pipe collection inside the same allowance.
    cleanup = min(2.0, remaining / 4)
    config["timeout"] = remaining - cleanup
    status, raw, _stderr = _run_bounded_subprocess(
        _ProcessRequest(
            command=[sys.executable, "-m", "diffstory.bounded_provider"],
            prompt=canonical_bytes(config) + b"\n" + body,
            cwd=str(Path(__file__).resolve().parent.parent),
            environment=os.environ.copy(),
            timeout=remaining - cleanup,
            output_limit=MAX_RESPONSE_BYTES + 4_096,
            cleanup_deadline=deadline,
        )
    )
    try:
        envelope = _decode_envelope(raw, status)
    except ProviderResponseError:
        raise
    except (ValueError, KeyError, TypeError):
        msg = "Narration provider process returned an invalid response"
        raise ProviderResponseError(msg) from None
    return envelope["response"], envelope["usage"]


def _decode_envelope(raw: bytes, status: int) -> dict:
    """Check the private child reply without forwarding child diagnostics."""
    envelope = json.loads(raw)
    if status != 0 or type(envelope) is not dict:
        raise ValueError
    if envelope.get("error") is True:
        msg = "Narration provider request failed"
        raise ProviderResponseError(
            msg,
            envelope.get("usage"),
            category=envelope["category"],
            reason=envelope["reason"],
        )
    if set(envelope) != {"response", "usage"}:
        raise ValueError
    if type(envelope["response"]) is not dict:
        raise ValueError
    return envelope


def _invoke(config: dict, body: bytes) -> dict:
    """Invoke the selected production adapter once without a nested session."""
    if config["provider"] == "codex":
        provider = CodexCLIProvider(
            model=config["model"], executable=config["executable"]
        )
        # The supervisor owns the group containing this process and Codex.
        # A second session would let Codex survive its supervisor's timeout.
        reply = provider.complete_reply(
            body, config["timeout"], isolate_process=False
        )
    elif config["provider"] == "openai":
        provider = OpenAIResponsesProvider(api_key=config["api_key"])
        reply = provider.complete_reply(body, config["timeout"])
    else:
        raise ValueError
    return {"response": reply.response, "usage": reply.usage}


def _worker() -> None:
    """Execute exactly one adapter call inside the parent's process group."""
    try:
        config = json.loads(sys.stdin.buffer.readline())
        result = _invoke(config, sys.stdin.buffer.read())
    except ProviderResponseError as error:
        result = {
            "error": True,
            "category": error.category,
            "reason": error.reason,
            "usage": error.usage,
        }
    except Exception:  # noqa: BLE001 # Child diagnostics must never expose source or credentials.
        result = {
            "error": True,
            "category": "provider_error",
            "reason": "transport",
            "usage": None,
        }
    sys.stdout.buffer.write(canonical_bytes(result))
    sys.stdout.buffer.flush()


if __name__ == "__main__":
    _worker()
