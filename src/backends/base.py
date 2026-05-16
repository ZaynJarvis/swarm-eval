"""Agent runtime interface.

The orchestrator intentionally talks to "agent runtimes" rather than directly
to a model vendor. A runtime can be Anthropic SDK, Claude Code, Codex, or
OpenCode as long as it can run one isolated worker/coordinator turn and return
the final text.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CallResult:
    text: str
    input_tokens: int = 0
    cache_read_tokens: int = 0
    output_tokens: int = 0
    backend: str = ""
    error: str | None = None


class Backend:
    name: str = "base"
    supports_hard_token_cap: bool = False

    async def call(
        self,
        *,
        model: str,
        system: str,
        user: str,
        max_tokens: int = 2048,
        cache_system: bool = True,
        role: str = "worker",
    ) -> CallResult:
        raise NotImplementedError
