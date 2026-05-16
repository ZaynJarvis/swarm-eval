"""Anthropic SDK backend — direct API calls."""
from __future__ import annotations

import os

from .base import Backend, CallResult


class SdkBackend(Backend):
    name = "sdk"
    supports_hard_token_cap = True

    def __init__(self) -> None:
        try:
            from anthropic import AsyncAnthropic  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "anthropic SDK not installed. Install with `uv pip install anthropic`."
            ) from e
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY not set; sdk backend unavailable.")
        self.client = AsyncAnthropic()

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
        del role
        if not model:
            return CallResult(text="", backend=self.name, error="model is required for sdk runtime")
        sys_param: list[dict] | str = system
        if cache_system:
            sys_param = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
        try:
            resp = await self.client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=sys_param,
                messages=[{"role": "user", "content": user}],
            )
        except Exception as e:
            return CallResult(text="", backend=self.name, error=f"{type(e).__name__}: {e}")
        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        usage = resp.usage
        input_tokens = getattr(usage, "input_tokens", 0) or 0
        cache_read_tokens = getattr(usage, "cache_read_input_tokens", 0) or 0
        output_tokens = getattr(usage, "output_tokens", 0) or 0
        return CallResult(
            text=text,
            input_tokens=input_tokens,
            cache_read_tokens=cache_read_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + cache_read_tokens + output_tokens,
            backend=self.name,
        )
