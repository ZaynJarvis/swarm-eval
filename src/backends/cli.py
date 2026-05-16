"""Claude CLI subprocess backend.

Spawns `claude -p --model <id> --system-prompt <sys> <user>` per call.
Two modes:
  - bare=True   uses ANTHROPIC_API_KEY, skips hooks/plugins/auto-memory (clean),
                lighter cold start. Recommended.
  - bare=False  uses OAuth subscription auth, but auto-discovers hooks/plugins
                which we don't want firing in a worker. Fallback only.

Mirrors zouk-daemon's spawn pattern (env injection, asyncio subprocess) but
without the chat MCP bridge — workers don't need chat tool access.
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil

from .base import Backend, CallResult


def resolve_claude_command() -> str:
    explicit = os.environ.get("CLAUDE_BIN")
    if explicit and os.path.isfile(explicit):
        return explicit
    found = shutil.which("claude")
    if found:
        return found
    candidates = [
        os.path.expanduser("~/Applications/Claude Code URL Handler.app/Contents/MacOS/claude"),
        "/Applications/Claude Code URL Handler.app/Contents/MacOS/claude",
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    raise RuntimeError("claude binary not found on PATH or in standard locations.")


def _clip(s: str, limit: int) -> str:
    if len(s) <= limit:
        return s
    return s[:limit] + f"... [truncated {len(s) - limit} chars]"


class CliBackend(Backend):
    name = "cli"

    def __init__(
        self,
        *,
        command: str | None = None,
        timeout_s: float = 240.0,
        error_log_chars: int = 1200,
        bare: bool | None = None,
        runtime_name: str = "cli",
    ) -> None:
        self.command = command or resolve_claude_command()
        self.timeout_s = timeout_s
        self.error_log_chars = error_log_chars
        self.name = runtime_name
        # bare=True needs ANTHROPIC_API_KEY; auto-detect.
        if bare is None:
            bare = bool(os.environ.get("ANTHROPIC_API_KEY"))
        self.bare = bare

    async def call(
        self,
        *,
        model: str,
        system: str,
        user: str,
        max_tokens: int = 2048,  # not enforced by CLI, kept for interface parity
        cache_system: bool = True,
        role: str = "worker",
    ) -> CallResult:
        del cache_system, role
        args = [
            self.command, "-p",
            "--system-prompt", system,
            "--output-format", "json",
            "--permission-mode", "bypassPermissions",
            "--no-session-persistence",
        ]
        if model:
            args.extend(["--model", model])
        if self.bare:
            args.append("--bare")
        args.append(user)

        env = os.environ.copy()
        env.setdefault("CLAUDE_DISABLE_TELEMETRY", "1")

        try:
            proc = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
        except FileNotFoundError as e:
            return CallResult(text="", backend=self.name, error=f"spawn failed: {e}")

        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=self.timeout_s)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return CallResult(text="", backend=self.name, error="timeout")

        if proc.returncode != 0:
            stdout_text = stdout.decode(errors="ignore")
            stderr_text = stderr.decode(errors="ignore")
            return CallResult(
                text="", backend=self.name,
                error=(
                    f"exit {proc.returncode}: "
                    f"stdout={_clip(stdout_text, self.error_log_chars)!r} "
                    f"stderr={_clip(stderr_text, self.error_log_chars)!r}"
                ),
            )
        return _parse_json_output(stdout.decode(errors="ignore"), backend_name=self.name)


def _parse_json_output(blob: str, *, backend_name: str) -> CallResult:
    blob = blob.strip()
    if not blob:
        return CallResult(text="", backend=backend_name, error="empty output")
    try:
        obj = json.loads(blob)
    except json.JSONDecodeError as e:
        return CallResult(text="", backend=backend_name,
                          error=f"json decode: {e}; head={blob[:200]!r}")
    text = ""
    if isinstance(obj.get("result"), str):
        text = obj["result"]
    elif isinstance(obj.get("text"), str):
        text = obj["text"]
    usage = obj.get("usage") or {}
    return CallResult(
        text=text,
        input_tokens=usage.get("input_tokens", 0) or 0,
        cache_read_tokens=usage.get("cache_read_input_tokens", 0) or 0,
        output_tokens=usage.get("output_tokens", 0) or 0,
        backend=backend_name,
    )
