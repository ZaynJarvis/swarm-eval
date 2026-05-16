"""OpenCode CLI runtime backend."""
from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import tempfile

from .base import Backend, CallResult
from .codex import format_agent_prompt


def resolve_opencode_command() -> str:
    explicit = os.environ.get("OPENCODE_BIN")
    if explicit and os.path.isfile(explicit):
        return explicit
    found = shutil.which("opencode")
    if found:
        return found
    raise RuntimeError("opencode binary not found on PATH.")


def _clip(s: str, limit: int) -> str:
    if len(s) <= limit:
        return s
    return s[:limit] + f"... [truncated {len(s) - limit} chars]"


_ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def _content_to_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    if isinstance(value, dict):
        text = value.get("text") or value.get("content") or value.get("message")
        if isinstance(text, str):
            return text
    return ""


def parse_opencode_output(blob: str) -> str:
    """Extract assistant text from OpenCode JSON event output.

    OpenCode versions differ slightly between `run --format json` and ACP event
    shapes, so keep this parser deliberately permissive.
    """
    chunks: list[str] = []
    for raw_line in blob.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue

        update = obj.get("params", {}).get("update") if isinstance(obj.get("params"), dict) else None
        if isinstance(update, dict) and update.get("sessionUpdate") == "agent_message_chunk":
            chunks.append(_content_to_text(update.get("content")))
            continue

        for key in ("message", "text", "content", "result", "output"):
            text = _content_to_text(obj.get(key))
            if text:
                chunks.append(text)
                break

        if obj.get("type") in {"assistant", "assistant_message"}:
            text = _content_to_text(obj.get("data") or obj.get("message") or obj.get("content"))
            if text:
                chunks.append(text)

    if chunks:
        return "".join(chunks).strip()
    return _ANSI_RE.sub("", blob).strip()


class OpencodeBackend(Backend):
    name = "opencode"

    def __init__(
        self,
        *,
        command: str | None = None,
        timeout_s: float = 300.0,
        error_log_chars: int = 1200,
        cwd: str | None = None,
    ) -> None:
        self.command = command or resolve_opencode_command()
        self.timeout_s = timeout_s
        self.error_log_chars = error_log_chars
        self.cwd = cwd or os.getcwd()

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
        del cache_system
        prompt = format_agent_prompt(
            role=role,
            system=system,
            user=user,
            max_tokens=max_tokens,
        )
        with tempfile.TemporaryDirectory(prefix="swarm-opencode-") as td:
            prompt_arg = prompt
            prompt_file = ""
            if len(prompt) > 30000:
                prompt_file = os.path.join(td, "prompt.md")
                with open(prompt_file, "w", encoding="utf-8") as f:
                    f.write(prompt)
                prompt_arg = (
                    "Analyze the attached prompt.md. Treat its SYSTEM section as "
                    "higher-priority instructions and return only the requested JSON."
                )

            args = [
                self.command,
                "run",
                "--pure",
                "--format", "json",
                "--dir", self.cwd,
            ]
            if model:
                args.extend(["--model", model])
            if prompt_file:
                args.extend(["--file", prompt_file])
            args.append(prompt_arg)

            env = os.environ.copy()
            env.setdefault("NO_COLOR", "1")
            env.setdefault("FORCE_COLOR", "0")

            try:
                proc = await asyncio.create_subprocess_exec(
                    *args,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=env,
                    cwd=self.cwd,
                )
            except FileNotFoundError as e:
                return CallResult(text="", backend=self.name, error=f"spawn failed: {e}")

            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=self.timeout_s)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return CallResult(text="", backend=self.name, error="timeout")

            stdout_text = stdout.decode(errors="ignore")
            stderr_text = stderr.decode(errors="ignore")
            if proc.returncode != 0:
                return CallResult(
                    text="",
                    backend=self.name,
                    error=(
                        f"exit {proc.returncode}: "
                        f"stdout={_clip(stdout_text, self.error_log_chars)!r} "
                        f"stderr={_clip(stderr_text, self.error_log_chars)!r}"
                    ),
                )
            return CallResult(text=parse_opencode_output(stdout_text), backend=self.name)
