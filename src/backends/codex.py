"""Codex CLI runtime backend.

Uses `codex exec` as an isolated, non-interactive agent turn. This mirrors the
runtime family exposed by zouk-daemon without embedding the daemon itself.
"""
from __future__ import annotations

import asyncio
import os
import shutil
import tempfile

from .base import Backend, CallResult


def resolve_codex_command() -> str:
    explicit = os.environ.get("CODEX_BIN")
    if explicit and os.path.isfile(explicit):
        return explicit
    found = shutil.which("codex")
    if found:
        return found
    candidate = "/Applications/Codex.app/Contents/Resources/codex"
    if os.path.isfile(candidate):
        return candidate
    raise RuntimeError("codex binary not found on PATH or in standard locations.")


def _clip(s: str, limit: int) -> str:
    if len(s) <= limit:
        return s
    return s[:limit] + f"... [truncated {len(s) - limit} chars]"


def format_agent_prompt(*, role: str, system: str, user: str, max_tokens: int) -> str:
    return (
        f"You are running as the swarm-eval {role} runtime.\n"
        "Treat the following SYSTEM section as higher-priority instructions.\n"
        "Do not edit files, run shell commands, or use external tools. Return only the requested JSON.\n"
        f"Keep the final answer within {max_tokens} output tokens.\n\n"
        "SYSTEM:\n"
        f"{system}\n\n"
        "USER:\n"
        f"{user}"
    )


class CodexBackend(Backend):
    name = "codex"

    def __init__(
        self,
        *,
        command: str | None = None,
        timeout_s: float = 300.0,
        error_log_chars: int = 1200,
        cwd: str | None = None,
    ) -> None:
        self.command = command or resolve_codex_command()
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

        with tempfile.TemporaryDirectory(prefix="swarm-codex-") as td:
            out_path = os.path.join(td, "last-message.txt")
            effort_env = (
                "SWARM_CODEX_COORDINATOR_REASONING"
                if role == "coordinator"
                else "SWARM_CODEX_WORKER_REASONING"
            )
            reasoning_effort = os.environ.get(
                effort_env,
                "medium" if role == "coordinator" else "low",
            )
            args = [
                self.command,
                "exec",
                "--skip-git-repo-check",
                "--ephemeral",
                "--ignore-rules",
                "--color", "never",
                "--sandbox", "read-only",
                "--config", 'approval_policy="never"',
                "--config", f'model_reasoning_effort="{reasoning_effort}"',
                "--output-last-message", out_path,
            ]
            if model:
                args.extend(["--model", model])
            args.append("-")

            env = os.environ.copy()
            env.setdefault("NO_COLOR", "1")
            env.setdefault("FORCE_COLOR", "0")

            try:
                proc = await asyncio.create_subprocess_exec(
                    *args,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=env,
                    cwd=self.cwd,
                )
            except FileNotFoundError as e:
                return CallResult(text="", backend=self.name, error=f"spawn failed: {e}")

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(prompt.encode()),
                    timeout=self.timeout_s,
                )
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

            text = ""
            if os.path.exists(out_path):
                with open(out_path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
            if not text.strip():
                text = stdout_text
            return CallResult(text=text.strip(), backend=self.name)
