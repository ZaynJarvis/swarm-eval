"""Runtime factory shared by CLI and tests."""
from __future__ import annotations

from .base import Backend


RUNTIME_CHOICES = (
    "sdk",
    "anthropic-sdk",
    "cli",
    "claude",
    "claude-code",
    "codex",
    "opencode",
)

_ALIASES = {
    "anthropic-sdk": "sdk",
    "cli": "claude-code",
    "claude": "claude-code",
}

_CLAUDE_WORKER_MODEL = "claude-haiku-4-5-20251001"
_CLAUDE_COORDINATOR_MODEL = "claude-opus-4-7"
_CODEX_WORKER_MODEL = "gpt-5.5"
_CODEX_COORDINATOR_MODEL = "gpt-5.5"


def normalize_runtime(name: str) -> str:
    normalized = name.strip().lower()
    return _ALIASES.get(normalized, normalized)


def default_model_for_runtime(runtime: str, role: str) -> str:
    runtime = normalize_runtime(runtime)
    if runtime in {"sdk", "claude-code"}:
        return _CLAUDE_COORDINATOR_MODEL if role == "coordinator" else _CLAUDE_WORKER_MODEL
    if runtime == "codex":
        return _CODEX_COORDINATOR_MODEL if role == "coordinator" else _CODEX_WORKER_MODEL
    # OpenCode should inherit the local runtime default unless the caller
    # supplies a model explicitly. Its installed provider varies by machine.
    return ""


def build_backend(runtime: str) -> Backend:
    runtime = normalize_runtime(runtime)
    if runtime == "sdk":
        try:
            from .sdk import SdkBackend
            return SdkBackend()
        except ImportError as e:
            raise RuntimeError(
                "sdk runtime unavailable: install the anthropic SDK with "
                "`uv pip install anthropic`."
            ) from e
        except RuntimeError as e:
            raise RuntimeError(f"sdk runtime unavailable: {e}") from e
    if runtime == "claude-code":
        from .cli import CliBackend
        return CliBackend(runtime_name=runtime)
    if runtime == "codex":
        from .codex import CodexBackend
        return CodexBackend()
    if runtime == "opencode":
        from .opencode import OpencodeBackend
        return OpencodeBackend()
    raise ValueError(
        f"unknown runtime: {runtime}. Available: {', '.join(RUNTIME_CHOICES)}"
    )
