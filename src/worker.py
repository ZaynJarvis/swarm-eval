"""Worker — analyze one session, return structured JSON or an error."""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

from backends.base import Backend, CallResult
from linker import LinkedSession, render_session_payload


PROMPT_PATH = os.path.join(os.path.dirname(__file__), "prompts", "worker.md")


@dataclass
class WorkerOutcome:
    session_id: str
    parsed: dict | None
    raw_text: str
    call: CallResult
    parse_error: str | None = None


def load_template() -> str:
    with open(PROMPT_PATH, encoding="utf-8") as f:
        return f.read()


def build_system(template: str, *, pattern_version: int, patterns_fragment: str) -> str:
    return (
        template
        .replace("{PATTERN_VERSION}", str(pattern_version))
        .replace("{PATTERNS_FRAGMENT}", patterns_fragment)
    )


def _strip_code_fence(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\n?", "", s)
        s = re.sub(r"\n?```\s*$", "", s)
    return s.strip()


def parse_worker_output(raw: str) -> tuple[dict | None, str | None]:
    s = _strip_code_fence(raw)
    try:
        return json.loads(s), None
    except json.JSONDecodeError:
        # last-resort: take the largest brace block
        m = re.search(r"\{.*\}", s, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0)), None
            except json.JSONDecodeError as e:
                return None, f"json decode (brace-block): {e}"
        return None, "no JSON object found"


def validate_worker_output(parsed: dict, *, expected_session_id: str) -> str | None:
    required = {"session_id", "verdict", "primary_pattern", "secondary_patterns",
                "evidence", "severity", "confidence", "escalate", "worker_notes"}
    missing = required - parsed.keys()
    if missing:
        return f"missing keys: {sorted(missing)}"
    if parsed["verdict"] not in {"correct", "wrong"}:
        return f"bad verdict: {parsed['verdict']!r}"
    if parsed["severity"] not in {"low", "med", "high"}:
        return f"bad severity: {parsed['severity']!r}"
    try:
        c = float(parsed["confidence"])
    except (TypeError, ValueError):
        return "confidence not float"
    if not 0.0 <= c <= 1.0:
        return f"confidence out of range: {c}"
    if not isinstance(parsed["escalate"], bool):
        return "escalate not bool"
    if parsed["escalate"] and not parsed.get("escalate_question"):
        return "escalate=true requires escalate_question"
    if parsed["session_id"] != expected_session_id:
        # tolerate but warn — coordinator can still match by row
        parsed["session_id"] = expected_session_id
    return None


async def run_worker(
    *,
    backend: Backend,
    model: str,
    linked: LinkedSession,
    pattern_version: int,
    patterns_fragment: str,
    template: str | None = None,
    max_tokens: int = 700,
    cost_context: str | None = None,
) -> WorkerOutcome:
    template = template or load_template()
    system = build_system(template, pattern_version=pattern_version,
                          patterns_fragment=patterns_fragment)
    user = render_session_payload(linked, cost_context=cost_context)
    user = f"session_id: {linked.session_id}\n\n{user}"
    call = await backend.call(
        model=model,
        system=system,
        user=user,
        max_tokens=max_tokens,
        role="worker",
    )
    if call.error:
        return WorkerOutcome(linked.session_id, None, "", call, parse_error=call.error)
    parsed, perr = parse_worker_output(call.text)
    if parsed is None:
        return WorkerOutcome(linked.session_id, None, call.text, call, parse_error=perr)
    verr = validate_worker_output(parsed, expected_session_id=linked.session_id)
    if verr:
        return WorkerOutcome(linked.session_id, None, call.text, call, parse_error=verr)
    return WorkerOutcome(linked.session_id, parsed, call.text, call)
