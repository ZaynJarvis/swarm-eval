"""Coordinator — Opus reflection that evolves the pattern library."""
from __future__ import annotations

import json
import os
import random
import re
from dataclasses import dataclass

from backends.base import Backend, CallResult
from pattern_store import Pattern, PatternStore


PROMPT_PATH = os.path.join(os.path.dirname(__file__), "prompts", "coordinator.md")


@dataclass
class ReflectionOutcome:
    parsed: dict | None
    raw_text: str
    call: CallResult
    parse_error: str | None = None
    applied_ops: int = 0
    resolved_escs: int = 0
    writebacks: int = 0


def load_template() -> str:
    with open(PROMPT_PATH, encoding="utf-8") as f:
        return f.read()


def _strip_code_fence(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\n?", "", s)
        s = re.sub(r"\n?```\s*$", "", s)
    return s.strip()


def _parse(raw: str) -> tuple[dict | None, str | None]:
    s = _strip_code_fence(raw)
    try:
        return json.loads(s), None
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", s, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0)), None
            except json.JSONDecodeError as e:
                return None, f"json decode: {e}"
        return None, "no JSON object found"


def _dump_patterns(store: PatternStore) -> str:
    rows = store.all_patterns()
    if not rows:
        return "(empty)"
    return "\n".join(
        f"- {r['signature']} | {r['category']} | n={r['count']} | "
        f"symptom: {r['symptom']} | rule: {r['rule_for_workers']}"
        for r in rows
    )


def _dump_escalations(store: PatternStore) -> tuple[str, list[int]]:
    rows = store.unresolved_escalations()
    if not rows:
        return "(none)", []
    lines = [f"- esc#{r['id']} session={r['session_id']}: {r['worker_question']}" for r in rows]
    return "\n".join(lines), [r["id"] for r in rows]


def _sample_recent(store: PatternStore, k: int = 12) -> str:
    cur = store.conn.cursor()
    cur.execute(
        "SELECT session_id, qa_sample_id, verdict, primary_pattern, severity, "
        "confidence, worker_notes FROM session_results ORDER BY written_at DESC LIMIT 60"
    )
    rows = cur.fetchall()
    if not rows:
        return "(none)"
    rows = list(rows)
    random.shuffle(rows)
    pick = rows[:k]
    return "\n".join(
        f"- {r['session_id'][:8]} {r['qa_sample_id']} verdict={r['verdict']} "
        f"pat={r['primary_pattern']} sev={r['severity']} conf={r['confidence']:.2f} "
        f"notes: {r['worker_notes']}"
        for r in pick
    )


def build_reflection_prompt(*, store: PatternStore, run_context: str,
                            pattern_version: int) -> tuple[str, str, list[int]]:
    template = load_template()
    pattern_dump = _dump_patterns(store)
    esc_text, esc_ids = _dump_escalations(store)
    sample = _sample_recent(store)
    system = (
        template
        .replace("{RUN_CONTEXT}", run_context)
        .replace("{PATTERN_VERSION}", str(pattern_version))
        .replace("{PATTERN_DUMP}", pattern_dump)
        .replace("{ESCALATIONS}", esc_text)
        .replace("{RECENT_RESULTS_SAMPLE}", sample)
    )
    user = "Produce the JSON now."
    return system, user, esc_ids


def apply_reflection(*, store: PatternStore, parsed: dict) -> tuple[int, int, int]:
    """Returns (applied_ops, resolved_escs, sessions_writeback)."""
    applied = 0
    for op in parsed.get("ops", []) or []:
        kind = op.get("kind")
        try:
            if kind == "add_pattern":
                store.upsert_pattern(Pattern(
                    signature=op["signature"],
                    category=op.get("category", "accuracy"),
                    symptom=op.get("symptom", ""),
                    rule_for_workers=op.get("rule_for_workers", ""),
                ))
                applied += 1
            elif kind == "refine_rule":
                cur = store.conn.cursor()
                cur.execute(
                    "UPDATE patterns SET symptom=?, rule_for_workers=?, last_updated=datetime('now') "
                    "WHERE signature=?",
                    (op.get("symptom", ""), op.get("rule_for_workers", ""), op["signature"]),
                )
                store.conn.commit()
                applied += 1
            elif kind == "merge_pattern":
                store.merge_patterns(op["src"], op["dst"])
                applied += 1
            elif kind == "dismiss":
                store.dismiss_pattern(op["signature"])
                applied += 1
        except Exception as e:
            print(f"[coordinator] op {kind} failed: {e}")

    resolved = 0
    writebacks = 0
    for r in parsed.get("escalation_resolutions", []) or []:
        try:
            esc_id = int(r["escalation_id"])
            sig = r.get("pattern_signature")
            pid = None
            if sig:
                cur = store.conn.cursor()
                cur.execute("SELECT id FROM patterns WHERE signature=?", (sig,))
                row = cur.fetchone()
                if row:
                    pid = row["id"]
            store.resolve_escalation(esc_id, r.get("resolution", ""), pid)
            resolved += 1

            # 1a — coordinator writeback: when resolving an escalation, override
            # the originating session's classification with the (possibly newly
            # created) pattern. This is cheaper than re-running the worker.
            if sig:
                cur = store.conn.cursor()
                cur.execute("SELECT session_id FROM escalations WHERE id=?", (esc_id,))
                erow = cur.fetchone()
                if erow:
                    sid = erow["session_id"]
                    cur.execute(
                        "UPDATE session_results "
                        "SET primary_pattern=?, worker_notes=?, written_at=datetime('now') "
                        "WHERE session_id=?",
                        (
                            sig,
                            "[coordinator-writeback] " + (r.get("resolution", "") or ""),
                            sid,
                        ),
                    )
                    if cur.rowcount > 0:
                        writebacks += 1
                        # Look up sample/qix for richer evidence display
                        cur.execute(
                            "SELECT qa_sample_id, qa_question_index FROM session_results WHERE session_id=?",
                            (sid,),
                        )
                        srow = cur.fetchone()
                        ev_item = {"session_id": sid, "via": "coordinator_writeback"}
                        if srow:
                            ev_item["sample_id"] = srow["qa_sample_id"]
                            ev_item["qix"] = srow["qa_question_index"]
                        store.bump_pattern_count(sig, ev_item)
                    store.conn.commit()
        except Exception as e:
            print(f"[coordinator] resolve esc failed: {e}")
    return applied, resolved, writebacks


async def run_reflection(
    *, backend: Backend, model: str, store: PatternStore,
    run_context: str, pattern_version: int, max_tokens: int = 4096,
) -> ReflectionOutcome:
    system, user, esc_ids = build_reflection_prompt(
        store=store, run_context=run_context, pattern_version=pattern_version,
    )
    call = await backend.call(
        model=model,
        system=system,
        user=user,
        max_tokens=max_tokens,
        role="coordinator",
    )
    if call.error:
        return ReflectionOutcome(None, "", call, parse_error=call.error)
    parsed, perr = _parse(call.text)
    if parsed is None:
        return ReflectionOutcome(None, call.text, call, parse_error=perr)
    applied, resolved, writebacks = apply_reflection(store=store, parsed=parsed)
    return ReflectionOutcome(parsed, call.text, call,
                             applied_ops=applied, resolved_escs=resolved,
                             writebacks=writebacks)
