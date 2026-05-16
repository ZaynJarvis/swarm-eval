"""Link transcript JSONL files to qa_results.csv rows.

Each transcript represents one QA session keyed by sessionId (UUID).
Each qa_results row is keyed by (sample_id, question_index).
The mapping is recovered by matching the FIRST user message in the transcript
against (sample_id, question) where sample_id is read from the transcript's
`cwd` field (`.../<sample_id>/`).
"""
from __future__ import annotations

import csv
import json
import os
import re
import sys
from dataclasses import dataclass, field
from glob import glob
from typing import Iterator

csv.field_size_limit(sys.maxsize)

QUESTION_LINE_RE = re.compile(
    r"Answer the question directly:\s*(?P<q>.+)",
    re.IGNORECASE,
)


@dataclass
class TranscriptEvents:
    session_id: str
    cwd: str
    sample_id: str
    question_text: str | None
    events: list[dict] = field(default_factory=list)


@dataclass
class LinkedSession:
    session_id: str
    sample_id: str
    question_index: int
    question: str
    gold_answer: str
    category: str
    response: str
    result: str  # CORRECT | WRONG
    grader_reasoning: str
    num_turns: int
    elapsed_seconds: float
    total_cost_usd: float
    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int
    ov_recall_hooks: int
    ov_mcp_calls: int
    transcript_path: str
    transcript_mcp_calls: int = 0


def parse_transcript(path: str) -> TranscriptEvents:
    events: list[dict] = []
    cwd = ""
    question = None
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            events.append(e)
            if not cwd and isinstance(e.get("cwd"), str):
                cwd = e["cwd"]
            if question is None and e.get("type") == "user":
                msg = e.get("message", {})
                content = msg.get("content")
                if isinstance(content, str):
                    m = QUESTION_LINE_RE.search(content)
                    question = m.group("q").strip() if m else content.strip()
    sample_id = os.path.basename(cwd.rstrip("/")) if cwd else ""
    sid = os.path.splitext(os.path.basename(path))[0]
    return TranscriptEvents(
        session_id=sid, cwd=cwd, sample_id=sample_id,
        question_text=question, events=events,
    )


def load_qa_index(qa_csv_path: str) -> dict[tuple[str, str], list[dict]]:
    """Index qa_results rows by (sample_id, normalized question).

    A few benchmark rows intentionally repeat the same question text at
    different question_index values, so each key maps to a list instead of one
    row. The linker assigns duplicate transcripts to duplicate rows
    deterministically.
    """
    out: dict[tuple[str, str], list[dict]] = {}
    with open(qa_csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["sample_id"], _norm(row["question"]))
            out.setdefault(key, []).append(row)
    return out


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def iter_linked(result_dir: str, limit: int | None = None) -> Iterator[LinkedSession]:
    qa_csv = os.path.join(result_dir, "qa_results.csv")
    qa_index = load_qa_index(qa_csv)
    transcripts = sorted(glob(os.path.join(result_dir, "transcripts", "*.jsonl")))
    transcripts_by_question: dict[tuple[str, str], list[tuple[str, TranscriptEvents]]] = {}
    for tp in transcripts:
        try:
            te = parse_transcript(tp)
        except Exception as e:
            print(f"[linker] parse fail {tp}: {e}", file=sys.stderr)
            continue
        if not te.sample_id or not te.question_text:
            continue
        key = (te.sample_id, _norm(te.question_text))
        if key not in qa_index:
            continue
        transcripts_by_question.setdefault(key, []).append((tp, te))

    n = 0
    selected: list[tuple[int, str, TranscriptEvents, dict]] = []
    for key, rows in qa_index.items():
        candidates = transcripts_by_question.get(key, [])
        if not candidates:
            continue
        rows_sorted = sorted(rows, key=lambda r: int(r["question_index"]))
        remaining = list(candidates)
        for row in rows_sorted:
            if not remaining:
                break
            ranked = sorted(
                (
                    (_score_transcript_for_row(te, row), tp, te)
                    for tp, te in remaining
                ),
                key=lambda x: (-x[0], x[1]),
            )
            score, tp, te = ranked[0]
            selected.append((score, tp, te, row))
            remaining = [(rtp, rte) for rtp, rte in remaining if rtp != tp]
    selected.sort(key=lambda x: x[1])

    for _, tp, te, row in selected:
        if limit is not None and n >= limit:
            return
        try:
            yield LinkedSession(
                session_id=te.session_id,
                sample_id=te.sample_id,
                question_index=int(row["question_index"]),
                question=row["question"],
                gold_answer=row["answer"],
                category=row["category"],
                response=row["response"],
                result=row["result"],
                grader_reasoning=row["reasoning"],
                num_turns=int(row["num_turns"]),
                elapsed_seconds=float(row["elapsed_seconds"]),
                total_cost_usd=float(row["total_cost_usd"]),
                input_tokens=int(row["input_tokens"]),
                output_tokens=int(row["output_tokens"]),
                cache_read_input_tokens=int(row["cache_read_input_tokens"] or 0),
                ov_recall_hooks=int(row["ov_recall_hooks"] or 0),
                ov_mcp_calls=int(row["ov_mcp_calls"] or 0),
                transcript_path=tp,
                transcript_mcp_calls=_count_transcript_mcp_calls(te.events),
            )
        except (KeyError, ValueError) as e:
            print(f"[linker] row coerce fail {tp}: {e}", file=sys.stderr)
            continue
        n += 1


def _last_assistant_text(events: list[dict]) -> str:
    last = ""
    for e in events:
        if e.get("type") != "assistant":
            continue
        content = (e.get("message") or {}).get("content", [])
        if isinstance(content, list):
            for c in content:
                if c.get("type") == "text":
                    last = c.get("text", "") or last
        elif isinstance(content, str):
            last = content
    return last


def _count_transcript_mcp_calls(events: list[dict]) -> int:
    count = 0
    for e in events:
        if e.get("type") != "assistant":
            continue
        content = (e.get("message") or {}).get("content", [])
        if not isinstance(content, list):
            continue
        for c in content:
            if c.get("type") == "tool_use" and str(c.get("name", "")).startswith("mcp__"):
                count += 1
    return count


def _score_transcript_for_row(te: TranscriptEvents, row: dict) -> int:
    expected = _norm(row.get("response", ""))
    actual = _norm(_last_assistant_text(te.events))
    if not actual or not expected:
        return 0
    if actual == expected:
        return 4
    if actual in expected or expected in actual:
        return 3
    actual_head = actual[:120]
    expected_head = expected[:120]
    if actual_head and actual_head == expected_head:
        return 2
    if actual_head and actual_head in expected:
        return 1
    return 0


def render_session_payload(
    linked: LinkedSession,
    max_chars: int = 12000,
    cost_context: str | None = None,
) -> str:
    """Render the transcript into a compact payload for the worker prompt."""
    lines: list[str] = []
    lines.append(f"sample_id: {linked.sample_id}")
    lines.append(f"category: {linked.category}")
    lines.append(f"question: {linked.question}")
    lines.append(f"gold_answer: {linked.gold_answer}")
    lines.append(f"model_response: {linked.response}")
    lines.append(f"grader_verdict: {linked.result}")
    lines.append(f"grader_reasoning: {linked.grader_reasoning}")
    lines.append(
        f"runtime: turns={linked.num_turns} elapsed_s={linked.elapsed_seconds:.1f} "
        f"cost=${linked.total_cost_usd:.4f} hooks={linked.ov_recall_hooks} "
        f"mcp_metric={linked.ov_mcp_calls} transcript_mcp={linked.transcript_mcp_calls} "
        f"input_tokens={linked.input_tokens} cache_read_tokens={linked.cache_read_input_tokens} "
        f"output_tokens={linked.output_tokens}"
    )
    if cost_context:
        lines.append(f"cost_context: {cost_context}")
    lines.append("")
    lines.append("=== TRANSCRIPT (assistant turns + tool_use names + tool_result snippets) ===")
    with open(linked.transcript_path, encoding="utf-8") as f:
        events = [json.loads(l) for l in f if l.strip()]
    for i, e in enumerate(events):
        et = e.get("type")
        msg = e.get("message", {}) or {}
        if et == "assistant":
            content = msg.get("content", [])
            if isinstance(content, list):
                for c in content:
                    if c.get("type") == "text":
                        lines.append(f"[{i}] assistant.text: {c.get('text','')[:600]}")
                    elif c.get("type") == "tool_use":
                        inp = json.dumps(c.get("input", {}))[:200]
                        lines.append(f"[{i}] assistant.tool_use({c.get('name')}): {inp}")
            elif isinstance(content, str):
                lines.append(f"[{i}] assistant.text: {content[:600]}")
        elif et == "user":
            content = msg.get("content", [])
            if isinstance(content, list):
                for c in content:
                    if c.get("type") == "tool_result":
                        body = c.get("content", "")
                        body_str = body if isinstance(body, str) else json.dumps(body)
                        lines.append(f"[{i}] tool_result: {body_str[:400]}")
        elif et == "attachment":
            att = e.get("attachment", {})
            if att.get("type") == "hook_additional_context":
                content = att.get("content", "")
                content_str = content if isinstance(content, str) else json.dumps(content)
                lines.append(f"[{i}] hook_context: {content_str[:800]}")
    payload = "\n".join(lines)
    if len(payload) > max_chars:
        payload = payload[:max_chars] + "\n... [TRUNCATED]"
    return payload


if __name__ == "__main__":
    rd = sys.argv[1] if len(sys.argv) > 1 else "/Users/bytedance/Downloads/result-ov-r6"
    n = 0
    correct = wrong = 0
    for ls in iter_linked(rd, limit=50):
        n += 1
        if ls.result == "CORRECT":
            correct += 1
        elif ls.result == "WRONG":
            wrong += 1
    print(f"linked {n} sessions: correct={correct} wrong={wrong}")
