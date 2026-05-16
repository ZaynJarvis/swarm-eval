"""Third-pass swarm analysis over tmux-journal logs.

This is intentionally separate from LoCoMo session analysis. It reuses the same
worker/coordinator runtime split to answer a different question: what recurring
workflow patterns and friction points are visible in a developer's tmux journal?
"""
from __future__ import annotations

import argparse
import asyncio
import difflib
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backends.base import Backend, CallResult
from backends.factory import build_backend, default_model_for_runtime, normalize_runtime


WORKER_SYSTEM = """You are a workflow-pattern analyst reviewing ONE tmux journal chunk.

Goal: extract recurring patterns that can become useful insight for Zayn's
engineering workflow and agent tooling. Focus on concrete repeated behavior,
friction, overload, missed automation, verification habits, and memory/journal
quality. Do not summarize every command.

The chunk may be selected by a script-level planner. Use planner metadata
(`planner_strategy`, source size, total entries, selected indices, noise ratio)
when judging coverage and when spotting overload/noise patterns.

Safety:
- Do not reveal secrets, tokens, API keys, credentials, or private hostnames.
- Evidence quotes must be short and sanitized.
- If the chunk is mostly noise, say so directly.

Output JSON only:
{
  "chunk_id": "<copy>",
  "source": "<copy>",
  "time_range": "<copy>",
  "patterns": [
    {
      "signature": "kebab-case",
      "category": "workflow_friction|debug_loop|context_switching|verification|automation|memory_quality|tooling|other",
      "severity": "low|med|high",
      "evidence": ["<=140 char sanitized quote", "..."],
      "insight": "one concrete sentence",
      "recommendation": "one concrete sentence"
    }
  ],
  "notable_events": ["short event", "..."],
  "confidence": 0.0
}
"""


COORDINATOR_SYSTEM = """You are the coordinator for a swarm of tmux-journal workers.

Aggregate worker JSON into a concise pattern library and user-facing insights.
Prefer recurring patterns across chunks over one-off events. Merge synonymous
signatures. Separate observed evidence from your interpretation.

Output JSON only:
{
  "top_patterns": [
    {
      "signature": "kebab-case",
      "category": "workflow_friction|debug_loop|context_switching|verification|automation|memory_quality|tooling|other",
      "count": 1,
      "severity": "low|med|high",
      "insight": "specific reusable insight",
      "evidence": ["short sanitized evidence", "..."],
      "recommendation": "specific process/tooling change"
    }
  ],
  "cross_cutting_insights": ["...", "..."],
  "recommended_experiments": ["...", "..."],
  "run_quality": {
    "coverage": "brief statement",
    "limitations": "brief statement"
  }
}
"""


SECRET_PATTERNS = [
    (re.compile(r"sk-[A-Za-z0-9_-]{12,}"), "sk-[REDACTED]"),
    (re.compile(r"(?i)(api[_-]?key|user[_-]?key|token|secret|password)\s*[:=]\s*['\"]?[^'\"\s]+"), r"\1=[REDACTED]"),
    (re.compile(r"(?i)\b(user[_-]?key|api[_-]?key|token|secret|password)\s+[0-9a-f]{32,}\b"), r"\1 [REDACTED]"),
    (re.compile(r"\b[0-9a-f]{40,}\b", re.I), "[REDACTED_HEX]"),
    (re.compile(r"(?i)authorization:\s*bearer\s+[A-Za-z0-9._-]+"), "authorization: bearer [REDACTED]"),
    (re.compile(r"AKIA[0-9A-Z]{16}"), "AKIA[REDACTED]"),
]

NOISE_MARKERS = (
    "Use /skills to list available skills",
    "Working (",
    "esc to interrupt",
    "Press enter to confirm",
    "Would you like to run the following command",
    "Yes, and don't ask again",
    "tokens used",
    "Received ping",
    "[TAILING] Tailing last",
)

KEY_SIGNAL_RE = re.compile(
    r"(?i)\b(error|failed|failure|exception|panic|timeout|denied|blocked|"
    r"commit|push|pull request|pr create|merge|test|typecheck|lint|build|"
    r"deploy|restart|pm2|git status|diff|verified|verification|fixed|done)\b"
)


@dataclass
class JournalChunk:
    chunk_id: str
    source: str
    pane_id: str
    pane_name: str
    first_ts: str
    last_ts: str
    entry_count: int
    total_entry_count: int
    source_bytes: int
    strategy: str
    chunk_part: int
    chunk_parts: int
    text: str


@dataclass
class JournalOutcome:
    chunk_id: str
    parsed: dict | None
    raw_text: str
    call: CallResult
    parse_error: str | None = None


def _redact(text: str) -> str:
    out = text
    for pattern, replacement in SECRET_PATTERNS:
        out = pattern.sub(replacement, out)
    return out


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [TRUNCATED {len(text) - limit} chars]"


def _entry_time(entry: str) -> datetime | None:
    m = re.match(r"^=== (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", entry)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def _split_entries(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"(?=^=== \d{4})", text, flags=re.MULTILINE) if p.strip()]


def _pane_id_from_path(path: Path) -> str:
    m = re.match(r"pane_(.+)\.log$", path.name)
    return m.group(1) if m else path.stem


def _pane_name(path: Path) -> str:
    name_path = path.with_suffix(".name")
    if not name_path.exists():
        return "unknown"
    return name_path.read_text(errors="replace").strip() or "unknown"


def _time_range(entries: list[str]) -> tuple[str, str, datetime | None]:
    times = [t for e in entries if (t := _entry_time(e)) is not None]
    if not times:
        return "unknown", "unknown", None
    return (
        min(times).strftime("%Y-%m-%d %H:%M:%S"),
        max(times).strftime("%Y-%m-%d %H:%M:%S"),
        max(times),
    )


def _trim_entries(entries: list[str], max_chars: int) -> list[str]:
    selected: list[str] = []
    total = 0
    for entry in reversed(entries):
        entry_len = len(entry)
        if selected and total + entry_len > max_chars:
            break
        selected.append(entry)
        total += entry_len
    return list(reversed(selected))


def _noise_ratio(text: str) -> float:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return 0.0
    noisy = 0
    for line in lines:
        if any(marker in line for marker in NOISE_MARKERS):
            noisy += 1
    return noisy / len(lines)


def _clip_entry(entry: str, limit: int) -> str:
    if len(entry) <= limit:
        return entry
    head = limit // 2
    tail = max(0, limit - head - 80)
    return (
        entry[:head]
        + f"\n... [ENTRY MIDDLE TRUNCATED {len(entry) - head - tail} chars] ...\n"
        + entry[-tail:]
    )


def _select_planned_entries(entries: list[str], max_chars: int) -> tuple[list[str], str, list[int]]:
    """Script-level planner: full small files, representative samples for large logs."""
    full_text = "\n\n".join(entries)
    if len(full_text) <= max_chars:
        return entries, "full", list(range(len(entries)))

    n = len(entries)
    indices: set[int] = {0, max(0, n // 4), max(0, n // 2), max(0, (3 * n) // 4), n - 1}
    signal_indices = [i for i, entry in enumerate(entries) if KEY_SIGNAL_RE.search(entry)]
    if signal_indices:
        step = max(1, len(signal_indices) // 8)
        indices.update(signal_indices[::step][:8])
        indices.update(signal_indices[-4:])

    ordered = sorted(i for i in indices if 0 <= i < n)
    if not ordered:
        return _trim_entries(entries, max_chars), "tail-fallback", []

    per_entry = max(600, max_chars // max(1, len(ordered)))
    selected: list[str] = []
    selected_indices: list[int] = []
    total = 0
    for idx in ordered:
        item = _clip_entry(entries[idx], per_entry)
        if selected and total + len(item) > max_chars:
            continue
        selected.append(item)
        selected_indices.append(idx)
        total += len(item)
    if not selected:
        selected = [_clip_entry(entries[-1], max_chars)]
        selected_indices = [n - 1]
    return selected, "planned-sample", selected_indices


def _split_entry(entry: str) -> tuple[str, list[str]]:
    lines = entry.splitlines()
    if not lines:
        return "(empty entry)", []
    return lines[0], lines[1:]


def _build_delta_full_records(entries: list[str]) -> list[tuple[int, str]]:
    """Represent every entry in order while collapsing repeated pane snapshots.

    tmux-journal captures whole pane snapshots, so adjacent entries are often
    cumulative. A unified diff against the previous snapshot preserves full
    coverage while avoiding re-sending the same screen hundreds of times.
    """
    records: list[tuple[int, str]] = []
    prev_body: list[str] | None = None
    for idx, entry in enumerate(entries):
        header, body = _split_entry(entry)
        if prev_body is None:
            payload = "\n".join(body)
            kind = "FULL_SNAPSHOT"
        else:
            diff = list(difflib.unified_diff(
                prev_body,
                body,
                fromfile=f"entry_{idx - 1}",
                tofile=f"entry_{idx}",
                lineterm="",
                n=2,
            ))
            payload = "\n".join(diff) if diff else "(no visible pane change)"
            kind = "DELTA_FROM_PREVIOUS"
        records.append((
            idx,
            (
                f"--- ENTRY {idx} {kind} ---\n"
                f"{header}\n"
                f"{payload}"
            ),
        ))
        prev_body = body
    return records


def _split_record_text(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    parts: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        parts.append(text[start:end])
        start = end
    total = len(parts)
    return [
        f"--- RECORD_PART {i + 1}/{total} ---\n{part}"
        for i, part in enumerate(parts)
    ]


def _pack_records(records: list[tuple[int, str]], max_chars: int) -> list[tuple[list[int], str]]:
    chunks: list[tuple[list[int], str]] = []
    cur_indices: list[int] = []
    cur_parts: list[str] = []
    cur_len = 0
    record_budget = max(1000, max_chars - 500)
    for idx, record in records:
        for piece in _split_record_text(record, record_budget):
            piece_len = len(piece) + 2
            if cur_parts and cur_len + piece_len > max_chars:
                chunks.append((cur_indices, "\n\n".join(cur_parts)))
                cur_indices = []
                cur_parts = []
                cur_len = 0
            cur_parts.append(piece)
            if idx not in cur_indices:
                cur_indices.append(idx)
            cur_len += piece_len
    if cur_parts:
        chunks.append((cur_indices, "\n\n".join(cur_parts)))
    return chunks


def discover_chunks(
    *,
    journal_dir: str,
    since_days: int,
    max_files: int,
    max_chunks: int,
    max_chars_per_chunk: int,
    coverage_mode: str = "delta-full",
) -> list[JournalChunk]:
    root = Path(journal_dir).expanduser()
    cutoff = datetime.now() - timedelta(days=since_days) if since_days > 0 else None
    candidates: list[tuple[datetime, int, Path, list[str]]] = []
    for path in sorted(root.glob("pane_*.log")):
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        entries = _split_entries(text)
        if cutoff is not None:
            entries = [e for e in entries if (t := _entry_time(e)) is not None and t >= cutoff]
        if not entries:
            continue
        _, _, latest = _time_range(entries)
        if latest is None:
            latest = datetime.fromtimestamp(path.stat().st_mtime)
        candidates.append((latest, path.stat().st_size, path, entries))

    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    chunks: list[JournalChunk] = []
    for _, _, path, entries in candidates[:max_files]:
        if coverage_mode == "sample":
            selected, strategy, selected_indices = _select_planned_entries(entries, max_chars_per_chunk)
            packed = [(selected_indices, "\n\n".join(selected))]
        else:
            strategy = "delta-full"
            packed = _pack_records(_build_delta_full_records(entries), max_chars_per_chunk)
        if not packed:
            continue
        pane_id = _pane_id_from_path(path)
        pane_name = _pane_name(path)
        source_bytes = path.stat().st_size
        chunk_parts = len(packed)
        for part_idx, (selected_indices, selected_text) in enumerate(packed, start=1):
            selected_entries = [entries[i] for i in selected_indices if 0 <= i < len(entries)]
            first_ts, last_ts, _ = _time_range(selected_entries)
            noise = _noise_ratio(selected_text)
            chunk_id = f"{pane_id.strip('%')}-{len(chunks):04d}"
            header = (
                f"chunk_id: {chunk_id}\n"
                f"source: {path}\n"
                f"pane_id: {pane_id}\n"
                f"pane_name: {pane_name}\n"
                f"time_range: {first_ts} -> {last_ts}\n"
                f"source_bytes: {source_bytes}\n"
                f"total_entries_in_file: {len(entries)}\n"
                f"selected_entries_in_chunk: {len(set(selected_indices))}\n"
                f"selected_entry_indices: {selected_indices[:30]}\n"
                f"planner_strategy: {strategy}\n"
                f"coverage_mode: {coverage_mode}\n"
                f"file_chunk_part: {part_idx}/{chunk_parts}\n"
                f"selected_noise_ratio: {noise:.2f}\n\n"
            )
            body = _redact(selected_text)
            chunks.append(JournalChunk(
                chunk_id=chunk_id,
                source=str(path),
                pane_id=pane_id,
                pane_name=pane_name,
                first_ts=first_ts,
                last_ts=last_ts,
                entry_count=len(set(selected_indices)),
                total_entry_count=len(entries),
                source_bytes=source_bytes,
                strategy=strategy,
                chunk_part=part_idx,
                chunk_parts=chunk_parts,
                text=header + body,
            ))
            if len(chunks) >= max_chunks:
                break
        if len(chunks) >= max_chunks:
            break
    return chunks


def _strip_code_fence(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\n?", "", s)
        s = re.sub(r"\n?```\s*$", "", s)
    return s.strip()


def _parse_json(raw: str) -> tuple[dict | None, str | None]:
    s = _strip_code_fence(raw)
    try:
        return json.loads(s), None
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", s, re.DOTALL)
        if not m:
            return None, "no JSON object found"
        try:
            return json.loads(m.group(0)), None
        except json.JSONDecodeError as e:
            return None, f"json decode: {e}"


async def analyze_chunk(
    *,
    backend: Backend,
    model: str,
    chunk: JournalChunk,
    max_tokens: int,
) -> JournalOutcome:
    call = await backend.call(
        model=model,
        system=WORKER_SYSTEM,
        user=chunk.text,
        max_tokens=max_tokens,
        role="worker",
    )
    if call.error:
        return JournalOutcome(chunk.chunk_id, None, "", call, parse_error=call.error)
    parsed, perr = _parse_json(call.text)
    if parsed is None:
        return JournalOutcome(chunk.chunk_id, None, call.text, call, parse_error=perr)
    parsed["chunk_id"] = chunk.chunk_id
    parsed.setdefault("source", chunk.source)
    parsed.setdefault("time_range", f"{chunk.first_ts} -> {chunk.last_ts}")
    return JournalOutcome(chunk.chunk_id, parsed, call.text, call)


def _call_fields(call: CallResult) -> dict:
    return {
        "backend": call.backend,
        "tokens_in": call.input_tokens,
        "tokens_cache": call.cache_read_tokens,
        "tokens_out": call.output_tokens,
        "tokens_total": call.total_tokens,
        "reasoning_effort": call.reasoning_effort,
    }


def _write_jsonl(path: str, obj: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _worker_row(chunk: JournalChunk, outcome: JournalOutcome, *, model: str) -> dict:
    return {
        "kind": "journal_worker",
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "role": "worker",
        "model": model,
        "chunk_id": chunk.chunk_id,
        "source": chunk.source,
        "pane_id": chunk.pane_id,
        "pane_name": chunk.pane_name,
        "first_ts": chunk.first_ts,
        "last_ts": chunk.last_ts,
        "entry_count": chunk.entry_count,
        "total_entry_count": chunk.total_entry_count,
        "source_bytes": chunk.source_bytes,
        "strategy": chunk.strategy,
        "chunk_part": chunk.chunk_part,
        "chunk_parts": chunk.chunk_parts,
        "ok": outcome.parsed is not None,
        "parse_error": outcome.parse_error,
        **_call_fields(outcome.call),
        "raw_head": (outcome.raw_text or "")[:500],
        "parsed": outcome.parsed,
    }


async def run_coordinator(
    *,
    backend: Backend,
    model: str,
    worker_rows: list[dict],
    max_tokens: int,
) -> JournalOutcome:
    compact = []
    for row in worker_rows:
        if row.get("parsed"):
            compact.append({
                "chunk_id": row["chunk_id"],
                "source": row["source"],
                "pane_name": row.get("pane_name"),
                "time_range": f"{row.get('first_ts')} -> {row.get('last_ts')}",
                "patterns": row["parsed"].get("patterns", []),
                "notable_events": row["parsed"].get("notable_events", []),
                "confidence": row["parsed"].get("confidence"),
            })
    user = (
        "Worker outputs:\n"
        + json.dumps(compact, ensure_ascii=False, indent=2)[:60000]
    )
    call = await backend.call(
        model=model,
        system=COORDINATOR_SYSTEM,
        user=user,
        max_tokens=max_tokens,
        role="coordinator",
    )
    if call.error:
        return JournalOutcome("coordinator", None, "", call, parse_error=call.error)
    parsed, perr = _parse_json(call.text)
    if parsed is None:
        return JournalOutcome("coordinator", None, call.text, call, parse_error=perr)
    return JournalOutcome("coordinator", parsed, call.text, call)


def _load_rows(path: str) -> list[dict]:
    rows: list[dict] = []
    if not os.path.isfile(path):
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _latest_ok_worker_rows_from_rows(rows: list[dict]) -> dict[str, dict]:
    rows_by_chunk: dict[str, dict] = {}
    for row in rows:
        if row.get("kind") != "journal_worker":
            continue
        if not row.get("ok") or not row.get("parsed") or not row.get("chunk_id"):
            continue
        rows_by_chunk[str(row["chunk_id"])] = row
    return rows_by_chunk


def _latest_ok_worker_rows(path: str) -> dict[str, dict]:
    return _latest_ok_worker_rows_from_rows(_load_rows(path))


def _load_manifest(out_dir: str) -> list[dict]:
    path = os.path.join(out_dir, "chunks_manifest.json")
    if not os.path.isfile(path):
        return []
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return []
    return data if isinstance(data, list) else []


def _usage_by_role_model(rows: list[dict]) -> dict[tuple[str, str], int]:
    usage: dict[tuple[str, str], int] = {}
    for row in rows:
        total = int(row.get("tokens_total") or 0)
        if not total:
            continue
        key = (str(row.get("role") or "?"), str(row.get("model") or row.get("backend") or "?"))
        usage[key] = usage.get(key, 0) + total
    return usage


def _aggregate_worker_patterns(worker_rows: list[dict]) -> list[dict]:
    buckets: dict[str, dict] = {}
    for row in worker_rows:
        parsed = row.get("parsed") or {}
        for pat in parsed.get("patterns", []) or []:
            sig = str(pat.get("signature") or "unknown")
            bucket = buckets.setdefault(sig, {
                "signature": sig,
                "category": pat.get("category", "other"),
                "count": 0,
                "severity": pat.get("severity", "med"),
                "insight": pat.get("insight", ""),
                "recommendation": pat.get("recommendation", ""),
                "evidence": [],
            })
            bucket["count"] += 1
            for ev in pat.get("evidence", []) or []:
                if len(bucket["evidence"]) < 4:
                    bucket["evidence"].append(str(ev)[:140])
    return sorted(buckets.values(), key=lambda x: (-x["count"], x["signature"]))


def write_digest(out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    rows = _load_rows(os.path.join(out_dir, "journal_results.jsonl"))
    manifest_rows = _load_manifest(out_dir)
    historical_worker_rows = [r for r in rows if r.get("kind") == "journal_worker"]
    latest_ok_by_chunk = _latest_ok_worker_rows_from_rows(rows)
    worker_rows = list(latest_ok_by_chunk.values())
    coord_rows = [r for r in rows if r.get("kind") == "journal_coordinator"]
    coord = coord_rows[-1] if coord_rows else None
    parsed = (coord or {}).get("parsed") or {}
    top_patterns = parsed.get("top_patterns") or _aggregate_worker_patterns(worker_rows)

    usage_rows = worker_rows + ([coord] if coord else [])
    usage = _usage_by_role_model(usage_rows)
    expected_chunk_ids = {str(r["chunk_id"]) for r in manifest_rows if r.get("chunk_id")}
    missing_chunk_ids = expected_chunk_ids - set(latest_ok_by_chunk)
    historical_error_count = sum(1 for r in historical_worker_rows if not r.get("ok"))

    out: list[str] = []
    out.append(f"# Tmux Journal Swarm Digest — {os.path.basename(out_dir)}\n")
    out.append(f"- chunks analyzed: **{len(worker_rows)}**")
    if worker_rows:
        source_meta: dict[str, dict] = {}
        for row in worker_rows:
            source = row.get("source")
            if source and source not in source_meta:
                source_meta[str(source)] = row
        total_source_bytes = sum(int(r.get("source_bytes") or 0) for r in source_meta.values())
        total_source_entries = sum(int(r.get("total_entry_count") or r.get("entry_count") or 0) for r in source_meta.values())
        selected_entries = sum(int(r.get("entry_count") or 0) for r in worker_rows)
        strategies: dict[str, int] = {}
        for row in worker_rows:
            strategy = str(row.get("strategy") or "unknown")
            strategies[strategy] = strategies.get(strategy, 0) + 1
        out.append(f"- source files covered: **{len(source_meta)}**")
        out.append(f"- source volume covered: **{total_source_bytes / 1024 / 1024:.2f} MB**, **{total_source_entries}** total entries")
        out.append(f"- entry references sent to workers: **{selected_entries}**")
        source_file_count = len(source_meta) or 1
        out.append("- planner strategies: " + ", ".join(f"`{k}`={v}" for k, v in sorted(strategies.items())))
        out.append(f"- avg chunks per source file: **{len(worker_rows) / source_file_count:.2f}**")
    out.append(f"- current missing/failed worker chunks: **{len(missing_chunk_ids)}**")
    if historical_error_count:
        out.append(f"- historical worker error attempts during retries: **{historical_error_count}**")
    out.append(f"- coordinator ok: **{bool(coord and coord.get('ok'))}**")
    if usage:
        out.append("- effective swarm token usage (latest successful workers + final coordinator):")
        for (role, model), total in sorted(usage.items()):
            out.append(f"  - `{role}` `{model}`: {total:,} total tokens")
        worker_token_rows = [r for r in worker_rows if int(r.get("tokens_total") or 0) > 0]
        if worker_token_rows:
            largest = max(worker_token_rows, key=lambda r: int(r.get("tokens_total") or 0))
            largest_total = int(largest.get("tokens_total") or 0)
            worker_total = sum(int(r.get("tokens_total") or 0) for r in worker_token_rows)
            if largest_total > 5_000_000 and largest_total > worker_total * 0.5:
                out.append(
                    f"- token usage caveat: chunk `{largest.get('chunk_id')}` reports "
                    f"{largest_total:,} worker tokens and dominates the total; treat usage accounting as noisy."
                )
    if parsed.get("run_quality"):
        rq = parsed["run_quality"]
        out.append(f"- coverage: {rq.get('coverage', '')}")
        out.append(f"- limitations: {rq.get('limitations', '')}")

    out.append("\n## Top Patterns\n")
    if not top_patterns:
        out.append("(none)")
    else:
        out.append("| pattern | count | severity | insight | recommendation |")
        out.append("|---|---:|---|---|---|")
        for pat in top_patterns[:20]:
            insight = str(pat.get("insight", "")).replace("|", "\\|")[:220]
            rec = str(pat.get("recommendation", "")).replace("|", "\\|")[:220]
            out.append(
                f"| `{pat.get('signature','unknown')}` | {int(pat.get('count') or 1)} | "
                f"{pat.get('severity','')} | {insight} | {rec} |"
            )

    if parsed.get("cross_cutting_insights"):
        out.append("\n## Cross-Cutting Insights\n")
        for item in parsed["cross_cutting_insights"]:
            out.append(f"- {item}")

    if parsed.get("recommended_experiments"):
        out.append("\n## Recommended Experiments\n")
        for item in parsed["recommended_experiments"]:
            out.append(f"- {item}")

    out.append("\n## Evidence Samples\n")
    for pat in top_patterns[:10]:
        evidence = pat.get("evidence") or []
        if not evidence:
            continue
        out.append(f"\n### `{pat.get('signature','unknown')}`\n")
        for ev in evidence[:4]:
            out.append(f"- {str(ev).replace(chr(10), ' ')[:180]}")

    if coord and coord.get("parse_error"):
        out.append("\n## Coordinator Error\n")
        out.append(str(coord["parse_error"]))

    digest = "\n".join(out)
    digest_path = os.path.join(out_dir, "tmux_journal_digest.md")
    with open(digest_path, "w", encoding="utf-8") as f:
        f.write(digest)
    return digest_path


async def run(args: argparse.Namespace) -> int:
    runtime = normalize_runtime(args.runtime)
    worker_model = args.worker_model or default_model_for_runtime(runtime, "worker")
    coordinator_model = args.coordinator_model or default_model_for_runtime(runtime, "coordinator")
    worker_backend = build_backend(runtime)
    coordinator_backend = worker_backend

    out_dir = os.path.abspath(os.path.join(args.output_dir, args.run_id))
    os.makedirs(out_dir, exist_ok=True)
    result_path = os.path.join(out_dir, "journal_results.jsonl")

    chunks = discover_chunks(
        journal_dir=args.journal_dir,
        since_days=args.since_days,
        max_files=args.max_files,
        max_chunks=args.max_chunks,
        max_chars_per_chunk=args.max_chars_per_chunk,
        coverage_mode=args.coverage_mode,
    )
    manifest_path = os.path.join(out_dir, "chunks_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump([{
            "chunk_id": c.chunk_id,
            "source": c.source,
            "pane_id": c.pane_id,
            "pane_name": c.pane_name,
            "first_ts": c.first_ts,
            "last_ts": c.last_ts,
            "entry_count": c.entry_count,
            "total_entry_count": c.total_entry_count,
            "source_bytes": c.source_bytes,
            "strategy": c.strategy,
            "chunk_part": c.chunk_part,
            "chunk_parts": c.chunk_parts,
        } for c in chunks], f, ensure_ascii=False, indent=2)

    worker_rows_by_chunk = _latest_ok_worker_rows(result_path)
    chunks_to_process = [c for c in chunks if c.chunk_id not in worker_rows_by_chunk]

    print(
        f"[tmux-journal] runtime={runtime} worker={worker_model} "
        f"coordinator={coordinator_model} chunks={len(chunks)} "
        f"to_process={len(chunks_to_process)} already_done={len(worker_rows_by_chunk)} "
        f"output={out_dir}"
    )

    queue: asyncio.Queue[JournalChunk | None] = asyncio.Queue()
    sem = asyncio.Semaphore(args.concurrency)
    worker_rows: list[dict] = list(worker_rows_by_chunk.values())

    async def worker(idx: int) -> None:
        while True:
            chunk = await queue.get()
            if chunk is None:
                queue.task_done()
                break
            async with sem:
                outcome = await analyze_chunk(
                    backend=worker_backend,
                    model=worker_model,
                    chunk=chunk,
                    max_tokens=args.worker_max_tokens,
                )
            row = _worker_row(chunk, outcome, model=worker_model)
            _write_jsonl(result_path, row)
            if row.get("ok") and row.get("parsed"):
                worker_rows_by_chunk[row["chunk_id"]] = row
            if outcome.parse_error:
                print(f"[tmux-journal] worker {idx} error {chunk.chunk_id}: {outcome.parse_error[:160]}")
            queue.task_done()

    workers = [asyncio.create_task(worker(i)) for i in range(args.concurrency)]
    for chunk in chunks_to_process:
        await queue.put(chunk)
    for _ in workers:
        await queue.put(None)
    await queue.join()
    await asyncio.gather(*workers)
    worker_rows = list(worker_rows_by_chunk.values())

    coord = await run_coordinator(
        backend=coordinator_backend,
        model=coordinator_model,
        worker_rows=worker_rows,
        max_tokens=args.coordinator_max_tokens,
    )
    _write_jsonl(result_path, {
        "kind": "journal_coordinator",
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "role": "coordinator",
        "model": coordinator_model,
        "ok": coord.parsed is not None,
        "parse_error": coord.parse_error,
        **_call_fields(coord.call),
        "raw_head": (coord.raw_text or "")[:500],
        "parsed": coord.parsed,
    })
    if coord.parse_error:
        print(f"[tmux-journal] coordinator error: {coord.parse_error[:200]}")

    digest_path = write_digest(out_dir)
    print(f"[tmux-journal] wrote {digest_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--journal-dir", default=str(Path.home() / ".tmux-journal"))
    p.add_argument("--runtime", default="codex")
    p.add_argument("--worker-model", default=None)
    p.add_argument("--coordinator-model", default=None)
    p.add_argument("--concurrency", type=int, default=2)
    p.add_argument("--since-days", type=int, default=21)
    p.add_argument("--max-files", type=int, default=24)
    p.add_argument("--max-chunks", type=int, default=12)
    p.add_argument("--max-chars-per-chunk", type=int, default=12000)
    p.add_argument("--coverage-mode", choices=["delta-full", "sample"], default="delta-full")
    p.add_argument("--worker-max-tokens", type=int, default=900)
    p.add_argument("--coordinator-max-tokens", type=int, default=3000)
    p.add_argument("--output-dir", default="runs")
    p.add_argument("--run-id", default=time.strftime("tmux-journal-%Y%m%d-%H%M%S"))
    p.add_argument("--summarize-only", action="store_true")
    args = p.parse_args(argv)
    if args.summarize_only:
        out_dir = os.path.abspath(os.path.join(args.output_dir, args.run_id))
        digest_path = write_digest(out_dir)
        print(f"[tmux-journal] wrote {digest_path}")
        return 0
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
