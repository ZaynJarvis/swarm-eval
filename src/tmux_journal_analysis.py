"""Third-pass swarm analysis over tmux-journal logs.

This is intentionally separate from LoCoMo session analysis. It reuses the same
worker/coordinator runtime split to answer a different question: what recurring
workflow patterns and friction points are visible in a developer's tmux journal?
"""
from __future__ import annotations

import argparse
import asyncio
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
    (re.compile(r"(?i)(api[_-]?key|token|secret|password)\s*[:=]\s*['\"]?[^'\"\s]+"), r"\1=[REDACTED]"),
    (re.compile(r"(?i)authorization:\s*bearer\s+[A-Za-z0-9._-]+"), "authorization: bearer [REDACTED]"),
    (re.compile(r"AKIA[0-9A-Z]{16}"), "AKIA[REDACTED]"),
]


@dataclass
class JournalChunk:
    chunk_id: str
    source: str
    pane_id: str
    pane_name: str
    first_ts: str
    last_ts: str
    entry_count: int
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


def discover_chunks(
    *,
    journal_dir: str,
    since_days: int,
    max_files: int,
    max_chunks: int,
    max_chars_per_chunk: int,
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
        selected = _trim_entries(entries, max_chars_per_chunk)
        if not selected:
            continue
        first_ts, last_ts, _ = _time_range(selected)
        pane_id = _pane_id_from_path(path)
        pane_name = _pane_name(path)
        chunk_id = f"{pane_id.strip('%')}-{len(chunks):03d}"
        header = (
            f"chunk_id: {chunk_id}\n"
            f"source: {path}\n"
            f"pane_id: {pane_id}\n"
            f"pane_name: {pane_name}\n"
            f"time_range: {first_ts} -> {last_ts}\n"
            f"entries_in_chunk: {len(selected)}\n\n"
        )
        body = _clip(_redact("\n\n".join(selected)), max_chars_per_chunk)
        chunks.append(JournalChunk(
            chunk_id=chunk_id,
            source=str(path),
            pane_id=pane_id,
            pane_name=pane_name,
            first_ts=first_ts,
            last_ts=last_ts,
            entry_count=len(selected),
            text=header + body,
        ))
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
    worker_rows = [r for r in rows if r.get("kind") == "journal_worker"]
    coord_rows = [r for r in rows if r.get("kind") == "journal_coordinator"]
    coord = coord_rows[-1] if coord_rows else None
    parsed = (coord or {}).get("parsed") or {}
    top_patterns = parsed.get("top_patterns") or _aggregate_worker_patterns(worker_rows)

    usage: dict[tuple[str, str], int] = {}
    for row in rows:
        total = int(row.get("tokens_total") or 0)
        if not total:
            continue
        key = (str(row.get("role") or "?"), str(row.get("model") or row.get("backend") or "?"))
        usage[key] = usage.get(key, 0) + total

    out: list[str] = []
    out.append(f"# Tmux Journal Swarm Digest — {os.path.basename(out_dir)}\n")
    out.append(f"- chunks analyzed: **{len(worker_rows)}**")
    out.append(f"- worker errors: **{sum(1 for r in worker_rows if not r.get('ok'))}**")
    out.append(f"- coordinator ok: **{bool(coord and coord.get('ok'))}**")
    if usage:
        out.append("- swarm token usage:")
        for (role, model), total in sorted(usage.items()):
            out.append(f"  - `{role}` `{model}`: {total:,} total tokens")
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
        } for c in chunks], f, ensure_ascii=False, indent=2)

    print(
        f"[tmux-journal] runtime={runtime} worker={worker_model} "
        f"coordinator={coordinator_model} chunks={len(chunks)} output={out_dir}"
    )

    queue: asyncio.Queue[JournalChunk | None] = asyncio.Queue()
    sem = asyncio.Semaphore(args.concurrency)
    worker_rows: list[dict] = []

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
            worker_rows.append(row)
            _write_jsonl(result_path, row)
            if outcome.parse_error:
                print(f"[tmux-journal] worker {idx} error {chunk.chunk_id}: {outcome.parse_error[:160]}")
            queue.task_done()

    workers = [asyncio.create_task(worker(i)) for i in range(args.concurrency)]
    for chunk in chunks:
        await queue.put(chunk)
    for _ in workers:
        await queue.put(None)
    await queue.join()
    await asyncio.gather(*workers)

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
