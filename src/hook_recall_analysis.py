"""Second-pass analysis: was the wrong answer already recoverable from hooks?

This intentionally stays separate from the pattern-library swarm run. It only
looks at grader-WRONG sessions and classifies hook recall quality:

- sufficient: hook context already contained enough information for the gold.
- partial: hook context was related but missed required detail/qualifier/count.
- absent: hook context did not retrieve useful target evidence.
- misleading: hook context surfaced distractors/conflicting evidence.
- unassessable: hook evidence is too truncated/ambiguous to judge.
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backends.base import Backend, CallResult
from backends.factory import build_backend, default_model_for_runtime, normalize_runtime
from linker import LinkedSession, iter_linked


SYSTEM_PROMPT = """You are a hook-recall quality analyst for LoCoMo memory QA sessions.

Analyze ONE grader-WRONG session. Your job is not to judge correctness; the
session is already WRONG. Decide whether the hook-injected recall context, by
itself, contained enough information to answer the gold answer.

Definitions:
- sufficient: hook context contains enough direct evidence to answer the gold,
  including required qualifier/item/count/date. The model answered wrong anyway.
- partial: hook context is related but lacks a required exact detail,
  qualifier, count, date, or one of multiple required items.
- absent: hook context lacks target evidence needed for the gold answer.
- misleading: hook context contains distractors/conflicting adjacent evidence
  that supports the wrong response or an easy wrong target.
- unassessable: hook context is too truncated, malformed, or ambiguous to judge.

Pick the best failure_owner:
- answer_extraction_or_reasoning: hook was sufficient; model failed after recall.
- hook_recall_gap: hook was absent or partial; recall quality is the primary gap.
- hook_ranking_noise: hook showed misleading/distractor evidence.
- needs_mcp_after_hook: hook was insufficient, but the transcript stats show MCP
  was available/used and should have compensated.
- unassessable: cannot decide.

Hard rules:
- Output JSON only.
- Do not re-grade the answer. Treat grader_verdict as WRONG.
- Quote only from HOOK_CONTEXT when filling hook_evidence_quotes.
- If gold is only partially supported, do not mark sufficient.
- Keep notes concise and specific.

Output schema:
{
  "session_id": "<copy from input>",
  "hook_sufficiency": "sufficient" | "partial" | "absent" | "misleading" | "unassessable",
  "failure_owner": "answer_extraction_or_reasoning" | "hook_recall_gap" | "hook_ranking_noise" | "needs_mcp_after_hook" | "unassessable",
  "gold_supported_in_hook": true | false,
  "response_supported_in_hook": true | false,
  "hook_evidence_quotes": ["<quote <=160 chars>", ...],
  "confidence": <0.0-1.0>,
  "notes": "<one concise sentence>"
}
"""


@dataclass
class HookOutcome:
    session_id: str
    parsed: dict | None
    raw_text: str
    call: CallResult
    parse_error: str | None = None


def _clip(s: str, limit: int) -> str:
    if len(s) <= limit:
        return s
    return s[:limit] + f"\n... [TRUNCATED {len(s) - limit} chars]"


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
            return None, f"json decode (brace-block): {e}"


def _load_hook_context(transcript_path: str, max_chars: int) -> str:
    chunks: list[str] = []
    with open(transcript_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") != "attachment":
                continue
            attachment = event.get("attachment", {}) or {}
            if attachment.get("type") != "hook_additional_context":
                continue
            content = attachment.get("content", "")
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False)
            chunks.append(f"[{i}] {content}")
    return _clip("\n\n".join(chunks) or "(no hook_additional_context found)", max_chars)


def _build_user_payload(linked: LinkedSession, *, max_hook_chars: int) -> str:
    hook_context = _load_hook_context(linked.transcript_path, max_chars=max_hook_chars)
    return (
        f"session_id: {linked.session_id}\n"
        f"sample_id: {linked.sample_id}\n"
        f"question_index: {linked.question_index}\n"
        f"category: {linked.category}\n"
        f"question: {linked.question}\n"
        f"gold_answer: {linked.gold_answer}\n"
        f"model_response: {linked.response}\n"
        f"grader_verdict: {linked.result}\n"
        f"grader_reasoning: {linked.grader_reasoning}\n"
        f"runtime: turns={linked.num_turns} cost=${linked.total_cost_usd:.4f} "
        f"mcp_metric={linked.ov_mcp_calls} transcript_mcp={linked.transcript_mcp_calls} "
        f"input_tokens={linked.input_tokens} cache_read_tokens={linked.cache_read_input_tokens} "
        f"output_tokens={linked.output_tokens}\n\n"
        "HOOK_CONTEXT:\n"
        f"{hook_context}"
    )


def _validate(parsed: dict, expected_session_id: str) -> str | None:
    required = {
        "session_id",
        "hook_sufficiency",
        "failure_owner",
        "gold_supported_in_hook",
        "response_supported_in_hook",
        "hook_evidence_quotes",
        "confidence",
        "notes",
    }
    missing = required - parsed.keys()
    if missing:
        return f"missing keys: {sorted(missing)}"
    if parsed["session_id"] != expected_session_id:
        parsed["session_id"] = expected_session_id
    if parsed["hook_sufficiency"] not in {
        "sufficient",
        "partial",
        "absent",
        "misleading",
        "unassessable",
    }:
        return f"bad hook_sufficiency: {parsed['hook_sufficiency']!r}"
    if parsed["failure_owner"] not in {
        "answer_extraction_or_reasoning",
        "hook_recall_gap",
        "hook_ranking_noise",
        "needs_mcp_after_hook",
        "unassessable",
    }:
        return f"bad failure_owner: {parsed['failure_owner']!r}"
    if not isinstance(parsed["gold_supported_in_hook"], bool):
        return "gold_supported_in_hook not bool"
    if not isinstance(parsed["response_supported_in_hook"], bool):
        return "response_supported_in_hook not bool"
    if not isinstance(parsed["hook_evidence_quotes"], list):
        return "hook_evidence_quotes not list"
    try:
        conf = float(parsed["confidence"])
    except (TypeError, ValueError):
        return "confidence not float"
    if not 0.0 <= conf <= 1.0:
        return f"confidence out of range: {conf}"
    return None


async def analyze_one(
    *,
    backend: Backend,
    model: str,
    linked: LinkedSession,
    max_hook_chars: int,
    max_tokens: int,
) -> HookOutcome:
    user = _build_user_payload(linked, max_hook_chars=max_hook_chars)
    call = await backend.call(
        model=model,
        system=SYSTEM_PROMPT,
        user=user,
        max_tokens=max_tokens,
        role="worker",
    )
    if call.error:
        return HookOutcome(linked.session_id, None, "", call, parse_error=call.error)
    parsed, perr = _parse_json(call.text)
    if parsed is None:
        return HookOutcome(linked.session_id, None, call.text, call, parse_error=perr)
    verr = _validate(parsed, linked.session_id)
    if verr:
        return HookOutcome(linked.session_id, None, call.text, call, parse_error=verr)
    return HookOutcome(linked.session_id, parsed, call.text, call)


def _read_done(path: str) -> set[str]:
    done: set[str] = set()
    if not os.path.isfile(path):
        return done
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("ok") and obj.get("parsed") and obj.get("session_id"):
                done.add(obj["session_id"])
    return done


def _write_jsonl(path: str, obj: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _row_from_outcome(linked: LinkedSession, outcome: HookOutcome) -> dict:
    return {
        "kind": "hook_recall_worker",
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "session_id": linked.session_id,
        "sample_id": linked.sample_id,
        "qix": linked.question_index,
        "category": linked.category,
        "num_turns": linked.num_turns,
        "cost_usd": linked.total_cost_usd,
        "input_tokens": linked.input_tokens,
        "cache_read_tokens": linked.cache_read_input_tokens,
        "output_tokens": linked.output_tokens,
        "transcript_mcp_calls": linked.transcript_mcp_calls,
        "ok": outcome.parsed is not None,
        "parse_error": outcome.parse_error,
        "backend": outcome.call.backend,
        "tokens_in": outcome.call.input_tokens,
        "tokens_cache": outcome.call.cache_read_tokens,
        "tokens_out": outcome.call.output_tokens,
        "raw_head": (outcome.raw_text or "")[:500],
        "parsed": outcome.parsed,
    }


def summarize(out_dir: str) -> str:
    path = os.path.join(out_dir, "hook_recall_results.jsonl")
    rows: list[dict] = []
    errors: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("ok") and obj.get("parsed"):
                rows.append(obj)
            else:
                errors.append(obj)

    def parsed_value(row: dict, key: str) -> str:
        return str(row["parsed"].get(key) or "unknown")

    def count_by(key: str) -> dict[str, int]:
        out: dict[str, int] = {}
        for row in rows:
            val = parsed_value(row, key)
            out[val] = out.get(val, 0) + 1
        return dict(sorted(out.items(), key=lambda x: (-x[1], x[0])))

    suff_counts = count_by("hook_sufficiency")
    owner_counts = count_by("failure_owner")
    gold_supported = sum(1 for r in rows if r["parsed"].get("gold_supported_in_hook"))
    response_supported = sum(1 for r in rows if r["parsed"].get("response_supported_in_hook"))
    sufficient_rows = [
        r for r in rows
        if r["parsed"].get("hook_sufficiency") == "sufficient"
        or r["parsed"].get("gold_supported_in_hook") is True
    ]

    out: list[str] = []
    out.append(f"# Hook Recall Quality Digest — {os.path.basename(out_dir)}\n")
    out.append(f"- WRONG sessions analyzed: **{len(rows)}**")
    out.append(f"- worker errors: **{len(errors)}**")
    out.append(
        f"- hook already supported gold: **{gold_supported}** "
        f"({(gold_supported / len(rows) * 100) if rows else 0:.1f}%)"
    )
    out.append(
        f"- sufficient hook / answer still wrong: **{len(sufficient_rows)}** "
        f"({(len(sufficient_rows) / len(rows) * 100) if rows else 0:.1f}%)"
    )
    out.append(
        f"- wrong response also supported by hook distractor/noise: **{response_supported}** "
        f"({(response_supported / len(rows) * 100) if rows else 0:.1f}%)"
    )

    out.append("\n## Hook Sufficiency\n")
    out.append("| bucket | count | pct |")
    out.append("|---|---:|---:|")
    for k, v in suff_counts.items():
        pct = v / len(rows) * 100 if rows else 0.0
        out.append(f"| `{k}` | {v} | {pct:.1f}% |")

    out.append("\n## Failure Owner\n")
    out.append("| owner | count | pct |")
    out.append("|---|---:|---:|")
    for k, v in owner_counts.items():
        pct = v / len(rows) * 100 if rows else 0.0
        out.append(f"| `{k}` | {v} | {pct:.1f}% |")

    out.append("\n## Examples — Sufficient Hook But Wrong Answer\n")
    out.append("| session | sample/qix | cost | turns/mcp | confidence | notes |")
    out.append("|---|---|---:|---|---:|---|")
    for r in sorted(
        sufficient_rows,
        key=lambda x: (-float(x.get("cost_usd") or 0.0), x["session_id"]),
    )[:20]:
        p = r["parsed"]
        note = str(p.get("notes", "")).replace("|", "\\|")[:160]
        out.append(
            f"| {r['session_id'][:8]} | {r['sample_id']}/q{r['qix']} | "
            f"${float(r.get('cost_usd') or 0.0):.4f} | "
            f"{r.get('num_turns')}/{r.get('transcript_mcp_calls')} | "
            f"{float(p.get('confidence') or 0.0):.2f} | {note} |"
        )

    out.append("\n## Examples By Hook Sufficiency\n")
    for bucket in ("sufficient", "partial", "absent", "misleading", "unassessable"):
        bucket_rows = [r for r in rows if r["parsed"].get("hook_sufficiency") == bucket]
        out.append(f"\n### `{bucket}` ({len(bucket_rows)})\n")
        if not bucket_rows:
            out.append("(none)")
            continue
        for r in bucket_rows[:8]:
            p = r["parsed"]
            quotes = "; ".join(str(q)[:120] for q in (p.get("hook_evidence_quotes") or [])[:2])
            note = str(p.get("notes", "")).replace("\n", " ")[:180]
            out.append(
                f"- `{r['session_id'][:8]}` {r['sample_id']}/q{r['qix']} "
                f"owner=`{p.get('failure_owner')}` conf={float(p.get('confidence') or 0.0):.2f}; "
                f"{note} Quotes: {quotes or '—'}"
            )

    if errors:
        out.append("\n## Worker Errors\n")
        for e in errors[:20]:
            out.append(f"- `{e.get('session_id','')[:8]}`: {e.get('parse_error')}")

    return "\n".join(out)


async def run(args: argparse.Namespace) -> int:
    runtime = normalize_runtime(args.runtime)
    model = args.model if args.model is not None else default_model_for_runtime(runtime, "worker")
    backend = build_backend(runtime)

    out_dir = os.path.abspath(os.path.join(args.output_dir, args.run_id))
    os.makedirs(out_dir, exist_ok=True)
    result_path = os.path.join(out_dir, "hook_recall_results.jsonl")
    done = _read_done(result_path)
    sessions = [
        s for s in iter_linked(args.result_dir, limit=args.limit)
        if s.result.upper() == "WRONG" and s.session_id not in done
    ]
    print(
        f"[hook-recall] runtime={runtime} model={model or '<runtime default>'} "
        f"to_process={len(sessions)} already_done={len(done)} output={out_dir}"
    )

    queue: asyncio.Queue[LinkedSession | None] = asyncio.Queue()
    sem = asyncio.Semaphore(args.concurrency)

    async def worker(idx: int) -> None:
        while True:
            linked = await queue.get()
            if linked is None:
                queue.task_done()
                break
            async with sem:
                outcome = await analyze_one(
                    backend=backend,
                    model=model,
                    linked=linked,
                    max_hook_chars=args.max_hook_chars,
                    max_tokens=args.max_tokens,
                )
            _write_jsonl(result_path, _row_from_outcome(linked, outcome))
            queue.task_done()
            if outcome.parse_error:
                print(f"[hook-recall] worker {idx} error {linked.session_id[:8]}: {outcome.parse_error[:160]}")

    workers = [asyncio.create_task(worker(i)) for i in range(args.concurrency)]
    for linked in sessions:
        await queue.put(linked)
    for _ in workers:
        await queue.put(None)
    await queue.join()
    await asyncio.gather(*workers)

    digest = summarize(out_dir)
    digest_path = os.path.join(out_dir, "hook_recall_digest.md")
    with open(digest_path, "w", encoding="utf-8") as f:
        f.write(digest)
    print(f"[hook-recall] wrote {digest_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--result-dir", required=True)
    p.add_argument("--runtime", default="codex")
    p.add_argument("--model", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--max-hook-chars", type=int, default=12000)
    p.add_argument("--max-tokens", type=int, default=900)
    p.add_argument("--output-dir", default="runs")
    p.add_argument("--run-id", default=time.strftime("hook-recall-%Y%m%d-%H%M%S"))
    p.add_argument("--summarize-only", action="store_true")
    args = p.parse_args(argv)
    if args.summarize_only:
        out_dir = os.path.abspath(os.path.join(args.output_dir, args.run_id))
        digest = summarize(out_dir)
        digest_path = os.path.join(out_dir, "hook_recall_digest.md")
        with open(digest_path, "w", encoding="utf-8") as f:
            f.write(digest)
        print(f"[hook-recall] wrote {digest_path}")
        return 0
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
