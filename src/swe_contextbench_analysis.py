"""Swarm analysis for SWE-ContextBench context-value cases.

The LoCoMo orchestrator is transcript/judge-result shaped. SWE-ContextBench is
relationship shaped: each related task links to one or more prior experience
tasks. This script reuses the same Codex/agent runtime layer but builds a
planner -> worker -> coordinator flow around those relationship pairs.
"""
from __future__ import annotations

import argparse
import asyncio
import ast
import json
import os
import re
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backends.base import Backend, CallResult
from backends.factory import build_backend, default_model_for_runtime, normalize_runtime


HF_BASE = "https://huggingface.co/datasets/jiayuanz3/SWEContextBench/resolve/main/"
HF_FILES = (
    "data/SWEContextBench_Experience.parquet",
    "data/SWEContextBench_Related.parquet",
    "data/SWEContextBench_Relationship.parquet",
)


PLANNER_SYSTEM = """You are the coordinator/planner for a SWE-ContextBench swarm.

Goal: inspect a deterministic sample of related task <-> prior experience task
pairs and propose a compact tag vocabulary for workers. The final research
question is: which cases have high context value, and what reusable context
categories do they belong to?

Use the paper/dataset construction facts in the prompt, but do not blindly
accept all relationships as valuable. A high-value case is one where the prior
experience likely gives reusable problem-solving state for the related task:
API invariant, file locality, regression oracle, patch recipe, edge case,
domain semantics, migration compatibility, or a warning about false analogy.

Output JSON only:
{
  "seed_tags": [
    {
      "tag": "kebab-case",
      "category": "api_contract|patch_recipe|test_oracle|localization|domain_semantics|compatibility|negative_transfer|workflow|other",
      "definition": "what this tag means",
      "positive_signal": "what workers should look for",
      "negative_signal": "when not to use it"
    }
  ],
  "sampling_observations": ["short observation", "..."],
  "worker_guidance": ["short instruction", "..."]
}
"""


WORKER_SYSTEM = """You are a SWE-ContextBench worker analyzing one chunk of full
related-task/experience-task text.

Question: does each related test case have meaningful context value from the
linked prior experience case(s), and if yes, what kind?

Read the full case text provided for each related case. Compare the related
task against linked experience tasks across: problem statement, hints, solution
patch, test patch, FAIL_TO_PASS tests, PASS_TO_PASS tests, repository/version,
and relationship metadata.

Use planner seed tags when they fit. Propose new tags only when the seed tags
miss a repeated or important context-value type.

Do not solve the SWE task. Do not judge benchmark correctness. Classify context
reuse value.

Context value scale:
- high: prior experience likely gives directly reusable implementation/test
  insight that would materially improve accuracy or reduce search.
- medium: prior experience gives useful localization, vocabulary, or caution,
  but the new fix still requires substantial fresh reasoning.
- low: relationship is real but mostly metadata/issue-reference level; little
  reusable coding context.
- negative: prior experience is likely to mislead because surface similarity
  hides a different invariant, layer, version, or fix strategy.
- uncertain: provided evidence is insufficient.

Output JSON only:
{
  "chunk_id": "<copy>",
  "case_results": [
    {
      "related_instance_id": "<id>",
      "repo": "<repo>",
      "experience_instance_ids": ["<id>"],
      "context_value": "high|medium|low|negative|uncertain",
      "context_value_score": 0,
      "primary_tags": ["kebab-case"],
      "new_tags": [
        {
          "tag": "kebab-case",
          "category": "api_contract|patch_recipe|test_oracle|localization|domain_semantics|compatibility|negative_transfer|workflow|other",
          "definition": "short definition"
        }
      ],
      "relationship_type_inferred": "same_pr|different_pr_same_issue_family|pr_reference|issue_reference|recursive_or_chain|unclear",
      "reusable_context_unit": "what should be stored/retrieved for this case",
      "why_context_helps": "specific reason",
      "why_context_may_hurt": "specific risk or empty string",
      "evidence": ["short quote or file/test/patch clue", "..."],
      "confidence": 0.0
    }
  ],
  "chunk_notes": "brief note",
  "unclear_cases": ["id", "..."]
}
"""


COORDINATOR_SYSTEM = """You are the final coordinator for a SWE-ContextBench
swarm. Workers have classified full related-task/experience-task pairs.

Aggregate the messy tags into a final taxonomy. Be critical: do not count every
GitHub reference as context value. Separate genuinely reusable context from
weak/misleading references. Preserve concrete case IDs.

Output JSON only:
{
  "final_categories": [
    {
      "category": "kebab-case",
      "label": "human readable label",
      "definition": "what makes this category context-valuable",
      "case_count": 0,
      "value_mix": {"high": 0, "medium": 0, "low": 0, "negative": 0, "uncertain": 0},
      "representative_cases": [
        {
          "related_instance_id": "<id>",
          "experience_instance_ids": ["<id>"],
          "why": "short reason"
        }
      ],
      "what_to_store_as_memory": "the reusable context unit"
    }
  ],
  "highest_value_cases": [
    {
      "related_instance_id": "<id>",
      "repo": "<repo>",
      "experience_instance_ids": ["<id>"],
      "category": "kebab-case",
      "score": 0,
      "why": "short reason"
    }
  ],
  "low_or_misleading_context_patterns": ["...", "..."],
  "retrieval_implications": ["...", "..."],
  "run_quality": {
    "coverage": "brief coverage statement",
    "limitations": "brief limitation statement"
  }
}
"""


@dataclass(frozen=True)
class ContextCase:
    related_id: str
    repo: str
    relationships: list[dict[str, Any]]
    related: dict[str, Any]
    experiences: list[dict[str, Any]]
    rendered_len: int


def _strip_code_fence(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\n?", "", s)
        s = re.sub(r"\n?```\s*$", "", s)
    return s.strip()


def _parse_json(raw: str) -> tuple[dict[str, Any] | None, str | None]:
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


def _clip(text: Any, limit: int) -> str:
    s = "" if text is None else str(text)
    if len(s) <= limit:
        return s
    head = limit // 2
    tail = max(0, limit - head - 80)
    return s[:head] + f"\n... [TRUNCATED {len(s) - head - tail} chars] ...\n" + s[-tail:]


def _decode_json_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str):
        try:
            data = json.loads(value)
            if isinstance(data, list):
                return [str(v) for v in data]
        except json.JSONDecodeError:
            pass
        try:
            data = ast.literal_eval(value)
            if isinstance(data, list):
                return [str(v) for v in data]
        except (ValueError, SyntaxError):
            pass
    return [str(value)]


def _case_projection(row: dict[str, Any], *, full: bool) -> dict[str, Any]:
    patch_limit = 2_500 if not full else 10_000_000
    problem_limit = 2_500 if not full else 10_000_000
    tests_limit = 30 if not full else 10_000
    return {
        "repo": row.get("repo"),
        "instance_id": row.get("instance_id"),
        "base_commit": row.get("base_commit"),
        "created_at": row.get("created_at"),
        "version": row.get("version"),
        "problem_statement": _clip(row.get("problem_statement"), problem_limit),
        "hints_text": _clip(row.get("hints_text"), problem_limit),
        "patch": _clip(row.get("patch"), patch_limit),
        "test_patch": _clip(row.get("test_patch"), patch_limit),
        "FAIL_TO_PASS": _decode_json_list(row.get("FAIL_TO_PASS"))[:tests_limit],
        "PASS_TO_PASS": _decode_json_list(row.get("PASS_TO_PASS"))[:tests_limit],
    }


def _repo_root_from_dataset_arg(dataset_dir: str) -> Path:
    root = Path(dataset_dir).expanduser().resolve()
    if (root / "cases").is_dir():
        return root
    if (root / "SWEContextBench").is_dir():
        return root / "SWEContextBench"
    raise FileNotFoundError(f"Cannot find SWEContextBench repo at {root}")


def ensure_hf_data(repo_root: Path) -> Path:
    data_dir = repo_root / ".cache" / "hf-data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for rel in HF_FILES:
        dest = data_dir / Path(rel).name
        if dest.exists() and dest.stat().st_size > 0:
            continue
        urllib.request.urlretrieve(HF_BASE + rel, dest)
    return data_dir


def _load_parquet_by_id(path: Path) -> dict[str, dict[str, Any]]:
    rows = pq.read_table(path).to_pylist()
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        instance_id = str(row.get("instance_id") or "")
        if instance_id and instance_id not in out:
            out[instance_id] = row
    return out


def load_context_cases(dataset_dir: str) -> list[ContextCase]:
    repo_root = _repo_root_from_dataset_arg(dataset_dir)
    data_dir = ensure_hf_data(repo_root)
    related_rows = _load_parquet_by_id(data_dir / "SWEContextBench_Related.parquet")
    experience_rows = _load_parquet_by_id(data_dir / "SWEContextBench_Experience.parquet")
    relationships = pq.read_table(data_dir / "SWEContextBench_Relationship.parquet").to_pylist()

    by_related: dict[str, list[dict[str, Any]]] = {}
    for rel in relationships:
        rid = str(rel.get("related_instance_id") or "")
        if not rid:
            continue
        by_related.setdefault(rid, []).append(rel)

    cases: list[ContextCase] = []
    for rid in sorted(by_related):
        related = related_rows.get(rid)
        if not related:
            continue
        seen_exp: set[str] = set()
        experiences: list[dict[str, Any]] = []
        for rel in by_related[rid]:
            eid = str(rel.get("experience_instance_id") or "")
            if not eid or eid in seen_exp:
                continue
            seen_exp.add(eid)
            if eid in experience_rows:
                experiences.append(experience_rows[eid])
        if not experiences:
            continue
        rendered = render_case_payload(ContextCase(
            related_id=rid,
            repo=str(related.get("repo") or ""),
            relationships=by_related[rid],
            related=related,
            experiences=experiences,
            rendered_len=0,
        ), full=True)
        cases.append(ContextCase(
            related_id=rid,
            repo=str(related.get("repo") or ""),
            relationships=by_related[rid],
            related=related,
            experiences=experiences,
            rendered_len=len(rendered),
        ))
    return cases


def _relationship_hint(case: ContextCase) -> dict[str, Any]:
    same_pr = sum(
        1 for rel in case.relationships
        if rel.get("related_pr_url") == rel.get("experience_pr_url")
    )
    same_issue = sum(
        1 for rel in case.relationships
        if rel.get("related_issue_url") == rel.get("experience_issue_url")
    )
    return {
        "relationship_rows": len(case.relationships),
        "unique_experience_count": len(case.experiences),
        "same_pr_url_rows": same_pr,
        "same_issue_url_rows": same_issue,
        "relationship_urls": case.relationships,
    }


def render_case_payload(case: ContextCase, *, full: bool) -> str:
    obj = {
        "related_instance_id": case.related_id,
        "repo": case.repo,
        "relationship_hint": _relationship_hint(case),
        "related_task": _case_projection(case.related, full=full),
        "experience_tasks": [_case_projection(exp, full=full) for exp in case.experiences],
    }
    return json.dumps(obj, ensure_ascii=False, indent=2)


def render_cases_chunk(chunk_id: str, cases: list[ContextCase], seed_tags: list[dict[str, Any]]) -> str:
    payload = {
        "chunk_id": chunk_id,
        "seed_tags": seed_tags,
        "cases": [json.loads(render_case_payload(case, full=True)) for case in cases],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def pick_planner_sample(cases: list[ContextCase], n: int) -> list[ContextCase]:
    selected: list[ContextCase] = []
    seen: set[str] = set()

    def add(case: ContextCase) -> None:
        if len(selected) >= n or case.related_id in seen:
            return
        selected.append(case)
        seen.add(case.related_id)

    for case in sorted(cases, key=lambda c: c.rendered_len, reverse=True)[: max(2, n // 4)]:
        add(case)
    for case in sorted(cases, key=lambda c: c.rendered_len)[: max(2, n // 4)]:
        add(case)
    for case in cases:
        hint = _relationship_hint(case)
        if hint["same_pr_url_rows"] or hint["unique_experience_count"] > 1:
            add(case)
    repo_seen: set[str] = set()
    for case in cases:
        if case.repo not in repo_seen:
            repo_seen.add(case.repo)
            add(case)
    for case in cases:
        add(case)
        if len(selected) >= n:
            break
    return selected


def build_chunks(cases: list[ContextCase], max_chars: int, max_cases: int | None) -> list[tuple[str, list[ContextCase]]]:
    selected = cases[:max_cases] if max_cases else cases
    # First-fit decreasing packs small cases together while leaving giant cases
    # as single chunks. This keeps all case text intact.
    chunks: list[list[ContextCase]] = []
    chunk_sizes: list[int] = []
    for case in sorted(selected, key=lambda c: (-c.rendered_len, c.related_id)):
        placed = False
        for i, size in enumerate(chunk_sizes):
            if size + case.rendered_len <= max_chars:
                chunks[i].append(case)
                chunk_sizes[i] += case.rendered_len
                placed = True
                break
        if not placed:
            chunks.append([case])
            chunk_sizes.append(case.rendered_len)
    return [(f"swectx-{i:04d}", chunk) for i, chunk in enumerate(chunks)]


def _call_fields(call: CallResult) -> dict[str, Any]:
    return {
        "backend": call.backend,
        "tokens_in": call.input_tokens,
        "tokens_cache": call.cache_read_tokens,
        "tokens_out": call.output_tokens,
        "tokens_total": call.total_tokens,
        "reasoning_effort": call.reasoning_effort,
    }


def _write_jsonl(path: str, obj: dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _load_rows(path: str) -> list[dict[str, Any]]:
    if not os.path.isfile(path):
        return []
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _latest_ok(rows: list[dict[str, Any]], kind: str, key: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("kind") == kind and row.get("ok") and row.get(key):
            out[str(row[key])] = row
    return out


async def call_json(
    *,
    backend: Backend,
    model: str,
    role: str,
    system: str,
    user: str,
    max_tokens: int,
) -> tuple[dict[str, Any] | None, str | None, str, CallResult]:
    call = await backend.call(
        model=model,
        system=system,
        user=user,
        max_tokens=max_tokens,
        role=role,
    )
    if call.error:
        return None, call.error, "", call
    parsed, err = _parse_json(call.text)
    return parsed, err, call.text, call


async def run_planner(
    *,
    backend: Backend,
    model: str,
    cases: list[ContextCase],
    sample_n: int,
    out_dir: str,
    result_path: str,
    max_tokens: int,
) -> dict[str, Any]:
    seed_path = os.path.join(out_dir, "seed_tags.json")
    if os.path.isfile(seed_path):
        with open(seed_path, encoding="utf-8") as f:
            return json.load(f)
    sample = pick_planner_sample(cases, sample_n)
    user = {
        "dataset_facts": {
            "source_repo": "https://github.com/jiayuanz3/SWEContextBench",
            "paper": "https://arxiv.org/abs/2602.08316",
            "full_related_cases": len(cases),
            "construction": "1,100 experience tasks; 376 relationship rows; related tasks are derived from issue/PR reference relationships.",
        },
        "sample_cases": [json.loads(render_case_payload(case, full=False)) for case in sample],
    }
    parsed, err, raw, call = await call_json(
        backend=backend,
        model=model,
        role="coordinator",
        system=PLANNER_SYSTEM,
        user=json.dumps(user, ensure_ascii=False, indent=2),
        max_tokens=max_tokens,
    )
    row = {
        "kind": "swectx_planner",
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "role": "coordinator",
        "model": model,
        "ok": parsed is not None,
        "parse_error": err,
        **_call_fields(call),
        "raw_head": raw[:500],
        "parsed": parsed,
    }
    _write_jsonl(result_path, row)
    if parsed is None:
        raise RuntimeError(f"planner failed: {err}")
    with open(seed_path, "w", encoding="utf-8") as f:
        json.dump(parsed, f, ensure_ascii=False, indent=2)
    return parsed


async def analyze_chunk(
    *,
    backend: Backend,
    model: str,
    chunk_id: str,
    cases: list[ContextCase],
    seed_tags: list[dict[str, Any]],
    max_tokens: int,
) -> dict[str, Any]:
    parsed, err, raw, call = await call_json(
        backend=backend,
        model=model,
        role="worker",
        system=WORKER_SYSTEM,
        user=render_cases_chunk(chunk_id, cases, seed_tags),
        max_tokens=max_tokens,
    )
    if parsed is not None:
        parsed["chunk_id"] = chunk_id
    return {
        "kind": "swectx_worker",
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "role": "worker",
        "model": model,
        "chunk_id": chunk_id,
        "case_ids": [c.related_id for c in cases],
        "case_count": len(cases),
        "payload_chars": sum(c.rendered_len for c in cases),
        "ok": parsed is not None,
        "parse_error": err,
        **_call_fields(call),
        "raw_head": raw[:500],
        "parsed": parsed,
    }


def flatten_case_results(worker_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in worker_rows:
        parsed = row.get("parsed") or {}
        for item in parsed.get("case_results", []) or []:
            if isinstance(item, dict):
                item = dict(item)
                item.setdefault("chunk_id", row.get("chunk_id"))
                out.append(item)
    return out


async def run_final_coordinator(
    *,
    backend: Backend,
    model: str,
    cases: list[ContextCase],
    planner: dict[str, Any],
    worker_rows: list[dict[str, Any]],
    result_path: str,
    max_tokens: int,
) -> dict[str, Any] | None:
    case_results = flatten_case_results(worker_rows)
    compact = []
    for r in case_results:
        compact.append({
            "related_instance_id": r.get("related_instance_id"),
            "repo": r.get("repo"),
            "experience_instance_ids": r.get("experience_instance_ids"),
            "context_value": r.get("context_value"),
            "context_value_score": r.get("context_value_score"),
            "primary_tags": r.get("primary_tags"),
            "relationship_type_inferred": r.get("relationship_type_inferred"),
            "reusable_context_unit": r.get("reusable_context_unit"),
            "why_context_helps": _clip(r.get("why_context_helps"), 600),
            "why_context_may_hurt": _clip(r.get("why_context_may_hurt"), 350),
            "evidence": (r.get("evidence") or [])[:2],
            "confidence": r.get("confidence"),
        })
    user = {
        "dataset_facts": {
            "expected_unique_related_cases": len(cases),
            "worker_case_results": len(case_results),
            "source_repo": "https://github.com/jiayuanz3/SWEContextBench",
            "hf_dataset": "https://huggingface.co/datasets/jiayuanz3/SWEContextBench",
            "paper": "https://arxiv.org/abs/2602.08316",
        },
        "planner_seed_tags": planner.get("seed_tags", []),
        "worker_results": compact,
    }
    parsed, err, raw, call = await call_json(
        backend=backend,
        model=model,
        role="coordinator",
        system=COORDINATOR_SYSTEM,
        user=json.dumps(user, ensure_ascii=False, indent=2),
        max_tokens=max_tokens,
    )
    _write_jsonl(result_path, {
        "kind": "swectx_coordinator",
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "role": "coordinator",
        "model": model,
        "ok": parsed is not None,
        "parse_error": err,
        **_call_fields(call),
        "raw_head": raw[:500],
        "parsed": parsed,
    })
    if err:
        print(f"[swectx] coordinator error: {err[:200]}")
    return parsed


def _usage_by_role_model(rows: list[dict[str, Any]]) -> dict[tuple[str, str], int]:
    usage: dict[tuple[str, str], int] = {}
    for row in rows:
        total = int(row.get("tokens_total") or 0)
        if not total:
            continue
        key = (str(row.get("role") or "?"), str(row.get("model") or row.get("backend") or "?"))
        usage[key] = usage.get(key, 0) + total
    return usage


def write_digest(out_dir: str, expected_cases: int | None = None) -> str:
    os.makedirs(out_dir, exist_ok=True)
    result_path = os.path.join(out_dir, "swe_contextbench_results.jsonl")
    rows = _load_rows(result_path)
    worker_rows = list(_latest_ok(rows, "swectx_worker", "chunk_id").values())
    planner_rows = [r for r in rows if r.get("kind") == "swectx_planner" and r.get("ok")]
    coord_rows = [r for r in rows if r.get("kind") == "swectx_coordinator" and r.get("ok")]
    planner = (planner_rows[-1].get("parsed") if planner_rows else {}) or {}
    coord = (coord_rows[-1].get("parsed") if coord_rows else {}) or {}
    case_results = flatten_case_results(worker_rows)

    value_counts: dict[str, int] = {}
    tag_counts: dict[str, int] = {}
    for item in case_results:
        value = str(item.get("context_value") or "unknown")
        value_counts[value] = value_counts.get(value, 0) + 1
        for tag in item.get("primary_tags") or []:
            tag = str(tag)
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

    rows_for_usage = worker_rows + planner_rows[-1:] + coord_rows[-1:]
    usage = _usage_by_role_model(rows_for_usage)
    expected = expected_cases or len({cid for row in worker_rows for cid in row.get("case_ids", [])})

    out: list[str] = []
    out.append(f"# SWEContextBench Context-Value Swarm Digest — {os.path.basename(out_dir)}\n")
    out.append("## Sources\n")
    out.append("- GitHub repo: https://github.com/jiayuanz3/SWEContextBench")
    out.append("- Hugging Face dataset: https://huggingface.co/datasets/jiayuanz3/SWEContextBench")
    out.append("- Paper: https://arxiv.org/abs/2602.08316")
    out.append("\n## Run Summary\n")
    out.append(f"- worker chunks completed: **{len(worker_rows)}**")
    out.append(f"- case results returned: **{len(case_results)} / {expected}**")
    out.append(f"- planner seed tags: **{len(planner.get('seed_tags') or [])}**")
    out.append(f"- final coordinator ok: **{bool(coord)}**")
    if value_counts:
        out.append("- context value distribution: " + ", ".join(
            f"`{k}`={v}" for k, v in sorted(value_counts.items())
        ))
    if usage:
        out.append("- effective swarm token usage:")
        for (role, model), total in sorted(usage.items()):
            out.append(f"  - `{role}` `{model}`: {total:,} total tokens")

    if coord.get("run_quality"):
        rq = coord["run_quality"]
        out.append(f"- coverage: {rq.get('coverage', '')}")
        out.append(f"- limitations: {rq.get('limitations', '')}")

    out.append("\n## Final Categories\n")
    categories = coord.get("final_categories") or []
    if categories:
        out.append("| category | cases | value mix | definition | memory unit |")
        out.append("|---|---:|---|---|---|")
        for cat in categories:
            mix = cat.get("value_mix") or {}
            mix_s = ", ".join(f"{k}:{v}" for k, v in mix.items())
            definition = str(cat.get("definition", "")).replace("|", "\\|")[:220]
            memory_unit = str(cat.get("what_to_store_as_memory", "")).replace("|", "\\|")[:220]
            out.append(
                f"| `{cat.get('category','unknown')}` | {cat.get('case_count', 0)} | "
                f"{mix_s} | {definition} | {memory_unit} |"
            )
    else:
        out.append("| tag | count |")
        out.append("|---|---:|")
        for tag, count in sorted(tag_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:30]:
            out.append(f"| `{tag}` | {count} |")

    if coord.get("highest_value_cases"):
        out.append("\n## Highest-Value Cases\n")
        out.append("| related case | repo | experience | category | score | why |")
        out.append("|---|---|---|---|---:|---|")
        for item in coord["highest_value_cases"][:30]:
            exps = ", ".join(item.get("experience_instance_ids") or [])
            why = str(item.get("why", "")).replace("|", "\\|")[:260]
            out.append(
                f"| `{item.get('related_instance_id','')}` | {item.get('repo','')} | "
                f"`{exps}` | `{item.get('category','')}` | {item.get('score','')} | {why} |"
            )

    if coord.get("low_or_misleading_context_patterns"):
        out.append("\n## Low Or Misleading Context Patterns\n")
        for item in coord["low_or_misleading_context_patterns"]:
            out.append(f"- {item}")

    if coord.get("retrieval_implications"):
        out.append("\n## Retrieval Implications\n")
        for item in coord["retrieval_implications"]:
            out.append(f"- {item}")

    out.append("\n## Worker Tag Counts\n")
    if tag_counts:
        for tag, count in sorted(tag_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:40]:
            out.append(f"- `{tag}`: {count}")
    else:
        out.append("(none)")

    digest_path = os.path.join(out_dir, "swe_contextbench_digest.md")
    with open(digest_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    return digest_path


async def run(args: argparse.Namespace) -> int:
    runtime = normalize_runtime(args.runtime)
    worker_model = args.worker_model or default_model_for_runtime(runtime, "worker")
    coordinator_model = args.coordinator_model or default_model_for_runtime(runtime, "coordinator")
    worker_backend = build_backend(runtime)
    coordinator_backend = worker_backend

    out_dir = os.path.abspath(os.path.join(args.output_dir, args.run_id))
    os.makedirs(out_dir, exist_ok=True)
    result_path = os.path.join(out_dir, "swe_contextbench_results.jsonl")

    cases = load_context_cases(args.dataset_dir)
    chunks = build_chunks(cases, args.max_chars_per_chunk, args.max_cases)
    manifest_path = os.path.join(out_dir, "swe_contextbench_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({
            "dataset_dir": str(_repo_root_from_dataset_arg(args.dataset_dir)),
            "case_count": len(cases),
            "selected_case_count": args.max_cases or len(cases),
            "chunk_count": len(chunks),
            "chunks": [
                {
                    "chunk_id": chunk_id,
                    "case_ids": [case.related_id for case in chunk_cases],
                    "case_count": len(chunk_cases),
                    "payload_chars": sum(c.rendered_len for c in chunk_cases),
                }
                for chunk_id, chunk_cases in chunks
            ],
        }, f, ensure_ascii=False, indent=2)

    print(
        f"[swectx] runtime={runtime} worker={worker_model} coordinator={coordinator_model} "
        f"cases={len(cases)} chunks={len(chunks)} output={out_dir}"
    )

    planner = await run_planner(
        backend=coordinator_backend,
        model=coordinator_model,
        cases=cases,
        sample_n=args.planner_sample_cases,
        out_dir=out_dir,
        result_path=result_path,
        max_tokens=args.planner_max_tokens,
    )
    seed_tags = planner.get("seed_tags") or []
    if not isinstance(seed_tags, list):
        seed_tags = []

    existing = _latest_ok(_load_rows(result_path), "swectx_worker", "chunk_id")
    chunks_to_process = [(cid, cs) for cid, cs in chunks if cid not in existing]
    print(
        f"[swectx] seed_tags={len(seed_tags)} to_process={len(chunks_to_process)} "
        f"already_done={len(existing)}"
    )

    queue: asyncio.Queue[tuple[str, list[ContextCase]] | None] = asyncio.Queue()

    async def worker(worker_idx: int) -> None:
        while True:
            item = await queue.get()
            if item is None:
                queue.task_done()
                break
            chunk_id, chunk_cases = item
            row = await analyze_chunk(
                backend=worker_backend,
                model=worker_model,
                chunk_id=chunk_id,
                cases=chunk_cases,
                seed_tags=seed_tags,
                max_tokens=args.worker_max_tokens,
            )
            _write_jsonl(result_path, row)
            if row.get("ok"):
                print(f"[swectx] worker {worker_idx} ok {chunk_id} cases={len(chunk_cases)} tokens={row.get('tokens_total')}")
            else:
                print(f"[swectx] worker {worker_idx} error {chunk_id}: {str(row.get('parse_error'))[:180]}")
            queue.task_done()

    workers = [asyncio.create_task(worker(i)) for i in range(args.concurrency)]
    for item in chunks_to_process:
        await queue.put(item)
    for _ in workers:
        await queue.put(None)
    await queue.join()
    await asyncio.gather(*workers)

    latest_rows = _latest_ok(_load_rows(result_path), "swectx_worker", "chunk_id")
    worker_rows = list(latest_rows.values())
    await run_final_coordinator(
        backend=coordinator_backend,
        model=coordinator_model,
        cases=cases[: args.max_cases] if args.max_cases else cases,
        planner=planner,
        worker_rows=worker_rows,
        result_path=result_path,
        max_tokens=args.coordinator_max_tokens,
    )
    digest = write_digest(out_dir, expected_cases=args.max_cases or len(cases))
    print(f"[swectx] wrote {digest}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-dir", default="/Users/bytedance/code/c/SWEContextBench")
    p.add_argument("--runtime", default="codex")
    p.add_argument("--worker-model", default="gpt-5.3-codex-spark")
    p.add_argument("--coordinator-model", default="gpt-5.5")
    p.add_argument("--concurrency", type=int, default=3)
    p.add_argument("--max-cases", type=int, default=None)
    p.add_argument("--max-chars-per-chunk", type=int, default=180000)
    p.add_argument("--planner-sample-cases", type=int, default=28)
    p.add_argument("--planner-max-tokens", type=int, default=3000)
    p.add_argument("--worker-max-tokens", type=int, default=1400)
    p.add_argument("--coordinator-max-tokens", type=int, default=6000)
    p.add_argument("--output-dir", default="runs")
    p.add_argument("--run-id", default=time.strftime("swectx-%Y%m%d-%H%M%S"))
    p.add_argument("--summarize-only", action="store_true")
    args = p.parse_args(argv)
    if args.summarize_only:
        out_dir = os.path.abspath(os.path.join(args.output_dir, args.run_id))
        digest = write_digest(out_dir, expected_cases=args.max_cases)
        print(f"[swectx] wrote {digest}")
        return 0
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
