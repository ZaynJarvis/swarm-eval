"""Orchestrator — single event loop wiring queue, semaphore, workers, coordinator.

Per tim's review (2026-05-05), retry-after-reflection is gone. Escalations are
resolved by the coordinator with a writeback to session_results, and any
sessions that finish with primary_pattern='unknown' OR confidence<0.6 are picked
up by a single bounded final-pass sweep against the final pattern_version.
"""
from __future__ import annotations

import asyncio
from collections import deque
import json
import os
import time
from dataclasses import dataclass, field

from backends.base import Backend
from coordinator import run_reflection
from linker import LinkedSession, iter_linked
from pattern_store import Pattern, PatternStore
from worker import WorkerOutcome, load_template, run_worker


@dataclass
class RunConfig:
    result_dir: str
    run_id: str
    out_dir: str
    worker_backend: Backend
    coordinator_backend: Backend
    worker_model: str = "claude-haiku-4-5-20251001"
    coordinator_model: str = "claude-opus-4-7"
    concurrency: int = 10
    limit: int | None = None
    reflect_every_n_sessions: int = 50          # bumped 25→50 per tim
    reflect_every_n_escalations: int = 5        # responsive path
    reflect_min_interval_s: float = 30.0        # min spacing between reflections
    reflect_idle_window_s: float = 240.0        # gated time-fallback (only if escalations>0)
    midrun_reflection: bool = False             # keep worker prompt/library frozen during the run
    final_reflection: bool = True               # one post-run coordinator pass before sweep
    final_sweep_low_conf: float = 0.6           # confidence threshold for sweep
    final_sweep_max_sessions: int = 200         # cap to keep cost bounded
    worker_max_tokens: int = 700                # enforced by SDK; CLI cannot enforce this
    coordinator_max_tokens: int = 4096
    circuit_breaker_consecutive_failures: int = 5
    circuit_breaker_rolling_window: int = 20
    circuit_breaker_rolling_failure_rate: float = 0.5
    scope: str = "all"                          # all | wrong-only | wrong-plus-correct-calibration
    correct_sample_per_bucket: int = 10


SEED_PATTERNS: list[Pattern] = [
    Pattern(
        signature="early-commit-wrong-date",
        category="accuracy",
        symptom="Model commits in 1 turn on hook-injected hint without verifying date arithmetic; off-by-one or wrong day-of-week.",
        rule_for_workers="If verdict=wrong AND num_turns<=2 AND question asks for a specific date AND grader_reasoning mentions a near-but-different date, classify here.",
    ),
    Pattern(
        signature="hook-thrash-no-mcp",
        category="turns",
        symptom="Model loops inside hook content for many turns without ever calling an MCP tool to fetch ground truth.",
        rule_for_workers="If num_turns>=5 AND ov_mcp_calls=0, classify here regardless of verdict.",
    ),
    Pattern(
        signature="missing-detail-generic-answer",
        category="accuracy",
        symptom="Model returns a vague paraphrase that misses the specific detail the gold answer requires (e.g. 'explore nature' vs 'roast marshmallows, tell stories').",
        rule_for_workers="If verdict=wrong AND grader_reasoning says response is generic/paraphrase or misses specific items present in gold, classify here.",
    ),
    Pattern(
        signature="hallucinated-event-window",
        category="accuracy",
        symptom="Hook content lacks the answer; model invents a plausible-sounding event/date instead of calling MCP to look it up.",
        rule_for_workers="If verdict=wrong AND ov_mcp_calls=0 AND grader_reasoning notes the response refers to an event/date not in evidence, classify here.",
    ),
    Pattern(
        signature="off-by-count",
        category="accuracy",
        symptom="Question asks 'how many'; model answers a different integer than gold (often 1-vs-2 from finding only one mention).",
        rule_for_workers="If verdict=wrong AND question starts with 'how many' or 'how often' AND response and gold are different small integers/words, classify here.",
    ),
    Pattern(
        signature="expensive-correct-mcp-thrash",
        category="tokens",
        symptom="Answer is CORRECT but cost/tokens are high because the model spent extra turns or MCP calls before converging.",
        rule_for_workers="If verdict=correct AND cost_context high_flags is non-empty AND turns>=4 or transcript_mcp>=3, classify here.",
    ),
    Pattern(
        signature="redundant-mcp-after-sufficient-hook",
        category="tool_use",
        symptom="Hook context already contains enough evidence, but the model still makes redundant MCP calls before the correct answer.",
        rule_for_workers="If verdict=correct AND hook_context contains the answer evidence AND transcript_mcp>=1, classify here.",
    ),
    Pattern(
        signature="high-output-overexplanation",
        category="tokens",
        symptom="Answer is CORRECT but output tokens are much higher than expected because the model over-explains a direct QA answer.",
        rule_for_workers="If verdict=correct AND cost_context flags output_tokens high while turns and MCP are not high, classify here.",
    ),
    Pattern(
        signature="metric-mcp-count-mismatch",
        category="tool_use",
        symptom="Logged ov_mcp_calls metric undercounts actual mcp__ tool_use events in the transcript.",
        rule_for_workers="If transcript_mcp > mcp_metric, classify here as secondary; primary only if this is the main anomaly.",
    ),
    Pattern(
        signature="truncated-context-extra-search",
        category="tokens",
        symptom="Relevant tool or hook content appears truncated, causing extra search/tool work before an ultimately correct answer.",
        rule_for_workers="If verdict=correct AND transcript/tool_result shows truncation markers AND high_flags has cost or tokens, classify here.",
    ),
]


def write_jsonl(path: str, obj: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _read_run_summary(result_dir: str) -> str:
    parts = []
    for fn in ("summary.txt", "experiment_report.md"):
        p = os.path.join(result_dir, fn)
        if os.path.isfile(p):
            with open(p, encoding="utf-8") as f:
                parts.append(f"=== {fn} ===\n" + f.read()[:4000])
    return "\n\n".join(parts) or "(no summary files)"


def _total_session_tokens(s: LinkedSession) -> int:
    return s.input_tokens + s.cache_read_input_tokens + s.output_tokens


def _session_mcp_calls(s: LinkedSession) -> int:
    return max(s.ov_mcp_calls, s.transcript_mcp_calls)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def _runtime_stats(sessions: list[LinkedSession]) -> dict[str, float]:
    return {
        "cost_usd": _mean([s.total_cost_usd for s in sessions]),
        "input_tokens": _mean([s.input_tokens for s in sessions]),
        "cache_read_tokens": _mean([s.cache_read_input_tokens for s in sessions]),
        "output_tokens": _mean([s.output_tokens for s in sessions]),
        "total_tokens": _mean([_total_session_tokens(s) for s in sessions]),
        "turns": _mean([s.num_turns for s in sessions]),
        "mcp_calls": _mean([_session_mcp_calls(s) for s in sessions]),
    }


def _runtime_p90(sessions: list[LinkedSession]) -> dict[str, float]:
    return {
        "cost_usd": _percentile([s.total_cost_usd for s in sessions], 0.9),
        "input_tokens": _percentile([s.input_tokens for s in sessions], 0.9),
        "cache_read_tokens": _percentile([s.cache_read_input_tokens for s in sessions], 0.9),
        "output_tokens": _percentile([s.output_tokens for s in sessions], 0.9),
        "total_tokens": _percentile([_total_session_tokens(s) for s in sessions], 0.9),
        "turns": _percentile([s.num_turns for s in sessions], 0.9),
        "mcp_calls": _percentile([_session_mcp_calls(s) for s in sessions], 0.9),
    }


def _ratio(value: float, baseline: float) -> float:
    return value / baseline if baseline > 0 else 0.0


def _fmt_metric_value(metric: str, value: float) -> str:
    if metric == "cost_usd":
        return f"${value:.4f}"
    if metric in {"turns", "mcp_calls"}:
        return f"{value:.1f}"
    return f"{value:.0f}"


def _build_cost_contexts(sessions: list[LinkedSession]) -> dict[str, str]:
    """Build compact per-session runtime context for expensive-correct analysis."""
    if not sessions:
        return {}
    correct_sessions = [s for s in sessions if s.result.upper() == "CORRECT"] or sessions
    all_avg = _runtime_stats(sessions)
    correct_p90 = _runtime_p90(correct_sessions)
    contexts: dict[str, str] = {}
    for s in sessions:
        values = {
            "cost_usd": s.total_cost_usd,
            "input_tokens": float(s.input_tokens),
            "cache_read_tokens": float(s.cache_read_input_tokens),
            "output_tokens": float(s.output_tokens),
            "total_tokens": float(_total_session_tokens(s)),
            "turns": float(s.num_turns),
            "mcp_calls": float(_session_mcp_calls(s)),
        }
        high_flags: list[str] = []
        for metric, value in values.items():
            high_threshold = max(all_avg.get(metric, 0.0) * 2.0, correct_p90.get(metric, 0.0))
            if s.result.upper() == "CORRECT" and high_threshold > 0 and value >= high_threshold:
                high_flags.append(metric)
        current = ", ".join(f"{k}={_fmt_metric_value(k, v)}" for k, v in values.items())
        ratios = ", ".join(
            f"{k}={_ratio(values[k], all_avg.get(k, 0.0)):.1f}x_avg"
            for k in ("cost_usd", "total_tokens", "output_tokens", "turns", "mcp_calls")
        )
        thresholds = ", ".join(
            f"{k}>={_fmt_metric_value(k, max(all_avg.get(k, 0.0) * 2.0, correct_p90.get(k, 0.0)))}"
            for k in ("cost_usd", "total_tokens", "output_tokens", "turns", "mcp_calls")
        )
        contexts[s.session_id] = (
            "Use only for CORRECT expensive-but-correct analysis; "
            f"high_flags={high_flags or '[]'}; current: {current}; "
            f"ratios_vs_dataset_avg: {ratios}; high_thresholds: {thresholds}"
        )
    return contexts


@dataclass
class _RunState:
    cfg: RunConfig
    store: PatternStore
    log_path: str
    cost_context_by_session: dict[str, str] = field(default_factory=dict)
    pattern_version: int = 1
    sessions_done_since_reflection: int = 0
    last_reflection_t: float = field(default_factory=time.time)
    frozen_patterns_fragment: str | None = None
    abort_event: asyncio.Event = field(default_factory=asyncio.Event)
    abort_reason: str | None = None
    breaker: "_CircuitBreaker | None" = None


@dataclass
class _CircuitBreaker:
    consecutive_limit: int
    rolling_window: int
    rolling_failure_rate: float
    consecutive_failures: int = 0
    outcomes: deque[bool] = field(default_factory=deque)
    tripped_reason: str | None = None

    def record(self, *, ok: bool, label: str) -> str | None:
        if self.tripped_reason:
            return self.tripped_reason
        self.outcomes.append(ok)
        if self.rolling_window > 0 and len(self.outcomes) > self.rolling_window:
            self.outcomes.popleft()
        if ok:
            self.consecutive_failures = 0
            return None
        self.consecutive_failures += 1
        if (
            self.consecutive_limit > 0
            and self.consecutive_failures >= self.consecutive_limit
        ):
            self.tripped_reason = (
                f"{label}: {self.consecutive_failures} consecutive backend failures"
            )
            return self.tripped_reason
        if self.rolling_window > 0 and len(self.outcomes) == self.rolling_window:
            failures = sum(1 for x in self.outcomes if not x)
            rate = failures / self.rolling_window
            if rate > self.rolling_failure_rate:
                self.tripped_reason = (
                    f"{label}: rolling backend failure rate {rate:.0%} "
                    f"({failures}/{self.rolling_window})"
                )
                return self.tripped_reason
        return None


async def _seed_patterns(store: PatternStore) -> None:
    for p in SEED_PATTERNS:
        store.upsert_pattern(p)


def _worker_patterns_fragment(state: _RunState) -> str:
    if state.frozen_patterns_fragment is not None:
        return state.frozen_patterns_fragment
    return state.store.build_worker_pattern_fragment()


def _record_backend_result(
    *,
    state: _RunState,
    ok: bool,
    label: str,
    session_id: str | None = None,
) -> None:
    if state.breaker is None or state.abort_event.is_set():
        return
    reason = state.breaker.record(ok=ok, label=label)
    if reason:
        state.abort_reason = reason
        state.abort_event.set()
        write_jsonl(state.log_path, {
            "kind": "run_aborted",
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "reason": reason,
            "session_id": session_id,
        })
        print(f"[abort] {reason}")


def _persist_worker_outcome(*, state: _RunState, name: str, linked: LinkedSession,
                             outcome: WorkerOutcome, attempt: int) -> None:
    log_obj = {
        "kind": "worker_outcome",
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "worker": name,
        "session_id": linked.session_id,
        "sample_id": linked.sample_id,
        "qix": linked.question_index,
        "num_turns": linked.num_turns,
        "ov_mcp_calls": linked.ov_mcp_calls,
        "transcript_mcp_calls": linked.transcript_mcp_calls,
        "session_cost_usd": linked.total_cost_usd,
        "session_input_tokens": linked.input_tokens,
        "session_cache_read_tokens": linked.cache_read_input_tokens,
        "session_output_tokens": linked.output_tokens,
        "session_total_tokens": _total_session_tokens(linked),
        "verdict_truth": linked.result,
        "attempt": attempt,
        "pattern_version": state.pattern_version,
        "ok": outcome.parsed is not None,
        "parse_error": outcome.parse_error,
        "tokens_in": outcome.call.input_tokens,
        "tokens_cache": outcome.call.cache_read_tokens,
        "tokens_out": outcome.call.output_tokens,
        "backend": outcome.call.backend,
        "raw_head": (outcome.raw_text or "")[:500],
        "parsed": outcome.parsed,
    }
    write_jsonl(state.log_path, log_obj)
    _record_backend_result(
        state=state,
        ok=outcome.call.error is None,
        label="worker",
        session_id=linked.session_id,
    )

    if outcome.parsed:
        sr = {
            "session_id": linked.session_id,
            "qa_sample_id": linked.sample_id,
            "qa_question_index": linked.question_index,
            **outcome.parsed,
            "attempt": attempt,
            "pattern_version": state.pattern_version,
        }
        state.store.write_session_result(sr)
        sig = outcome.parsed.get("primary_pattern")
        if sig and sig != "unknown":
            state.store.bump_pattern_count(
                sig,
                {"session_id": linked.session_id, "sample_id": linked.sample_id,
                 "qix": linked.question_index},
            )
        if outcome.parsed.get("escalate") and outcome.parsed.get("escalate_question"):
            state.store.add_escalation(
                linked.session_id, outcome.parsed["escalate_question"]
            )


async def _worker_task(*, name: str, queue: asyncio.Queue, sem: asyncio.Semaphore,
                       state: _RunState, template: str) -> None:
    cfg = state.cfg
    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            break
        linked: LinkedSession = item
        if state.abort_event.is_set():
            queue.task_done()
            continue
        async with sem:
            patterns_fragment = _worker_patterns_fragment(state)
            outcome: WorkerOutcome = await run_worker(
                backend=cfg.worker_backend,
                model=cfg.worker_model,
                linked=linked,
                pattern_version=state.pattern_version,
                patterns_fragment=patterns_fragment,
                template=template,
                max_tokens=cfg.worker_max_tokens,
                cost_context=state.cost_context_by_session.get(linked.session_id),
            )
        _persist_worker_outcome(state=state, name=name, linked=linked,
                                outcome=outcome, attempt=1)
        state.sessions_done_since_reflection += 1
        queue.task_done()


async def _reflection_task(*, state: _RunState, run_context: str,
                           shutdown: asyncio.Event) -> None:
    cfg = state.cfg
    while not shutdown.is_set():
        await asyncio.sleep(5.0)
        if state.abort_event.is_set():
            shutdown.set()
            break
        store = state.store
        unresolved = len(store.unresolved_escalations())
        elapsed = time.time() - state.last_reflection_t
        if elapsed < cfg.reflect_min_interval_s:
            continue
        # Tim's adjusted triggers:
        #   responsive: ≥5 unresolved escalations
        #   batch:      ≥50 sessions since last reflection
        #   gated time: only if there's actually unresolved escalations
        responsive = unresolved >= cfg.reflect_every_n_escalations
        batch = state.sessions_done_since_reflection >= cfg.reflect_every_n_sessions
        time_gated = unresolved > 0 and elapsed >= cfg.reflect_idle_window_s
        if not (responsive or batch or time_gated):
            continue
        print(f"[reflect] trigger reasons: responsive={responsive} "
              f"batch={batch} time={time_gated} | unresolved={unresolved} "
              f"done_since={state.sessions_done_since_reflection} elapsed={elapsed:.0f}s")
        outcome = await run_reflection(
            backend=cfg.coordinator_backend,
            model=cfg.coordinator_model,
            store=store,
            run_context=run_context,
            pattern_version=state.pattern_version,
            max_tokens=cfg.coordinator_max_tokens,
        )
        write_jsonl(state.log_path, {
            "kind": "reflection",
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "ok": outcome.parsed is not None,
            "parse_error": outcome.parse_error,
            "applied_ops": outcome.applied_ops,
            "resolved_escs": outcome.resolved_escs,
            "writebacks": outcome.writebacks,
            "tokens_in": outcome.call.input_tokens,
            "tokens_cache": outcome.call.cache_read_tokens,
            "tokens_out": outcome.call.output_tokens,
            "raw_head": (outcome.raw_text or "")[:500],
        })
        _record_backend_result(
            state=state,
            ok=outcome.call.error is None,
            label="reflection",
        )
        if outcome.parsed:
            state.pattern_version += 1
            state.store.set_meta("pattern_version", str(state.pattern_version))
        state.sessions_done_since_reflection = 0
        state.last_reflection_t = time.time()


def _select_sweep_targets(state: _RunState, all_sessions: list[LinkedSession]) -> list[LinkedSession]:
    """Pick sessions whose current classification is 'unknown' or below confidence
    threshold for re-running with the final pattern library."""
    cfg = state.cfg
    cur = state.store.conn.cursor()
    cur.execute(
        "SELECT session_id, verdict, primary_pattern, confidence "
        "FROM session_results"
    )
    by_id = {
        r["session_id"]: (r["verdict"], r["primary_pattern"], r["confidence"])
        for r in cur.fetchall()
    }
    targets: list[LinkedSession] = []
    for ls in all_sessions:
        if ls.session_id not in by_id:
            # never produced a result — sweep too
            targets.append(ls)
            continue
        verdict, sig, conf = by_id[ls.session_id]
        if verdict == "correct" and sig == "unknown" and (
            conf is not None and conf >= cfg.final_sweep_low_conf
        ):
            # "unknown" is an acceptable terminal label for efficient CORRECT
            # sessions: no failure and no overload pattern to diagnose.
            continue
        if sig == "unknown" or (conf is not None and conf < cfg.final_sweep_low_conf):
            targets.append(ls)
    return targets[: cfg.final_sweep_max_sessions]


async def _final_sweep(*, state: _RunState, all_sessions: list[LinkedSession],
                       template: str) -> int:
    cfg = state.cfg
    targets = _select_sweep_targets(state, all_sessions)
    if not targets:
        print("[sweep] no targets")
        return 0
    print(f"[sweep] running final-pass on {len(targets)} sessions at pv={state.pattern_version}")
    queue: asyncio.Queue = asyncio.Queue()
    sem = asyncio.Semaphore(cfg.concurrency)

    async def _sweep_worker(idx: int) -> None:
        while True:
            item = await queue.get()
            if item is None:
                queue.task_done()
                break
            linked: LinkedSession = item
            if state.abort_event.is_set():
                queue.task_done()
                continue
            async with sem:
                patterns_fragment = _worker_patterns_fragment(state)
                outcome = await run_worker(
                    backend=cfg.worker_backend,
                    model=cfg.worker_model,
                    linked=linked,
                    pattern_version=state.pattern_version,
                    patterns_fragment=patterns_fragment,
                    template=template,
                    max_tokens=cfg.worker_max_tokens,
                    cost_context=state.cost_context_by_session.get(linked.session_id),
                )
            # log under "sweep_outcome" so digest can distinguish
            log_obj = {
                "kind": "sweep_outcome",
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "worker": f"sw{idx}",
                "session_id": linked.session_id,
                "sample_id": linked.sample_id,
                "qix": linked.question_index,
                "num_turns": linked.num_turns,
                "ov_mcp_calls": linked.ov_mcp_calls,
                "transcript_mcp_calls": linked.transcript_mcp_calls,
                "session_cost_usd": linked.total_cost_usd,
                "session_input_tokens": linked.input_tokens,
                "session_cache_read_tokens": linked.cache_read_input_tokens,
                "session_output_tokens": linked.output_tokens,
                "session_total_tokens": _total_session_tokens(linked),
                "verdict_truth": linked.result,
                "pattern_version": state.pattern_version,
                "ok": outcome.parsed is not None,
                "parse_error": outcome.parse_error,
                "tokens_in": outcome.call.input_tokens,
                "tokens_cache": outcome.call.cache_read_tokens,
                "tokens_out": outcome.call.output_tokens,
                "backend": outcome.call.backend,
                "raw_head": (outcome.raw_text or "")[:500],
                "parsed": outcome.parsed,
            }
            write_jsonl(state.log_path, log_obj)
            _record_backend_result(
                state=state,
                ok=outcome.call.error is None,
                label="sweep",
                session_id=linked.session_id,
            )
            if outcome.parsed:
                # Only overwrite session_results if the sweep actually improved
                # the classification (i.e. moved away from 'unknown' OR raised
                # confidence). Don't regress.
                cur = state.store.conn.cursor()
                cur.execute(
                    "SELECT primary_pattern, confidence FROM session_results WHERE session_id=?",
                    (linked.session_id,),
                )
                prev = cur.fetchone()
                new_sig = outcome.parsed.get("primary_pattern")
                new_conf = outcome.parsed.get("confidence", 0.0)
                allow = (
                    prev is None
                    or (prev["primary_pattern"] == "unknown" and new_sig != "unknown")
                    or (new_sig != "unknown" and float(new_conf) > float(prev["confidence"] or 0))
                )
                if allow:
                    sr = {
                        "session_id": linked.session_id,
                        "qa_sample_id": linked.sample_id,
                        "qa_question_index": linked.question_index,
                        **outcome.parsed,
                        "attempt": 2,
                        "pattern_version": state.pattern_version,
                    }
                    state.store.write_session_result(sr)
                    if new_sig and new_sig != "unknown":
                        state.store.bump_pattern_count(
                            new_sig,
                            {"session_id": linked.session_id,
                             "sample_id": linked.sample_id,
                             "qix": linked.question_index, "via": "final_sweep"},
                        )
            queue.task_done()

    workers = [asyncio.create_task(_sweep_worker(i)) for i in range(cfg.concurrency)]
    for ls in targets:
        await queue.put(ls)
    for _ in workers:
        await queue.put(None)
    await queue.join()
    await asyncio.gather(*workers, return_exceptions=True)
    print(f"[sweep] done.")
    return len(targets)


def _turn_bucket(num_turns: int) -> str:
    if num_turns == 1:
        return "turns=1"
    if num_turns <= 3:
        return "turns=2-3"
    if num_turns <= 7:
        return "turns=4-7"
    return "turns=8+"


def _mcp_sample_bucket(ov_mcp_calls: int) -> str:
    return "mcp=0" if ov_mcp_calls == 0 else "mcp=1+"


def _runtime_pressure_score(s: LinkedSession) -> tuple[float, int, int, int]:
    return (
        s.total_cost_usd,
        _total_session_tokens(s),
        s.num_turns,
        _session_mcp_calls(s),
    )


def _apply_scope(sessions: list[LinkedSession], cfg: RunConfig) -> list[LinkedSession]:
    if cfg.scope == "all":
        return sessions
    if cfg.scope == "wrong-only":
        return [s for s in sessions if s.result.upper() == "WRONG"]
    if cfg.scope != "wrong-plus-correct-calibration":
        raise ValueError(f"unknown scope: {cfg.scope}")

    selected: list[LinkedSession] = []
    for s in sessions:
        if s.result.upper() == "WRONG":
            selected.append(s)

    per_bucket: dict[tuple[str, str], int] = {}
    correct_sessions = [s for s in sessions if s.result.upper() == "CORRECT"]
    correct_sessions.sort(key=_runtime_pressure_score, reverse=True)
    for s in correct_sessions:
        mcp_count = max(s.ov_mcp_calls, s.transcript_mcp_calls)
        key = (_turn_bucket(s.num_turns), _mcp_sample_bucket(mcp_count))
        if per_bucket.get(key, 0) >= cfg.correct_sample_per_bucket:
            continue
        selected.append(s)
        per_bucket[key] = per_bucket.get(key, 0) + 1

    order = {s.session_id: i for i, s in enumerate(sessions)}
    selected.sort(key=lambda s: order[s.session_id])
    return selected


def _partition_resume_sessions(
    all_sessions: list[LinkedSession],
    already_classified: set[str],
) -> tuple[list[LinkedSession], list[LinkedSession]]:
    """Return (sessions_to_process, sessions_visible_to_final_sweep).

    Final sweep must see the original scoped session set, including already
    classified rows, so resumed runs can refresh stale unknown/low-confidence
    classifications instead of silently shipping them.
    """
    if not already_classified:
        return all_sessions, all_sessions
    to_process = [s for s in all_sessions if s.session_id not in already_classified]
    return to_process, all_sessions


async def run(cfg: RunConfig) -> None:
    os.makedirs(cfg.out_dir, exist_ok=True)
    db_path = os.path.join(cfg.out_dir, "pattern_store.db")
    log_path = os.path.join(cfg.out_dir, "results.jsonl")
    store = PatternStore(db_path)
    await _seed_patterns(store)

    all_linked_sessions = list(iter_linked(cfg.result_dir, limit=None))
    linked_sessions = all_linked_sessions[: cfg.limit] if cfg.limit is not None else all_linked_sessions
    cost_context_by_session = _build_cost_contexts(all_linked_sessions)
    scoped_sessions = _apply_scope(linked_sessions, cfg)
    if len(scoped_sessions) != len(linked_sessions):
        print(
            f"[orchestrator] scope={cfg.scope}: selected {len(scoped_sessions)} "
            f"of {len(linked_sessions)} linked sessions"
        )
    # Resume-safe: skip sessions already classified in this run dir.
    cur = store.conn.cursor()
    cur.execute("SELECT session_id FROM session_results")
    already = {r["session_id"] for r in cur.fetchall()}
    sessions, sweep_sessions = _partition_resume_sessions(scoped_sessions, already)
    if already:
        before = len(scoped_sessions)
        print(f"[orchestrator] resume: skipping {before - len(sessions)} already-classified sessions")
    print(f"[orchestrator] linked {len(sessions)} sessions to process; concurrency={cfg.concurrency}")

    run_context = _read_run_summary(cfg.result_dir)
    pattern_version = int(store.get_meta("pattern_version", "1") or "1")
    state = _RunState(cfg=cfg, store=store, log_path=log_path,
                      cost_context_by_session=cost_context_by_session,
                      pattern_version=pattern_version)
    if cfg.midrun_reflection is False:
        state.frozen_patterns_fragment = store.build_worker_pattern_fragment()
        print("[orchestrator] worker pattern library frozen for this run")
    state.breaker = _CircuitBreaker(
        consecutive_limit=cfg.circuit_breaker_consecutive_failures,
        rolling_window=cfg.circuit_breaker_rolling_window,
        rolling_failure_rate=cfg.circuit_breaker_rolling_failure_rate,
    )

    queue: asyncio.Queue = asyncio.Queue()
    sem = asyncio.Semaphore(cfg.concurrency)
    template = load_template()
    shutdown = asyncio.Event()

    workers = [
        asyncio.create_task(
            _worker_task(name=f"w{i}", queue=queue, sem=sem, state=state,
                         template=template)
        )
        for i in range(cfg.concurrency)
    ]
    reflector = None
    if cfg.midrun_reflection:
        reflector = asyncio.create_task(
            _reflection_task(state=state, run_context=run_context, shutdown=shutdown)
        )

    for ls in sessions:
        await queue.put(ls)
    for _ in workers:
        await queue.put(None)

    await queue.join()
    shutdown.set()

    # final reflection: integrate any leftover escalations and any new wrong cases
    if (
        cfg.final_reflection
        and not state.abort_event.is_set()
        and (state.sessions_done_since_reflection > 0 or store.unresolved_escalations())
    ):
        outcome = await run_reflection(
            backend=cfg.coordinator_backend,
            model=cfg.coordinator_model,
            store=store,
            run_context=run_context,
            pattern_version=state.pattern_version,
            max_tokens=cfg.coordinator_max_tokens,
        )
        write_jsonl(log_path, {
            "kind": "reflection_final",
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "ok": outcome.parsed is not None,
            "parse_error": outcome.parse_error,
            "applied_ops": outcome.applied_ops,
            "resolved_escs": outcome.resolved_escs,
            "writebacks": outcome.writebacks,
            "tokens_in": outcome.call.input_tokens,
            "tokens_cache": outcome.call.cache_read_tokens,
            "tokens_out": outcome.call.output_tokens,
            "raw_head": (outcome.raw_text or "")[:500],
        })
        _record_backend_result(
            state=state,
            ok=outcome.call.error is None,
            label="reflection_final",
        )
        if outcome.parsed:
            state.pattern_version += 1
            state.store.set_meta("pattern_version", str(state.pattern_version))
            state.frozen_patterns_fragment = None

    # final-pass sweep over remaining unknown / low-confidence sessions
    swept = 0
    if not state.abort_event.is_set():
        swept = await _final_sweep(state=state, all_sessions=sweep_sessions, template=template)
    write_jsonl(log_path, {
        "kind": "sweep_summary",
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "swept": swept,
        "pattern_version": state.pattern_version,
        "aborted": state.abort_event.is_set(),
        "abort_reason": state.abort_reason,
    })

    if reflector is not None:
        reflector.cancel()
        try:
            await reflector
        except asyncio.CancelledError:
            pass
    await asyncio.gather(*workers, return_exceptions=True)
    print("[orchestrator] done.", store.stats())
