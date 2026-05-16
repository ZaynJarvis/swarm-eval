# Swarm Eval — Design

Builder: @zeus  •  Reviewer: @tim  •  Final review: @hela

## Goal

Reusable analysis pipeline for eval session files (locomo10 first; future locomo / new eval datasets). Output: structured **issue patterns** explaining WHY sessions go wrong (accuracy / token waste / turn thrash), so the team can fix prompts, plugin params, or memory ingest.

Constraints from spec:
- 1 Opus coordinator (persistent state) + N Haiku workers (ephemeral).
- Workers escalate when low-confidence; coordinator answers; pattern library grows over the run.
- Concurrent-safe with queue. Local files only (no OV integration in v1).
- Reference zouk-daemon **runtime/spawn pattern** — not a hard plug-in.

## Non-goals (v1)

- Connecting coordinator/workers to OV.
- Real-time eval (this analyzes a finished result dir).
- Cross-dataset generalization beyond simple prompt swap.

---

## Data shape (locomo10 / `result-ov-r6`)

| File | Use |
|---|---|
| `transcripts/<sessionId>.jsonl` | full event stream per QA session: `user`, `attachment` (skill_listing + hook_additional_context), `assistant` (text + tool_use), tool_results (in user msgs), `last-prompt`, `queue-operation`. 1550 files. |
| `qa_results.csv` | 1540 rows: ground truth (`answer`, `evidence`), prediction (`response`), grader verdict (`result` ∈ {CORRECT, WRONG}, `reasoning`), runtime (`num_turns`, `total_cost_usd`, `elapsed_seconds`), recall counters (`ov_recall_hooks`, `ov_mcp_calls`). |
| `sample_mapping.json` | `sample_id → project_dir` (10 conv-* dirs). |
| `summary.txt`, `experiment_report.md` | hand-written run-level stats; coordinator gets these as seed context. |

Linker: transcript JSONL has `cwd` ending in `/conv-XX` (= sample_id) and a first user message containing the question text. Joining transcript ↔ qa_results by (sample_id, normalized question) is sufficient.

### Failure-mode buckets (already observed in r6)

| Bucket | Count | Hypothesis |
|---|---|---|
| `turns=1, mcp=0`, WRONG | 133 | injected hook content was wrong/insufficient; model committed too early without checking MCP. |
| `turns=4-7, mcp=0`, WRONG | 57 | model thrashed inside hook content, never escalated to MCP. |
| `turns=2-3, mcp=0`, WRONG | 44 | |
| `turns=8+, mcp=0`, WRONG | 37 | severe thrash; cost outliers ($0.5-0.7) live here; model often gives "based on the clues" guess instead of fetching. |

Concrete examples:
- Date arithmetic off-by-one (Sun-before-25-May → 20 not 21).
- Count missed (2 → "Once").
- Hallucinated event window when actual evidence missing from injected content.

These are **seed patterns** for the coordinator's pattern library so workers don't start blind.

---

## Architecture

```
                 ┌────────────────────────┐
                 │  Coordinator (Opus)    │
                 │  ─ pattern_store.db    │
                 │  ─ reflection loop     │
                 │  ─ escalation handler  │
                 └──────┬───────┬─────────┘
                        ▲       │
            escalations │       │ patterns_v_N (system-prompt fragment)
                        │       ▼
       ┌────────────────┴───────────────────────┐
       │            asyncio queues               │
       │   work_q: session_ids                   │
       │   esc_q:  worker→coordinator questions │
       │   pat_q:  coordinator→workers updates   │
       └─────────────────┬───────────────────────┘
                         ▼
           ┌─────────────┴────────────┐
           │  Worker pool (N=100      │
           │  logical, sem=K real)    │
           │  Haiku 4.5, single-shot  │
           └──────────────────────────┘
```

### Concurrency model

- `Semaphore(K)` — caps in-flight Anthropic calls (K tunable; default 20). 100 logical "workers" is a misread of zayn's spec; what matters is queue depth & throughput, not literal process count. We document this and run K=20 by default; raise if rate limit allows.
- All queues are `asyncio.Queue`; single event loop owns mutation of `pattern_store` (no cross-thread locks needed).
- Worker is **stateless coroutine**: dequeue session → fetch transcript+qa → call Haiku → parse → write result file → optionally enqueue escalation.

### Escalation model — async, not sync

Worker that is low-confidence DOES NOT block waiting for coordinator. Instead:
1. Worker emits result with `confidence < threshold` AND an `escalate_question`.
2. Worker writes a "deferred" marker for that session (so we can re-process after pattern update).
3. Coordinator drains escalation queue every M seconds (or every N escalations); produces a clarification + new pattern entry; bumps `pattern_version`.
4. After bump, deferred sessions are re-enqueued with updated pattern context. Bound by max retries (default 2).

Why async: synchronous round-trip would serialize 100 workers behind 1 opus reflection — kills throughput. Quality cost is small because the SAME pattern likely affects many sessions (one opus reflection benefits the batch).

### Pattern store

SQLite (file `pattern_store.db`). Schema:

```sql
CREATE TABLE patterns (
  id INTEGER PRIMARY KEY,
  signature TEXT UNIQUE,        -- short canonical id like "early-commit-wrong-date"
  category TEXT,                -- "accuracy" | "tokens" | "turns" | "tool_use"
  symptom TEXT,                 -- 1-line description for system prompt
  evidence TEXT,                -- JSON list of (session_id, snippet) examples
  count INTEGER DEFAULT 1,
  first_seen TEXT,
  last_updated TEXT,
  rule_for_workers TEXT         -- short instruction for haiku ("if hook content lacks date, escalate")
);
CREATE TABLE escalations (
  id INTEGER PRIMARY KEY,
  session_id TEXT,
  worker_question TEXT,
  raised_at TEXT,
  resolved_at TEXT,
  resolution TEXT,
  pattern_id INTEGER REFERENCES patterns(id)
);
CREATE TABLE session_results (
  session_id TEXT PRIMARY KEY,
  qa_sample_id TEXT,
  qa_question_index INTEGER,
  classification TEXT,    -- pattern signature(s), JSON
  severity TEXT,          -- "low" | "med" | "high"
  confidence REAL,
  worker_notes TEXT,
  attempt INTEGER,
  pattern_version INTEGER,
  written_at TEXT
);
```

The system-prompt fragment fed to workers is materialized from `patterns` (top K by recent count, summarized to fit a token budget — default 2K tokens). Coordinator reflection rebuilds this fragment on every bump.

### Coordinator reflection loop

Triggered when EITHER condition met:
1. `escalations_unhandled >= 5`
2. `sessions_done_since_last_reflection >= 25`
3. Time-based fallback: every 60s if work in flight.

Reflection prompt (Opus) input:
- current `patterns` table
- new escalations text
- new wrong/anomalous session result snippets (sampled, not all)
Reflection output: JSON ops list — `add_pattern`, `merge_pattern`, `refine_rule`, `dismiss`. Coordinator applies, increments `pattern_version`, broadcasts new system-prompt fragment.

### Worker prompt skeleton

System prompt:
```
You analyze ONE eval session at a time. Output strict JSON.
Run context: <run-level stats>
Known issue patterns (v_<N>):
<bulleted patterns with rule_for_workers>
For each session, classify the failure mode (or success), cite at most 3 evidence snippets,
and rate confidence 0-1. If confidence < 0.6 OR session doesn't fit any known pattern, set
escalate=true and write a clear question to the coordinator.

Output schema:
{
  "session_id": "...",
  "verdict": "correct|wrong",
  "primary_pattern": "<signature or 'unknown'>",
  "secondary_patterns": [],
  "evidence": [{"event_idx": int, "snippet": "..."}],
  "severity": "low|med|high",
  "confidence": 0.0..1.0,
  "escalate": bool,
  "escalate_question": "..." (if escalate)
}
```

User prompt: serialized session payload (compact: question, gold answer, model response, grader reasoning, num_turns, mcp_count, hook_count, ALL assistant text turns concatenated, list of tool_use names + first-arg only).

Token budget per worker call: keep < 8K input tokens (transcripts compress well). Use prompt caching on system prompt + run context.

### Coordinator prompt skeleton

System: "You are the senior analyst. Maintain a stable, deduplicated pattern library. When workers escalate, decide whether the case fits an existing pattern (refine its rule), introduces a new pattern (add), or is noise (dismiss)."

User (per reflection): current patterns + new escalations + sample of recent results.

Output: JSON ops list (add/merge/refine/dismiss).

---

## Runtime backends — pluggable by role

Zayn's desired shape is role-level runtime switching: the worker pool and the
coordinator can be swapped independently across the runtimes exposed by
zouk-daemon. In this repo the orchestrator owns the queue/persistence logic and
calls a thin runtime interface:

```python
await runtime.call(
    model=model,
    system=system_prompt,
    user=session_or_reflection_payload,
    max_tokens=max_tokens,
    role="worker" | "coordinator",
)
```

The daemon registry currently includes `claude`, `codex`, `opencode` plus
other runtimes. `swarm-eval` implements the three requested runtime families
without depending on the daemon process itself:

| Runtime | Driver shape | Tradeoff |
|---|---|---|
| `sdk` | Anthropic Python SDK direct API calls | hard output cap and prompt caching; requires `ANTHROPIC_API_KEY` |
| `claude-code` | `claude -p --model <id>` subprocess | OAuth fallback; mirrors daemon Claude command resolution |
| `codex` | `codex exec` subprocess | OpenAI/Codex runtime comparison; prompt-side cap only |
| `opencode` | `opencode run --format json` subprocess | provider/OpenCode comparison; prompt-side cap only |

Runtime selection:

```bash
python3 src/cli.py runtimes
python3 src/cli.py run --runtime claude-code ...
python3 src/cli.py run --worker-runtime codex --coordinator-runtime claude-code ...
python3 src/cli.py run --worker-runtime opencode --coordinator-runtime opencode ...
python3 src/cli.py run --output-dir runs --run-id calibration-round-1 ...
```

Reference to zouk-daemon:
- `zouk-daemon/src/drivers/index.ts` is the source of available runtime names.
- `zouk-daemon/src/drivers/claude.ts`, `codex.ts`, and `opencode.ts` define the
  CLI/protocol shapes mirrored here.
- Lifecycle pieces (idle cache, busy delivery, MCP chat bridge) are deliberately
  not borrowed; swarm workers are one-shot local analysis calls.

---

## File layout

```
swarm-eval/
├── docs/DESIGN.md                  (this)
├── src/
│   ├── linker.py                   transcripts ↔ qa_results
│   ├── pattern_store.py            SQLite CRUD + system-prompt builder
│   ├── prompts/
│   │   ├── worker.md               system prompt template
│   │   └── coordinator.md
│   ├── backends/
│   │   ├── base.py
│   │   ├── factory.py
│   │   ├── cli.py
│   │   ├── codex.py
│   │   ├── opencode.py
│   │   └── sdk.py
│   ├── coordinator.py              reflection loop
│   ├── worker.py                   one-session analysis
│   ├── orchestrator.py             queues, semaphore, retry, main
│   └── cli.py                      `python -m swarm_eval.cli run --result-dir ... --limit 10`
├── runs/<run-id>/                  per-run logs, pattern_store.db, results.jsonl
└── data/                           cached parsed transcripts (optional)
```

---

## Sample-batch protocol

1. Run `--limit 10` first. Inspect: did each worker output valid JSON? did coordinator generate ≥1 pattern? did escalation round-trip work?
2. `--limit 100` if 10 looked OK.
3. Hand to @tim with: results.jsonl summary, pattern_store dump, run log, cost breakdown.
4. Iterate prompts based on tim feedback until accept.
5. Full 1540 run.
6. @hela final review of methodology + output quality.

## Open design questions for @tim

- (a) Worker talk-back async vs sync — design proposes async (broadcast model) to avoid serializing 100 workers behind 1 reflection. Confirm.
- (b) Pattern library size budget — 2K token system-prompt fragment, top-K recent patterns. Acceptable?
- (c) Re-process attempt cap — 2. Acceptable?
- (d) Concurrency cap K — start at 20. Adjust based on rate limit.
- (e) Reflection trigger — escalation queue OR session count OR time. Confirm thresholds.

## Open ops questions for @zaynjarvis

- (1) ANTHROPIC_API_KEY available? Otherwise default to `cli` backend.
- (2) Worker uses `claude-haiku-4-5` correct? Coordinator `claude-opus-4-7`?
- (3) zouk-daemon "reference" — borrow spawn pattern only (current plan), or actually plug into daemon?
