"""Produce a markdown digest of a swarm-eval run for human review."""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys


SEED_SIGNATURES = {
    "early-commit-wrong-date",
    "hook-thrash-no-mcp",
    "missing-detail-generic-answer",
    "hallucinated-event-window",
    "off-by-count",
    "expensive-correct-mcp-thrash",
    "redundant-mcp-after-sufficient-hook",
    "high-output-overexplanation",
    "metric-mcp-count-mismatch",
    "truncated-context-extra-search",
}


def _edit_distance_le1(a: str, b: str) -> bool:
    """Return True iff Damerau-Levenshtein distance(a, b) <= 1.

    Counts a single insert / delete / substitute / adjacent-transpose as 1 edit.
    Catches common typos like `mcp`↔`mpc` that pure Levenshtein scores as 2.
    """
    if a == b:
        return True
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    if la == lb:
        # find first mismatch
        i = 0
        while i < la and a[i] == b[i]:
            i += 1
        if i == la:
            return True
        # try substitution: rest must match
        if a[i + 1 :] == b[i + 1 :]:
            return True
        # try adjacent transposition: a[i]a[i+1] swapped with b
        if (
            i + 1 < la
            and a[i] == b[i + 1]
            and a[i + 1] == b[i]
            and a[i + 2 :] == b[i + 2 :]
        ):
            return True
        return False
    # length differs by 1: must be one insert/delete
    short, long = (a, b) if la < lb else (b, a)
    i = j = 0
    found_gap = False
    while i < len(short) and j < len(long):
        if short[i] == long[j]:
            i += 1
            j += 1
            continue
        if found_gap:
            return False
        found_gap = True
        j += 1
    return True


def _compute_signature_aliases(
    db: sqlite3.Connection, prefix_len: int = 6
) -> tuple[dict[str, str], list[tuple[str, str]]]:
    """Build a typo-alias map: raw_sig -> canonical_sig.

    Rule: two signatures alias iff they share the first `prefix_len` chars AND
    Levenshtein distance <=1 (lowercase). The canonical is the one present in
    the `patterns` table; if both/neither, the higher-count one wins; ties
    broken by longer-string-first then lexicographic.

    Returns (alias_map, alias_pairs) where alias_pairs is the list of
    (raw, canonical) edges actually applied (for digest disclosure).
    """
    cur = db.cursor()
    cur.execute("SELECT signature, count FROM patterns")
    pattern_rows = {r["signature"]: r["count"] for r in cur.fetchall()}
    cur.execute(
        "SELECT primary_pattern, COUNT(*) c FROM session_results "
        "WHERE primary_pattern IS NOT NULL AND primary_pattern != 'unknown' "
        "GROUP BY primary_pattern"
    )
    sr_rows = {r["primary_pattern"]: r["c"] for r in cur.fetchall()}

    all_sigs = set(pattern_rows.keys()) | set(sr_rows.keys())
    sigs = sorted(all_sigs)

    def score(sig: str) -> tuple[int, int, int, str]:
        in_patterns = 1 if sig in pattern_rows else 0
        cnt = max(pattern_rows.get(sig, 0), sr_rows.get(sig, 0))
        return (in_patterns, cnt, len(sig), sig)

    alias: dict[str, str] = {}
    edges: list[tuple[str, str]] = []
    for i, a in enumerate(sigs):
        for b in sigs[i + 1 :]:
            if a[:prefix_len].lower() != b[:prefix_len].lower():
                continue
            if not _edit_distance_le1(a.lower(), b.lower()):
                continue
            sa, sb = score(a), score(b)
            winner, loser = (a, b) if sa >= sb else (b, a)
            cur_canonical = alias.get(winner, winner)
            alias[loser] = cur_canonical
            edges.append((loser, cur_canonical))
            for k, v in list(alias.items()):
                if v == loser:
                    alias[k] = cur_canonical
    return alias, edges


def _canon(sig: str | None, alias: dict[str, str]) -> str | None:
    if sig is None:
        return None
    return alias.get(sig, sig)


def _load_log_meta(log_path: str) -> dict[str, dict]:
    meta: dict[str, dict] = {}
    if not os.path.isfile(log_path):
        return meta
    with open(log_path) as f:
        for line in f:
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            if o.get("kind") not in ("worker_outcome", "sweep_outcome"):
                continue
            sid = o["session_id"]
            meta[sid] = {
                "truth": (o.get("verdict_truth") or "?").upper(),
                "sample_id": o.get("sample_id"),
                "qix": o.get("qix"),
                "num_turns": o.get("num_turns"),
                "ov_mcp_calls": o.get("ov_mcp_calls"),
                "transcript_mcp_calls": o.get("transcript_mcp_calls"),
                "session_cost_usd": float(o.get("session_cost_usd", 0.0) or 0.0),
                "session_total_tokens": int(o.get("session_total_tokens", 0) or 0),
                "session_input_tokens": int(o.get("session_input_tokens", 0) or 0),
                "session_cache_read_tokens": int(o.get("session_cache_read_tokens", 0) or 0),
                "session_output_tokens": int(o.get("session_output_tokens", 0) or 0),
            }
    return meta


def summarize(run_dir: str) -> str:
    db = sqlite3.connect(os.path.join(run_dir, "pattern_store.db"))
    db.row_factory = sqlite3.Row
    log_path = os.path.join(run_dir, "results.jsonl")
    log_meta = _load_log_meta(log_path)

    out: list[str] = []
    out.append(f"# Run digest — {os.path.basename(run_dir)}\n")

    # ---- raw counters
    cur = db.cursor()
    cur.execute("SELECT COUNT(*) c FROM session_results")
    sr_count = cur.fetchone()["c"]
    cur.execute(
        "SELECT verdict, COUNT(*) c FROM session_results GROUP BY verdict"
    )
    verdict_dist = {r["verdict"]: r["c"] for r in cur.fetchall()}
    cur.execute("SELECT COUNT(*) c FROM patterns")
    pat_count = cur.fetchone()["c"]
    cur.execute("SELECT COUNT(*) c FROM escalations")
    esc_total = cur.fetchone()["c"]
    cur.execute("SELECT COUNT(*) c FROM escalations WHERE resolved_at IS NULL")
    esc_open = cur.fetchone()["c"]

    out.append(f"- sessions analyzed: **{sr_count}** (verdicts: {verdict_dist})")
    out.append(f"- patterns in library: **{pat_count}**")
    out.append(f"- escalations: {esc_total} total ({esc_open} unresolved)")

    # ---- token / cost from log
    tot_in = tot_out = tot_cache = 0
    worker_runs = reflections = 0
    parse_errors = 0
    pv_max = 0
    if os.path.isfile(log_path):
        with open(log_path) as f:
            for line in f:
                try:
                    o = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if o["kind"] == "worker_outcome":
                    worker_runs += 1
                    if not o.get("ok"):
                        parse_errors += 1
                    pv_max = max(pv_max, int(o.get("pattern_version", 0) or 0))
                elif "reflection" in o["kind"]:
                    reflections += 1
                tot_in += int(o.get("tokens_in", 0) or 0)
                tot_cache += int(o.get("tokens_cache", 0) or 0)
                tot_out += int(o.get("tokens_out", 0) or 0)
    out.append(
        f"- worker calls: {worker_runs} (parse errors: {parse_errors}); reflections: {reflections}; max pattern_version: {pv_max}"
    )
    out.append(
        f"- tokens (input / cache_read / output): "
        f"{tot_in:,} / {tot_cache:,} / {tot_out:,}"
    )

    # ---- post-hoc signature normalization (typo alias map)
    alias, alias_edges = _compute_signature_aliases(db)
    if alias_edges:
        out.append("\n## Signature normalization (post-hoc)\n")
        out.append(
            "The following typo-aliased signatures were canonicalized for display "
            "(lowercase + Levenshtein ≤ 1 + shared 6-char prefix). The pattern_store.db "
            "is unchanged; this only affects digest counts and per-session classification."
        )
        out.append("\n| raw | canonical |")
        out.append("|---|---|")
        seen: set[tuple[str, str]] = set()
        for raw, canon in alias_edges:
            key = (raw, canon)
            if key in seen:
                continue
            seen.add(key)
            out.append(f"| `{raw}` | `{canon}` |")

    # ---- seed vs final diff
    out.append("\n## Seed → final pattern diff\n")
    raw_final_sigs = {r["signature"] for r in db.execute("SELECT signature FROM patterns").fetchall()}
    final_sigs = {_canon(s, alias) for s in raw_final_sigs}
    seed_dismissed = SEED_SIGNATURES - final_sigs
    seed_kept = SEED_SIGNATURES & final_sigs
    new_added = final_sigs - SEED_SIGNATURES
    out.append(f"- seed kept (still in library): {len(seed_kept)} — {sorted(seed_kept) or '—'}")
    out.append(f"- seed dismissed/merged out: {len(seed_dismissed)} — {sorted(seed_dismissed) or '—'}")
    out.append(f"- new patterns added by coordinator: {len(new_added)} — {sorted(new_added) or '—'}")

    # which seeds got actual evidence vs are still empty?
    out.append("\n### Seed evidence accumulation")
    out.append("| seed signature | count | rule mutated? |")
    out.append("|---|---|---|")
    for sig in sorted(SEED_SIGNATURES):
        row = db.execute(
            "SELECT count, rule_for_workers FROM patterns WHERE signature=?", (sig,)
        ).fetchone()
        if not row:
            out.append(f"| `{sig}` | (dismissed) | n/a |")
        else:
            out.append(f"| `{sig}` | {row['count']} | (compare with seed if needed) |")

    # ---- top patterns (with alias rollup)
    # Compute extra count per canonical from aliased session_results that aren't in patterns.
    extra_by_canon: dict[str, int] = {}
    cur_extra = db.execute(
        "SELECT primary_pattern, COUNT(*) c FROM session_results "
        "WHERE primary_pattern IS NOT NULL AND primary_pattern != 'unknown' "
        "GROUP BY primary_pattern"
    )
    pattern_signatures_in_db = {
        r["signature"] for r in db.execute("SELECT signature FROM patterns").fetchall()
    }
    for r in cur_extra.fetchall():
        raw = r["primary_pattern"]
        canon = _canon(raw, alias) or raw
        if raw == canon:
            continue
        if canon in pattern_signatures_in_db:
            extra_by_canon[canon] = extra_by_canon.get(canon, 0) + r["c"]

    out.append("\n## Patterns (top by count)\n")
    rows = db.execute(
        "SELECT signature, category, count, symptom, rule_for_workers, evidence "
        "FROM patterns ORDER BY count DESC, signature"
    ).fetchall()
    for r in rows:
        sig = r["signature"]
        canon_sig = _canon(sig, alias) or sig
        if sig != canon_sig:
            # this row is a typo-alias of another canonical row; skip — its
            # count is rolled into the canonical via extra_by_canon.
            continue
        ev = json.loads(r["evidence"]) if r["evidence"] else []
        def _fmt_ev(e: dict) -> str:
            if e.get("sample_id") and e.get("qix") is not None:
                return f"{e['sample_id']}/q{e['qix']}"
            sid = e.get("session_id") or ""
            return sid[:8] if sid else "?"
        ev_short = ", ".join(_fmt_ev(e) for e in ev[:3]) or "—"
        eff = r["count"] + extra_by_canon.get(sig, 0)
        eff_note = f" (+{extra_by_canon[sig]} aliased)" if extra_by_canon.get(sig) else ""
        out.append(
            f"### `{sig}` ({r['category']}, n={eff}{eff_note})\n"
            f"- symptom: {r['symptom']}\n"
            f"- rule: {r['rule_for_workers']}\n"
            f"- examples: {ev_short}\n"
        )

    # ---- split the core questions: failure patterns vs expensive-correct patterns
    pattern_categories = {
        r["signature"]: r["category"]
        for r in db.execute("SELECT signature, category FROM patterns").fetchall()
    }
    session_rows = db.execute(
        "SELECT session_id, verdict, primary_pattern, severity, confidence, worker_notes "
        "FROM session_results"
    ).fetchall()

    out.append("\n## WRONG failure patterns\n")
    wrong_counts: dict[str, dict] = {}
    for r in session_rows:
        meta = log_meta.get(r["session_id"], {})
        if meta.get("truth") != "WRONG":
            continue
        sig = _canon(r["primary_pattern"], alias) or r["primary_pattern"] or "unknown"
        bucket = wrong_counts.setdefault(sig, {"count": 0, "examples": []})
        bucket["count"] += 1
        if len(bucket["examples"]) < 3:
            bucket["examples"].append(
                f"{meta.get('sample_id','?')}/q{meta.get('qix','?')}:{r['session_id'][:8]}"
            )
    if not wrong_counts:
        out.append("(none)")
    else:
        out.append("| pattern | count | examples |")
        out.append("|---|---:|---|")
        for sig, v in sorted(wrong_counts.items(), key=lambda x: (-x[1]["count"], x[0])):
            out.append(f"| `{sig}` | {v['count']} | {', '.join(v['examples'])} |")

    out.append("\n## CORRECT high-cost/token patterns\n")
    correct_counts: dict[str, dict] = {}
    top_correct: list[tuple[float, sqlite3.Row, dict, str]] = []
    for r in session_rows:
        meta = log_meta.get(r["session_id"], {})
        if meta.get("truth") != "CORRECT":
            continue
        sig = _canon(r["primary_pattern"], alias) or r["primary_pattern"] or "unknown"
        category = pattern_categories.get(sig)
        if sig != "unknown" and category in {"tokens", "turns", "tool_use"}:
            bucket = correct_counts.setdefault(sig, {"count": 0, "examples": []})
            bucket["count"] += 1
            if len(bucket["examples"]) < 3:
                bucket["examples"].append(
                    f"{meta.get('sample_id','?')}/q{meta.get('qix','?')}:{r['session_id'][:8]}"
                )
        top_correct.append((float(meta.get("session_cost_usd", 0.0) or 0.0), r, meta, sig))
    if not correct_counts:
        out.append("(no non-unknown token/turn/tool patterns on CORRECT sessions)")
    else:
        out.append("| pattern | count | examples |")
        out.append("|---|---:|---|")
        for sig, v in sorted(correct_counts.items(), key=lambda x: (-x[1]["count"], x[0])):
            out.append(f"| `{sig}` | {v['count']} | {', '.join(v['examples'])} |")

    out.append("\n### Top expensive CORRECT sessions")
    top_correct.sort(key=lambda x: x[0], reverse=True)
    top_correct = [x for x in top_correct if x[0] > 0][:10]
    if not top_correct:
        out.append("(cost metadata unavailable)")
    else:
        out.append("| session | sample/qix | cost | total_tokens | turns | mcp | pattern | note |")
        out.append("|---|---|---:|---:|---:|---:|---|---|")
        for cost, r, meta, sig in top_correct:
            mcp = max(int(meta.get("ov_mcp_calls") or 0), int(meta.get("transcript_mcp_calls") or 0))
            note = (r["worker_notes"] or "").replace("|", "\\|")[:120]
            out.append(
                f"| {r['session_id'][:8]} | {meta.get('sample_id','?')}/q{meta.get('qix','?')} | "
                f"${cost:.4f} | {int(meta.get('session_total_tokens') or 0):,} | "
                f"{meta.get('num_turns') or 0} | {mcp} | `{sig}` | {note} |"
            )

    # ---- escalations & resolutions
    out.append("\n## Escalations\n")
    rows = db.execute(
        "SELECT id, session_id, worker_question, resolved_at, resolution, pattern_id "
        "FROM escalations ORDER BY id"
    ).fetchall()
    if not rows:
        out.append("(none)")
    for r in rows:
        sig = "—"
        if r["pattern_id"]:
            row = db.execute("SELECT signature FROM patterns WHERE id=?", (r["pattern_id"],)).fetchone()
            if row:
                sig = row["signature"]
        out.append(
            f"### esc#{r['id']} session={r['session_id'][:8]} → `{sig}`\n"
            f"- Q: {r['worker_question']}\n"
            f"- R: {r['resolution'] or '(unresolved)'}\n"
        )

    # ---- agreement with grader
    # ---- coverage: WRONG sessions still classified 'unknown' (per tim's ask)
    out.append("\n## Coverage — WRONG cases by classification\n")
    truth_per_log: dict[str, dict] = {}
    if os.path.isfile(log_path):
        with open(log_path) as f:
            for line in f:
                try:
                    o = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if o.get("kind") in ("worker_outcome", "sweep_outcome"):
                    sid = o["session_id"]
                    truth_per_log[sid] = {
                        "truth": (o.get("verdict_truth") or "?").upper(),
                        "num_turns": o.get("num_turns"),
                        "ov_mcp_calls": o.get("ov_mcp_calls"),
                    }
    cur2 = db.execute("SELECT session_id, primary_pattern FROM session_results")
    by_truth_bucket: dict[tuple, dict[str, int]] = {}
    wrong_unknown = 0
    wrong_total = 0
    for r in cur2.fetchall():
        meta = truth_per_log.get(r["session_id"])
        if not meta or meta["truth"] != "WRONG":
            continue
        wrong_total += 1
        nt = meta.get("num_turns") or 0
        mcp = meta.get("ov_mcp_calls") or 0
        if nt == 1:
            tb = "turns=1"
        elif nt <= 3:
            tb = "turns=2-3"
        elif nt <= 7:
            tb = "turns=4-7"
        else:
            tb = "turns=8+"
        mb = "mcp=0" if mcp == 0 else ("mcp=1-3" if mcp <= 3 else "mcp=4+")
        key = (tb, mb)
        bucket = by_truth_bucket.setdefault(key, {"total": 0, "unknown": 0})
        bucket["total"] += 1
        canon_pat = _canon(r["primary_pattern"], alias) or "unknown"
        if canon_pat == "unknown":
            bucket["unknown"] += 1
            wrong_unknown += 1
    out.append(f"- WRONG sessions: {wrong_total} • still 'unknown' after sweep: {wrong_unknown} "
               f"({(wrong_unknown / wrong_total * 100) if wrong_total else 0:.1f}%)")
    out.append("\n| bucket | wrong | unknown | unknown% |")
    out.append("|---|---|---|---|")
    for (tb, mb), v in sorted(by_truth_bucket.items(), key=lambda x: -x[1]["total"]):
        pct = (v["unknown"] / v["total"] * 100) if v["total"] else 0
        out.append(f"| {tb} / {mb} | {v['total']} | {v['unknown']} | {pct:.1f}% |")

    out.append("\n## Worker vs grader agreement\n")
    rows = db.execute(
        "SELECT session_id, verdict, primary_pattern, severity, confidence "
        "FROM session_results"
    ).fetchall()
    # join with link from log truth
    truth: dict[str, str] = {}
    if os.path.isfile(log_path):
        with open(log_path) as f:
            for line in f:
                try:
                    o = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if o.get("kind") == "worker_outcome":
                    sid = o["session_id"]
                    truth[sid] = o.get("verdict_truth", "?")
    correct_agree = wrong_agree = mismatch = 0
    rows_for_show: list = []
    for r in rows:
        truth_v = truth.get(r["session_id"], "?").lower()
        wv = r["verdict"]
        if truth_v == wv:
            if wv == "correct":
                correct_agree += 1
            else:
                wrong_agree += 1
        else:
            mismatch += 1
        rows_for_show.append((r, truth_v))
    out.append(
        f"- agree on CORRECT: {correct_agree} • agree on WRONG: {wrong_agree} • mismatch: {mismatch}"
    )

    out.append("\n## Per-session table\n")
    out.append("| session | sample/qix | grader | worker | pattern | sev | conf |")
    out.append("|---|---|---|---|---|---|---|")
    for r, truth_v in rows_for_show:
        # find sample/qix from log (cheap re-read OK at this scale)
        sid = r["session_id"]
        meta = ""
        with open(log_path) as f:
            for line in f:
                o = json.loads(line)
                if o.get("kind") == "worker_outcome" and o["session_id"] == sid:
                    meta = f"{o['sample_id']}/q{o['qix']}"
                    break
        canon_pat = _canon(r["primary_pattern"], alias) or r["primary_pattern"]
        pat_disp = (
            f"`{canon_pat}` ←`{r['primary_pattern']}`"
            if canon_pat != r["primary_pattern"]
            else f"`{r['primary_pattern']}`"
        )
        out.append(
            f"| {sid[:8]} | {meta} | {truth_v} | {r['verdict']} | "
            f"{pat_disp} | {r['severity']} | {r['confidence']:.2f} |"
        )

    return "\n".join(out)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    md = summarize(args.run_dir)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"wrote {args.out}")
    else:
        print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
