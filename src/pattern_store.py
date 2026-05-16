"""SQLite-backed pattern store + system-prompt builder.

Single-writer assumption: only the orchestrator's main coroutine touches the DB.
Worker results go through a queue back to the main loop, which persists them.
This avoids cross-thread locking and keeps SQLite simple.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator


SCHEMA = """
CREATE TABLE IF NOT EXISTS patterns (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  signature TEXT UNIQUE NOT NULL,
  category TEXT NOT NULL,
  symptom TEXT NOT NULL,
  rule_for_workers TEXT NOT NULL,
  evidence TEXT NOT NULL DEFAULT '[]',
  count INTEGER NOT NULL DEFAULT 0,
  first_seen TEXT NOT NULL,
  last_updated TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS escalations (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT NOT NULL,
  worker_question TEXT NOT NULL,
  raised_at TEXT NOT NULL,
  resolved_at TEXT,
  resolution TEXT,
  pattern_id INTEGER REFERENCES patterns(id)
);
CREATE TABLE IF NOT EXISTS session_results (
  session_id TEXT PRIMARY KEY,
  qa_sample_id TEXT NOT NULL,
  qa_question_index INTEGER NOT NULL,
  verdict TEXT NOT NULL,
  primary_pattern TEXT,
  secondary_patterns TEXT NOT NULL DEFAULT '[]',
  evidence TEXT NOT NULL DEFAULT '[]',
  severity TEXT,
  confidence REAL,
  worker_notes TEXT,
  attempt INTEGER NOT NULL DEFAULT 1,
  pattern_version INTEGER NOT NULL DEFAULT 0,
  written_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS run_meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
"""


@dataclass
class Pattern:
    signature: str
    category: str
    symptom: str
    rule_for_workers: str
    count: int = 0
    evidence: list[dict] = field(default_factory=list)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


class PatternStore:
    def __init__(self, db_path: str):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Cursor]:
        cur = self.conn.cursor()
        try:
            yield cur
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise
        finally:
            cur.close()

    # ---- pattern ops --------------------------------------------------------

    def upsert_pattern(self, p: Pattern) -> int:
        with self._tx() as cur:
            cur.execute(
                """
                INSERT INTO patterns(signature, category, symptom, rule_for_workers,
                                     evidence, count, first_seen, last_updated)
                VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(signature) DO UPDATE SET
                  symptom = excluded.symptom,
                  rule_for_workers = excluded.rule_for_workers,
                  category = excluded.category,
                  last_updated = excluded.last_updated
                """,
                (
                    p.signature, p.category, p.symptom, p.rule_for_workers,
                    json.dumps(p.evidence), p.count, _now(), _now(),
                ),
            )
            cur.execute("SELECT id FROM patterns WHERE signature = ?", (p.signature,))
            return cur.fetchone()[0]

    def bump_pattern_count(self, signature: str, evidence_item: dict | None = None) -> None:
        with self._tx() as cur:
            cur.execute("SELECT id, evidence, count FROM patterns WHERE signature=?", (signature,))
            row = cur.fetchone()
            if not row:
                return
            ev = json.loads(row["evidence"]) if row["evidence"] else []
            if evidence_item is not None and len(ev) < 5:
                ev.append(evidence_item)
            cur.execute(
                "UPDATE patterns SET count=count+1, evidence=?, last_updated=? WHERE id=?",
                (json.dumps(ev), _now(), row["id"]),
            )

    def list_patterns(self, top_k: int = 12) -> list[sqlite3.Row]:
        cur = self.conn.cursor()
        cur.execute(
            "SELECT * FROM patterns ORDER BY count DESC, last_updated DESC LIMIT ?",
            (top_k,),
        )
        return cur.fetchall()

    def all_patterns(self) -> list[sqlite3.Row]:
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM patterns ORDER BY count DESC")
        return cur.fetchall()

    def merge_patterns(self, src_signature: str, dst_signature: str) -> None:
        with self._tx() as cur:
            cur.execute("SELECT id, count, evidence FROM patterns WHERE signature=?", (src_signature,))
            src = cur.fetchone()
            cur.execute("SELECT id, count, evidence FROM patterns WHERE signature=?", (dst_signature,))
            dst = cur.fetchone()
            if not src or not dst:
                return
            src_ev = json.loads(src["evidence"]) if src["evidence"] else []
            dst_ev = json.loads(dst["evidence"]) if dst["evidence"] else []
            merged_ev = (dst_ev + src_ev)[:8]
            cur.execute(
                "UPDATE patterns SET count=count+?, evidence=?, last_updated=? WHERE id=?",
                (src["count"], json.dumps(merged_ev), _now(), dst["id"]),
            )
            cur.execute("DELETE FROM patterns WHERE id=?", (src["id"],))

    def dismiss_pattern(self, signature: str) -> None:
        with self._tx() as cur:
            cur.execute("DELETE FROM patterns WHERE signature=?", (signature,))

    # ---- escalation / result ops -------------------------------------------

    def add_escalation(self, session_id: str, question: str) -> int:
        with self._tx() as cur:
            cur.execute(
                "INSERT INTO escalations(session_id, worker_question, raised_at) VALUES(?,?,?)",
                (session_id, question, _now()),
            )
            return cur.lastrowid

    def unresolved_escalations(self) -> list[sqlite3.Row]:
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM escalations WHERE resolved_at IS NULL ORDER BY id")
        return cur.fetchall()

    def resolve_escalation(self, esc_id: int, resolution: str, pattern_id: int | None) -> None:
        with self._tx() as cur:
            cur.execute(
                "UPDATE escalations SET resolved_at=?, resolution=?, pattern_id=? WHERE id=?",
                (_now(), resolution, pattern_id, esc_id),
            )

    def write_session_result(self, sr: dict) -> None:
        with self._tx() as cur:
            cur.execute(
                """
                INSERT INTO session_results(session_id, qa_sample_id, qa_question_index,
                  verdict, primary_pattern, secondary_patterns, evidence, severity,
                  confidence, worker_notes, attempt, pattern_version, written_at)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(session_id) DO UPDATE SET
                  verdict = excluded.verdict,
                  primary_pattern = excluded.primary_pattern,
                  secondary_patterns = excluded.secondary_patterns,
                  evidence = excluded.evidence,
                  severity = excluded.severity,
                  confidence = excluded.confidence,
                  worker_notes = excluded.worker_notes,
                  attempt = excluded.attempt,
                  pattern_version = excluded.pattern_version,
                  written_at = excluded.written_at
                """,
                (
                    sr["session_id"], sr["qa_sample_id"], sr["qa_question_index"],
                    sr["verdict"], sr.get("primary_pattern"),
                    json.dumps(sr.get("secondary_patterns", [])),
                    json.dumps(sr.get("evidence", [])),
                    sr.get("severity"), sr.get("confidence"),
                    sr.get("worker_notes"),
                    sr.get("attempt", 1), sr.get("pattern_version", 0), _now(),
                ),
            )

    def stats(self) -> dict:
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) c FROM session_results")
        done = cur.fetchone()["c"]
        cur.execute("SELECT COUNT(*) c FROM patterns")
        npat = cur.fetchone()["c"]
        cur.execute("SELECT COUNT(*) c FROM escalations WHERE resolved_at IS NULL")
        esc_open = cur.fetchone()["c"]
        cur.execute("SELECT COUNT(*) c FROM escalations")
        esc_total = cur.fetchone()["c"]
        return {"sessions_done": done, "patterns": npat,
                "escalations_open": esc_open, "escalations_total": esc_total}

    def get_meta(self, key: str, default: str = "") -> str:
        cur = self.conn.cursor()
        cur.execute("SELECT value FROM run_meta WHERE key=?", (key,))
        row = cur.fetchone()
        return row["value"] if row else default

    def set_meta(self, key: str, value: str) -> None:
        with self._tx() as cur:
            cur.execute(
                "INSERT INTO run_meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value),
            )

    # ---- system-prompt fragment for workers --------------------------------

    def build_worker_pattern_fragment(self, top_k: int = 12, max_chars: int = 4000) -> str:
        rows = self.list_patterns(top_k=top_k)
        if not rows:
            return "(no known patterns yet — be careful and escalate when uncertain)"
        lines = []
        for r in rows:
            lines.append(
                f"- [{r['signature']}] ({r['category']}, n={r['count']}) "
                f"{r['symptom']}\n  rule: {r['rule_for_workers']}"
            )
        out = "\n".join(lines)
        if len(out) > max_chars:
            out = out[:max_chars] + "\n... [more patterns truncated]"
        return out


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        s = PatternStore(os.path.join(d, "x.db"))
        s.upsert_pattern(Pattern("early-commit-wrong-date", "accuracy",
                                 "Model commits on injected hint without verifying date arithmetic.",
                                 "If question asks for a specific date and hints contain only relative anchors, escalate."))
        s.upsert_pattern(Pattern("hook-thrash", "tokens",
                                 "Model loops on hook content without escalating to MCP.",
                                 "If turns >= 5 and no MCP tool calls, classify as hook-thrash."))
        print(s.build_worker_pattern_fragment())
        print(s.stats())
