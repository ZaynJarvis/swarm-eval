from __future__ import annotations

import os
import sys
import tempfile
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from linker import LinkedSession
from orchestrator import (  # noqa: E402
    RunConfig,
    _CircuitBreaker,
    _RunState,
    _apply_scope,
    _partition_resume_sessions,
    _select_sweep_targets,
)
from pattern_store import PatternStore  # noqa: E402


class DummyBackend:
    name = "dummy"


def make_session(
    session_id: str,
    *,
    result: str = "CORRECT",
    num_turns: int = 1,
    ov_mcp_calls: int = 0,
    transcript_mcp_calls: int = 0,
) -> LinkedSession:
    return LinkedSession(
        session_id=session_id,
        sample_id="conv-test",
        question_index=0,
        question="question?",
        gold_answer="gold",
        category="test",
        response="response",
        result=result,
        grader_reasoning="reasoning",
        num_turns=num_turns,
        elapsed_seconds=1.0,
        total_cost_usd=0.0,
        input_tokens=0,
        output_tokens=0,
        cache_read_input_tokens=0,
        ov_recall_hooks=0,
        ov_mcp_calls=ov_mcp_calls,
        transcript_path="/tmp/nonexistent.jsonl",
        transcript_mcp_calls=transcript_mcp_calls,
    )


def make_state(tmpdir: str, *, final_sweep_low_conf: float = 0.6) -> _RunState:
    cfg = RunConfig(
        result_dir=tmpdir,
        run_id="test",
        out_dir=tmpdir,
        worker_backend=DummyBackend(),
        coordinator_backend=DummyBackend(),
        final_sweep_low_conf=final_sweep_low_conf,
        final_sweep_max_sessions=20,
    )
    return _RunState(
        cfg=cfg,
        store=PatternStore(os.path.join(tmpdir, "pattern_store.db")),
        log_path=os.path.join(tmpdir, "results.jsonl"),
    )


class ResumeSweepTests(unittest.TestCase):
    def test_resume_keeps_low_conf_and_wrong_unknown_visible_to_sweep(self) -> None:
        sessions = [
            make_session("already-ok"),
            make_session("already-unknown"),
            make_session("already-low"),
            make_session("never-ran"),
        ]
        already = {"already-ok", "already-unknown", "already-low"}
        to_process, sweep_pool = _partition_resume_sessions(sessions, already)

        self.assertEqual([s.session_id for s in to_process], ["never-ran"])
        self.assertEqual([s.session_id for s in sweep_pool], [s.session_id for s in sessions])

        with tempfile.TemporaryDirectory() as td:
            state = make_state(td)
            state.store.write_session_result({
                "session_id": "already-ok",
                "qa_sample_id": "conv-test",
                "qa_question_index": 0,
                "verdict": "correct",
                "primary_pattern": "known-good",
                "secondary_patterns": [],
                "evidence": [],
                "severity": "low",
                "confidence": 0.95,
                "worker_notes": "stable",
                "attempt": 1,
                "pattern_version": 1,
            })
            state.store.write_session_result({
                "session_id": "already-unknown",
                "qa_sample_id": "conv-test",
                "qa_question_index": 0,
                "verdict": "correct",
                "primary_pattern": "unknown",
                "secondary_patterns": [],
                "evidence": [],
                "severity": "low",
                "confidence": 0.95,
                "worker_notes": "needs final library",
                "attempt": 1,
                "pattern_version": 1,
            })
            state.store.write_session_result({
                "session_id": "wrong-unknown",
                "qa_sample_id": "conv-test",
                "qa_question_index": 0,
                "verdict": "wrong",
                "primary_pattern": "unknown",
                "secondary_patterns": [],
                "evidence": [],
                "severity": "med",
                "confidence": 0.95,
                "worker_notes": "wrong unknown needs final library",
                "attempt": 1,
                "pattern_version": 1,
            })
            state.store.write_session_result({
                "session_id": "already-low",
                "qa_sample_id": "conv-test",
                "qa_question_index": 0,
                "verdict": "wrong",
                "primary_pattern": "some-pattern",
                "secondary_patterns": [],
                "evidence": [],
                "severity": "med",
                "confidence": 0.40,
                "worker_notes": "needs retry",
                "attempt": 1,
                "pattern_version": 1,
            })

            targets = _select_sweep_targets(state, sweep_pool)
            self.assertEqual(
                [s.session_id for s in targets],
                ["already-low", "never-ran"],
            )

            wrong_pool = sweep_pool + [make_session("wrong-unknown", result="WRONG")]
            targets = _select_sweep_targets(state, wrong_pool)
            self.assertEqual(
                [s.session_id for s in targets],
                ["already-low", "never-ran", "wrong-unknown"],
            )


class CircuitBreakerTests(unittest.TestCase):
    def test_consecutive_failures_trip_breaker(self) -> None:
        breaker = _CircuitBreaker(
            consecutive_limit=3,
            rolling_window=10,
            rolling_failure_rate=0.5,
        )
        self.assertIsNone(breaker.record(ok=False, label="worker"))
        self.assertIsNone(breaker.record(ok=False, label="worker"))
        reason = breaker.record(ok=False, label="worker")
        self.assertEqual(reason, "worker: 3 consecutive backend failures")

    def test_rolling_failure_rate_trips_breaker(self) -> None:
        breaker = _CircuitBreaker(
            consecutive_limit=10,
            rolling_window=4,
            rolling_failure_rate=0.5,
        )
        for ok in (False, True, False):
            self.assertIsNone(breaker.record(ok=ok, label="reflection"))
        reason = breaker.record(ok=False, label="reflection")
        self.assertEqual(reason, "reflection: rolling backend failure rate 75% (3/4)")


class ScopeTests(unittest.TestCase):
    def test_wrong_plus_correct_calibration_samples_each_non_empty_bucket(self) -> None:
        sessions = [
            make_session("w1", result="WRONG", num_turns=1, ov_mcp_calls=0),
            make_session("w2", result="WRONG", num_turns=8, ov_mcp_calls=2),
            make_session("c-1-0-a", result="CORRECT", num_turns=1, ov_mcp_calls=0),
            make_session("c-1-0-b", result="CORRECT", num_turns=1, ov_mcp_calls=0),
            make_session("c-1-mcp", result="CORRECT", num_turns=1, transcript_mcp_calls=1),
            make_session("c-23-0", result="CORRECT", num_turns=2, ov_mcp_calls=0),
            make_session("c-47-mcp", result="CORRECT", num_turns=5, ov_mcp_calls=4),
        ]
        cfg = RunConfig(
            result_dir="/tmp",
            run_id="scope",
            out_dir="/tmp",
            worker_backend=DummyBackend(),
            coordinator_backend=DummyBackend(),
            scope="wrong-plus-correct-calibration",
            correct_sample_per_bucket=1,
        )
        selected = _apply_scope(sessions, cfg)
        self.assertEqual(
            [s.session_id for s in selected],
            ["w1", "w2", "c-1-0-a", "c-1-mcp", "c-23-0", "c-47-mcp"],
        )


if __name__ == "__main__":
    unittest.main()
