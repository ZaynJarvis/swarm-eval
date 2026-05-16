from __future__ import annotations

import csv
import json
import os
import sys
import tempfile
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from linker import iter_linked  # noqa: E402


def write_transcript(path: str, *, sample_id: str, question: str, answer_text: str) -> None:
    events = [
        {
            "type": "user",
            "cwd": f"/tmp/{sample_id}",
            "message": {"content": f"Answer the question directly: {question}"},
        },
        {
            "type": "assistant",
            "message": {"content": [{"type": "text", "text": answer_text}]},
        },
    ]
    with open(path, "w", encoding="utf-8") as f:
        for event in events:
            f.write(json.dumps(event) + "\n")


class LinkerTests(unittest.TestCase):
    def test_duplicate_transcripts_for_same_qa_row_choose_response_match(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            os.makedirs(os.path.join(td, "transcripts"))
            with open(os.path.join(td, "qa_results.csv"), "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "sample_id",
                        "question_index",
                        "question",
                        "answer",
                        "category",
                        "response",
                        "reasoning",
                        "num_turns",
                        "elapsed_seconds",
                        "total_cost_usd",
                        "input_tokens",
                        "output_tokens",
                        "cache_read_input_tokens",
                        "ov_recall_hooks",
                        "ov_mcp_calls",
                        "result",
                    ],
                )
                writer.writeheader()
                writer.writerow({
                    "sample_id": "conv-test",
                    "question_index": "7",
                    "question": "What happened?",
                    "answer": "right answer",
                    "category": "test",
                    "response": "right answer",
                    "reasoning": "ok",
                    "num_turns": "1",
                    "elapsed_seconds": "1.0",
                    "total_cost_usd": "0",
                    "input_tokens": "0",
                    "output_tokens": "0",
                    "cache_read_input_tokens": "0",
                    "ov_recall_hooks": "0",
                    "ov_mcp_calls": "0",
                    "result": "CORRECT",
                })

            write_transcript(
                os.path.join(td, "transcripts", "aaa-wrong.jsonl"),
                sample_id="conv-test",
                question="What happened?",
                answer_text="wrong answer",
            )
            write_transcript(
                os.path.join(td, "transcripts", "zzz-right.jsonl"),
                sample_id="conv-test",
                question="What happened?",
                answer_text="right answer",
            )

            linked = list(iter_linked(td))
            self.assertEqual(len(linked), 1)
            self.assertEqual(linked[0].session_id, "zzz-right")


if __name__ == "__main__":
    unittest.main()
