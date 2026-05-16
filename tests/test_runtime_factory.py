from __future__ import annotations

import argparse
import builtins
import os
import sys
import unittest
from unittest import mock


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from backends.codex import format_agent_prompt, parse_codex_usage  # noqa: E402
from backends.factory import build_backend, default_model_for_runtime, normalize_runtime  # noqa: E402
from backends.opencode import parse_opencode_output  # noqa: E402
from cli import _resolve_run_output_dir, _resolve_runtime_args  # noqa: E402


class RuntimeFactoryTests(unittest.TestCase):
    def test_runtime_aliases_and_defaults(self) -> None:
        self.assertEqual(normalize_runtime("cli"), "claude-code")
        self.assertEqual(normalize_runtime("claude"), "claude-code")
        self.assertEqual(default_model_for_runtime("claude-code", "worker"), "claude-haiku-4-5-20251001")
        self.assertEqual(default_model_for_runtime("codex", "worker"), "gpt-5.5")
        self.assertEqual(default_model_for_runtime("codex", "coordinator"), "gpt-5.5")

    def test_runtime_args_allow_split_worker_and_coordinator(self) -> None:
        args = argparse.Namespace(
            backend=None,
            runtime="claude",
            worker_runtime="codex",
            coordinator_runtime=None,
            worker_model=None,
            coordinator_model=None,
        )
        worker_runtime, coordinator_runtime, worker_model, coordinator_model = _resolve_runtime_args(args)
        self.assertEqual(worker_runtime, "codex")
        self.assertEqual(coordinator_runtime, "claude-code")
        self.assertEqual(worker_model, "gpt-5.5")
        self.assertEqual(coordinator_model, "claude-opus-4-7")

    def test_legacy_backend_conflicts_with_runtime_flags(self) -> None:
        args = argparse.Namespace(
            backend="cli",
            runtime="codex",
            worker_runtime=None,
            coordinator_runtime=None,
            worker_model=None,
            coordinator_model=None,
        )
        with self.assertRaisesRegex(ValueError, "legacy --backend"):
            _resolve_runtime_args(args)

    def test_invalid_runtime_name_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown runtime: bogus"):
            build_backend("bogus")

    def test_sdk_missing_dependency_error_is_friendly(self) -> None:
        real_import = builtins.__import__

        def fake_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "anthropic":
                raise ImportError("missing anthropic")
            return real_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=fake_import):
            with self.assertRaisesRegex(RuntimeError, "sdk runtime unavailable.*anthropic"):
                build_backend("sdk")

    def test_output_dir_is_not_machine_hardcoded(self) -> None:
        self.assertEqual(
            _resolve_run_output_dir("runs", "abc"),
            os.path.abspath(os.path.join("runs", "abc")),
        )


class RuntimeParserTests(unittest.TestCase):
    def test_codex_prompt_wraps_system_and_user(self) -> None:
        prompt = format_agent_prompt(
            role="worker",
            system="SYSTEM RULE",
            user="USER PAYLOAD",
            max_tokens=700,
        )
        self.assertIn("swarm-eval worker runtime", prompt)
        self.assertIn("SYSTEM RULE", prompt)
        self.assertIn("USER PAYLOAD", prompt)
        self.assertIn("700 output tokens", prompt)

    def test_codex_usage_parser_handles_total_only_stderr(self) -> None:
        stderr = "OpenAI Codex v0.130.0\nmodel: gpt-5.5\n\ntokens used\n11,756\n"
        self.assertEqual(parse_codex_usage(stderr)["total_tokens"], 11756)

    def test_opencode_parser_handles_acp_chunks(self) -> None:
        blob = "\n".join([
            '{"params":{"update":{"sessionUpdate":"agent_message_chunk","content":{"type":"text","text":"{\\"a\\":"}}}}',
            '{"params":{"update":{"sessionUpdate":"agent_message_chunk","content":{"type":"text","text":"1}"}}}}',
        ])
        self.assertEqual(parse_opencode_output(blob), '{"a":1}')


if __name__ == "__main__":
    unittest.main()
