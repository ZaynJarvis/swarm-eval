"""CLI entry: python -m src.cli run --result-dir ... --runtime claude-code."""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time

# Allow running as `python src/cli.py` from project root.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backends.factory import (
    RUNTIME_CHOICES,
    build_backend,
    default_model_for_runtime,
    normalize_runtime,
)
from orchestrator import RunConfig, run


def _resolve_runtime_args(args: argparse.Namespace) -> tuple[str, str, str, str]:
    if args.backend and (args.runtime or args.worker_runtime or args.coordinator_runtime):
        raise ValueError("Use either legacy --backend or the runtime flags, not both.")
    base_runtime = args.runtime or args.backend or "sdk"
    worker_runtime = args.worker_runtime or base_runtime
    coordinator_runtime = args.coordinator_runtime or base_runtime
    worker_runtime = normalize_runtime(worker_runtime)
    coordinator_runtime = normalize_runtime(coordinator_runtime)
    worker_model = (
        args.worker_model
        if args.worker_model is not None
        else default_model_for_runtime(worker_runtime, "worker")
    )
    coordinator_model = (
        args.coordinator_model
        if args.coordinator_model is not None
        else default_model_for_runtime(coordinator_runtime, "coordinator")
    )
    return worker_runtime, coordinator_runtime, worker_model, coordinator_model


def _resolve_run_output_dir(output_dir: str, run_id: str) -> str:
    output_base = os.path.abspath(os.path.expanduser(output_dir))
    return os.path.join(output_base, run_id)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run")
    pr.add_argument("--result-dir", required=True)
    pr.add_argument("--limit", type=int, default=None)
    pr.add_argument("--concurrency", type=int, default=4)
    pr.add_argument("--backend", choices=["sdk", "cli"], default=None,
                    help="legacy alias: sets both worker/coordinator runtimes")
    pr.add_argument("--runtime", choices=RUNTIME_CHOICES, default=None,
                    help="sets both worker and coordinator runtimes")
    pr.add_argument("--worker-runtime", choices=RUNTIME_CHOICES, default=None)
    pr.add_argument("--coordinator-runtime", choices=RUNTIME_CHOICES, default=None)
    pr.add_argument("--worker-model", default=None)
    pr.add_argument("--coordinator-model", default=None)
    pr.add_argument("--run-id", default=None)
    pr.add_argument("--output-dir", default="runs",
                    help="base directory for per-run artifacts; run-id is appended")
    pr.add_argument("--reflect-every-n", type=int, default=25)
    pr.add_argument("--worker-max-tokens", type=int, default=700)
    pr.add_argument("--coordinator-max-tokens", type=int, default=4096)
    pr.add_argument("--midrun-reflection", action="store_true",
                    help="enable coordinator reflection while workers are running")
    pr.add_argument("--no-final-reflection", action="store_true",
                    help="skip the single post-run coordinator reflection")
    pr.add_argument("--circuit-breaker-consecutive-failures", type=int, default=5)
    pr.add_argument("--circuit-breaker-rolling-window", type=int, default=20)
    pr.add_argument("--circuit-breaker-rolling-failure-rate", type=float, default=0.5)
    pr.add_argument(
        "--scope",
        choices=["all", "wrong-only", "wrong-plus-correct-calibration"],
        default="all",
        help="session subset to analyze; calibration scope samples CORRECT by turn/mcp buckets",
    )
    pr.add_argument("--correct-sample-per-bucket", type=int, default=10)

    pdry = sub.add_parser("dry-link")
    pdry.add_argument("--result-dir", required=True)
    pdry.add_argument("--limit", type=int, default=20)

    sub.add_parser("runtimes")

    args = p.parse_args(argv)

    if args.cmd == "dry-link":
        from linker import iter_linked
        n = 0
        for ls in iter_linked(args.result_dir, limit=args.limit):
            n += 1
            print(f"{n:3d} {ls.session_id[:8]} {ls.sample_id} q{ls.question_index} "
                  f"{ls.result} turns={ls.num_turns} "
                  f"mcp_metric={ls.ov_mcp_calls} transcript_mcp={ls.transcript_mcp_calls}")
        return 0

    if args.cmd == "runtimes":
        print("available runtimes:")
        seen: set[str] = set()
        for runtime in RUNTIME_CHOICES:
            normalized = normalize_runtime(runtime)
            if normalized in seen:
                continue
            seen.add(normalized)
            print(
                f"- {normalized}: worker_default="
                f"{default_model_for_runtime(normalized, 'worker') or '<runtime default>'}; "
                f"coordinator_default="
                f"{default_model_for_runtime(normalized, 'coordinator') or '<runtime default>'}"
            )
        return 0

    if args.cmd == "run":
        run_id = args.run_id or time.strftime("run-%Y%m%d-%H%M%S")
        out_dir = _resolve_run_output_dir(args.output_dir, run_id)
        try:
            worker_runtime, coordinator_runtime, worker_model, coordinator_model = _resolve_runtime_args(args)
            worker_backend = build_backend(worker_runtime)
            coordinator_backend = (
                worker_backend
                if coordinator_runtime == worker_runtime
                else build_backend(coordinator_runtime)
            )
        except (ValueError, RuntimeError) as e:
            p.error(str(e))
        print(
            "[runtime] "
            f"worker={worker_runtime} model={worker_model or '<runtime default>'} "
            f"coordinator={coordinator_runtime} model={coordinator_model or '<runtime default>'} "
            f"output={out_dir}"
        )
        cfg = RunConfig(
            result_dir=args.result_dir,
            run_id=run_id,
            out_dir=out_dir,
            worker_backend=worker_backend,
            coordinator_backend=coordinator_backend,
            worker_model=worker_model,
            coordinator_model=coordinator_model,
            concurrency=args.concurrency,
            limit=args.limit,
            reflect_every_n_sessions=args.reflect_every_n,
            midrun_reflection=args.midrun_reflection,
            final_reflection=not args.no_final_reflection,
            worker_max_tokens=args.worker_max_tokens,
            coordinator_max_tokens=args.coordinator_max_tokens,
            circuit_breaker_consecutive_failures=args.circuit_breaker_consecutive_failures,
            circuit_breaker_rolling_window=args.circuit_breaker_rolling_window,
            circuit_breaker_rolling_failure_rate=args.circuit_breaker_rolling_failure_rate,
            scope=args.scope,
            correct_sample_per_bucket=args.correct_sample_per_bucket,
        )
        asyncio.run(run(cfg))
        print(f"Output: {out_dir}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
