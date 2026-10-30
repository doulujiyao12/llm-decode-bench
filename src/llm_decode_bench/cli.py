import argparse
import asyncio
import json
import math
from pathlib import Path
import sys

from .analysis import analyze, compare
from .io import endpoint, write_json
from .metrics import SGLANG, SGLANG_PD, validate_spec


def parser():
    p = argparse.ArgumentParser(description="Mixed-length, fixed-cohort LLM decode diagnostics (SSE by default)")
    sub = p.add_subparsers(dest="command", required=True)
    g = sub.add_parser("generate", help="Generate valid synthetic token IDs using a model tokenizer")
    g.add_argument("--tokenizer", required=True)
    g.add_argument("--tokenizer-revision", default=None)
    g.add_argument("--lengths", nargs="+", type=int, required=True, help="Token counts, equal-probability buckets")
    g.add_argument("--count", type=int, required=True)
    g.add_argument("--seed", type=int, default=60)
    g.add_argument("--shared-prefix-tokens", type=int, default=0)
    g.add_argument("--prefix-families", type=int, default=1)
    g.add_argument("--output", required=True)
    for name in ("run", "sweep"):
        r = sub.add_parser(name, help="Run one cohort" if name == "run" else "Sequential BS matrix, same nested dataset")
        r.add_argument("--base-url", required=True, help="API origin/path prefix WITHOUT /v1; no credentials in URL")
        r.add_argument("--backend", choices=["sglang", "openai", "openai-chat"], default="openai")
        r.add_argument("--model", required=True)
        r.add_argument("--dataset", required=True, help="JSONL: one text or input_ids per line")
        r.add_argument("--dp-size", type=int, default=1)
        if name == "run":
            r.add_argument("--batch-size", type=int, required=True, help="Target running requests per DP rank")
        else:
            r.add_argument("--batch-sizes", type=int, nargs="+", required=True)
            r.add_argument("--repeats", type=int, default=1)
        r.add_argument("--route-dp", action="store_true", help="Native SGLang only; require returned dp_rank confirmation")
        r.add_argument("--output-tokens", type=int, default=2048)
        r.add_argument("--temperature", type=float, default=0.)
        r.add_argument("--ignore-eos", action="store_true", help="Server extension; may produce meaningless text, performance only")
        r.add_argument("--warmup", action="store_true", help="Preload the same prefixes with 1-token requests; no cache flush")
        r.add_argument("--warmup-concurrency", type=int, default=8)
        r.add_argument("--steady-mode", choices=["client", "metrics"], default="client")
        r.add_argument("--metrics-url", default="")
        r.add_argument("--metrics-preset", choices=["sglang", "sglang-pd"], default="sglang")
        r.add_argument("--metrics-map", help="JSON metric names, rank label, and fixed label selectors")
        r.add_argument("--metrics-interval", type=float, default=.5)
        r.add_argument("--max-metric-gap", type=float, default=2.)
        r.add_argument("--window-seconds", type=float, default=5.)
        r.add_argument("--guard-seconds", type=float, default=3.)
        r.add_argument("--settle-events", type=int, default=5)
        r.add_argument("--min-gaps", type=int, default=20)
        r.add_argument("--request-timeout", type=float, default=7200.)
        r.add_argument("--api-key-env", default="OPENAI_API_KEY")
        r.add_argument("--extra-body", help="Path to server-specific JSON options (never copied into results)")
        r.add_argument("--event-dir", help="OPTIONAL shared control directory; requires server hook installed before startup")
        r.add_argument("--event-steps", type=int, default=128)
        r.add_argument("--event-timeout", type=float, default=600.)
        r.add_argument("--label", default="unspecified", help="e.g. baseline or candidate; does not reconfigure server")
        r.add_argument("--output", required=True, help="NEW result directory; refuses overwrite")
    a = sub.add_parser("analyze", help="Recompute steady windows from saved measurements")
    a.add_argument("result")
    c = sub.add_parser("compare", help="Compare matching inputs/configs; refuse invalid cohorts")
    c.add_argument("baseline")
    c.add_argument("candidate")
    c.add_argument("--output")
    v = sub.add_parser("plot", help="Write SVG using saved analyses; no matplotlib needed")
    v.add_argument("results", nargs="+")
    v.add_argument("--output", required=True)
    return p


def config(args, bs):
    if not all(math.isfinite(x) for x in (args.temperature, args.window_seconds, args.metrics_interval,
            args.max_metric_gap, args.request_timeout, args.event_timeout, args.guard_seconds)):
        raise ValueError("Numeric options must be finite")
    spec = json.loads(Path(args.metrics_map).read_text()) if args.metrics_map else (
        SGLANG_PD if args.metrics_preset == "sglang-pd" else SGLANG)
    validate_spec(spec)
    if min(args.dp_size, bs, args.output_tokens, args.warmup_concurrency, args.min_gaps, args.settle_events) < 1:
        raise ValueError("Counts and batch sizes must be positive")
    if min(args.window_seconds, args.metrics_interval, args.max_metric_gap, args.request_timeout, args.event_timeout) <= 0 or args.guard_seconds < 0:
        raise ValueError("Durations must be positive, guard-seconds >= 0")
    if args.temperature < 0:
        raise ValueError("temperature must be nonnegative")
    endpoint(args.base_url, "")
    if args.base_url.rstrip("/").endswith("/v1"):
        raise ValueError("base-url should not include /v1")
    if args.metrics_url:
        endpoint(args.metrics_url, "")
    if args.steady_mode == "metrics" and not args.metrics_url:
        raise ValueError("metrics steady mode requires --metrics-url; no silent fallback")
    if args.steady_mode == "metrics" and args.dp_size > 1 and not spec.get("rank_label"):
        raise ValueError("Global metrics cannot verify per-DP batch sizes")
    if (args.route_dp or args.event_dir) and args.backend != "sglang":
        raise ValueError("Explicit DP routing/event request identity currently requires native SGLang")
    if args.event_dir and not 5 <= args.event_steps <= 4096:
        raise ValueError("event-steps must be 5..4096")
    keys = ("base_url", "backend", "model", "dp_size", "route_dp", "output_tokens", "temperature", "ignore_eos",
            "warmup", "warmup_concurrency", "steady_mode", "metrics_url", "metrics_interval", "max_metric_gap",
            "window_seconds", "guard_seconds", "settle_events", "min_gaps", "request_timeout", "event_timeout", "label")
    return {**{k: getattr(args, k) for k in keys}, "batch_size": bs, "metrics_spec": spec,
            "event_steps": args.event_steps if args.event_dir else 0}


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "generate":
            from .inputs import generate
            generate(args)
        elif args.command in {"run", "sweep"}:
            from .runner import run
            extra = json.loads(Path(args.extra_body).read_text()) if args.extra_body else {}
            if not isinstance(extra, dict):
                raise ValueError("extra-body must be a JSON object")
            if args.command == "run":
                result = asyncio.run(run(config(args, args.batch_size), args.dataset, args.output, extra, args.api_key_env, args.event_dir))
                if result["status"] != "valid":
                    return 2
            else:
                if args.repeats < 1 or len(set(args.batch_sizes)) != len(args.batch_sizes):
                    raise ValueError("repeats >=1 and unique batch-sizes required")
                configs = [config(args, bs) for bs in args.batch_sizes]
                root = Path(args.output)
                root.mkdir(parents=True, exist_ok=False)
                rows = []
                for repeat in range(1, args.repeats + 1):
                    for cfg in configs:
                        name = f"bs{cfg['batch_size']}-r{repeat}"
                        result = asyncio.run(run(cfg, args.dataset, root / name, extra, args.api_key_env, args.event_dir))
                        rows.append({"case": name, "status": result["status"], "sse_gap_p50_ms": result["sse_gap_p50_ms"]})
                        write_json(root / "matrix.json", rows)
                        if result["successful"] < result["requests"]:
                            raise ValueError("Request failure: stopping matrix to avoid adding more load")
                if any(r["status"] != "valid" for r in rows):
                    return 2
        elif args.command == "analyze":
            result = analyze(args.result)
            print(json.dumps(result, indent=2))
            if result["status"] != "valid":
                return 2
        elif args.command == "compare":
            result = compare(args.baseline, args.candidate)
            if args.output:
                if Path(args.output).exists():
                    raise FileExistsError(args.output)
                write_json(args.output, result)
            print(json.dumps(result, indent=2))
            if not result["comparable"]:
                return 2
        elif args.command == "plot":
            from .plot import plot
            print(plot(args.results, args.output))
    except (ValueError, OSError, ImportError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0


def entrypoint():
    raise SystemExit(main())
