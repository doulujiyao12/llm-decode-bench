import asyncio
import json
import os
import platform
import time
import uuid
from pathlib import Path

import aiohttp

from . import __version__
from .analysis import analyze
from .inputs import load_dataset, manifest
from .io import append_jsonl, digest, endpoint, write_json
from .metrics import parse, ready
from .sse import Progress, StreamError, messages


def make_body(cfg, row, index, rid, warmup, extra):
    limit = 1 if warmup else cfg["output_tokens"]
    if cfg["backend"] == "sglang":
        params = {"max_new_tokens": limit, "temperature": cfg["temperature"], "ignore_eos": cfg["ignore_eos"]}
        body = {**row, "sampling_params": params, "stream": True, "rid": rid}
        if cfg["route_dp"]:
            body["routed_dp_rank"] = index % cfg["dp_size"]
    else:
        body = {"model": cfg["model"], "max_tokens": limit, "temperature": cfg["temperature"], "stream": True, "n": 1}
        if cfg["backend"] == "openai-chat":
            if "input_ids" in row:
                raise ValueError("Chat requires text JSONL; token IDs cannot be round-tripped without changing tokenization")
            body["messages"] = [{"role": "user", "content": row["text"]}]
        else:
            body["prompt"] = row.get("input_ids", row.get("text"))
        if cfg["ignore_eos"]:
            body["ignore_eos"] = True  # Explicit opt-in nonstandard extension.
    reserved = {"input_ids", "text", "prompt", "messages", "sampling_params", "model", "stream", "n", "rid",
                "routed_dp_rank", "temperature", "max_tokens", "max_new_tokens", "ignore_eos"}
    if reserved & set(extra):
        raise ValueError("extra-body must not override benchmark-owned fields")
    return body | extra


async def run(cfg, dataset, output, extra=None, api_key_env="OPENAI_API_KEY", event_dir=None):
    extra = extra or {}
    count = cfg["dp_size"] * cfg["batch_size"]
    inputs = load_dataset(dataset, count)
    # Validate before creating files or sending requests.
    for i, row in enumerate(inputs):
        make_body(cfg, row, i, "validation", False, extra)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    for name in ("requests.jsonl", "metrics.jsonl", "capture.jsonl"):
        (out / name).touch()
    config = dict(cfg, schema_version=1, tool_version=__version__, python=platform.python_version(),
                  client_platform=platform.platform(), created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  extra_body_hash=digest(extra))
    write_json(out / "config.json", config)
    write_json(out / "manifest.json", manifest(inputs, cfg["dp_size"]))
    namespace = "ldb_" + uuid.uuid4().hex
    clock = time.perf_counter()
    now = lambda: time.perf_counter() - clock
    active, measured = {}, []
    latest = None
    stop = asyncio.Event()
    headers = {"Authorization": f"Bearer {os.environ[api_key_env]}"} if os.environ.get(api_key_env) else {}
    timeout = aiohttp.ClientTimeout(total=cfg["request_timeout"], connect=30)
    suffix = {"sglang": "/generate", "openai": "/v1/completions", "openai-chat": "/v1/chat/completions"}[cfg["backend"]]
    url = endpoint(cfg["base_url"], suffix)
    async with aiohttp.ClientSession(timeout=timeout, headers=headers, trust_env=False,
            connector=aiohttp.TCPConnector(limit=0), read_bufsize=256 * 1024) as session:
        async def request(i, prompt, warmup=False):
            rid = f"{namespace}_{'warm' if warmup else 'run'}_{i}"
            row = {"index": i, "rid": rid, "warmup": warmup, "start": now(), "events": [], "ok": False}
            if not warmup:
                active[rid] = row
            progress = Progress(cfg["backend"])
            try:
                async with session.post(url, json=make_body(cfg, prompt, i, rid, warmup, extra)) as response:
                    if response.status != 200:
                        raise StreamError(f"HTTP {response.status}; response body omitted")
                    if "text/event-stream" not in response.headers.get("Content-Type", ""):
                        raise StreamError("Response is not text/event-stream")
                    async for payload in messages(response.content.iter_any()):
                        changed, tokens = progress.consume(payload)
                        if changed:
                            row["events"].append({"t": now(), "tokens": tokens})
                    if not (progress.done or progress.finished):
                        raise StreamError("Stream ended without completion marker")
                    if not row["events"]:
                        raise StreamError("No generation progress events")
                    if cfg["route_dp"] and progress.dp_rank != i % cfg["dp_size"]:
                        raise StreamError("Missing or incorrect returned dp_rank")
                    if "input_ids" in prompt and progress.prompt_tokens is not None and progress.prompt_tokens != len(prompt["input_ids"]):
                        raise StreamError("Server prompt token count differs from input_ids")
                    if cfg["backend"] == "sglang" and cfg["ignore_eos"] and progress.tokens != (1 if warmup else cfg["output_tokens"]):
                        raise StreamError("ignore_eos requested but output count differs from limit")
                    row["ok"] = True
            except (aiohttp.ClientError, asyncio.TimeoutError, StreamError, UnicodeError, ValueError) as e:
                # Never persist server bodies, input text, headers, or arbitrary exception reprs.
                row["error"] = str(e) if isinstance(e, StreamError) else type(e).__name__
            finally:
                row.update(end=now(), completion_tokens=progress.tokens or None,
                           prompt_tokens=progress.prompt_tokens, returned_dp_rank=progress.dp_rank,
                           cached_tokens=progress.cached_tokens)
                active.pop(rid, None)
                append_jsonl(out / "requests.jsonl", row)
                if not warmup:
                    measured.append(row)
            return row

        async def poll_metrics():
            nonlocal latest
            if not cfg["metrics_url"]:
                return
            while not stop.is_set():
                started = now()
                try:
                    async with session.get(cfg["metrics_url"], timeout=aiohttp.ClientTimeout(total=10)) as r:
                        if r.status != 200:
                            raise ValueError("HTTP error")
                        # Cap scrape size; only readiness fields are persisted.
                        pieces, size = [], 0
                        async for chunk in r.content.iter_chunked(65536):
                            size += len(chunk)
                            if size > 8 * 1024 * 1024:
                                raise ValueError("Metrics response exceeds limit")
                            pieces.append(chunk)
                        body = b"".join(pieces)
                        ranks = parse(body.decode(), cfg["metrics_spec"])
                    latest = {"t": now(), "latency": now() - started, "ok": True, "ranks": ranks}
                except (aiohttp.ClientError, asyncio.TimeoutError, ValueError, UnicodeError) as e:
                    latest = {"t": now(), "latency": now() - started, "ok": False, "error": type(e).__name__}
                append_jsonl(out / "metrics.jsonl", latest)
                try:
                    await asyncio.wait_for(stop.wait(), cfg["metrics_interval"])
                except asyncio.TimeoutError:
                    pass

        async def capture_events():
            if not event_dir:
                return
            root = Path(event_dir)
            if not root.is_dir():
                raise ValueError("event-dir must exist and be shared with the instrumented server")
            # Exclusive lock prevents this client from clobbering another capture.
            lock = root / "client.lock"
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            os.close(fd)
            armed = False
            try:
                stable_since = None
                until = now() + cfg["event_timeout"]
                while not stop.is_set() and now() < until:
                    client_ready = len(active) == count and all(len(r["events"]) >= cfg["settle_events"] for r in active.values())
                    metric_ready = (cfg["steady_mode"] != "metrics" or (latest and latest["ok"]
                        and now() - latest["t"] <= cfg["max_metric_gap"]
                        and ready(latest["ranks"], cfg["metrics_spec"], cfg["dp_size"], cfg["batch_size"])))
                    if client_ready and metric_ready:
                        stable_since = stable_since if stable_since is not None else now()
                    else:
                        stable_since = None
                    if stable_since is not None and now() - stable_since >= cfg["guard_seconds"]:
                        break
                    await asyncio.sleep(.2)
                else:
                    raise ValueError("No readiness before capture timeout/cohort end")
                control = {"case": namespace, "steps": cfg["event_steps"], "batch_size": cfg["batch_size"],
                           "dp_size": cfg["dp_size"], "rid_prefix": namespace + "_run_",
                           "expires": time.time() + cfg["event_timeout"]}
                write_json(root / "control.json", control)
                append_jsonl(out / "capture.jsonl", {"action": "arm", "t": now(), "case": namespace})
                armed = True
                until = now() + cfg["event_timeout"]
                while now() < until and not stop.is_set():
                    paths = list(root.glob(f"{namespace}-rank*-done.json"))
                    if len(paths) >= cfg["dp_size"]:
                        results = [json.loads(p.read_text()) for p in paths]
                        if {r["rank"] for r in results} != set(range(cfg["dp_size"])):
                            raise ValueError("GPU capture rank mismatch")
                        for p in root.glob(f"{namespace}-rank*-events.jsonl"):
                            (out / p.name).write_bytes(p.read_bytes())
                        write_json(out / "gpu_completion.json", results)
                        append_jsonl(out / "capture.jsonl", {"action": "done", "t": now()})
                        return
                    await asyncio.sleep(.2)
                raise ValueError("GPU event capture incomplete; inspect hook status files")
            finally:
                if armed:
                    write_json(root / "control.json", {})
                lock.unlink()

        poll = asyncio.create_task(poll_metrics())
        capture = None
        try:
            if cfg["warmup"]:
                semaphore = asyncio.Semaphore(cfg["warmup_concurrency"])
                async def warm(i, row):
                    async with semaphore:
                        return await request(i, row, True)
                results = await asyncio.gather(*(warm(i, row) for i, row in enumerate(inputs)))
                if not all(r["ok"] for r in results):
                    raise ValueError("Warmup failed; inspect requests.jsonl (no measurement started)")
                await asyncio.sleep(cfg["guard_seconds"])
            if event_dir:
                capture = asyncio.create_task(capture_events())
            print(f"Fixed cohort: {count} requests, target BS={cfg['batch_size']}/rank; no replenishment", flush=True)
            await asyncio.gather(*(request(i, row) for i, row in enumerate(inputs)))
        finally:
            # Take another scrape for bracketing steady windows, then stop.
            await asyncio.sleep(cfg["metrics_interval"] if cfg["metrics_url"] else 0)
            stop.set()
            await poll
            if capture:
                try:
                    await capture
                except (ValueError, OSError) as e:
                    append_jsonl(out / "capture.jsonl", {"action": "failed", "t": now(), "reason": type(e).__name__})
    if event_dir:
        from .events import analyze_events
        analyze_events(out, cfg["dp_size"], cfg["batch_size"], cfg["event_steps"])
    result = analyze(out)
    print(json.dumps({k: v for k, v in result.items() if k != "windows"}, indent=2), flush=True)
    return result
