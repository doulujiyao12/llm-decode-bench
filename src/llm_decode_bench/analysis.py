import json
import statistics as st
from pathlib import Path

from .io import percentile, read_jsonl, write_json
from .metrics import window_ready


def analyze(root):
    root = Path(root)
    cfg = json.loads((root / "config.json").read_text())
    rows = [r for r in read_jsonl(root / "requests.jsonl") if not r["warmup"]]
    samples = read_jsonl(root / "metrics.jsonl")
    captures = read_jsonl(root / "capture.jsonl")
    expected = cfg["dp_size"] * cfg["batch_size"]
    out = {"schema_version": 1, "status": "invalid", "requests": len(rows),
           "successful": sum(bool(r.get("ok")) for r in rows), "windows": [],
           "batch_verification": "sampled_metrics" if cfg["steady_mode"] == "metrics" else "client_only_unverified",
           "sse_gap_p50_ms": None, "sse_gap_p95_ms": None, "reasons": []}
    if (len(rows) != expected or len({r["index"] for r in rows}) != expected
            or not all(r.get("ok") for r in rows)):
        out["reasons"].append("missing/failed/duplicate requests; never drop failed requests from the denominator")
    elif any(len(r["events"]) < cfg["settle_events"] + 2 for r in rows):
        out["reasons"].append("not enough progress events from every request")
    else:
        a = max(r["events"][cfg["settle_events"] - 1]["t"] for r in rows) + cfg["guard_seconds"]
        end = min(r["events"][-1]["t"] for r in rows) - cfg["guard_seconds"]
        out["common_window"] = [a, end]
        arm = next((c["t"] for c in captures if c["action"] == "arm"), None)
        done = next((c["t"] for c in captures if c["action"] in {"done", "failed"}), float("inf"))
        while a + cfg["window_seconds"] <= end:
            b = a + cfg["window_seconds"]
            medians, p95s, deltas = [], [], []
            for r in rows:
                pairs = [(x, y) for x, y in zip(r["events"], r["events"][1:]) if a <= x["t"] < y["t"] <= b]
                gaps = [(y["t"] - x["t"]) * 1000 for x, y in pairs]
                if len(gaps) >= cfg["min_gaps"]:
                    medians.append(st.median(gaps))
                    p95s.append(percentile(gaps, .95))
                    deltas.extend(y["tokens"] for _, y in pairs if y["tokens"] is not None)
            verified = cfg["steady_mode"] == "client" or window_ready(samples, a, b, cfg)
            overlap = arm is not None and a < done + cfg["guard_seconds"] and b > arm - cfg["guard_seconds"]
            reasons = []
            if len(medians) != expected:
                reasons.append("insufficient_gaps")
            if not verified:
                reasons.append("metrics_not_exact_or_missing")
            if overlap:
                reasons.append("gpu_event_capture_overlap")
            out["windows"].append({"start": a, "end": b, "valid": not reasons, "reasons": reasons,
                "sse_gap_p50_ms": st.median(medians) if medians else None,
                "sse_gap_p95_ms": st.median(p95s) if p95s else None,
                "mean_tokens_per_progress_event": st.mean(deltas) if deltas else None})
            a = b
        valid = [w for w in out["windows"] if w["valid"]]
        out["valid_window_count"] = len(valid)
        if valid:
            out["status"] = "valid"
            out["sse_gap_p50_ms"] = st.median(w["sse_gap_p50_ms"] for w in valid)
            out["sse_gap_p95_ms"] = st.median(w["sse_gap_p95_ms"] for w in valid)
        else:
            out["reasons"].append("no complete steady window; increase output length or investigate scheduling/memory")
    if cfg.get("event_steps", 0):
        # Offline analysis must not turn an incomplete requested capture into a valid run.
        from .events import analyze_events
        gpu = analyze_events(root, cfg["dp_size"], cfg["batch_size"], cfg["event_steps"])
        if not gpu["valid"]:
            out["status"] = "invalid"
            out["reasons"].append("requested GPU capture incomplete or invalid")
    out["metric_definition"] = "median of per-window medians of per-request SSE gaps; P95 column uses per-request P95, NOT pooled P95"
    write_json(root / "analysis.json", out)
    return out


COMPARISON_FIELDS = ("backend", "model", "dp_size", "batch_size", "route_dp", "output_tokens",
                     "temperature", "ignore_eos", "warmup", "warmup_concurrency", "steady_mode", "metrics_spec",
                     "window_seconds", "min_gaps", "settle_events", "guard_seconds",
                     "max_metric_gap", "metrics_interval", "extra_body_hash", "event_steps")


def compare(a, b):
    a, b = Path(a), Path(b)
    ca, cb = (json.loads((p / "config.json").read_text()) for p in (a, b))
    mismatch = [k for k in COMPARISON_FIELDS if ca.get(k) != cb.get(k)]
    ma, mb = (json.loads((p / "manifest.json").read_text()) for p in (a, b))
    if ma != mb:
        mismatch.append("input_manifest")
    aa, ab = analyze(a), analyze(b)
    if mismatch or aa["status"] != "valid" or ab["status"] != "valid":
        return {"comparable": False, "mismatches": mismatch,
                "baseline_status": aa["status"], "candidate_status": ab["status"]}
    x, y = aa["sse_gap_p50_ms"], ab["sse_gap_p50_ms"]
    return {"comparable": True, "baseline_sse_gap_ms": x, "candidate_sse_gap_ms": y,
            "saved_ms": x - y, "reduction_percent": (x - y) / x * 100 if x else None,
            "batch_verification": aa["batch_verification"],
            "caution": "Input/config checks passed; confirm model weights, server flags, clocks and competing load separately"}
