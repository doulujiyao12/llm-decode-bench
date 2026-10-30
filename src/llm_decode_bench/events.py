"""GPU event output audit; client module does not import torch."""
import json
import statistics as st
from pathlib import Path

from .io import read_jsonl, write_json


def analyze_events(root, dp_size, batch_size, steps):
    root = Path(root)
    paths = list(root.glob("ldb_*-rank*-events.jsonl"))
    rows = [r for p in paths for r in read_jsonl(p)]
    manifests = json.loads((root / "manifest.json").read_text())
    reqs = [r for r in read_jsonl(root / "requests.jsonl") if not r["warmup"]]
    config = json.loads((root / "config.json").read_text())
    groups = {rank: [r for r in rows if r["rank"] == rank] for rank in range(dp_size)}
    reasons, summaries = [], []
    if {r["rank"] for r in rows} != set(groups):
        reasons.append("missing or extra ranks")
    if not (root / "gpu_completion.json").exists():
        reasons.append("capture not completed")
    all_rids = {r["rid"] for r in reqs}
    memberships = []
    for rank, rank_rows in groups.items():
        expected = {r["rid"] for r in reqs if r["index"] % dp_size == rank}
        first = set(rank_rows[0]["rids"]) if rank_rows else set()
        memberships.append(first)
        if len(rank_rows) != steps or {r["step"] for r in rank_rows} != set(range(1, steps + 1)):
            reasons.append(f"rank {rank}: incomplete/duplicate steps")
        for r in rank_rows:
            if (r["bs"] != batch_size or len(set(r["rids"])) != batch_size
                    or set(r["rids"]) != first or not set(r["rids"]) <= all_rids
                    or (config["route_dp"] and set(r["rids"]) != expected)
                    or r["gpu_ms"] < 0):
                reasons.append(f"rank {rank}: batch membership or timing mismatch")
                break
        trimmed = rank_rows[2:-2] if len(rank_rows) > 4 else []
        if trimmed:
            summaries.append({"rank": rank, "steps": len(rank_rows),
                "gpu_step_p50_ms": st.median(r["gpu_ms"] for r in trimmed),
                "gpu_gap_before_p50_ms": st.median(r["gap_ms"] for r in trimmed if r.get("gap_ms") is not None),
                "context_sum_mean": st.mean(r["context_sum"] for r in trimmed)})
    if len(set().union(*memberships)) != len(manifests) or sum(map(len, memberships)) != len(manifests):
        reasons.append("membership union does not match entire cohort")
    out = {"valid": not reasons, "reasons": reasons, "ranks": summaries,
           "gpu_step_p50_ms": st.median(r["gpu_step_p50_ms"] for r in summaries) if summaries and not reasons else None,
           "definition": "median of rank medians, first/last 2 steps removed; forward-stream event interval, not kernel busy-time"}
    write_json(root / "gpu_analysis.json", out)
    return out
