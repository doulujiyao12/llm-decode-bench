"""Small Prometheus text reader with explicit labels and no silent replica sum."""
import json
import math
import re

SAMPLE = re.compile(r'^([\w:]+)(?:\{(.*)\})?\s+([^\s]+)(?:\s+\S+)?$')
LABEL = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"(?:,\s*|$)')
SGLANG = {"rank_label": "dp_rank", "labels": {}, "metrics": {
    "running": "sglang:num_running_reqs", "waiting": "sglang:num_queue_reqs"}}
SGLANG_PD = {**SGLANG, "metrics": {**SGLANG["metrics"],
    "prealloc": "sglang:num_decode_prealloc_queue_reqs",
    "transfer": "sglang:num_decode_transfer_queue_reqs"}}


def validate_spec(spec):
    if not isinstance(spec.get("metrics"), dict) or not {"running", "waiting"} <= spec["metrics"].keys():
        raise ValueError("Metrics mapping must include running and waiting")
    if not set(spec["metrics"]) <= {"running", "waiting", "prealloc", "transfer"}:
        raise ValueError("Only running/waiting/prealloc/transfer are readiness fields")
    if len(set(spec["metrics"].values())) != len(spec["metrics"]):
        raise ValueError("Each readiness field needs a distinct metric")
    return spec


def parse(text, spec):
    inverse = {name: field for field, name in spec["metrics"].items()}
    ranks = {}
    for raw in text.splitlines():
        if not raw or raw.startswith("#"):
            continue
        m = SAMPLE.match(raw)
        if not m or m[1] not in inverse:
            continue
        labels = {}
        raw_labels = m[2] or ""
        pos = 0
        for label in LABEL.finditer(raw_labels):
            if label.start() != pos:
                raise ValueError("Malformed metric labels")
            labels[label[1]] = json.loads('"' + label[2] + '"')
            pos = label.end()
        if pos != len(raw_labels):
            raise ValueError("Malformed metric labels")
        if any(labels.get(k) != v for k, v in spec.get("labels", {}).items()):
            continue
        rank_label = spec.get("rank_label")
        if rank_label:
            if rank_label not in labels:
                raise ValueError("Missing rank label; provide a mapping for this service")
            rank = int(labels[rank_label])
        else:
            rank = 0  # Global gauges allowed only with dp_size=1 (validated by client).
        value = float(m[3])
        if not math.isfinite(value) or value < 0:
            raise ValueError("Invalid metric value")
        target = ranks.setdefault(str(rank), {})
        field = inverse[m[1]]
        if field in target:
            raise ValueError("Duplicate metric per rank; select model/worker/TP labels explicitly")
        target[field] = value
    return ranks


def ready(ranks, spec, dp_size, batch_size):
    if set(ranks) != {str(i) for i in range(dp_size)}:
        return False
    for fields in ranks.values():
        if set(fields) != set(spec["metrics"]):
            return False
        if fields["running"] != batch_size:
            return False
        if any(v != 0 for k, v in fields.items() if k != "running"):
            return False
    return True


def window_ready(samples, a, b, cfg):
    """Require fresh, bracketing successful samples; polling cannot prove every instant."""
    before = [s for s in samples if s["t"] <= a]
    after = [s for s in samples if s["t"] >= b]
    if not before or not after:
        return False
    selected = [before[-1]] + [s for s in samples if a < s["t"] < b] + [after[0]]
    gap = cfg["max_metric_gap"]
    if a - selected[0]["t"] > gap or selected[-1]["t"] - b > gap:
        return False
    if any(y["t"] - x["t"] > gap for x, y in zip(selected, selected[1:])):
        return False
    return all(s.get("ok") and s.get("latency", 0) <= gap and ready(
        s["ranks"], cfg["metrics_spec"], cfg["dp_size"], cfg["batch_size"])
        for s in selected)
