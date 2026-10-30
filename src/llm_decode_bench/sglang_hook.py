"""Opt-in SGLang adapter. No synchronize, tensor reads, SDK injection or signals.

Only the exact original run_batch is executed once. Instrumentation errors disable
measurement and leave serving intact. Original serving exceptions propagate.
"""
import functools
import importlib.abc
import importlib.machinery
import json
import os
from pathlib import Path
import re
import sys
import time

from .io import append_jsonl, write_json


def rank_attr(scheduler, name):
    ps = getattr(scheduler, "ps", scheduler)
    value = getattr(ps, name, None)
    if value is None:
        value = getattr(scheduler, name, None)
    if value is None:
        raise ValueError(f"Unsupported SGLang: missing {name}")
    return int(value)


class Recorder:
    def __init__(self, root, torch, leaders):
        self.root, self.torch, self.leaders = Path(root), torch, set(leaders)
        self.pending, self.done = [], set()
        self.control = None
        self.last_poll, self.count, self.previous = 0., 0, None
        self.disabled = False

    def fail(self, error):
        self.disabled = True
        # Do not let a filesystem/diagnostic failure kill model serving.
        try:
            write_json(self.root / f"status-{os.getpid()}.json", {"state": "disabled", "error_type": type(error).__name__})
        except Exception:
            pass

    def drain(self):
        remaining = []
        for begin, end, previous, row in self.pending:
            if end.query():
                row["gpu_ms"] = begin.elapsed_time(end)
                row["gap_ms"] = previous.elapsed_time(begin) if previous is not None and previous.query() else None
                append_jsonl(self.root / f"{row['case']}-rank{row['rank']}-events.jsonl", row)
            else:
                remaining.append((begin, end, previous, row))
        self.pending = remaining
        if self.control and self.count >= self.control["steps"] and not remaining:
            case, rank = self.control["case"], self.rank
            write_json(self.root / f"{case}-rank{rank}-done.json", {"case": case, "rank": rank, "steps": self.count})
            self.done.add(case)
            self.control = None

    def before(self, scheduler, batch):
        tp = rank_attr(scheduler, "tp_rank")
        if self.disabled or tp not in self.leaders:
            return None
        self.rank = rank_attr(scheduler, "attn_dp_rank")
        if getattr(getattr(scheduler, "server_args", None), "pp_size", 1) != 1:
            raise ValueError("Pipeline parallel event measurement is unsupported")
        self.drain()
        t = time.monotonic()
        if t - self.last_poll > .2:
            self.last_poll = t
            try:
                c = json.loads((self.root / "control.json").read_text())
            except FileNotFoundError:
                c = {}
            if (c.get("case") not in self.done and re.fullmatch(r"ldb_[0-9a-f]{32}", c.get("case", ""))
                    and c.get("expires", 0) > time.time() and 5 <= c.get("steps", 0) <= 4096
                    and c.get("rid_prefix") == c["case"] + "_run_" and c.get("batch_size", 0) > 0):
                if self.control is None or c["case"] != self.control["case"]:
                    if self.pending:
                        return None
                    self.control, self.count, self.previous = c, 0, None
            elif not self.pending:
                self.control = None
        c = self.control
        if not c or time.time() > c["expires"] or self.count >= c["steps"]:
            return None
        if not batch.forward_mode.is_decode() or len(batch.reqs) != c["batch_size"]:
            return None
        rids = [r.rid for r in batch.reqs]
        if not all(isinstance(r, str) and r.startswith(c["rid_prefix"]) for r in rids):
            return None
        self.count += 1
        lengths = [len(r.origin_input_ids) + len(r.output_ids) for r in batch.reqs]
        row = {"case": c["case"], "step": self.count, "rank": self.rank, "tp_rank": tp,
               "bs": len(rids), "rids": rids, "context_sum": sum(lengths), "context_max": max(lengths),
               "wall_time": time.time()}
        begin, end = self.torch.cuda.Event(enable_timing=True), self.torch.cuda.Event(enable_timing=True)
        begin.record(scheduler.forward_stream)
        return begin, end, self.previous, row

    def after(self, scheduler, item):
        item[1].record(scheduler.forward_stream)
        self.pending.append(item)
        self.previous = item[1]


def patch_scheduler(module, root, leaders):
    import torch
    cls = module.Scheduler
    if getattr(cls.run_batch, "_ldb_instrumented", False):
        return
    recorder = Recorder(root, torch, leaders)
    original = cls.run_batch

    @functools.wraps(original)
    def run_batch(self, batch, *args, **kwargs):
        item = None
        if not recorder.disabled:
            try:
                item = recorder.before(self, batch)
            except Exception as error:
                recorder.fail(error)
        try:
            return original(self, batch, *args, **kwargs)
        finally:
            if item is not None:
                try:
                    recorder.after(self, item)
                except Exception as error:
                    recorder.fail(error)

    run_batch._ldb_instrumented = True
    cls.run_batch = run_batch
    try:
        write_json(Path(root) / f"status-{os.getpid()}.json", {"state": "installed", "leader_tp_ranks": leaders,
            "torch_version": torch.__version__, "hip": getattr(torch.version, "hip", None)})
    except OSError as e:
        recorder.fail(e)


def install(root, leaders):
    target_name = "sglang.srt.managers.scheduler"
    if target_name in sys.modules:
        patch_scheduler(sys.modules[target_name], root, leaders)
        return

    class Loader:
        def __init__(self, loader):
            self.loader = loader

        def create_module(self, spec):
            return self.loader.create_module(spec)

        def exec_module(self, module):
            self.loader.exec_module(module)
            try:
                patch_scheduler(module, root, leaders)
            except Exception as e:
                try:
                    write_json(Path(root) / f"status-{os.getpid()}.json", {"state": "unsupported", "error_type": type(e).__name__})
                except OSError:
                    pass

        def __getattr__(self, name):
            return getattr(self.loader, name)

    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname != target_name:
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec and spec.loader:
                spec.loader = Loader(spec.loader)
            return spec

    sys.meta_path.insert(0, Finder())


def install_from_env():
    root = os.environ.get("LDB_EVENT_DIR")
    if root:
        leaders = [int(v) for v in os.environ.get("LDB_EVENT_LEADER_TP_RANKS", "0").split(",")]
        Path(root).mkdir(parents=True, exist_ok=True)
        install(root, leaders)
