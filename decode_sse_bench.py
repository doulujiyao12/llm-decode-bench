#!/usr/bin/env python3
"""Standalone prefix-cache-aware SSE benchmark (Python 3.10+).

Copy ONLY this file. Dependencies: pip install aiohttp transformers
No server changes, GPU profiler, package imports, cache flush or global abort.

Modes: 1=uniform deterministic, 2=mixed deterministic, 3=mixed fresh random,
4=unaltered real text/messages selected from --shadow-dir (JSON/JSONL).
Modes 1-3 use token-ID prompts: /generate or /v1/completions, NOT chat-only APIs.
Mode 4 also supports /v1/chat/completions with a matching local chat template.

Capacity is per ATTENTION DP rank, not the sum over TP GPUs. Default: read
SGLang /server_info; otherwise require --max-total-kvcache-length-k (K=1024).
Default over-capacity behavior: exit BEFORE warmup. --allow-shared-prefix
opts synthetic cases into shared prefixes, retaining unique suffixes/lengths.
It never rewrites semantic shadow traffic. Reserve includes full output budget.

Example (no change to already-running server):
  python3 decode_sse_bench.py --base-url http://127.0.0.1:31000 --model hy4 \
    --tokenizer /data/models/hy4 --dp-size 8 --batch-sizes 16 32 --mode 2 \
    --input-lengths-k 32 64 100 --max-total-kvcache-length-k 3500 --plot

SSE gap != token TPOT != GPU scheduler step. No MTP division is applied.
Cache/pinning/metrics not reported by the server cannot be invented. Without
per-rank telemetry results are CLIENT-ONLY, even with all streams active.
Do not use an ordinary /generate call directly against a PD decode-only node;
use an existing complete PD/router entrypoint (which may not support pinning).
Prefix caching must already be enabled. Outputs/seeds aren't guaranteed bitwise
deterministic across servers. Synthetic tokens are NOT quality/MTP datasets.
Run --self-test for built-in unit and localhost HTTP integration tests.
"""
from __future__ import annotations

import argparse
import asyncio
from array import array
import codecs
import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import html
import gzip
import json
import math
import os
from pathlib import Path
import random
import re
import secrets
import statistics
import sys
import time
from urllib.parse import urlsplit


class BenchError(RuntimeError):
    pass


def log(message):
    print(f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {message}", flush=True)


def check(condition, message):
    if not condition:
        raise BenchError(message)


def rounded(n, page):
    return ((n + page - 1) // page) * page


def digest_ids(ids):
    a = array('I', ids)
    if sys.byteorder != 'little':
        a.byteswap()
    return hashlib.sha256(a.tobytes()).hexdigest()


def quantile(values, p):
    values = sorted(values)
    if not values:
        return None
    idx = (len(values) - 1) * p
    lo = int(idx)
    return values[lo] + (values[min(lo + 1, len(values) - 1)] - values[lo]) * (idx - lo)


def mixed_lengths(bs, mean, low_ratio, high_ratio, rng):
    """Exact mean at each rank, with deterministic bounded integer transfers."""
    low, high = max(1, math.ceil(mean * low_ratio)), math.floor(mean * high_ratio)
    check(low <= mean <= high, 'Length range must contain the requested mean')
    values = [mean] * bs
    if bs > 1:
        for _ in range(bs * 24):
            i, j = rng.sample(range(bs), 2)
            lo = max(low - values[i], values[j] - high)
            hi = min(high - values[i], values[j] - low)
            delta = rng.randint(lo, hi)
            values[i] += delta
            values[j] -= delta
    return values


def rank_cost(lengths, shared, output_tokens, page):
    # Common prefix is a nested family, all split points page-aligned. This
    # conservatively rounds each independent suffix and output separately.
    return (rounded(max(shared, default=0), page)
            + sum(rounded(n - s, page) for n, s in zip(lengths, shared))
            + len(lengths) * rounded(output_tokens, page))


def plan_sharing(lengths, budget, output_tokens, page, allow, min_suffix):
    none = [0] * len(lengths)
    independent = rank_cost(lengths, none, output_tokens, page)
    if independent <= budget:
        return none, independent
    check(allow, f'Independent KV needs {independent:,} tokens/rank, usable budget '
                 f'{budget:,}. Reduce BS/context/output or explicitly add --allow-shared-prefix.')
    max_shared = [max(0, n - min_suffix) // page * page for n in lengths]
    check(rank_cost(lengths, max_shared, output_tokens, page) <= budget,
          'Even maximal shared prefixes + unique suffixes + output budget do not fit. '
          'Reduce BS/context/output; no requests were sent for this case.')
    # Nested common prefixes work for mixed lengths too. Rounding is monotone:
    # each step removes whole pages from suffixes; max(prefix) grows at most as fast.
    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = (lo + hi) / 2
        shares = [int(s * mid) // page * page for s in max_shared]
        if rank_cost(lengths, shares, output_tokens, page) <= budget:
            hi = mid
        else:
            lo = mid
    shares = [int(s * hi) // page * page for s in max_shared]
    if rank_cost(lengths, shares, output_tokens, page) > budget:
        shares = max_shared
    return shares, rank_cost(lengths, shares, output_tokens, page)


def natural_prefix_cost(sequences, output_tokens, page):
    """Conservative prefix-trie page union; no dedup of equal interior content."""
    previous = ()
    total = 0
    for seq in sorted(sequences):
        common = 0
        for a, b in zip(seq, previous):
            if a != b:
                break
            common += 1
        total += rounded(len(seq), page) - (common // page * page)
        previous = seq
    return total + len(sequences) * rounded(output_tokens, page)


@dataclass
class Prompt:
    index: int
    rank: int
    ids: array
    seed: int
    shared_tokens: int = 0
    messages: list | None = None

    def manifest(self):
        return dict(index=self.index, planned_rank=self.rank, input_tokens=len(self.ids),
                    sha256=digest_ids(self.ids), generation_seed=self.seed,
                    synthetic_shared_prefix_tokens=self.shared_tokens)


def capacity_from_info(info, dp_size):
    """Only recognized token-pool fields; never bytes/free VRAM heuristics."""
    if not isinstance(info, dict):
        return None
    if info.get('dp_size') is not None:
        check(info['dp_size'] == dp_size, 'Service dp_size differs from --dp-size')
    # Newer SGLang reports every scheduler's capacity. Use the minimum, not sum.
    states = info.get('internal_states') or []
    if not isinstance(states, list):
        states = []
    capacities = []
    for state in states:
        if not isinstance(state, dict):
            continue
        memory = state.get('memory_usage') or {}
        if not isinstance(memory, dict):
            continue
        value = memory.get('token_capacity')
        if type(value) is int and value > 0:
            capacities.append(value)
    if len(capacities) >= dp_size:
        return min(capacities), 'server_info.internal_states.memory_usage.token_capacity (minimum)'
    value = info.get('max_total_num_tokens')
    if type(value) is int and value > 0 and (dp_size == 1 or info.get('dp_size') == dp_size):
        return value, 'server_info.max_total_num_tokens (per-scheduler startup limit)'
    return None


def shadow_messages(row):
    """No truncation, repetition or synthetic padding of semantic traffic."""
    if not isinstance(row, dict):
        return None
    for wrapper in ('request', 'body'):
        if isinstance(row.get(wrapper), dict):
            row = row[wrapper]
    messages = row.get('messages')
    if messages is None and isinstance(row.get('conversations'), list):
        roles = {'human': 'user', 'gpt': 'assistant', 'system': 'system'}
        messages = [dict(role=roles.get(m.get('from')), content=m.get('value'))
                    for m in row['conversations'] if isinstance(m, dict)]
        if len(messages) != len(row['conversations']):
            return None
    if messages is None:
        text = row.get('text', row.get('prompt'))
        if isinstance(text, str) and text:
            return [dict(role='user', content=text)]
        return None
    if not isinstance(messages, list) or not messages:
        return None
    # Reject tool/image/multimodal records rather than silently changing them.
    if any(not isinstance(m, dict) or set(m) - {'role', 'content'}
           or m.get('role') not in {'system', 'user', 'assistant'}
           or not isinstance(m.get('content'), str) for m in messages):
        return None
    if row.get('tools') or row.get('images') or row.get('image_data'):
        return None
    return messages


def load_shadow(folder, tokenizer):
    candidates, seen, skipped = [], set(), 0
    root = Path(folder)
    supported = lambda path: path.name.endswith(('.json', '.jsonl', '.json.gz', '.jsonl.gz'))
    files = [root] if root.is_file() and supported(root) else sorted(
        p for p in root.rglob('*') if p.is_file() and supported(p) and 'manifest' not in p.name)
    check(files, 'No .json/.jsonl/.json.gz/.jsonl.gz shadow files found; '
          'copy the shadow folder beside this script or pass --shadow-dir FILE_OR_FOLDER')
    for path in files:
        opener = gzip.open if path.suffix == '.gz' else open
        with opener(path, 'rt', encoding='utf-8') as f:
            is_lines = path.name.endswith(('.jsonl', '.jsonl.gz'))
            rows = (json.loads(line) for line in f if line.strip()) if is_lines else json.load(f)
            if isinstance(rows, dict):
                rows = rows.get('data', [rows])
            for row in rows:
                msgs = shadow_messages(row)
                if msgs is None:
                    skipped += 1
                    continue
                ids = array('I', tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True))
                key = digest_ids(ids)
                if not ids or key in seen:
                    skipped += 1
                    continue
                seen.add(key)
                candidates.append((ids, msgs))
    lengths = [len(ids) for ids, _ in candidates]
    log(f'Shadow: {len(candidates)} unique text/message prompts; skipped {skipped} unsupported/duplicate records'
        + (f'; target-tokenizer input min/mean/max={min(lengths)}/{statistics.mean(lengths):.1f}/{max(lengths)}'
           if lengths else ''))
    return candidates


def select_shadow(candidates, dp, bs, mean, tolerance, rng, low_ratio, high_ratio):
    usable = [x for x in candidates if mean * low_ratio <= len(x[0]) <= mean * high_ratio]
    check(len(usable) >= dp * bs, 'Not enough distinct semantic prompts within length bounds')
    available = list(range(len(usable)))
    selected = []
    for _rank in range(dp):
        target = bs * mean
        best, error = None, float('inf')
        # Random restarts + substitutions. Never alter selected prompt content.
        for _ in range(12):
            chosen = rng.sample(available, bs)
            total = sum(len(usable[i][0]) for i in chosen)
            for _ in range(2000):
                if abs(total - target) <= target * tolerance:
                    break
                pos, new = rng.randrange(bs), rng.choice(available)
                if new in chosen:
                    continue
                alt = total - len(usable[chosen[pos]][0]) + len(usable[new][0])
                if abs(alt - target) < abs(total - target):
                    chosen[pos], total = new, alt
            if abs(total - target) < error:
                best, error = chosen[:], abs(total - target)
            if error <= target * tolerance:
                break
        check(error <= target * tolerance,
              'Shadow cohort cannot reach requested per-rank mean within tolerance; '
              'add real samples, change mean/bounds/tolerance. Content is not padded/truncated.')
        selected.append([usable[i] for i in best])
        chosen_set = set(best)
        available = [i for i in available if i not in chosen_set]
    return selected


def make_case(args, tokenizer, capacity, page, bs, mean, run_seed, shadows=None):
    budget = math.floor(capacity * args.capacity_fraction)
    seed = f'{run_seed}:mode{args.mode}:bs{bs}:mean{mean}'
    rng = random.Random(seed)
    by_rank, costs, plans = [], [], []
    if args.mode == 4:
        cohorts = select_shadow(shadows, args.dp_size, bs, mean, args.shadow_mean_tolerance,
                                rng, args.min_length_ratio, args.max_length_ratio)
    else:
        vocab = sorted(set(tokenizer.get_vocab().values()) - set(tokenizer.all_special_ids))
        check(len(vocab) >= 2, 'Tokenizer has fewer than 2 non-special token IDs')
    for rank in range(args.dp_size):
        if args.mode == 4:
            cohort = cohorts[rank]
            lengths = [len(x[0]) for x in cohort]
            cost = natural_prefix_cost([x[0] for x in cohort], args.output_tokens, page)
            check(cost <= budget, f'Shadow rank {rank}: {cost:,} KV tokens > usable {budget:,}; '
                  'artificial prefix sharing would change semantics and is not applied.')
            shared = [0] * bs
        else:
            lengths = ([mean] * bs if args.mode == 1 else mixed_lengths(
                bs, mean, args.min_length_ratio, args.max_length_ratio, random.Random(f'{seed}:lengths:{rank}')))
            shared, cost = plan_sharing(lengths, budget, args.output_tokens, page,
                                       args.allow_shared_prefix, args.min_unique_suffix_tokens)
            prefix_rng = random.Random(f'{seed}:common:{rank}')
            common = array('I', prefix_rng.choices(vocab, k=max(shared)))
        rank_prompts = []
        for local, length in enumerate(lengths):
            index = local * args.dp_size + rank
            if args.mode == 4:
                ids, msgs = cohort[local]
            else:
                prng = random.Random(f'{seed}:input:{rank}:{local}')
                ids = common[:shared[local]]
                ids.extend(prng.choices(vocab, k=length - shared[local]))
                msgs = None
            gseed = (args.generation_seed + index if args.mode in (1, 2)
                     else rng.randrange(0, 2**31 - 1))
            rank_prompts.append(Prompt(index, rank, ids, gseed, shared[local], msgs))
        by_rank.extend(rank_prompts)
        costs.append(cost)
        plans.append(dict(rank=rank, min=min(lengths), max=max(lengths), mean=statistics.mean(lengths),
                          sum_input_tokens=sum(lengths), estimated_kv_tokens=cost,
                          max_synthetic_shared_tokens=max(shared)))
    check(len({digest_ids(p.ids) for p in by_rank}) == len(by_rank),
          'Duplicate prompts found; increase input/suffix lengths or vocabulary')
    return sorted(by_rank, key=lambda p: p.index), dict(
        mode=args.mode, bs_per_dp=bs, dp_size=args.dp_size, mean_input_tokens_requested=mean,
        run_seed=run_seed, case_seed=seed, capacity_tokens_per_rank=capacity,
        usable_capacity_tokens_per_rank=budget, output_reserve_tokens_per_request=args.output_tokens,
        page_size=page, rank_plans=plans,
        prefix_workload=('capacity_shared_synthetic' if any(p.shared_tokens for p in by_rank)
                         else 'natural_semantic' if args.mode == 4 else 'independent_synthetic'))


async def sse_frames(chunks):
    decoder = codecs.getincrementaldecoder('utf-8-sig')('strict')
    buf, data = '', []
    size = 0
    async for chunk in chunks:
        buf += decoder.decode(chunk)
        while True:
            ends = [p for p in (buf.find('\n'), buf.find('\r')) if p >= 0]
            if not ends:
                break
            i = min(ends)
            if buf[i] == '\r' and i == len(buf) - 1:
                break
            width = 2 if buf[i:i + 2] == '\r\n' else 1
            line, buf = buf[:i], buf[i + width:]
            if not line:
                if data:
                    yield '\n'.join(data)
                data, size = [], 0
            elif line.startswith('data:'):
                value = line[5:]
                data.append(value[1:] if value.startswith(' ') else value)
                size += len(line)
                check(size < 16 * 1024 * 1024, 'Oversized SSE frame')
        check(size + len(buf) < 16 * 1024 * 1024, 'Oversized SSE frame')
    buf += decoder.decode(b'', final=True)
    if buf == '\r':
        if data:
            yield '\n'.join(data)
        data, buf = [], ''
    check(not buf.strip() and not data, 'Truncated SSE frame at EOF')


@dataclass
class Record:
    prompt: Prompt
    events: list = field(default_factory=list)
    prompt_tokens: int | None = None
    cached_tokens: int | None = None
    actual_rank: int | None = None
    tokens: int = 0
    finished: bool = False
    done: bool = False
    end: float | None = None
    error: str | None = None
    cancelled: bool = False

    def consume(self, raw, backend):
        if raw.strip() == '[DONE]':
            self.done = True
            return
        obj = json.loads(raw)
        check(isinstance(obj, dict) and not obj.get('error'), 'SSE error event (body omitted)')
        if backend == 'sglang':
            meta = obj.get('meta_info') or {}
            count = meta.get('completion_tokens', self.tokens)
            check(type(count) is int and count >= self.tokens, 'Non-monotonic completion count')
            progress = count > self.tokens
            self.tokens = count
            self.update_counts(meta.get('prompt_tokens'), meta.get('cached_tokens'), meta.get('dp_rank'))
            finish = meta.get('finish_reason')
            if finish is not None:
                self.finished = True
                reason = finish.get('type') if isinstance(finish, dict) else finish
                check(reason not in {'abort', 'error', 'cancelled', 'canceled'}, 'Server aborted request')
        else:
            usage = obj.get('usage') or {}
            details = usage.get('prompt_tokens_details') or {}
            self.update_counts(usage.get('prompt_tokens'), details.get('cached_tokens'), None)
            if usage.get('completion_tokens') is not None:
                count = usage['completion_tokens']
                check(type(count) is int and count >= self.tokens, 'Invalid usage completion count')
                self.tokens = count
            choices = obj.get('choices') or []
            check(len(choices) <= 1, 'Only n=1 responses supported')
            progress = False
            for choice in choices:
                delta = choice.get('delta') or {}
                progress |= bool(choice.get('text') or delta.get('content') or delta.get('reasoning_content')
                                 or delta.get('reasoning') or delta.get('tool_calls'))
                if choice.get('finish_reason') is not None:
                    self.finished = True
                    check(choice['finish_reason'] not in {'error', 'cancelled', 'canceled', 'content_filter'},
                          'Abnormal completion finish')
        if progress:
            self.events.append(time.monotonic())

    def update_counts(self, prompt, cached, rank):
        for name, value in [('prompt_tokens', prompt), ('cached_tokens', cached), ('actual_rank', rank)]:
            if value is not None:
                check(type(value) is int and value >= 0, f'Invalid {name} metadata')
                setattr(self, name, value)


def cache_status(record, args):
    if record.prompt_tokens is not None:
        check(record.prompt_tokens == len(record.prompt.ids),
              f'Request {record.prompt.index}: server prompt token count differs from local tokenizer; '
              'chat template/tokenization mismatch, cannot trust capacity or target lengths')
    if record.actual_rank is not None and args.route:
        check(record.actual_rank == record.prompt.rank, 'Server did not honor requested DP rank')
    if record.cached_tokens is None:
        check(args.allow_unverified_cache, 'Service did not report cached_tokens. Cannot verify prefix cache; '
              'use a supported endpoint or explicitly --allow-unverified-cache (unverified results).')
        return 'unknown'
    n = len(record.prompt.ids)
    check(0 <= record.cached_tokens <= n, 'Invalid cached token count')
    miss = n - record.cached_tokens
    check(miss <= args.cache_tail_tokens and record.cached_tokens / n >= args.min_cache_hit_ratio,
          f'Request {record.prompt.index}: only {record.cached_tokens}/{n} input tokens cached; '
          'cache disabled/evicted/routed elsewhere. This is not a warm-prefix decode test.')
    return 'reported_hit'


def payload(args, prompt, output, stream):
    if args.backend == 'sglang':
        sampling = dict(max_new_tokens=output, temperature=args.temperature,
                        top_p=args.top_p, sampling_seed=prompt.seed)
        if args.ignore_eos:
            sampling['ignore_eos'] = True
        body = dict(input_ids=list(prompt.ids), sampling_params=sampling, stream=stream,
                    rid=f'ssebench_{args.run_id}_{prompt.index}_{secrets.token_hex(4)}')
        if args.route:
            body['routed_dp_rank'] = prompt.rank
        return '/generate', body
    body = dict(model=args.model, max_tokens=output, temperature=args.temperature,
                top_p=args.top_p, seed=prompt.seed, stream=stream, n=1)
    if stream:
        body['stream_options'] = {'include_usage': True}
    if args.ignore_eos:
        body['ignore_eos'] = True  # explicit opt-in, not standard OpenAI
    if args.backend == 'openai-chat':
        check(prompt.messages is not None, 'Chat mode requires semantic messages (mode 4)')
        body['messages'] = prompt.messages
        return '/v1/chat/completions', body
    body['prompt'] = list(prompt.ids)
    return '/v1/completions', body


async def request(session, args, record, output):
    path, body = payload(args, record.prompt, output, True)
    try:
        async with session.post(args.base_url + path, json=body,
                                timeout=args.request_timeout) as response:
            check(response.status == 200, f'HTTP {response.status} (response body omitted)')
            check('text/event-stream' in response.headers.get('Content-Type', ''), 'Not an SSE response')
            async for raw in sse_frames(response.content.iter_any()):
                record.consume(raw, args.backend)
                if record.done:
                    break
            check(record.done or record.finished, 'Stream ended without terminal event')
            check(record.events, 'No generation progress received')
    except asyncio.CancelledError:
        record.cancelled = True
        raise
    except Exception as exc:
        # Do not leak request/response contents, credentials, proxy URLs or text.
        record.error = str(exc) if isinstance(exc, BenchError) else type(exc).__name__
    finally:
        record.end = time.monotonic()


async def warm_phase(session, args, prompts, verify):
    semaphore = asyncio.Semaphore(args.warmup_concurrency)
    async def one(prompt):
        async with semaphore:
            record = Record(prompt)
            await request(session, args, record, 1)
            check(not record.error, f'Warmup/probe failed: {record.error}')
            status = cache_status(record, args) if verify else None
            # Validate lengths and routing even during first, cold warmup.
            if record.prompt_tokens is not None:
                check(record.prompt_tokens == len(prompt.ids), 'Warmup tokenizer length mismatch')
            if args.route and record.actual_rank is not None:
                check(record.actual_rank == prompt.rank, 'Warmup DP routing mismatch')
            return dict(index=prompt.index, cached_tokens=record.cached_tokens, status=status)
    tasks = [asyncio.create_task(one(p)) for p in prompts]
    try:
        return await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


METRIC_LINE = re.compile(r'^([\w:]+)(?:\{(.*)\})?\s+([^\s]+)(?:\s+\S+)?$')
METRIC_LABEL = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"(?:,\s*|$)')


def parse_metrics(text, args):
    names = {args.running_metric: 'running', args.waiting_metric: 'waiting'}
    if args.pd_metrics:
        names.update({'sglang:num_decode_prealloc_queue_reqs': 'prealloc',
                      'sglang:num_decode_transfer_queue_reqs': 'transfer'})
    result = {}
    for line in text.splitlines():
        m = METRIC_LINE.match(line)
        if not m or m[1] not in names:
            continue
        labels, end = {}, 0
        raw = m[2] or ''
        for label in METRIC_LABEL.finditer(raw):
            check(label.start() == end, 'Malformed metric labels')
            labels[label[1]] = json.loads('"' + label[2] + '"')
            end = label.end()
        check(end == len(raw), 'Malformed metric labels')
        if any(labels.get(k) != v for k, v in args.metric_selectors.items()):
            continue
        if args.rank_label:
            check(args.rank_label in labels, 'Metric missing DP rank label')
            rank = int(labels[args.rank_label])
        else:
            check(args.dp_size == 1, 'Global running metric cannot verify multiple DP ranks')
            rank = 0
        check(0 <= rank < args.dp_size, 'Metric DP rank outside configured range')
        value = float(m[3])
        check(math.isfinite(value) and value >= 0, 'Invalid metric value')
        fields = result.setdefault(rank, {})
        check(names[m[1]] not in fields, 'Duplicate metrics per rank; use --metric-label key=value')
        fields[names[m[1]]] = value
    check(set(result) == set(range(args.dp_size)), 'Not all DP ranks exposed in metrics')
    check(all(set(fields) == set(names.values()) for fields in result.values()), 'Missing queue/running metric')
    return result


def metrics_ready(ranks, bs):
    return all(fields['running'] == bs and all(v == 0 for k, v in fields.items() if k != 'running')
               for fields in ranks.values())


async def read_metrics(session, args):
    start = time.monotonic()
    async with session.get(args.metrics_url, timeout=3) as response:
        check(response.status == 200, f'Metrics HTTP {response.status}')
        ranks = parse_metrics(await response.text(), args)
    return dict(t=time.monotonic(), latency=time.monotonic() - start, ranks=ranks)


def summarize(records, begin, end):
    per_request, all_gaps = [], []
    for r in records:
        times = [t for t in r.events if begin <= t <= end]
        gaps = [(b - a) * 1000 for a, b in zip(times, times[1:])]
        check(len(gaps) >= 3, f'Request {r.prompt.index} has too few SSE gaps in common window')
        all_gaps.extend(gaps)
        per_request.append(dict(index=r.prompt.index, planned_rank=r.prompt.rank,
                                actual_rank=r.actual_rank, events=len(times),
                                mean_ms=statistics.mean(gaps), median_ms=statistics.median(gaps),
                                p99_ms=quantile(gaps, .99)))
    return dict(window_seconds=end - begin, request_count=len(records),
                sse_gap_mean_ms=statistics.mean([r['mean_ms'] for r in per_request]),
                sse_gap_median_ms=statistics.median([r['median_ms'] for r in per_request]),
                pooled_sse_gap_p99_ms=quantile(all_gaps, .99), per_request=per_request,
                metric='client SSE progress interval; not tokens, GPU steps or MTP-normalized TPOT')


async def measure(session, args, prompts, bs, use_metrics):
    records = [Record(p) for p in prompts]
    tasks = [asyncio.create_task(request(session, args, r, args.output_tokens)) for r in records]
    deadline = time.monotonic() + args.wait_timeout
    ready_since = begin = None
    samples = []
    last_report, last_metric = 0.0, None
    try:
        while True:
            now = time.monotonic()
            errors = [r.error for r in records if r.error]
            check(not errors, f'Measured request failed: {errors[0] if errors else ""}')
            # Finishing early invalidates the common window, no survivor bias.
            check(not any(r.end is not None for r in records),
                  'A request finished before the common steady window ended. Increase --output-tokens '
                  '(and capacity), or --ignore-eos if supported; no partial cohort is reported.')
            check(now <= deadline, 'Stable decode timeout; insufficient running BS/cache/capacity/output budget')
            client_ready = all(len(r.events) >= args.min_events and now - r.events[-1] <= args.max_event_stall
                               for r in records)
            metric_ok = True
            if use_metrics:
                try:
                    sample = await read_metrics(session, args)
                    samples.append(sample)
                    metric_ok = (sample['latency'] <= args.max_metric_gap
                                 and (last_metric is None or sample['t'] - last_metric <= args.max_metric_gap)
                                 and metrics_ready(sample['ranks'], bs))
                    last_metric = sample['t']
                except Exception as exc:
                    metric_ok = False
                    samples.append(dict(t=time.monotonic(), error=type(exc).__name__))
                    last_metric = None
            now = time.monotonic()
            if client_ready and metric_ok:
                if ready_since is None:
                    ready_since = now
                if begin is None and now - ready_since >= args.settle_seconds:
                    begin = now
                    log(f'Stable window started: {args.measure_seconds:g}s; '
                        f'{"per-rank metrics checked" if use_metrics else "CLIENT ONLY, per-rank BS unverified"}')
                if begin is not None and now - begin >= args.measure_seconds:
                    end = now
                    break
            else:
                ready_since = begin = None
            if now - last_report >= 15:
                log(f'Waiting/measuring: progressed {sum(len(r.events) >= args.min_events for r in records)}/{len(records)}, '
                    f'metrics_ready={metric_ok if use_metrics else "unverified"}')
                last_report = now
            await asyncio.sleep(args.poll_interval)
        # Requests may have completed while reading the bracketing metrics sample.
        check(all(r.end is None or r.end >= end for r in records), 'Cohort ended inside measurement window')
        if args.stop_after_measure:
            log('Closing only this benchmark\'s HTTP streams; generic servers may continue computing after disconnect.')
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        else:
            log('Steady window captured; draining outputs to validate final cache usage (no new requests).')
            await asyncio.gather(*tasks)
        check(not any(r.error for r in records), 'A request failed after measurement; case invalid')
        statuses = [cache_status(r, args) for r in records]
        result = summarize(records, begin, end)
        result.update(
            bs_verification='sampled_per_rank_metrics' if use_metrics else 'client_only_unverified',
            dp_routing=('reported_rank_verified_all' if args.route and all(r.actual_rank is not None for r in records)
                        else 'requested_native_unverified' if args.route else 'service_assigned_unverified'),
            cache_verification='reported_hit_all_requests' if all(s == 'reported_hit' for s in statuses) else 'unverified',
            cache_scope='server reported cached tokens; GPU residency/tier not proven by this field alone',
            metrics_samples=samples,
            requests=[dict(**r.prompt.manifest(), cached_tokens=r.cached_tokens,
                           actual_prompt_tokens=r.prompt_tokens,
                           actual_rank=r.actual_rank, completion_tokens=r.tokens,
                           cancelled=r.cancelled, sse_times_relative_s=[t - begin for t in r.events]) for r in records])
        return result
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


def plot_svg(results, path):
    good = [r for r in results if r.get('status') == 'ok']
    if not good:
        return
    w, h, pad = 900, 500, 70
    xs = [r['mean_input_tokens_requested'] / 1024 for r in good]
    ys = [r['measurement']['sse_gap_median_ms'] for r in good]
    xmax, ymax = max(xs) or 1, max(ys) * 1.15 or 1
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
           '<rect width="100%" height="100%" fill="white"/>',
           '<text x="70" y="25" font-family="sans-serif" font-size="17">Steady SSE gap (not GPU step / token TPOT)</text>']
    for i in range(6):
        y = h - pad - (h - 2 * pad) * i / 5
        out.append(f'<path d="M {pad} {y} H {w-pad}" stroke="#ddd"/>')
        out.append(f'<text x="15" y="{y+4}" font-size="12">{ymax*i/5:.1f}</text>')
    colors = ['#1261a0', '#df7130', '#17835f', '#9b4db2', '#a63341']
    for j, bs in enumerate(sorted({r['bs_per_dp'] for r in good})):
        group = sorted((r for r in good if r['bs_per_dp'] == bs), key=lambda r: r['mean_input_tokens_requested'])
        pts = [(pad + (w - 2 * pad) * r['mean_input_tokens_requested'] / 1024 / xmax,
                h - pad - (h - 2 * pad) * r['measurement']['sse_gap_median_ms'] / ymax) for r in group]
        color = colors[j % len(colors)]
        out.append(f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x,y in pts)}" fill="none" stroke="{color}" stroke-width="2"/>')
        for (x, y), r in zip(pts, group):
            label = html.escape(f"BS{bs}, {r['prefix_workload']}, {r['measurement']['bs_verification']}, "
                                f"{r['measurement']['sse_gap_median_ms']:.3f} ms")
            out.append(f'<circle cx="{x}" cy="{y}" r="4" fill="{color}"><title>{label}</title></circle>')
        out.append(f'<text x="{pad+j*140}" y="{h-12}" fill="{color}" font-size="13">BS/rank={bs}</text>')
    for x in sorted(set(xs)):
        px = pad + (w - 2 * pad) * x / xmax
        out.append(f'<text x="{px-12}" y="{h-pad+20}" font-size="12">{x:g}K</text>')
    out += ['<text x="14" y="55" font-size="12">ms</text>', '</svg>']
    path.write_text('\n'.join(out), encoding='utf-8')


async def get_info(session, base):
    for endpoint in ('/server_info', '/get_server_info'):
        try:
            async with session.get(base + endpoint, timeout=5) as response:
                if response.status == 200:
                    obj = await response.json()
                    if isinstance(obj, dict):
                        return obj
        except Exception:
            pass
    return {}


def resolve_capacity(args, info):
    # This benchmark models full-resident, prefix-cached KV. HiSparse GPU hot
    # token capacity and host logical capacity are different and not interchangeable.
    check(not info.get('enable_hisparse') and not info.get('hisparse_config'),
          'Detected HiSparse: full-prefix GPU-residency model is inapplicable. Use a HiSparse-specific benchmark.')
    check(not info.get('disable_radix_cache') and not info.get('disable_prefix_caching'),
          'Server explicitly reports prefix cache disabled; cannot run a cache-warm benchmark')
    automatic = capacity_from_info(info, args.dp_size)
    manual = args.kv_capacity_tokens or (round(args.max_total_kvcache_length_k * 1024)
                                        if args.max_total_kvcache_length_k else None)
    if manual:
        # A user cap may intentionally be lower; never exceed the detected physical limit.
        if automatic and manual > automatic[0]:
            log(f'Manual cap {manual:,} exceeds detected {automatic[0]:,}; using detected lower cap')
            return automatic[0], automatic[1] + ' (manual cap clamped)'
        return manual, 'manual per-rank token limit'
    check(automatic is not None, 'Cannot detect reliable per-rank KV token capacity. Supply '
          '--max-total-kvcache-length-k (K=1024) or --kv-capacity-tokens. No VRAM guess/OOM probing is used.')
    return automatic


async def run(args):
    import aiohttp
    args.base_url = args.base_url.rstrip('/')
    if args.base_url.endswith('/v1'):
        args.base_url = args.base_url[:-3]
    url = urlsplit(args.base_url)
    check(url.scheme in {'http', 'https'} and url.netloc and not url.username and not url.password
          and not url.query and not url.fragment, 'Use an http(s) base URL without embedded credentials/query')
    key = os.environ.get(args.api_key_env)
    headers = {'Authorization': f'Bearer {key}'} if key else {}
    async with aiohttp.ClientSession(headers=headers, connector=aiohttp.TCPConnector(limit=0),
                                     trust_env=False) as session:
        info = await get_info(session, args.base_url)
        if args.backend == 'auto':
            args.backend = 'sglang' if 'max_total_num_tokens' in info or 'internal_states' in info else 'openai'
        check(not (args.backend == 'openai-chat' and args.mode != 4),
              'Exact synthetic token prompts require /generate or /v1/completions. '
              'Chat-only APIs are supported with --mode 4 and the matching chat-template tokenizer.')
        args.route = args.backend == 'sglang' and not args.no_route_dp
        if not args.route and args.dp_size > 1:
            log('WARNING: no DP pinning API; per-rank input distribution/cache affinity cannot be guaranteed.')
        capacity, source = resolve_capacity(args, info)
        page = args.page_size or info.get('page_size') or 256
        check(type(page) is int and page > 0, 'Invalid page size')
        log(f'KV capacity/rank {capacity:,}; usable {int(capacity*args.capacity_fraction):,}; '
            f'source={source}; page={page} tokens')
        log('Capacity preflight is a pool-budget estimate, not an OOM guarantee: activations, '
            'other traffic, cache eviction and backend-specific transient allocations still matter.')
        if not args.tokenizer:
            args.tokenizer = info.get('tokenizer_path') or info.get('model_path')
        check(args.tokenizer, 'Supply --tokenizer matching the serving model; no guessed vocabulary')
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=False)
        shadows = load_shadow(args.shadow_dir, tokenizer) if args.mode == 4 else None
        args.metrics_url = args.metrics_url or args.base_url + '/metrics'
        metrics_url = urlsplit(args.metrics_url)
        check(metrics_url.scheme in {'http', 'https'} and metrics_url.netloc and not metrics_url.username
              and not metrics_url.password and not metrics_url.query and not metrics_url.fragment,
              'Metrics URL must not contain embedded credentials/query/fragment')
        # Do not forward the API key to a second host accidentally.
        check(not key or (metrics_url.scheme, metrics_url.netloc) == (url.scheme, url.netloc),
              'Authenticated metrics must use the same origin as base-url; separate-host credentials '
              'are not forwarded by this single-client benchmark')
        use_metrics = args.steady_check != 'client'
        if use_metrics:
            try:
                await read_metrics(session, args)
            except Exception as exc:
                check(args.steady_check != 'required', f'Cannot verify DP metrics: {type(exc).__name__}')
                use_metrics = False
                log('WARNING: metrics unavailable/ambiguous; CLIENT-ONLY results. Use --steady-check required '
                    'to refuse this, or configure metric names/labels.')
        run_seed = (args.replay_seed if args.replay_seed is not None else args.seed if args.mode in (1, 2)
                    else secrets.randbits(63))
        args.run_id = secrets.token_hex(4)
        out = Path(args.save_path) / (datetime.now(timezone.utc).strftime('sse_%Y%m%d-%H%M%S_') + args.run_id)
        out.mkdir(parents=True, exist_ok=False)
        log(f'Output: {out}; effective input/length seed={run_seed}')
        # Don't persist the full server_info: it may contain API keys and private paths.
        params = vars(args).copy()
        params['metric_selectors'] = args.metric_selectors
        service_hints = {k: info[k] for k in ('dp_size', 'tp_size', 'page_size', 'context_length',
                         'stream_interval', 'speculative_algorithm', 'speculative_num_steps',
                         'speculative_num_draft_tokens', 'disable_radix_cache') if k in info}
        (out / 'config.json').write_text(json.dumps(dict(arguments=params, capacity_source=source,
            service_hints=service_hints,
            run_seed=run_seed, notes='No generated text or source prompts saved; seeds do not guarantee deterministic outputs'),
            ensure_ascii=False, indent=2), encoding='utf-8')
        results = []
        try:
            for bs in args.batch_sizes:
                for mean_k in args.input_lengths_k:
                    mean = round(mean_k * 1024)
                    case = dict(bs_per_dp=bs, mean_input_tokens_requested=mean, status='failed')
                    results.append(case)
                    prompts, plan = make_case(args, tokenizer, capacity, page, bs, mean, run_seed, shadows)
                    case.update(plan)
                    context = info.get('context_length')
                    if isinstance(context, int):
                        check(all(len(p.ids) + args.output_tokens <= context for p in prompts),
                              'Input+output exceeds service context_length')
                    effective_limits = [s.get('effective_max_running_requests_per_dp')
                                        for s in info.get('internal_states', []) if isinstance(s, dict)]
                    limits = [v for v in effective_limits if type(v) is int]
                    if limits:
                        check(bs <= min(limits), 'Requested BS exceeds service per-DP max running requests')
                    case['prompts'] = [p.manifest() for p in prompts]
                    log(f'Mode {args.mode}, BS/rank={bs}, mean={mean:,}, global={len(prompts)}, '
                        f'prefix={plan["prefix_workload"]}, KV estimate/rank '
                        f'{min(p["estimated_kv_tokens"] for p in plan["rank_plans"]):,}..'
                        f'{max(p["estimated_kv_tokens"] for p in plan["rank_plans"]):,}')
                    if args.plan_only:
                        case['status'] = 'planned_not_run'
                        continue
                    log('Warmup all exact prompts (1 generated token each)')
                    await warm_phase(session, args, prompts, False)
                    log('Probe all exact prompts again; verify cache residency/hits before formal run')
                    case['cache_probe'] = await warm_phase(session, args, prompts, True)
                    case['measurement'] = await measure(session, args, prompts, bs, use_metrics)
                    case['status'] = 'ok'
                    m = case['measurement']
                    log(f'SSE gap mean/median/P99={m["sse_gap_mean_ms"]:.3f}/'
                        f'{m["sse_gap_median_ms"]:.3f}/{m["pooled_sse_gap_p99_ms"]:.3f} ms; '
                        f'{m["bs_verification"]}, {m["cache_verification"]}')
        except Exception as exc:
            if results:
                results[-1]['error'] = str(exc) if isinstance(exc, BenchError) else type(exc).__name__
            raise
        finally:
            (out / 'results.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
            with (out / 'summary.csv').open('w', newline='', encoding='utf-8') as f:
                fields = ['status', 'bs_per_dp', 'mean_input_tokens_requested', 'prefix_workload',
                          'sse_gap_mean_ms', 'sse_gap_median_ms', 'pooled_sse_gap_p99_ms',
                          'bs_verification', 'cache_verification']
                writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
                writer.writeheader()
                for result in results:
                    writer.writerow({**result, **result.get('measurement', {})})
            if args.plot:
                plot_svg(results, out / 'sse_gap.svg')


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--base-url', default='http://127.0.0.1:31000')
    p.add_argument('--model', default='hy4')
    p.add_argument('--tokenizer', help='Matching tokenizer path/HF ID; never trust_remote_code')
    p.add_argument('--backend', choices=['auto', 'sglang', 'openai', 'openai-chat'], default='auto')
    p.add_argument('--mode', type=int, choices=[1, 2, 3, 4], default=1)
    p.add_argument('--dp-size', type=int, default=8, help='Attention-DP ranks, NOT TP size')
    p.add_argument('--batch-sizes', type=int, nargs='+', default=[32], help='Requests per attention-DP rank')
    p.add_argument('--input-lengths-k', type=float, nargs='+', default=[32], help='Per-rank mean input tokens, K=1024')
    p.add_argument('--min-length-ratio', type=float, default=.25)
    p.add_argument('--max-length-ratio', type=float, default=1.75)
    p.add_argument('--seed', type=int, default=60)
    p.add_argument('--generation-seed', type=int, default=60)
    p.add_argument('--replay-seed', type=int, help='Replay recorded run_seed for modes 3/4')
    p.add_argument('--shadow-dir', help='Mode 4: text/messages file or folder; .json/.jsonl and gzip supported; '
                   'default: bundled shadow/basic.jsonl.gz')
    p.add_argument('--shadow-mean-tolerance', type=float, default=.01)
    cap = p.add_mutually_exclusive_group()
    cap.add_argument('--max-total-kvcache-length-k', type=float, help='Manual token capacity PER RANK, K=1024; default auto')
    cap.add_argument('--kv-capacity-tokens', type=int, help='Manual token capacity PER RANK (literal tokens)')
    p.add_argument('--capacity-fraction', type=float, default=.95, help='Use this fraction of reported/manual token pool')
    p.add_argument('--page-size', type=int, help='KV page size in tokens; default server value, else conservative 256')
    p.add_argument('--allow-shared-prefix', action='store_true', help='Opt-in capacity fallback for synthetic modes ONLY')
    p.add_argument('--min-unique-suffix-tokens', type=int, default=128)
    p.add_argument('--output-tokens', '--request-output-length', type=int, default=2048)
    p.add_argument('--temperature', type=float, default=.6)
    p.add_argument('--top-p', type=float, default=.95)
    p.add_argument('--ignore-eos', action='store_true', help='Nonstandard OpenAI extension; only if supported')
    p.add_argument('--warmup-concurrency', type=int, default=8)
    p.add_argument('--min-cache-hit-ratio', type=float, default=.98)
    p.add_argument('--cache-tail-tokens', type=int, default=256, help='Maximum uncached input tail, AND ratio must pass')
    p.add_argument('--allow-unverified-cache', action='store_true', help='Accept MISSING cache counters, never known misses')
    p.add_argument('--no-route-dp', action='store_true', help='Disable native routed_dp_rank; distribution/affinity unverified')
    p.add_argument('--steady-check', choices=['auto', 'required', 'client'], default='auto')
    p.add_argument('--metrics-url')
    p.add_argument('--running-metric', default='sglang:num_running_reqs')
    p.add_argument('--waiting-metric', default='sglang:num_queue_reqs')
    p.add_argument('--rank-label', default='dp_rank', help='Empty string permitted only with DP=1')
    p.add_argument('--metric-label', action='append', default=[], help='Filter duplicate workers, e.g. model_name=hy4')
    p.add_argument('--pd-metrics', action='store_true', help='Also require D prealloc/transfer queues empty')
    p.add_argument('--measure-seconds', type=float, default=10)
    p.add_argument('--settle-seconds', type=float, default=3)
    p.add_argument('--min-events', type=int, default=5)
    p.add_argument('--max-event-stall', type=float, default=5)
    p.add_argument('--poll-interval', type=float, default=.5)
    p.add_argument('--max-metric-gap', type=float, default=3)
    p.add_argument('--wait-timeout', type=float, default=600)
    p.add_argument('--request-timeout', type=float, default=7200)
    p.add_argument('--stop-after-measure', action='store_true', help='Close own streams early; final usage may be unavailable')
    p.add_argument('--api-key-env', default='OPENAI_API_KEY')
    p.add_argument('--save-path', default='results')
    p.add_argument('--plot', action='store_true', help='Generate dependency-free SVG chart')
    p.add_argument('--plan-only', action='store_true', help='Read-only metadata/plan; no generation requests')
    p.add_argument('--self-test', action='store_true', help='Embedded unit + local mock HTTP tests; no GPU/model downloads')
    return p


def validate(args):
    for name in ('dp_size', 'output_tokens', 'warmup_concurrency', 'min_events', 'min_unique_suffix_tokens'):
        check(getattr(args, name) > 0, f'{name} must be positive')
    for name in ('measure_seconds', 'max_event_stall', 'poll_interval', 'max_metric_gap', 'wait_timeout', 'request_timeout'):
        check(math.isfinite(getattr(args, name)) and getattr(args, name) > 0, f'{name} must be positive/finite')
    check(all(bs > 0 for bs in args.batch_sizes), 'BS must be positive')
    check(all(math.isfinite(k) and k * 1024 >= 1 for k in args.input_lengths_k), 'Invalid input lengths')
    check(0 < args.capacity_fraction <= 1 and 0 < args.min_cache_hit_ratio <= 1, 'Invalid capacity/cache fraction')
    check(0 < args.min_length_ratio <= 1 <= args.max_length_ratio < float('inf'), 'Invalid length bounds')
    check(0 <= args.shadow_mean_tolerance < 1 and args.settle_seconds >= 0, 'Invalid tolerance/settle time')
    check(args.cache_tail_tokens >= 0 and args.temperature >= 0 and 0 < args.top_p <= 1, 'Invalid sampling/cache parameters')
    check(math.isfinite(args.temperature) and math.isfinite(args.settle_seconds), 'Non-finite sampling/settle value')
    if args.mode == 4 and not args.shadow_dir:
        args.shadow_dir = str(Path(__file__).resolve().parent / 'shadow' / 'basic.jsonl.gz')
    check(args.mode == 4 or not args.shadow_dir, '--shadow-dir is only for mode 4')
    for name in ('kv_capacity_tokens', 'max_total_kvcache_length_k', 'page_size'):
        value = getattr(args, name)
        check(value is None or (math.isfinite(value) and value > 0), f'{name} must be positive/finite')
    try:
        args.metric_selectors = dict(item.split('=', 1) for item in args.metric_label)
    except ValueError:
        raise BenchError('--metric-label must be key=value') from None


def self_test():
    """Tests travel with the single file; mocks do not require a GPU/tokenizer download."""
    import contextlib
    import io
    import tempfile
    import types
    import unittest
    from unittest.mock import patch
    from aiohttp import web, ClientSession, TCPConnector

    class FakeTokenizer:
        all_special_ids = [0, 1]

        def get_vocab(self):
            return {str(i): i for i in range(128)}

        def apply_chat_template(self, messages, **kwargs):
            # One codepoint per fake token + role boundaries, exact and portable.
            return [1] + [ord(ch) + 2 for m in messages for ch in m['content']] + [1]

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            assert kwargs.get('trust_remote_code') is False
            return cls()

    def config(**changes):
        a = parser().parse_args([])
        a.dp_size = 2
        a.batch_sizes = [2]
        a.input_lengths_k = [.125]
        a.backend = 'sglang'
        a.route = True
        a.run_id = 'test'
        a.tokenizer = 'fake'
        a.output_tokens = 50
        a.measure_seconds = .08
        a.settle_seconds = .02
        a.poll_interval = .01
        a.max_metric_gap = 1
        a.wait_timeout = 4
        a.request_timeout = 4
        a.min_events = 2
        for k, v in changes.items():
            setattr(a, k, v)
        validate(a)
        return a

    class PlanningTests(unittest.TestCase):
        def test_equal_and_mixed_exact_rank_means(self):
            for mode in (1, 2, 3):
                for bs in (1, 8, 16, 17, 32, 64):
                    a = config(mode=mode)
                    p, plan = make_case(a, FakeTokenizer(), 10000000, 16, bs, 100, 60)
                    self.assertEqual(len(p), bs * 2)
                    for rank in range(2):
                        lengths = [len(x.ids) for x in p if x.rank == rank]
                        self.assertEqual(sum(lengths), bs * 100)
                        self.assertTrue(all(25 <= n <= 175 for n in lengths))
                        if mode == 1:
                            self.assertEqual(set(lengths), {100})
                    self.assertEqual(len({digest_ids(x.ids) for x in p}), len(p))

        def test_reproducible_and_seed_changes(self):
            for mode in (1, 2, 3):
                a = config(mode=mode)
                one, _ = make_case(a, FakeTokenizer(), 100000, 16, 8, 128, 60)
                two, _ = make_case(a, FakeTokenizer(), 100000, 16, 8, 128, 60)
                other, _ = make_case(a, FakeTokenizer(), 100000, 16, 8, 128, 61)
                self.assertEqual([p.manifest() for p in one], [p.manifest() for p in two])
                self.assertNotEqual([digest_ids(p.ids) for p in one], [digest_ids(p.ids) for p in other])
                if mode in (1, 2):
                    self.assertEqual([p.seed for p in one], [60 + i for i in range(16)])
                else:
                    self.assertNotEqual([p.seed for p in one], [p.seed for p in other])

        def test_capacity_fails_by_default_before_requests(self):
            a = config()
            with self.assertRaisesRegex(BenchError, 'Independent KV'):
                make_case(a, FakeTokenizer(), 1000, 16, 8, 1000, 60)

        def test_shared_capacity_and_unique_suffix(self):
            a = config(mode=2, allow_shared_prefix=True, output_tokens=128)
            prompts, plan = make_case(a, FakeTokenizer(), 6000, 16, 8, 1000, 60)
            self.assertEqual(plan['prefix_workload'], 'capacity_shared_synthetic')
            for rank in range(2):
                group = [p for p in prompts if p.rank == rank]
                self.assertEqual(sum(len(p.ids) for p in group), 8000)
                self.assertTrue(all(len(p.ids) - p.shared_tokens >= 128 for p in group))
                for x in group:
                    for y in group:
                        n = min(x.shared_tokens, y.shared_tokens)
                        self.assertEqual(x.ids[:n], y.ids[:n])
                cost = natural_prefix_cost([p.ids for p in group], a.output_tokens, 16)
                self.assertLessEqual(cost, 5700)
            self.assertEqual(len({digest_ids(p.ids) for p in prompts}), len(prompts))

        def test_shared_flag_does_not_change_already_fitting_cases(self):
            a = config()
            independent, _ = make_case(a, FakeTokenizer(), 100000, 16, 8, 128, 60)
            a.allow_shared_prefix = True
            enabled, plan = make_case(a, FakeTokenizer(), 100000, 16, 8, 128, 60)
            self.assertEqual([p.manifest() for p in independent], [p.manifest() for p in enabled])
            self.assertEqual(plan['prefix_workload'], 'independent_synthetic')

        def test_max_shared_still_fails(self):
            with self.assertRaisesRegex(BenchError, 'Even maximal'):
                plan_sharing([1000] * 32, 1000, 2048, 16, True, 128)

        def test_page_union_no_interior_dedup(self):
            a = array('I', [1, 2, 3, 4, 7, 8])
            b = array('I', [1, 2, 3, 4, 9, 10])
            self.assertEqual(natural_prefix_cost([a, b], 2, 2), 12)
            c = array('I', [9, 2, 3, 4, 7, 8])
            self.assertEqual(natural_prefix_cost([a, c], 2, 2), 16)

        def test_auto_capacity_min_not_sum_and_manual(self):
            info = {'dp_size': 2, 'max_total_num_tokens': 8000, 'internal_states': [
                {'memory_usage': {'token_capacity': 7000}}, {'memory_usage': {'token_capacity': 6000}}]}
            self.assertEqual(capacity_from_info(info, 2)[0], 6000)
            self.assertEqual(resolve_capacity(config(), info)[0], 6000)
            self.assertEqual(resolve_capacity(config(kv_capacity_tokens=4000), info)[0], 4000)
            self.assertEqual(resolve_capacity(config(kv_capacity_tokens=9000), info)[0], 6000)
            self.assertEqual(resolve_capacity(config(max_total_kvcache_length_k=10), {})[0], 10240)
            with self.assertRaises(BenchError):
                capacity_from_info(info, 8)
            with self.assertRaises(BenchError):
                resolve_capacity(config(), {'free_gpu_memory': 1000000000})

        def test_disabled_cache_and_hisparse_refuse(self):
            for info in ({'disable_radix_cache': True}, {'enable_hisparse': True}):
                with self.assertRaises(BenchError):
                    resolve_capacity(config(kv_capacity_tokens=4000), info)

        def test_shadow_preserves_semantics_and_mean(self):
            candidates = [(array('I', [i] * (100 + i % 2 * 20)), [{'role': 'user', 'content': str(i)}])
                          for i in range(1, 65)]
            a = config(mode=4, shadow_dir='unused', shadow_mean_tolerance=0)
            prompts, plan = make_case(a, FakeTokenizer(), 100000, 16, 8, 110, 60, candidates)
            source_hashes = {digest_ids(x[0]) for x in candidates}
            self.assertTrue(all(digest_ids(p.ids) in source_hashes for p in prompts))
            self.assertTrue(all(r['mean'] == 110 for r in plan['rank_plans']))
            with self.assertRaisesRegex(BenchError, 'semantics'):
                a.allow_shared_prefix = True
                make_case(a, FakeTokenizer(), 100, 16, 8, 110, 60, candidates)

        def test_shadow_unsupported_and_no_repetition(self):
            self.assertIsNone(shadow_messages({'messages': [{'role': 'user', 'content': [{'type': 'image'}]}]}))
            self.assertIsNone(shadow_messages({'messages': [{'role': 'assistant', 'content': 'a', 'tool_calls': []}]}))
            with tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / 'traffic.jsonl'
                path.write_text('\n'.join(json.dumps(x) for x in [{'text': 'hello'}, {'text': 'hello'}, {'text': 'world'}]))
                data = load_shadow(temp, FakeTokenizer())
                self.assertEqual(len(data), 2)
                with self.assertRaisesRegex(BenchError, 'Not enough'):
                    select_shadow(data, 2, 2, 7, .01, random.Random(60), .25, 1.75)

        def test_gzip_shadow_file_and_folder_skip_manifest(self):
            rows = [{'text': 'a real sample'}, {'messages': [{'role': 'user', 'content': 'another example'}]}]
            with tempfile.TemporaryDirectory() as temp:
                directory = Path(temp)
                path = directory / 'basic.jsonl.gz'
                path.write_bytes(gzip.compress(('\n'.join(json.dumps(r) for r in rows) + '\n').encode(), mtime=0))
                (directory / 'manifest.json').write_text('{"source":"metadata, not prompts"}')
                from_file = load_shadow(path, FakeTokenizer())
                from_folder = load_shadow(directory, FakeTokenizer())
                self.assertEqual(len(from_file), 2)
                self.assertEqual(from_file, from_folder)
                json_path = directory / 'array.json.gz'
                json_path.write_bytes(gzip.compress(json.dumps(rows).encode(), mtime=0))
                self.assertEqual(load_shadow(json_path, FakeTokenizer()), from_file)

        def test_mode4_bundled_dataset_default(self):
            a = config(mode=4)
            self.assertTrue(a.shadow_dir.endswith('shadow/basic.jsonl.gz'))

        def test_native_and_openai_seed_placement(self):
            a = config()
            p = Prompt(0, 1, array('I', [4, 5]), 61)
            path, body = payload(a, p, 32, True)
            self.assertEqual(path, '/generate')
            self.assertEqual(body['sampling_params']['sampling_seed'], 61)
            self.assertEqual(body['routed_dp_rank'], 1)
            a.backend = 'openai'
            path, body = payload(a, p, 32, True)
            self.assertEqual(body['seed'], 61)
            self.assertNotIn('routed_dp_rank', body)

        def test_cache_unknown_miss_tail_and_length_mismatch(self):
            a = config()
            r = Record(Prompt(0, 0, array('I', [1] * 1024), 60))
            with self.assertRaises(BenchError):
                cache_status(r, a)
            a.allow_unverified_cache = True
            self.assertEqual(cache_status(r, a), 'unknown')
            r.cached_tokens = 0
            with self.assertRaises(BenchError):
                cache_status(r, a)
            r.cached_tokens = 1023
            self.assertEqual(cache_status(r, a), 'reported_hit')
            r.prompt_tokens = 1000
            with self.assertRaises(BenchError):
                cache_status(r, a)

        def test_metrics_duplicates_missing_and_no_summing(self):
            a = config()
            text = '\n'.join(f'sglang:{n}{{dp_rank="{rank}"}} {v}' for rank in range(2)
                             for n, v in [('num_running_reqs', 2), ('num_queue_reqs', 0)])
            ranks = parse_metrics(text, a)
            self.assertTrue(metrics_ready(ranks, 2))
            self.assertFalse(metrics_ready(ranks, 1))
            with self.assertRaisesRegex(BenchError, 'Duplicate'):
                parse_metrics(text + '\n' + text, a)
            with self.assertRaises(BenchError):
                parse_metrics(text.splitlines()[0], a)

        def test_sse_progress_does_not_count_tokens_or_usage_as_frames(self):
            r = Record(Prompt(0, 0, array('I', [1] * 128), 60))
            for count in (1, 1, 4, 4, 8):
                r.consume(json.dumps({'meta_info': {'completion_tokens': count}}), 'sglang')
            self.assertEqual(len(r.events), 3)
            r.consume('[DONE]', 'sglang')
            self.assertEqual(len(r.events), 3)
            r = Record(r.prompt)
            r.consume('{"choices":[{"delta":{"role":"assistant"}}]}', 'openai-chat')
            r.consume('{"choices":[{"delta":{"content":"a"}}]}', 'openai-chat')
            r.consume('{"usage":{"prompt_tokens":128,"prompt_tokens_details":{"cached_tokens":127}}}', 'openai-chat')
            self.assertEqual(len(r.events), 1)
            self.assertEqual(r.cached_tokens, 127)

        def test_statistics_do_not_divide_by_mtp(self):
            r = Record(Prompt(0, 0, array('I', [1]), 60))
            r.events = [i * .06 for i in range(20)]
            result = summarize([r], 0, 1)
            self.assertAlmostEqual(result['sse_gap_median_ms'], 60)

    class HTTPTests(unittest.IsolatedAsyncioTestCase):
        async def asyncSetUp(self):
            self.running = [0, 0]
            self.cache = set()
            self.mode = 'normal'
            self.posts = 0
            self.temp = tempfile.TemporaryDirectory()
            app = web.Application()
            for path in ('/generate', '/v1/completions', '/v1/chat/completions'):
                app.router.add_post(path, self.generate)
            app.router.add_get('/server_info', self.info)
            app.router.add_get('/metrics', self.metrics)
            self.server = web.AppRunner(app)
            await self.server.setup()
            site = web.TCPSite(self.server, '127.0.0.1', 0)
            await site.start()
            self.url = f'http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}'

        async def asyncTearDown(self):
            await self.server.cleanup()
            self.temp.cleanup()

        async def info(self, request):
            return web.json_response(dict(dp_size=2, max_total_num_tokens=100000, page_size=16,
                                          context_length=8192, api_key='DO_NOT_SAVE_SERVER_SECRET'))

        async def metrics(self, request):
            text = '\n'.join(f'sglang:{n}{{dp_rank="{rank}"}} {v}' for rank in range(2)
                             for n, v in [('num_running_reqs', self.running[rank]),
                                          ('num_queue_reqs', 1 if self.mode == 'queued' else 0)])
            return web.Response(text=text)

        async def generate(self, request):
            self.posts += 1
            body = await request.json()
            native = request.path == '/generate'
            ids = body.get('input_ids', body.get('prompt'))
            if ids is None:
                ids = FakeTokenizer().apply_chat_template(body['messages'])
            n = len(ids)
            rank = body.get('routed_dp_rank', 0)
            key = (rank, tuple(ids))
            cached = max(0, n - 1) if key in self.cache else 0
            if self.mode == 'cache-miss':
                cached = 0
            self.cache.add(key)
            limit = body.get('sampling_params', {}).get('max_new_tokens', body.get('max_tokens'))
            if self.mode == 'early' and limit > 1:
                limit = 2
            if self.mode == 'http-error':
                return web.Response(status=503, text='DO_NOT_SAVE_RESPONSE_SECRET')
            response = web.StreamResponse(headers={'Content-Type': 'text/event-stream'})
            await response.prepare(request)
            self.running[rank] += 1
            try:
                for i in range(1, limit + 1):
                    await asyncio.sleep(.007)
                    if native:
                        obj = {'text': 'DO_NOT_SAVE_GENERATED_TEXT', 'meta_info': {
                            'prompt_tokens': n, 'cached_tokens': cached, 'completion_tokens': i,
                            'dp_rank': (1 - rank if self.mode == 'bad-route' else rank)}}
                    else:
                        obj = {'choices': [{'delta': {'content': 'hello'}}]}
                    if self.mode == 'missing-cache' and native:
                        obj['meta_info'].pop('cached_tokens')
                    raw = ('data: ' + json.dumps(obj) + '\r\n\r\n').encode()
                    await response.write(raw[:7])
                    await response.write(raw[7:])
                    if self.mode == 'truncated' and i == 2:
                        await response.write(b'data: {')
                        return response
                if not native:
                    usage = {'choices': [], 'usage': {'prompt_tokens': n, 'completion_tokens': limit,
                                                     'prompt_tokens_details': {'cached_tokens': cached}}}
                    await response.write(('data: ' + json.dumps(usage) + '\n\n').encode())
                await response.write(b'data: [DONE]\n\n')
            except (ConnectionResetError, asyncio.CancelledError):
                pass
            finally:
                self.running[rank] -= 1
            return response

        async def invoke(self, **changes):
            settings = dict(base_url=self.url, metrics_url=self.url + '/metrics', save_path=self.temp.name,
                            steady_check='required', plot=True)
            settings.update(changes)
            a = config(**settings)
            fake_module = types.SimpleNamespace(AutoTokenizer=FakeTokenizer)
            with patch.dict(sys.modules, {'transformers': fake_module}), contextlib.redirect_stdout(io.StringIO()):
                await run(a)
            directories = list(Path(self.temp.name).glob('sse_*'))
            return json.loads((directories[-1] / 'results.json').read_text()), directories[-1]

        async def test_native_end_to_end_warm_probe_measure_plot(self):
            result, directory = await self.invoke()
            self.assertEqual(result[0]['status'], 'ok')
            m = result[0]['measurement']
            self.assertEqual(m['cache_verification'], 'reported_hit_all_requests')
            self.assertEqual(m['bs_verification'], 'sampled_per_rank_metrics')
            self.assertEqual(self.posts, 12)  # 4 warm + 4 probe + 4 formal
            self.assertGreater(m['sse_gap_median_ms'], 4)
            self.assertTrue((directory / 'sse_gap.svg').exists())
            for path in directory.glob('*'):
                content = path.read_text()
                self.assertNotIn('DO_NOT_SAVE_GENERATED_TEXT', content)
                self.assertNotIn('DO_NOT_SAVE_SERVER_SECRET', content)

        async def test_openai_completion_no_server_hook(self):
            result, _ = await self.invoke(backend='openai', steady_check='client')
            self.assertEqual(result[0]['status'], 'ok')
            self.assertEqual(result[0]['measurement']['bs_verification'], 'client_only_unverified')
            self.assertEqual(result[0]['measurement']['cache_verification'], 'reported_hit_all_requests')

        async def test_semantic_chat_pipeline(self):
            source = Path(self.temp.name) / 'shadow'
            source.mkdir()
            rows = [{'messages': [{'role': 'user', 'content': (f'Public example {i}. ' + 'a' * 128)[:126]}]}
                    for i in range(16)]
            (source / 'data.jsonl').write_text('\n'.join(json.dumps(row) for row in rows))
            result, _ = await self.invoke(backend='openai-chat', mode=4, shadow_dir=str(source), steady_check='client')
            self.assertEqual(result[0]['status'], 'ok')
            self.assertEqual(result[0]['prefix_workload'], 'natural_semantic')
            self.assertEqual(result[0]['measurement']['cache_verification'], 'reported_hit_all_requests')

        async def test_queued_metrics_never_count_as_stable(self):
            self.mode = 'queued'
            with self.assertRaisesRegex(BenchError, 'timeout'):
                await self.invoke(wait_timeout=.15)

        async def test_unknown_cache_refused(self):
            self.mode = 'missing-cache'
            with self.assertRaisesRegex(BenchError, 'cached_tokens'):
                await self.invoke()
            self.assertLessEqual(self.posts, 8)  # never formal measurement

        async def test_unknown_cache_explicitly_unverified(self):
            self.mode = 'missing-cache'
            result, _ = await self.invoke(allow_unverified_cache=True)
            self.assertEqual(result[0]['measurement']['cache_verification'], 'unverified')

        async def test_known_miss_not_silently_accepted(self):
            self.mode = 'cache-miss'
            with self.assertRaisesRegex(BenchError, 'cached'):
                await self.invoke(allow_unverified_cache=True)

        async def test_bad_route_refused(self):
            self.mode = 'bad-route'
            with self.assertRaisesRegex(BenchError, 'routing mismatch'):
                await self.invoke()

        async def test_early_eos_invalidates_whole_cohort(self):
            self.mode = 'early'
            with self.assertRaisesRegex(BenchError, 'finished before'):
                await self.invoke()

        async def test_truncated_sse_invalid(self):
            self.mode = 'truncated'
            with self.assertRaisesRegex(BenchError, 'Truncated'):
                await self.invoke()

        async def test_capacity_failure_sends_zero_requests(self):
            with self.assertRaisesRegex(BenchError, 'Independent KV'):
                await self.invoke(kv_capacity_tokens=100)
            self.assertEqual(self.posts, 0)

        async def test_plan_only_has_zero_generation(self):
            result, _ = await self.invoke(plan_only=True)
            self.assertEqual(result[0]['status'], 'planned_not_run')
            self.assertEqual(self.posts, 0)

        async def test_http_error_body_never_saved(self):
            self.mode = 'http-error'
            with self.assertRaises(BenchError):
                await self.invoke()
            for path in Path(self.temp.name).rglob('*.json'):
                self.assertNotIn('DO_NOT_SAVE_RESPONSE_SECRET', path.read_text())

        async def test_sse_utf8_arbitrary_boundaries(self):
            raw = 'data: {"text":"中文"}\r\n\r\n:keepalive\n\ndata: [DONE]\n\n'.encode()
            async def chunks():
                for b in raw:
                    yield bytes([b])
            frames = [frame async for frame in sse_frames(chunks())]
            self.assertEqual(frames, ['{"text":"中文"}', '[DONE]'])

    suite = unittest.TestSuite()
    for cls in (PlanningTests, HTTPTests):
        suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(cls))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return result.wasSuccessful()


if __name__ == '__main__':
    arguments = parser().parse_args()
    try:
        if arguments.self_test:
            sys.exit(0 if self_test() else 1)
        validate(arguments)
        asyncio.run(run(arguments))
    except KeyboardInterrupt:
        print('Interrupted; own client streams closed. No global service changes.', file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print('ERROR: ' + (str(exc) if isinstance(exc, BenchError) else type(exc).__name__), file=sys.stderr)
        sys.exit(2)
