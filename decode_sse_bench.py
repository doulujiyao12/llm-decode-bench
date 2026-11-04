#!/usr/bin/env python3
"""Standalone prefix-cache-aware SSE benchmark (Python 3.10+).

Copy ONLY this file. Dependencies: pip install aiohttp
Synthetic modes / optional local length counting also need transformers.
No server changes, GPU profiler, package imports, cache flush or global abort.

Modes: 0=identical shared prefix, 1=uniform deterministic,
2=mixed deterministic, 3=mixed fresh random,
4=real text with matched per-rank mean, 5=random real text from a length bucket.
Modes 0-3 use token-ID prompts: /generate or /v1/completions, NOT chat-only APIs.
Modes 4/5 send plain text; local tokenizer is optional. No chat template needed.
Mode 0 uses one seeded token sequence across all requests/ranks/BS values;
shorter inputs are slices of that sequence. KV sharing is budgeted PER rank,
with private input-tail and output reserves, never as a cross-GPU cache.
Modes 0-3 support --warmup-strategy once-max: one covering input bank,
descending BS/length sweep, no per-case warmup/probe; cache hits are recorded,
not required. Missing KV is computed by the server before the steady window.

Capacity is per ATTENTION DP rank, not the sum over TP GPUs. Default: read
SGLang /server_info; otherwise require --max-total-kvcache-length-k (K=1024).
Default synthetic sweeps stop each BS at its capacity/context input bound,
adding the bound once if it replaces larger requested lengths, then move to
the next BS. Runtime failures still fail the run. --allow-shared-prefix
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
from contextlib import contextmanager
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
import tempfile
import time
from urllib.parse import urlsplit


class BenchError(RuntimeError):
    pass


class CapacityError(BenchError):
    """A recognized KV budget failure, distinct from service/protocol failures."""


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


def identical_prefix_cost(input_tokens, bs, output_tokens, page):
    """One shared prefix per rank, but reserve a private tail page per request.

    Even an aligned identical prompt may recompute its last page (last-token
    cache matching / copy-on-write). Do not count that page as fully shared,
    and never assume generated output is identical or shared between requests.
    """
    shared_pages = (input_tokens - 1) // page * page
    return shared_pages + bs * (page + rounded(output_tokens, page))


def plan_sharing(lengths, budget, output_tokens, page, allow, min_suffix, max_shared=None):
    none = [0] * len(lengths)
    independent = rank_cost(lengths, none, output_tokens, page)
    if independent <= budget:
        return none, independent
    if not allow:
        raise CapacityError(f'Independent KV needs {independent:,} tokens/rank, usable budget '
                            f'{budget:,}. Reduce BS/context/output or explicitly add --allow-shared-prefix.')
    if max_shared is None:
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


def plan_input_sweep(args, capacity, page, bs, context_length=None):
    """Plan means before allocating prompts; never exceed the user's max target.

    Uniform input has an exact page-aligned pool bound. Mixed input reserves
    worst-case page tails for every request and uses its configured max length
    ratio for context, keeping the selected endpoint safe for every rank/seed.
    Real text has cohort-dependent prefix sharing: its original content is
    checked by make_case, never shortened or assigned an artificial endpoint.
    """
    requested = [round(k * 1024) for k in args.input_lengths_k]
    summary = dict(bs_per_dp=bs, requested_input_tokens=requested,
                   budget_tokens_per_rank=math.floor(capacity * args.capacity_fraction),
                   output_reserve_tokens_per_request=rounded(args.output_tokens, page))
    if args.mode != 0 and (args.allow_shared_prefix or args.mode in (4, 5)):
        summary.update(policy='shared_prefix_requested_grid' if args.allow_shared_prefix else 'real_text_preflight',
                       auto_added_input_tokens=[], omitted_requested_input_tokens=[])
        grid = requested if args.allow_shared_prefix else sorted(set(requested))
        return [dict(mean=n, origin='requested') for n in grid], summary

    budget = summary['budget_tokens_per_rank']
    output = summary['output_reserve_tokens_per_request']
    mixed = args.mode in (2, 3) and bs > 1
    if args.mode == 0:
        # Invert identical_prefix_cost: the final input page is private even
        # when the full input is page-aligned, and every output is private.
        prefix_budget = budget - bs * (output + page)
        kv_limit = (prefix_budget // page + 1) * page if prefix_budget >= 0 else 0
    else:
        per_input_budget = budget // bs - output
        if mixed:
            per_input_budget -= page - 1  # Every request may have a different last-page tail.
        kv_limit = max(0, per_input_budget // page * page)
    context_limit = None
    if type(context_length) is int and context_length > 0:
        remaining = max(0, context_length - args.output_tokens)
        context_limit = math.floor(remaining / args.max_length_ratio) if mixed else remaining
    limit = min(kv_limit, context_limit) if context_limit is not None else kv_limit
    limit = max(0, limit)
    unique = sorted(set(requested))
    selected = [n for n in unique if n <= limit]
    omitted = [n for n in unique if n > limit]
    if context_limit is not None and context_limit < kv_limit:
        reason = 'context_limit'
    elif context_limit == kv_limit:
        reason = 'kv_and_context_limit'
    else:
        reason = 'kv_capacity'
    added = [limit] if omitted and limit > 0 and limit not in selected else []
    entries = [dict(mean=n, origin='requested') for n in selected]
    entries.extend(dict(mean=n, origin='automatic_' + reason + '_boundary') for n in added)
    summary.update(policy='identical_prefix_input_bound' if args.mode == 0 else 'independent_input_bound',
                   bound_kind=('shared_prefix_private_tail_reserved' if args.mode == 0
                               else 'conservative_mixed_mean' if mixed else 'uniform_page_aligned'),
                   kv_input_limit_tokens=kv_limit, context_input_limit_tokens=context_limit,
                   effective_input_limit_tokens=limit, limiting_factor=reason,
                   selected_input_tokens=[entry['mean'] for entry in entries],
                   auto_added_input_tokens=added, omitted_requested_input_tokens=omitted)
    return entries, summary


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
    ids: array | None
    seed: int
    shared_tokens: int = 0
    messages: list | None = None
    text: str | None = None
    planned_tokens: int | None = None
    length_source: str = 'token_ids'
    observed_tokens: int | None = None
    source_bucket: tuple | None = None

    @property
    def input_tokens(self):
        if self.text is None:
            return len(self.ids)
        return self.observed_tokens if self.observed_tokens is not None else self.planned_tokens

    @property
    def sha256(self):
        return hashlib.sha256(self.text.encode('utf-8')).hexdigest() if self.text is not None else digest_ids(self.ids)

    def manifest(self):
        return dict(index=self.index, planned_rank=self.rank, input_tokens=self.input_tokens,
                    sha256=self.sha256, generation_seed=self.seed,
                    synthetic_shared_prefix_tokens=self.shared_tokens,
                    planned_input_tokens=self.planned_tokens if self.text is not None else len(self.ids),
                    actual_input_tokens=self.observed_tokens, length_source=self.length_source,
                    source_bucket=self.source_bucket)


@dataclass
class ShadowSample:
    text: str
    tokens: int
    length_source: str
    bucket: tuple
    sha256: str


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


def shadow_text(row):
    """Extract real content into a fixed, model-independent text format.

    Plain text is unchanged. Structured histories keep roles, reasoning, tool
    definitions, calls and results; no captured routing/sampling fields enter
    the prompt. No truncation, repetition, padding or model control tokens.
    """
    if not isinstance(row, dict):
        return None
    for wrapper in ('request', 'body'):
        if isinstance(row.get(wrapper), dict):
            row = row[wrapper]
    if any(row.get(k) for k in ('images', 'image_data', 'audio', 'video')):
        return None
    messages = row.get('messages')
    if messages is None and isinstance(row.get('conversations'), list):
        roles = {'human': 'user', 'gpt': 'assistant', 'system': 'system'}
        messages = [dict(role=roles.get(m.get('from')), content=m.get('value'))
                    for m in row['conversations'] if isinstance(m, dict)]
        if len(messages) != len(row['conversations']):
            return None
    if messages is None:
        text = row.get('text', row.get('prompt'))
        if isinstance(text, str) and text.strip():
            return text
        return None
    if not isinstance(messages, list) or not messages:
        return None
    blocks = []
    for key in ('tools', 'functions'):
        if row.get(key):
            blocks.append(f'[{key}]\n' + json.dumps(row[key], ensure_ascii=False, sort_keys=True))
    for m in messages:
        if not isinstance(m, dict) or not isinstance(m.get('role'), str):
            return None
        content = m.get('content')
        if isinstance(content, list):
            if any(not isinstance(p, dict) or p.get('type') != 'text'
                   or not isinstance(p.get('text'), str) or set(p) - {'type', 'text'} for p in content):
                return None
            content = '\n'.join(p['text'] for p in content)
        if content is not None and not isinstance(content, str):
            return None
        blocks.append(f'[{m["role"]}]')
        if m.get('reasoning_content') is not None:
            if not isinstance(m['reasoning_content'], str):
                return None
            blocks.append('[reasoning]\n' + m['reasoning_content'])
        if content is not None:
            blocks.append(content)
        extra = {k: v for k, v in m.items() if k not in {'role', 'content', 'reasoning_content'} and v is not None}
        if extra:
            blocks.append(json.dumps(extra, ensure_ascii=False, sort_keys=True))
    text = '\n'.join(blocks)
    return text if text.strip() else None


def dotted_value(row, key):
    value = row
    for part in key.split('.'):
        value = value.get(part) if isinstance(value, dict) else None
    return value


def shadow_reference(row, field_name=None):
    fields = [field_name] if field_name else [
        'reference_content_tokens', 'reference_input_tokens', 'input_tokens', 'prompt_tokens',
        'input_len', 'meta.prompt_tokens_offline', 'meta.prompt_tokens',
        'metadata.prompt_tokens', 'usage.prompt_tokens']
    for key in fields:
        value = dotted_value(row, key)
        if type(value) is int and value > 0:
            return value, key
    return None, None


def shadow_bucket(row, path, tokens, boundaries=None):
    if boundaries is None:
        names = [dotted_value(row, 'meta.prompt_bucket'), row.get('bucket')]
        names += list(reversed(Path(path).parts))
        for name in names:
            if not isinstance(name, str):
                continue
            match = re.search(r'(\d+)[_-](\d+)[kK](?:\b|[_.-])', name)
            if match:
                lo, hi = (int(n) * 1024 for n in match.groups())
                if lo < hi:
                    return lo, hi
            match = re.search(r'(\d+)[kK][_-]plus(?:\b|[_.-])', name)
            if match:
                return int(match[1]) * 1024, None
    bounds = boundaries if boundaries is not None else [0, 32*1024, 64*1024, 128*1024, 208*1024, 256*1024]
    for lo, hi in zip(bounds, bounds[1:] + [None]):
        if lo <= tokens and (hi is None or tokens < hi):
            return lo, hi
    raise BenchError('Reference length does not fit configured shadow buckets')


def load_shadow(folder, tokenizer=None, length_field=None, bucket_boundaries=None, summary=None):
    candidates, seen = [], set()
    counts = dict(records=0, accepted=0, unsupported=0, duplicates=0)
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
            for number, row in enumerate(rows, 1):
                counts['records'] += 1
                text = shadow_text(row)
                if text is None:
                    counts['unsupported'] += 1
                    continue
                key = hashlib.sha256(text.encode('utf-8')).hexdigest()
                if key in seen:
                    counts['duplicates'] += 1
                    continue
                if tokenizer is not None:
                    tokens, source = len(tokenizer.encode(text, add_special_tokens=True)), 'local_tokenizer'
                else:
                    tokens, field_name = shadow_reference(row, length_field)
                    check(tokens is not None, f'Shadow record {number} in {path.name} has no reference token length; '
                          'supply --tokenizer or --shadow-length-field. Character counts are not token counts.')
                    source = 'recorded_reference:' + field_name
                check(tokens > 0, 'Shadow input has no tokens')
                seen.add(key)
                candidates.append(ShadowSample(text, tokens, source,
                    shadow_bucket(row, path, tokens, bucket_boundaries), key))
                counts['accepted'] += 1
                if counts['records'] % 500 == 0:
                    log(f'Shadow: scanned {counts["records"]:,}; accepted {counts["accepted"]:,} unique texts')
    lengths = [sample.tokens for sample in candidates]
    buckets = sorted(set(s.bucket for s in candidates), key=lambda b: b[0])
    if summary is not None:
        summary.update(counts, format='plain_text_v1', length_basis='local_tokenizer' if tokenizer else 'recorded_reference',
                       buckets=[dict(low=lo, high=hi, count=sum(s.bucket == (lo, hi) for s in candidates)) for lo, hi in buckets])
    log(f'Shadow: {len(candidates)} unique plain texts; skipped {counts["unsupported"]} unsupported / '
        f'{counts["duplicates"]} duplicate records; '
        f'lengths={"local tokenizer" if tokenizer else "recorded reference, NOT measured target tokens"}'
        + (f'; input min/mean/max={min(lengths)}/{statistics.mean(lengths):.1f}/{max(lengths)}'
           if lengths else ''))
    return candidates


def shadow_length(sample):
    return sample.tokens if isinstance(sample, ShadowSample) else len(sample[0])


def select_shadow(candidates, dp, bs, mean, tolerance, rng, low_ratio, high_ratio):
    usable = [x for x in candidates if mean * low_ratio <= shadow_length(x) <= mean * high_ratio]
    check(len(usable) >= dp * bs, 'Not enough distinct semantic prompts within length bounds')
    available = list(range(len(usable)))
    selected = []
    for _rank in range(dp):
        target = bs * mean
        best, error = None, float('inf')
        # Random restarts + substitutions. Never alter selected prompt content.
        for _ in range(12):
            chosen = rng.sample(available, bs)
            total = sum(shadow_length(usable[i]) for i in chosen)
            for _ in range(2000):
                if abs(total - target) <= target * tolerance:
                    break
                pos, new = rng.randrange(bs), rng.choice(available)
                if new in chosen:
                    continue
                alt = total - shadow_length(usable[chosen[pos]]) + shadow_length(usable[new])
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


def select_shadow_bucket(candidates, dp, bs, target, rng):
    buckets = {s.bucket for s in candidates}
    matches = [b for b in buckets if b[0] <= target and (b[1] is None or target < b[1])]
    check(len(matches) == 1, 'Target length must match exactly one populated shadow bucket; '
          'check the dataset or use --shadow-bucket-boundaries-k to define non-overlapping buckets')
    bucket = matches[0]
    usable = [s for s in candidates if s.bucket == bucket]
    check(len(usable) >= dp * bs,
          f'Bucket [{bucket[0]}, {bucket[1]}) has {len(usable)} distinct texts; needs {dp * bs}. '
          'Requests are not repeated to fill the batch.')
    chosen = rng.sample(usable, dp * bs)
    return [chosen[rank * bs:(rank + 1) * bs] for rank in range(dp)], bucket


def synthetic_case_seed(args, bs, mean, run_seed):
    return (f'{run_seed}:mode0:global-prefix' if args.mode == 0
            else f'{run_seed}:mode{args.mode}:bs{bs}:mean{mean}')


def synthetic_lengths(args, bs, mean, run_seed, rank):
    if args.mode in (0, 1):
        return [mean] * bs
    seed = synthetic_case_seed(args, bs, mean, run_seed)
    return mixed_lengths(bs, mean, args.min_length_ratio, args.max_length_ratio,
                         random.Random(f'{seed}:lengths:{rank}'))


def build_once_max_bank(args, tokenizer, capacity, page, run_seed, sweeps):
    """One stream per (rank, local slot), covering ALL feasible matrix points.

    Mixed lengths / capacity-added endpoints need not peak at the largest BS.
    Take each slot's actual maximum, not merely the largest requested mean.
    Budget the union plus the largest output cohort BEFORE sending requests.
    """
    active = [(bs, entries) for bs, entries, _ in sweeps if entries]
    check(active, 'No requested BS has a feasible input case; no warmup sent.')
    max_bs = max(bs for bs, _ in active)
    budget = math.floor(capacity * args.capacity_fraction)
    vocab = sorted(set(tokenizer.get_vocab().values()) - set(tokenizer.all_special_ids))
    check(len(vocab) >= 2, 'Tokenizer has fewer than 2 non-special token IDs')
    streams, warm, rank_plans = {}, [], []
    mode0_ids = None
    if args.mode == 0:
        length = max(e['mean'] for _, entries in active for e in entries)
        mode0_ids = array('I', random.Random(f'{run_seed}:mode0:global-prefix').choices(vocab, k=length))
    for rank in range(args.dp_size):
        maxima = [0] * max_bs
        minima = [math.inf] * max_bs
        for bs, entries in active:
            for entry in entries:
                for local, length in enumerate(synthetic_lengths(args, bs, entry['mean'], run_seed, rank)):
                    maxima[local] = max(maxima[local], length)
                    minima[local] = min(minima[local], length)
        if args.mode == 0:
            cost = identical_prefix_cost(len(mode0_ids), max_bs, args.output_tokens, page)
            shared = [len(mode0_ids)] * max_bs
        else:
            # Sharing must be stable even at the shortest measured truncation.
            # Never turn a requested independent suffix into a fully shared one.
            max_shared = [max(0, n - args.min_unique_suffix_tokens) // page * page for n in minima]
            try:
                shared, cost = plan_sharing(maxima, budget, args.output_tokens, page,
                    args.allow_shared_prefix, args.min_unique_suffix_tokens, max_shared=max_shared)
            except BenchError as exc:
                raise CapacityError(f'Once-max input bank on rank {rank} cannot fit: {exc} '
                    'The union of all BS/length cases can exceed any single case. '
                    'Reduce the matrix, increase the valid KV budget, or use --warmup-strategy per-case.') from None
        check(cost <= budget, f'Once-max input bank needs {cost:,} tokens/rank; budget {budget:,}. '
              'Reduce the matrix or use --warmup-strategy per-case; no warmup sent.')
        common = (array('I', random.Random(f'{run_seed}:mode{args.mode}:bank:common:{rank}').choices(
                    vocab, k=max(shared))) if args.mode != 0 else None)
        for local, length in enumerate(maxima):
            index = local * args.dp_size + rank
            if args.mode == 0:
                ids = mode0_ids
            else:
                rng = random.Random(f'{run_seed}:mode{args.mode}:bank:input:{rank}:{local}')
                ids = common[:shared[local]]
                ids.extend(rng.choices(vocab, k=length - shared[local]))
            prompt = Prompt(index, rank, ids, args.generation_seed + index, shared[local])
            streams[rank, local] = prompt
            # Mode 0 needs only one identical warmup per DP, not max_bs copies.
            if args.mode != 0 or local == 0:
                warm.append(prompt)
        rank_plans.append(dict(rank=rank, stream_count=max_bs,
            longest_input_tokens=max(map(len, (streams[rank, i].ids for i in range(max_bs)))),
            sum_input_tokens=sum(len(streams[rank, i].ids) for i in range(max_bs)),
            estimated_union_and_output_tokens=cost, usable_capacity_tokens=budget))
    plan = dict(strategy='once-max', status='planned', max_bs_per_dp=max_bs,
                warmup_request_count=len(warm), warmup_output_tokens=1,
                warmup_concurrency_effective=effective_warmup_concurrency(args, len(warm)),
                measured_order='bs_descending_then_length_descending',
                input_policy='fixed_per_rank_slot_streams_truncated_to_case_length',
                cache_policy='report_only', rank_plans=rank_plans,
                requests=[p.manifest() for p in warm])
    return streams, sorted(warm, key=lambda p: p.index), plan


def make_case(args, tokenizer, capacity, page, bs, mean, run_seed, shadows=None, synthetic_bank=None):
    budget = math.floor(capacity * args.capacity_fraction)
    seed = synthetic_case_seed(args, bs, mean, run_seed)
    rng = random.Random(seed)
    by_rank, costs, plans = [], [], []
    identical_ids = None
    selected_bucket = None
    if args.mode == 4:
        cohorts = select_shadow(shadows, args.dp_size, bs, mean, args.shadow_mean_tolerance,
                                rng, args.min_length_ratio, args.max_length_ratio)
    elif args.mode == 5:
        cohorts, selected_bucket = select_shadow_bucket(shadows, args.dp_size, bs, mean, rng)
    else:
        vocab = sorted(set(tokenizer.get_vocab().values()) - set(tokenizer.all_special_ids))
        check(len(vocab) >= 2, 'Tokenizer has fewer than 2 non-special token IDs')
        if args.mode == 0:
            cost = identical_prefix_cost(mean, bs, args.output_tokens, page)
            if cost > budget:
                raise CapacityError(f'Identical-prefix KV needs {cost:,} tokens/rank, usable budget '
                                    f'{budget:,}; includes private tail pages and output tokens.')
            # No BS/rank/length in the seed: every case uses the same logical
            # infinite prefix. Read-only shared array; payload() copies to list.
            identical_ids = (synthetic_bank[0, 0].ids[:mean] if synthetic_bank is not None
                             else array('I', rng.choices(vocab, k=mean)))
    for rank in range(args.dp_size):
        if args.mode == 0:
            lengths = [mean] * bs
            shared = [mean] * bs
        elif args.mode in (4, 5):
            cohort = cohorts[rank]
            lengths = [shadow_length(x) for x in cohort]
            cost = (sum(rounded(n, page) + rounded(args.output_tokens, page) for n in lengths)
                    if isinstance(cohort[0], ShadowSample)
                    else natural_prefix_cost([x[0] for x in cohort], args.output_tokens, page))
            if cost > budget:
                raise CapacityError(f'Shadow rank {rank}: {cost:,} KV tokens > usable {budget:,}; '
                                    'artificial prefix sharing would change semantics and is not applied.')
            shared = [0] * bs
        else:
            lengths = synthetic_lengths(args, bs, mean, run_seed, rank)
            if synthetic_bank is not None:
                shared = [synthetic_bank[rank, i].shared_tokens for i in range(bs)]
                cost = rank_cost(lengths, shared, args.output_tokens, page)
                check(cost <= budget, 'Truncated bank exceeds case budget')
            else:
                shared, cost = plan_sharing(lengths, budget, args.output_tokens, page,
                                           args.allow_shared_prefix, args.min_unique_suffix_tokens)
                prefix_rng = random.Random(f'{seed}:common:{rank}')
                common = array('I', prefix_rng.choices(vocab, k=max(shared)))
        rank_prompts = []
        for local, length in enumerate(lengths):
            index = local * args.dp_size + rank
            sample = None
            if args.mode == 0:
                ids, msgs = identical_ids, None
            elif args.mode in (4, 5):
                if isinstance(cohort[local], ShadowSample):
                    sample = cohort[local]
                    ids, msgs = None, None
                else:
                    ids, msgs = cohort[local]
            elif synthetic_bank is not None:
                source = synthetic_bank[rank, local].ids
                check(length <= len(source), 'Measured input exceeds prewarmed stream length')
                ids, msgs = source[:length], None
            else:
                prng = random.Random(f'{seed}:input:{rank}:{local}')
                ids = common[:shared[local]]
                ids.extend(prng.choices(vocab, k=length - shared[local]))
                msgs = None
            gseed = (args.generation_seed + index if args.mode in (0, 1, 2)
                     else rng.randrange(0, 2**31 - 1))
            rank_prompts.append(Prompt(index, rank, ids, gseed, shared[local], msgs,
                text=sample.text if sample else None,
                planned_tokens=sample.tokens if sample else None,
                length_source=sample.length_source if sample else 'token_ids',
                source_bucket=sample.bucket if sample else None))
        by_rank.extend(rank_prompts)
        costs.append(cost)
        plans.append(dict(rank=rank, min=min(lengths), max=max(lengths), mean=statistics.mean(lengths),
                          sum_input_tokens=sum(lengths), estimated_kv_tokens=cost,
                          max_synthetic_shared_tokens=max(shared)))
    unique_inputs = len({p.sha256 for p in by_rank})
    if args.mode == 0:
        check(unique_inputs == 1, 'Mode 0 requires identical input tokens for every request')
    else:
        check(unique_inputs == len(by_rank),
              'Duplicate prompts found; increase input/suffix lengths or vocabulary')
    return sorted(by_rank, key=lambda p: p.index), dict(
        mode=args.mode, bs_per_dp=bs, dp_size=args.dp_size, mean_input_tokens_requested=mean,
        run_seed=run_seed, case_seed=seed, capacity_tokens_per_rank=capacity,
        usable_capacity_tokens_per_rank=budget, output_reserve_tokens_per_request=args.output_tokens,
        page_size=page, rank_plans=plans,
        selection=('matched_per_rank_mean' if args.mode == 4 else 'random_within_bucket' if args.mode == 5 else 'synthetic'),
        selected_bucket=selected_bucket,
        planned_mean_input_tokens=statistics.mean(p.input_tokens for p in by_rank),
        length_basis=sorted({p.length_source for p in by_rank}),
        prefix_workload=('fully_shared_synthetic' if args.mode == 0
                         else 'capacity_shared_synthetic' if any(p.shared_tokens for p in by_rank)
                         else 'natural_semantic' if args.mode in (4, 5) else 'independent_synthetic'),
        input_bank_reused=synthetic_bank is not None)


async def received_chunks(content, stats):
    """Bounded socket reader, independent of JSON consumption on this event loop.

    Timestamps are application read times, NOT NIC/kernel arrival times. A
    blocked event loop/socket buffer can still be behind: the live-token gate
    below checks that separately. Never manufacture evenly spaced timestamps.
    """
    queue = asyncio.Queue(maxsize=4)  # <= 256 KiB queued bytes per stream.

    async def reader():
        try:
            async for chunk in content.iter_chunked(65536):
                at = time.monotonic()
                stats['bytes_read'] += len(chunk)
                if queue.full():
                    stats['queue_full_count'] += 1
                await queue.put((chunk, at))
                stats['queue_wait_max_seconds'] = max(stats['queue_wait_max_seconds'], time.monotonic() - at)
            await queue.put(None)
        except Exception as exc:
            await queue.put(exc)

    task = asyncio.create_task(reader())
    try:
        while True:
            item = await queue.get()
            if item is None:
                return
            if isinstance(item, Exception):
                raise item
            yield item
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def sse_frames(chunks, timed=False):
    decoder = codecs.getincrementaldecoder('utf-8-sig')('strict')
    buf, data = '', []
    size = 0
    async for chunk in chunks:
        if timed:
            chunk, at = chunk
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
                    raw = '\n'.join(data)
                    yield (raw, at) if timed else raw
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
            raw = '\n'.join(data)
            yield (raw, at) if timed else raw
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
    request_id: str | None = None
    submitted: bool = False
    observations: list = field(default_factory=list)
    read_stats: dict = field(default_factory=lambda: dict(
        bytes_read=0, queue_full_count=0, queue_wait_max_seconds=0.0))

    def consume(self, raw, backend, received_at=None):
        # Called with a complete frame's last-byte READ timestamp, before JSON
        # parsing. Frames completed by one read keep the same timestamp.
        received_at = time.monotonic() if received_at is None else received_at
        parse_start = time.monotonic()
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
            parsed_at = time.monotonic()
            self.events.append(received_at)
            self.observations.append((received_at, self.tokens if backend == 'sglang' else None,
                                      parsed_at - received_at, parsed_at - parse_start))

    def update_counts(self, prompt, cached, rank):
        for name, value in [('prompt_tokens', prompt), ('cached_tokens', cached), ('actual_rank', rank)]:
            if value is not None:
                check(type(value) is int and value >= 0, f'Invalid {name} metadata')
                setattr(self, name, value)


def validate_prompt_length(record):
    if record.prompt.text is not None:
        check(type(record.prompt_tokens) is int and record.prompt_tokens > 0,
              'Plain-text requests require service-reported prompt_tokens/usage for actual length validation')
        if record.prompt.observed_tokens is None:
            record.prompt.observed_tokens = record.prompt_tokens
        else:
            check(record.prompt.observed_tokens == record.prompt_tokens,
                  'The same plain-text request has inconsistent service-reported prompt token counts')
    elif record.prompt_tokens is not None:
        check(record.prompt_tokens == len(record.prompt.ids),
              f'Request {record.prompt.index}: server prompt token count differs from local tokenizer; '
              'chat template/tokenization mismatch, cannot trust capacity or target lengths')


def cache_status(record, args):
    validate_prompt_length(record)
    if record.actual_rank is not None and args.route:
        check(record.actual_rank == record.prompt.rank, 'Server did not honor requested DP rank')
    if record.cached_tokens is None:
        if args.warmup_strategy == 'once-max':
            return 'unknown'
        check(args.allow_unverified_cache, 'Service did not report cached_tokens. Cannot verify prefix cache; '
              'use a supported endpoint or explicitly --allow-unverified-cache (unverified results).')
        return 'unknown'
    n = record.prompt.input_tokens
    check(0 <= record.cached_tokens <= n, 'Invalid cached token count')
    if args.warmup_strategy == 'once-max':
        return 'reported_only'
    miss = n - record.cached_tokens
    check(miss <= args.cache_tail_tokens and record.cached_tokens / n >= args.min_cache_hit_ratio,
          f'Request {record.prompt.index}: only {record.cached_tokens}/{n} input tokens cached; '
          'cache disabled/evicted/routed elsewhere. This is not a warm-prefix decode test.')
    return 'reported_hit'


def summarize_cache(records):
    """Service-reported INPUT hits of formal requests, not warmup or SSE tokens.

    Unknown counters are not treated as zero or silently included in the ratio.
    A partial summary is explicitly labelled and includes its coverage.
    """
    rows = []
    for record in records:
        n, cached = record.prompt.input_tokens, record.cached_tokens
        rows.append(dict(index=record.prompt.index, planned_rank=record.prompt.rank,
                         actual_rank=record.actual_rank, input_tokens=n, cached_tokens=cached,
                         uncached_tokens=n - cached if cached is not None else None,
                         hit_ratio=cached / n if cached is not None else None))

    def totals(items):
        known = [r for r in items if r['cached_tokens'] is not None]
        inputs = sum(r['input_tokens'] for r in known)
        cached = sum(r['cached_tokens'] for r in known)
        ratios = [r['hit_ratio'] for r in known]
        return dict(request_count=len(items), reported_request_count=len(known),
                    unknown_request_count=len(items) - len(known),
                    input_tokens_reported=inputs, cached_tokens=cached,
                    uncached_tokens=inputs - cached,
                    weighted_hit_ratio=cached / inputs if inputs else None,
                    mean_request_hit_ratio=statistics.mean(ratios) if ratios else None,
                    min_request_hit_ratio=min(ratios) if ratios else None,
                    max_request_hit_ratio=max(ratios) if ratios else None)

    return dict(source='formal_requests_service_reported_input_cache',
                coverage='complete' if all(r['cached_tokens'] is not None for r in rows)
                         else 'partial' if any(r['cached_tokens'] is not None for r in rows) else 'unknown',
                **totals(rows),
                by_planned_rank=[dict(planned_rank=rank, **totals([r for r in rows if r['planned_rank'] == rank]))
                                 for rank in sorted({r['planned_rank'] for r in rows})],
                requests=rows)


def payload(args, prompt, output, stream):
    if args.backend == 'sglang':
        sampling = dict(max_new_tokens=output, temperature=args.temperature,
                        top_p=args.top_p, sampling_seed=prompt.seed)
        if args.ignore_eos:
            sampling['ignore_eos'] = True
        body = dict(sampling_params=sampling, stream=stream,
                    rid=f'ssebench_{args.run_id}_{prompt.index}_{secrets.token_hex(4)}')
        body['text' if prompt.text is not None else 'input_ids'] = (
            prompt.text if prompt.text is not None else list(prompt.ids))
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
        check(prompt.messages is not None or prompt.text is not None, 'Chat mode requires real text (mode 4/5)')
        body['messages'] = ([dict(role='user', content=prompt.text)] if prompt.text is not None else prompt.messages)
        return '/v1/chat/completions', body
    body['prompt'] = prompt.text if prompt.text is not None else list(prompt.ids)
    return '/v1/completions', body


async def request(session, args, record, output):
    path, body = payload(args, record.prompt, output, True)
    record.request_id = body.get('rid')
    try:
        record.submitted = True
        async with session.post(args.base_url + path, json=body,
                                timeout=args.request_timeout) as response:
            check(response.status == 200, f'HTTP {response.status} (response body omitted)')
            check('text/event-stream' in response.headers.get('Content-Type', ''), 'Not an SSE response')
            chunks = received_chunks(response.content, record.read_stats)
            frames = sse_frames(chunks, timed=True)
            try:
                async for raw, received_at in frames:
                    record.consume(raw, args.backend, received_at)
                    if record.done:
                        break
                    # All modes, including full-output drain: let socket readers
                    # and the controller run even when many frames are buffered.
                    await asyncio.sleep(0)
            finally:
                await frames.aclose()
                await chunks.aclose()
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


def effective_warmup_concurrency(args, request_count):
    return min(request_count, args.warmup_concurrency) if args.warmup_concurrency is not None else request_count


async def warm_phase(session, args, prompts, verify):
    concurrency = effective_warmup_concurrency(args, len(prompts))
    log(f'{"Cache probe" if verify else "Warmup"}: global client concurrency={concurrency}, '
        f'requests={len(prompts)}, {"automatic DP × current BS" if args.warmup_concurrency is None else "explicit cap"}')
    semaphore = asyncio.Semaphore(concurrency)
    phase_records = []
    async def one(prompt):
        async with semaphore:
            record = Record(prompt)
            phase_records.append(record)
            await request(session, args, record, 1)
            check(not record.error, f'Warmup/probe failed: {record.error}')
            validate_prompt_length(record)
            status = cache_status(record, args) if verify else None
            if args.route and record.actual_rank is not None:
                check(record.actual_rank == prompt.rank, 'Warmup DP routing mismatch')
            return dict(index=prompt.index, cached_tokens=record.cached_tokens,
                        actual_prompt_tokens=record.prompt_tokens, status=status)
    tasks = [asyncio.create_task(one(p)) for p in prompts]
    try:
        return await asyncio.gather(*tasks)
    except BaseException:
        if args.stop_after_measure:
            try:
                await cancel_cohort(session, args, phase_records, tasks, {})
            except Exception as exc:
                log(f'Warmup/probe cleanup did not finish: {type(exc).__name__}; stopping the matrix.')
        raise
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


def actual_shadow_plan(prompts, args, capacity, page, context=None):
    if not prompts or prompts[0].text is None:
        return {}
    ranks = []
    budget = math.floor(capacity * args.capacity_fraction)
    for rank in range(args.dp_size):
        group = [p for p in prompts if p.rank == rank]
        lengths = [p.input_tokens for p in group]
        check(all(p.observed_tokens is not None for p in group), 'Actual shadow lengths missing after warmup')
        cost = sum(rounded(n, page) + rounded(args.output_tokens, page) for n in lengths)
        check(cost <= budget, f'Actual service lengths require {cost:,} conservative KV tokens on rank {rank}, '
              f'exceeding budget {budget:,}; reduce BS/input/output. Formal requests were not sent.')
        if type(context) is int:
            check(max(lengths) + args.output_tokens <= context,
                  'Actual service input+output exceeds context_length; formal requests were not sent')
        ranks.append(dict(rank=rank, min=min(lengths), mean=statistics.mean(lengths), max=max(lengths),
                          conservative_kv_tokens=cost))
    return dict(actual_mean_input_tokens=statistics.mean(p.input_tokens for p in prompts),
                actual_lengths_by_planned_rank=ranks)


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


def parse_live_tokens(text, args):
    """Known SGLang scheduler counter, not the finish-only frontend counter.

    This counter counts committed decode output tokens (MTP accepted tokens,
    not draft tokens), excluding the first token generated by prefill. Require
    exactly one series per DP rank; NEVER sum TP/worker duplicates.
    """
    result, identities = {}, {}
    for line in text.splitlines():
        m = METRIC_LINE.match(line)
        if not m or m[1] != 'sglang:realtime_tokens_total':
            continue
        labels, end = {}, 0
        raw = m[2] or ''
        for label in METRIC_LABEL.finditer(raw):
            check(label.start() == end, 'Malformed live-token labels')
            labels[label[1]] = json.loads('"' + label[2] + '"')
            end = label.end()
        check(end == len(raw), 'Malformed live-token labels')
        if labels.get('mode') != 'decode' or any(labels.get(k) != v for k, v in args.metric_selectors.items()):
            continue
        if args.rank_label:
            check(args.rank_label in labels, 'Live-token counter missing DP rank')
            rank = int(labels[args.rank_label])
        else:
            check(args.dp_size == 1, 'Global counter cannot verify multiple DP ranks')
            rank = 0
        check(0 <= rank < args.dp_size and rank not in result, 'Ambiguous live-token series; filter --metric-label')
        value = float(m[3])
        check(math.isfinite(value) and value >= 0, 'Invalid live-token counter')
        result[rank] = value
        identities[rank] = sorted(labels.items())
    check(set(result) == set(range(args.dp_size)), 'Missing per-rank live decode counters')
    return result, identities


async def read_metrics(session, args):
    start = time.monotonic()
    async with session.get(args.metrics_url, timeout=3) as response:
        check(response.status == 200, f'Metrics HTTP {response.status}')
        body = await response.text()
        ranks = parse_metrics(body, args)
    result = dict(t=time.monotonic(), start=start, latency=time.monotonic() - start, ranks=ranks)
    try:
        result['live_tokens'], result['live_token_series'] = parse_live_tokens(body, args)
    except (BenchError, ValueError) as exc:
        result['live_token_error'] = str(exc) if isinstance(exc, BenchError) else 'Invalid live-token rank'
    return result


class CatchupGate:
    """Compare absolute backlog AND rolling token rates, with explicit bounds.

    Sampled counters cannot prove per-frame GPU identity or sub-poll freshness.
    The tolerance bounds HTTP sampling skew, normal in-flight output and one
    MTP packet/request. A constant large backlog must not pass a rate-only test.
    """
    def __init__(self, args, baseline):
        self.args, self.baseline = args, baseline
        self.history = []
        self.previous = baseline['live_tokens']
        self.rejections = 0

    def observe(self, sample, records):
        a = self.args
        if 'live_tokens' not in sample:
            self.history.clear()
            self.rejections += 1
            return dict(ready=False, reason=sample.get('live_token_error', 'live counter unavailable'))
        check(sample['live_token_series'] == self.baseline['live_token_series'],
              'Live-token series identity changed during this cohort; cannot verify catch-up')
        values = sample['live_tokens']
        check(all(values[r] >= self.previous[r] for r in values),
              'Live-token counter reset during this cohort; cannot verify catch-up')
        self.previous = values
        client = {rank: 0 for rank in range(a.dp_size)}
        started = dict.fromkeys(client, 0)
        for record in records:
            check(record.actual_rank is None or record.actual_rank == record.prompt.rank,
                  'Server did not honor requested DP rank')
            client[record.prompt.rank] += record.tokens
            started[record.prompt.rank] += int(record.tokens > 0)
        current = dict(t=sample['t'], server={r: values[r] - self.baseline['live_tokens'][r] for r in client},
                       client=client, started=started)
        self.history.append(current)
        cutoff = current['t'] - a.catchup_window_seconds
        # Retain a bracketing sample at/before the rolling window boundary.
        while len(self.history) > 2 and self.history[1]['t'] <= cutoff:
            self.history.pop(0)
        old = self.history[0]
        dt = current['t'] - old['t']
        if dt < a.catchup_window_seconds:
            return dict(ready=False, reason='collecting_live_token_window')
        rows = []
        for rank in client:
            server_delta = current['server'][rank] - old['server'][rank]
            client_delta = client[rank] - old['client'][rank]
            prefill_delta = started[rank] - old['started'][rank]
            expected_delta = server_delta + prefill_delta
            rate = server_delta / dt
            slack = sum(r.prompt.rank == rank for r in records) * a.catchup_token_slack
            # One initial prefill token/request is absent from mode=decode.
            gap = current['server'][rank] + started[rank] - client[rank]
            limit = slack + rate * (a.catchup_max_lag_seconds + sample['latency'])
            rate_error = abs(client_delta - expected_delta)
            rate_ok = server_delta > 0 and client_delta > 0 and rate_error <= (
                slack + expected_delta * a.catchup_rate_tolerance + rate * sample['latency'])
            ready = rate_ok and abs(gap) <= limit
            rows.append(dict(rank=rank, ready=ready, server_tokens=current['server'][rank],
                             client_tokens=client[rank], prefill_tokens=started[rank],
                             server_rate=rate, client_rate=client_delta / dt,
                             backlog_tokens=gap, backlog_limit_tokens=limit,
                             backlog_seconds=gap / rate if rate > 0 else None,
                             rate_ok=rate_ok))
        ready = all(row['ready'] for row in rows)
        if not ready:
            self.rejections += 1
        return dict(ready=ready, reason='within_sampled_bounds' if ready else 'token_backlog_or_rate_mismatch',
                    interval_seconds=dt, ranks=rows)


async def prepare_catchup(session, args, use_metrics):
    """Establish an exclusive idle baseline AFTER warmup/probe, BEFORE formal requests."""
    reason = None
    sample = None
    if args.catchup_check == 'off':
        reason = 'explicitly_disabled'
    elif args.backend != 'sglang' or not args.route:
        reason = 'requires_native_completion_counts_and_pinned_DP'
    elif not use_metrics:
        reason = 'per_rank_metrics_unavailable'
    else:
        sample = await read_metrics(session, args)
        if 'live_tokens' not in sample:
            reason = sample.get('live_token_error', 'live counters unavailable')
    if reason:
        check(args.catchup_check != 'required', 'Cannot verify SSE catch-up: ' + reason)
        log('WARNING: SSE catch-up UNVERIFIED: ' + reason)
        return None, dict(status='disabled' if args.catchup_check == 'off' else 'unverified', reason=reason)
    # A preceding frontend response may finish before the running gauge clears.
    # Wait without aborting existing work; never baseline over foreign traffic.
    await wait_idle(session, args, {}, time.monotonic() + args.cleanup_timeout)
    sample = await read_metrics(session, args)
    check(metrics_ready(sample['ranks'], 0) and 'live_tokens' in sample,
          'No valid idle live-token baseline; formal requests were not sent')
    return CatchupGate(args, sample), dict(status='enabled',
        counter='sglang:realtime_tokens_total{mode="decode"}', baseline=sample,
        initial_prefill_output_tokens_per_request=1, window_seconds=args.catchup_window_seconds,
        max_lag_seconds=args.catchup_max_lag_seconds, token_slack_per_request=args.catchup_token_slack,
        rate_tolerance=args.catchup_rate_tolerance,
        scope='sampled per-rank accepted output counts within configured bounds, not GPU-step identity')


async def wait_idle(session, args, state, deadline):
    """Confirm all selected ranks idle; never abort existing/foreign traffic."""
    consecutive, last_report = 0, 0.0
    while time.monotonic() < deadline:
        try:
            sample = await asyncio.wait_for(read_metrics(session, args),
                timeout=max(.001, min(3, deadline - time.monotonic())))
            state['last_ranks'] = sample['ranks']
            idle = sample['latency'] <= args.max_metric_gap and all(
                all(v == 0 for v in fields.values()) for fields in sample['ranks'].values())
            consecutive = consecutive + 1 if idle else 0
            if consecutive >= args.cleanup_polls:
                state.update(status='verified', confirmation='sampled_per_rank_idle_metrics',
                             idle_polls=consecutive)
                return
        except Exception as exc:
            consecutive = 0
            state['last_metric_error'] = type(exc).__name__
        if time.monotonic() - last_report >= 5:
            log(f'Waiting for idle ranks: {consecutive}/{args.cleanup_polls} polls; '
                f'last={state.get("last_ranks", "unavailable")}')
            last_report = time.monotonic()
        await asyncio.sleep(min(args.poll_interval, max(0, deadline - time.monotonic())))
    state.update(status='failed', reason='idle_confirmation_timeout')
    raise BenchError('Cleanup/idle timeout: selected ranks are not confirmed idle. '
                     'No next test point will be started; other requests are never aborted.')


async def cancel_cohort(session, args, records, tasks, state):
    """Close readers, abort only this cohort's native RIDs, then prove idle.

    An HTTP 200 from /abort_request is acknowledgement, not completion.
    Final usage is deliberately not required; cached INPUT counters must
    already have been checked at the end of the measured window.
    """
    started = time.monotonic()
    deadline = started + args.cleanup_timeout
    state.update(status='pending', method=('native_rid_abort_and_stream_close'
                 if args.backend == 'sglang' else 'stream_close'),
                 abort_requests_accepted=0)
    for task in tasks:
        if not task.done():
            task.cancel()
    try:
        await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True),
                               timeout=max(.001, deadline - time.monotonic()))
        if args.backend == 'sglang':
            prefix = f'ssebench_{args.run_id}_'
            rids = {r.request_id for r in records if r.submitted and r.request_id}
            check(all(rid.startswith(prefix) and len(rid) > len(prefix) for rid in rids),
                  'Refuse to abort a request not owned by this benchmark')
            state['abort_requests_sent'] = len(rids)
            semaphore = asyncio.Semaphore(16)
            async def abort_one(rid):
                async with semaphore:
                    try:
                        async with session.post(args.base_url + '/abort_request',
                            json={'rid': rid, 'abort_all': False},
                            timeout=max(.001, min(10, deadline - time.monotonic()))) as response:
                            # Native SGLang returns an EMPTY 200; do not call json().
                            if response.status == 200:
                                state['abort_requests_accepted'] += 1
                                return None
                            return f'HTTP {response.status}'
                    except Exception as exc:
                        return type(exc).__name__
            errors = await asyncio.wait_for(asyncio.gather(*(abort_one(rid) for rid in rids)),
                                            timeout=max(.001, deadline - time.monotonic()))
            errors = [error for error in errors if error]
            if errors:
                state['abort_error_count'] = len(errors)
                state['abort_error_types'] = sorted(set(errors))
                raise BenchError('Own-request abort failed; no next test point will be started. '
                                 'Check /abort_request support/admin permission and cleanup metadata.')
        log('Own streams closed; confirming selected ranks have no running/waiting requests.')
        await wait_idle(session, args, state, deadline)
    except BaseException as exc:
        state.update(status='interrupted' if isinstance(exc, asyncio.CancelledError) else 'failed',
                     error_type=type(exc).__name__)
        raise
    finally:
        state['seconds'] = time.monotonic() - started


def measured_result(records, args, begin, end, samples, use_metrics, final_usage_complete, catchup=None):
    """Snapshot immutable window observations BEFORE cancelling or draining."""
    statuses = [cache_status(r, args) for r in records]
    result = summarize(records, begin, end)
    result.update(
        measurement_valid=True,
        timestamp_basis='complete SSE frame last-byte application read, before JSON parsing; not NIC arrival',
        catchup_verification=('sampled_live_tokens_verified' if catchup and catchup['status'] == 'verified'
                              else 'disabled' if catchup and catchup['status'] == 'disabled' else 'unverified'),
        catchup=catchup or dict(status='unverified'),
        timing_warnings=([] if catchup and catchup['status'] == 'verified' else
                         ['Client/server catch-up not verified; do not interpret a low SSE gap as GPU speedup']) +
                        (['Multiple progress frames completed in the same application read; burst delivery present']
                         if result['same_read_gap_fraction'] > .05 else []),
        output_handling='cancel_after_window' if args.stop_after_measure else 'drain_to_completion',
        final_usage_complete=final_usage_complete,
        usage_scope='final' if final_usage_complete else 'stream_observed_at_window_end',
        sse_times_scope='measurement_window_only' if args.stop_after_measure else 'full_request',
        bs_verification='sampled_per_rank_metrics' if use_metrics else 'client_only_unverified',
        dp_routing=('reported_rank_verified_all' if args.route and all(r.actual_rank is not None for r in records)
                    else 'requested_native_unverified' if args.route else 'service_assigned_unverified'),
        cache_verification=('report_only' if args.warmup_strategy == 'once-max' else
                            'reported_hit_all_requests' if all(s == 'reported_hit' for s in statuses) else 'unverified'),
        cache_summary=summarize_cache(records),
        cache_scope='server reported cached tokens; GPU residency/tier not proven by this field alone',
        metrics_samples=list(samples),
        requests=[dict(**r.prompt.manifest(), request_id=r.request_id, cached_tokens=r.cached_tokens,
                       actual_prompt_tokens=r.prompt_tokens,
                       actual_rank=r.actual_rank, completion_tokens=r.tokens,
                       cancelled=r.cancelled,
                       read_diagnostics=dict(r.read_stats),
                       window_progress=[dict(received_s=t - begin, completion_tokens=count,
                                             read_to_parsed_ms=lag * 1000, json_parse_ms=parse * 1000)
                                        for t, count, lag, parse in r.observations if begin <= t <= end],
                       sse_times_relative_s=[t - begin for t in r.events
                                            if not args.stop_after_measure or begin <= t <= end])
                  for r in records])
    return result


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
    lags = [lag * 1000 for r in records for t, _, lag, _ in r.observations if begin <= t <= end]
    return dict(window_seconds=end - begin, request_count=len(records),
                sse_gap_mean_ms=statistics.mean([r['mean_ms'] for r in per_request]),
                sse_gap_median_ms=statistics.median([r['median_ms'] for r in per_request]),
                pooled_sse_gap_p99_ms=quantile(all_gaps, .99), per_request=per_request,
                read_to_parsed_p99_ms=quantile(lags, .99) if lags else None,
                read_to_parsed_max_ms=max(lags) if lags else None,
                same_read_gap_fraction=sum(gap == 0 for gap in all_gaps) / len(all_gaps),
                metric='client SSE progress interval; not tokens, GPU steps or MTP-normalized TPOT')


async def measure(session, args, prompts, bs, use_metrics, on_window=None):
    check(not args.stop_after_measure or use_metrics,
          '--stop-after-measure requires readable per-rank metrics to confirm cleanup')
    gate, catchup = await prepare_catchup(session, args, use_metrics)
    records = [Record(p) for p in prompts]
    tasks = [asyncio.create_task(request(session, args, r, args.output_tokens)) for r in records]
    deadline = time.monotonic() + args.wait_timeout
    ready_since = begin = None
    samples = []
    last_report, last_metric = 0.0, None
    result, cleanup_attempted = None, False
    checked_observations = [0] * len(records)
    try:
        while True:
            now = time.monotonic()
            errors = [r.error for r in records if r.error]
            check(not errors, f'Measured request failed: {errors[0] if errors else ""}')
            # Finishing early invalidates the common window, no survivor bias.
            check(not any(r.end is not None for r in records),
                  'A request finished before the common steady window ended. Increase --output-tokens '
                  '(and capacity), or --ignore-eos if supported; no partial cohort is reported.')
            check(now <= deadline, 'Stable decode timeout; insufficient running BS/cache/capacity/output budget '
                  'or SSE client has not caught up; inspect timing/catchup diagnostics')
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
                    sample = None
            now = time.monotonic()
            parse_ok = True
            for i, record in enumerate(records):
                recent = record.observations[checked_observations[i]:]
                if any(row[2] > args.client_max_frame_lag for row in recent):
                    parse_ok = False
                checked_observations[i] = len(record.observations)
            client_ready = parse_ok and all(
                len(r.events) >= args.min_events and now - r.events[-1] <= args.max_event_stall
                and r.observations[-1][2] <= args.client_max_frame_lag for r in records)
            catchup_ok = True
            if gate is not None:
                if not metric_ok:
                    gate.history.clear()
                state = (gate.observe(sample, records) if sample is not None else
                         dict(ready=False, reason='metrics_read_failed'))
                catchup_ok = state['ready']
                if sample is not None:
                    sample['catchup'] = state
                catchup['last_check'] = state
            if client_ready and metric_ok and catchup_ok:
                if ready_since is None:
                    ready_since = now
                if begin is None and now - ready_since >= args.settle_seconds:
                    begin = now
                    log(f'Stable window started: {args.measure_seconds:g}s; '
                        f'{"per-rank metrics checked" if use_metrics else "CLIENT ONLY, per-rank BS unverified"}; '
                        f'catch-up={"live tokens checked" if gate else catchup["status"]}')
                if begin is not None and now - begin >= args.measure_seconds:
                    end = now
                    break
            else:
                ready_since = begin = None
            if now - last_report >= 15:
                log(f'Waiting/measuring: progressed {sum(len(r.events) >= args.min_events for r in records)}/{len(records)}, '
                    f'metrics_ready={metric_ok if use_metrics else "unverified"}, '
                    f'catchup={catchup.get("last_check", {}).get("reason", catchup["status"])}')
                last_report = now
            await asyncio.sleep(args.poll_interval)
        # Requests may have completed while reading the bracketing metrics sample.
        check(all(r.end is None or r.end >= end for r in records), 'Cohort ended inside measurement window')
        check(not any(r.error for r in records), 'A measured request failed; case invalid')
        if gate is not None:
            catchup.update(status='verified', rejected_polls=gate.rejections)
        if args.stop_after_measure:
            result = measured_result(records, args, begin, end, samples, use_metrics, False, catchup)
            result['cleanup'] = dict(status='pending')
            if on_window is not None:
                await on_window(result)
            log('Steady window saved; cancelling only this cohort (no full-output drain).')
            cleanup_attempted = True
            await cancel_cohort(session, args, records, tasks, result['cleanup'])
            for row, record in zip(result['requests'], records):
                row['cancelled'] = record.cancelled
        else:
            log('Steady window captured; draining outputs to record final usage (no new requests).')
            await asyncio.gather(*tasks)
            check(not any(r.error for r in records), 'A request failed after measurement; case invalid')
            result = measured_result(records, args, begin, end, samples, use_metrics, True, catchup)
        return result
    except BaseException as exc:
        if isinstance(exc, BenchError):
            # Attach only counters/timing, never request text or response bodies.
            exc.diagnostics = dict(catchup=catchup, metrics_samples=samples[-20:],
                read_diagnostics=[dict(index=r.prompt.index, **r.read_stats,
                    last_read_to_parsed_ms=r.observations[-1][2] * 1000 if r.observations else None)
                                  for r in records])
        if args.stop_after_measure and not cleanup_attempted:
            state = result['cleanup'] if result is not None else {}
            try:
                await cancel_cohort(session, args, records, tasks, state)
            except Exception as exc:
                log(f'Failure cleanup did not finish: {type(exc).__name__}; no next point will be started.')
        raise
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


@contextmanager
def atomic_text_output(path):
    """Replace one complete file; interruption cannot truncate its prior version."""
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='',
                                         dir=path.parent, prefix=f'.{path.name}.',
                                         suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def save_results(out, results, sweep_plans, plot):
    """Checkpoint between cases, never in an active SSE measurement window.

    Atomicity is per file, not a transaction across JSON/CSV/SVG. Full JSON is
    streamed to disk to avoid a second large in-memory serialization buffer.
    Only validated, completed cases are plotted by plot_svg().
    """
    with atomic_text_output(out / 'results.json') as stream:
        json.dump(results, stream, ensure_ascii=False, indent=2)
    with atomic_text_output(out / 'sweep_plan.json') as stream:
        json.dump(sweep_plans, stream, ensure_ascii=False, indent=2)
    with atomic_text_output(out / 'summary.csv') as stream:
        fields = ['status', 'bs_per_dp', 'mean_input_tokens_requested', 'prefix_workload',
                  'selection', 'selected_bucket', 'planned_mean_input_tokens', 'actual_mean_input_tokens',
                  'input_length_origin', 'theoretical_input_limit_tokens',
                  'sse_gap_mean_ms', 'sse_gap_median_ms', 'pooled_sse_gap_p99_ms',
                  'bs_verification', 'cache_verification', 'output_handling',
                  'catchup_verification', 'read_to_parsed_p99_ms', 'read_to_parsed_max_ms', 'same_read_gap_fraction',
                  'final_usage_complete', 'cleanup_status']
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for result in results:
            measurement = result.get('measurement', {})
            writer.writerow({**result, **measurement,
                             'cleanup_status': measurement.get('cleanup', {}).get('status')})
    # A separate incremental cache report makes misses visible without turning
    # them into failed steady-decode results in once-max mode.
    cache_rows = []
    for result in results:
        measurement = result.get('measurement', {})
        cache_rows.append(dict(status=result['status'], bs_per_dp=result['bs_per_dp'],
            mean_input_tokens_requested=result.get('mean_input_tokens_requested'),
            warmup_strategy=result.get('warmup_strategy', 'per-case'),
            cache_verification=measurement.get('cache_verification'),
            **measurement.get('cache_summary', {})))
    with atomic_text_output(out / 'cache_hits.json') as stream:
        json.dump(cache_rows, stream, ensure_ascii=False, indent=2)
    with atomic_text_output(out / 'cache_hits.csv') as stream:
        fields = ['status', 'bs_per_dp', 'mean_input_tokens_requested', 'warmup_strategy',
                  'cache_verification', 'coverage', 'request_count', 'reported_request_count',
                  'unknown_request_count', 'input_tokens_reported', 'cached_tokens', 'uncached_tokens',
                  'weighted_hit_ratio', 'mean_request_hit_ratio', 'min_request_hit_ratio', 'max_request_hit_ratio']
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(cache_rows)
    if plot:
        plot_svg(results, out / 'sse_gap.svg')


def plot_svg(results, path):
    good = [r for r in results if r.get('status') == 'ok']
    if not good:
        return
    w, h, pad = 900, 500, 70
    def xvalue(r):
        return r.get('actual_mean_input_tokens', r['mean_input_tokens_requested']) / 1024
    xs = [xvalue(r) for r in good]
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
        group = sorted((r for r in good if r['bs_per_dp'] == bs), key=xvalue)
        pts = [(pad + (w - 2 * pad) * xvalue(r) / xmax,
                h - pad - (h - 2 * pad) * r['measurement']['sse_gap_median_ms'] / ymax) for r in group]
        color = colors[j % len(colors)]
        out.append(f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x,y in pts)}" fill="none" stroke="{color}" stroke-width="2"/>')
        for (x, y), r in zip(pts, group):
            label = html.escape(f"BS{bs}, {r['prefix_workload']}, {r['measurement']['bs_verification']}, "
                                f"catch-up={r['measurement'].get('catchup_verification', 'unverified')}, "
                                f"{r['measurement']['sse_gap_median_ms']:.3f} ms, "
                                f"requested={r['mean_input_tokens_requested']/1024:g}K, "
                                f"x={xvalue(r):g}K, bucket={r.get('selected_bucket')}")
            verified = r['measurement'].get('catchup_verification') == 'sampled_live_tokens_verified'
            out.append(f'<circle cx="{x}" cy="{y}" r="4" fill="{color if verified else "white"}" '
                       f'stroke="{color}"><title>{label}</title></circle>')
        out.append(f'<text x="{pad+j*140}" y="{h-12}" fill="{color}" font-size="13">BS/rank={bs}</text>')
    for x in sorted(set(xs)):
        px = pad + (w - 2 * pad) * x / xmax
        out.append(f'<text x="{px-12}" y="{h-pad+20}" font-size="12">{x:g}K</text>')
    out += ['<text x="14" y="55" font-size="12">ms</text>',
            '<text x="70" y="45" font-size="12">X: actual mean input K for real text; requested K for synthetic</text>',
            '<text x="70" y="62" font-size="11">Hollow point = catch-up UNVERIFIED; filled = sampled live-token checks passed</text>', '</svg>']
    with atomic_text_output(path) as stream:
        stream.write('\n'.join(out))


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
    # A long warmup may outlive the server's idle keep-alive timeout. Reusing
    # those connections for non-idempotent POSTs can race a server-side close.
    # Each request gets one connection; its SSE stream still stays open for the
    # entire generation, so this does not add a handshake per measured event.
    async with aiohttp.ClientSession(headers=headers, connector=aiohttp.TCPConnector(limit=0, force_close=True),
                                     trust_env=False) as session:
        info = await get_info(session, args.base_url)
        if args.backend == 'auto':
            args.backend = 'sglang' if 'max_total_num_tokens' in info or 'internal_states' in info else 'openai'
        check(not (args.backend == 'openai-chat' and args.mode not in (4, 5)),
              'Exact synthetic token prompts require /generate or /v1/completions. '
              'Chat-only APIs are supported with --mode 4/5 (server applies its own template).')
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
        if not args.tokenizer and args.mode not in (4, 5):
            args.tokenizer = info.get('tokenizer_path') or info.get('model_path')
        check(args.tokenizer or args.mode in (4, 5), 'Supply --tokenizer matching the serving model; no guessed vocabulary')
        tokenizer = None
        if args.tokenizer:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=False)
        shadow_summary = {}
        shadows = (load_shadow(args.shadow_dir, tokenizer, args.shadow_length_field,
                               args.shadow_bucket_tokens, shadow_summary) if args.mode in (4, 5) else None)
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
        initial_metrics = None
        if use_metrics:
            try:
                initial_metrics = await read_metrics(session, args)
            except Exception as exc:
                check(args.steady_check != 'required', f'Cannot verify DP metrics: {type(exc).__name__}')
                use_metrics = False
                log('WARNING: metrics unavailable/ambiguous; CLIENT-ONLY results. Use --steady-check required '
                    'to refuse this, or configure metric names/labels.')
        if args.catchup_check == 'required' and not args.plan_only:
            check(args.backend == 'sglang' and args.route and use_metrics and initial_metrics
                  and 'live_tokens' in initial_metrics,
                  'Cannot verify SSE catch-up: requires native completion counts, pinned DP, '
                  'and per-rank sglang:realtime_tokens_total{mode="decode"}; '
                  'finished-request generation_tokens_total is not sufficient')
        check(not args.stop_after_measure or args.plan_only or use_metrics,
              '--stop-after-measure requires readable per-rank metrics to confirm cleanup; '
              'without metrics, omit it and drain outputs normally')
        run_seed = (args.replay_seed if args.replay_seed is not None else args.seed if args.mode in (0, 1, 2)
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
            shadow_summary=shadow_summary,
            run_seed=run_seed, notes='No generated text or source prompts saved; seeds do not guarantee deterministic outputs'),
            ensure_ascii=False, indent=2), encoding='utf-8')
        results = []
        sweep_plans = []
        try:
            once_max = args.warmup_strategy == 'once-max'
            synthetic_bank = None
            batch_sizes = sorted(set(args.batch_sizes), reverse=True) if once_max else args.batch_sizes
            if once_max:
                if args.stop_after_measure and not args.plan_only:
                    log('Checking selected ranks idle before the covering warmup; no existing traffic is aborted.')
                    await wait_idle(session, args, {}, time.monotonic() + args.cleanup_timeout)
                planned = [(bs, *plan_input_sweep(args, capacity, page, bs, info.get('context_length')))
                           for bs in batch_sizes]
                limits = [s.get('effective_max_running_requests_per_dp')
                          for s in info.get('internal_states', []) if isinstance(s, dict)]
                limits = [v for v in limits if type(v) is int]
                feasible_bs = [bs for bs, entries, _ in planned if entries]
                if feasible_bs and limits:
                    check(max(feasible_bs) <= min(limits), 'Requested BS exceeds service per-DP max running requests')
                synthetic_bank, warm_prompts, warm_plan = build_once_max_bank(
                    args, tokenizer, capacity, page, run_seed, planned)
                context = info.get('context_length')
                if type(context) is int:
                    check(all(len(p.ids) + args.output_tokens <= context for p in warm_prompts),
                          'Once-max input+output exceeds service context_length; no warmup sent.')
                log(f'Once-max: {len(warm_prompts)} covering warmup requests, '
                    f'max BS/rank={warm_plan["max_bs_per_dp"]}; descending BS then length; '
                    'cache hits are report-only, no per-case warmup/probe.')
                warm_plan['status'] = 'planned_not_run' if args.plan_only else 'running'
                with atomic_text_output(out / 'once_warmup.json') as stream:
                    json.dump(warm_plan, stream, ensure_ascii=False, indent=2)
                if not args.plan_only:
                    try:
                        warm_plan['responses'] = await warm_phase(session, args, warm_prompts, False)
                        warm_plan['status'] = 'ok'
                    except BaseException:
                        warm_plan['status'] = 'failed_or_interrupted'
                        raise
                    finally:
                        with atomic_text_output(out / 'once_warmup.json') as stream:
                            json.dump(warm_plan, stream, ensure_ascii=False, indent=2)
                del warm_prompts
            for bs in batch_sizes:
                entries, sweep = plan_input_sweep(args, capacity, page, bs, info.get('context_length'))
                if once_max:
                    entries = sorted(entries, key=lambda e: e['mean'], reverse=True)
                    sweep['measured_input_tokens'] = [e['mean'] for e in entries]
                sweep_plans.append(sweep)
                if sweep.get('omitted_requested_input_tokens'):
                    prefix_kind = 'identical-prefix' if args.mode == 0 else 'independent'
                    log(f'BS/rank={bs}: {prefix_kind} input bound '
                        f'{sweep["effective_input_limit_tokens"] / 1024:g}K '
                        f'({sweep["limiting_factor"]}); larger requested lengths omitted. '
                        f'Automatic boundary: {sweep["auto_added_input_tokens"] or "already on grid / no positive input fits"}. '
                        'Continue remaining BS values after this grid.')
                if not entries:
                    results.append(dict(bs_per_dp=bs, status='capacity_limited',
                        theoretical_input_limit_tokens=sweep['effective_input_limit_tokens'],
                        reason='No positive input fits this BS after output, page and safety reserves'))
                    save_results(out, results, sweep_plans, args.plot)
                    continue
                for entry in entries:
                    mean = entry['mean']
                    case = dict(bs_per_dp=bs, mean_input_tokens_requested=mean, status='running',
                                warmup_strategy=args.warmup_strategy,
                                input_length_origin=entry['origin'],
                                theoretical_input_limit_tokens=sweep.get('effective_input_limit_tokens'))
                    results.append(case)
                    try:
                        prompts, plan = make_case(args, tokenizer, capacity, page, bs, mean, run_seed,
                                                  shadows, synthetic_bank=synthetic_bank)
                    except CapacityError as exc:
                        if args.allow_shared_prefix:
                            raise
                        case.update(status='capacity_limited', reason=str(exc))
                        sweep['stopped_at_input_tokens'] = mean
                        log(f'BS/rank={bs}: {exc}; ending this BS, continuing the next BS.')
                        save_results(out, results, sweep_plans, args.plot)
                        break
                    case.update(plan)
                    context = info.get('context_length')
                    if isinstance(context, int):
                        check(all(p.input_tokens + args.output_tokens <= context for p in prompts),
                              'Input+output exceeds service context_length')
                    effective_limits = [s.get('effective_max_running_requests_per_dp')
                                        for s in info.get('internal_states', []) if isinstance(s, dict)]
                    limits = [v for v in effective_limits if type(v) is int]
                    if limits:
                        check(bs <= min(limits), 'Requested BS exceeds service per-DP max running requests')
                    case['prompts'] = [p.manifest() for p in prompts]
                    case['warmup_concurrency_effective'] = (0 if once_max else
                                                           effective_warmup_concurrency(args, len(prompts)))
                    target_label = 'bucket_selector' if args.mode == 5 else 'mean'
                    log(f'Mode {args.mode}, BS/rank={bs}, {target_label}={mean:,}, global={len(prompts)}, '
                        f'prefix={plan["prefix_workload"]}, KV estimate/rank '
                        f'{min(p["estimated_kv_tokens"] for p in plan["rank_plans"]):,}..'
                        f'{max(p["estimated_kv_tokens"] for p in plan["rank_plans"]):,}')
                    if args.plan_only:
                        case['status'] = 'planned_not_run'
                        save_results(out, results, sweep_plans, args.plot)
                        continue
                    # Persist prior completed cases plus the active case before a
                    # potentially long warmup. No generated output is serialized.
                    save_results(out, results, sweep_plans, args.plot)
                    if args.stop_after_measure:
                        log('Checking selected ranks idle before this point; no existing traffic is aborted.')
                        await wait_idle(session, args, {}, time.monotonic() + args.cleanup_timeout)
                    if once_max:
                        case['cache_probe_skipped'] = True
                        log('Send formal cohort directly; server may compute missing KV. '
                            'Wait for common steady window; no extra warmup/probe.')
                    else:
                        log('Warmup all exact prompts (1 generated token each)')
                        await warm_phase(session, args, prompts, False)
                        case.update(actual_shadow_plan(prompts, args, capacity, page, context))
                        case['prompts'] = [p.manifest() for p in prompts]
                        if case.get('actual_mean_input_tokens') is not None:
                            log(f'Shadow planned mean={statistics.mean(p.planned_tokens for p in prompts):,.1f}; '
                                f'service actual mean={case["actual_mean_input_tokens"]:,.1f} tokens; '
                                f'selection={case["selection"]}, bucket={case["selected_bucket"]}')
                        log('Probe all exact prompts again; verify cache residency/hits before formal run')
                        case['cache_probe'] = await warm_phase(session, args, prompts, True)
                    async def checkpoint_window(measurement):
                        case.update(status='measured_cleanup_pending', measurement=measurement)
                        save_results(out, results, sweep_plans, args.plot)
                        log('Window JSON/CSV checkpoint saved before cancellation; cleanup is still pending.')

                    case['measurement'] = await measure(session, args, prompts, bs, use_metrics,
                                                        on_window=checkpoint_window)
                    case['status'] = 'ok'
                    m = case['measurement']
                    save_results(out, results, sweep_plans, args.plot)
                    log(f'SSE gap mean/median/P99={m["sse_gap_mean_ms"]:.3f}/'
                        f'{m["sse_gap_median_ms"]:.3f}/{m["pooled_sse_gap_p99_ms"]:.3f} ms; '
                        f'{m["bs_verification"]}, {m["cache_verification"]}, {m["catchup_verification"]}; '
                        f'read→parsed P99={m["read_to_parsed_p99_ms"]:.3f} ms')
                    cache = m['cache_summary']
                    ratio = cache['weighted_hit_ratio']
                    hit_label = f'{ratio:.2%}' if ratio is not None else 'unknown'
                    log(f'Formal input cache hit={hit_label}; '
                        f'coverage={cache["reported_request_count"]}/{cache["request_count"]}; '
                        'saved to cache_hits.json / cache_hits.csv')
                    log(f'Checkpoint saved: {sum(r["status"] == "ok" for r in results)} completed cases; '
                        f'JSON/CSV{"/SVG" if args.plot else ""} updated in {out}')
            check(any(r['status'] in {'ok', 'planned_not_run'} for r in results),
                  'No requested BS has a feasible input case; reduce BS/output or increase the KV budget.')
        except (asyncio.CancelledError, KeyboardInterrupt):
            if results and results[-1]['status'] in {'running', 'measured_cleanup_pending'}:
                results[-1].update(status='interrupted', error='Benchmark interrupted')
            raise
        except Exception as exc:
            if results and results[-1]['status'] in {'running', 'measured_cleanup_pending'}:
                status = 'cleanup_failed' if results[-1]['status'] == 'measured_cleanup_pending' else 'failed'
                results[-1].update(status=status,
                    error=str(exc) if isinstance(exc, BenchError) else type(exc).__name__)
                if getattr(exc, 'diagnostics', None):
                    results[-1]['failure_diagnostics'] = exc.diagnostics
            raise
        finally:
            save_results(out, results, sweep_plans, args.plot)


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--base-url', default='http://127.0.0.1:31000')
    p.add_argument('--model', default='hy4')
    p.add_argument('--tokenizer', help='Matching tokenizer path/HF ID; optional for modes 4/5 with reference lengths')
    p.add_argument('--backend', choices=['auto', 'sglang', 'openai', 'openai-chat'], default='auto')
    p.add_argument('--mode', type=int, choices=[0, 1, 2, 3, 4, 5], default=1,
                   help='0=identical shared prefix, 1=uniform independent, 2=mixed fixed, '
                        '3=mixed random, 4=real text matched mean, 5=random real-text bucket')
    p.add_argument('--dp-size', type=int, default=8, help='Attention-DP ranks, NOT TP size')
    p.add_argument('--batch-sizes', type=int, nargs='+', default=[32], help='Requests per attention-DP rank')
    p.add_argument('--input-lengths-k', type=float, nargs='+', default=[32],
                   help='Per-rank target mean; mode 5 selects the bucket containing this length. K=1024')
    p.add_argument('--min-length-ratio', type=float, default=.25)
    p.add_argument('--max-length-ratio', type=float, default=1.75)
    p.add_argument('--seed', type=int, default=60)
    p.add_argument('--generation-seed', type=int, default=60)
    p.add_argument('--replay-seed', type=int, help='Replay recorded run_seed for modes 3/4/5')
    p.add_argument('--shadow-dir', help='Modes 4/5: text/messages file or folder; .json/.jsonl and gzip supported; '
                   'default: bundled shadow/basic.jsonl.gz')
    p.add_argument('--shadow-mean-tolerance', type=float, default=.01)
    p.add_argument('--shadow-length-field', help='Dotted reference token-count field when no tokenizer, e.g. meta.prompt_tokens_offline')
    p.add_argument('--shadow-bucket-boundaries-k', type=float, nargs='+',
                   help='Override source buckets, e.g. 0 32 64 128 208 256; last bucket is unbounded. '
                        'Default: source bucket names/metadata, else these same boundaries.')
    cap = p.add_mutually_exclusive_group()
    cap.add_argument('--max-total-kvcache-length-k', type=float, help='Manual token capacity PER RANK, K=1024; default auto')
    cap.add_argument('--kv-capacity-tokens', type=int, help='Manual token capacity PER RANK (literal tokens)')
    p.add_argument('--capacity-fraction', type=float, default=.95, help='Use this fraction of reported/manual token pool')
    p.add_argument('--page-size', type=int, help='KV page size in tokens; default server value, else conservative 256')
    p.add_argument('--allow-shared-prefix', action='store_true',
                   help='Opt-in capacity fallback for modes 1-3; mode 0 is always fully shared')
    p.add_argument('--min-unique-suffix-tokens', type=int, default=128)
    p.add_argument('--output-tokens', '--request-output-length', type=int, default=2048)
    p.add_argument('--temperature', type=float, default=.6)
    p.add_argument('--top-p', type=float, default=.95)
    p.add_argument('--ignore-eos', action='store_true', help='Nonstandard OpenAI extension; only if supported')
    p.add_argument('--warmup-concurrency', type=int,
                   help='Global warmup/probe client cap; omitted = all requests of current point (DP × BS)')
    p.add_argument('--warmup-strategy', choices=['per-case', 'once-max'], default='per-case',
                   help='once-max (modes 0-3): one covering warmup, descending BS/length, '
                        'no per-case warmup/probe; cache counters are report-only. Default: strict per-case.')
    p.add_argument('--min-cache-hit-ratio', type=float, default=.98,
                   help='Minimum cached/input token ratio in [0, 1]; 0 disables only this threshold, not the tail limit')
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
    p.add_argument('--catchup-check', choices=['auto', 'required', 'off'], default='auto',
                   help='All modes: compare native live completion tokens vs per-rank scheduler decode counters; '
                        'auto marks unsupported services unverified, required refuses them, off explicitly disables')
    p.add_argument('--catchup-window-seconds', type=float, default=2,
                   help='Rolling token-rate agreement window before/during steady measurement (default 2s)')
    p.add_argument('--catchup-max-lag-seconds', type=float, default=.25,
                   help='Maximum sampled token backlog in seconds, plus sampling skew and per-request token slack')
    p.add_argument('--catchup-rate-tolerance', type=float, default=.2,
                   help='Allowed rolling client/server token-rate difference (default 20%%), plus token/skew slack')
    p.add_argument('--catchup-token-slack', type=int, default=8,
                   help='In-flight/MTP token allowance per request; not an MTP time divisor (default 8 tokens)')
    p.add_argument('--client-max-frame-lag', type=float, default=.25,
                   help='All modes/backends: max latest-frame read-to-parsed lag before/during measurement (seconds)')
    p.add_argument('--measure-seconds', type=float, default=10)
    p.add_argument('--settle-seconds', type=float, default=3)
    p.add_argument('--min-events', type=int, default=5)
    p.add_argument('--max-event-stall', type=float, default=5)
    p.add_argument('--poll-interval', type=float, default=.5)
    p.add_argument('--max-metric-gap', type=float, default=3)
    p.add_argument('--wait-timeout', type=float, default=600)
    p.add_argument('--request-timeout', type=float, default=7200)
    p.add_argument('--stop-after-measure', action='store_true',
                   help='All modes: save window first, cancel own requests, confirm idle ranks before next point; '
                        'requires metrics. Default: drain full outputs.')
    p.add_argument('--cleanup-timeout', type=float, default=120,
                   help='Maximum seconds for own-request cancellation/idle confirmation; timeout stops the matrix')
    p.add_argument('--cleanup-polls', type=int, default=3,
                   help='Consecutive per-rank idle polls required after cancellation (default 3)')
    p.add_argument('--api-key-env', default='OPENAI_API_KEY')
    p.add_argument('--save-path', default='results')
    p.add_argument('--plot', action='store_true', help='Update dependency-free SVG chart after each completed case')
    p.add_argument('--plan-only', action='store_true', help='Read-only metadata/plan; no generation requests')
    p.add_argument('--self-test', action='store_true', help='Embedded unit + local mock HTTP tests; no GPU/model downloads')
    return p


def validate(args):
    check(args.warmup_strategy != 'once-max' or args.mode in (0, 1, 2, 3),
          '--warmup-strategy once-max supports only modes 0-3')
    for name in ('dp_size', 'output_tokens', 'min_events', 'min_unique_suffix_tokens', 'cleanup_polls'):
        check(getattr(args, name) > 0, f'{name} must be positive')
    check(args.warmup_concurrency is None or args.warmup_concurrency > 0,
          'warmup_concurrency must be positive when specified; omit for automatic concurrency')
    for name in ('measure_seconds', 'max_event_stall', 'poll_interval', 'max_metric_gap', 'wait_timeout',
                 'request_timeout', 'cleanup_timeout', 'catchup_window_seconds', 'catchup_max_lag_seconds',
                 'client_max_frame_lag'):
        check(math.isfinite(getattr(args, name)) and getattr(args, name) > 0, f'{name} must be positive/finite')
    check(all(bs > 0 for bs in args.batch_sizes), 'BS must be positive')
    check(all(math.isfinite(k) and k * 1024 >= 1 for k in args.input_lengths_k), 'Invalid input lengths')
    check(0 < args.capacity_fraction <= 1 and 0 <= args.min_cache_hit_ratio <= 1, 'Invalid capacity/cache fraction')
    check(math.isfinite(args.catchup_rate_tolerance) and 0 <= args.catchup_rate_tolerance < 1
          and args.catchup_token_slack >= 0, 'Invalid catch-up rate tolerance/token slack')
    check(0 < args.min_length_ratio <= 1 <= args.max_length_ratio < float('inf'), 'Invalid length bounds')
    check(0 <= args.shadow_mean_tolerance < 1 and args.settle_seconds >= 0, 'Invalid tolerance/settle time')
    check(args.cache_tail_tokens >= 0 and args.temperature >= 0 and 0 < args.top_p <= 1, 'Invalid sampling/cache parameters')
    check(math.isfinite(args.temperature) and math.isfinite(args.settle_seconds), 'Non-finite sampling/settle value')
    if args.mode in (4, 5) and not args.shadow_dir:
        args.shadow_dir = str(Path(__file__).resolve().parent / 'shadow' / 'basic.jsonl.gz')
    check(args.mode in (4, 5) or not (args.shadow_dir or args.shadow_length_field or args.shadow_bucket_boundaries_k),
          'Shadow data options require --mode 4/5')
    args.shadow_bucket_tokens = None
    if args.shadow_bucket_boundaries_k is not None:
        bounds = args.shadow_bucket_boundaries_k
        check(len(bounds) >= 2 and bounds[0] == 0
              and all(math.isfinite(n) and n >= 0 for n in bounds)
              and all(a < b for a, b in zip(bounds, bounds[1:])),
              'Bucket boundaries must start at 0 and be strictly increasing')
        args.shadow_bucket_tokens = [round(n * 1024) for n in bounds]
        check(all(a < b for a, b in zip(args.shadow_bucket_tokens, args.shadow_bucket_tokens[1:])),
              'Bucket boundaries collapse after token rounding')
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

        def encode(self, text, add_special_tokens=True):
            return [ord(ch) + 2 for ch in text]

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
        a.cleanup_timeout = 1
        a.cleanup_polls = 2
        a.min_events = 2
        a.catchup_window_seconds = .03
        for k, v in changes.items():
            setattr(a, k, v)
        validate(a)
        return a

    class PlanningTests(unittest.TestCase):
        def test_once_max_nested_streams_cover_every_case_all_synthetic_modes(self):
            for mode in range(4):
                a = config(mode=mode, warmup_strategy='once-max', batch_sizes=[1, 2, 4],
                           input_lengths_k=[.125, .25])
                sweeps = [(bs, *plan_input_sweep(a, 100000, 16, bs, 8192)) for bs in a.batch_sizes]
                bank, warm, meta = build_once_max_bank(a, FakeTokenizer(), 100000, 16, 60, sweeps)
                repeat, _, _ = build_once_max_bank(a, FakeTokenizer(), 100000, 16, 60, sweeps)
                changed, _, _ = build_once_max_bank(a, FakeTokenizer(), 100000, 16, 61, sweeps)
                self.assertEqual(len(warm), 2 if mode == 0 else 8)
                self.assertEqual(meta['max_bs_per_dp'], 4)
                self.assertEqual([p.sha256 for p in bank.values()], [p.sha256 for p in repeat.values()])
                self.assertNotEqual([p.sha256 for p in bank.values()], [p.sha256 for p in changed.values()])
                for bs, entries, _ in sweeps:
                    for entry in entries:
                        prompts, plan = make_case(a, FakeTokenizer(), 100000, 16, bs, entry['mean'], 60,
                                                  synthetic_bank=bank)
                        self.assertTrue(plan['input_bank_reused'])
                        for p in prompts:
                            self.assertEqual(p.ids, bank[p.rank, p.index // a.dp_size].ids[:len(p.ids)])
                            self.assertFalse(set(p.ids) & {0, 1})
                        for rank in range(2):
                            self.assertEqual(statistics.mean(len(p.ids) for p in prompts if p.rank == rank), entry['mean'])
                        self.assertEqual(len({p.sha256 for p in prompts}), 1 if mode == 0 else len(prompts))

        def test_once_max_union_capacity_is_checked_not_only_individual_cases(self):
            a = config(warmup_strategy='once-max', batch_sizes=[1, 4], input_lengths_k=[.25, .5])
            sweeps = [(bs, *plan_input_sweep(a, 2048, 16, bs, 8192)) for bs in a.batch_sizes]
            for bs, entries, _ in sweeps:
                for e in entries:
                    make_case(a, FakeTokenizer(), 2048, 16, bs, e['mean'], 60)
            with self.assertRaisesRegex(CapacityError, 'union of all BS/length cases'):
                build_once_max_bank(a, FakeTokenizer(), 2048, 16, 60, sweeps)

        def test_once_max_explicit_shared_bank_preserves_prefixes_and_suffixes(self):
            a = config(warmup_strategy='once-max', allow_shared_prefix=True,
                       batch_sizes=[1, 2], input_lengths_k=[1], output_tokens=32)
            sweeps = [(bs, *plan_input_sweep(a, 1600, 16, bs)) for bs in a.batch_sizes]
            bank, warm, meta = build_once_max_bank(a, FakeTokenizer(), 1600, 16, 60, sweeps)
            self.assertTrue(any(p.shared_tokens for p in warm))
            self.assertTrue(all(len(p.ids) - p.shared_tokens >= a.min_unique_suffix_tokens for p in warm))
            self.assertTrue(all(r['estimated_union_and_output_tokens'] <= r['usable_capacity_tokens']
                                for r in meta['rank_plans']))
            for bs in a.batch_sizes:
                prompts, _ = make_case(a, FakeTokenizer(), 1600, 16, bs, 1024, 60, synthetic_bank=bank)
                self.assertTrue(all(p.ids == bank[p.rank, p.index // 2].ids for p in prompts))

        def test_once_max_cache_report_only_keeps_protocol_and_routing_checks(self):
            a = config(warmup_strategy='once-max', min_cache_hit_ratio=1, cache_tail_tokens=0)
            r = Record(Prompt(0, 0, array('I', [2] * 1024), 60))
            r.cached_tokens, r.prompt_tokens, r.actual_rank = 0, 1024, 0
            self.assertEqual(cache_status(r, a), 'reported_only')
            r.cached_tokens = None
            self.assertEqual(cache_status(r, a), 'unknown')
            for cached in (-1, 1025):
                r.cached_tokens = cached
                with self.assertRaises(BenchError):
                    cache_status(r, a)
            r.cached_tokens, r.actual_rank = 0, 1
            with self.assertRaisesRegex(BenchError, 'DP rank'):
                cache_status(r, a)
            for mode in (4, 5):
                with self.assertRaisesRegex(BenchError, 'only modes 0-3'):
                    config(mode=mode, warmup_strategy='once-max')

        def test_cache_summary_weighted_unweighted_partial_and_unknown(self):
            records = [Record(Prompt(i, i % 2, array('I', [2] * n), 60))
                       for i, n in enumerate((100, 200, 300))]
            records[0].cached_tokens, records[1].cached_tokens = 25, 150
            s = summarize_cache(records[:2])
            self.assertAlmostEqual(s['weighted_hit_ratio'], 175 / 300)
            self.assertEqual(s['mean_request_hit_ratio'], .5)
            self.assertEqual(s['coverage'], 'complete')
            s = summarize_cache(records)
            self.assertEqual(s['coverage'], 'partial')
            self.assertEqual(s['unknown_request_count'], 1)
            self.assertEqual(s['input_tokens_reported'], 300)
            self.assertEqual(len(s['by_planned_rank']), 2)
            self.assertIsNone(s['requests'][-1]['hit_ratio'])
            self.assertIsNone(summarize_cache(records[2:])['weighted_hit_ratio'])

        def test_mode0_identical_across_ranks_bs_and_nested_lengths(self):
            a = config(mode=0)
            short, plan = make_case(a, FakeTokenizer(), 100000, 16, 2, 128, 60)
            long, _ = make_case(a, FakeTokenizer(), 100000, 16, 4, 256, 60)
            repeat, _ = make_case(a, FakeTokenizer(), 100000, 16, 2, 128, 60)
            changed, _ = make_case(a, FakeTokenizer(), 100000, 16, 2, 128, 61)
            self.assertEqual(plan['prefix_workload'], 'fully_shared_synthetic')
            self.assertEqual(len(short), 4)
            self.assertEqual({p.rank for p in short}, {0, 1})
            self.assertEqual(len({digest_ids(p.ids) for p in short}), 1)
            self.assertEqual(len({digest_ids(p.ids) for p in long}), 1)
            self.assertEqual(short[0].ids, long[-1].ids[:128])
            self.assertEqual([p.manifest() for p in short], [p.manifest() for p in repeat])
            self.assertNotEqual(short[0].ids, changed[0].ids)
            self.assertTrue(all(p.shared_tokens == 128 for p in short))
            self.assertEqual([p.seed for p in short], [60, 61, 62, 63])

        def test_mode0_private_tail_and_output_budget(self):
            # 112 shared input + 4 * (16 private input tail + 32 output).
            self.assertEqual(identical_prefix_cost(128, 4, 17, 16), 304)
            self.assertEqual(identical_prefix_cost(129, 4, 17, 16), 320)
            a = config(mode=0, output_tokens=17, capacity_fraction=1)
            prompts, plan = make_case(a, FakeTokenizer(), 304, 16, 4, 128, 60)
            self.assertEqual(len(prompts), 8)
            self.assertTrue(all(r['estimated_kv_tokens'] == 304 for r in plan['rank_plans']))
            with self.assertRaisesRegex(CapacityError, 'Identical-prefix KV'):
                make_case(a, FakeTokenizer(), 304, 16, 4, 129, 60)
            a.allow_shared_prefix = True
            again, same = make_case(a, FakeTokenizer(), 304, 16, 4, 128, 60)
            self.assertEqual([p.manifest() for p in prompts], [p.manifest() for p in again])
            self.assertEqual(plan, same)

        def test_mode0_sweep_uses_shared_budget_not_bs_times_input(self):
            a = config(mode=0, output_tokens=8192, input_lengths_k=[4, 64, 100, 150, 190])
            entries, plan = plan_input_sweep(a, 3500 * 1024, 256, 64, 1048576)
            self.assertEqual([e['mean'] for e in entries], [n * 1024 for n in a.input_lengths_k])
            self.assertEqual(plan['policy'], 'identical_prefix_input_bound')
            self.assertEqual(plan['omitted_requested_input_tokens'], [])

        def test_mode0_sweep_boundary_and_output_exhaustion(self):
            a = config(mode=0, output_tokens=17, capacity_fraction=1, input_lengths_k=[.0625, .25])
            entries, plan = plan_input_sweep(a, 304, 16, 4)
            self.assertEqual([e['mean'] for e in entries], [64, 128])
            self.assertEqual(entries[-1]['origin'], 'automatic_kv_capacity_boundary')
            self.assertEqual(plan['effective_input_limit_tokens'], 128)
            self.assertEqual(plan_input_sweep(a, 191, 16, 4)[0], [])
            self.assertEqual(plan_input_sweep(a, 192, 16, 4)[0][0]['mean'], 16)
            a.allow_shared_prefix = True  # This flag must not bypass mode 0 sizing.
            self.assertEqual(plan_input_sweep(a, 304, 16, 4)[0], entries)

        def test_mode0_sweep_context_limit_and_boundary_inversion(self):
            a = config(mode=0, output_tokens=17, capacity_fraction=1, input_lengths_k=[.0625, .125])
            entries, plan = plan_input_sweep(a, 10000, 16, 4, 100)
            self.assertEqual([e['mean'] for e in entries], [64, 83])
            self.assertEqual(plan['limiting_factor'], 'context_limit')
            for bs in (1, 2, 16, 64):
                for page in (16, 256):
                    for capacity in (1, 512, 1000, 100000):
                        _, plan = plan_input_sweep(a, capacity, page, bs)
                        limit = plan['kv_input_limit_tokens']
                        if limit:
                            self.assertLessEqual(identical_prefix_cost(limit, bs, 17, page), capacity)
                            self.assertGreater(identical_prefix_cost(limit + 1, bs, 17, page), capacity)
                        else:
                            self.assertGreater(identical_prefix_cost(1, bs, 17, page), capacity)

        def test_default_sweep_adds_3500k_bs64_boundary(self):
            a = config(output_tokens=8192, input_lengths_k=[.5, 1, 2, 4, 8, 16, 32, 64, 100, 128, 160, 190, 200])
            entries, summary = plan_input_sweep(a, 3500 * 1024, 256, 64, 1048576)
            self.assertEqual(summary['effective_input_limit_tokens'], 44800)  # 43.75K
            self.assertEqual([e['mean'] for e in entries][-2:], [32768, 44800])
            self.assertEqual(entries[-1]['origin'], 'automatic_kv_capacity_boundary')
            self.assertEqual(summary['auto_added_input_tokens'], [44800])
            self.assertIn(200 * 1024, summary['omitted_requested_input_tokens'])
            self.assertLessEqual(rank_cost([44800] * 64, [0] * 64, 8192, 256), 3500 * 1024 * .95)
            self.assertGreater(rank_cost([44801] * 64, [0] * 64, 8192, 256), 3500 * 1024 * .95)

        def test_sweep_never_extends_past_requested_max(self):
            a = config(input_lengths_k=[.5, 1, 2])
            entries, summary = plan_input_sweep(a, 1000000, 16, 2, 8192)
            self.assertEqual([e['mean'] for e in entries], [512, 1024, 2048])
            self.assertEqual(summary['auto_added_input_tokens'], [])

        def test_sweep_existing_boundary_is_not_duplicated(self):
            a = config(output_tokens=8192, input_lengths_k=[32, 43.75, 200])
            entries, summary = plan_input_sweep(a, 3500 * 1024, 256, 64)
            self.assertEqual([e['mean'] for e in entries], [32768, 44800])
            self.assertEqual(entries[-1]['origin'], 'requested')
            self.assertEqual(summary['auto_added_input_tokens'], [])

        def test_sweep_sorted_deduplicated_and_all_above_bound(self):
            a = config(output_tokens=8192, input_lengths_k=[200, 32, 32, .5, 64])
            entries, _ = plan_input_sweep(a, 3500 * 1024, 256, 64)
            self.assertEqual([e['mean'] for e in entries], [512, 32768, 44800])
            a.input_lengths_k = [200]
            entries, _ = plan_input_sweep(a, 3500 * 1024, 256, 64)
            self.assertEqual([e['mean'] for e in entries], [44800])

        def test_sweep_no_positive_input_with_output_budget_exhausted(self):
            entries, summary = plan_input_sweep(config(output_tokens=8192), 1000, 256, 64)
            self.assertEqual(entries, [])
            self.assertEqual(summary['effective_input_limit_tokens'], 0)

        def test_sweep_context_limit_includes_output(self):
            a = config(output_tokens=512, input_lengths_k=[1, 4])
            entries, summary = plan_input_sweep(a, 100000, 16, 1, 4096)
            self.assertEqual([e['mean'] for e in entries], [1024, 3584])
            self.assertEqual(summary['limiting_factor'], 'context_limit')
            self.assertEqual(entries[-1]['origin'], 'automatic_context_limit_boundary')

        def test_mixed_sweep_bound_is_safe_across_lengths_and_page_tails(self):
            for mode in (2, 3):
                a = config(mode=mode, output_tokens=128, input_lengths_k=[1, 2])
                for bs in (1, 8, 32):
                    entries, summary = plan_input_sweep(a, 12000, 16, bs, 2048)
                    mean = entries[-1]['mean']
                    for seed in range(20):
                        lengths = mixed_lengths(bs, mean, a.min_length_ratio, a.max_length_ratio, random.Random(seed))
                        self.assertLessEqual(rank_cost(lengths, [0] * bs, a.output_tokens, 16), 11400)
                        self.assertLessEqual(max(lengths) + a.output_tokens, 2048)

        def test_shared_and_semantic_grids_are_not_rewritten(self):
            a = config(allow_shared_prefix=True, input_lengths_k=[200, 32, 32])
            entries, summary = plan_input_sweep(a, 100, 256, 64)
            self.assertEqual([e['mean'] for e in entries], [204800, 32768, 32768])
            self.assertEqual(summary['auto_added_input_tokens'], [])
            a.mode, a.allow_shared_prefix = 4, False
            entries, summary = plan_input_sweep(a, 100, 256, 64)
            self.assertEqual([e['mean'] for e in entries], [32768, 204800])
            self.assertEqual(summary['policy'], 'real_text_preflight')

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
            self.assertIsNone(shadow_text({'messages': [{'role': 'user', 'content': [{'type': 'image'}]}]}))
            self.assertIn('a', shadow_text({'messages': [{'role': 'assistant', 'content': 'a', 'tool_calls': []}]}))
            with tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / 'traffic.jsonl'
                path.write_text('\n'.join(json.dumps(x) for x in [{'text': 'hello'}, {'text': 'hello'}, {'text': 'world'}]))
                data = load_shadow(temp, FakeTokenizer())
                self.assertEqual(len(data), 2)
                with self.assertRaisesRegex(BenchError, 'Not enough'):
                    select_shadow(data, 2, 2, 7, .01, random.Random(60), .25, 1.75)

        def test_plaintext_tools_reference_lengths_and_no_template(self):
            row = dict(meta=dict(prompt_tokens_offline=100000, prompt_bucket='064_128k'), body=dict(
                model='DO_NOT_REPLAY_MODEL', user='DO_NOT_REPLAY_USER',
                tools=[dict(type='function', function=dict(name='lookup', description='Real schema'))],
                messages=[dict(role='user', content=[dict(type='text', text='First'), dict(type='text', text='Second')]),
                          dict(role='assistant', content=None, reasoning_content='Private reasoning',
                               tool_calls=[dict(id='c1', type='function', function=dict(name='lookup', arguments='{"q":1}'))]),
                          dict(role='tool', tool_call_id='c1', content='Real result')]))
            text = shadow_text(row)
            for value in ('First\nSecond', 'Private reasoning', 'Real result', 'lookup', 'c1', 'Real schema'):
                self.assertIn(value, text)
            self.assertNotIn('DO_NOT_REPLAY', text)
            with tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / 'data.jsonl'
                path.write_text(json.dumps(row) + '\n' + json.dumps(row))
                summary = {}
                loaded = load_shadow(path, summary=summary)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0].tokens, 100000)
            self.assertEqual(loaded[0].text, text)
            self.assertEqual(loaded[0].bucket, (64*1024, 128*1024))
            self.assertEqual(summary['duplicates'], 1)
            self.assertEqual(shadow_text({'text': '  unchanged\n'}), '  unchanged\n')

        def test_plaintext_no_reference_requires_explicit_tokenizer_or_field(self):
            with tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / 'data.jsonl'
                path.write_text(json.dumps(dict(text='Real text', custom=dict(length=42))))
                with self.assertRaisesRegex(BenchError, 'no reference token length'):
                    load_shadow(path)
                self.assertEqual(load_shadow(path, length_field='custom.length')[0].tokens, 42)
                self.assertEqual(load_shadow(path, FakeTokenizer())[0].tokens, len('Real text'))

        def test_mode4_matches_mean_using_complete_plain_text_samples(self):
            data = [ShadowSample(f'Real text {i}', 100 if i % 2 else 120, 'recorded_reference',
                                 (0, 32768), str(i)) for i in range(64)]
            a = config(mode=4, tokenizer=None, shadow_mean_tolerance=0)
            prompts, plan = make_case(a, None, 100000, 16, 8, 110, 60, data)
            self.assertTrue(all(p.ids is None for p in prompts))
            self.assertTrue(all(r['mean'] == 110 for r in plan['rank_plans']))
            self.assertTrue(all(p.text in {s.text for s in data} for p in prompts))
            self.assertEqual(plan['selection'], 'matched_per_rank_mean')

        def test_mode5_bucket_boundaries_random_replay_and_no_mean_matching(self):
            data = [ShadowSample(f'Bucket real {i}', 33000+i, 'recorded_reference', (32768, 65536), str(i))
                    for i in range(1024)]
            a = config(mode=5, tokenizer=None, dp_size=8, min_length_ratio=.99, max_length_ratio=1.01,
                       shadow_mean_tolerance=0)
            first, plan = make_case(a, None, 100000000, 256, 64, 40000, 60, data)
            again, _ = make_case(a, None, 100000000, 256, 64, 40000, 60, data)
            other, _ = make_case(a, None, 100000000, 256, 64, 40000, 61, data)
            self.assertEqual([p.sha256 for p in first], [p.sha256 for p in again])
            self.assertNotEqual([p.sha256 for p in first], [p.sha256 for p in other])
            self.assertEqual(len({p.sha256 for p in first}), 512)
            self.assertEqual(plan['selected_bucket'], (32768, 65536))
            self.assertTrue(all(r['mean'] < 35000 for r in plan['rank_plans']))
            self.assertEqual(shadow_bucket({}, 'prompt_032_064k.jsonl.gz', 42), (32768, 65536))
            self.assertEqual(shadow_bucket({}, 'x', 32768), (32768, 65536))
            self.assertEqual(shadow_bucket({}, 'prompt_256k_plus/part.jsonl.gz', 1), (262144, None))
            self.assertEqual(shadow_bucket({}, 'prompt_032_064k.jsonl.gz', 42, [0, 64, 128]), (0, 64))
            with self.assertRaisesRegex(BenchError, 'exactly one'):
                select_shadow_bucket(data, 2, 2, 65536, random.Random(60))
            with self.assertRaisesRegex(BenchError, 'needs 2048'):
                select_shadow_bucket(data, 2, 1024, 32768, random.Random(60))

        def test_plaintext_cache_uses_actual_usage_not_reference(self):
            p = Prompt(0, 0, None, 60, text='Real text', planned_tokens=1000, length_source='recorded_reference')
            r = Record(p)
            r.prompt_tokens, r.cached_tokens = 100, 99
            self.assertEqual(cache_status(r, config()), 'reported_hit')
            self.assertEqual(p.observed_tokens, 100)
            r.prompt_tokens = 101
            with self.assertRaisesRegex(BenchError, 'inconsistent'):
                cache_status(r, config())
            for backend, key in (('sglang', 'text'), ('openai', 'prompt')):
                _, body = payload(config(backend=backend), p, 1, True)
                self.assertEqual(body[key], 'Real text')
                self.assertNotIn('input_ids', body)

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

        def test_cache_ratio_zero_keeps_tail_and_metadata_checks(self):
            a = config(min_cache_hit_ratio=0, cache_tail_tokens=256)
            r = Record(Prompt(0, 0, array('I', [1] * 4), 60))
            r.cached_tokens = 0
            self.assertEqual(cache_status(r, a), 'reported_hit')
            with self.assertRaises(BenchError):
                cache_status(r, config(min_cache_hit_ratio=.5))
            with self.assertRaises(BenchError):
                cache_status(r, config(min_cache_hit_ratio=0, cache_tail_tokens=3))
            r.cached_tokens = None
            with self.assertRaises(BenchError):
                cache_status(r, a)
            for cached in (-1, 5):
                r.cached_tokens = cached
                with self.assertRaises(BenchError):
                    cache_status(r, a)
            r = Record(Prompt(0, 0, array('I', [1] * 1024), 60))
            r.cached_tokens = 0
            with self.assertRaises(BenchError):
                cache_status(r, a)

        def test_cache_ratio_inclusive_bounds(self):
            for ratio in (0, .5, 1):
                self.assertEqual(config(min_cache_hit_ratio=ratio).min_cache_hit_ratio, ratio)
            for ratio in (-.01, 1.01, float('nan'), float('inf'), -float('inf')):
                with self.assertRaisesRegex(BenchError, 'Invalid capacity/cache fraction'):
                    config(min_cache_hit_ratio=ratio)
            with self.assertRaisesRegex(BenchError, 'Invalid capacity/cache fraction'):
                config(capacity_fraction=0, min_cache_hit_ratio=0)

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

        def test_live_counter_semantics_and_duplicate_rejection(self):
            a = config()
            live = '\n'.join(f'sglang:realtime_tokens_total{{dp_rank="{r}",mode="decode"}} 100' for r in range(2))
            other = '\nsglang:realtime_tokens_total{dp_rank="0",mode="prefill_compute"} 99999'
            counts, _ = parse_live_tokens(live + other, a)
            self.assertEqual(counts, {0: 100, 1: 100})
            for bad in (live + '\n' + live, live.splitlines()[0],
                        'sglang:generation_tokens_total 999999', live.replace('100', 'NaN')):
                with self.assertRaises(BenchError):
                    parse_live_tokens(bad, a)

        def test_catchup_gate_rejects_equal_rate_large_backlog_and_fast_replay(self):
            a = config(catchup_window_seconds=1, catchup_token_slack=0,
                       catchup_max_lag_seconds=.1, catchup_rate_tolerance=.1)
            identities = {0: [('dp_rank', '0')], 1: [('dp_rank', '1')]}
            def sample(t, value):
                return dict(t=t, latency=0, live_tokens={0: value, 1: value}, live_token_series=identities)
            records = [Record(Prompt(r, r, array('I', [1]), 60), actual_rank=r) for r in range(2)]
            gate = CatchupGate(a, sample(0, 1000))
            for r in records:
                r.tokens = 51  # Decode counter 100, client only 50+prefill.
            self.assertFalse(gate.observe(sample(1, 1100), records)['ready'])
            for r in records:
                r.tokens = 151
            state = gate.observe(sample(2, 1200), records)
            self.assertFalse(state['ready'])  # Rates both100/s, but constant50 token backlog.
            self.assertTrue(all(row['rate_ok'] for row in state['ranks']))
            for r in records:
                r.tokens = 301
            self.assertFalse(gate.observe(sample(3, 1300), records)['ready'])  # Catch-up replay rate150 vs100.
            for r in records:
                r.tokens = 401
            self.assertTrue(gate.observe(sample(4, 1400), records)['ready'])
            with self.assertRaisesRegex(BenchError, 'counter reset'):
                gate.observe(sample(5, 1300), records)

        def test_catchup_gate_stale_missing_and_changed_counter_never_verify(self):
            a = config(catchup_window_seconds=1, catchup_token_slack=0)
            base = dict(t=0, latency=0, live_tokens={0: 100, 1: 100}, live_token_series={0: [], 1: []})
            records = [Record(Prompt(r, r, array('I', [1]), 60), tokens=1) for r in range(2)]
            gate = CatchupGate(a, base)
            gate.observe({**base, 't': 1}, records)
            self.assertFalse(gate.observe({**base, 't': 2}, records)['ready'])
            self.assertFalse(gate.observe(dict(t=3, live_token_error='unavailable'), records)['ready'])
            self.assertEqual(gate.history, [])
            with self.assertRaisesRegex(BenchError, 'identity changed'):
                gate.observe({**base, 't': 4, 'live_token_series': {0: ['new'], 1: []}}, records)

        def test_frame_read_timestamp_precedes_json_and_mtp_counts_are_separate(self):
            record = Record(Prompt(0, 0, array('I', [1]), 60))
            at = time.monotonic() - .03
            record.consume('{"meta_info":{"completion_tokens":5}}', 'sglang', at)
            record.consume('{"meta_info":{"completion_tokens":10}}', 'sglang', at)
            self.assertEqual(record.events, [at, at])
            self.assertEqual([row[1] for row in record.observations], [5, 10])
            self.assertGreaterEqual(record.observations[0][2], .03)

        def test_catchup_argument_validation(self):
            for name in ('catchup_window_seconds', 'catchup_max_lag_seconds', 'client_max_frame_lag'):
                for value in (0, -1, float('nan'), float('inf')):
                    with self.assertRaises(BenchError):
                        config(**{name: value})
            for kwargs in (dict(catchup_token_slack=-1), dict(catchup_rate_tolerance=-.1),
                           dict(catchup_rate_tolerance=1), dict(catchup_rate_tolerance=float('nan'))):
                with self.assertRaises(BenchError):
                    config(**kwargs)

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

    class PersistenceTests(unittest.TestCase):
        def test_atomic_output_keeps_old_file_until_replace(self):
            with tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / 'result.json'
                path.write_text('old')
                with atomic_text_output(path) as stream:
                    stream.write('new')
                    self.assertEqual(path.read_text(), 'old')
                self.assertEqual(path.read_text(), 'new')
                self.assertEqual(list(Path(temp).glob('.*.tmp')), [])

        def test_failed_atomic_write_preserves_previous_result(self):
            with tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / 'result.json'
                path.write_text('previous complete snapshot')
                with self.assertRaisesRegex(RuntimeError, 'simulated'):
                    with atomic_text_output(path) as stream:
                        stream.write('incomplete')
                        raise RuntimeError('simulated interruption')
                self.assertEqual(path.read_text(), 'previous complete snapshot')
                self.assertEqual(list(Path(temp).glob('.*.tmp')), [])

        def test_snapshot_plot_opt_in_and_only_completed_points(self):
            good = dict(status='ok', bs_per_dp=2, mean_input_tokens_requested=128,
                prefix_workload='independent_synthetic', measurement=dict(
                    sse_gap_mean_ms=13, sse_gap_median_ms=12, pooled_sse_gap_p99_ms=20,
                    bs_verification='sampled_per_rank_metrics',
                    cache_verification='reported_hit_all_requests'))
            pending = dict(status='running', bs_per_dp=2, mean_input_tokens_requested=256)
            failed = dict(status='failed', bs_per_dp=2, mean_input_tokens_requested=512)
            with tempfile.TemporaryDirectory() as temp:
                out = Path(temp)
                save_results(out, [good], [{'bs': 2}], False)
                self.assertFalse((out / 'sse_gap.svg').exists())
                save_results(out, [good, pending, failed], [{'bs': 2}], True)
                self.assertEqual((out / 'sse_gap.svg').read_text().count('<circle '), 1)
                self.assertEqual(json.loads((out / 'results.json').read_text()), [good, pending, failed])
                self.assertEqual(json.loads((out / 'sweep_plan.json').read_text()), [{'bs': 2}])
                with (out / 'summary.csv').open() as stream:
                    self.assertEqual([r['status'] for r in csv.DictReader(stream)], ['ok', 'running', 'failed'])

    class HTTPTests(unittest.IsolatedAsyncioTestCase):
        async def asyncSetUp(self):
            self.running = [0, 0]
            self.generated = [0, 0]
            self.cache = set()
            self.mode = 'normal'
            self.posts = 0
            self.posted_bodies = []
            self.active = {}
            self.abort_seen = []
            self.cleanup_stuck = False
            self.info_dp_size = 2
            self.temp = tempfile.TemporaryDirectory()
            app = web.Application()
            for path in ('/generate', '/v1/completions', '/v1/chat/completions'):
                app.router.add_post(path, self.generate)
            app.router.add_get('/server_info', self.info)
            app.router.add_get('/metrics', self.metrics)
            app.router.add_post('/abort_request', self.abort)
            self.server = web.AppRunner(app)
            await self.server.setup()
            site = web.TCPSite(self.server, '127.0.0.1', 0)
            await site.start()
            self.url = f'http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}'

        async def asyncTearDown(self):
            await self.server.cleanup()
            self.temp.cleanup()

        async def test_warmup_and_probe_auto_follow_current_point_and_explicit_caps(self):
            a = config(batch_sizes=[2, 64])
            prompts, _ = make_case(a, FakeTokenizer(), 100000, 16, 2, 128, 60)
            for cap, expected in ((None, 4), (2, 2), (100, 4)):
                a.warmup_concurrency = cap
                for verify in (False, True):
                    active = peak = 0
                    async def fake_request(session, args, record, output):
                        nonlocal active, peak
                        active += 1
                        peak = max(peak, active)
                        try:
                            await asyncio.sleep(.01)
                            record.prompt_tokens = len(record.prompt.ids)
                            record.cached_tokens = record.prompt_tokens - 1
                            record.actual_rank = record.prompt.rank
                        finally:
                            active -= 1
                    with patch.dict(warm_phase.__globals__, request=fake_request):
                        rows = await warm_phase(None, a, prompts, verify)
                    self.assertEqual(peak, expected)
                    self.assertEqual(len(rows), 4)
            for invalid in (0, -1):
                with self.assertRaisesRegex(BenchError, 'warmup_concurrency'):
                    config(warmup_concurrency=invalid)

        async def info(self, request):
            return web.json_response(dict(dp_size=self.info_dp_size, max_total_num_tokens=100000, page_size=16,
                                          context_length=8192, api_key='DO_NOT_SAVE_SERVER_SECRET'))

        async def metrics(self, request):
            text = '\n'.join(f'sglang:{n}{{dp_rank="{rank}"}} {v}' for rank in range(2)
                             for n, v in [('num_running_reqs', self.running[rank] + int(self.cleanup_stuck)),
                                          ('num_queue_reqs', 1 if self.mode == 'queued' else 0)])
            if self.mode != 'missing-live' and not (self.mode == 'drop-live' and any(self.running)):
                text += '\n' + '\n'.join(f'sglang:realtime_tokens_total{{dp_rank="{r}",mode="decode"}} {v}'
                                         for r, v in enumerate(self.generated))
            return web.Response(text=text)

        async def abort(self, request):
            body = await request.json()
            self.assertEqual(body.get('abort_all'), False)
            self.assertTrue(isinstance(body.get('rid'), str) and body['rid'].startswith('ssebench_'))
            self.assertEqual(set(body), {'rid', 'abort_all'})
            self.abort_seen.append(body['rid'])
            if self.mode == 'abort-error':
                return web.Response(status=503)
            if self.mode == 'abort-stuck':
                self.cleanup_stuck = True
            stop = self.active.get(body['rid'])
            if stop is not None:
                stop.set()
            return web.Response(status=200)  # Deliberately empty, like native SGLang.

        async def generate(self, request):
            self.posts += 1
            body = await request.json()
            self.posted_bodies.append(body)
            native = request.path == '/generate'
            ids = body.get('input_ids', body.get('prompt', body.get('text')))
            if isinstance(ids, str):
                ids = FakeTokenizer().encode(ids)
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
            rid = body.get('rid')
            stop = asyncio.Event()
            if rid:
                self.active[rid] = stop
            self.running[rank] += 1
            try:
                for i in range(1, limit + 1):
                    await asyncio.sleep(.007)
                    if stop.is_set():
                        break
                    if i > 1:
                        self.generated[rank] += 1  # One initial token came from prefill.
                    if self.mode == 'backlogged' and i == 1 and limit > 1:
                        self.generated[rank] += 100  # Simulate output still buffered upstream.
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
                if rid:
                    self.active.pop(rid, None)
            return response

        async def invoke(self, **changes):
            settings = dict(base_url=self.url, metrics_url=self.url + '/metrics', save_path=self.temp.name,
                            steady_check='required', plot=True)
            settings.update(changes)
            a = config(**settings)
            fake_module = types.SimpleNamespace(AutoTokenizer=FakeTokenizer)
            before = set(Path(self.temp.name).glob('sse_*'))
            with patch.dict(sys.modules, {'transformers': fake_module}), contextlib.redirect_stdout(io.StringIO()):
                await run(a)
            created = set(Path(self.temp.name).glob('sse_*')) - before
            self.assertEqual(len(created), 1)
            directory = created.pop()
            return json.loads((directory / 'results.json').read_text()), directory

        async def test_native_end_to_end_warm_probe_measure_plot(self):
            result, directory = await self.invoke()
            self.assertEqual(result[0]['status'], 'ok')
            m = result[0]['measurement']
            self.assertEqual(m['cache_verification'], 'reported_hit_all_requests')
            self.assertEqual(m['bs_verification'], 'sampled_per_rank_metrics')
            self.assertEqual(m['catchup_verification'], 'sampled_live_tokens_verified')
            self.assertEqual(self.posts, 12)  # 4 warm + 4 probe + 4 formal
            self.assertGreater(m['sse_gap_median_ms'], 4)
            self.assertTrue((directory / 'sse_gap.svg').exists())
            for path in directory.glob('*'):
                content = path.read_text()
                self.assertNotIn('DO_NOT_SAVE_GENERATED_TEXT', content)
                self.assertNotIn('DO_NOT_SAVE_SERVER_SECRET', content)

        async def test_cancel_after_window_all_six_modes_and_next_point(self):
            shadow = Path(self.temp.name) / 'stop-shadow.jsonl'
            shadow.write_text('\n'.join(json.dumps(dict(text=(f'real request {i} ' + 'x' * 128)[:128],
                reference_input_tokens=128)) for i in range(16)))
            for mode in range(6):
                with self.subTest(mode=mode):
                    before = len(self.abort_seen)
                    changes = dict(mode=mode, stop_after_measure=True, output_tokens=1000, catchup_check='required',
                                   batch_sizes=[1, 2], replay_seed=60, min_cache_hit_ratio=.9)
                    if mode in (4, 5):
                        changes.update(shadow_dir=str(shadow), tokenizer=None)
                    rows, directory = await self.invoke(**changes)
                    self.assertEqual([r['status'] for r in rows], ['ok', 'ok'])
                    self.assertEqual(self.running, [0, 0])
                    self.assertEqual(len(self.abort_seen) - before, 6)
                    for case in rows:
                        m = case['measurement']
                        self.assertTrue(m['measurement_valid'])
                        self.assertEqual(m['catchup_verification'], 'sampled_live_tokens_verified')
                        self.assertIn('before JSON parsing', m['timestamp_basis'])
                        self.assertFalse(m['final_usage_complete'])
                        self.assertEqual(m['output_handling'], 'cancel_after_window')
                        self.assertEqual(m['usage_scope'], 'stream_observed_at_window_end')
                        self.assertEqual(m['cleanup']['status'], 'verified')
                        self.assertEqual(m['cleanup']['abort_requests_accepted'], m['request_count'])
                        for req in m['requests']:
                            self.assertLess(req['completion_tokens'], 1000)
                            self.assertTrue(req['cancelled'])
                            self.assertTrue(all(0 <= t <= m['window_seconds'] for t in req['sse_times_relative_s']))
                    self.assertEqual((directory / 'sse_gap.svg').read_text().count('<circle '), 2)

        async def test_catchup_unsupported_auto_marks_required_refuses_before_generation(self):
            self.mode = 'missing-live'
            rows, directory = await self.invoke()
            self.assertEqual(rows[0]['measurement']['catchup_verification'], 'unverified')
            self.assertIn('Hollow point = catch-up UNVERIFIED', (directory / 'sse_gap.svg').read_text())
            before = self.posts
            with self.assertRaisesRegex(BenchError, 'Cannot verify SSE catch-up'):
                await self.invoke(catchup_check='required')
            self.assertEqual(self.posts, before)
            self.mode = 'normal'
            rows, _ = await self.invoke(catchup_check='off')
            self.assertEqual(rows[0]['measurement']['catchup_verification'], 'disabled')

        async def test_backlogged_stream_does_not_enter_window_and_failure_is_auditable(self):
            self.mode = 'backlogged'
            with self.assertRaisesRegex(BenchError, 'Stable decode timeout'):
                await self.invoke(catchup_check='required', catchup_token_slack=0,
                                  catchup_max_lag_seconds=.01, stop_after_measure=True,
                                  output_tokens=1000, wait_timeout=.15)
            row = json.loads(next(Path(self.temp.name).glob('sse_*/results.json')).read_text())[0]
            self.assertEqual(row['status'], 'failed')
            self.assertNotIn('measurement', row)
            self.assertEqual(row['failure_diagnostics']['catchup']['last_check']['reason'],
                             'token_backlog_or_rate_mismatch')
            self.assertEqual(self.running, [0, 0])

        async def test_counter_lost_after_baseline_cannot_silently_downgrade(self):
            self.mode = 'drop-live'
            with self.assertRaisesRegex(BenchError, 'Stable decode timeout'):
                await self.invoke(catchup_check='required', stop_after_measure=True,
                                  output_tokens=1000, wait_timeout=.15)
            row = json.loads(next(Path(self.temp.name).glob('sse_*/results.json')).read_text())[0]
            self.assertEqual(row['status'], 'failed')
            self.assertNotIn('measurement', row)
            self.assertFalse(row['failure_diagnostics']['catchup']['last_check']['ready'])

        async def test_frame_parse_backlog_prevents_window_even_with_live_token_agreement(self):
            original = sse_frames
            async def old_reads(chunks, timed=False):
                async for raw, at in original(chunks, timed=True):
                    yield raw, at - .5
            with patch.dict(request.__globals__, sse_frames=old_reads):
                with self.assertRaisesRegex(BenchError, 'Stable decode timeout'):
                    await self.invoke(catchup_check='required', stop_after_measure=True,
                                      output_tokens=1000, wait_timeout=.15, client_max_frame_lag=.01)
            row = json.loads(next(Path(self.temp.name).glob('sse_*/results.json')).read_text())[0]
            self.assertNotIn('measurement', row)
            self.assertTrue(all(r['last_read_to_parsed_ms'] >= 500
                                for r in row['failure_diagnostics']['read_diagnostics']))

        async def test_generic_openai_cannot_guess_live_token_counts(self):
            for backend in ('openai', 'openai-chat'):
                shadow = Path(self.temp.name) / 'generic-shadow.jsonl'
                shadow.write_text('\n'.join(json.dumps(dict(text=f'real {i} ' + 'x' * 128,
                    reference_input_tokens=128)) for i in range(8)))
                rows, _ = await self.invoke(backend=backend, mode=4, shadow_dir=str(shadow),
                                            tokenizer=None, allow_unverified_cache=True, steady_check='client')
                self.assertEqual(rows[0]['measurement']['catchup_verification'], 'unverified')
                self.assertTrue(all(p['completion_tokens'] is None
                    for r in rows[0]['measurement']['requests'] for p in r['window_progress']))
                before = self.posts
                with self.assertRaisesRegex(BenchError, 'Cannot verify SSE catch-up'):
                    await self.invoke(backend=backend, mode=4, shadow_dir=str(shadow),
                                      tokenizer=None, catchup_check='required')
                self.assertEqual(before, self.posts)

        async def test_all_synthetic_modes_once_max_use_the_same_catchup_gate(self):
            for mode in range(4):
                with self.subTest(mode=mode):
                    rows, _ = await self.invoke(mode=mode, warmup_strategy='once-max',
                        catchup_check='required', stop_after_measure=True, output_tokens=1000)
                    self.assertEqual(rows[0]['measurement']['catchup_verification'], 'sampled_live_tokens_verified')

        async def test_window_checkpoint_exists_before_any_abort(self):
            entered, release = asyncio.Event(), asyncio.Event()
            original = cancel_cohort
            async def paused_cleanup(*args, **kwargs):
                entered.set()
                await release.wait()
                return await original(*args, **kwargs)
            with patch.dict(measure.__globals__, cancel_cohort=paused_cleanup):
                task = asyncio.create_task(self.invoke(stop_after_measure=True, output_tokens=1000))
                try:
                    await asyncio.wait_for(entered.wait(), 4)
                    directory = next(Path(self.temp.name).glob('sse_*'))
                    row = json.loads((directory / 'results.json').read_text())[0]
                    self.assertEqual(row['status'], 'measured_cleanup_pending')
                    self.assertTrue(row['measurement']['measurement_valid'])
                    self.assertEqual(row['measurement']['cleanup']['status'], 'pending')
                    self.assertEqual(self.abort_seen, [])
                    self.assertGreater(sum(self.running), 0)
                    release.set()
                    rows, _ = await task
                    self.assertEqual(rows[0]['status'], 'ok')
                    self.assertEqual(rows[0]['measurement']['cleanup']['status'], 'verified')
                finally:
                    release.set()
                    if not task.done():
                        task.cancel()
                    await asyncio.gather(task, return_exceptions=True)

        async def test_cleanup_failure_keeps_window_and_stops_matrix(self):
            for mode in ('abort-error', 'abort-stuck'):
                with self.subTest(mode=mode):
                    self.mode = mode
                    self.cleanup_stuck = False
                    before_dirs = set(Path(self.temp.name).glob('sse_*'))
                    before_posts = self.posts
                    with self.assertRaises(BenchError):
                        await self.invoke(stop_after_measure=True, output_tokens=1000,
                                          input_lengths_k=[.125, .25], cleanup_timeout=.2)
                    directory = (set(Path(self.temp.name).glob('sse_*')) - before_dirs).pop()
                    rows = json.loads((directory / 'results.json').read_text())
                    self.assertEqual(len(rows), 1)
                    self.assertEqual(rows[0]['status'], 'cleanup_failed')
                    self.assertTrue(rows[0]['measurement']['measurement_valid'])
                    self.assertEqual(rows[0]['measurement']['cleanup']['status'], 'failed')
                    self.assertEqual(self.posts - before_posts, 12)
                    self.cleanup_stuck = False

        async def test_stop_requires_metrics_and_does_not_abort_foreign_work(self):
            with self.assertRaisesRegex(BenchError, 'requires readable per-rank metrics'):
                await self.invoke(stop_after_measure=True, steady_check='client')
            self.assertEqual(self.posts, 0)
            self.running[0] = 1  # Existing foreign request: wait, never abort it.
            with self.assertRaisesRegex(BenchError, 'idle timeout'):
                await self.invoke(stop_after_measure=True, cleanup_timeout=.06)
            self.assertEqual(self.posts, 0)
            self.assertEqual(self.abort_seen, [])
            self.running[0] = 0

        async def test_stop_once_max_preserves_report_only_cache(self):
            rows, _ = await self.invoke(stop_after_measure=True, output_tokens=1000,
                                        mode=1, warmup_strategy='once-max', batch_sizes=[1, 2])
            self.assertTrue(all(r['measurement']['cleanup']['status'] == 'verified' for r in rows))
            self.assertTrue(all(r['measurement']['cache_verification'] == 'report_only' for r in rows))
            self.assertTrue(all(r['cache_probe_skipped'] for r in rows))

        async def test_cancel_512_ids_is_bounded_and_accepts_empty_200(self):
            a = config(stop_after_measure=True, dp_size=8, cleanup_timeout=2)
            records = [Record(Prompt(i, i % 8, array('I', [1]), 60),
                              request_id=f'ssebench_test_{i}_nonce', submitted=True)
                       for i in range(512)]
            tasks = [asyncio.create_task(asyncio.sleep(30)) for _ in records]
            active = peak = 0
            sent = []
            class Reply:
                status = 200
                async def __aenter__(inner):
                    nonlocal active, peak
                    active += 1
                    peak = max(peak, active)
                    await asyncio.sleep(.001)
                    return inner
                async def __aexit__(inner, *exc):
                    nonlocal active
                    active -= 1
            class Session:
                def post(inner, url, *, json, timeout):
                    self.assertTrue(url.endswith('/abort_request'))
                    self.assertFalse(json['abort_all'])
                    sent.append(json['rid'])
                    return Reply()
            async def idle_metrics(*unused):
                return dict(t=time.monotonic(), latency=.001,
                            ranks={rank: dict(running=0, waiting=0) for rank in range(8)})
            state = {}
            with patch.dict(cancel_cohort.__globals__, read_metrics=idle_metrics):
                await cancel_cohort(Session(), a, records, tasks, state)
            self.assertEqual(state['status'], 'verified')
            self.assertEqual(state['abort_requests_accepted'], 512)
            self.assertEqual(len(set(sent)), 512)
            self.assertLessEqual(peak, 16)
            self.assertTrue(all(task.cancelled() for task in tasks))

        async def test_stop_openai_disconnect_is_verified_not_claimed_native_abort(self):
            self.info_dp_size = 1
            rows, _ = await self.invoke(stop_after_measure=True, output_tokens=1000,
                backend='openai', dp_size=1, batch_sizes=[2], metric_label=['dp_rank=0'],
                allow_unverified_cache=True)
            cleanup = rows[0]['measurement']['cleanup']
            self.assertEqual(cleanup['method'], 'stream_close')
            self.assertEqual(cleanup['abort_requests_accepted'], 0)
            self.assertEqual(cleanup['status'], 'verified')
            self.assertEqual(self.abort_seen, [])

        async def test_once_max_all_modes_one_warmup_descending_matrix_zero_hits(self):
            for mode in range(4):
                self.mode = 'cache-miss'
                self.posts = 0
                self.posted_bodies.clear()
                results, directory = await self.invoke(mode=mode, warmup_strategy='once-max',
                    batch_sizes=[1, 2, 4], input_lengths_k=[.125, .25],
                    min_cache_hit_ratio=1, cache_tail_tokens=0, replay_seed=60)
                self.assertEqual([(r['bs_per_dp'], r['mean_input_tokens_requested']) for r in results],
                                 [(4, 256), (4, 128), (2, 256), (2, 128), (1, 256), (1, 128)])
                warm_count = 2 if mode == 0 else 8
                self.assertEqual(self.posts, warm_count + (4 + 2 + 1) * 2 * 2)
                warm = self.posted_bodies[:warm_count]
                self.assertTrue(all(b['sampling_params']['max_new_tokens'] == 1 for b in warm))
                for body in self.posted_bodies[warm_count:]:
                    self.assertEqual(body['sampling_params']['max_new_tokens'], 50)
                    self.assertTrue(any(w['routed_dp_rank'] == body['routed_dp_rank']
                        and body['input_ids'] == w['input_ids'][:len(body['input_ids'])] for w in warm))
                for r in results:
                    self.assertEqual(r['status'], 'ok')
                    self.assertTrue(r['cache_probe_skipped'])
                    self.assertNotIn('cache_probe', r)
                    self.assertEqual(r['measurement']['cache_verification'], 'report_only')
                    self.assertEqual(r['measurement']['cache_summary']['weighted_hit_ratio'], 0)
                    self.assertEqual(r['measurement']['bs_verification'], 'sampled_per_rank_metrics')
                    self.assertEqual(r['measurement']['dp_routing'], 'reported_rank_verified_all')
                warm_plan = json.loads((directory / 'once_warmup.json').read_text())
                self.assertEqual(warm_plan['status'], 'ok')
                self.assertEqual(warm_plan['warmup_request_count'], warm_count)
                cache = json.loads((directory / 'cache_hits.json').read_text())
                self.assertEqual(len(cache), 6)
                self.assertTrue(all(row['coverage'] == 'complete' and row['weighted_hit_ratio'] == 0 for row in cache))
                self.assertEqual(len(cache[0]['requests']), 8)
                with (directory / 'cache_hits.csv').open() as stream:
                    self.assertEqual(len(list(csv.DictReader(stream))), 6)
                self.assertEqual((directory / 'sse_gap.svg').read_text().count('<circle '), 6)

        async def test_once_max_unknown_cache_is_not_fabricated_or_a_failure(self):
            self.mode = 'missing-cache'
            results, directory = await self.invoke(warmup_strategy='once-max')
            cache = results[0]['measurement']['cache_summary']
            self.assertEqual(cache['coverage'], 'unknown')
            self.assertEqual(cache['reported_request_count'], 0)
            self.assertIsNone(cache['weighted_hit_ratio'])
            self.assertEqual(results[0]['status'], 'ok')
            with (directory / 'cache_hits.csv').open() as stream:
                self.assertEqual(next(csv.DictReader(stream))['weighted_hit_ratio'], '')

        async def test_once_max_plan_only_and_union_failure_send_no_generation(self):
            results, directory = await self.invoke(warmup_strategy='once-max', plan_only=True,
                                                   batch_sizes=[1, 2], input_lengths_k=[.125, .25])
            self.assertEqual(self.posts, 0)
            self.assertTrue(all(r['status'] == 'planned_not_run' for r in results))
            self.assertEqual(json.loads((directory / 'once_warmup.json').read_text())['status'], 'planned_not_run')
            with self.assertRaisesRegex(CapacityError, 'union of all BS/length cases'):
                await self.invoke(warmup_strategy='once-max', kv_capacity_tokens=2048,
                                  batch_sizes=[1, 4], input_lengths_k=[.25, .5])
            self.assertEqual(self.posts, 0)

        async def test_once_max_does_not_bypass_steady_or_early_finish_failures(self):
            self.mode = 'queued'
            with self.assertRaisesRegex(BenchError, 'Stable decode timeout'):
                await self.invoke(warmup_strategy='once-max', wait_timeout=.08, catchup_check='off')
            self.mode = 'early'
            with self.assertRaisesRegex(BenchError, 'finished before'):
                await self.invoke(warmup_strategy='once-max')

        async def test_mode0_shared_input_keeps_all_requests_and_per_rank_checks(self):
            results, directory = await self.invoke(mode=0, batch_sizes=[1, 2])
            self.assertEqual([r['status'] for r in results], ['ok', 'ok'])
            self.assertEqual(self.posts, 18)  # (2 + 4) requests * warm/probe/formal.
            self.assertEqual(len(self.cache), 2)  # Same input, separate DP caches.
            self.assertEqual(len({p['sha256'] for r in results for p in r['prompts']}), 1)
            for r in results:
                self.assertEqual(r['prefix_workload'], 'fully_shared_synthetic')
                m = r['measurement']
                self.assertEqual(m['request_count'], r['bs_per_dp'] * 2)
                self.assertEqual(m['cache_verification'], 'reported_hit_all_requests')
                self.assertEqual(m['bs_verification'], 'sampled_per_rank_metrics')
                self.assertEqual(m['dp_routing'], 'reported_rank_verified_all')
            self.assertEqual(json.loads((directory / 'config.json').read_text())['run_seed'], 60)
            self.assertIn('fully_shared_synthetic', (directory / 'sse_gap.svg').read_text())

        async def test_mode0_known_cache_miss_still_fails(self):
            self.mode = 'cache-miss'
            with self.assertRaisesRegex(BenchError, 'input tokens cached'):
                await self.invoke(mode=0)

        async def test_mode0_output_capacity_failure_sends_zero_requests(self):
            with self.assertRaisesRegex(BenchError, 'No requested BS'):
                await self.invoke(mode=0, kv_capacity_tokens=100)
            self.assertEqual(self.posts, 0)

        async def test_checkpoint_and_plot_visible_while_next_case_waits(self):
            entered, release = asyncio.Event(), asyncio.Event()
            original_warm = warm_phase

            async def pause_second(session, args, prompts, verify):
                if not verify and len(prompts[0].ids) == 256:
                    entered.set()
                    await release.wait()
                return await original_warm(session, args, prompts, verify)

            with patch.dict(run.__globals__, warm_phase=pause_second):
                task = asyncio.create_task(self.invoke(input_lengths_k=[.125, .25]))
                try:
                    await asyncio.wait_for(entered.wait(), 5)
                    self.assertFalse(task.done())
                    directory = next(Path(self.temp.name).glob('sse_*'))
                    saved = json.loads((directory / 'results.json').read_text())
                    self.assertEqual([r['status'] for r in saved], ['ok', 'running'])
                    initial_plot = (directory / 'sse_gap.svg').read_text()
                    self.assertEqual(initial_plot.count('<circle '), 1)
                    with (directory / 'summary.csv').open() as stream:
                        self.assertEqual([r['status'] for r in csv.DictReader(stream)], ['ok', 'running'])
                    self.assertTrue((directory / 'sweep_plan.json').exists())
                    release.set()
                    results, _ = await task
                    self.assertEqual([r['status'] for r in results], ['ok', 'ok'])
                    self.assertEqual((directory / 'sse_gap.svg').read_text().count('<circle '), 2)
                    self.assertNotEqual(initial_plot, (directory / 'sse_gap.svg').read_text())
                finally:
                    release.set()
                    if not task.done():
                        task.cancel()
                    await asyncio.gather(task, return_exceptions=True)

        async def test_later_failure_keeps_completed_checkpoint_and_plot(self):
            original_warm = warm_phase

            async def fail_second(session, args, prompts, verify):
                if not verify and len(prompts[0].ids) == 256:
                    raise BenchError('simulated later-point failure')
                return await original_warm(session, args, prompts, verify)

            with patch.dict(run.__globals__, warm_phase=fail_second):
                with self.assertRaisesRegex(BenchError, 'later-point'):
                    await self.invoke(input_lengths_k=[.125, .25])
            directory = next(Path(self.temp.name).glob('sse_*'))
            results = json.loads((directory / 'results.json').read_text())
            self.assertEqual([r['status'] for r in results], ['ok', 'failed'])
            self.assertEqual((directory / 'sse_gap.svg').read_text().count('<circle '), 1)

        async def test_cancelled_later_case_preserves_completed_results(self):
            entered, release = asyncio.Event(), asyncio.Event()
            original_warm = warm_phase

            async def pause_second(session, args, prompts, verify):
                if not verify and len(prompts[0].ids) == 256:
                    entered.set()
                    await release.wait()
                return await original_warm(session, args, prompts, verify)

            with patch.dict(run.__globals__, warm_phase=pause_second):
                task = asyncio.create_task(self.invoke(input_lengths_k=[.125, .25]))
                try:
                    await asyncio.wait_for(entered.wait(), 5)
                    task.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await task
                finally:
                    release.set()
                    if not task.done():
                        task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
            directory = next(Path(self.temp.name).glob('sse_*'))
            results = json.loads((directory / 'results.json').read_text())
            self.assertEqual([r['status'] for r in results], ['ok', 'interrupted'])
            self.assertEqual((directory / 'sse_gap.svg').read_text().count('<circle '), 1)

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
            result, _ = await self.invoke(backend='openai-chat', mode=4, shadow_dir=str(source), steady_check='client',
                                          input_lengths_k=[len(shadow_text(rows[0]))/1024])
            self.assertEqual(result[0]['status'], 'ok')
            self.assertEqual(result[0]['prefix_workload'], 'natural_semantic')
            self.assertEqual(result[0]['measurement']['cache_verification'], 'reported_hit_all_requests')

        async def test_plaintext_modes_4_5_without_tokenizer_warm_probe_measure_plot(self):
            source = Path(self.temp.name) / 'shadow'
            source.mkdir()
            rows = [dict(text=f'DO_NOT_SAVE_SOURCE_{i} ' + 'x' * 100, reference_input_tokens=128) for i in range(16)]
            (source / 'prompt_000_032k.jsonl.gz').write_bytes(gzip.compress(
                ('\n'.join(json.dumps(row) for row in rows) + '\n').encode(), mtime=0))
            for mode, backend in ((4, 'sglang'), (5, 'sglang'), (5, 'openai')):
                self.posts = 0
                self.posted_bodies.clear()
                with patch.dict(sys.modules, {'transformers': None}):
                    result, directory = await self.invoke(mode=mode, backend=backend, tokenizer=None,
                        shadow_dir=str(source), steady_check='required' if backend == 'sglang' else 'client')
                self.assertEqual(result[0]['status'], 'ok')
                self.assertEqual(self.posts, 12)
                key = 'text' if backend == 'sglang' else 'prompt'
                texts = [b[key] for b in self.posted_bodies]
                self.assertTrue(all(isinstance(text, str) for text in texts))
                self.assertEqual(len(set(texts)), 4)
                self.assertTrue(all(texts.count(t) == 3 for t in set(texts)))
                self.assertEqual(result[0]['measurement']['cache_verification'], 'reported_hit_all_requests')
                actual = statistics.mean(len(t) for t in set(texts))
                self.assertEqual(result[0]['actual_mean_input_tokens'], actual)
                self.assertNotEqual(result[0]['actual_mean_input_tokens'], result[0]['planned_mean_input_tokens'])
                self.assertIn(f'x={actual/1024:g}K', (directory / 'sse_gap.svg').read_text())
                self.assertTrue(all(p['actual_input_tokens'] for p in result[0]['prompts']))
                for path in directory.glob('*'):
                    self.assertNotIn('DO_NOT_SAVE_SOURCE', path.read_text())

        async def test_plaintext_actual_capacity_checked_before_formal_requests(self):
            source = Path(self.temp.name) / 'shadow'
            source.mkdir()
            rows = [dict(text=str(i) + 'x'*10000, input_tokens=128) for i in range(16)]
            (source / 'data.jsonl').write_text('\n'.join(json.dumps(row) for row in rows))
            with self.assertRaisesRegex(BenchError, 'Actual service input\\+output'):
                await self.invoke(mode=4, tokenizer=None, shadow_dir=str(source))
            self.assertEqual(self.posts, 4)  # Only first warmups; no probes/formal long outputs.

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
            with self.assertRaisesRegex(BenchError, 'No requested BS'):
                await self.invoke(kv_capacity_tokens=100)
            self.assertEqual(self.posts, 0)

        async def test_bounded_sweep_continues_next_bs_and_records_actual_grid(self):
            results, directory = await self.invoke(kv_capacity_tokens=512, plan_only=True,
                batch_sizes=[1, 2], input_lengths_k=[.125, .25, .5])
            self.assertEqual([(r['bs_per_dp'], r['mean_input_tokens_requested']) for r in results],
                             [(1, 128), (1, 256), (1, 416), (2, 128), (2, 176)])
            self.assertTrue(all(r['status'] == 'planned_not_run' for r in results))
            self.assertEqual(results[-1]['input_length_origin'], 'automatic_kv_capacity_boundary')
            self.assertEqual(results[-1]['prefix_workload'], 'independent_synthetic')
            plans = json.loads((directory / 'sweep_plan.json').read_text())
            self.assertEqual([p['auto_added_input_tokens'] for p in plans], [[416], [176]])
            self.assertEqual(self.posts, 0)

        async def test_bounded_sweep_measures_auto_added_endpoint(self):
            results, _ = await self.invoke(kv_capacity_tokens=512, input_lengths_k=[.125, .25])
            self.assertEqual([r['mean_input_tokens_requested'] for r in results], [128, 176])
            self.assertTrue(all(r['status'] == 'ok' for r in results))
            self.assertTrue(all(r['measurement']['cache_verification'] == 'reported_hit_all_requests' for r in results))
            self.assertEqual(self.posts, 24)

        async def test_impossible_bs_does_not_stop_other_bs(self):
            results, directory = await self.invoke(kv_capacity_tokens=200, plan_only=True,
                batch_sizes=[4, 1], input_lengths_k=[.125])
            self.assertEqual(results[0]['status'], 'capacity_limited')
            self.assertEqual(results[1]['status'], 'planned_not_run')
            self.assertEqual(results[1]['bs_per_dp'], 1)
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

        async def test_timed_frames_keep_read_bursts_and_fragment_completion_time(self):
            async def chunks():
                yield b'data: {"x":', 10.
                yield b'1}\n\ndata: {"x":2}\n\n', 12.
            frames = [frame async for frame in sse_frames(chunks(), timed=True)]
            self.assertEqual(frames, [('{"x":1}', 12.), ('{"x":2}', 12.)])

        async def test_bounded_reader_reports_backpressure_and_cleans_up(self):
            class Content:
                async def iter_chunked(inner, limit):
                    self.assertEqual(limit, 65536)
                    for _ in range(20):
                        yield b'x' * 1024
            stats = dict(bytes_read=0, queue_full_count=0, queue_wait_max_seconds=0.)
            chunks = received_chunks(Content(), stats)
            await anext(chunks)
            await asyncio.sleep(.02)
            await chunks.aclose()
            self.assertGreater(stats['queue_full_count'], 0)
            self.assertLessEqual(stats['bytes_read'], 6 * 1024)  # consumer1 + queue4 + blocked read1

    suite = unittest.TestSuite()
    for cls in (PlanningTests, PersistenceTests, HTTPTests):
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
