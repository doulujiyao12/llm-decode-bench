"""Synthetic inputs and user-selected text/token JSONL (optionally gzip)."""
import gzip
import json
import random
from pathlib import Path

from .io import digest, write_json


def load_dataset(path, count):
    rows = []
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("Each dataset line must be an object")
            if ("input_ids" in row) == ("text" in row):
                raise ValueError("Each line needs exactly one of input_ids or text")
            if "input_ids" in row:
                ids = row["input_ids"]
                if not isinstance(ids, list) or not ids or any(type(x) is not int or x < 0 for x in ids):
                    raise ValueError("input_ids must be a non-empty list of nonnegative integers")
                rows.append({"input_ids": ids})
            else:
                if not isinstance(row["text"], str) or not row["text"]:
                    raise ValueError("text must be a non-empty string")
                rows.append({"text": row["text"]})
            if len(rows) == count:
                break
    if len(rows) != count:
        raise ValueError(f"Need {count} distinct dataset rows; got {len(rows)}. No implicit recycling")
    return rows


def manifest(rows, dp_size):
    return [{"index": i, "target_rank": i % dp_size, "sha256": digest(row),
             "input_tokens": len(row["input_ids"]) if "input_ids" in row else None,
             "input_kind": "tokens" if "input_ids" in row else "text"}
            for i, row in enumerate(rows)]


def synthetic_rows(valid_ids, lengths, count, seed, shared_prefix=0, families=1):
    if not valid_ids or not lengths or min(lengths) < 1 or count < 1 or families < 1:
        raise ValueError("Positive lengths/count/families and a nonempty vocabulary are required")
    if shared_prefix < 0 or shared_prefix >= min(lengths):
        raise ValueError("shared-prefix must be >=0 and < the shortest input")
    for i in range(count):
        # Per-request seed makes smaller cohorts prefixes of larger ones. The
        # length is independent of DP index (unlike lengths[i % len(lengths)]).
        rng = random.Random(f"{seed}:request:{i}")
        length = rng.choice(lengths)
        family = rng.randrange(families)
        prefix_rng = random.Random(f"{seed}:prefix:{family}")
        ids = prefix_rng.choices(valid_ids, k=shared_prefix)
        ids.extend(rng.choices(valid_ids, k=length - shared_prefix))
        yield {"input_ids": ids}


def generate(args):
    from transformers import AutoTokenizer
    # Do not execute repository-provided Python code.
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=False,
                                       revision=args.tokenizer_revision)
    valid = sorted(set(tok.get_vocab().values()) - set(tok.all_special_ids))
    path = Path(args.output)
    if path.exists() or path.with_suffix(path.suffix + ".meta.json").exists():
        raise FileExistsError("Dataset or metadata already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = synthetic_rows(valid, args.lengths, args.count, args.seed,
                          args.shared_prefix_tokens, args.prefix_families)
    with path.open("x") as f:
        for row in rows:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")
    write_json(path.with_suffix(path.suffix + ".meta.json"), {
        "type": "synthetic_valid_token_ids_not_quality_data", "tokenizer": args.tokenizer,
        "revision": args.tokenizer_revision, "vocab_hash": digest(tok.get_vocab()),
        "lengths": args.lengths, "count": args.count, "seed": args.seed,
        "shared_prefix_tokens": args.shared_prefix_tokens, "prefix_families": args.prefix_families,
    })
