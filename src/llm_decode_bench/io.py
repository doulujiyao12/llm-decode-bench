import hashlib
import json
import math
import os
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temp, path)


def read_jsonl(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path, value):
    with Path(path).open("a") as f:
        f.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def endpoint(url, suffix):
    parts = urlsplit(url)
    if (parts.scheme not in {"http", "https"} or not parts.netloc or parts.username
            or parts.password or parts.query or parts.fragment):
        raise ValueError("Use an http(s) URL without userinfo/query/fragment; auth uses an env variable")
    return urlunsplit((parts.scheme, parts.netloc, parts.path.rstrip("/") + suffix, "", ""))


def percentile(values, q):
    if not values:
        return None
    xs = sorted(values)
    pos = (len(xs) - 1) * q
    lo = math.floor(pos)
    return xs[lo] + (xs[math.ceil(pos)] - xs[lo]) * (pos - lo)
