"""Validate the shipped, authorized data rather than relying on file names."""
import gzip
import hashlib
import json
from pathlib import Path
import random
import unittest

from decode_sse_bench import select_shadow
from llm_decode_bench.inputs import load_dataset


class ReleasedShadowTests(unittest.TestCase):
    root = Path(__file__).resolve().parents[1] / 'shadow'

    def test_compressed_and_content_hashes_counts_and_unique_text(self):
        manifest = json.loads((self.root / 'manifest.json').read_text())
        self.assertEqual(manifest['source']['original_requests'], 40)
        self.assertEqual(manifest['basic']['records'], 512)
        self.assertEqual(manifest['basic']['distinct_original_requests'], 39)
        self.assertLess(sum(m['bytes'] for m in manifest['files'].values()), 15_000_000)
        for name, spec in manifest['files'].items():
            with self.subTest(name=name):
                raw = (self.root / name).read_bytes()
                self.assertEqual(hashlib.sha256(raw).hexdigest(), spec['sha256'])
                plain = gzip.decompress(raw)
                self.assertEqual(hashlib.sha256(plain).hexdigest(), spec['jsonl_sha256'])
                rows = [json.loads(line) for line in plain.splitlines()]
                self.assertEqual(len(rows), spec['records'])
                self.assertEqual(len({r['text'] for r in rows}), len(rows))
                self.assertEqual(len({r['source_id'] for r in rows}), spec['source_requests'])
                for row in rows:
                    self.assertEqual(hashlib.sha256(row['text'].encode()).hexdigest(), row['content_sha256'])

    def test_package_loads_gzip_without_unpacking(self):
        rows = load_dataset(self.root / 'basic.jsonl.gz', 512)
        self.assertEqual(len(rows), 512)
        self.assertTrue(all(row['text'] for row in rows))
        with self.assertRaises(ValueError):
            load_dataset(self.root / 'long-context.jsonl.gz', 256)

    def test_basic_excerpts_are_labelled_not_claimed_independent_requests(self):
        with gzip.open(self.root / 'basic.jsonl.gz', 'rt', encoding='utf-8') as f:
            rows = [json.loads(line) for line in f]
        self.assertEqual({row['kind'] for row in rows}, {'sanitized_real_traffic_excerpt'})
        self.assertEqual(len({row['text'][:128] for row in rows}), 512)
        for row in rows:
            self.assertIn(row['sanitized_source_token_end'] - row['sanitized_source_token_start'],
                          {2048, 4096, 8192, 16384})

    def test_reference_lengths_can_form_dp8_cohorts_without_repeating_text(self):
        with gzip.open(self.root / 'basic.jsonl.gz', 'rt', encoding='utf-8') as f:
            rows = [json.loads(line) for line in f]
        candidates = [(range(r['reference_content_tokens']), [{'role': 'user', 'content': r['id']}]) for r in rows]
        for bs, mean in ((16, 6144), (32, 6144), (64, 7680)):
            with self.subTest(bs=bs):
                groups = select_shadow(candidates, 8, bs, mean, .01, random.Random(60), .1, 4)
                selected = [messages[0]['content'] for group in groups for _, messages in group]
                self.assertEqual(len(set(selected)), 8 * bs)
                self.assertTrue(all(abs(sum(len(ids) for ids, _ in group) / bs - mean) <= mean * .01
                                    for group in groups))


if __name__ == '__main__':
    unittest.main()
