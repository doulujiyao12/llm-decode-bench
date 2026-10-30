import asyncio
import json
from pathlib import Path
import tempfile
import unittest
import types
from unittest.mock import patch

from llm_decode_bench.analysis import analyze, compare
from llm_decode_bench.cli import config, main, parser
from llm_decode_bench.inputs import load_dataset, manifest, synthetic_rows
from llm_decode_bench.io import append_jsonl, endpoint, write_json
from llm_decode_bench.metrics import SGLANG, SGLANG_PD, parse, ready, window_ready
from llm_decode_bench.plot import plot
from llm_decode_bench.runner import make_body
from llm_decode_bench.sse import Progress, StreamError, messages


def cfg(backend="sglang", bs=1, dp=2, extra=()):
    args = parser().parse_args(["run", "--base-url", "http://127.0.0.1:8000", "--model", "test-model",
        "--dataset", "unused", "--output", "unused", "--backend", backend,
        "--batch-size", str(bs), "--dp-size", str(dp), *extra])
    return config(args, bs)


async def chunks(data, size=1):
    for i in range(0, len(data), size):
        yield data[i:i + size]


class SSETests(unittest.IsolatedAsyncioTestCase):
    async def test_fragmented_utf8_crlf(self):
        data = '\ufeff: ping\r\ndata: {"text":"中文"}\r\n\r\ndata: [DONE]\n\n'.encode()
        for size in (1, 2, 7, 4096):
            got = [x async for x in messages(chunks(data, size))]
            self.assertEqual(got, ['{"text":"中文"}', '[DONE]'])

    async def test_multiline_and_heartbeat(self):
        data = b': ping\n\ndata: {\ndata: "a":1}\n\n'
        self.assertEqual([x async for x in messages(chunks(data))], ['{\n"a":1}'])

    async def test_truncated(self):
        for data in (b'data: {}', b'data: {}\n'):
            with self.assertRaises(StreamError):
                _ = [x async for x in messages(chunks(data))]

    async def test_frame_limit(self):
        with self.assertRaises(StreamError):
            _ = [x async for x in messages(chunks(b'data: ' + b'x' * 128), 16)]

    async def test_invalid_utf8(self):
        with self.assertRaises(UnicodeError):
            _ = [x async for x in messages(chunks(b'data: \xff\n\n'))]

    async def test_cr_only(self):
        self.assertEqual([x async for x in messages(chunks(b'data: x\r\r'))], ['x'])

    def test_malformed_shapes(self):
        for backend, payload in [('sglang', '{"meta_info":[]}'), ('openai', '{"choices":{}}'),
                                 ('openai', '{"choices":[{"delta":5}]}'),
                                 ('openai', '{"usage":{"completion_tokens":-1}}')]:
            with self.subTest(backend=backend, payload=payload):
                with self.assertRaises(StreamError):
                    Progress(backend).consume(payload)

    def test_native_counts_not_text_frames(self):
        p = Progress("sglang")
        self.assertEqual(p.consume('{"meta_info":{"completion_tokens":4}}'), (True, 4))
        self.assertEqual(p.consume('{"text":"same","meta_info":{"completion_tokens":4}}'), (False, 0))
        self.assertEqual(p.consume('{"meta_info":{"completion_tokens":7}}'), (True, 3))
        with self.assertRaises(StreamError):
            p.consume('{"meta_info":{"completion_tokens":6}}')

    def test_chat_role_usage_not_generation(self):
        p = Progress("openai-chat")
        self.assertFalse(p.consume('{"choices":[{"delta":{"role":"assistant"}}]}')[0])
        self.assertTrue(p.consume('{"choices":[{"delta":{"reasoning_content":"x"}}]}')[0])
        self.assertFalse(p.consume('{"choices":[],"usage":{"completion_tokens":10}}')[0])
        self.assertEqual(p.tokens, 10)
        p.consume('{"choices":[{"finish_reason":"stop","delta":{}}]}')
        self.assertTrue(p.finished)

    def test_error_without_private_body(self):
        for payload in ('{"error":"SECRET_BODY"}', 'not-json'):
            with self.assertRaises(StreamError) as ctx:
                Progress("openai").consume(payload)
            self.assertNotIn("SECRET_BODY", str(ctx.exception))

    def test_abort_not_success(self):
        with self.assertRaises(StreamError):
            Progress('sglang').consume('{"meta_info":{"finish_reason":{"type":"abort"}}}')

    def test_metadata_not_arbitrary_content(self):
        with self.assertRaises(StreamError):
            Progress('sglang').consume('{"meta_info":{"prompt_tokens":"private text"}}')


class InputTests(unittest.TestCase):
    def test_nested_cohort(self):
        a = list(synthetic_rows([3, 9, 11], [8, 16, 32], 8, 60))
        b = list(synthetic_rows([3, 9, 11], [8, 16, 32], 16, 60))
        self.assertEqual(a, b[:8])
        self.assertTrue(all(set(r["input_ids"]) <= {3, 9, 11} for r in b))
        self.assertEqual(manifest(a, 2)[3]["target_rank"], 1)

    def test_shared_prefix(self):
        rows = list(synthetic_rows(list(range(100)), [20, 40], 8, 60, 10))
        self.assertTrue(all(r["input_ids"][:10] == rows[0]["input_ids"][:10] for r in rows))
        self.assertNotEqual(rows[0]["input_ids"], rows[1]["input_ids"])

    def test_bad_generation(self):
        with self.assertRaises(ValueError):
            list(synthetic_rows([3], [10], 2, 60, shared_prefix=10))

    def test_generate_command_with_mock_tokenizer(self):
        tok = types.SimpleNamespace(get_vocab=lambda: {'special': 0, 'a': 1, 'b': 2}, all_special_ids=[0])
        mock = types.SimpleNamespace(AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a, **kw: tok))
        with tempfile.TemporaryDirectory() as d, patch.dict('sys.modules', {'transformers': mock}):
            path = Path(d) / 'generated.jsonl'
            status = main(['generate', '--tokenizer', 'mock', '--lengths', '8', '16', '--count', '4', '--output', str(path)])
            self.assertEqual(status, 0)
            rows = load_dataset(path, 4)
            self.assertTrue(all(0 not in r['input_ids'] for r in rows))
            self.assertTrue(Path(str(path) + '.meta.json').exists())

    def test_dataset_validation_and_no_recycling(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.jsonl"
            for row in ({"input_ids": [True]}, {"text": ""}, {"text": "a", "input_ids": [1]}):
                p.write_text(json.dumps(row) + '\n')
                with self.assertRaises(ValueError):
                    load_dataset(p, 1)
            p.write_text('{"text":"one"}\n')
            with self.assertRaises(ValueError):
                load_dataset(p, 2)
            self.assertIsNone(manifest(load_dataset(p, 1), 1)[0]["input_tokens"])


class MetricsTests(unittest.TestCase):
    def metric_text(self):
        return '\n'.join(f'sglang:{name}{{dp_rank="{r}",model="x"}} {v}' for r in range(2)
                         for name, v in [('num_running_reqs', 16), ('num_queue_reqs', 0)])

    def test_exact(self):
        ranks = parse(self.metric_text(), SGLANG)
        self.assertTrue(ready(ranks, SGLANG, 2, 16))
        self.assertFalse(ready(ranks, SGLANG, 2, 32))
        self.assertFalse(ready(ranks, SGLANG_PD, 2, 16))  # Missing PD gauges not zero.

    def test_duplicate_not_summed(self):
        with self.assertRaises(ValueError):
            parse(self.metric_text() + '\n' + self.metric_text(), SGLANG)

    def test_missing_rank(self):
        with self.assertRaises(ValueError):
            parse('sglang:num_running_reqs{model="x"} 16', SGLANG)

    def test_nonfinite(self):
        with self.assertRaises(ValueError):
            parse('sglang:num_running_reqs{dp_rank="0"} NaN', SGLANG)

    def test_global(self):
        spec = {"metrics": {"running": "running", "waiting": "waiting"}, "rank_label": None}
        self.assertTrue(ready(parse('running 32\nwaiting 0', spec), spec, 1, 32))

    def test_filter_labels(self):
        spec = {**SGLANG, "labels": {"model": "x"}}
        text = self.metric_text() + '\n' + self.metric_text().replace('model="x"', 'model="other"')
        self.assertTrue(ready(parse(text, spec), spec, 2, 16))

    def test_window_bracketing_and_gap(self):
        c = cfg(bs=16)
        ranks = parse(self.metric_text(), SGLANG)
        samples = [{"t": i * .5, "ok": True, "ranks": ranks} for i in range(11)]
        self.assertTrue(window_ready(samples, .25, 4.75, c))
        self.assertFalse(window_ready(samples[:3] + samples[9:], .25, 4.75, c))
        samples[3]["ok"] = False
        self.assertFalse(window_ready(samples, .25, 4.75, c))


class AnalysisTests(unittest.TestCase):
    def make_run(self, root, failure=False):
        root.mkdir()
        c = cfg(extra=("--window-seconds", "1", "--guard-seconds", "0.1", "--min-gaps", "5"))
        write_json(root / "config.json", c)
        write_json(root / "manifest.json", manifest([{"input_ids": [1]}] * 2, 2))
        for i in range(2):
            append_jsonl(root / "requests.jsonl", {"index": i, "rid": f"test_{i}", "warmup": False, "ok": not (failure and i == 1),
                "events": [{"t": j * .05 + i * .001, "tokens": 3} for j in range(200)]})
        (root / "metrics.jsonl").touch()
        (root / "capture.jsonl").touch()
        return c

    def test_known_gap_no_mtp_division(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / 'a'; self.make_run(root)
            out = analyze(root)
            self.assertEqual(out["status"], "valid")
            self.assertAlmostEqual(out["sse_gap_p50_ms"], 50)
            self.assertEqual(out["windows"][0]["mean_tokens_per_progress_event"], 3)

    def test_failure_invalidates_all(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / 'a'; self.make_run(root, True)
            self.assertEqual(analyze(root)["status"], "invalid")

    def test_no_metrics_no_downgrade(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / 'a'; c = self.make_run(root)
            c["steady_mode"] = "metrics"; write_json(root / "config.json", c)
            self.assertEqual(analyze(root)["status"], "invalid")

    def test_capture_exclusion(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / 'a'; self.make_run(root)
            append_jsonl(root / 'capture.jsonl', {"action": "arm", "t": .1})
            append_jsonl(root / 'capture.jsonl', {"action": "done", "t": 9.9})
            self.assertEqual(analyze(root)["status"], "invalid")

    def test_comparison_mismatch(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d) / 'a', Path(d) / 'b'
            self.make_run(a); c = self.make_run(b)
            self.assertTrue(compare(a, b)["comparable"])
            c['temperature'] = .6; write_json(b / 'config.json', c)
            self.assertFalse(compare(a, b)["comparable"])
            c['temperature'] = 0; write_json(b / 'config.json', c)
            write_json(b / 'manifest.json', [])
            self.assertIn('input_manifest', compare(a, b)['mismatches'])

    def test_plot(self):
        with tempfile.TemporaryDirectory() as d:
            a = Path(d) / 'a'; self.make_run(a); analyze(a)
            path = Path(d) / 'plot.svg'
            plot([a], path)
            self.assertIn('50.000 ms', path.read_text())
            with self.assertRaises(FileExistsError):
                plot([a], path)

    def test_incomplete_requested_capture_stays_invalid(self):
        with tempfile.TemporaryDirectory() as d:
            a = Path(d) / 'a'; c = self.make_run(a)
            c['event_steps'] = 8; write_json(a / 'config.json', c)
            result = analyze(a)
            self.assertEqual(result['status'], 'invalid')
            self.assertIn('requested GPU capture incomplete or invalid', result['reasons'])


class ConfigTests(unittest.TestCase):
    def test_secret_url_forbidden(self):
        for url in ('http://user:secret@localhost', 'http://localhost/?key=secret', 'file:///tmp/a'):
            with self.assertRaises(ValueError):
                endpoint(url, '')

    def test_native_body(self):
        c = cfg(extra=("--route-dp",))
        body = make_body(c, {"input_ids": [1, 2]}, 3, "id", False, {})
        self.assertEqual(body['routed_dp_rank'], 1)
        self.assertEqual(body['sampling_params']['max_new_tokens'], 2048)

    def test_extra_cannot_override(self):
        with self.assertRaises(ValueError):
            make_body(cfg(), {"input_ids": [1]}, 0, 'id', False, {"stream": False})

    def test_chat_token_ids_forbidden(self):
        with self.assertRaises(ValueError):
            make_body(cfg('openai-chat'), {"input_ids": [1]}, 0, 'id', False, {})

    def test_config_rejections(self):
        for extra in (("--steady-mode", "metrics"), ("--base-url", "http://localhost/v1"), ("--min-gaps", "0")):
            with self.assertRaises(ValueError):
                cfg(extra=extra)


if __name__ == '__main__':
    unittest.main()
