from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import time

from llm_decode_bench.io import read_jsonl, write_json
from llm_decode_bench.sglang_hook import Recorder, patch_scheduler


class FakeEvent:
    complete = False

    def __init__(self, enable_timing):
        assert enable_timing

    def record(self, stream):
        self.stream = stream

    def query(self):
        return self.complete

    def elapsed_time(self, other):
        return 42.

    def synchronize(self):
        raise AssertionError('Instrumentation must never synchronize')


TORCH = types.SimpleNamespace(cuda=types.SimpleNamespace(Event=FakeEvent),
                              __version__='fake', version=types.SimpleNamespace(hip='fake'))


class EventTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.case = 'ldb_' + 'a' * 32
        self.scheduler = types.SimpleNamespace(ps=types.SimpleNamespace(tp_rank=0, attn_dp_rank=0), forward_stream='fake')
        self.batch = types.SimpleNamespace(forward_mode=types.SimpleNamespace(is_decode=lambda: True),
            reqs=[types.SimpleNamespace(rid=self.case + '_run_0', origin_input_ids=[1, 2], output_ids=[3])])
        write_json(self.root / 'control.json', {'case': self.case, 'rid_prefix': self.case + '_run_',
                    'steps': 5, 'batch_size': 1, 'dp_size': 1, 'expires': time.time() + 30})
        FakeEvent.complete = False

    def tearDown(self):
        self.temp.cleanup()

    def test_deferred_query_no_synchronize(self):
        r = Recorder(self.root, TORCH, [0])
        for _ in range(5):
            item = r.before(self.scheduler, self.batch)
            r.after(self.scheduler, item)
        self.assertEqual(len(r.pending), 5)
        r.drain()
        self.assertEqual(len(r.pending), 5)
        FakeEvent.complete = True
        r.drain()
        self.assertEqual(r.pending, [])
        rows = read_jsonl(self.root / f'{self.case}-rank0-events.jsonl')
        self.assertEqual(len(rows), 5)
        self.assertTrue(all(row['gpu_ms'] == 42 for row in rows))
        self.assertTrue((self.root / f'{self.case}-rank0-done.json').exists())

    def test_other_request_not_measured(self):
        self.batch.reqs[0].rid = 'someone_else'
        r = Recorder(self.root, TORCH, [0])
        self.assertIsNone(r.before(self.scheduler, self.batch))

    def test_nonleader_skipped(self):
        r = Recorder(self.root, TORCH, [1])
        self.assertIsNone(r.before(self.scheduler, self.batch))

    def test_original_called_once_and_exceptions_preserved(self):
        calls = []
        class Scheduler:
            def run_batch(self, batch):
                calls.append(batch)
                raise RuntimeError('original-model-error')
        module = types.SimpleNamespace(Scheduler=Scheduler)
        with patch.dict('sys.modules', {'torch': TORCH}):
            patch_scheduler(module, self.root, [0])
        scheduler = Scheduler()  # Missing rank attributes disables diagnostics.
        with self.assertRaisesRegex(RuntimeError, 'original-model-error'):
            scheduler.run_batch(self.batch)
        self.assertEqual(len(calls), 1)
        with self.assertRaisesRegex(RuntimeError, 'original-model-error'):
            scheduler.run_batch(self.batch)
        self.assertEqual(len(calls), 2)


if __name__ == '__main__':
    unittest.main()
