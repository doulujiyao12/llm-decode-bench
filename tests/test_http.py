import asyncio
import contextlib
import io
import json
import os
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from aiohttp import web

from llm_decode_bench.runner import run
from llm_decode_bench.io import write_json, append_jsonl
from test_core import cfg


class HTTPTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.mode = "normal"
        self.running = [0, 0]
        self.seen_auth = []
        app = web.Application()
        app.router.add_post('/generate', self.generate)
        app.router.add_post('/v1/completions', self.generate)
        app.router.add_post('/v1/chat/completions', self.generate)
        app.router.add_get('/metrics', self.metrics)
        self.server = web.AppRunner(app)
        await self.server.setup()
        site = web.TCPSite(self.server, '127.0.0.1', 0)
        await site.start()
        self.url = f'http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}'

    async def asyncTearDown(self):
        await self.server.cleanup()
        self.temp.cleanup()

    async def metrics(self, request):
        text = '\n'.join(f'sglang:{key}{{dp_rank="{rank}"}} {value}'
            for rank in range(2) for key, value in (
                ('num_running_reqs', self.running[rank]), ('num_queue_reqs', 0)))
        return web.Response(text=text)

    async def generate(self, request):
        body = await request.json()
        self.seen_auth.append(request.headers.get('Authorization'))
        if self.mode == 'http-error':
            return web.Response(status=503, text='DO_NOT_PERSIST_THIS_PRIVATE_SERVER_BODY')
        native = request.path == '/generate'
        chat = 'messages' in body
        rank = body.get('routed_dp_rank', 0)
        self.running[rank] += 1
        response = web.StreamResponse(headers={'Content-Type': 'text/event-stream'})
        await response.prepare(request)
        try:
            tokens = body.get('sampling_params', {}).get('max_new_tokens', body.get('max_tokens', 60))
            for i in range(1, tokens + 1):
                await asyncio.sleep(.01)
                if native:
                    obj = {'text': 'DO_NOT_PERSIST_GENERATED_TEXT', 'meta_info': {
                        'completion_tokens': i, 'prompt_tokens': len(body.get('input_ids', [])),
                        'dp_rank': 1 - rank if self.mode == 'bad-route' else rank}}
                elif chat:
                    obj = {'choices': [{'delta': {'content': 'hello'}}]}
                else:
                    obj = {'choices': [{'text': 'hello'}]}
                payload = ('data: ' + json.dumps(obj) + '\r\n\r\n').encode()
                await response.write(payload[:7])
                await response.write(payload[7:])
                if self.mode == 'truncated' and i == 4:
                    await response.write(b'data: {')
                    return response
            await response.write(b'data: [DONE]\n\n')
        finally:
            self.running[rank] -= 1
        return response

    async def measure(self, backend='sglang', strict=True):
        data = self.root / f'dataset-{backend}.jsonl'
        row = {'text': 'A small public example.'} if backend == 'openai-chat' else {'input_ids': [1, 2, 3]}
        data.write_text((json.dumps(row) + '\n') * 4)
        c = cfg(backend, bs=2, extra=(
            '--base-url', self.url, '--output-tokens', '60', '--window-seconds', '.12',
            '--guard-seconds', '.02', '--settle-events', '2', '--min-gaps', '3',
            '--metrics-interval', '.02', '--max-metric-gap', '.2'))
        if strict:
            c.update(steady_mode='metrics', metrics_url=self.url + '/metrics', route_dp=True)
        with contextlib.redirect_stdout(io.StringIO()):
            return await run(c, data, self.root / 'result')

    async def test_native_strict(self):
        result = await self.measure()
        self.assertEqual(result['status'], 'valid')
        self.assertEqual(result['successful'], 4)
        self.assertGreater(result['sse_gap_p50_ms'], 8)
        self.assertLess(result['sse_gap_p50_ms'], 100)
        text = (self.root / 'result' / 'requests.jsonl').read_text()
        self.assertNotIn('DO_NOT_PERSIST_GENERATED_TEXT', text)

    async def test_completion_existing_service_no_hook(self):
        result = await self.measure('openai', False)
        self.assertEqual(result['status'], 'valid')
        self.assertEqual(result['batch_verification'], 'client_only_unverified')

    async def test_chat(self):
        result = await self.measure('openai-chat', False)
        self.assertEqual(result['status'], 'valid')

    async def test_truncated_stream_invalid(self):
        self.mode = 'truncated'
        result = await self.measure()
        self.assertEqual(result['status'], 'invalid')
        self.assertEqual(result['successful'], 0)

    async def test_bad_route_invalid(self):
        self.mode = 'bad-route'
        result = await self.measure()
        self.assertEqual(result['successful'], 0)

    async def test_auth_not_saved_or_server_body(self):
        self.mode = 'http-error'
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'UNIT_TEST_ONLY_SECRET'}):
            result = await self.measure()
        self.assertEqual(result['status'], 'invalid')
        self.assertTrue(all(x == 'Bearer UNIT_TEST_ONLY_SECRET' for x in self.seen_auth))
        for path in (self.root / 'result').glob('*.json*'):
            self.assertNotIn('UNIT_TEST_ONLY_SECRET', path.read_text())
            self.assertNotIn('DO_NOT_PERSIST_THIS_PRIVATE_SERVER_BODY', path.read_text())

    async def test_cli_sweep_analyze_compare_and_plot(self):
        data = self.root / 'cli-data.jsonl'
        data.write_text('{"input_ids":[1,2,3]}\n' * 4)

        async def cli(*args):
            p = await asyncio.create_subprocess_exec(sys.executable, '-m', 'llm_decode_bench', *map(str, args),
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            stdout, stderr = await p.communicate()
            self.assertEqual(p.returncode, 0, stdout.decode() + stderr.decode())

        matrix = self.root / 'matrix'
        await cli('sweep', '--base-url', self.url, '--backend', 'sglang', '--model', 'mock',
            '--dataset', data, '--dp-size', '2', '--batch-sizes', '1', '2', '--route-dp', '--output-tokens', '60',
            '--steady-mode', 'metrics', '--metrics-url', self.url + '/metrics',
            '--metrics-interval', '.02', '--window-seconds', '.12', '--guard-seconds', '.02', '--min-gaps', '3',
            '--output', matrix)
        await cli('analyze', matrix / 'bs2-r1')
        await cli('compare', matrix / 'bs2-r1', matrix / 'bs2-r1')
        await cli('plot', matrix / 'bs1-r1', matrix / 'bs2-r1', '--output', matrix / 'plot.svg')
        self.assertTrue((matrix / 'plot.svg').exists())

    async def test_capture_handshake_with_simulated_event_files(self):
        # Tests orchestration only. There is no real GPU or SGLang in this test.
        telemetry = self.root / 'telemetry'; telemetry.mkdir()
        data = self.root / 'event-data.jsonl'
        data.write_text('{"input_ids":[1,2,3]}\n' * 4)
        c = cfg(bs=2, extra=(
            '--base-url', self.url, '--output-tokens', '120', '--window-seconds', '.12',
            '--guard-seconds', '.02', '--settle-events', '2', '--min-gaps', '3',
            '--metrics-interval', '.02', '--max-metric-gap', '.3'))
        c.update(steady_mode='metrics', metrics_url=self.url + '/metrics', route_dp=True, event_steps=8, event_timeout=5)

        async def fake_server_event_writer():
            for _ in range(300):
                if (telemetry / 'control.json').exists():
                    control = json.loads((telemetry / 'control.json').read_text())
                    if control.get('case'):
                        break
                await asyncio.sleep(.01)
            else:
                self.fail('capture not armed')
            case = control['case']
            for rank in range(2):
                for step in range(1, 9):
                    append_jsonl(telemetry / f'{case}-rank{rank}-events.jsonl', {
                        'case': case, 'rank': rank, 'step': step, 'bs': 2,
                        'rids': [f'{case}_run_{i}' for i in range(rank, 4, 2)],
                        'context_sum': 100, 'gpu_ms': 42., 'gap_ms': 1.})
                write_json(telemetry / f'{case}-rank{rank}-done.json', {'rank': rank, 'steps': 8})

        writer = asyncio.create_task(fake_server_event_writer())
        with contextlib.redirect_stdout(io.StringIO()):
            result = await run(c, data, self.root / 'event-result', event_dir=telemetry)
        await writer
        self.assertEqual(result['status'], 'valid')
        gpu = json.loads((self.root / 'event-result/gpu_analysis.json').read_text())
        self.assertTrue(gpu['valid'])
        self.assertEqual(gpu['gpu_step_p50_ms'], 42.)
        self.assertFalse((telemetry / 'client.lock').exists())


if __name__ == '__main__':
    unittest.main()
