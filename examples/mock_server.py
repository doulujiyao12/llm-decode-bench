"""CPU-only protocol demo, NOT an inference engine or performance reference."""
import argparse
import asyncio
import json

from aiohttp import web


def app(dp_size=2, delay=.02):
    running = [0] * dp_size

    async def generate(request):
        body = await request.json()
        rank = body.get("routed_dp_rank", 0)
        if type(rank) is not int or not 0 <= rank < dp_size:
            return web.json_response({"error": "invalid rank"}, status=400)
        native = request.path == '/generate'
        limit = body.get('sampling_params', {}).get('max_new_tokens', body.get('max_tokens', 128))
        if not 1 <= limit <= 10000:
            return web.json_response({"error": "mock output limit is 1..10000"}, status=400)
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        running[rank] += 1
        try:
            for i in range(1, limit + 1):
                await asyncio.sleep(delay)
                if native:
                    obj = {"text": "demo", "meta_info": {"completion_tokens": i,
                        "prompt_tokens": len(body.get('input_ids', [])), "dp_rank": rank}}
                elif 'messages' in body:
                    obj = {"choices": [{"delta": {"content": "demo "}}]}
                else:
                    obj = {"choices": [{"text": "demo "}]}
                await response.write(('data: ' + json.dumps(obj) + '\n\n').encode())
            await response.write(b'data: [DONE]\n\n')
        except (ConnectionResetError, asyncio.CancelledError):
            pass
        finally:
            running[rank] -= 1
        return response

    async def metrics(request):
        text = '\n'.join(f'sglang:{name}{{dp_rank="{rank}"}} {value}'
            for rank in range(dp_size) for name, value in (
                ('num_running_reqs', running[rank]), ('num_queue_reqs', 0)))
        return web.Response(text=text)

    server = web.Application()
    for route in ('/generate', '/v1/completions', '/v1/chat/completions'):
        server.router.add_post(route, generate)
    server.router.add_get('/metrics', metrics)
    return server


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--port', type=int, default=8765)
    p.add_argument('--dp-size', type=int, default=2)
    p.add_argument('--delay-ms', type=float, default=20)
    a = p.parse_args()
    if a.dp_size < 1 or a.delay_ms <= 0:
        p.error('positive dp-size and delay-ms required')
    web.run_app(app(a.dp_size, a.delay_ms / 1000), host='127.0.0.1', port=a.port)
