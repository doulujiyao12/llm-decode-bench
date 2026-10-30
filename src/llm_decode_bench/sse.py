"""Incremental SSE framing. Network chunks are NOT SSE messages or tokens."""
import codecs
import json


class StreamError(ValueError):
    pass


def optional_count(value):
    if value is not None and (type(value) is not int or value < 0):
        raise StreamError("Invalid token/rank count metadata")
    return value


async def messages(chunks, max_frame_bytes=16 * 1024 * 1024):
    decoder = codecs.getincrementaldecoder("utf-8-sig")("strict")
    buffer, data, frame_size = "", [], 0
    # Normalize CR, LF and CRLF across arbitrary byte boundaries.
    async for chunk in chunks:
        buffer += decoder.decode(chunk)
        while True:
            candidates = [p for p in (buffer.find("\r"), buffer.find("\n")) if p >= 0]
            if not candidates:
                break
            p = min(candidates)
            if buffer[p] == "\r" and p == len(buffer) - 1:
                break
            width = 2 if buffer[p:p + 2] == "\r\n" else 1
            line, buffer = buffer[:p], buffer[p + width:]
            if not line:
                if data:
                    yield "\n".join(data)
                data, frame_size = [], 0
            elif line.startswith("data:"):
                value = line[5:]
                if value.startswith(" "):
                    value = value[1:]
                data.append(value)
                frame_size += len(value.encode())
            if frame_size + len(buffer.encode()) > max_frame_bytes:
                raise StreamError("SSE frame too large")
        if frame_size + len(buffer.encode()) > max_frame_bytes:
            raise StreamError("SSE frame too large")
    buffer += decoder.decode(b"", final=True)
    if buffer == "\r":  # A CR-only blank line is also a valid final delimiter.
        if data:
            yield "\n".join(data)
        buffer, data = "", []
    if buffer.strip() or data:
        raise StreamError("Truncated SSE frame at EOF")


class Progress:
    def __init__(self, backend):
        self.backend = backend
        self.tokens = 0
        self.prompt_tokens = None
        self.dp_rank = None
        self.finished = False
        self.done = False
        self.finish_reason = None
        self.cached_tokens = None

    def consume(self, payload):
        """Return (has_generation_progress, token_delta_or_None), never retain text."""
        if payload.strip() == "[DONE]":
            self.done = True
            return False, None
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            raise StreamError("Non-JSON SSE data") from None
        if not isinstance(obj, dict) or obj.get("error"):
            raise StreamError("Server returned an error event (body omitted)")
        if self.backend == "sglang":
            meta = obj.get("meta_info", {})
            if not isinstance(meta, dict):
                raise StreamError("Invalid meta_info")
            count = meta.get("completion_tokens", self.tokens)
            if type(count) is not int or count < self.tokens:
                raise StreamError("Invalid/non-monotonic completion_tokens")
            delta = count - self.tokens
            self.tokens = count
            self.prompt_tokens = optional_count(meta.get("prompt_tokens", self.prompt_tokens))
            self.dp_rank = optional_count(meta.get("dp_rank", self.dp_rank))
            self.cached_tokens = optional_count(meta.get("cached_tokens", self.cached_tokens))
            if meta.get("finish_reason") is not None:
                self.finish_reason = meta["finish_reason"]
                reason = self.finish_reason.get("type") if isinstance(self.finish_reason, dict) else self.finish_reason
                if reason in {"abort", "error", "cancelled", "canceled"}:
                    raise StreamError("Server aborted generation")
                self.finished = True
            return delta > 0, delta
        usage = obj.get("usage") or {}
        if not isinstance(usage, dict):
            raise StreamError("Invalid usage")
        if usage:
            count = usage.get("completion_tokens", self.tokens)
            if type(count) is not int or count < self.tokens:
                raise StreamError("Invalid/non-monotonic usage token count")
            self.tokens = count
            self.prompt_tokens = optional_count(usage.get("prompt_tokens", self.prompt_tokens))
        choices = obj.get("choices", [])
        if choices is None:
            choices = []
        if not isinstance(choices, list) or any(not isinstance(x, dict) for x in choices):
            raise StreamError("Invalid choices")
        if len(choices) > 1:
            raise StreamError("Only n=1 is supported")
        progress = False
        for choice in choices:
            delta = choice.get("delta") or {}
            if not isinstance(delta, dict):
                raise StreamError("Invalid choice delta")
            progress = bool(choice.get("text") or delta.get("content")
                            or delta.get("reasoning_content") or delta.get("reasoning")
                            or delta.get("tool_calls"))
            if choice.get("finish_reason") is not None:
                self.finish_reason = choice["finish_reason"]
                if self.finish_reason in {"error", "cancelled", "canceled", "content_filter"}:
                    raise StreamError("Server stopped generation abnormally")
                self.finished = True
        return progress, None
