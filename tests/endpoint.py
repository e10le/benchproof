"""A fake model endpoint that serves recorded replies to a real OpenAI client, offline."""

import io
import json
from collections.abc import Callable

import httpx2
import openai
from PIL import Image

from benchproof.llm import BASETEN_BASE_URL, SPARK_OPEN_MODEL


def png_bytes(color: str = "white") -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 6), color).save(buffer, format="PNG")
    return buffer.getvalue()


def reply(content: str | None, finish_reason: str = "stop") -> tuple[int, dict]:
    # Shape recorded from a real Baseten reply for meta-models/muse-spark-open-1026.
    return 200, {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1793132479,
        "model": SPARK_OPEN_MODEL,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content, "reasoning_content": "Looking at the photo."},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {"prompt_tokens": 151, "completion_tokens": 22, "total_tokens": 173},
    }


class FakeEndpoint:
    """Serves queued HTTP responses and keeps every request body it received."""

    def __init__(self, *responses: tuple[int, dict]) -> None:
        self._responses = list(responses)
        self.requests: list[dict] = []

    def handle(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(json.loads(request.content))
        status, body = self._responses.pop(0)
        return httpx2.Response(status, json=body)

    def client(self) -> openai.OpenAI:
        return client_for(self.handle)


def client_for(handler: Callable[[httpx2.Request], httpx2.Response]) -> openai.OpenAI:
    return openai.OpenAI(
        api_key="test",
        base_url=BASETEN_BASE_URL,
        max_retries=0,
        http_client=httpx2.Client(transport=httpx2.MockTransport(handler)),
    )


def event_stream(*events: dict) -> httpx2.Response:
    body = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)
    return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=body.encode("utf-8"))
