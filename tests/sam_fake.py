"""A fake SAM 3.1 endpoint that answers each prompt with masks in Meta's wire format."""

import json
import threading

import httpx2
import numpy as np

# The parser package has no public encoder; its private one produces masks the real decoder accepts.
from meta_sam_parser._mask_codec import _encode_raster

from benchproof.llm import CallBudget
from benchproof.segmenter import Segmenter
from endpoint import client_for, event_stream

LANE = {"item_id": "msg_1", "output_index": 0, "content_index": 0}


def record(
    object_id: int, box: tuple[int, int, int, int], frame: tuple[int, int], confidence: float | None = 0.9
) -> str:
    """One SAM record for a fully filled rectangle (x, y, w, h) in a photo of size frame (w, h)."""
    x, y, w, h = box
    raster = np.ones((h, w), dtype=np.uint8)
    payload = _encode_raster(raster.ravel().tolist(), w, h)
    c = "" if confidence is None else f";c={confidence}"
    return (
        f"{object_id}<|box;x1={x};y1={y};x2={x + w - 1};y2={y + h - 1};w={frame[0]};h={frame[1]}{c}|>"
        f"<|mask;x=0;y=0{c};data={h},{w},{payload}|>"
    )


def line(*records: str) -> str:
    return "<0f>" + ",".join(records)


def completed() -> dict:
    return {
        "type": "response.completed",
        "sequence_number": 99,
        "response": {"id": "resp_1", "object": "response", "status": "completed", "output": [], "created_at": 0},
    }


def text_reply(text: str) -> httpx2.Response:
    deltas = [text[i : i + 40] for i in range(0, len(text), 40)]
    events = [
        {"type": "response.output_text.delta", **LANE, "delta": d, "sequence_number": i, "logprobs": []}
        for i, d in enumerate(deltas)
    ]
    return event_stream(*events, completed())


class FakeSam:
    """Answers each request by its prompt phrase, so parallel calls can arrive in any order."""

    def __init__(self, replies: dict[str, httpx2.Response | str]) -> None:
        self._replies = replies
        self.requests: list[dict] = []
        self._lock = threading.Lock()

    def handle(self, request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        with self._lock:
            self.requests.append(body)
        prompt = body["input"][0]["content"][0]["text"]
        reply = self._replies.get(prompt, "")
        return text_reply(reply) if isinstance(reply, str) else reply

    def segmenter(self, budget: CallBudget | None = None) -> Segmenter:
        return Segmenter(client_for(self.handle), budget or CallBudget(20))
