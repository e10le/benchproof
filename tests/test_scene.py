import base64
import io
import json

import httpx2
import openai
import pytest
from PIL import Image, UnidentifiedImageError

from benchproof import scene
from benchproof.models import Visibility
from benchproof.scene import BASETEN_BASE_URL, SPARK_OPEN_MODEL, CallBudget, PerceptionError, SceneJudge

PACKED_READING = {
    "interior_visibility": "clear",
    "lid_closed": False,
    "item_counts": {"notebook": 1, "marker": 1, "mouse": 1, "cable": 1},
    "unexpected_items": [],
}


def png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 6), "white").save(buffer, format="PNG")
    return buffer.getvalue()


def completion(content: str | None, finish_reason: str = "stop") -> dict:
    # Shape recorded from a real Baseten reply for meta-models/muse-spark-open-1026.
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1793132479,
        "model": SPARK_OPEN_MODEL,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content, "reasoning_content": "Looking at the box."},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {"prompt_tokens": 151, "completion_tokens": 22, "total_tokens": 173},
    }


class FakeEndpoint:
    """Serves queued HTTP responses to a real OpenAI client and keeps the request bodies."""

    def __init__(self, *responses: tuple[int, dict]) -> None:
        self._responses = list(responses)
        self.requests: list[dict] = []

    def handle(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(json.loads(request.content))
        status, body = self._responses.pop(0)
        return httpx2.Response(status, json=body)

    def judge(self, budget: CallBudget | None = None) -> SceneJudge:
        client = openai.OpenAI(
            api_key="test",
            base_url=BASETEN_BASE_URL,
            max_retries=0,
            http_client=httpx2.Client(transport=httpx2.MockTransport(self.handle)),
        )
        return SceneJudge(client, SPARK_OPEN_MODEL, budget or CallBudget(5))


def reply(content: str | None, finish_reason: str = "stop") -> tuple[int, dict]:
    return 200, completion(content, finish_reason)


def test_valid_reply_becomes_a_scene_assessment(procedure):
    endpoint = FakeEndpoint(reply(json.dumps(PACKED_READING)))

    reading = endpoint.judge().assess(procedure, png_bytes())

    assert reading.interior_visibility is Visibility.CLEAR
    assert reading.item_counts == {"notebook": 1, "marker": 1, "mouse": 1, "cable": 1}


def test_request_sends_the_photo_and_the_kit_items(procedure):
    endpoint = FakeEndpoint(reply(json.dumps(PACKED_READING)))
    photo = png_bytes()

    endpoint.judge().assess(procedure, photo)

    body = endpoint.requests[0]
    assert body["model"] == SPARK_OPEN_MODEL
    assert body["reasoning_effort"] == "medium"
    text, image = body["messages"][1]["content"]
    for item in procedure.items:
        assert f"- {item.id}: {item.name}" in text["text"]
    prefix = "data:image/png;base64,"
    assert image["image_url"]["url"].startswith(prefix)
    assert base64.b64decode(image["image_url"]["url"][len(prefix) :]) == photo


def test_json_inside_a_code_fence_is_accepted(procedure):
    endpoint = FakeEndpoint(reply(f"Here it is:\n```json\n{json.dumps(PACKED_READING)}\n```"))
    assert endpoint.judge().assess(procedure, png_bytes()).lid_closed is False


@pytest.mark.parametrize(
    "bad_reply",
    [
        None,
        "The box looks packed.",
        json.dumps({**PACKED_READING, "item_counts": {"notebook": 1}}),
        json.dumps({**PACKED_READING, "interior_visibility": "mostly"}),
        json.dumps({**PACKED_READING, "item_counts": {**PACKED_READING["item_counts"], "mouse": True}}),
    ],
)
def test_unusable_reply_is_retried_with_the_problem_stated(procedure, bad_reply):
    endpoint = FakeEndpoint(reply(bad_reply), reply(json.dumps(PACKED_READING)))
    budget = CallBudget(5)

    reading = endpoint.judge(budget).assess(procedure, png_bytes())

    assert reading.item_counts["mouse"] == 1
    assert budget.used == 2
    assert "not usable" in endpoint.requests[1]["messages"][-1]["content"]


def test_two_unusable_replies_raise_a_perception_error(procedure):
    endpoint = FakeEndpoint(reply("no idea"), reply("still no idea"))
    with pytest.raises(PerceptionError, match="no usable reply in 2 attempts"):
        endpoint.judge().assess(procedure, png_bytes())


@pytest.mark.parametrize("status", [401, 404, 429, 500])
def test_api_errors_raise_a_perception_error(procedure, status):
    endpoint = FakeEndpoint((status, {"error": {"message": "failed"}}))
    with pytest.raises(PerceptionError, match="request failed"):
        endpoint.judge().assess(procedure, png_bytes())


def test_spent_budget_stops_before_any_request(procedure):
    budget = CallBudget(1)
    budget.spend()
    endpoint = FakeEndpoint()

    with pytest.raises(PerceptionError, match="all 1 model calls"):
        endpoint.judge(budget).assess(procedure, png_bytes())
    assert endpoint.requests == []


def test_retries_stop_at_the_budget(procedure):
    endpoint = FakeEndpoint(reply("no idea"))
    with pytest.raises(PerceptionError, match="all 1 model calls"):
        endpoint.judge(CallBudget(1)).assess(procedure, png_bytes())
    assert len(endpoint.requests) == 1


def test_budget_needs_at_least_one_call():
    with pytest.raises(ValueError):
        CallBudget(0)


def test_non_image_bytes_are_rejected_before_any_request(procedure):
    endpoint = FakeEndpoint()
    with pytest.raises(UnidentifiedImageError):
        endpoint.judge().assess(procedure, b"not an image")
    assert endpoint.requests == []


def test_unsupported_image_format_is_rejected(procedure):
    buffer = io.BytesIO()
    Image.new("RGB", (8, 6)).save(buffer, format="BMP")
    with pytest.raises(ValueError, match="must be one of"):
        FakeEndpoint().judge().assess(procedure, buffer.getvalue())


def test_oversized_photo_is_rejected(procedure, monkeypatch):
    monkeypatch.setattr(scene, "MAX_IMAGE_BYTES", 10)
    with pytest.raises(ValueError, match="larger than"):
        FakeEndpoint().judge().assess(procedure, png_bytes())


def test_baseten_client_requires_the_api_key(monkeypatch):
    monkeypatch.delenv("BASETEN_API_KEY", raising=False)
    with pytest.raises(ValueError, match="BASETEN_API_KEY"):
        scene.baseten_client()


def test_baseten_client_targets_baseten(monkeypatch):
    monkeypatch.setenv("BASETEN_API_KEY", "test-key")
    client = scene.baseten_client()
    assert str(client.base_url).rstrip("/") == BASETEN_BASE_URL
    assert client.timeout == scene.REQUEST_TIMEOUT_SECONDS
