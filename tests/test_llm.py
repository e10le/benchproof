import base64
import io
import json

import pytest
from PIL import Image, UnidentifiedImageError

from benchproof import llm
from benchproof.llm import (
    BASETEN_BASE_URL,
    SPARK_OPEN_MODEL,
    CallBudget,
    ModelError,
    image_part,
    json_object,
    request_json,
)
from endpoint import FakeEndpoint, png_bytes, reply

MESSAGES = [{"role": "user", "content": "Count to one."}]


def parse_count(content: str | None) -> int:
    data = json.loads(json_object(content))
    if data.get("count") != 1:
        raise ValueError("count must be 1")
    return data["count"]


def ask(endpoint: FakeEndpoint, budget: CallBudget | None = None, max_attempts: int = 2) -> int:
    return request_json(
        endpoint.client(), SPARK_OPEN_MODEL, budget or CallBudget(5), MESSAGES, parse_count, max_attempts
    )


def test_usable_reply_is_parsed_and_the_request_is_shaped_for_spark_open():
    endpoint = FakeEndpoint(reply('{"count": 1}'))

    assert ask(endpoint) == 1
    body = endpoint.requests[0]
    assert body["model"] == SPARK_OPEN_MODEL
    assert body["reasoning_effort"] == "medium"
    assert body["max_tokens"] == llm.MAX_TOKENS


@pytest.mark.parametrize("bad_reply", [None, "", "one", '{"count": 2}', "{not json}"])
def test_unusable_reply_is_retried_with_the_problem_stated(bad_reply):
    endpoint = FakeEndpoint(reply(bad_reply), reply('{"count": 1}'))
    budget = CallBudget(5)

    assert ask(endpoint, budget) == 1
    assert budget.used == 2
    retry = endpoint.requests[1]["messages"]
    assert retry[:-2] == MESSAGES
    assert "not usable" in retry[-1]["content"]


def test_attempts_run_out_with_a_model_error():
    endpoint = FakeEndpoint(reply("one"), reply("still one"))
    with pytest.raises(ModelError, match="no usable reply in 2 attempts"):
        ask(endpoint)


def test_caller_messages_are_not_changed_by_retries():
    original = [dict(m) for m in MESSAGES]
    ask(FakeEndpoint(reply("one"), reply('{"count": 1}')))
    assert MESSAGES == original


@pytest.mark.parametrize("status", [401, 404, 429, 500])
def test_api_errors_become_model_errors(status):
    endpoint = FakeEndpoint((status, {"error": {"message": "failed"}}))
    with pytest.raises(ModelError, match="request failed"):
        ask(endpoint)


def test_spent_budget_stops_before_any_request():
    budget = CallBudget(1)
    budget.spend()
    endpoint = FakeEndpoint()

    with pytest.raises(ModelError, match="all 1 model calls"):
        ask(endpoint, budget)
    assert endpoint.requests == []


def test_retries_stop_at_the_budget():
    endpoint = FakeEndpoint(reply("one"))
    with pytest.raises(ModelError, match="all 1 model calls"):
        ask(endpoint, CallBudget(1))
    assert len(endpoint.requests) == 1


def test_budget_needs_at_least_one_call():
    with pytest.raises(ValueError):
        CallBudget(0)


@pytest.mark.parametrize(
    ("content", "expected"),
    [('{"a": 1}', '{"a": 1}'), ('```json\n{"a": {"b": 2}}\n```', '{"a": {"b": 2}}'), ('Here: {"a": 1}.', '{"a": 1}')],
)
def test_json_object_takes_the_outermost_object(content, expected):
    assert json_object(content) == expected


@pytest.mark.parametrize("content", [None, "", "no braces", "} backwards {"])
def test_json_object_rejects_replies_without_an_object(content):
    with pytest.raises(ValueError):
        json_object(content)


def test_image_part_encodes_the_photo_as_a_data_url():
    photo = png_bytes()
    url = image_part(photo)["image_url"]["url"]

    prefix = "data:image/png;base64,"
    assert url.startswith(prefix)
    assert base64.b64decode(url[len(prefix) :]) == photo


def test_image_part_rejects_non_images():
    with pytest.raises(UnidentifiedImageError):
        image_part(b"not an image")


def test_image_part_rejects_unsupported_formats():
    buffer = io.BytesIO()
    Image.new("RGB", (8, 6)).save(buffer, format="BMP")
    with pytest.raises(ValueError, match="must be one of"):
        image_part(buffer.getvalue())


def test_image_part_rejects_oversized_photos(monkeypatch):
    monkeypatch.setattr(llm, "MAX_IMAGE_BYTES", 10)
    with pytest.raises(ValueError, match="larger than"):
        image_part(png_bytes())


def test_baseten_client_requires_the_api_key(monkeypatch):
    monkeypatch.delenv("BASETEN_API_KEY", raising=False)
    with pytest.raises(ValueError, match="BASETEN_API_KEY"):
        llm.baseten_client()


def test_baseten_client_targets_baseten_with_a_timeout(monkeypatch):
    monkeypatch.setenv("BASETEN_API_KEY", "test-key")
    client = llm.baseten_client()
    assert str(client.base_url).rstrip("/") == BASETEN_BASE_URL
    assert client.timeout == llm.REQUEST_TIMEOUT_SECONDS
