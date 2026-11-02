import json

import pytest

from benchproof.llm import SPARK_OPEN_MODEL, CallBudget, ModelError
from benchproof.models import Visibility
from benchproof.scene import SceneJudge
from endpoint import FakeEndpoint, png_bytes, reply

PACKED_READING = {
    "interior_visibility": "clear",
    "lid_closed": False,
    "item_counts": {"notebook": 1, "marker": 1, "mouse": 1, "cable": 1},
    "unexpected_items": [],
}


def judge(endpoint: FakeEndpoint, budget: CallBudget | None = None) -> SceneJudge:
    return SceneJudge(endpoint.client(), SPARK_OPEN_MODEL, budget or CallBudget(5))


def test_valid_reply_becomes_a_scene_assessment(procedure):
    endpoint = FakeEndpoint(reply(json.dumps(PACKED_READING)))

    reading = judge(endpoint).assess(procedure, png_bytes())

    assert reading.interior_visibility is Visibility.CLEAR
    assert reading.item_counts == {"notebook": 1, "marker": 1, "mouse": 1, "cable": 1}


def test_request_sends_the_photo_and_the_kit_items(procedure):
    endpoint = FakeEndpoint(reply(json.dumps(PACKED_READING)))

    judge(endpoint).assess(procedure, png_bytes())

    text, image = endpoint.requests[0]["messages"][1]["content"]
    for item in procedure.items:
        assert f"- {item.id}: {item.name}" in text["text"]
    assert image["image_url"]["url"].startswith("data:image/png;base64,")


def test_the_divider_named_in_the_procedure_is_not_an_unexpected_item(procedure):
    endpoint = FakeEndpoint(reply(json.dumps(PACKED_READING)))

    judge(endpoint).assess(procedure.model_copy(update={"divider_prompt": "paperback book"}), png_bytes())

    text = endpoint.requests[0]["messages"][1]["content"][0]["text"]
    assert "The paperback book across the box is the divider, not an unexpected item." in text
    assert "cardboard" not in text


@pytest.mark.parametrize(
    "bad_reading",
    [
        {**PACKED_READING, "item_counts": {"notebook": 1}},
        {**PACKED_READING, "interior_visibility": "mostly"},
        {**PACKED_READING, "item_counts": {**PACKED_READING["item_counts"], "mouse": True}},
        {**PACKED_READING, "item_counts": {**PACKED_READING["item_counts"], "pen": 1}},
    ],
)
def test_reading_that_does_not_fit_the_kit_is_retried(procedure, bad_reading):
    endpoint = FakeEndpoint(reply(json.dumps(bad_reading)), reply(json.dumps(PACKED_READING)))
    budget = CallBudget(5)

    assert judge(endpoint, budget).assess(procedure, png_bytes()).item_counts["mouse"] == 1
    assert budget.used == 2


def test_two_unusable_readings_raise_a_model_error(procedure):
    endpoint = FakeEndpoint(reply("no idea"), reply("still no idea"))
    with pytest.raises(ModelError, match="no usable reply in 2 attempts"):
        judge(endpoint).assess(procedure, png_bytes())
