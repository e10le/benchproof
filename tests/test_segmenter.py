import base64

import httpx2
import numpy as np
import pytest
from sam_fake import LANE, FakeSam, line, record

from benchproof import llm
from benchproof.llm import META_BASE_URL, CallBudget, ModelError
from benchproof.photos import prepare
from benchproof.segmenter import SAM_MODEL, scene_prompts
from endpoint import event_stream, png_bytes

PHOTO = prepare(png_bytes())  # 8 x 6
FRAME = (PHOTO.width, PHOTO.height)


def test_mask_lands_at_its_box_in_the_full_photo():
    sam = FakeSam({"mouse": line(record(0, (2, 1, 3, 2), FRAME, confidence=0.87))})

    [mouse] = sam.segmenter().segment(PHOTO, "mouse", "mouse")

    expected = np.zeros((6, 8), dtype=bool)
    expected[1:3, 2:5] = True
    assert mouse.label == "mouse"
    assert mouse.score == pytest.approx(0.87)
    assert np.array_equal(mouse.mask, expected)


def test_several_matches_become_several_detections():
    sam = FakeSam({"marker": line(record(0, (0, 0, 2, 2), FRAME), record(3, (5, 3, 2, 2), FRAME))})
    assert len(sam.segmenter().segment(PHOTO, "marker", "marker")) == 2


def test_request_is_shaped_like_meta_examples():
    sam = FakeSam({})
    sam.segmenter().segment(PHOTO, "cable", "coiled cable")

    body = sam.requests[0]
    assert body["model"] == SAM_MODEL
    assert body["stream"] is True
    assert body["metadata"] == {"include_confidence": "true"}
    text, image = body["input"][0]["content"]
    assert text == {"type": "input_text", "text": "coiled cable"}
    prefix = "data:image/jpeg;base64,"
    assert image["image_url"].startswith(prefix)
    assert base64.b64decode(image["image_url"][len(prefix) :]) == PHOTO.jpeg


def test_no_match_is_an_empty_result_not_an_error():
    assert FakeSam({"lid": ""}).segmenter().segment(PHOTO, "lid", "lid") == []


def test_box_with_an_empty_mask_is_not_a_detection():
    from meta_sam_parser._mask_codec import _encode_raster

    empty = _encode_raster([0] * 6, 3, 2)
    reply = line(f"0<|box;x1=1;y1=1;x2=3;y2=2;w=8;h=6;c=0.5|><|mask;x=0;y=0;c=0.5;data=2,3,{empty}|>")
    assert FakeSam({"mouse": reply}).segmenter().segment(PHOTO, "mouse", "mouse") == []


def test_mask_raster_smaller_than_its_box_is_scaled_to_the_box():
    from meta_sam_parser._mask_codec import _encode_raster

    half = _encode_raster([1, 1, 1, 1], 2, 2)
    reply = line(f"0<|box;x1=0;y1=0;x2=3;y2=3;w=8;h=6;c=0.9|><|mask;x=0;y=0;c=0.9;data=2,2,{half}|>")

    [found] = FakeSam({"mouse": reply}).segmenter().segment(PHOTO, "mouse", "mouse")

    assert found.mask[0:4, 0:4].all()
    assert found.mask.sum() == 16


def test_malformed_output_is_a_model_error():
    sam = FakeSam({"mouse": "<0f>0<|box;x1=oops|>"})
    with pytest.raises(ModelError, match="malformed"):
        sam.segmenter().segment(PHOTO, "mouse", "mouse")


def test_box_from_a_larger_image_is_a_model_error():
    sam = FakeSam({"mouse": line(record(0, (10, 1, 3, 2), (20, 6)))})
    with pytest.raises(ModelError, match="outside the photo"):
        sam.segmenter().segment(PHOTO, "mouse", "mouse")


def test_missing_confidence_is_a_model_error():
    sam = FakeSam({"mouse": line(record(0, (1, 1, 2, 2), FRAME, confidence=None))})
    with pytest.raises(ModelError, match="without confidence"):
        sam.segmenter().segment(PHOTO, "mouse", "mouse")


@pytest.mark.parametrize("ending", ["response.failed", "response.incomplete", "error"])
def test_stream_that_does_not_complete_is_a_model_error(ending):
    reply = event_stream({"type": ending, "sequence_number": 1, "message": "stopped", "code": "x", "param": None})
    with pytest.raises(ModelError, match=ending):
        FakeSam({"mouse": reply}).segmenter().segment(PHOTO, "mouse", "mouse")


def test_stream_cut_off_before_completion_is_a_model_error():
    cut = event_stream({"type": "response.output_text.delta", **LANE, "delta": "<0f>", "sequence_number": 1})
    with pytest.raises(ModelError, match="before it completed"):
        FakeSam({"mouse": cut}).segmenter().segment(PHOTO, "mouse", "mouse")


@pytest.mark.parametrize("status", [401, 429, 500])
def test_http_errors_are_model_errors(status):
    reply = httpx2.Response(status, json={"error": {"message": "failed"}})
    with pytest.raises(ModelError, match="request for 'mouse' failed"):
        FakeSam({"mouse": reply}).segmenter().segment(PHOTO, "mouse", "mouse")


def test_segment_all_asks_once_per_scene_object(procedure):
    prompts = scene_prompts(procedure)
    sam = FakeSam({procedure.box_prompt: line(record(0, (0, 0, 8, 6), FRAME))})
    budget = CallBudget(20)

    detections = sam.segmenter(budget).segment_all(procedure, PHOTO)

    assert sorted(r["input"][0]["content"][0]["text"] for r in sam.requests) == sorted(prompts.values())
    assert budget.used == len(prompts) == 7
    assert [d.label for d in detections] == ["box"]


def test_segment_all_stops_when_the_budget_runs_out(procedure):
    with pytest.raises(ModelError, match="model calls"):
        FakeSam({}).segmenter(CallBudget(3)).segment_all(procedure, PHOTO)


def test_scene_prompts_cover_box_divider_lid_and_every_item(procedure):
    prompts = scene_prompts(procedure)
    assert prompts["box"] == procedure.box_prompt
    assert prompts["lid"] == procedure.lid_prompt
    assert {item.id: item.sam_prompt for item in procedure.items}.items() <= prompts.items()


def test_meta_client_requires_its_key(monkeypatch):
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    with pytest.raises(ValueError, match="MODEL_API_KEY"):
        llm.meta_client()


def test_meta_client_targets_the_meta_api(monkeypatch):
    monkeypatch.setenv("MODEL_API_KEY", "test-key")
    assert str(llm.meta_client().base_url).rstrip("/") == META_BASE_URL
