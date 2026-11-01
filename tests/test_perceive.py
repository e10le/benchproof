import io
import json
from datetime import UTC, datetime

from PIL import Image
from sam_fake import FakeSam, line, record

from benchproof.llm import SPARK_OPEN_MODEL, CallBudget
from benchproof.models import Outcome
from benchproof.perceive import perceive
from benchproof.photos import prepare
from benchproof.predicates import calibrate, evaluate
from benchproof.scene import SceneJudge
from endpoint import FakeEndpoint, reply
from scenes import BOX, DIVIDER, HEIGHT, PACKED, T0, WIDTH, ZONES, observation

FRAME = (WIDTH, HEIGHT)
TAKEN = datetime(2026, 11, 1, 22, 0, tzinfo=UTC)


def photo():
    buffer = io.BytesIO()
    Image.new("RGB", FRAME, "white").save(buffer, format="PNG")
    return prepare(buffer.getvalue())


def sam_replies(procedure, items: dict[str, tuple[int, int, int, int]]) -> dict[str, str]:
    replies = {
        procedure.box_prompt: line(record(0, BOX, FRAME)),
        procedure.divider_prompt: line(record(0, DIVIDER, FRAME)),
    }
    for item in procedure.items:
        if item.id in items:
            replies[item.sam_prompt] = line(record(0, items[item.id], FRAME))
    return replies


def judge_reply(counts: dict[str, int]) -> FakeEndpoint:
    reading = {"interior_visibility": "clear", "lid_closed": False, "item_counts": counts, "unexpected_items": []}
    return FakeEndpoint(reply(json.dumps(reading)))


def run(procedure, items, counts, budget=None):
    budget = budget or CallBudget(20)
    judge = SceneJudge(judge_reply(counts).client(), SPARK_OPEN_MODEL, budget)
    obs = perceive(procedure, photo(), "c1", 4, TAKEN, FakeSam(sam_replies(procedure, items)).segmenter(budget), judge)
    calibration = calibrate(observation("reference", T0), ZONES, revision=1, approved_by="author", approved_at=T0)
    return obs, evaluate(procedure, calibration, obs, ["mouse_right", "cable_right", "contents_exact"])


def test_packed_kit_passes_end_to_end(procedure):
    budget = CallBudget(20)
    obs, results = run(procedure, PACKED, {k: 1 for k in PACKED}, budget)

    assert [r.outcome for r in results] == [Outcome.PASS] * 3
    assert (obs.width, obs.height, obs.sequence, obs.captured_at) == (WIDTH, HEIGHT, 4, TAKEN)
    assert obs.source == "sam-3.1 + meta-models/muse-spark-open-1026"
    assert budget.used == 8


def test_missing_cable_fails_end_to_end(procedure):
    items = {k: v for k, v in PACKED.items() if k != "cable"}
    _, results = run(procedure, items, {**{k: 1 for k in PACKED}, "cable": 0})

    outcomes = {r.condition_id: r.outcome for r in results}
    assert outcomes == {"mouse_right": Outcome.PASS, "cable_right": Outcome.FAIL, "contents_exact": Outcome.FAIL}


def test_detections_and_scene_reading_must_agree_end_to_end(procedure):
    _, results = run(procedure, PACKED, {**{k: 1 for k in PACKED}, "mouse": 0})

    assert {r.condition_id: r.outcome for r in results}["mouse_right"] is Outcome.INSUFFICIENT
