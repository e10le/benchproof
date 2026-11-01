import json

import pytest

from benchproof.drafting import EXAMPLE_DRAFT, MAX_PHOTOS, ProcedureDrafter
from benchproof.llm import SPARK_OPEN_MODEL, CallBudget, ModelError
from benchproof.models import Procedure
from endpoint import FakeEndpoint, png_bytes, reply
from scenes import PROCEDURE_PATH

NARRATION = "First show the empty box. Put the divider in. Notebook and marker go left, mouse and cable go right."


def kit_draft() -> dict:
    return json.loads(PROCEDURE_PATH.read_text(encoding="utf-8"))


def drafter(endpoint: FakeEndpoint, budget: CallBudget | None = None) -> ProcedureDrafter:
    return ProcedureDrafter(endpoint.client(), SPARK_OPEN_MODEL, budget or CallBudget(5))


def test_valid_draft_becomes_an_unapproved_procedure():
    endpoint = FakeEndpoint(reply(json.dumps(kit_draft())))

    draft = drafter(endpoint).draft(NARRATION, [png_bytes()])

    assert draft.id == "mobile_desk_kit"
    assert draft.approved_by is None
    assert draft.final_step.id == "inspect_contents"


def test_request_carries_the_narration_and_every_photo():
    endpoint = FakeEndpoint(reply(json.dumps(kit_draft())))

    drafter(endpoint).draft(NARRATION, [png_bytes("white"), png_bytes("black")])

    text, *images = endpoint.requests[0]["messages"][1]["content"]
    assert NARRATION in text["text"]
    assert len(images) == 2
    assert all(part["image_url"]["url"].startswith("data:image/png;base64,") for part in images)


def test_the_model_cannot_set_the_revision():
    endpoint = FakeEndpoint(reply(json.dumps({**kit_draft(), "revision": 7})))
    assert drafter(endpoint).draft(NARRATION, []).revision == 1


@pytest.mark.parametrize(
    "approval", [{"approved_by": "model", "approved_at": "2026-10-27T20:00:00Z"}, {"approved_by": None}]
)
def test_draft_that_tries_to_approve_itself_is_retried(approval):
    endpoint = FakeEndpoint(reply(json.dumps({**kit_draft(), **approval})), reply(json.dumps(kit_draft())))
    budget = CallBudget(5)

    draft = drafter(endpoint, budget).draft(NARRATION, [])

    assert draft.approved_by is None
    assert budget.used == 2
    assert "only the author approves" in endpoint.requests[1]["messages"][-1]["content"]


def cyclic() -> dict:
    data = kit_draft()
    data["steps"][0]["requires"] = ["inspect_contents"]
    return data


def unknown_condition() -> dict:
    data = kit_draft()
    data["steps"][2]["conditions"].append("ghost")
    return data


def no_final_inspection() -> dict:
    data = kit_draft()
    data["steps"][4]["final_inspection"] = False
    return data


@pytest.mark.parametrize("bad_draft", [cyclic, unknown_condition, no_final_inspection])
def test_draft_the_engine_cannot_run_is_retried_with_the_reason(bad_draft):
    endpoint = FakeEndpoint(reply(json.dumps(bad_draft())), reply(json.dumps(kit_draft())))

    drafter(endpoint).draft(NARRATION, [])

    assert "not usable" in endpoint.requests[1]["messages"][-1]["content"]


def test_three_unusable_drafts_raise_a_model_error():
    endpoint = FakeEndpoint(*(reply(json.dumps(cyclic())) for _ in range(3)))
    with pytest.raises(ModelError, match="no usable reply in 3 attempts"):
        drafter(endpoint).draft(NARRATION, [])


@pytest.mark.parametrize("narration", ["", "   \n"])
def test_empty_narration_is_rejected_before_any_request(narration):
    endpoint = FakeEndpoint()
    with pytest.raises(ValueError, match="narration is empty"):
        drafter(endpoint).draft(narration, [])
    assert endpoint.requests == []


def test_too_many_photos_are_rejected_before_any_request():
    endpoint = FakeEndpoint()
    with pytest.raises(ValueError, match=f"at most {MAX_PHOTOS}"):
        drafter(endpoint).draft(NARRATION, [png_bytes()] * (MAX_PHOTOS + 1))
    assert endpoint.requests == []


def test_prompt_example_is_itself_a_valid_draft():
    example = Procedure.model_validate(EXAMPLE_DRAFT)
    assert example.approved_by is None
    assert example.id != kit_draft()["id"]
