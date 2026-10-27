import json

import pytest
from pydantic import ValidationError

from benchproof.models import BBox, ConditionResult, Judgment, Outcome, Procedure, SceneAssessment, Zone, aggregate
from scenes import PROCEDURE_PATH, T0


def kit() -> dict:
    return json.loads(PROCEDURE_PATH.read_text(encoding="utf-8"))


def test_kit_procedure_is_valid_and_unapproved():
    procedure = Procedure.model_validate(kit())
    assert procedure.approved_by is None
    assert procedure.final_step.id == "inspect_contents"


def test_approve_returns_an_approved_copy():
    procedure = Procedure.model_validate(kit())
    approved = procedure.approve("author", T0)
    assert approved.approved_by == "author"
    assert procedure.approved_by is None


def test_approval_needs_author_and_time_together():
    with pytest.raises(ValidationError, match="together"):
        Procedure.model_validate({**kit(), "approved_by": "author"})


def test_prerequisite_cycle_is_rejected():
    data = kit()
    data["steps"][0]["requires"] = ["inspect_contents"]
    with pytest.raises(ValidationError, match="cycle"):
        Procedure.model_validate(data)


def test_unknown_condition_reference_is_rejected():
    data = kit()
    data["steps"][2]["conditions"].append("ghost")
    with pytest.raises(ValidationError, match="unknown condition 'ghost'"):
        Procedure.model_validate(data)


def test_condition_checked_by_two_steps_is_rejected():
    data = kit()
    data["steps"][3]["conditions"].append("notebook_left")
    with pytest.raises(ValidationError, match="used by both"):
        Procedure.model_validate(data)


def test_unchecked_condition_is_rejected():
    data = kit()
    data["steps"][3]["conditions"] = ["mouse_right"]
    with pytest.raises(ValidationError, match="cable_right"):
        Procedure.model_validate(data)


def test_final_inspection_must_follow_every_step():
    data = kit()
    data["steps"][4]["requires"] = ["pack_left"]
    with pytest.raises(ValidationError, match="does not depend on steps"):
        Procedure.model_validate(data)


def test_two_final_inspections_are_rejected():
    data = kit()
    data["steps"][3]["final_inspection"] = True
    with pytest.raises(ValidationError, match="exactly one final inspection"):
        Procedure.model_validate(data)


def test_reserved_item_id_is_rejected():
    data = kit()
    data["items"][0]["id"] = "lid"
    data["conditions"][2]["item"] = "lid"
    with pytest.raises(ValidationError, match="reserved"):
        Procedure.model_validate(data)


def test_box_empty_cannot_be_an_invariant():
    data = kit()
    data["conditions"][0]["role"] = "invariant"
    with pytest.raises(ValidationError, match="observation"):
        Procedure.model_validate(data)


def test_item_condition_needs_a_known_zone():
    data = kit()
    data["conditions"][2]["zone"] = "middle"
    with pytest.raises(ValidationError, match="unknown zone"):
        Procedure.model_validate(data)


@pytest.mark.parametrize(
    ("outcomes", "expected"),
    [
        ([Outcome.PASS, Outcome.PASS], Outcome.PASS),
        ([Outcome.PASS, Outcome.INSUFFICIENT], Outcome.INSUFFICIENT),
        ([Outcome.INSUFFICIENT, Outcome.FAIL, Outcome.PASS], Outcome.FAIL),
    ],
)
def test_aggregate_precedence(outcomes, expected):
    assert aggregate(outcomes) is expected


def test_aggregate_of_nothing_is_not_a_pass():
    with pytest.raises(ValueError):
        aggregate([])


def test_non_pass_result_needs_an_instruction():
    with pytest.raises(ValidationError, match="needs an instruction"):
        ConditionResult(condition_id="x", outcome=Outcome.FAIL, reasons=("bad",))


def test_judgment_outcome_must_match_results():
    result = ConditionResult(condition_id="x", outcome=Outcome.FAIL, reasons=("bad",), instruction="fix it")
    with pytest.raises(ValidationError, match="does not match"):
        Judgment(
            capture_id="c",
            capture_sequence=0,
            image_sha256="0" * 64,
            target_step=None,
            captured_at=T0,
            judged_at=T0,
            perception_source="synthetic",
            procedure_revision=1,
            calibration_revision=None,
            results=(result,),
            outcome=Outcome.PASS,
        )


@pytest.mark.parametrize("author", ["", "   "])
def test_direct_approval_requires_a_named_author(author):
    with pytest.raises(ValidationError, match="author"):
        Procedure.model_validate({**kit(), "approved_by": author, "approved_at": T0})


@pytest.mark.parametrize("index", [1, 2, 6])
def test_persistent_condition_cannot_be_an_observation(index):
    data = kit()
    data["conditions"][index]["role"] = "observation"
    with pytest.raises(ValidationError, match="invariant"):
        Procedure.model_validate(data)


def test_final_inspection_requires_an_exact_inventory_condition():
    data = kit()
    data["conditions"][6] = {**data["conditions"][6], "kind": "divider_in_place"}
    with pytest.raises(ValidationError, match="inventory"):
        Procedure.model_validate(data)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_box_coordinates_must_be_finite(value):
    with pytest.raises(ValidationError, match="finite"):
        BBox(x=value, y=0, w=10, h=10)


@pytest.mark.parametrize(
    "polygon",
    [
        ((10, 10), (10, 10), (10, 10)),
        ((10, 10), (20, 20), (30, 30)),
        ((10, 10), (90, 90), (90, 10), (10, 90)),
        ((10, 10), (90, 10), (10, 90), (90, 90), (50, 0)),
    ],
)
def test_zone_rejects_degenerate_or_self_crossing_polygon(polygon):
    with pytest.raises(ValidationError, match="polygon"):
        Zone(id="left", polygon=polygon)


def test_scene_counts_reject_boolean_values():
    with pytest.raises(ValidationError):
        SceneAssessment(interior_visibility="clear", lid_closed=False, item_counts={"mouse": True})


@pytest.mark.parametrize("zone_id", ["left-zone", ""])
def test_procedure_zone_ids_must_match_calibration_ids(zone_id):
    data = kit()
    data["zones"] = [zone_id, "right"]
    for condition in data["conditions"]:
        if condition.get("zone") == "left":
            condition["zone"] = zone_id
    with pytest.raises(ValidationError, match="zones"):
        Procedure.model_validate(data)
