from dataclasses import replace
from datetime import datetime

import cv2
import numpy as np
import pytest

from benchproof.models import Outcome, Visibility, Zone
from benchproof.predicates import (
    CaptureObservation,
    ClosureState,
    Detection,
    calibrate,
    check_registration,
    closure_state,
    evaluate,
)
from scenes import BOX, HEIGHT, LEFT_SLOTS, PACKED, T0, WIDTH, ZONES, observation, placed, rect_mask


@pytest.fixture
def calibration(clock):
    return calibrate(observation("reference", clock()), ZONES, revision=1, approved_by="author", approved_at=clock())


def box_empty(procedure, obs):
    return evaluate(procedure, None, obs, ["box_empty"])[0]


def test_empty_box_passes_when_both_sources_agree(procedure):
    result = box_empty(procedure, observation("c", T0))
    assert result.outcome is Outcome.PASS


def test_box_with_an_item_fails_empty_check(procedure):
    result = box_empty(procedure, observation("c", T0, placed({"mouse": (220, 70, 40, 50)})))
    assert result.outcome is Outcome.FAIL
    assert "mouse" in result.reasons[0]


def test_zero_masks_alone_do_not_prove_empty(procedure):
    # SAM found nothing, but the scene model saw a mouse: the sources disagree.
    result = box_empty(procedure, observation("c", T0, counts={"mouse": 1}))
    assert result.outcome is Outcome.INSUFFICIENT


def test_unexpected_object_counts_against_empty(procedure):
    result = box_empty(procedure, observation("c", T0, unexpected=("phone",)))
    assert result.outcome is Outcome.INSUFFICIENT


def test_empty_check_needs_full_visibility(procedure):
    result = box_empty(procedure, observation("c", T0, visibility=Visibility.PARTIAL))
    assert result.outcome is Outcome.INSUFFICIENT


def test_empty_check_needs_the_box(procedure):
    result = box_empty(procedure, observation("c", T0, box=None))
    assert result.outcome is Outcome.INSUFFICIENT
    assert result.reasons[0] == "Box not found"


def test_items_without_calibration_are_insufficient(procedure):
    result = evaluate(procedure, None, observation("c", T0, placed(PACKED)), ["notebook_left"])[0]
    assert result.outcome is Outcome.INSUFFICIENT
    assert result.reasons[0] == "Zones are not calibrated yet"


def test_item_outside_the_box_fails(procedure, calibration):
    obs = observation("c", T0, {"notebook": [(355, 10, 40, 40)]}, counts={"notebook": 0})
    result = evaluate(procedure, calibration, obs, ["notebook_left"])[0]
    assert result.outcome is Outcome.FAIL
    assert result.reasons[0] == "notebook is outside the box"


def test_item_straddling_the_divider_fails(procedure, calibration):
    obs = observation("c", T0, {"notebook": [(160, 100, 60, 40)]})
    result = evaluate(procedure, calibration, obs, ["notebook_left"])[0]
    assert result.outcome is Outcome.FAIL
    assert result.reasons[0] == "notebook is not fully inside the left zone"


def test_detected_item_the_scene_model_missed_is_insufficient(procedure, calibration):
    obs = observation("c", T0, placed(LEFT_SLOTS), counts={"notebook": 0})
    result = evaluate(procedure, calibration, obs, ["notebook_left"])[0]
    assert result.outcome is Outcome.INSUFFICIENT


def test_duplicate_masks_count_as_one_item(procedure, calibration):
    x, y, w, h = PACKED["marker"]
    doubled = {**placed(PACKED), "marker": [(x, y, w, h), (x + 1, y, w, h)]}
    result = evaluate(procedure, calibration, observation("c", T0, doubled, counts={"marker": 1}), ["contents_exact"])[
        0
    ]
    assert result.outcome is Outcome.PASS


def test_inventory_disagreement_is_insufficient(procedure, calibration):
    obs = observation("c", T0, placed(PACKED), counts={"cable": 2})
    result = evaluate(procedure, calibration, obs, ["contents_exact"])[0]
    assert result.outcome is Outcome.INSUFFICIENT
    assert "cable" in result.reasons[0]


def test_assessment_must_count_every_kit_item(procedure):
    obs = observation("c", T0)
    bad = CaptureObservation(
        capture_id=obs.capture_id,
        sequence=obs.sequence,
        image_sha256=obs.image_sha256,
        captured_at=obs.captured_at,
        width=obs.width,
        height=obs.height,
        scene=obs.scene.model_copy(update={"item_counts": {"mouse": 0}}),
        detections=obs.detections,
        source=obs.source,
    )
    with pytest.raises(ValueError, match="count exactly"):
        evaluate(procedure, None, bad, ["box_empty"])


def test_registration_accepts_small_jitter(calibration):
    x, y, w, h = BOX
    assert check_registration(calibration, observation("c", T0, box=(x + 2, y + 1, w, h))) is None


def test_registration_accepts_near_square_mask_noise():
    reference = observation("reference", T0, box=(100, 50, 200, 199), divider=(195, 50, 10, 199))
    zones = (
        Zone(id="left", polygon=((110, 60), (190, 60), (190, 238), (110, 238))),
        Zone(id="right", polygon=((210, 60), (288, 60), (288, 238), (210, 238))),
    )
    calibration = calibrate(reference, zones, revision=1, approved_by="author", approved_at=T0)
    noisy = observation("noisy", T0, box=(100, 50, 199, 200), divider=(195, 50, 10, 199))
    assert check_registration(calibration, noisy) is None


def test_registration_rejects_moved_box(calibration):
    x, y, w, h = BOX
    problem = check_registration(calibration, observation("c", T0, box=(x + 40, y, w, h)))
    assert problem == "Box moved from its calibrated position"


def test_registration_rejects_resized_box(calibration):
    x, y, w, h = BOX
    problem = check_registration(calibration, observation("c", T0, box=(x - 15, y - 10, w + 30, h + 20)))
    assert problem == "Box size changed since calibration; the camera may have moved"


def test_registration_rejects_missing_divider(calibration):
    assert check_registration(calibration, observation("c", T0, divider=None)) == "Divider not found"


def test_calibration_needs_box_and_divider(clock):
    with pytest.raises(ValueError, match="box and the divider"):
        calibrate(observation("r", clock(), divider=None), ZONES, revision=1, approved_by="a", approved_at=clock())


@pytest.mark.parametrize(
    ("lid_closed", "lid_detected", "expected"),
    [
        (False, False, ClosureState.OPEN),
        (True, True, ClosureState.CLOSED),
        (True, False, ClosureState.UNCLEAR),
        (False, True, ClosureState.UNCLEAR),
    ],
)
def test_closure_needs_both_sources(lid_closed, lid_detected, expected):
    obs = observation("c", T0, lid_closed=lid_closed, lid_detected=lid_detected)
    assert closure_state(obs) is expected


def test_observation_rejects_mask_of_wrong_shape():
    obs = observation("c", T0)
    with pytest.raises(ValueError, match="shape"):
        CaptureObservation(
            capture_id="c",
            sequence=0,
            image_sha256=obs.image_sha256,
            captured_at=T0,
            width=WIDTH,
            height=HEIGHT,
            scene=obs.scene,
            detections=(Detection(label="box", score=0.9, mask=np.ones((10, 10), dtype=bool)),),
            source="synthetic",
        )


def test_observation_rejects_empty_mask():
    obs = observation("c", T0)
    with pytest.raises(ValueError, match="empty mask"):
        CaptureObservation(
            capture_id="c",
            sequence=0,
            image_sha256=obs.image_sha256,
            captured_at=T0,
            width=WIDTH,
            height=HEIGHT,
            scene=obs.scene,
            detections=(Detection(label="box", score=0.9, mask=np.zeros((HEIGHT, WIDTH), dtype=bool)),),
            source="synthetic",
        )


def test_observation_rejects_naive_time():
    obs = observation("c", T0)
    with pytest.raises(ValueError, match="timezone"):
        CaptureObservation(
            capture_id="c",
            sequence=0,
            image_sha256=obs.image_sha256,
            captured_at=datetime(2026, 10, 27, 9, 0),
            width=WIDTH,
            height=HEIGHT,
            scene=obs.scene,
            detections=(),
            source="synthetic",
        )


def test_partial_view_does_not_suppress_source_disagreement(procedure, calibration):
    obs = observation("c", T0, placed(LEFT_SLOTS), visibility=Visibility.PARTIAL, counts={"notebook": 0})
    result = evaluate(procedure, calibration, obs, ["notebook_left"])[0]
    assert result.outcome is Outcome.INSUFFICIENT


def test_registration_rejects_centered_quarter_turn(calibration):
    obs = observation("c", T0, box=(100, 0, 200, 300), divider=(100, 145, 200, 10))
    assert check_registration(calibration, obs) is not None


@pytest.mark.parametrize("angle", [40, 45, 50])
def test_registration_rejects_diagonal_quarter_turn(angle):
    reference, zones = _rotated_box(angle)
    calibration = calibrate(reference, zones, revision=1, approved_by="author", approved_at=T0)
    rotated, _ = _rotated_box(angle + 90)
    assert check_registration(calibration, rotated) is not None


def test_registration_rejects_diagonal_divider_quarter_turn():
    reference, _ = _rotated_box(45)
    rotated, _ = _rotated_box(135)
    base = observation("reference", T0)
    reference = replace(
        base, detections=tuple(d for d in base.detections if d.label != "divider") + (reference.largest("divider"),)
    )
    calibration = calibrate(reference, ZONES, revision=1, approved_by="author", approved_at=T0)
    rotated = replace(
        base, detections=tuple(d for d in base.detections if d.label != "divider") + (rotated.largest("divider"),)
    )
    assert check_registration(calibration, rotated) == "Divider rotated since calibration"


def _rotated_box(angle):
    transform = cv2.getRotationMatrix2D((200, 150), angle, 1.0)
    reference = observation("reference", T0, box=(120, 100, 160, 100), divider=(195, 100, 10, 100))
    detections = tuple(
        Detection(
            d.label,
            d.score,
            cv2.warpAffine(d.mask.astype(np.uint8), transform, (WIDTH, HEIGHT), flags=cv2.INTER_NEAREST).astype(bool),
        )
        for d in reference.detections
    )
    zones = tuple(
        Zone(
            id=zone_id,
            polygon=tuple(map(tuple, cv2.transform(np.array([polygon], dtype=np.float64), transform)[0])),
        )
        for zone_id, polygon in (
            ("left", ((130, 110), (185, 110), (185, 190), (130, 190))),
            ("right", ((215, 110), (270, 110), (270, 190), (215, 190))),
        )
    )
    return replace(reference, detections=detections), zones


def test_registration_rejects_divider_rotation_without_center_shift(calibration):
    obs = observation("c", T0, divider=(100, 145, 200, 10))
    assert check_registration(calibration, obs) is not None


def test_registration_rejects_divider_shortening_without_center_shift(calibration):
    obs = observation("c", T0, divider=(195, 100, 10, 100))
    assert check_registration(calibration, obs) is not None


def test_calibration_rejects_zone_outside_box(clock):
    zones = (Zone(id="left", polygon=((0, 0), (30, 0), (30, 30), (0, 30))), ZONES[1])
    with pytest.raises(ValueError, match="box"):
        calibrate(observation("r", clock()), zones, revision=1, approved_by="a", approved_at=clock())


def test_calibration_rejects_overlapping_zones(clock):
    zones = (ZONES[0], Zone(id="right", polygon=ZONES[0].polygon))
    with pytest.raises(ValueError, match="overlap"):
        calibrate(observation("r", clock()), zones, revision=1, approved_by="a", approved_at=clock())


@pytest.mark.parametrize("field", ["box_angle_deg", "divider_center", "box_aspect_ratio", "divider_aspect_ratio"])
def test_calibration_rejects_nonfinite_registration(field, calibration):
    value = (float("nan"), 0) if field == "divider_center" else float("nan")
    with pytest.raises(ValueError, match="finite"):
        type(calibration).model_validate({**calibration.model_dump(), field: value})


@pytest.mark.parametrize("field", ["box_aspect_ratio", "divider_aspect_ratio"])
@pytest.mark.parametrize("value", [0.5, float("inf"), -float("inf")])
def test_calibration_rejects_invalid_aspect_ratio(field, value, calibration):
    with pytest.raises(ValueError):
        type(calibration).model_validate({**calibration.model_dump(), field: value})


def test_lid_away_from_box_is_not_closure():
    obs = observation("c", T0, lid_closed=True)
    lid = next(d for d in obs.detections if d.label == "lid")
    mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
    mask[0:20, 0:20] = True
    detections = tuple(Detection("lid", lid.score, mask) if d is lid else d for d in obs.detections)
    assert closure_state(replace(obs, detections=detections)) is ClosureState.UNCLEAR


def test_lid_beside_an_open_box_does_not_block_checks():
    obs = observation("c", T0)
    beside = Detection("lid", 0.9, rect_mask(355, 10, 40, 40))
    assert closure_state(replace(obs, detections=obs.detections + (beside,))) is ClosureState.OPEN


def test_closure_needs_the_target_box():
    obs = observation("c", T0, box=None, lid_closed=True)
    assert closure_state(obs) is ClosureState.UNCLEAR


def test_capture_requires_a_valid_image_hash():
    obs = observation("c", T0)
    with pytest.raises(ValueError, match="image_sha256"):
        replace(obs, image_sha256="not-a-hash")


def test_capture_requires_a_nonnegative_sequence():
    with pytest.raises(ValueError, match="sequence"):
        observation("c", T0, sequence=-1)
