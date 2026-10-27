import threading
from dataclasses import replace
from datetime import timedelta, timezone

import pytest

from benchproof.models import CLOSURE_CONDITION_ID, COMPLETION_CLAIM, Outcome, Visibility
from benchproof.predicates import Detection, calibrate
from benchproof.session import Session
from scenes import DIVIDER, LEFT_SLOTS, PACKED, RIGHT_SLOTS, ZONES, observation, placed, rect_mask


def without(*item_ids: str) -> dict[str, tuple[int, int, int, int]]:
    return {k: v for k, v in PACKED.items() if k not in item_ids}


def test_unapproved_procedure_cannot_start(procedure):
    unapproved = procedure.model_copy(update={"approved_by": None, "approved_at": None})
    with pytest.raises(ValueError, match="approved"):
        Session(unapproved)


def test_full_run_produces_record_with_bounded_claim(inspected, clock):
    closing = inspected.check(observation("closed", clock(), lid_closed=True), None)

    assert closing.outcome is Outcome.PASS
    record = inspected.record()
    assert record.claim == COMPLETION_CLAIM
    assert record.inspection_capture_id == "inspect"
    assert record.closure_capture_id == "closed"
    assert [s.step_id for s in record.steps] == [
        "show_empty_box",
        "place_divider",
        "pack_left",
        "pack_right",
        "inspect_contents",
    ]
    assert record.overridden == ()


def test_record_before_completion_raises(inspected):
    with pytest.raises(ValueError, match="not completed"):
        inspected.record()


def test_pack_steps_are_interchangeable(calibrated, clock):
    right_first = calibrated.check(observation("right", clock(), placed(RIGHT_SLOTS)), "pack_right")
    left_second = calibrated.check(observation("left", clock(), placed(PACKED)), "pack_left")

    assert right_first.outcome is Outcome.PASS
    assert left_second.outcome is Outcome.PASS


def test_step_before_its_prerequisite_is_rejected(session, clock):
    with pytest.raises(ValueError, match="not available"):
        session.check(observation("early", clock(), placed(LEFT_SLOTS)), "pack_left")


def test_missing_cable_fails_and_names_the_item(calibrated, clock):
    calibrated.check(observation("left", clock(), placed(LEFT_SLOTS)), "pack_left")
    judgment = calibrated.check(observation("right", clock(), placed(without("cable"))), "pack_right")

    assert judgment.outcome is Outcome.FAIL
    cable = judgment.result("cable_right")
    assert cable.outcome is Outcome.FAIL
    assert "cable was not found" in cable.reasons[0]
    assert "pack_right" not in calibrated.passed_steps


def test_mouse_in_wrong_compartment_fails_with_location(calibrated, clock):
    calibrated.check(observation("left", clock(), placed(LEFT_SLOTS)), "pack_left")
    wrong = {**placed(PACKED), "mouse": [(80, 170, 40, 50)]}
    judgment = calibrated.check(observation("right", clock(), wrong), "pack_right")

    mouse = judgment.result("mouse_right")
    assert mouse.outcome is Outcome.FAIL
    assert "left zone, not the right zone" in mouse.reasons[0]


def test_extra_marker_fails_final_inspection(calibrated, clock):
    calibrated.check(observation("left", clock(), placed(LEFT_SLOTS)), "pack_left")
    calibrated.check(observation("right", clock(), placed(PACKED)), "pack_right")
    extra = {**placed(PACKED), "marker": [LEFT_SLOTS["marker"], (140, 150, 40, 20)]}
    judgment = calibrated.check(observation("inspect", clock(), extra), "inspect_contents")

    assert judgment.outcome is Outcome.FAIL
    assert "2 x marker" in judgment.result("contents_exact").reasons[0]
    assert not calibrated.ready_to_close


def test_pen_mistaken_for_marker_never_passes(calibrated, clock):
    calibrated.check(observation("left", clock(), placed(LEFT_SLOTS)), "pack_left")
    calibrated.check(observation("right", clock(), placed(PACKED)), "pack_right")
    # SAM's marker prompt matched the pen; the scene model saw a pen and no marker.
    pen = observation("inspect", clock(), placed(PACKED), counts={"marker": 0}, unexpected=("pen",))
    judgment = calibrated.check(pen, "inspect_contents")

    assert judgment.outcome is not Outcome.PASS
    assert judgment.result("marker_left").outcome is Outcome.INSUFFICIENT
    assert "unexpected pen" in judgment.result("contents_exact").reasons[0]
    assert not calibrated.ready_to_close


def test_hand_covering_an_item_asks_for_a_clear_view(calibrated, clock):
    calibrated.check(observation("left", clock(), placed(LEFT_SLOTS)), "pack_left")
    covered = observation(
        "right", clock(), placed(without("cable")), visibility=Visibility.PARTIAL, counts={"cable": 0}
    )
    judgment = calibrated.check(covered, "pack_right")

    assert judgment.outcome is Outcome.INSUFFICIENT
    assert judgment.result("cable_right").outcome is Outcome.INSUFFICIENT
    assert any("whole box interior is visible" in step for step in judgment.recovery)


def test_premature_closure_fails_then_reopening_allows_completion(calibrated, clock):
    calibrated.check(observation("left", clock(), placed(LEFT_SLOTS)), "pack_left")
    calibrated.check(observation("right", clock(), placed(PACKED)), "pack_right")
    early = calibrated.check(observation("closed_early", clock(), lid_closed=True), None)

    assert early.outcome is Outcome.FAIL
    assert early.result(CLOSURE_CONDITION_ID).instruction == "Reopen the box so the contents can be inspected."
    assert not calibrated.completed

    calibrated.check(observation("inspect", clock(), placed(PACKED)), "inspect_contents")
    closed = calibrated.check(observation("closed", clock(), lid_closed=True), None)
    assert closed.outcome is Outcome.PASS
    assert calibrated.completed


def test_item_removed_after_its_step_passed_is_caught(calibrated, clock):
    calibrated.check(observation("left", clock(), placed(LEFT_SLOTS)), "pack_left")
    judgment = calibrated.check(observation("right", clock(), placed(without("notebook"))), "pack_right")

    assert judgment.outcome is Outcome.FAIL
    assert judgment.result("notebook_left").outcome is Outcome.FAIL
    assert calibrated.invariants["notebook_left"].outcome is Outcome.FAIL
    assert "pack_right" not in calibrated.passed_steps


def test_shifted_divider_requests_recalibration(calibrated, clock):
    calibrated.check(observation("left", clock(), placed(LEFT_SLOTS)), "pack_left")
    x, y, w, h = DIVIDER
    judgment = calibrated.check(observation("right", clock(), placed(PACKED), divider=(x + 30, y, w, h)), "pack_right")

    assert judgment.outcome is Outcome.INSUFFICIENT
    assert judgment.result("divider_placed").reasons[0] == "Divider moved from its calibrated position"
    assert judgment.result("mouse_right").outcome is Outcome.INSUFFICIENT


def test_step_passes_with_the_lid_set_beside_the_box(calibrated, clock):
    obs = observation("left", clock(), placed(LEFT_SLOTS))
    beside = Detection("lid", 0.9, rect_mask(355, 10, 40, 40))
    judgment = calibrated.check(replace(obs, detections=obs.detections + (beside,)), "pack_left")

    assert judgment.outcome is Outcome.PASS
    assert "pack_left" in calibrated.passed_steps


def test_correction_counts_only_with_fresh_capture(calibrated, clock):
    calibrated.check(observation("left", clock(), placed(LEFT_SLOTS)), "pack_left")
    failed = calibrated.check(observation("right", clock(), placed(without("cable"))), "pack_right")
    assert failed.outcome is Outcome.FAIL

    with pytest.raises(ValueError, match="already judged"):
        calibrated.check(observation("right", clock(), placed(PACKED)), "pack_right")

    fixed = calibrated.check(observation("right_fixed", clock(), placed(PACKED)), "pack_right")
    assert fixed.outcome is Outcome.PASS
    assert calibrated.passed_steps["pack_right"] == "right_fixed"


def test_capture_order_does_not_depend_on_wall_clock(calibrated, clock):
    later = clock()
    earlier = later - timedelta(minutes=5)
    calibrated.check(observation("left", later, placed(LEFT_SLOTS)), "pack_left")

    judgment = calibrated.check(observation("right", earlier, placed(PACKED)), "pack_right")

    assert judgment.outcome is Outcome.PASS
    assert [j.capture_id for j in calibrated.judgments][-2:] == ["left", "right"]


def test_visible_change_after_inspection_revokes_closure(inspected, clock):
    assert inspected.ready_to_close

    recheck = inspected.check(observation("recheck", clock(), placed(without("mouse"))), None)
    assert recheck.outcome is Outcome.FAIL
    assert not inspected.ready_to_close
    assert "inspect_contents" in {s.id for s in inspected.available_steps()}

    closed = inspected.check(observation("closed", clock(), lid_closed=True), None)
    assert closed.outcome is Outcome.FAIL
    assert not inspected.completed


def test_obscured_view_after_inspection_keeps_closure_allowed(inspected, clock):
    obscured = inspected.check(observation("hand", clock(), visibility=Visibility.NONE), None)

    assert obscured.outcome is Outcome.INSUFFICIENT
    assert inspected.ready_to_close
    assert not inspected.completed


def test_lid_seen_by_only_one_source_is_not_closure(inspected, clock):
    judgment = inspected.check(observation("ambiguous", clock(), placed(PACKED), lid_detected=True), None)

    assert judgment.result(CLOSURE_CONDITION_ID).outcome is Outcome.INSUFFICIENT
    assert not inspected.completed


def test_recalibration_requires_a_new_inspection(inspected, clock):
    reference = observation("reference2", clock())
    inspected.set_calibration(calibrate(reference, ZONES, revision=2, approved_by="author", approved_at=clock()))

    assert not inspected.ready_to_close
    closed = inspected.check(observation("closed", clock(), lid_closed=True), None)
    assert closed.outcome is Outcome.FAIL


def test_recalibration_needs_higher_revision(calibrated, clock):
    reference = observation("reference2", clock())
    with pytest.raises(ValueError, match="higher revision"):
        calibrated.set_calibration(calibrate(reference, ZONES, revision=1, approved_by="author", approved_at=clock()))


def test_override_is_recorded_and_unblocks_the_step(calibrated, clock):
    calibrated.check(observation("left", clock(), placed(LEFT_SLOTS)), "pack_left")
    calibrated.check(observation("right", clock(), placed(without("cable"))), "pack_right")

    judgment = calibrated.override("cable_right", Outcome.PASS, "author", "Cable is under the mouse", clock())

    cable = judgment.result("cable_right")
    assert judgment.outcome is Outcome.PASS
    assert cable.override is not None
    assert cable.override.original is Outcome.FAIL
    assert calibrated.passed_steps["pack_right"] == "right"


def test_override_shows_in_the_completion_record(calibrated, clock):
    calibrated.check(observation("left", clock(), placed(LEFT_SLOTS)), "pack_left")
    calibrated.check(observation("right", clock(), placed(without("cable"))), "pack_right")
    calibrated.override("cable_right", Outcome.PASS, "author", "Cable is under the mouse", clock())
    calibrated.check(observation("inspect", clock(), placed(PACKED)), "inspect_contents")
    calibrated.check(observation("closed", clock(), lid_closed=True), None)

    assert calibrated.record().overridden == (("right", "cable_right"),)


def test_override_to_fail_reverts_a_passed_step(calibrated, clock):
    calibrated.check(observation("left", clock(), placed(LEFT_SLOTS)), "pack_left")

    judgment = calibrated.override("marker_left", Outcome.FAIL, "author", "That is a pen", clock())

    assert judgment.outcome is Outcome.FAIL
    assert judgment.result("marker_left").instruction == "That is a pen"
    assert "pack_left" not in calibrated.passed_steps


def test_closure_cannot_be_overridden(calibrated, clock):
    calibrated.check(observation("closed_early", clock(), lid_closed=True), None)
    with pytest.raises(ValueError, match="closure"):
        calibrated.override(CLOSURE_CONDITION_ID, Outcome.PASS, "author", "trust me", clock())


def test_override_to_the_same_outcome_is_rejected(calibrated, clock):
    calibrated.check(observation("left", clock(), placed(LEFT_SLOTS)), "pack_left")
    with pytest.raises(ValueError, match="already pass"):
        calibrated.override("marker_left", Outcome.PASS, "author", "fine", clock())


def test_completed_session_accepts_no_more_captures(inspected, clock):
    inspected.check(observation("closed", clock(), lid_closed=True), None)
    with pytest.raises(ValueError, match="completed"):
        inspected.check(observation("after", clock(), placed(PACKED)), None)


def test_judgment_keeps_the_capture_digest(calibrated, clock):
    capture = observation("left", clock(), placed(LEFT_SLOTS))

    judgment = calibrated.check(capture, "pack_left")

    assert judgment.image_sha256 == capture.image_sha256


def test_same_image_with_a_new_id_and_time_is_rejected(calibrated, clock):
    capture = observation("left", clock(), placed(LEFT_SLOTS))
    calibrated.check(capture, "pack_left")
    replay = replace(capture, capture_id="replayed", captured_at=clock())

    with pytest.raises(ValueError, match="already judged"):
        calibrated.check(replay, None)


@pytest.mark.parametrize("scene", [{"lid_closed": True}, {"lid_detected": True}])
def test_invalid_target_is_rejected_even_when_the_box_is_not_open(inspected, clock, scene):
    with pytest.raises(ValueError, match="not available"):
        inspected.check(observation("bad_target", clock(), **scene), "missing_step")


def test_unavailable_target_is_rejected_on_a_closed_box(session, clock):
    with pytest.raises(ValueError, match="not available"):
        session.check(observation("too_early", clock(), lid_closed=True), "pack_left")


def test_a_condition_cannot_be_overridden_twice_on_one_capture(calibrated, clock):
    calibrated.check(observation("left", clock(), placed(LEFT_SLOTS)), "pack_left")
    calibrated.override("marker_left", Outcome.FAIL, "author", "That is a pen", clock())

    with pytest.raises(ValueError, match="already overridden"):
        calibrated.override("marker_left", Outcome.PASS, "author", "Changed my mind", clock())

    marker = calibrated.judgments[-1].result("marker_left")
    assert marker.outcome is Outcome.FAIL
    assert marker.override.original is Outcome.PASS
    assert marker.override.reason == "That is a pen"


def test_different_conditions_can_be_overridden_on_one_capture(calibrated, clock):
    calibrated.check(observation("right", clock(), placed(LEFT_SLOTS)), "pack_right")
    calibrated.override("mouse_right", Outcome.PASS, "author", "Mouse is covered", clock())

    judgment = calibrated.override("cable_right", Outcome.PASS, "author", "Cable is covered", clock())

    assert judgment.outcome is Outcome.PASS
    assert judgment.result("mouse_right").override.reason == "Mouse is covered"
    assert judgment.result("cable_right").override.reason == "Cable is covered"
    assert calibrated.passed_steps["pack_right"] == "right"


def test_completed_session_rejects_overrides(inspected, clock):
    inspected.check(observation("closed", clock(), lid_closed=True), None)

    with pytest.raises(ValueError, match="completed"):
        inspected.override("marker_left", Outcome.FAIL, "author", "Wrong marker", clock())


def test_session_procedure_cannot_be_reassigned(session, procedure, clock):
    replacement = procedure.approve("different_author", clock())

    with pytest.raises(AttributeError):
        session.procedure = replacement

    assert session.procedure is procedure


def test_session_calibration_cannot_bypass_recalibration(calibrated, clock):
    original = calibrated.calibration
    replacement = calibrate(observation("new_reference", clock()), ZONES, 2, "author", clock())

    with pytest.raises(AttributeError):
        calibrated.calibration = replacement

    assert calibrated.calibration is original


@pytest.mark.parametrize("changed", ["box", "divider"])
def test_registration_drift_revokes_the_final_inspection(inspected, clock, changed):
    rect = (80, 50, 300, 200) if changed == "box" else (225, 50, 10, 200)
    drifted = observation("drifted", clock(), placed(PACKED), **{changed: rect})

    judgment = inspected.check(drifted, None)

    assert judgment.outcome is Outcome.INSUFFICIENT
    assert not inspected.ready_to_close
    assert "inspect_contents" in {s.id for s in inspected.available_steps()}
    closed = inspected.check(observation("closed", clock(), lid_closed=True), None)
    assert closed.outcome is Outcome.FAIL
    assert not inspected.completed


def test_inspection_after_realignment_restores_permission_to_close(inspected, clock):
    inspected.check(observation("drifted", clock(), placed(PACKED), divider=(225, 50, 10, 200)), None)

    judgment = inspected.check(observation("inspect_again", clock(), placed(PACKED)), "inspect_contents")

    assert judgment.outcome is Outcome.PASS
    assert inspected.ready_to_close
    inspected.check(observation("closed", clock(), lid_closed=True), None)
    assert inspected.record().inspection_capture_id == "inspect_again"


def test_overriding_registration_does_not_restore_the_old_inspection(inspected, clock):
    inspected.check(observation("drifted", clock(), placed(PACKED), divider=(225, 50, 10, 200)), None)

    inspected.override("divider_placed", Outcome.PASS, "author", "I checked the divider", clock())

    assert not inspected.ready_to_close
    closed = inspected.check(observation("closed", clock(), lid_closed=True), None)
    assert closed.outcome is Outcome.FAIL


def test_photo_taken_before_the_inspection_cannot_close_the_box(calibrated, clock):
    calibrated.check(observation("left", clock(), placed(LEFT_SLOTS)), "pack_left")
    calibrated.check(observation("right", clock(), placed(PACKED)), "pack_right")
    stale_closed = observation("stale_closed", clock(), lid_closed=True)
    calibrated.check(observation("inspect", clock(), placed(PACKED)), "inspect_contents")

    with pytest.raises(ValueError, match="taken before"):
        calibrated.check(stale_closed, None)

    assert not calibrated.completed
    assert calibrated.ready_to_close


def test_judgment_keeps_the_acquisition_sequence(session, clock):
    obs = observation("empty", clock(), sequence=7)
    assert session.check(obs, "show_empty_box").capture_sequence == 7


def test_capture_times_are_stored_in_utc(session, clock):
    taken = clock().astimezone(timezone(timedelta(hours=-7)))
    judgment = session.check(observation("empty", taken), "show_empty_box")

    assert judgment.captured_at.utcoffset() == timedelta(0)
    assert judgment.captured_at == taken


def test_checks_on_one_session_do_not_interleave(inspected, clock, monkeypatch):
    closing = observation("closed", clock(), lid_closed=True)
    removal = observation("removed", clock(), placed(without("mouse")))
    closure_judged = threading.Event()
    release = threading.Event()
    real_closure_result = Session._closure_result

    def paused_closure_result(self, state):
        result = real_closure_result(self, state)
        closure_judged.set()
        release.wait(timeout=5)
        return result

    monkeypatch.setattr(Session, "_closure_result", paused_closure_result)
    close_thread = threading.Thread(target=inspected.check, args=(closing, None))
    close_thread.start()
    assert closure_judged.wait(timeout=5)

    errors: list[ValueError] = []

    def remove_mouse():
        try:
            inspected.check(removal, None)
        except ValueError as e:
            errors.append(e)

    # Without serialization this check would revoke the inspection while the closure is mid-flight.
    removal_thread = threading.Thread(target=remove_mouse)
    removal_thread.start()
    removal_thread.join(timeout=0.2)
    blocked = removal_thread.is_alive()

    release.set()
    close_thread.join(timeout=5)
    removal_thread.join(timeout=5)

    assert blocked
    assert [str(e) for e in errors] == ["the session is already completed"]
    assert inspected.completed
    assert inspected.record().closure_capture_id == "closed"
