import threading
from pathlib import Path

import pytest
from pydantic import ValidationError

from benchproof import session as session_module
from benchproof.history import Event, EventKind, RunLog, load_run_log
from benchproof.models import Outcome
from benchproof.predicates import calibrate
from benchproof.session import Session
from scenes import LEFT_SLOTS, ZONES, observation, placed, without


def kinds(session: Session) -> list[EventKind]:
    return [record.kind for record in load_run_log(session.log_path)]


def test_full_run_is_recorded_in_order(inspected, clock):
    inspected.check(observation("closed", clock(), lid_closed=True), None)

    records = load_run_log(inspected.log_path)
    assert [r.kind for r in records] == [
        EventKind.SESSION_STARTED,
        EventKind.CHECK_REQUESTED,
        EventKind.CHECK_JUDGED,
        EventKind.CALIBRATION_SET,
        *[EventKind.CHECK_REQUESTED, EventKind.CHECK_JUDGED] * 5,
        EventKind.SESSION_COMPLETED,
    ]
    assert records[0].procedure == inspected.procedure
    assert records[-2].judgment == inspected.judgments[-1]
    assert records[-1].capture_id == "closed"


def test_rejected_check_is_recorded_without_changing_state(calibrated, clock):
    calibrated.check(observation("left", clock(), placed(LEFT_SLOTS)), "pack_left")
    passed = calibrated.passed_steps

    with pytest.raises(ValueError, match="already judged"):
        calibrated.check(observation("left", clock(), placed(LEFT_SLOTS)), "pack_right")

    rejected = load_run_log(calibrated.log_path)[-1]
    assert rejected.kind is EventKind.CHECK_REJECTED
    assert rejected.capture_id == "left"
    assert "already judged" in rejected.reason
    assert calibrated.passed_steps == passed


def test_check_that_crashes_stays_visibly_unfinished(calibrated, clock, monkeypatch):
    def crash(*args):
        raise RuntimeError("perception bug")

    monkeypatch.setattr(session_module, "evaluate", crash)
    with pytest.raises(RuntimeError):
        calibrated.check(observation("left", clock(), placed(LEFT_SLOTS)), "pack_left")

    last = load_run_log(calibrated.log_path)[-1]
    assert last.kind is EventKind.CHECK_REQUESTED
    assert last.capture_id == "left"


def test_failed_log_write_leaves_state_unchanged_and_stops_the_session(calibrated, clock, monkeypatch):
    original_open = Path.open
    writes = []

    def fail_second_write(path, mode="r", *args, **kwargs):
        if path == calibrated.log_path and "a" in mode:
            writes.append(mode)
            if len(writes) == 2:
                raise OSError("disk full")
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_second_write)
    with pytest.raises(OSError, match="disk full"):
        calibrated.check(observation("left", clock(), placed(LEFT_SLOTS)), "pack_left")

    assert "pack_left" not in calibrated.passed_steps
    assert calibrated.judgments[-1].capture_id == "divider"
    with pytest.raises(ValueError, match="earlier run log write failed"):
        calibrated.check(observation("left_again", clock(), placed(LEFT_SLOTS)), "pack_left")


def test_revocation_is_recorded_with_its_reason(inspected, clock):
    inspected.check(observation("removed", clock(), placed(without("mouse"))), None)

    judged, revoked = load_run_log(inspected.log_path)[-2:]
    assert judged.kind is EventKind.CHECK_JUDGED
    assert revoked.kind is EventKind.INSPECTION_REVOKED
    assert "mouse_right" in revoked.reason


def test_recalibration_records_the_calibration_and_the_revocation(inspected, clock):
    calibration = calibrate(observation("reference2", clock()), ZONES, revision=2, approved_by="a", approved_at=clock())
    inspected.set_calibration(calibration)

    calibrated, revoked = load_run_log(inspected.log_path)[-2:]
    assert calibrated.kind is EventKind.CALIBRATION_SET
    assert calibrated.calibration == calibration
    assert revoked.kind is EventKind.INSPECTION_REVOKED
    assert revoked.reason == "Zones were recalibrated"


def test_rejected_calibration_is_recorded(calibrated, clock):
    stale = calibrate(observation("reference2", clock()), ZONES, revision=1, approved_by="a", approved_at=clock())
    with pytest.raises(ValueError, match="higher revision"):
        calibrated.set_calibration(stale)

    last = load_run_log(calibrated.log_path)[-1]
    assert last.kind is EventKind.CALIBRATION_REJECTED
    assert calibrated.calibration.revision == 1


def test_override_is_a_new_event_that_keeps_the_original_judgment(calibrated, clock):
    calibrated.check(observation("left", clock(), placed(LEFT_SLOTS)), "pack_left")
    calibrated.check(observation("right", clock(), placed(without("cable"))), "pack_right")
    calibrated.override("cable_right", Outcome.PASS, "author", "Cable is under the mouse", clock())

    original, overridden = load_run_log(calibrated.log_path)[-2:]
    assert original.kind is EventKind.CHECK_JUDGED
    assert original.judgment.result("cable_right").outcome is Outcome.FAIL
    assert overridden.kind is EventKind.OVERRIDE_APPLIED
    assert overridden.judgment.result("cable_right").override.original is Outcome.FAIL
    assert overridden.by == "author"


def test_rejected_override_is_recorded(calibrated, clock):
    calibrated.check(observation("closed_early", clock(), lid_closed=True), None)
    with pytest.raises(ValueError, match="closure"):
        calibrated.override("closure", Outcome.PASS, "author", "trust me", clock())

    last = load_run_log(calibrated.log_path)[-1]
    assert last.kind is EventKind.OVERRIDE_REJECTED
    assert last.by == "author"


def test_cancelled_session_records_cancel_and_rejects_checks(calibrated, clock):
    calibrated.cancel("author", "Wrong box on the mat")

    assert calibrated.cancelled
    assert calibrated.available_steps() == []
    with pytest.raises(ValueError, match="cancelled"):
        calibrated.check(observation("left", clock(), placed(LEFT_SLOTS)), "pack_left")
    with pytest.raises(ValueError, match="cancelled"):
        calibrated.cancel("author", "again")
    assert kinds(calibrated)[-3:] == [EventKind.SESSION_CANCELLED, EventKind.CHECK_REQUESTED, EventKind.CHECK_REJECTED]


def test_a_run_log_cannot_be_shared_by_two_sessions(procedure, tmp_path):
    log = RunLog(tmp_path)
    Session(procedure, log)

    with pytest.raises(ValueError, match="exactly one session_started"):
        Session(procedure, log)


def test_each_run_log_gets_its_own_file(tmp_path):
    first, second = RunLog(tmp_path), RunLog(tmp_path)
    assert first.path != second.path
    assert first.session_id != second.session_id


@pytest.mark.parametrize("fields", [{"capture_id": "c"}, {"capture_id": "c", "reason": ""}])
def test_event_missing_a_required_field_is_rejected(fields):
    with pytest.raises(ValidationError, match="needs"):
        Event(kind=EventKind.CHECK_REJECTED, **fields)


def test_cancel_needs_a_named_person_and_reason(calibrated):
    with pytest.raises(ValidationError, match="needs"):
        calibrated.cancel("", "")
    assert not calibrated.cancelled


def test_out_of_order_log_is_rejected_on_load(session, clock):
    session.check(observation("empty", clock()), "show_empty_box")
    path = session.log_path
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join([lines[0], lines[2], lines[1]]) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="out of order"):
        load_run_log(path)


def test_empty_log_loads_as_no_records(tmp_path):
    assert load_run_log(RunLog(tmp_path).path) == []


@pytest.mark.parametrize("separator", ["\u2028", "\u2029", "\x85", "\x1c"])
def test_log_with_unicode_line_separators_in_text_loads(calibrated, separator):
    calibrated.cancel("author", f"first{separator}second")

    assert load_run_log(calibrated.log_path)[-1].reason == f"first{separator}second"


def test_concurrent_writers_cannot_both_take_the_same_index(procedure, tmp_path, monkeypatch):
    log = RunLog(tmp_path)
    original_open = Path.open
    first_writer_inside = threading.Event()

    def slow_first_write(path, mode="r", *args, **kwargs):
        if path == log.path and "a" in mode and not first_writer_inside.is_set():
            first_writer_inside.set()
            # Hold the first write open long enough for the second session to reach the log.
            threading.Event().wait(0.2)
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", slow_first_write)
    outcomes: list[str] = []

    def start_session():
        try:
            Session(procedure, log)
            outcomes.append("started")
        except ValueError:
            outcomes.append("rejected")

    threads = [threading.Thread(target=start_session) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert sorted(outcomes) == ["rejected", "started"]
    assert [r.index for r in load_run_log(log.path)] == [0]


def test_override_of_an_unknown_condition_is_rejected_and_recorded(calibrated, clock):
    calibrated.check(observation("left", clock(), placed(LEFT_SLOTS)), "pack_left")

    with pytest.raises(ValueError, match="not in the latest judgment"):
        calibrated.override("ghost", Outcome.PASS, "author", "typo", clock())

    last = load_run_log(calibrated.log_path)[-1]
    assert last.kind is EventKind.OVERRIDE_REJECTED
    assert last.condition_id == "ghost"
