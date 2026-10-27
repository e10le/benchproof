import itertools
from datetime import timedelta

import pytest

from benchproof.models import Procedure
from benchproof.predicates import calibrate
from benchproof.session import Session
from scenes import LEFT_SLOTS, PACKED, PROCEDURE_PATH, T0, ZONES, observation, placed


@pytest.fixture
def procedure() -> Procedure:
    return Procedure.model_validate_json(PROCEDURE_PATH.read_text(encoding="utf-8")).approve("author", T0)


@pytest.fixture
def clock():
    ticks = itertools.count(1)
    return lambda: T0 + timedelta(seconds=next(ticks))


@pytest.fixture
def session(procedure: Procedure) -> Session:
    return Session(procedure)


@pytest.fixture
def calibrated(session: Session, clock) -> Session:
    """A session that has passed the empty-box and divider steps."""
    session.check(observation("empty", clock()), "show_empty_box")

    reference = observation("reference", clock())
    session.set_calibration(calibrate(reference, ZONES, revision=1, approved_by="author", approved_at=clock()))
    session.check(observation("divider", clock()), "place_divider")
    return session


@pytest.fixture
def inspected(calibrated: Session, clock) -> Session:
    """A session whose final inspection has passed and is ready to close."""
    calibrated.check(observation("left", clock(), placed(LEFT_SLOTS)), "pack_left")
    calibrated.check(observation("right", clock(), placed(PACKED)), "pack_right")
    calibrated.check(observation("inspect", clock(), placed(PACKED)), "inspect_contents")
    return calibrated
