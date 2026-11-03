"""Everything the check page does, without the page: capture, perceive, calibrate and judge."""

import itertools
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
from pydantic import AwareDatetime

from benchproof import geometry
from benchproof.history import RunLog
from benchproof.llm import SPARK_OPEN_MODEL, CallBudget, baseten_client, meta_client
from benchproof.models import Judgment, Procedure, Zone, ZoneCalibration
from benchproof.perceive import perceive
from benchproof.photos import Photo, prepare
from benchproof.predicates import CaptureObservation, calibrate
from benchproof.scene import SceneJudge
from benchproof.segmenter import Segmenter
from benchproof.session import Session

# A photo costs 8 calls (7 SAM + 1 judge) plus retries; this caps one session at about 25 photos.
SESSION_CALL_LIMIT = 200
# Proposed zones stay this share of the box diagonal away from walls and divider.
ZONE_INSET = 0.02


@dataclass(frozen=True)
class Capture:
    photo: Photo
    observation: CaptureObservation


class Checker:
    def __init__(self, session: Session, segmenter: Segmenter, judge: SceneJudge, budget: CallBudget) -> None:
        self.session = session
        self.budget = budget
        self._segmenter = segmenter
        self._judge = judge
        self._sequence = itertools.count()

    def capture(self, image_bytes: bytes) -> Capture:
        photo = prepare(image_bytes)
        # The sequence is fixed when the photo arrives, before the slow model calls.
        sequence = next(self._sequence)
        captured_at = datetime.now(UTC)
        observation = perceive(
            self.session.procedure,
            photo,
            capture_id=f"{captured_at:%Y%m%dT%H%M%S%f}-{sequence}",
            sequence=sequence,
            captured_at=captured_at,
            segmenter=self._segmenter,
            judge=self._judge,
        )
        return Capture(photo=photo, observation=observation)

    def check(self, capture: Capture, target_step: str | None) -> Judgment:
        return self.session.check(capture.observation, target_step)

    def calibrate(
        self, capture: Capture, zones: tuple[Zone, ...], approved_by: str, at: AwareDatetime
    ) -> ZoneCalibration:
        current = self.session.calibration
        revision = 1 if current is None else current.revision + 1
        calibration = calibrate(capture.observation, zones, revision, approved_by, at)
        self.session.set_calibration(calibration)
        return calibration


def start(procedure: Procedure, approved_by: str, runs_dir: Path) -> Checker:
    # Clients are built first so a missing key fails before a run log is created.
    segmenter_client, judge_client = meta_client(), baseten_client()
    budget = CallBudget(SESSION_CALL_LIMIT)
    session = Session(procedure.approve(approved_by, datetime.now(UTC)), RunLog(runs_dir))
    return Checker(
        session,
        Segmenter(segmenter_client, budget),
        SceneJudge(judge_client, SPARK_OPEN_MODEL, budget),
        budget,
    )


def propose_zones(observation: CaptureObservation, zone_ids: tuple[str, ...]) -> tuple[Zone, ...]:
    """Split the box at the divider into two rectangles for the author to adjust and approve."""
    if len(zone_ids) != 2:
        raise ValueError("automatic zones need a procedure with exactly two zones")

    box, divider = observation.largest("box"), observation.largest("divider")
    if box is None or divider is None:
        raise ValueError("the photo must show both the box and the divider")

    b, _, _ = geometry.box_pose(box.mask)
    d, _, _ = geometry.box_pose(divider.mask)
    inset = ZONE_INSET * b.diagonal

    if d.w >= d.h:
        # The divider runs across the photo, so the zones lie above and below it.
        left, right = d.x + inset, d.x + d.w - inset
        first = (left, b.y + inset, right, d.y - inset)
        second = (left, d.y + d.h + inset, right, b.y + b.h - inset)
    else:
        top, bottom = d.y + inset, d.y + d.h - inset
        first = (b.x + inset, top, d.x - inset, bottom)
        second = (d.x + d.w + inset, top, b.x + b.w - inset, bottom)

    for x0, y0, x1, y1 in (first, second):
        if x1 <= x0 or y1 <= y0:
            raise ValueError("the divider is too close to a box wall to split the box into two zones")

    rectangles = tuple(
        Zone(id=zone_id, polygon=rectangle(*corners))
        for zone_id, corners in zip(zone_ids, (first, second), strict=True)
    )
    return fit_to_box(observation, rectangles)


def fit_to_box(observation: CaptureObservation, zones: tuple[Zone, ...]) -> tuple[Zone, ...]:
    """Trim each zone to the box outline, kept a small margin inside it.

    A camera looking into the box at an angle sees a trapezoid, so a rectangle drawn from the box's
    bounding box has corners outside the box, and calibration rejects it.
    """
    box = observation.largest("box")
    if box is None:
        raise ValueError("the photo must show the box")

    b, _, _ = geometry.box_pose(box.mask)
    margin = max(1, round(ZONE_INSET * b.diagonal))
    kernel = np.ones((2 * margin + 1, 2 * margin + 1), dtype=np.uint8)
    inner = cv2.erode(geometry.filled_hull(box.mask).astype(np.uint8), kernel)
    points = cv2.findNonZero(inner)
    if points is None:
        raise ValueError("the box is too small in the photo to place zones; move the camera closer")
    outline = cv2.convexHull(points).reshape(-1, 2).astype(np.float32)

    fitted = []
    for zone in zones:
        area, polygon = cv2.intersectConvexConvex(np.array(zone.polygon, dtype=np.float32), outline)
        if polygon is None or area < 1:
            raise ValueError(f"zone '{zone.id}' lies outside the box; move it back over the box")
        fitted.append(Zone(id=zone.id, polygon=tuple((float(x), float(y)) for x, y in polygon.reshape(-1, 2))))
    return tuple(fitted)


def rectangle(x0: float, y0: float, x1: float, y1: float) -> tuple[tuple[float, float], ...]:
    return ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
