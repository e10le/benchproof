"""Deterministic checks over one capture's perception output.

Perception (SAM masks plus a whole-scene model assessment) arrives as a CaptureObservation. The
functions here turn it into condition results without calling any model, so the rules for pass,
fail and insufficient are testable offline and identical across providers.
"""

import re
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

import numpy as np
from pydantic import AwareDatetime

from benchproof import geometry
from benchproof.models import (
    IMAGE_SHA256_PATTERN,
    Condition,
    ConditionKind,
    ConditionResult,
    Item,
    Outcome,
    Procedure,
    SceneAssessment,
    Visibility,
    Zone,
    ZoneCalibration,
)

# Starting thresholds; tune them on development captures and freeze them before held-out evaluation.
ZONE_FRACTION = 0.8
INSIDE_BOX_FRACTION = 0.5
DUPLICATE_IOU = 0.5
LID_COVERAGE_FRACTION = 0.8

CLEAR_VIEW = "Clear the view so the whole box interior is visible, then take a new photo."
REALIGN = "Put the box and divider back in the calibrated position, or recalibrate the zones."
RETAKE = "Take a new photo from the fixed camera; the two checks disagree."
SHOW_BOX = "Place the open box fully inside the camera view, then take a new photo."


# eq=False: comparing masks element-wise has no single truth value, so detections compare by identity.
@dataclass(frozen=True, eq=False)
class Detection:
    label: str
    score: float
    mask: np.ndarray


@dataclass(frozen=True)
class CaptureObservation:
    capture_id: str
    # Assigned by the capture source when the photo is taken, before any model call. It orders
    # evidence because wall-clock times can jump or differ between devices.
    sequence: int
    image_sha256: str
    captured_at: datetime
    width: int
    height: int
    scene: SceneAssessment
    detections: tuple[Detection, ...]
    source: str

    def __post_init__(self) -> None:
        if self.sequence < 0:
            raise ValueError("sequence must be non-negative")
        if re.fullmatch(IMAGE_SHA256_PATTERN, self.image_sha256) is None:
            raise ValueError("image_sha256 must be a lowercase SHA-256 digest")
        if self.captured_at.tzinfo is None:
            raise ValueError("captured_at must be timezone-aware")

        for d in self.detections:
            if d.mask.shape != (self.height, self.width) or d.mask.dtype != bool:
                raise ValueError(f"{d.label}: mask must be a bool array of shape ({self.height}, {self.width})")
            if not d.mask.any():
                raise ValueError(f"{d.label}: empty mask")

    def largest(self, label: str) -> Detection | None:
        found = [d for d in self.detections if d.label == label]
        if not found:
            return None
        return max(found, key=lambda d: np.count_nonzero(d.mask))


class ClosureState(StrEnum):
    OPEN = "open"
    CLOSED = "closed"
    UNCLEAR = "unclear"


def closure_state(obs: CaptureObservation) -> ClosureState:
    # Both sources must agree either way. Only a lid over the box counts: an open box's flaps, or a
    # lid set beside it, must not block every check.
    lid = obs.largest("lid")
    box = obs.largest("box")
    lid_covers_box = (
        lid is not None
        and box is not None
        and geometry.overlap_fraction(geometry.filled_hull(box.mask), lid.mask) >= LID_COVERAGE_FRACTION
    )
    if obs.scene.lid_closed and lid_covers_box:
        return ClosureState.CLOSED
    if not obs.scene.lid_closed and not lid_covers_box:
        return ClosureState.OPEN
    return ClosureState.UNCLEAR


def calibrate(
    obs: CaptureObservation,
    zones: tuple[Zone, ...],
    revision: int,
    approved_by: str,
    approved_at: AwareDatetime,
) -> ZoneCalibration:
    box = obs.largest("box")
    divider = obs.largest("divider")
    if box is None or divider is None:
        raise ValueError("calibration needs both the box and the divider in the reference capture")

    bbox, angle, aspect_ratio = geometry.box_pose(box.mask)
    divider_bbox, divider_angle, divider_aspect_ratio = geometry.box_pose(divider.mask)
    box_region = geometry.filled_hull(box.mask)
    for zone in zones:
        region = geometry.polygon_mask(zone.polygon, obs.width, obs.height)
        if not region.any() or np.any(region & ~box_region):
            raise ValueError(f"zone '{zone.id}' must be inside the visible box")
    return ZoneCalibration(
        revision=revision,
        reference_capture_id=obs.capture_id,
        image_width=obs.width,
        image_height=obs.height,
        box=bbox,
        box_angle_deg=angle,
        box_aspect_ratio=aspect_ratio,
        divider_center=geometry.mask_center(divider.mask),
        divider=divider_bbox,
        divider_angle_deg=divider_angle,
        divider_aspect_ratio=divider_aspect_ratio,
        zones=zones,
        approved_by=approved_by,
        approved_at=approved_at,
    )


def check_registration(calibration: ZoneCalibration | None, obs: CaptureObservation) -> str | None:
    """Return why this capture cannot be read with the approved zones, or None if it can."""
    if calibration is None:
        return "Zones are not calibrated yet"
    if (obs.width, obs.height) != (calibration.image_width, calibration.image_height):
        return "Capture resolution differs from the calibration capture"

    box = obs.largest("box")
    if box is None:
        return "Box not found"

    ref = calibration.box
    bbox, angle, aspect_ratio = geometry.box_pose(box.mask)
    shift = np.hypot(bbox.center[0] - ref.center[0], bbox.center[1] - ref.center[1]) / ref.diagonal
    if shift > calibration.max_box_shift:
        return "Box moved from its calibrated position"
    if max(abs(bbox.w / ref.w - 1.0), abs(bbox.h / ref.h - 1.0)) > calibration.max_box_scale_change:
        return "Box size changed since calibration; the camera may have moved"
    if (
        geometry.angle_difference(angle, calibration.box_angle_deg, aspect_ratio, calibration.box_aspect_ratio)
        > calibration.max_angle_deg
    ):
        return "Box rotated since calibration"

    divider = obs.largest("divider")
    if divider is None:
        return "Divider not found"
    dx, dy = geometry.mask_center(divider.mask)
    ref_x, ref_y = calibration.divider_center
    if np.hypot(dx - ref_x, dy - ref_y) / ref.diagonal > calibration.max_divider_shift:
        return "Divider moved from its calibrated position"
    divider_bbox, divider_angle, divider_aspect_ratio = geometry.box_pose(divider.mask)
    ref_divider = calibration.divider
    if (
        geometry.angle_difference(
            divider_angle, calibration.divider_angle_deg, divider_aspect_ratio, calibration.divider_aspect_ratio
        )
        > calibration.max_angle_deg
    ):
        return "Divider rotated since calibration"
    divider_scale = max(abs(divider_bbox.w / ref_divider.w - 1.0), abs(divider_bbox.h / ref_divider.h - 1.0))
    if divider_scale > calibration.max_divider_scale_change:
        return "Divider size changed since calibration"

    return None


@dataclass(frozen=True)
class _Scene:
    obs: CaptureObservation
    box_region: np.ndarray | None
    items: dict[str, tuple[Detection, ...]]
    registration_problem: str | None
    zone_regions: dict[str, np.ndarray]

    def inside_box(self, item_id: str) -> tuple[Detection, ...]:
        if self.box_region is None:
            return ()
        return tuple(
            d for d in self.items[item_id] if geometry.overlap_fraction(d.mask, self.box_region) >= INSIDE_BOX_FRACTION
        )


def evaluate(
    procedure: Procedure,
    calibration: ZoneCalibration | None,
    obs: CaptureObservation,
    condition_ids: list[str],
) -> list[ConditionResult]:
    check_assessment(procedure, obs.scene)

    box = obs.largest("box")
    registration_problem = check_registration(calibration, obs)
    zone_regions = {}
    if registration_problem is None:
        zone_regions = {z.id: geometry.polygon_mask(z.polygon, obs.width, obs.height) for z in calibration.zones}

    scene = _Scene(
        obs=obs,
        box_region=geometry.filled_hull(box.mask) if box is not None else None,
        items={i.id: _dedupe([d for d in obs.detections if d.label == i.id]) for i in procedure.items},
        registration_problem=registration_problem,
        zone_regions=zone_regions,
    )

    results = []
    for cid in condition_ids:
        condition = procedure.condition(cid)
        match condition.kind:
            case ConditionKind.BOX_EMPTY:
                results.append(_box_empty(condition, scene))
            case ConditionKind.DIVIDER_IN_PLACE:
                results.append(_divider_in_place(condition, scene))
            case ConditionKind.ITEM_IN_ZONE:
                results.append(_item_in_zone(condition, procedure.item(condition.item), scene))
            case ConditionKind.INVENTORY_EXACT:
                results.append(_inventory_exact(condition, procedure.items, scene))
    return results


def check_assessment(procedure: Procedure, scene: SceneAssessment) -> None:
    expected = {i.id for i in procedure.items}
    reported = set(scene.item_counts)
    if reported != expected:
        raise ValueError(f"scene assessment must count exactly {sorted(expected)}, got {sorted(reported)}")


def _dedupe(detections: list[Detection]) -> tuple[Detection, ...]:
    # SAM can return overlapping masks for one object; keep the strongest of each overlapping group.
    kept: list[Detection] = []
    for d in sorted(detections, key=lambda d: d.score, reverse=True):
        if all(geometry.iou(d.mask, k.mask) < DUPLICATE_IOU for k in kept):
            kept.append(d)
    return tuple(kept)


def _result(condition: Condition, outcome: Outcome, reason: str, instruction: str | None = None) -> ConditionResult:
    if outcome is Outcome.FAIL:
        instruction = condition.recovery
    return ConditionResult(condition_id=condition.id, outcome=outcome, reasons=(reason,), instruction=instruction)


def _box_empty(condition: Condition, scene: _Scene) -> ConditionResult:
    if scene.box_region is None:
        return _result(condition, Outcome.INSUFFICIENT, "Box not found", SHOW_BOX)

    assessment = scene.obs.scene
    if assessment.interior_visibility is not Visibility.CLEAR:
        return _result(condition, Outcome.INSUFFICIENT, "Box interior is not fully visible", CLEAR_VIEW)

    # Absence needs both witnesses: no masks inside the box and a model reading of an empty interior.
    detected = [item_id for item_id in scene.items if scene.inside_box(item_id)]
    reported = [item_id for item_id, n in assessment.item_counts.items() if n > 0]
    reported += list(assessment.unexpected_items)

    if detected and reported:
        return _result(condition, Outcome.FAIL, f"Box is not empty: {', '.join(sorted(set(detected + reported)))}")
    if detected or reported:
        reason = "Detection and scene assessment disagree about whether the box is empty"
        return _result(condition, Outcome.INSUFFICIENT, reason, RETAKE)
    return _result(condition, Outcome.PASS, "Box interior is visible and empty")


def _divider_in_place(condition: Condition, scene: _Scene) -> ConditionResult:
    if scene.registration_problem is not None:
        return _result(condition, Outcome.INSUFFICIENT, scene.registration_problem, REALIGN)
    return _result(condition, Outcome.PASS, "Box and divider match the calibrated position")


def _item_in_zone(condition: Condition, item: Item, scene: _Scene) -> ConditionResult:
    if scene.registration_problem is not None:
        return _result(condition, Outcome.INSUFFICIENT, scene.registration_problem, REALIGN)

    assessment = scene.obs.scene
    if assessment.interior_visibility is Visibility.NONE:
        return _result(condition, Outcome.INSUFFICIENT, "Box interior is not visible", CLEAR_VIEW)

    detections = scene.items[item.id]
    reported = assessment.item_counts[item.id]
    target = scene.zone_regions[condition.zone]
    disagreement = f"Detection and scene assessment disagree about the {item.name}"

    if any(geometry.overlap_fraction(d.mask, target) >= ZONE_FRACTION for d in detections):
        if reported == 0:
            return _result(condition, Outcome.INSUFFICIENT, disagreement, RETAKE)
        return _result(condition, Outcome.PASS, f"{item.name} is inside the {condition.zone} zone")

    if detections:
        return _result(condition, Outcome.FAIL, _misplacement(item, condition.zone, detections, scene))

    if assessment.interior_visibility is Visibility.PARTIAL:
        return _result(condition, Outcome.INSUFFICIENT, f"{item.name} is not visible; it may be covered", CLEAR_VIEW)
    if reported > 0:
        return _result(condition, Outcome.INSUFFICIENT, disagreement, RETAKE)
    return _result(condition, Outcome.FAIL, f"{item.name} was not found in the box")


def _misplacement(item: Item, zone_id: str, detections: tuple[Detection, ...], scene: _Scene) -> str:
    largest = max(detections, key=lambda d: np.count_nonzero(d.mask))

    for other, region in scene.zone_regions.items():
        if other != zone_id and geometry.overlap_fraction(largest.mask, region) >= ZONE_FRACTION:
            return f"{item.name} is in the {other} zone, not the {zone_id} zone"

    if geometry.overlap_fraction(largest.mask, scene.box_region) >= INSIDE_BOX_FRACTION:
        return f"{item.name} is not fully inside the {zone_id} zone"
    return f"{item.name} is outside the box"


def _inventory_exact(condition: Condition, items: tuple[Item, ...], scene: _Scene) -> ConditionResult:
    if scene.box_region is None:
        return _result(condition, Outcome.INSUFFICIENT, "Box not found", SHOW_BOX)

    assessment = scene.obs.scene
    if assessment.interior_visibility is not Visibility.CLEAR:
        return _result(condition, Outcome.INSUFFICIENT, "Box interior is not fully visible", CLEAR_VIEW)

    problems: list[str] = []
    disagreements: list[str] = []
    for item in items:
        detected = len(scene.inside_box(item.id))
        reported = assessment.item_counts[item.id]
        if detected != reported:
            disagreements.append(item.name)
        elif detected > 1:
            problems.append(f"{detected} x {item.name} (expected 1)")
        elif detected == 0:
            problems.append(f"{item.name} missing")

    # SAM is only prompted for kit items, so an unexpected object can only come from the scene model.
    # It still fails: a false alarm costs a retake, a missed extra item costs a wrong record.
    problems += [f"unexpected {name}" for name in assessment.unexpected_items]

    if problems:
        return _result(condition, Outcome.FAIL, f"Contents do not match the kit: {'; '.join(problems)}")
    if disagreements:
        return _result(
            condition,
            Outcome.INSUFFICIENT,
            f"Detection and scene assessment disagree about: {', '.join(disagreements)}",
            RETAKE,
        )
    return _result(condition, Outcome.PASS, "Box holds exactly the kit items")
