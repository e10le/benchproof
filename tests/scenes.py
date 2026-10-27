"""Synthetic captures with known geometry.

These stand in for perception output so the rules can be tested offline. They are not model
results and say nothing about how well real perception works.
"""

import hashlib
import itertools
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from benchproof.models import SceneAssessment, Visibility, Zone
from benchproof.predicates import CaptureObservation, Detection

WIDTH, HEIGHT = 400, 300
BOX = (50, 50, 300, 200)
DIVIDER = (195, 50, 10, 200)
ZONES = (
    Zone(id="left", polygon=((60, 60), (190, 60), (190, 240), (60, 240))),
    Zone(id="right", polygon=((210, 60), (340, 60), (340, 240), (210, 240))),
)

# Rectangles (x, y, w, h) that sit fully inside the intended compartment.
LEFT_SLOTS = {"notebook": (70, 70, 60, 80), "marker": (140, 180, 40, 20)}
RIGHT_SLOTS = {"mouse": (220, 70, 40, 50), "cable": (270, 160, 50, 50)}
PACKED = {**LEFT_SLOTS, **RIGHT_SLOTS}

PROCEDURE_PATH = Path(__file__).parent.parent / "procedures" / "mobile_desk_kit.json"
T0 = datetime(2026, 10, 27, 9, 0, tzinfo=UTC)

# Building an observation stands in for taking the photo, so construction order is acquisition order.
_ACQUISITION = itertools.count()


def rect_mask(x: int, y: int, w: int, h: int) -> np.ndarray:
    mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
    mask[y : y + h, x : x + w] = True
    return mask


def observation(
    capture_id: str,
    captured_at: datetime,
    items: dict[str, list[tuple[int, int, int, int]]] | None = None,
    visibility: Visibility = Visibility.CLEAR,
    counts: dict[str, int] | None = None,
    unexpected: tuple[str, ...] = (),
    lid_closed: bool = False,
    lid_detected: bool | None = None,
    box: tuple[int, int, int, int] | None = BOX,
    divider: tuple[int, int, int, int] | None = DIVIDER,
    sequence: int | None = None,
) -> CaptureObservation:
    """Build a capture whose SAM masks are the given rectangles.

    By default the scene assessment counts exactly the rectangles placed, so the two sources
    agree; pass counts to make them disagree.
    """
    items = items or {}
    detections = [Detection(label=item_id, score=0.9, mask=rect_mask(*r)) for item_id, rs in items.items() for r in rs]
    if box is not None:
        detections.append(Detection(label="box", score=0.95, mask=rect_mask(*box)))
    if divider is not None:
        detections.append(Detection(label="divider", score=0.9, mask=rect_mask(*divider)))
    if lid_detected is None:
        lid_detected = lid_closed
    if lid_detected:
        detections.append(Detection(label="lid", score=0.9, mask=rect_mask(*BOX)))

    reported = {item_id: len(items.get(item_id, [])) for item_id in PACKED}
    reported.update(counts or {})

    return CaptureObservation(
        capture_id=capture_id,
        sequence=next(_ACQUISITION) if sequence is None else sequence,
        image_sha256=hashlib.sha256(capture_id.encode("utf-8")).hexdigest(),
        captured_at=captured_at,
        width=WIDTH,
        height=HEIGHT,
        scene=SceneAssessment(
            interior_visibility=visibility,
            lid_closed=lid_closed,
            item_counts=reported,
            unexpected_items=unexpected,
        ),
        detections=tuple(detections),
        source="synthetic",
    )


def placed(slots: dict[str, tuple[int, int, int, int]]) -> dict[str, list[tuple[int, int, int, int]]]:
    return {item_id: [r] for item_id, r in slots.items()}
