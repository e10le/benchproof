"""Draw what the models found, and the approved zones, on top of a photo."""

import io

import cv2
import numpy as np
from PIL import Image

from benchproof.models import Zone
from benchproof.photos import Photo
from benchproof.predicates import Detection

# Distinct RGB colors so neighbouring matches can be told apart in the image.
COLORS = [(230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48), (145, 30, 180), (70, 240, 240)]
ZONE_COLOR = (255, 225, 25)


def overlay(photo: Photo, detections: list[Detection], zones: tuple[Zone, ...] = (), named: bool = False) -> np.ndarray:
    with Image.open(io.BytesIO(photo.jpeg)) as image:
        canvas = np.array(image.convert("RGB"))

    # Line and text size follow the photo so labels stay readable from thumbnails to full size.
    scale = max(0.5, photo.width / 1000)
    thickness = max(2, round(2 * scale))

    for i, detection in enumerate(sorted(detections, key=lambda d: d.score, reverse=True)):
        color = COLORS[i % len(COLORS)]
        canvas[detection.mask] = (0.5 * canvas[detection.mask] + 0.5 * np.array(color)).astype(np.uint8)

        ys, xs = np.nonzero(detection.mask)
        top_left = (int(xs.min()), int(ys.min()))
        cv2.rectangle(canvas, top_left, (int(xs.max()), int(ys.max())), color, thickness)
        text = f"{detection.label} {detection.score:.2f}" if named else f"{detection.score:.2f}"
        _label(canvas, text, top_left, color, scale, thickness)

    for zone in zones:
        points = np.round(np.array(zone.polygon)).astype(np.int32)
        cv2.polylines(canvas, [points], isClosed=True, color=ZONE_COLOR, thickness=thickness)
        # Zone names go just inside the top-left corner, clear of the box and divider labels above it.
        inside = (int(points[:, 0].min()) + 3 * thickness, int(points[:, 1].min()) + round(24 * scale) + thickness)
        cv2.putText(canvas, zone.id, inside, cv2.FONT_HERSHEY_SIMPLEX, scale, ZONE_COLOR, thickness)
    return canvas


def jpeg(canvas: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(canvas).save(buffer, format="JPEG", quality=85)
    return buffer.getvalue()


def _label(canvas: np.ndarray, text: str, at: tuple[int, int], color: tuple, scale: float, thickness: int) -> None:
    position = (at[0], max(at[1] - 6 * thickness, round(24 * scale)))
    cv2.putText(canvas, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)
