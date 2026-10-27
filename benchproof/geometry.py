from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from benchproof.models import BBox

# Initial, unmeasured tolerance for ambiguous square orientation; tune on development captures.
NEAR_SQUARE_RATIO = 1.02


def polygon_mask(polygon: tuple[tuple[float, float], ...], width: int, height: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    points = np.round(np.array(polygon, dtype=np.float64)).astype(np.int32)
    cv2.fillPoly(mask, [points], 1)
    return mask.astype(bool)


def filled_hull(mask: np.ndarray) -> np.ndarray:
    # An overhead box mask may cover only the walls; the hull also covers the floor where items sit.
    points = cv2.findNonZero(mask.astype(np.uint8))
    hull = cv2.convexHull(points)
    region = np.zeros(mask.shape, dtype=np.uint8)
    cv2.fillPoly(region, [hull], 1)
    return region.astype(bool)


def overlap_fraction(mask: np.ndarray, region: np.ndarray) -> float:
    """Share of mask pixels that fall inside region. The mask must not be empty."""
    return float(np.count_nonzero(mask & region)) / float(np.count_nonzero(mask))


def iou(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.count_nonzero(a & b)) / float(np.count_nonzero(a | b))


def box_pose(mask: np.ndarray) -> tuple["BBox", float, float]:
    # Models validate polygons through this module, so importing BBox here breaks the cycle.
    from benchproof.models import BBox

    points = cv2.findNonZero(mask.astype(np.uint8))
    x, y, w, h = cv2.boundingRect(points)
    _, (rect_width, rect_height), angle = cv2.minAreaRect(points)
    if rect_width < rect_height:
        angle += 90.0
    # OpenCV measures between pixel centers; include their extent so thin masks stay finite.
    aspect_ratio = (max(rect_width, rect_height) + 1.0) / (min(rect_width, rect_height) + 1.0)
    return BBox(x=x, y=y, w=w, h=h), float(angle % 180.0), float(aspect_ratio)


def mask_center(mask: np.ndarray) -> tuple[float, float]:
    ys, xs = np.nonzero(mask)
    return float(xs.mean()), float(ys.mean())


def angle_difference(a: float, b: float, aspect_a: float, aspect_b: float) -> float:
    # A near-square mask's major axis can flip with one pixel of noise.
    period = 90.0 if max(aspect_a, aspect_b) <= NEAR_SQUARE_RATIO else 180.0
    d = abs(a - b) % period
    return min(d, period - d)


def polygon_is_simple(polygon: tuple[tuple[float, float], ...]) -> bool:
    """Reject repeated vertices, zero area and crossing polygon edges."""
    if len(set(polygon)) != len(polygon):
        return False

    edges = list(zip(polygon, polygon[1:] + polygon[:1], strict=True))
    area = sum(a[0] * b[1] - a[1] * b[0] for a, b in edges)
    if area == 0:
        return False

    for i, (a, b) in enumerate(edges):
        for j in range(i + 1, len(edges)):
            if j == i + 1 or (i == 0 and j == len(edges) - 1):
                continue
            if _segments_intersect(a, b, *edges[j]):
                return False
    return True


def polygons_overlap(a: tuple[tuple[float, float], ...], b: tuple[tuple[float, float], ...]) -> bool:
    """Return whether polygons intersect, contain one another, or touch."""
    edges_a = zip(a, a[1:] + a[:1], strict=True)
    edges_b = list(zip(b, b[1:] + b[:1], strict=True))
    for first, second in edges_a:
        if any(_segments_intersect(first, second, *edge) for edge in edges_b):
            return True
    contour_a = np.asarray(a, dtype=np.float32)
    contour_b = np.asarray(b, dtype=np.float32)
    return cv2.pointPolygonTest(contour_a, b[0], False) >= 0 or cv2.pointPolygonTest(contour_b, a[0], False) >= 0


def _segments_intersect(a, b, c, d) -> bool:
    if max(a[0], b[0]) < min(c[0], d[0]) or max(c[0], d[0]) < min(a[0], b[0]):
        return False
    if max(a[1], b[1]) < min(c[1], d[1]) or max(c[1], d[1]) < min(a[1], b[1]):
        return False

    def side(start, end, point):
        return (end[0] - start[0]) * (point[1] - start[1]) - (end[1] - start[1]) * (point[0] - start[0])

    return side(a, b, c) * side(a, b, d) <= 0 and side(c, d, a) * side(c, d, b) <= 0
