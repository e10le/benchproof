import numpy as np
import pytest

from benchproof import geometry
from scenes import HEIGHT, WIDTH, rect_mask


def test_polygon_mask_covers_the_polygon_area():
    mask = geometry.polygon_mask(((10, 10), (110, 10), (110, 60), (10, 60)), WIDTH, HEIGHT)
    assert mask.shape == (HEIGHT, WIDTH)
    # fillPoly includes both edges, so a 100 x 50 polygon covers 101 x 51 pixels.
    assert np.count_nonzero(mask) == 101 * 51


def test_overlap_fraction_counts_only_mask_pixels_inside_region():
    mask = rect_mask(0, 0, 10, 10)
    region = rect_mask(5, 0, 50, 50)
    assert geometry.overlap_fraction(mask, region) == pytest.approx(0.5)


def test_iou_of_half_shifted_squares():
    a = rect_mask(0, 0, 10, 10)
    b = rect_mask(5, 0, 10, 10)
    assert geometry.iou(a, b) == pytest.approx(50 / 150)


def test_filled_hull_covers_the_floor_inside_box_walls():
    walls = rect_mask(50, 50, 100, 100) & ~rect_mask(60, 60, 80, 80)
    floor_point = (100, 100)
    assert not walls[floor_point]
    assert geometry.filled_hull(walls)[floor_point]


def test_box_pose_of_axis_aligned_rectangle():
    bbox, _, aspect_ratio = geometry.box_pose(rect_mask(50, 40, 200, 100))
    assert (bbox.x, bbox.y, bbox.w, bbox.h) == (50, 40, 200, 100)
    assert aspect_ratio == pytest.approx(2.0)


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [(10.0, 12.0, 2.0), (89.0, 1.0, 88.0), (-90.0, 0.0, 90.0), (45.0, 0.0, 45.0), (179.0, 1.0, 2.0)],
)
def test_non_square_angle_difference_wraps_at_180_degrees(a, b, expected):
    assert geometry.angle_difference(a, b, 2.0, 2.0) == pytest.approx(expected)


def test_near_square_box_angle_is_stable_under_one_pixel_of_noise():
    _, wide, wide_aspect = geometry.box_pose(rect_mask(100, 100, 200, 199))
    _, tall, tall_aspect = geometry.box_pose(rect_mask(100, 100, 199, 200))
    assert geometry.angle_difference(wide, tall, wide_aspect, tall_aspect) == pytest.approx(0)


@pytest.mark.parametrize(("aspect_a", "aspect_b"), [(1.02, 1.02), (1.01, 1.02)])
def test_square_ambiguity_includes_the_tolerance_boundary(aspect_a, aspect_b):
    assert geometry.angle_difference(0, 90, aspect_a, aspect_b) == pytest.approx(0)


@pytest.mark.parametrize(("aspect_a", "aspect_b"), [(1.021, 1.0), (1.0, 1.021)])
def test_square_ambiguity_requires_both_shapes_near_square(aspect_a, aspect_b):
    assert geometry.angle_difference(0, 90, aspect_a, aspect_b) == pytest.approx(90)


@pytest.mark.parametrize(("width", "height"), [(1, 100), (100, 1), (1, 1)])
def test_box_pose_has_finite_aspect_ratio_for_thin_masks(width, height):
    _, angle, aspect_ratio = geometry.box_pose(rect_mask(10, 10, width, height))
    assert np.isfinite(angle)
    assert np.isfinite(aspect_ratio)
    assert aspect_ratio >= 1


def test_simple_concave_polygon_is_valid():
    polygon = ((10, 10), (90, 10), (90, 90), (50, 50), (10, 90))
    assert geometry.polygon_is_simple(polygon)


def test_nested_polygons_overlap_without_crossing_edges():
    outside = ((0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0))
    inside = ((10.0, 10.0), (20.0, 10.0), (20.0, 20.0), (10.0, 20.0))
    assert geometry.polygons_overlap(outside, inside)
