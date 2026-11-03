from dataclasses import replace
from datetime import UTC, datetime

import numpy as np
import pytest

from bench import EMPTY, WITH_DIVIDER, Bench, perspective
from benchproof import checking
from benchproof.checking import propose_zones
from benchproof.models import Outcome
from benchproof.predicates import Detection
from scenes import BOX, DIVIDER, LEFT_SLOTS, T0, observation, rect_mask

NOW = datetime(2026, 11, 1, 23, 0, tzinfo=UTC)


def test_full_run_from_empty_box_to_packed_compartment(procedure, tmp_path):
    bench = Bench(procedure, tmp_path, [EMPTY, WITH_DIVIDER, {**WITH_DIVIDER, **LEFT_SLOTS}])

    assert bench.checker.check(bench.capture(), "show_empty_box").outcome is Outcome.PASS

    reference = bench.capture()
    zones = propose_zones(reference.observation, procedure.zones)
    calibration = bench.checker.calibrate(reference, zones, "author", NOW)
    assert calibration.revision == 1
    assert bench.checker.check(reference, "place_divider").outcome is Outcome.PASS

    packed = bench.checker.check(bench.capture(), "pack_left")
    assert packed.outcome is Outcome.PASS
    assert bench.checker.session.passed_steps.keys() == {"show_empty_box", "place_divider", "pack_left"}


def test_photos_get_increasing_sequences_and_unique_ids(procedure, tmp_path):
    bench = Bench(procedure, tmp_path, [EMPTY, EMPTY])
    first, second = bench.capture(), bench.capture()

    assert (first.observation.sequence, second.observation.sequence) == (0, 1)
    assert first.observation.capture_id != second.observation.capture_id


def test_recalibration_raises_the_revision(procedure, tmp_path):
    bench = Bench(procedure, tmp_path, [WITH_DIVIDER, WITH_DIVIDER])
    first, second = bench.capture(), bench.capture()

    bench.checker.calibrate(first, propose_zones(first.observation, procedure.zones), "author", NOW)
    again = bench.checker.calibrate(second, propose_zones(second.observation, procedure.zones), "author", NOW)

    assert again.revision == 2


def test_the_same_image_cannot_be_judged_twice(procedure, tmp_path):
    bench = Bench(procedure, tmp_path, [EMPTY, EMPTY])
    bench.checker.check(bench.capture(), "show_empty_box")
    bench._taken = 0

    with pytest.raises(ValueError, match="already judged"):
        bench.checker.check(bench.capture(), None)


def test_each_photo_spends_one_call_per_object_plus_the_judge(procedure, tmp_path):
    bench = Bench(procedure, tmp_path, [EMPTY])
    bench.capture()
    assert bench.checker.budget.used == 7 + 1


def test_vertical_divider_splits_the_box_left_and_right(procedure):
    left, right = propose_zones(observation("c", T0), procedure.zones)

    assert (left.id, right.id) == ("left", "right")
    assert max(x for x, _ in left.polygon) < DIVIDER[0] < min(x for x, _ in right.polygon)
    assert min(y for _, y in left.polygon) >= DIVIDER[1]


def test_horizontal_divider_splits_the_box_top_and_bottom(procedure):
    across = (50, 145, 300, 10)
    top, bottom = propose_zones(observation("c", T0, divider=across), ("top", "bottom"))

    assert max(y for _, y in top.polygon) < across[1] < min(y for _, y in bottom.polygon)
    assert min(x for x, _ in top.polygon) >= across[0]


def test_proposed_zones_pass_calibration_checks(procedure):
    from benchproof.predicates import calibrate

    reference = observation("c", T0)
    calibration = calibrate(reference, propose_zones(reference, procedure.zones), 1, "author", NOW)
    assert {z.id for z in calibration.zones} == set(procedure.zones)


def test_zones_need_both_box_and_divider(procedure):
    with pytest.raises(ValueError, match="box and the divider"):
        propose_zones(observation("c", T0, divider=None), procedure.zones)


def test_zones_need_exactly_two_zone_ids():
    with pytest.raises(ValueError, match="exactly two zones"):
        propose_zones(observation("c", T0), ("a", "b", "c"))


def test_divider_against_a_wall_cannot_be_split(procedure):
    against_wall = (BOX[0], BOX[1], 6, BOX[3])
    with pytest.raises(ValueError, match="too close"):
        propose_zones(observation("c", T0, divider=against_wall), procedure.zones)


def test_largest_divider_mask_sets_the_split(procedure):
    obs = observation("c", T0)
    sliver = Detection("divider", 0.99, rect_mask(60, 60, 4, 4))
    left, _ = propose_zones(replace(obs, detections=obs.detections + (sliver,)), procedure.zones)
    assert max(x for x, _ in left.polygon) < DIVIDER[0]


def test_start_without_keys_fails_before_creating_a_run_log(procedure, tmp_path, monkeypatch):
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    monkeypatch.setenv("BASETEN_API_KEY", "test")

    with pytest.raises(ValueError, match="MODEL_API_KEY"):
        checking.start(procedure, "author", tmp_path / "runs")
    assert not (tmp_path / "runs").exists()


def test_start_approves_the_procedure_in_the_authors_name(procedure, tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_API_KEY", "test")
    monkeypatch.setenv("BASETEN_API_KEY", "test")
    unapproved = procedure.model_copy(update={"approved_by": None, "approved_at": None})

    checker = checking.start(unapproved, "Phong", tmp_path / "runs")

    assert checker.session.procedure.approved_by == "Phong"
    assert checker.budget.limit == checking.SESSION_CALL_LIMIT
    assert np.array(list(checker.session.log_path.parent.iterdir())).size == 1


def test_zones_for_a_box_seen_at_an_angle_pass_calibration(procedure):
    from benchproof.predicates import calibrate

    reference = perspective(observation("c", T0))
    zones = propose_zones(reference, procedure.zones)

    calibration = calibrate(reference, zones, 1, "author", NOW)
    assert {z.id for z in calibration.zones} == set(procedure.zones)


def test_zones_adjusted_past_the_box_are_trimmed_to_it(procedure):
    from benchproof.predicates import calibrate

    reference = perspective(observation("c", T0))
    left, right = propose_zones(reference, procedure.zones)
    wide = left.model_copy(update={"polygon": checking.rectangle(0, 0, DIVIDER[0] - 5, 299)})

    trimmed = checking.fit_to_box(reference, (wide, right))

    calibrate(reference, trimmed, 1, "author", NOW)
    assert max(x for x, _ in trimmed[0].polygon) <= DIVIDER[0] - 5


def test_a_zone_entirely_outside_the_box_is_rejected(procedure):
    reference = perspective(observation("c", T0))
    _, right = propose_zones(reference, procedure.zones)
    outside = right.model_copy(update={"polygon": checking.rectangle(360, 10, 395, 40)})

    with pytest.raises(ValueError, match="outside the box"):
        checking.fit_to_box(reference, (outside,))
