from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from bench import EMPTY, WITH_DIVIDER, Bench, perspective
from benchproof import checking
from benchproof.checking import Capture, Checker
from benchproof.drawing import jpeg, overlay
from benchproof.llm import CallBudget
from benchproof.models import Outcome
from scenes import LEFT_SLOTS, observation

PAGE = Path(__file__).parent.parent / "pages" / "1_Check.py"
# The first run imports Streamlit cold, which can exceed the 3-second default on a fresh checkout.
PAGE_TIMEOUT = 30


def page() -> AppTest:
    return AppTest.from_file(str(PAGE), default_timeout=PAGE_TIMEOUT)


def started(checker: Checker) -> AppTest:
    return started_with(page(), checker)


def started_with(at: AppTest, checker: Checker) -> AppTest:
    at.session_state["checker"] = checker
    at.session_state["author"] = "author"
    return at.run()


def test_start_page_lists_the_real_kit_steps():
    at = page().run()

    assert not at.exception
    assert at.title[0].value == "BenchProof: check"
    assert any("Pack the left compartment" in m.value for m in at.markdown)


def test_missing_keys_show_an_error_instead_of_crashing(monkeypatch):
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    monkeypatch.delenv("BASETEN_API_KEY", raising=False)
    at = page().run()

    at.text_input[0].input("Phong")
    at.button[0].click().run()

    assert not at.exception
    assert "Could not start a session" in at.error[0].value
    assert "checker" not in at.session_state


def test_a_name_is_required_to_approve(monkeypatch):
    monkeypatch.setattr(checking, "start", lambda *args: pytest.fail("start must not run without a name"))
    at = page().run()

    at.text_input[0].input("   ")
    at.button[0].click().run()

    assert "Enter your name" in at.error[0].value


def test_starting_offers_the_first_step_and_a_recheck(procedure, tmp_path, monkeypatch):
    bench = Bench(procedure, tmp_path, [])
    monkeypatch.setattr(checking, "start", lambda *args: bench.checker)
    at = page().run()

    at.text_input[0].input("Phong")
    at.button[0].click().run()

    assert not at.exception
    assert at.session_state["author"] == "Phong"
    assert at.radio[0].options == ["Show the open, empty box", "Re-check without a step"]


def test_a_failed_check_shows_the_outcome_and_the_fix(procedure, tmp_path):
    bench = Bench(procedure, tmp_path, [{**EMPTY, **LEFT_SLOTS}])
    capture = bench.capture()
    judgment = bench.checker.check(capture, "show_empty_box")
    assert judgment.outcome is Outcome.FAIL

    at = page()
    at.session_state["last"] = (judgment, jpeg(overlay(capture.photo, list(capture.observation.detections))))
    started_with(at, bench.checker)

    assert not at.exception
    assert "FAIL" in at.error[0].value
    fixes = [m.value for m in at.markdown if m.value.startswith("**To fix:**")]
    assert judgment.recovery[0] in fixes[0]


def test_a_closed_box_offers_the_evidence_record(inspected, clock):
    inspected.check(observation("closed", clock(), lid_closed=True), None)
    at = started(Checker(inspected, None, None, CallBudget(1)))

    assert not at.exception
    assert "Contents were observed before closure" in at.success[0].value
    assert at.radio == []


def test_approving_the_proposed_zones_calibrates_and_checks_the_divider(procedure, tmp_path):
    bench = Bench(procedure, tmp_path, [EMPTY, WITH_DIVIDER])
    bench.checker.check(bench.capture(), "show_empty_box")
    reference = bench.capture()
    at = page()
    at.session_state["pending"] = (reference, checking.propose_zones(reference.observation, procedure.zones))
    started_with(at, bench.checker)

    assert len(at.slider) == 2 * len(procedure.zones)
    approve = next(b for b in at.button if b.label == "Approve zones and check")
    approve.click().run()

    assert not at.exception
    assert "pending" not in at.session_state
    assert bench.checker.session.calibration.revision == 1
    assert "place_divider" in bench.checker.session.passed_steps
    assert "PASS" in at.success[0].value


def test_zones_for_a_box_seen_at_an_angle_can_be_approved(procedure, tmp_path):
    bench = Bench(procedure, tmp_path, [EMPTY, WITH_DIVIDER])
    bench.checker.check(bench.capture(), "show_empty_box")
    taken = bench.capture()
    reference = Capture(photo=taken.photo, observation=perspective(taken.observation))
    at = page()
    at.session_state["pending"] = (reference, checking.propose_zones(reference.observation, procedure.zones))
    started_with(at, bench.checker)

    assert at.error == []
    next(b for b in at.button if b.label == "Approve zones and check").click().run()

    assert not at.exception
    assert at.error == []
    assert bench.checker.session.calibration is not None
