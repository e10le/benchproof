from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from benchproof import labels

APP = Path(__file__).parent.parent / "app.py"

# The first run imports Streamlit cold, which can exceed the 3-second default on a fresh checkout.
APP_TIMEOUT = 30


def test_labeling_page_renders_with_all_steps():
    at = AppTest.from_file(str(APP), default_timeout=APP_TIMEOUT).run()

    assert not at.exception
    assert at.title[0].value == "BenchProof: capture and label"
    assert at.selectbox[1].options == [
        "None (closure or recheck)",
        "Show the open, empty box",
        "Position the divider",
        "Pack the left compartment",
        "Pack the right compartment",
        "Inspect before closing",
    ]


def test_expected_outcome_follows_the_scenario():
    at = AppTest.from_file(str(APP), default_timeout=APP_TIMEOUT).run()
    at.selectbox[0].set_value("hand_occlusion").run()
    assert at.selectbox[2].value == "insufficient"


@pytest.mark.parametrize("error", [OSError("access denied"), ValueError("invalid label log")])
def test_label_read_failures_are_visible_without_app_crash(monkeypatch, error):
    def fail_load(directory):
        raise error

    monkeypatch.setattr(labels, "load_labels", fail_load)
    at = AppTest.from_file(str(APP), default_timeout=APP_TIMEOUT).run()

    assert not at.exception
    assert "Could not load labels" in at.error[0].value
    assert str(error) in at.error[0].value
