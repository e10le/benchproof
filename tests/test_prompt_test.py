import io

import numpy as np
import pytest
from PIL import Image

from benchproof.llm import CallBudget, ModelError
from benchproof.photos import prepare
from benchproof.predicates import Detection
from benchproof.prompt_test import overlay, run, slug
from sam_fake import FakeSam, line, record

SIZE = (40, 30)


def photo_file(tmp_path, name="My Photo.png"):
    path = tmp_path / name
    Image.new("RGB", SIZE, "white").save(path)
    return path


def test_overlay_tints_only_the_masked_pixels():
    photo = prepare(open_png())
    mask = np.zeros((SIZE[1], SIZE[0]), dtype=bool)
    mask[10:20, 10:20] = True

    canvas = overlay(photo, [Detection(label="x", score=0.9, mask=mask)])

    assert canvas.shape == (SIZE[1], SIZE[0], 3)
    assert (canvas[14, 14] != 255).any()
    # Below the mask and away from the score label drawn above it.
    assert (canvas[27, 2] == 255).all()


def test_overlay_without_matches_is_the_plain_photo():
    photo = prepare(open_png())
    assert (overlay(photo, []) == 255).all()


def test_run_saves_one_image_per_phrase_and_reports_counts(tmp_path):
    sam = FakeSam({"black cable": line(record(0, (5, 5, 10, 4), SIZE, 0.81)), "wire": ""})
    budget = CallBudget(2)

    lines = run(sam.segmenter(budget), photo_file(tmp_path), ["black cable", "wire"], tmp_path / "out")

    folder = tmp_path / "out" / "my_photo"
    assert sorted(p.name for p in folder.iterdir()) == ["black_cable.png", "wire.png"]
    assert "black cable: 1 found  0.81" in lines[0]
    assert "wire: 0 found" in lines[1]
    assert budget.used == 2


def test_repeated_phrases_are_asked_once(tmp_path):
    sam = FakeSam({})
    run(sam.segmenter(), photo_file(tmp_path), ["wire", " wire ", "wire"], tmp_path / "out")
    assert len(sam.requests) == 1


@pytest.mark.parametrize("phrases", [["wire", ""], ["usb cable", "USB-cable"]])
def test_unusable_phrase_lists_are_rejected_before_any_call(tmp_path, phrases):
    sam = FakeSam({})
    with pytest.raises(ValueError):
        run(sam.segmenter(), photo_file(tmp_path), phrases, tmp_path / "out")
    assert sam.requests == []


def test_budget_limits_the_number_of_phrases(tmp_path):
    with pytest.raises(ModelError, match="model calls"):
        run(FakeSam({}).segmenter(CallBudget(1)), photo_file(tmp_path), ["a", "b"], tmp_path / "out")


@pytest.mark.parametrize(("text", "expected"), [("Black Cable", "black_cable"), ("usb-c / wire!", "usb_c_wire")])
def test_slug_makes_safe_file_names(text, expected):
    assert slug(text) == expected


def open_png() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", SIZE, "white").save(buffer, format="PNG")
    return buffer.getvalue()
