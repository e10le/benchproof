import io
from pathlib import Path

import pytest
from PIL import Image, UnidentifiedImageError

from benchproof.labels import LABELS_FILE, Scenario, Split, load_labels, save_capture
from benchproof.models import Outcome


def png_bytes(color="white", mode="RGB", size=(8, 6)) -> bytes:
    buffer = io.BytesIO()
    Image.new(mode, size, color).save(buffer, format="PNG")
    return buffer.getvalue()


def save(directory, index=0, **overrides):
    args = dict(
        split=Split.DEVELOPMENT,
        scenario=Scenario.MISSING_CABLE,
        sequence_id="run_01",
        sequence_index=index,
        target_step="pack_right",
        expected=Outcome.FAIL,
        note="cable on the desk",
    )
    args.update(overrides)
    # Each index gets distinct pixels; index 0 stays white for the cross-encoding test.
    return save_capture(directory, png_bytes((255, 255, 255 - index)), **args)


def test_saved_label_round_trips_with_its_image(tmp_path):
    label = save(tmp_path)

    assert load_labels(tmp_path) == [label]
    assert Image.open(tmp_path / label.image_file).size == (8, 6)


def test_labels_append_in_order(tmp_path):
    first = save(tmp_path, index=0)
    second = save(tmp_path, index=1)
    assert [lab.capture_id for lab in load_labels(tmp_path)] == [first.capture_id, second.capture_id]


def test_same_image_cannot_be_saved_twice_in_one_split(tmp_path):
    first = save(tmp_path, index=0)

    with pytest.raises(ValueError, match="already saved"):
        save_capture(
            tmp_path,
            png_bytes(),
            split=Split.DEVELOPMENT,
            scenario=Scenario.CORRECT,
            sequence_id="run_01",
            sequence_index=1,
            target_step=None,
            expected=Outcome.PASS,
            note="",
        )

    assert load_labels(tmp_path) == [first]
    assert list(tmp_path.glob("*.png")) == [tmp_path / first.image_file]


def test_missing_directory_has_no_labels(tmp_path):
    assert load_labels(tmp_path / "nothing") == []


def test_duplicate_sequence_position_is_rejected(tmp_path):
    save(tmp_path, index=0)
    with pytest.raises(ValueError, match="already has a capture"):
        save(tmp_path, index=0)


def test_invalid_sequence_id_writes_nothing(tmp_path):
    with pytest.raises(ValueError):
        save(tmp_path, sequence_id="Run 1")
    assert not (tmp_path / LABELS_FILE).exists()


def test_non_image_bytes_write_no_label(tmp_path):
    with pytest.raises(UnidentifiedImageError):
        save_capture(
            tmp_path,
            b"not an image",
            split=Split.DEVELOPMENT,
            scenario=Scenario.CORRECT,
            sequence_id="run_01",
            sequence_index=0,
            target_step=None,
            expected=Outcome.PASS,
            note="",
        )
    assert not (tmp_path / LABELS_FILE).exists()


def test_sequence_cannot_cross_evaluation_splits(tmp_path):
    save(tmp_path / "development")

    with pytest.raises(ValueError, match="sequence.*development"):
        save_capture(
            tmp_path / "heldout",
            png_bytes("black"),
            split=Split.HELDOUT,
            scenario=Scenario.CORRECT,
            sequence_id="run_01",
            sequence_index=1,
            target_step=None,
            expected=Outcome.PASS,
            note="",
        )
    assert load_labels(tmp_path / "heldout") == []


def test_same_pixels_cannot_cross_splits_with_different_encoding(tmp_path):
    save(tmp_path / "development")

    with pytest.raises(ValueError, match="image.*development"):
        save_capture(
            tmp_path / "heldout",
            png_bytes(255, mode="L"),
            split=Split.HELDOUT,
            scenario=Scenario.CORRECT,
            sequence_id="run_02",
            sequence_index=0,
            target_step=None,
            expected=Outcome.PASS,
            note="",
        )
    assert load_labels(tmp_path / "heldout") == []


def test_distinct_attempts_can_use_different_splits(tmp_path):
    save(tmp_path / "development")
    heldout = save_capture(
        tmp_path / "heldout",
        png_bytes("black"),
        split=Split.HELDOUT,
        scenario=Scenario.CORRECT,
        sequence_id="run_02",
        sequence_index=0,
        target_step=None,
        expected=Outcome.PASS,
        note="",
    )

    assert load_labels(tmp_path / "heldout") == [heldout]


def test_image_digest_includes_dimensions(tmp_path):
    first = save(tmp_path, index=0)
    second = save_capture(
        tmp_path,
        png_bytes(size=(6, 8)),
        split=Split.DEVELOPMENT,
        scenario=Scenario.CORRECT,
        sequence_id="run_01",
        sequence_index=1,
        target_step=None,
        expected=Outcome.PASS,
        note="",
    )

    assert first.image_sha256 != second.image_sha256


def test_failed_label_write_preserves_labels_and_removes_new_image(tmp_path, monkeypatch):
    first = save(tmp_path, index=0)
    original_labels = (tmp_path / LABELS_FILE).read_bytes()
    original_open = Path.open

    def reject_label_write(path, mode="r", *args, **kwargs):
        if path.suffix == ".jsonl" and any(flag in mode for flag in "awx"):
            raise OSError("disk full")
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", reject_label_write)
    with pytest.raises(OSError, match="disk full"):
        save(tmp_path, index=1)

    assert (tmp_path / LABELS_FILE).read_bytes() == original_labels
    assert list(tmp_path.glob("*.png")) == [tmp_path / first.image_file]


def test_concurrent_writer_is_rejected_without_writing_capture(tmp_path):
    (tmp_path / ".labels.lock").touch()

    with pytest.raises(ValueError, match="another.*save"):
        save(tmp_path / "development")

    assert not (tmp_path / "development" / LABELS_FILE).exists()


def test_failed_label_replacement_leaves_previous_capture_intact(tmp_path, monkeypatch):
    first = save(tmp_path, index=0)
    original_labels = (tmp_path / LABELS_FILE).read_bytes()

    def reject_replacement(path, target):
        raise OSError("replacement denied")

    monkeypatch.setattr(Path, "replace", reject_replacement)
    with pytest.raises(OSError, match="replacement denied"):
        save(tmp_path, index=1)

    assert (tmp_path / LABELS_FILE).read_bytes() == original_labels
    assert set(tmp_path.iterdir()) == {tmp_path / first.image_file, tmp_path / LABELS_FILE}
    assert not (tmp_path.parent / ".labels.lock").exists()


def test_corrupt_label_log_stays_visible(tmp_path):
    (tmp_path / LABELS_FILE).write_text("not json\n", encoding="utf-8")

    with pytest.raises(ValueError):
        load_labels(tmp_path)


def test_oversized_image_is_rejected_without_writing(tmp_path, monkeypatch):
    # Lowering Pillow's limit lets a tiny image stand in for an oversized phone photo.
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 10)

    with pytest.raises(ValueError, match="too large"):
        save(tmp_path / "development")

    assert not (tmp_path / "development").exists()


def test_note_with_unicode_line_separator_loads(tmp_path):
    label = save(tmp_path, note="cable\u2028under the mouse")

    assert load_labels(tmp_path) == [label]
