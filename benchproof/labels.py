"""Hand labels for test captures, written before any model sees the image."""

import io
import os
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

from PIL import Image, ImageOps
from pydantic import Field

from benchproof.models import ID_PATTERN, IMAGE_SHA256_PATTERN, Outcome, StrictModel, UtcDatetime

LABELS_FILE = "labels.jsonl"


class Split(StrEnum):
    DEVELOPMENT = "development"
    HELDOUT = "heldout"


class Scenario(StrEnum):
    CORRECT = "correct"
    MISSING_CABLE = "missing_cable"
    MOUSE_WRONG_ZONE = "mouse_wrong_zone"
    EXTRA_MARKER = "extra_marker"
    PEN_FOR_MARKER = "pen_for_marker"
    HAND_OCCLUSION = "hand_occlusion"
    PREMATURE_CLOSURE = "premature_closure"
    ITEM_REMOVED_AFTER_PASS = "item_removed_after_pass"
    DIVIDER_SHIFTED = "divider_shifted"
    CORRECTION = "correction"
    CLOSE_AFTER_INSPECTION = "close_after_inspection"


# Suggested label per scenario. The person labeling still confirms what the photo actually shows.
SCENARIO_EXPECTED = {
    Scenario.CORRECT: Outcome.PASS,
    Scenario.MISSING_CABLE: Outcome.FAIL,
    Scenario.MOUSE_WRONG_ZONE: Outcome.FAIL,
    Scenario.EXTRA_MARKER: Outcome.FAIL,
    Scenario.PEN_FOR_MARKER: Outcome.FAIL,
    Scenario.HAND_OCCLUSION: Outcome.INSUFFICIENT,
    Scenario.PREMATURE_CLOSURE: Outcome.FAIL,
    Scenario.ITEM_REMOVED_AFTER_PASS: Outcome.FAIL,
    Scenario.DIVIDER_SHIFTED: Outcome.INSUFFICIENT,
    Scenario.CORRECTION: Outcome.PASS,
    Scenario.CLOSE_AFTER_INSPECTION: Outcome.PASS,
}


class CaptureLabel(StrictModel):
    capture_id: str = Field(min_length=1)
    split: Split
    scenario: Scenario
    sequence_id: str = Field(pattern=ID_PATTERN)
    sequence_index: int = Field(ge=0)
    target_step: str | None
    expected: Outcome
    note: str = ""
    image_file: str = Field(min_length=1)
    image_sha256: str = Field(pattern=IMAGE_SHA256_PATTERN)
    labeled_at: UtcDatetime


def load_labels(directory: Path) -> list[CaptureLabel]:
    path = directory / LABELS_FILE
    if not path.exists():
        return []

    lines = path.read_text(encoding="utf-8").splitlines()
    return [CaptureLabel.model_validate_json(line) for line in lines if line.strip()]


@contextmanager
def _capture_lock(directory: Path) -> Iterator[None]:
    directory.mkdir(parents=True, exist_ok=True)
    lock_path = directory / ".labels.lock"
    try:
        lock_file = lock_path.open("x", encoding="utf-8")
    except FileExistsError as e:
        raise ValueError(
            "another label save is active; retry after it finishes. If a save was interrupted, "
            f"close every app instance before removing {lock_path}"
        ) from e

    try:
        with lock_file:
            yield
    finally:
        lock_path.unlink()


def _write_capture(directory: Path, image: Image.Image, label: CaptureLabel, existing: list[CaptureLabel]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    image_path = directory / label.image_file
    temporary_path = directory / f"labels.{uuid4().hex}.jsonl"
    image_created = False
    try:
        with image_path.open("xb") as f:
            image_created = True
            image.save(f, format="PNG")
            f.flush()
            os.fsync(f.fileno())
        with temporary_path.open("x", encoding="utf-8") as f:
            for saved in [*existing, label]:
                f.write(saved.model_dump_json() + "\n")
            f.flush()
            os.fsync(f.fileno())
        temporary_path.replace(directory / LABELS_FILE)
    except OSError:
        if image_created:
            image_path.unlink(missing_ok=True)
        raise
    finally:
        temporary_path.unlink(missing_ok=True)


def save_capture(
    directory: Path,
    image_bytes: bytes,
    split: Split,
    scenario: Scenario,
    sequence_id: str,
    sequence_index: int,
    target_step: str | None,
    expected: Outcome,
    note: str,
) -> CaptureLabel:
    """Store the image as PNG and append its label; the label is final once written."""
    labeled_at = datetime.now(UTC)
    capture_id = f"{labeled_at:%Y%m%dT%H%M%S%f}-{scenario}"

    try:
        with Image.open(io.BytesIO(image_bytes)) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
    except Image.DecompressionBombError as e:
        raise ValueError("photo is too large; lower the camera resolution and retake it") from e
    dimensions = image.width.to_bytes(8, "big") + image.height.to_bytes(8, "big")
    image_sha256 = sha256(dimensions + image.tobytes()).hexdigest()

    label = CaptureLabel(
        capture_id=capture_id,
        split=split,
        scenario=scenario,
        sequence_id=sequence_id,
        sequence_index=sequence_index,
        target_step=target_step,
        expected=expected,
        note=note,
        image_file=f"{capture_id}.png",
        image_sha256=image_sha256,
        labeled_at=labeled_at,
    )

    # Both splits share a lock so a concurrent save cannot bypass the leakage checks.
    with image, _capture_lock(directory.parent):
        existing = load_labels(directory)
        if any(lab.sequence_id == sequence_id and lab.sequence_index == sequence_index for lab in existing):
            raise ValueError(f"sequence {sequence_id} already has a capture at index {sequence_index}")
        # A session rejects a reused image, so a duplicate would make the sequence unreplayable.
        if any(lab.image_sha256 == image_sha256 for lab in existing):
            raise ValueError("this image is already saved; take a new photo")
        for other_split in Split:
            if other_split == split:
                continue
            other_labels = load_labels(directory.parent / other_split.value)
            if any(lab.sequence_id == sequence_id for lab in other_labels):
                raise ValueError(f"sequence {sequence_id} already belongs to {other_split.value}")
            if any(lab.image_sha256 == image_sha256 for lab in other_labels):
                raise ValueError(f"image already belongs to {other_split.value}; capture a separate attempt")
        _write_capture(directory, image, label, existing)
    return label
