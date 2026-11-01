"""Try SAM 3.1 phrases on one photo and save images showing what each phrase matched.

Usage: python -m benchproof.prompt_test path/to/photo.jpg "black cable" "usb cable" "wire"
Each phrase is one SAM call. Images are written to prompt_tests/<photo name>/.
"""

import argparse
import io
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from benchproof.llm import CallBudget, meta_client
from benchproof.photos import Photo, prepare
from benchproof.predicates import Detection
from benchproof.segmenter import Segmenter

OUTPUT_DIR = Path("prompt_tests")
# Distinct RGB colors so neighbouring matches can be told apart in the image.
COLORS = [(230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48), (145, 30, 180), (70, 240, 240)]


def try_phrases(segmenter: Segmenter, photo: Photo, phrases: list[str]) -> dict[str, list[Detection]]:
    with ThreadPoolExecutor(max_workers=len(phrases)) as pool:
        found = pool.map(lambda phrase: segmenter.segment(photo, phrase, phrase), phrases)
        return dict(zip(phrases, found, strict=True))


def overlay(photo: Photo, detections: list[Detection]) -> np.ndarray:
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
        label_at = (top_left[0], max(top_left[1] - 6 * thickness, round(24 * scale)))
        cv2.putText(canvas, f"{detection.score:.2f}", label_at, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)
    return canvas


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def run(segmenter: Segmenter, photo_path: Path, phrases: list[str], output_dir: Path) -> list[str]:
    cleaned = list(dict.fromkeys(p.strip() for p in phrases))
    if not all(cleaned):
        raise ValueError("phrases must not be empty")
    if len({slug(p) for p in cleaned}) != len(cleaned):
        raise ValueError("two phrases would save to the same file name; make them differ by more than punctuation")

    photo = prepare(photo_path.read_bytes())
    results = try_phrases(segmenter, photo, cleaned)

    folder = output_dir / slug(photo_path.stem)
    folder.mkdir(parents=True, exist_ok=True)
    lines = []
    for phrase, detections in results.items():
        image_path = folder / f"{slug(phrase)}.png"
        Image.fromarray(overlay(photo, detections)).save(image_path)
        scores = ", ".join(f"{d.score:.2f}" for d in sorted(detections, key=lambda d: d.score, reverse=True))
        lines.append(f"{phrase:>24}: {len(detections)} found  {scores}  -> {image_path}")
    return lines


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Try SAM 3.1 phrases on one photo.")
    parser.add_argument("photo", type=Path)
    parser.add_argument("phrases", nargs="+")
    args = parser.parse_args()

    segmenter = Segmenter(meta_client(), CallBudget(len(args.phrases)))
    for line in run(segmenter, args.photo, args.phrases, OUTPUT_DIR):
        sys.stdout.write(line + "\n")
