"""Object masks from SAM 3.1 on Meta Model API.

SAM takes one short noun phrase per request and returns every match as a box plus a box-local mask.
Output is decoded with Meta's meta-sam-parser and pasted into full-photo masks for the predicates.
"""

import threading
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import openai
from meta_sam_parser import (
    CompletedOutcome,
    ImageSegmentationResult,
    SegmentationMaskRecord,
    decode_mask_to_raster,
    image_segmentation_format,
)

from benchproof.llm import CallBudget, ModelError, data_url
from benchproof.models import Procedure
from benchproof.photos import Photo
from benchproof.predicates import Detection

SAM_MODEL = "sam-3.1"

# The openai library builds its stream event types the first time it parses one, and threads doing
# that at the same moment can fail with pydantic errors. The first request in a process therefore
# runs alone; once it has parsed a full stream, requests run in parallel.
_first_request = threading.Lock()
_types_built = threading.Event()


def scene_prompts(procedure: Procedure) -> dict[str, str]:
    """Detection label -> SAM phrase for every object the predicates look for."""
    prompts = {"box": procedure.box_prompt, "divider": procedure.divider_prompt, "lid": procedure.lid_prompt}
    prompts.update({item.id: item.sam_prompt for item in procedure.items})
    return prompts


class Segmenter:
    def __init__(self, client: openai.OpenAI, budget: CallBudget) -> None:
        self._client = client
        self._budget = budget

    def segment_all(self, procedure: Procedure, photo: Photo) -> tuple[Detection, ...]:
        prompts = scene_prompts(procedure)
        with ThreadPoolExecutor(max_workers=len(prompts)) as pool:
            found = pool.map(lambda label: self.segment(photo, label, prompts[label]), prompts)
            return tuple(d for detections in found for d in detections)

    def segment(self, photo: Photo, label: str, prompt: str) -> list[Detection]:
        result = self._parse(self._request(photo, prompt))

        errors = [d for d in result.diagnostics if d.severity == "error"]
        if errors:
            raise ModelError(f"SAM output for '{prompt}' was malformed: {errors[0].message}")

        detections = []
        for record in result.records:
            if record.kind != "mask":
                continue
            mask = _full_mask(record, photo)
            # A box with an empty mask outlines nothing, so it is not evidence of an object.
            if mask.any():
                detections.append(Detection(label=label, score=_confidence(record, prompt), mask=mask))
        return detections

    def _request(self, photo: Photo, prompt: str) -> str:
        if not _types_built.is_set():
            with _first_request:
                if not _types_built.is_set():
                    text = self._send(photo, prompt)
                    _types_built.set()
                    return text
        return self._send(photo, prompt)

    def _send(self, photo: Photo, prompt: str) -> str:
        self._budget.spend()
        try:
            stream = self._client.responses.create(
                model=SAM_MODEL,
                stream=True,
                metadata={"include_confidence": "true"},
                input=[
                    {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {"type": "input_image", "image_url": data_url(photo.jpeg)},
                        ],
                    }
                ],
            )
            return _collect_text(stream, prompt)
        except openai.APIError as e:
            raise ModelError(f"SAM request for '{prompt}' failed: {e}") from e

    def _parse(self, text: str) -> ImageSegmentationResult:
        parser = image_segmentation_format().create_parser()
        parser.push(text)
        return parser.finish(CompletedOutcome()).result


def _collect_text(stream, prompt: str) -> str:
    parts: list[str] = []
    with stream:
        for event in stream:
            if event.type == "response.output_text.delta":
                parts.append(event.delta)
            elif event.type == "response.completed":
                return "".join(parts)
            elif event.type in ("response.failed", "response.incomplete", "error"):
                raise ModelError(f"SAM response for '{prompt}' ended with {event.type}")
    raise ModelError(f"SAM response for '{prompt}' ended before it completed")


def _confidence(record: SegmentationMaskRecord, prompt: str) -> float:
    # Confidence is requested on every call, so a missing value means the response format changed.
    if record.confidence is None:
        raise ModelError(f"SAM returned a mask without confidence for '{prompt}'")
    return record.confidence


def _full_mask(record: SegmentationMaskRecord, photo: Photo) -> np.ndarray:
    bounds = record.bounds
    left, top, right, bottom = int(bounds.left), int(bounds.top), int(bounds.right), int(bounds.bottom)
    if right > photo.width or bottom > photo.height:
        raise ModelError("SAM returned a box outside the photo; it may have seen a different image size")

    raster = np.frombuffer(decode_mask_to_raster(record.mask), dtype=np.uint8)
    raster = raster.reshape(record.mask.height, record.mask.width)
    if raster.shape != (bottom - top, right - left):
        raster = cv2.resize(raster, (right - left, bottom - top), interpolation=cv2.INTER_NEAREST)

    full = np.zeros((photo.height, photo.width), dtype=bool)
    full[top:bottom, left:right] = raster.astype(bool)
    return full
