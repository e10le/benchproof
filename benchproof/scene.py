"""Whole-scene reading of one capture by a vision model.

The model only reports what it sees; every pass, fail or insufficient decision stays in the
deterministic predicates. Its reply is untrusted and is validated before anything uses it.
"""

import base64
import io
import os
import re
import sys
import threading
from pathlib import Path

import openai
from PIL import Image

from benchproof.models import Procedure, SceneAssessment
from benchproof.predicates import check_assessment

BASETEN_BASE_URL = "https://inference.baseten.co/v1"
SPARK_OPEN_MODEL = "meta-models/muse-spark-open-1026"

# Reasoning is always on and defaults to the slowest, most expensive level; tune on development photos.
REASONING_EFFORT = "medium"
# Reasoning tokens count toward max_tokens, so leave room for the answer after them.
MAX_TOKENS = 8000
REQUEST_TIMEOUT_SECONDS = 120
MAX_IMAGE_BYTES = 10 * 1024 * 1024
IMAGE_MEDIA_TYPES = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp"}

SYSTEM_PROMPT = (
    "You inspect one overhead photo of a box for a packing check. Report only what is visible. "
    "Reply with one JSON object and nothing else."
)


class PerceptionError(Exception):
    """A model could not produce a usable reading; the capture cannot be judged."""


class CallBudget:
    def __init__(self, limit: int) -> None:
        if limit < 1:
            raise ValueError("a call budget needs at least one call")
        self.limit = limit
        self._used = 0
        self._lock = threading.Lock()

    @property
    def used(self) -> int:
        return self._used

    def spend(self) -> None:
        with self._lock:
            if self._used >= self.limit:
                raise PerceptionError(f"the session used all {self.limit} model calls; start a new session")
            self._used += 1


def baseten_client() -> openai.OpenAI:
    api_key = os.environ.get("BASETEN_API_KEY")
    if not api_key:
        raise ValueError("set BASETEN_API_KEY in this terminal before calling the model")
    return openai.OpenAI(api_key=api_key, base_url=BASETEN_BASE_URL, timeout=REQUEST_TIMEOUT_SECONDS, max_retries=2)


class SceneJudge:
    def __init__(self, client: openai.OpenAI, model: str, budget: CallBudget, max_attempts: int = 2) -> None:
        self._client = client
        self._model = model
        self._budget = budget
        self._max_attempts = max_attempts

    def assess(self, procedure: Procedure, image_bytes: bytes) -> SceneAssessment:
        messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _instructions(procedure)},
                    {"type": "image_url", "image_url": {"url": _data_url(image_bytes)}},
                ],
            },
        ]

        problem = ""
        for _ in range(self._max_attempts):
            self._budget.spend()
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    reasoning_effort=REASONING_EFFORT,
                    max_tokens=MAX_TOKENS,
                )
            except openai.APIError as e:
                raise PerceptionError(f"scene model request failed: {e}") from e

            content = response.choices[0].message.content
            try:
                return _parse(content, procedure)
            except ValueError as e:
                problem = str(e)
                messages += [
                    {"role": "assistant", "content": content or ""},
                    {
                        "role": "user",
                        "content": f"That reply was not usable: {problem}. Reply with only the JSON object.",
                    },
                ]

        raise PerceptionError(f"scene model gave no usable reply in {self._max_attempts} attempts: {problem}")


def _instructions(procedure: Procedure) -> str:
    items = "\n".join(f"- {item.id}: {item.name}" for item in procedure.items)
    counts = ", ".join(f'"{item.id}": 0' for item in procedure.items)
    return f"""Kit items:
{items}

Reply with exactly this JSON shape:
{{"interior_visibility": "clear", "lid_closed": false, "item_counts": {{{counts}}}, "unexpected_items": []}}

Rules:
- interior_visibility: "clear" if the whole inside of the box is visible, "partial" if part of it is
  covered (for example by a hand), "none" if the inside cannot be seen.
- lid_closed: true only if a lid or flaps cover the box opening.
- item_counts: how many of each kit item are inside the box. Use 0 if an item is absent or hidden.
- unexpected_items: short names of objects inside the box that are not kit items. Ignore the cardboard divider."""


def _data_url(image_bytes: bytes) -> str:
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise ValueError(f"photo is larger than {MAX_IMAGE_BYTES // (1024 * 1024)} MB; lower the resolution")

    with Image.open(io.BytesIO(image_bytes)) as image:
        media_type = IMAGE_MEDIA_TYPES.get(image.format)
    if media_type is None:
        raise ValueError(f"photo must be one of {sorted(IMAGE_MEDIA_TYPES)}")
    return f"data:{media_type};base64,{base64.b64encode(image_bytes).decode('ascii')}"


def _parse(content: str | None, procedure: Procedure) -> SceneAssessment:
    if not content:
        raise ValueError("the reply was empty")

    # Models sometimes wrap JSON in a code fence or a sentence; take the outermost object.
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if match is None:
        raise ValueError("the reply contained no JSON object")

    scene = SceneAssessment.model_validate_json(match.group(0))
    check_assessment(procedure, scene)
    return scene


if __name__ == "__main__":
    # Manual check against the live model: python -m benchproof.scene path/to/photo.jpg
    kit = Procedure.model_validate_json(Path("procedures/mobile_desk_kit.json").read_text(encoding="utf-8"))
    budget = CallBudget(2)
    reading = SceneJudge(baseten_client(), SPARK_OPEN_MODEL, budget).assess(kit, Path(sys.argv[1]).read_bytes())
    sys.stdout.write(reading.model_dump_json(indent=2) + "\n")
    sys.stdout.write(f"model calls used: {budget.used}\n")
