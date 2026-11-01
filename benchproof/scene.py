"""Whole-scene reading of one capture by a vision model.

The model only reports what it sees; every pass, fail or insufficient decision stays in the
deterministic predicates. Its reply is untrusted and is validated before anything uses it.
"""

import sys
from pathlib import Path

import openai

from benchproof.llm import SPARK_OPEN_MODEL, CallBudget, baseten_client, image_part, json_object, request_json
from benchproof.models import Procedure, SceneAssessment
from benchproof.predicates import check_assessment

SYSTEM_PROMPT = (
    "You inspect one overhead photo of a box for a packing check. Report only what is visible. "
    "Reply with one JSON object and nothing else."
)


class SceneJudge:
    def __init__(self, client: openai.OpenAI, model: str, budget: CallBudget, max_attempts: int = 2) -> None:
        self._client = client
        self._model = model
        self._budget = budget
        self._max_attempts = max_attempts

    def assess(self, procedure: Procedure, image_bytes: bytes) -> SceneAssessment:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [{"type": "text", "text": _instructions(procedure)}, image_part(image_bytes)],
            },
        ]

        def parse(content: str | None) -> SceneAssessment:
            scene = SceneAssessment.model_validate_json(json_object(content))
            check_assessment(procedure, scene)
            return scene

        return request_json(self._client, self._model, self._budget, messages, parse, self._max_attempts)


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


if __name__ == "__main__":
    # Manual check against the live model: python -m benchproof.scene path/to/photo.jpg
    kit = Procedure.model_validate_json(Path("procedures/mobile_desk_kit.json").read_text(encoding="utf-8"))
    budget = CallBudget(2)
    reading = SceneJudge(baseten_client(), SPARK_OPEN_MODEL, budget).assess(kit, Path(sys.argv[1]).read_bytes())
    sys.stdout.write(reading.model_dump_json(indent=2) + "\n")
    sys.stdout.write(f"model calls used: {budget.used}\n")
