"""Draft a procedure from a narrated demonstration.

The model proposes; the Procedure validator decides whether the draft is executable, and only the
author can approve it. A draft never carries approval fields.
"""

import json
import sys
from pathlib import Path

import openai

from benchproof.llm import SPARK_OPEN_MODEL, CallBudget, baseten_client, image_part, json_object, request_json
from benchproof.models import Procedure

# Muse Spark Open on Baseten accepts at most 8 images per request.
MAX_PHOTOS = 8
APPROVAL_FIELDS = frozenset({"approved_by", "approved_at"})

SYSTEM_PROMPT = (
    "You turn a person's narrated demonstration of a packing task into a checkable procedure. "
    "Reply with one JSON object and nothing else."
)

# A different kit from the one being drafted, so the model learns the format without copying content.
EXAMPLE_DRAFT = {
    "id": "first_aid_pouch",
    "revision": 1,
    "title": "Pack the first-aid pouch",
    "items": [
        {"id": "bandages", "name": "bandage roll", "sam_prompt": "white bandage roll"},
        {"id": "scissors", "name": "safety scissors", "sam_prompt": "small scissors"},
        {"id": "gloves", "name": "glove pack", "sam_prompt": "blue glove packet"},
    ],
    "zones": ["top", "bottom"],
    "box_prompt": "plastic tray",
    "divider_prompt": "plastic divider",
    "lid_prompt": "tray lid",
    "conditions": [
        {
            "id": "box_empty",
            "kind": "box_empty",
            "role": "observation",
            "description": "The open tray is empty.",
            "recovery": "Take everything out of the tray, then take a new photo.",
        },
        {
            "id": "divider_placed",
            "kind": "divider_in_place",
            "role": "invariant",
            "description": "The divider sits in its approved position.",
            "recovery": "Put the divider back in its approved position.",
        },
        {
            "id": "bandages_top",
            "kind": "item_in_zone",
            "role": "invariant",
            "item": "bandages",
            "zone": "top",
            "description": "The bandage roll is in the top compartment.",
            "recovery": "Put the bandage roll fully inside the top compartment.",
        },
        {
            "id": "scissors_top",
            "kind": "item_in_zone",
            "role": "invariant",
            "item": "scissors",
            "zone": "top",
            "description": "The scissors are in the top compartment.",
            "recovery": "Put the scissors fully inside the top compartment.",
        },
        {
            "id": "gloves_bottom",
            "kind": "item_in_zone",
            "role": "invariant",
            "item": "gloves",
            "zone": "bottom",
            "description": "The glove pack is in the bottom compartment.",
            "recovery": "Put the glove pack fully inside the bottom compartment.",
        },
        {
            "id": "contents_exact",
            "kind": "inventory_exact",
            "role": "invariant",
            "description": "The tray holds exactly one of each item and nothing else.",
            "recovery": "Remove extra objects and add anything missing.",
        },
    ],
    "steps": [
        {
            "id": "show_empty_tray",
            "title": "Show the empty tray",
            "action": "Place the open tray under the camera with nothing inside.",
            "conditions": ["box_empty"],
        },
        {
            "id": "place_divider",
            "title": "Place the divider",
            "action": "Place the divider so both compartments are visible.",
            "requires": ["show_empty_tray"],
            "conditions": ["divider_placed"],
        },
        {
            "id": "pack_top",
            "title": "Pack the top compartment",
            "action": "Place the bandage roll and the scissors in the top compartment.",
            "requires": ["place_divider"],
            "conditions": ["bandages_top", "scissors_top"],
        },
        {
            "id": "pack_bottom",
            "title": "Pack the bottom compartment",
            "action": "Place the glove pack in the bottom compartment.",
            "requires": ["place_divider"],
            "conditions": ["gloves_bottom"],
        },
        {
            "id": "inspect_contents",
            "title": "Inspect before closing",
            "action": "Keep the lid open and take one photo of the whole inside.",
            "requires": ["pack_top", "pack_bottom"],
            "conditions": ["contents_exact"],
            "final_inspection": True,
        },
    ],
}


class ProcedureDrafter:
    def __init__(self, client: openai.OpenAI, model: str, budget: CallBudget, max_attempts: int = 3) -> None:
        self._client = client
        self._model = model
        self._budget = budget
        self._max_attempts = max_attempts

    def draft(self, narration: str, photos: list[bytes]) -> Procedure:
        if not narration.strip():
            raise ValueError("the narration is empty; describe the task first")
        if len(photos) > MAX_PHOTOS:
            raise ValueError(f"use at most {MAX_PHOTOS} photos")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [{"type": "text", "text": _instructions(narration)}, *(image_part(p) for p in photos)],
            },
        ]
        return request_json(self._client, self._model, self._budget, messages, _parse, self._max_attempts)


def _parse(content: str | None) -> Procedure:
    data = json.loads(json_object(content))

    approval = sorted(APPROVAL_FIELDS & data.keys())
    if approval:
        raise ValueError(f"a draft must not include {approval}; only the author approves")

    # The application numbers revisions; a fresh draft is always revision 1.
    return Procedure.model_validate({**data, "revision": 1})


def _instructions(narration: str) -> str:
    schema = json.dumps(Procedure.model_json_schema(), separators=(",", ":"))
    return f"""BenchProof checks an open box, split by a divider into named zones, from overhead photos.
It can check only these condition kinds:
- box_empty (role "observation"): the open box is empty before packing.
- divider_in_place (role "invariant"): the divider stays where it was approved.
- item_in_zone (role "invariant", needs "item" and "zone"): one kit item lies inside one zone.
- inventory_exact (role "invariant"): the box holds exactly one of each kit item and nothing else.

Rules:
- Ids are lowercase letters, digits and underscores, starting with a letter.
- Item ids must not be box, divider or lid. No condition id may be closure.
- Every condition is checked by exactly one step.
- Exactly one step has "final_inspection": true. It checks the inventory_exact condition and requires,
  directly or through other steps, every other step.
- Add a "requires" entry only when the narration gives a reason for the order or the step needs the
  divider in place. Steps that can happen in either order must not require each other.
- sam_prompt, box_prompt, divider_prompt and lid_prompt are short noun phrases describing how the
  object looks, for an object detector.
- recovery tells the person what to do when the condition fails.
- Leave out approved_by and approved_at; the author approves the draft.

Example for a different kit:
{json.dumps(EXAMPLE_DRAFT)}

JSON schema:
{schema}

Narration:
{narration}

The photos, if any, show the demonstration in order."""


if __name__ == "__main__":
    # Manual check against the live model: python -m benchproof.drafting narration.txt [photo ...]
    budget = CallBudget(3)
    drafter = ProcedureDrafter(baseten_client(), SPARK_OPEN_MODEL, budget)
    draft = drafter.draft(Path(sys.argv[1]).read_text(encoding="utf-8"), [Path(p).read_bytes() for p in sys.argv[2:]])

    out = Path("drafts") / f"{draft.id}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(draft.model_dump_json(indent=2, exclude_none=True) + "\n", encoding="utf-8")
    sys.stdout.write(f"draft written to {out} ({len(draft.steps)} steps, {budget.used} model calls)\n")
