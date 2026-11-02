"""A checker wired to fake SAM and Muse Spark endpoints that answer from scripted scenes."""

import io
import json

from PIL import Image

from benchproof.checking import Capture, Checker
from benchproof.history import RunLog
from benchproof.llm import SPARK_OPEN_MODEL, CallBudget
from benchproof.scene import SceneJudge
from benchproof.session import Session
from endpoint import FakeEndpoint, reply
from sam_fake import FakeSam, line, record
from scenes import BOX, DIVIDER, HEIGHT, PACKED, WIDTH

FRAME = (WIDTH, HEIGHT)


def photo_bytes(shade: int) -> bytes:
    # Each photo differs by one pixel, because the session refuses to judge the same image twice.
    image = Image.new("RGB", FRAME, "white")
    image.putpixel((0, 0), (shade, shade, shade))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def reading(counts: dict[str, int]) -> tuple[int, dict]:
    full = {item: counts.get(item, 0) for item in PACKED}
    scene = {"interior_visibility": "clear", "lid_closed": False, "item_counts": full, "unexpected_items": []}
    return reply(json.dumps(scene))


class Bench:
    """A checker whose models answer from a scripted list of scenes, one scene per photo."""

    def __init__(self, procedure, tmp_path, scenes: list[dict[str, tuple[int, int, int, int]]]):
        self.procedure = procedure
        self._scenes = list(scenes)
        self._taken = 0
        self.sam = FakeSam({})
        self.judge = FakeEndpoint(*(reading({k: 1 for k in s if k in PACKED}) for s in scenes))
        budget = CallBudget(200)
        self.checker = Checker(
            Session(procedure, RunLog(tmp_path / "runs")),
            self.sam.segmenter(budget),
            SceneJudge(self.judge.client(), SPARK_OPEN_MODEL, budget),
            budget,
        )

    def capture(self) -> Capture:
        scene = self._scenes.pop(0)
        prompts = {"box": self.procedure.box_prompt, "divider": self.procedure.divider_prompt}
        prompts |= {item.id: item.sam_prompt for item in self.procedure.items}
        self.sam._replies = {prompts[label]: line(record(0, rect, FRAME)) for label, rect in scene.items()}
        self._taken += 1
        return self.checker.capture(photo_bytes(self._taken))


EMPTY = {"box": BOX}
WITH_DIVIDER = {"box": BOX, "divider": DIVIDER}
