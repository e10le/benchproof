"""Run both perception models on one photo and hand the engine a CaptureObservation."""

import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

from benchproof.llm import SPARK_OPEN_MODEL, CallBudget, baseten_client, meta_client
from benchproof.models import Procedure
from benchproof.photos import Photo, prepare
from benchproof.predicates import CaptureObservation
from benchproof.scene import SceneJudge
from benchproof.segmenter import SAM_MODEL, Segmenter


def perceive(
    procedure: Procedure,
    photo: Photo,
    capture_id: str,
    sequence: int,
    captured_at: datetime,
    segmenter: Segmenter,
    judge: SceneJudge,
) -> CaptureObservation:
    # The scene judge reasons for many seconds, so SAM runs alongside it rather than after it.
    with ThreadPoolExecutor(max_workers=2) as pool:
        scene = pool.submit(judge.assess, procedure, photo.jpeg)
        detections = pool.submit(segmenter.segment_all, procedure, photo)
        return CaptureObservation(
            capture_id=capture_id,
            sequence=sequence,
            image_sha256=photo.sha256,
            captured_at=captured_at,
            width=photo.width,
            height=photo.height,
            scene=scene.result(),
            detections=detections.result(),
            source=f"{SAM_MODEL} + {SPARK_OPEN_MODEL}",
        )


if __name__ == "__main__":
    # Manual check against both live models: python -m benchproof.perceive path/to/photo.jpg
    kit = Procedure.model_validate_json(Path("procedures/mobile_desk_kit.json").read_text(encoding="utf-8"))
    budget = CallBudget(12)
    obs = perceive(
        kit,
        prepare(Path(sys.argv[1]).read_bytes()),
        capture_id="manual",
        sequence=0,
        captured_at=datetime.now(UTC),
        segmenter=Segmenter(meta_client(), budget),
        judge=SceneJudge(baseten_client(), SPARK_OPEN_MODEL, budget),
    )

    sys.stdout.write(f"photo: {obs.width} x {obs.height}\n")
    sys.stdout.write(f"scene: {obs.scene.model_dump_json()}\n")
    for label in ("box", "divider", "lid", *(item.id for item in kit.items)):
        found = [d for d in obs.detections if d.label == label]
        scores = ", ".join(f"{d.score:.2f}" for d in found)
        sys.stdout.write(f"{label:>8}: {len(found)} found  {scores}\n")
    sys.stdout.write(f"model calls used: {budget.used}\n")
