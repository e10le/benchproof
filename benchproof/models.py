from collections.abc import Iterable
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Literal, get_args

from pydantic import AfterValidator, AwareDatetime, BaseModel, ConfigDict, Field, NonNegativeInt, model_validator

from benchproof import geometry

ID_PATTERN = r"^[a-z][a-z0-9_]*$"
IMAGE_SHA256_PATTERN = r"^[0-9a-f]{64}$"

# SAM prompts for these scene objects live on the Procedure, so item ids must not collide with them.
RESERVED_LABELS = frozenset({"box", "divider", "lid"})

CLOSURE_CONDITION_ID = "closure"

CompletionClaim = Literal["contents observed before closure"]
COMPLETION_CLAIM: str = get_args(CompletionClaim)[0]


class Outcome(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    INSUFFICIENT = "insufficient"


class ConditionKind(StrEnum):
    BOX_EMPTY = "box_empty"
    DIVIDER_IN_PLACE = "divider_in_place"
    ITEM_IN_ZONE = "item_in_zone"
    INVENTORY_EXACT = "inventory_exact"


class ConditionRole(StrEnum):
    OBSERVATION = "observation"
    INVARIANT = "invariant"


class Visibility(StrEnum):
    CLEAR = "clear"
    PARTIAL = "partial"
    NONE = "none"


def _to_utc(value: datetime) -> datetime:
    return value.astimezone(UTC)


UtcDatetime = Annotated[AwareDatetime, AfterValidator(_to_utc)]


class StrictModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


def aggregate(outcomes: Iterable[Outcome]) -> Outcome:
    """Combine condition outcomes: any fail wins, then any insufficient, else pass.

    An empty input is rejected because "nothing was checked" must never read as a pass.
    """
    collected = list(outcomes)
    if not collected:
        raise ValueError("cannot aggregate zero outcomes")

    if Outcome.FAIL in collected:
        return Outcome.FAIL
    if Outcome.INSUFFICIENT in collected:
        return Outcome.INSUFFICIENT
    return Outcome.PASS


# --- Procedure ---------------------------------------------------------------


class Item(StrictModel):
    id: str = Field(pattern=ID_PATTERN)
    name: str = Field(min_length=1)
    sam_prompt: str = Field(min_length=1)


class Condition(StrictModel):
    id: str = Field(pattern=ID_PATTERN)
    kind: ConditionKind
    role: ConditionRole
    description: str = Field(min_length=1)
    recovery: str = Field(min_length=1)
    item: str | None = None
    zone: str | None = None

    @model_validator(mode="after")
    def _check_kind_fields(self) -> "Condition":
        is_placement = self.kind is ConditionKind.ITEM_IN_ZONE
        if is_placement != (self.item is not None):
            raise ValueError(f"{self.id}: item is required for item_in_zone and not allowed otherwise")
        if is_placement != (self.zone is not None):
            raise ValueError(f"{self.id}: zone is required for item_in_zone and not allowed otherwise")

        # An empty box stops being true as soon as packing starts, so it can only be a one-time observation.
        if self.kind is ConditionKind.BOX_EMPTY and self.role is not ConditionRole.OBSERVATION:
            raise ValueError(f"{self.id}: box_empty must be an observation")
        if self.kind is not ConditionKind.BOX_EMPTY and self.role is not ConditionRole.INVARIANT:
            raise ValueError(f"{self.id}: {self.kind} must be an invariant")
        return self


class Step(StrictModel):
    id: str = Field(pattern=ID_PATTERN)
    title: str = Field(min_length=1)
    action: str = Field(min_length=1)
    requires: tuple[str, ...] = ()
    conditions: tuple[str, ...] = Field(min_length=1)
    final_inspection: bool = False


class Procedure(StrictModel):
    id: str = Field(pattern=ID_PATTERN)
    revision: int = Field(ge=1)
    title: str = Field(min_length=1)
    items: tuple[Item, ...] = Field(min_length=1)
    zones: tuple[Annotated[str, Field(pattern=ID_PATTERN)], ...] = Field(min_length=1)
    box_prompt: str = Field(min_length=1)
    divider_prompt: str = Field(min_length=1)
    lid_prompt: str = Field(min_length=1)
    conditions: tuple[Condition, ...] = Field(min_length=1)
    steps: tuple[Step, ...] = Field(min_length=1)
    approved_by: str | None = None
    approved_at: UtcDatetime | None = None

    @model_validator(mode="after")
    def _check_references(self) -> "Procedure":
        item_ids = _unique_ids("item", [i.id for i in self.items])
        zone_ids = _unique_ids("zone", list(self.zones))
        condition_ids = _unique_ids("condition", [c.id for c in self.conditions])
        _unique_ids("step", [s.id for s in self.steps])

        reserved = item_ids & RESERVED_LABELS
        if reserved:
            raise ValueError(f"item ids {sorted(reserved)} are reserved for scene objects")
        if CLOSURE_CONDITION_ID in condition_ids:
            raise ValueError(f"condition id '{CLOSURE_CONDITION_ID}' is reserved")

        for c in self.conditions:
            if c.item is not None and c.item not in item_ids:
                raise ValueError(f"{c.id}: unknown item '{c.item}'")
            if c.zone is not None and c.zone not in zone_ids:
                raise ValueError(f"{c.id}: unknown zone '{c.zone}'")

        used: dict[str, str] = {}
        for s in self.steps:
            for cid in s.conditions:
                if cid not in condition_ids:
                    raise ValueError(f"{s.id}: unknown condition '{cid}'")
                if cid in used:
                    raise ValueError(f"condition '{cid}' is used by both {used[cid]} and {s.id}")
                used[cid] = s.id

        unused = condition_ids - used.keys()
        if unused:
            raise ValueError(f"conditions {sorted(unused)} are not checked by any step")

        if (self.approved_by is None) != (self.approved_at is None):
            raise ValueError("approved_by and approved_at must be set together")
        if self.approved_by is not None and not self.approved_by.strip():
            raise ValueError("an approving author is required")

        self._check_graph()
        return self

    def _check_graph(self) -> None:
        step_ids = {s.id for s in self.steps}
        for s in self.steps:
            unknown = set(s.requires) - step_ids
            if unknown:
                raise ValueError(f"{s.id}: unknown prerequisites {sorted(unknown)}")

        finals = [s for s in self.steps if s.final_inspection]
        if len(finals) != 1:
            raise ValueError(f"exactly one final inspection step is required, found {len(finals)}")

        ancestors = {s.id: self._ancestors(s.id, path=()) for s in self.steps}

        # The final inspection gates closure, so every other step has to be finished before it.
        final = finals[0]
        missing = step_ids - {final.id} - ancestors[final.id]
        if missing:
            raise ValueError(f"final inspection does not depend on steps {sorted(missing)}")
        if not any(self.condition(cid).kind is ConditionKind.INVENTORY_EXACT for cid in final.conditions):
            raise ValueError("final inspection must check an exact inventory condition")

    def _ancestors(self, step_id: str, path: tuple[str, ...]) -> set[str]:
        if step_id in path:
            cycle = " -> ".join(path + (step_id,))
            raise ValueError(f"prerequisite cycle: {cycle}")

        found: set[str] = set()
        for parent in self.step(step_id).requires:
            found.add(parent)
            found |= self._ancestors(parent, path + (step_id,))
        return found

    def item(self, item_id: str) -> Item:
        for i in self.items:
            if i.id == item_id:
                return i
        raise KeyError(item_id)

    def condition(self, condition_id: str) -> Condition:
        for c in self.conditions:
            if c.id == condition_id:
                return c
        raise KeyError(condition_id)

    def step(self, step_id: str) -> Step:
        for s in self.steps:
            if s.id == step_id:
                return s
        raise KeyError(step_id)

    @property
    def final_step(self) -> Step:
        return next(s for s in self.steps if s.final_inspection)

    def approve(self, author: str, at: AwareDatetime) -> "Procedure":
        return self.model_validate({**self.model_dump(), "approved_by": author, "approved_at": at})


def _unique_ids(kind: str, ids: list[str]) -> set[str]:
    unique = set(ids)
    if len(unique) != len(ids):
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"duplicate {kind} ids: {duplicates}")
    return unique


# --- Zone calibration --------------------------------------------------------


class BBox(StrictModel):
    x: float
    y: float
    w: float = Field(gt=0)
    h: float = Field(gt=0)

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.w / 2, self.y + self.h / 2)

    @property
    def diagonal(self) -> float:
        return (self.w**2 + self.h**2) ** 0.5


class Zone(StrictModel):
    id: str = Field(pattern=ID_PATTERN)
    polygon: tuple[tuple[float, float], ...] = Field(min_length=3)

    @model_validator(mode="after")
    def _check_polygon(self) -> "Zone":
        if not geometry.polygon_is_simple(self.polygon):
            raise ValueError("zone polygon must have nonzero area and no repeated or crossing vertices")
        return self


class ZoneCalibration(StrictModel):
    revision: int = Field(ge=1)
    reference_capture_id: str = Field(min_length=1)
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    box: BBox
    box_angle_deg: float
    box_aspect_ratio: float = Field(ge=1)
    divider_center: tuple[float, float]
    divider: BBox
    divider_angle_deg: float
    divider_aspect_ratio: float = Field(ge=1)
    zones: tuple[Zone, ...] = Field(min_length=1)

    # Starting tolerances only. Replace them with values measured on real captures from the fixed camera.
    max_box_shift: float = Field(default=0.03, gt=0)
    max_box_scale_change: float = Field(default=0.05, gt=0)
    max_angle_deg: float = Field(default=3.0, gt=0)
    max_divider_shift: float = Field(default=0.03, gt=0)
    max_divider_scale_change: float = Field(default=0.05, gt=0)

    approved_by: str = Field(min_length=1)
    approved_at: UtcDatetime

    @model_validator(mode="after")
    def _check_zones(self) -> "ZoneCalibration":
        _unique_ids("zone", [z.id for z in self.zones])

        for name, box in (("box", self.box), ("divider", self.divider)):
            if box.x < 0 or box.y < 0 or box.x + box.w > self.image_width or box.y + box.h > self.image_height:
                raise ValueError(f"{name} geometry must fit inside the image")

        for z in self.zones:
            for x, y in z.polygon:
                if not (0 <= x <= self.image_width and 0 <= y <= self.image_height):
                    raise ValueError(f"zone '{z.id}' has a point outside the image: ({x}, {y})")
                if not (self.box.x <= x < self.box.x + self.box.w and self.box.y <= y < self.box.y + self.box.h):
                    raise ValueError(f"zone '{z.id}' must be inside the box")
        for i, zone in enumerate(self.zones):
            for other in self.zones[i + 1 :]:
                if geometry.polygons_overlap(zone.polygon, other.polygon):
                    raise ValueError(f"zones '{zone.id}' and '{other.id}' overlap or touch")
        return self

    def zone(self, zone_id: str) -> Zone:
        for z in self.zones:
            if z.id == zone_id:
                return z
        raise KeyError(zone_id)


# --- Perception and judgments -------------------------------------------------


class SceneAssessment(StrictModel):
    """Whole-scene reading of one capture from a vision model.

    This is also the structured-output schema the judge model fills in, so every field here is
    something the model reports, never audit metadata the application supplies.
    """

    interior_visibility: Visibility
    lid_closed: bool
    item_counts: dict[str, Annotated[NonNegativeInt, Field(strict=True)]]
    unexpected_items: tuple[str, ...] = ()


class Override(StrictModel):
    original: Outcome
    by: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    at: UtcDatetime


class ConditionResult(StrictModel):
    condition_id: str = Field(min_length=1)
    outcome: Outcome
    reasons: tuple[str, ...] = Field(min_length=1)
    instruction: str | None = None
    override: Override | None = None

    @model_validator(mode="after")
    def _check_instruction(self) -> "ConditionResult":
        if self.outcome is not Outcome.PASS and not self.instruction:
            raise ValueError(f"{self.condition_id}: a {self.outcome} result needs an instruction")
        return self


class Judgment(StrictModel):
    capture_id: str = Field(min_length=1)
    capture_sequence: int = Field(ge=0)
    image_sha256: str = Field(pattern=IMAGE_SHA256_PATTERN)
    target_step: str | None
    captured_at: UtcDatetime
    judged_at: UtcDatetime
    perception_source: str = Field(min_length=1)
    procedure_revision: int = Field(ge=1)
    calibration_revision: int | None
    results: tuple[ConditionResult, ...] = Field(min_length=1)
    outcome: Outcome

    @model_validator(mode="after")
    def _check_outcome(self) -> "Judgment":
        expected = aggregate(r.outcome for r in self.results)
        if self.outcome is not expected:
            raise ValueError(f"outcome {self.outcome} does not match results ({expected})")
        return self

    @property
    def recovery(self) -> tuple[str, ...]:
        steps: list[str] = []
        for r in self.results:
            if r.instruction and r.instruction not in steps:
                steps.append(r.instruction)
        return tuple(steps)

    def result(self, condition_id: str) -> ConditionResult:
        for r in self.results:
            if r.condition_id == condition_id:
                return r
        raise KeyError(condition_id)


class StepRecord(StrictModel):
    step_id: str
    capture_id: str


class CompletionRecord(StrictModel):
    claim: CompletionClaim = COMPLETION_CLAIM
    procedure_id: str
    procedure_revision: int
    calibration_revision: int
    steps: tuple[StepRecord, ...]
    inspection_capture_id: str
    inspected_at: UtcDatetime
    closure_capture_id: str
    closed_at: UtcDatetime
    judgments: tuple[Judgment, ...]

    @property
    def overridden(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            (j.capture_id, r.condition_id) for j in self.judgments for r in j.results if r.override is not None
        )
