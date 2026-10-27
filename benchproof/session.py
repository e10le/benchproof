"""Runtime state for one attempt at an approved procedure, in snapshot mode.

Every result is tied to the capture it came from. A result changes only when a new capture
arrives (or the author overrides the latest judgment, which is recorded as an override).
"""

import copy
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime

from pydantic import AwareDatetime

from benchproof.models import (
    CLOSURE_CONDITION_ID,
    CompletionRecord,
    ConditionResult,
    ConditionRole,
    Judgment,
    Outcome,
    Override,
    Procedure,
    Step,
    StepRecord,
    Visibility,
    ZoneCalibration,
    aggregate,
)
from benchproof.predicates import CaptureObservation, ClosureState, check_registration, closure_state, evaluate


@dataclass
class _Progress:
    passed_steps: dict[str, str] = field(default_factory=dict)
    invariants: dict[str, ConditionResult] = field(default_factory=dict)
    inspection: tuple[str, datetime] | None = None
    closure: tuple[str, datetime] | None = None


class Session:
    def __init__(self, procedure: Procedure) -> None:
        if procedure.approved_by is None:
            raise ValueError("the procedure must be approved by its author before a session starts")

        self._procedure = procedure
        self._calibration: ZoneCalibration | None = None
        self.judgments: list[Judgment] = []
        self._progress = _Progress()
        self._before_latest: _Progress | None = None
        # A check reads, judges and then updates progress; overlapping calls would mix two states.
        self._lock = threading.RLock()

    @property
    def procedure(self) -> Procedure:
        return self._procedure

    @property
    def calibration(self) -> ZoneCalibration | None:
        return self._calibration

    @property
    def completed(self) -> bool:
        return self._progress.closure is not None

    @property
    def ready_to_close(self) -> bool:
        return self._progress.inspection is not None and not self.completed

    @property
    def passed_steps(self) -> dict[str, str]:
        with self._lock:
            return dict(self._progress.passed_steps)

    @property
    def invariants(self) -> dict[str, ConditionResult]:
        with self._lock:
            return dict(self._progress.invariants)

    def available_steps(self) -> list[Step]:
        with self._lock:
            if self.completed:
                return []
            passed = self._progress.passed_steps
            return [s for s in self.procedure.steps if s.id not in passed and all(r in passed for r in s.requires)]

    def set_calibration(self, calibration: ZoneCalibration) -> None:
        with self._lock:
            if self.completed:
                raise ValueError("the session is already completed")
            if {z.id for z in calibration.zones} != set(self.procedure.zones):
                raise ValueError(f"calibration zones must be exactly {sorted(self.procedure.zones)}")
            if self.calibration is not None and calibration.revision <= self.calibration.revision:
                raise ValueError("a recalibration needs a higher revision")

            self._calibration = calibration

            # Zone evidence gathered under the old calibration no longer counts toward closure.
            self._revoke_inspection()
            self._before_latest = None

    def check(self, obs: CaptureObservation, target_step: str | None) -> Judgment:
        """Judge a fresh capture, targeting one available step or (None) only re-checking invariants."""
        with self._lock:
            if self.completed:
                raise ValueError("the session is already completed")
            if any(j.capture_id == obs.capture_id for j in self.judgments):
                raise ValueError(f"capture {obs.capture_id} was already judged; take a new photo")
            if any(j.image_sha256 == obs.image_sha256 for j in self.judgments):
                raise ValueError("this image was already judged; take a new photo")
            if self.judgments and obs.sequence <= self.judgments[-1].capture_sequence:
                raise ValueError("this photo was taken before the last judged photo; take a new photo")
            if target_step is not None and target_step not in {s.id for s in self.available_steps()}:
                raise ValueError(f"step {target_step} is not available")

            judgment = self._judge(obs, target_step)
            if (
                self.ready_to_close
                and closure_state(obs) is ClosureState.OPEN
                and obs.scene.interior_visibility is Visibility.CLEAR
                and obs.largest("box") is not None
                and obs.largest("divider") is not None
                and check_registration(self.calibration, obs) is not None
            ):
                self._revoke_inspection()
            snapshot = copy.deepcopy(self._progress)
            self._apply(judgment)
            self._before_latest = snapshot
            self.judgments.append(judgment)
            return judgment

    def override(self, condition_id: str, outcome: Outcome, by: str, reason: str, at: AwareDatetime) -> Judgment:
        """Replace one result of the latest judgment with the author's call, marked as an override."""
        with self._lock:
            if self.completed:
                raise ValueError("the session is already completed")
            if self._before_latest is None:
                raise ValueError("only the latest judgment of this calibration can be overridden")
            if condition_id == CLOSURE_CONDITION_ID:
                raise ValueError("closure cannot be overridden")

            latest = self.judgments[-1]
            original = latest.result(condition_id)
            if original.override is not None:
                raise ValueError(f"{condition_id} was already overridden; take a new photo")
            if original.outcome is outcome:
                raise ValueError(f"{condition_id} is already {outcome}")

            replaced = ConditionResult(
                condition_id=condition_id,
                outcome=outcome,
                reasons=original.reasons + (f"Overridden by {by}: {reason}",),
                instruction=None if outcome is Outcome.PASS else reason,
                override=Override(original=original.outcome, by=by, reason=reason, at=at),
            )
            results = tuple(replaced if r.condition_id == condition_id else r for r in latest.results)
            judgment = Judgment.model_validate(
                {**latest.model_dump(), "results": results, "outcome": aggregate(r.outcome for r in results)}
            )

            self._progress = copy.deepcopy(self._before_latest)
            self._apply(judgment)
            self.judgments[-1] = judgment
            return judgment

    def record(self) -> CompletionRecord:
        with self._lock:
            if not self.completed:
                raise ValueError("the session is not completed")

            inspection_id, inspected_at = self._progress.inspection
            closure_id, closed_at = self._progress.closure
            return CompletionRecord(
                procedure_id=self.procedure.id,
                procedure_revision=self.procedure.revision,
                calibration_revision=self.calibration.revision,
                steps=tuple(StepRecord(step_id=s, capture_id=c) for s, c in self._progress.passed_steps.items()),
                inspection_capture_id=inspection_id,
                inspected_at=inspected_at,
                closure_capture_id=closure_id,
                closed_at=closed_at,
                judgments=tuple(self.judgments),
            )

    def _judge(self, obs: CaptureObservation, target_step: str | None) -> Judgment:
        state = closure_state(obs)
        if state is ClosureState.OPEN:
            results = self._open_box_results(obs, target_step)
        else:
            results = [self._closure_result(state)]

        return Judgment(
            capture_id=obs.capture_id,
            capture_sequence=obs.sequence,
            image_sha256=obs.image_sha256,
            target_step=target_step,
            captured_at=obs.captured_at,
            judged_at=datetime.now(UTC),
            perception_source=obs.source,
            procedure_revision=self.procedure.revision,
            calibration_revision=self.calibration.revision if self.calibration else None,
            results=tuple(results),
            outcome=aggregate(r.outcome for r in results),
        )

    def _open_box_results(self, obs: CaptureObservation, target_step: str | None) -> list[ConditionResult]:
        condition_ids: list[str] = []
        if target_step is not None:
            condition_ids += self.procedure.step(target_step).conditions

        condition_ids += [cid for cid in self._progress.invariants if cid not in condition_ids]
        if not condition_ids:
            raise ValueError("nothing to check: choose an available step")

        return evaluate(self.procedure, self.calibration, obs, condition_ids)

    def _closure_result(self, state: ClosureState) -> ConditionResult:
        if state is ClosureState.UNCLEAR:
            return ConditionResult(
                condition_id=CLOSURE_CONDITION_ID,
                outcome=Outcome.INSUFFICIENT,
                reasons=("Cannot tell whether the box is closed",),
                instruction="Take a new photo with nothing covering the box.",
            )
        if self._progress.inspection is None:
            return ConditionResult(
                condition_id=CLOSURE_CONDITION_ID,
                outcome=Outcome.FAIL,
                reasons=("Box closed before the final inspection passed",),
                instruction="Reopen the box so the contents can be inspected.",
            )
        return ConditionResult(
            condition_id=CLOSURE_CONDITION_ID,
            outcome=Outcome.PASS,
            reasons=("Box closed after the final inspection passed",),
        )

    def _apply(self, judgment: Judgment) -> None:
        progress = self._progress
        results = {r.condition_id: r for r in judgment.results}

        closure = results.get(CLOSURE_CONDITION_ID)
        if closure is not None:
            if closure.outcome is Outcome.PASS:
                progress.closure = (judgment.capture_id, judgment.captured_at)
            return

        for cid in progress.invariants:
            if cid in results:
                progress.invariants[cid] = results[cid]

        # A visible change after inspection voids it; an obscured view proves nothing either way.
        if any(r.outcome is Outcome.FAIL for r in progress.invariants.values()):
            self._revoke_inspection()

        if judgment.target_step is None or judgment.outcome is not Outcome.PASS:
            return

        step = self.procedure.step(judgment.target_step)
        progress.passed_steps[step.id] = judgment.capture_id
        for cid in step.conditions:
            if self.procedure.condition(cid).role is ConditionRole.INVARIANT:
                progress.invariants[cid] = results[cid]
        if step.final_inspection:
            progress.inspection = (judgment.capture_id, judgment.captured_at)

    def _revoke_inspection(self) -> None:
        self._progress.inspection = None
        self._progress.passed_steps.pop(self.procedure.final_step.id, None)
