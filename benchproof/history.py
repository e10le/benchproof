"""Append-only run history for one session.

Each batch of events is synced to disk before the session applies the change it describes, so a
crash leaves a request without an outcome rather than an outcome without a record.
"""

import os
import threading
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from uuid import uuid4

from pydantic import Field, model_validator

from benchproof.jsonl import read_lines
from benchproof.models import Judgment, Procedure, StrictModel, UtcDatetime, ZoneCalibration


class EventKind(StrEnum):
    SESSION_STARTED = "session_started"
    CALIBRATION_SET = "calibration_set"
    CALIBRATION_REJECTED = "calibration_rejected"
    CHECK_REQUESTED = "check_requested"
    CHECK_REJECTED = "check_rejected"
    CHECK_JUDGED = "check_judged"
    INSPECTION_REVOKED = "inspection_revoked"
    OVERRIDE_APPLIED = "override_applied"
    OVERRIDE_REJECTED = "override_rejected"
    SESSION_COMPLETED = "session_completed"
    SESSION_CANCELLED = "session_cancelled"


_REQUIRED_FIELDS = {
    EventKind.SESSION_STARTED: ("procedure",),
    EventKind.CALIBRATION_SET: ("calibration",),
    EventKind.CALIBRATION_REJECTED: ("calibration", "reason"),
    EventKind.CHECK_REQUESTED: ("capture_id", "capture_sequence"),
    EventKind.CHECK_REJECTED: ("capture_id", "reason"),
    EventKind.CHECK_JUDGED: ("judgment",),
    EventKind.INSPECTION_REVOKED: ("reason",),
    EventKind.OVERRIDE_APPLIED: ("condition_id", "by", "reason", "judgment"),
    EventKind.OVERRIDE_REJECTED: ("condition_id", "by", "reason"),
    EventKind.SESSION_COMPLETED: ("capture_id",),
    EventKind.SESSION_CANCELLED: ("by", "reason"),
}


class Event(StrictModel):
    kind: EventKind
    capture_id: str | None = None
    capture_sequence: int | None = Field(default=None, ge=0)
    target_step: str | None = None
    condition_id: str | None = None
    by: str | None = None
    reason: str | None = None
    procedure: Procedure | None = None
    calibration: ZoneCalibration | None = None
    judgment: Judgment | None = None

    @model_validator(mode="after")
    def _check_required(self) -> "Event":
        missing = [name for name in _REQUIRED_FIELDS[self.kind] if getattr(self, name) in (None, "")]
        if missing:
            raise ValueError(f"{self.kind} event needs {missing}")
        return self


class LogRecord(Event):
    session_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    # Order comes from index; the wall-clock time is audit metadata only.
    index: int = Field(ge=0)
    at: UtcDatetime


class RunLog:
    def __init__(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.session_id = uuid4().hex
        self.path = directory / f"{self.session_id}.jsonl"
        self.path.touch(exist_ok=False)
        self._count = 0
        self._failed = False
        # Index assignment and the write must happen together, or two writers could both take index 0.
        self._lock = threading.Lock()

    def write(self, *events: Event) -> None:
        with self._lock:
            self._write(events)

    def _write(self, events: tuple[Event, ...]) -> None:
        if self._failed:
            raise ValueError("an earlier run log write failed; start a new session")

        at = datetime.now(UTC)
        records = [
            LogRecord.model_validate(
                {**event.model_dump(), "session_id": self.session_id, "index": self._count + offset, "at": at}
            )
            for offset, event in enumerate(events)
        ]
        for record in records:
            if (record.index == 0) != (record.kind is EventKind.SESSION_STARTED):
                raise ValueError("a run log starts with exactly one session_started event")

        text = "".join(record.model_dump_json() + "\n" for record in records)
        try:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(text)
                f.flush()
                os.fsync(f.fileno())
        except OSError:
            # Part of the batch may be on disk, so nothing written after it could be trusted.
            self._failed = True
            raise
        self._count += len(records)


def load_run_log(path: Path) -> list[LogRecord]:
    records = [LogRecord.model_validate_json(line) for line in read_lines(path)]

    for position, record in enumerate(records):
        if record.index != position or record.session_id != records[0].session_id:
            raise ValueError(f"{path} is out of order or mixes sessions at line {position + 1}")
    return records
