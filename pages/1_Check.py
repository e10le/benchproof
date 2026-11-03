"""Check page: take a photo, let SAM 3.1 and Muse Spark look at it, and see pass, fail or insufficient."""

from datetime import UTC, datetime
from pathlib import Path

import streamlit as st
from PIL import UnidentifiedImageError

from benchproof import checking
from benchproof.drawing import jpeg, overlay
from benchproof.llm import ModelError
from benchproof.models import Judgment, Outcome, Procedure, Zone
from benchproof.photos import CAMERA_RESOLUTION, Photo

ROOT = Path(__file__).parent.parent
PROCEDURE_PATH = ROOT / "procedures" / "mobile_desk_kit.json"
RUNS_DIR = ROOT / "runs"
DIVIDER_STEP = "place_divider"
CLOSE = "close"

SHOW = {Outcome.PASS: st.success, Outcome.FAIL: st.error, Outcome.INSUFFICIENT: st.warning}
MARK = {Outcome.PASS: "✅", Outcome.FAIL: "❌", Outcome.INSUFFICIENT: "⚠️"}

# The kit file is read only to start a session; afterwards the session's approved copy is the one source.
KIT = Procedure.model_validate_json(PROCEDURE_PATH.read_text(encoding="utf-8"))
state = st.session_state
state.setdefault("shots", 0)

st.set_page_config(page_title="BenchProof check", layout="wide")
st.title("BenchProof: check")


def picture(photo: Photo, detections, zones: tuple[Zone, ...]) -> bytes:
    return jpeg(overlay(photo, list(detections), zones, named=True))


def show_judgment(procedure: Procedure, judgment: Judgment, image: bytes) -> None:
    image_col, result_col = st.columns(2)
    image_col.image(image, caption=f"Photo {judgment.capture_sequence + 1}: what the models found")
    with result_col:
        step = procedure.step(judgment.target_step).title if judgment.target_step else "Re-check / close"
        SHOW[judgment.outcome](f"{step}: {judgment.outcome.value.upper()}")
        if judgment.recovery:
            st.markdown("**To fix:**\n" + "\n".join(f"- {line}" for line in judgment.recovery))
        for r in judgment.results:
            marked = " (overridden)" if r.override else ""
            st.markdown(
                f"{MARK[r.outcome]} **{procedure.condition(r.condition_id).description}**{marked}  \n"
                + "; ".join(r.reasons)
            )


def start_form() -> None:
    st.write(f"**{KIT.title}** (revision {KIT.revision})")
    for number, step in enumerate(KIT.steps, start=1):
        st.write(f"{number}. {step.title}: {step.action}")

    with st.form("start"):
        author = st.text_input("Your name", help="You approve this procedure and, later, the zones.")
        if not st.form_submit_button("Start session", type="primary"):
            return

    if not author.strip():
        st.error("Enter your name to approve the procedure.")
        return
    try:
        state.checker = checking.start(KIT, author.strip(), RUNS_DIR)
    except (ValueError, OSError) as e:
        st.error(f"Could not start a session: {e}")
        return
    state.author = author.strip()
    st.rerun()


def sidebar(checker: checking.Checker) -> None:
    session = checker.session
    st.sidebar.write(f"Approved by **{state.author}**")
    st.sidebar.write(f"Model calls: {checker.budget.used} of {checker.budget.limit}")
    st.sidebar.write("Steps passed:")
    for step in session.procedure.steps:
        st.sidebar.write(("✅ " if step.id in session.passed_steps else "⬜ ") + step.title)
    st.sidebar.caption(f"Run log: {session.log_path}")

    if session.completed or session.cancelled:
        if st.sidebar.button("New session"):
            for key in ("checker", "pending", "last"):
                state.pop(key, None)
            st.rerun()
    elif st.sidebar.button("Cancel session"):
        session.cancel(state.author, "cancelled from the check page")
        st.rerun()


def approve_zones(checker: checking.Checker) -> None:
    capture, proposed = state.pending
    photo = capture.photo
    st.subheader("Approve the zones")
    st.write("The yellow outlines are the compartments every later photo is checked against. Adjust them if needed.")

    zones = []
    with st.expander("Adjust zones"):
        for zone in proposed:
            xs, ys = [x for x, _ in zone.polygon], [y for _, y in zone.polygon]
            x0, x1 = st.slider(f"{zone.id}: left and right", 0, photo.width, (round(min(xs)), round(max(xs))))
            y0, y1 = st.slider(f"{zone.id}: top and bottom", 0, photo.height, (round(min(ys)), round(max(ys))))
            zones.append(zone.model_copy(update={"polygon": checking.rectangle(x0, y0, x1, y1)}))
    # Slider values are rectangles; trim them to the box outline as the proposal was.
    try:
        zones = checking.fit_to_box(capture.observation, tuple(zones))
        problem = None
    except ValueError as e:
        zones, problem = (), str(e)

    st.image(picture(photo, capture.observation.detections, zones), caption="Proposed zones")
    if problem:
        st.error(f"These zones cannot be used: {problem}")
    approve_col, retake_col = st.columns(2)
    if retake_col.button("Retake the photo"):
        del state.pending
        st.rerun()
    if not approve_col.button("Approve zones and check", type="primary", disabled=problem is not None):
        return
    try:
        calibration = checker.calibrate(capture, zones, state.author, datetime.now(UTC))
        judgment = checker.check(capture, DIVIDER_STEP)
    except ValueError as e:
        st.error(f"Zones not accepted: {e}")
        return
    del state.pending
    state.last = (judgment, picture(photo, capture.observation.detections, calibration.zones))
    st.rerun()


def check_photo(checker: checking.Checker, image: bytes, target: str | None) -> None:
    state.shots += 1
    try:
        with st.spinner("SAM 3.1 is finding objects and Muse Spark is reading the scene..."):
            capture = checker.capture(image)
    except (ModelError, ValueError, UnidentifiedImageError) as e:
        st.error(f"Could not check this photo: {e}")
        return

    session = checker.session
    if target == DIVIDER_STEP and session.calibration is None:
        try:
            state.pending = (capture, checking.propose_zones(capture.observation, session.procedure.zones))
        except ValueError as e:
            st.error(f"Could not propose zones: {e}. Retake the photo with the divider clearly visible.")
            st.image(picture(capture.photo, capture.observation.detections, ()), caption="What the models found")
            return
        st.rerun()

    try:
        judgment = checker.check(capture, target)
    except ValueError as e:
        st.error(f"Could not judge this photo: {e}")
        return
    zones = session.calibration.zones if session.calibration else ()
    state.last = (judgment, picture(capture.photo, capture.observation.detections, zones))
    st.rerun()


def main() -> None:
    checker: checking.Checker | None = state.get("checker")
    if checker is None:
        start_form()
        return

    sidebar(checker)
    session = checker.session
    procedure = session.procedure

    # The latest result comes first so it is in view right after a check, above the camera.
    if "last" in state and "pending" not in state:
        st.subheader("Latest result")
        show_judgment(procedure, *state.last)

    if session.completed:
        st.success("Box closed. Contents were observed before closure.")
        record = session.record().model_dump_json(indent=2)
        st.download_button("Download the evidence record", record, file_name="benchproof_record.json")
    elif session.cancelled:
        st.warning("This session was cancelled. Start a new one from the sidebar.")
    elif "pending" in state:
        approve_zones(checker)
    else:
        available = [s.id for s in session.available_steps()]
        titles = {s.id: s.title for s in procedure.steps}
        closing = "Close the box (lid on)" if session.ready_to_close else "Re-check without a step"
        target = st.radio(
            "What are you checking?",
            [*available, CLOSE],
            format_func=lambda sid: closing if sid == CLOSE else titles[sid],
        )
        if target != CLOSE:
            st.info(procedure.step(target).action)

        source = st.radio("Image source", ["Webcam", "Upload"], horizontal=True)
        key = f"photo-{state.shots}"
        if source == "Webcam":
            image = st.camera_input("Photo from the fixed camera", key=key, resolution=CAMERA_RESOLUTION)
        else:
            image = st.file_uploader("Photo from the fixed camera", type=["png", "jpg", "jpeg", "webp"], key=key)

        if st.button("Check photo", type="primary", disabled=image is None):
            check_photo(checker, image.getvalue(), None if target == CLOSE else target)


main()
