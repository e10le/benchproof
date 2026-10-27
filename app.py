from pathlib import Path

import streamlit as st
from PIL import UnidentifiedImageError

from benchproof.labels import SCENARIO_EXPECTED, Scenario, Split, load_labels, save_capture
from benchproof.models import Outcome, Procedure

ROOT = Path(__file__).parent
PROCEDURE_PATH = ROOT / "procedures" / "mobile_desk_kit.json"
CAPTURES_DIR = ROOT / "captures"
NO_STEP = "none"

procedure = Procedure.model_validate_json(PROCEDURE_PATH.read_text(encoding="utf-8"))

st.set_page_config(page_title="BenchProof labeling", layout="wide")
st.title("BenchProof: capture and label")
st.caption("Label what each photo truly shows before any model sees it.")

split = Split(st.sidebar.radio("Split", [s.value for s in Split]))
directory = CAPTURES_DIR / split.value
try:
    labels = load_labels(directory)
except (OSError, ValueError) as e:
    st.error(f"Could not load labels from {directory}: {e}. Check the file before continuing.")
    st.stop()

capture_col, label_col = st.columns(2)

with capture_col:
    source = st.radio("Image source", ["Webcam", "Upload"], horizontal=True)
    if source == "Webcam":
        image = st.camera_input("Photo from the fixed camera")
    else:
        image = st.file_uploader("Photo from the fixed camera", type=["png", "jpg", "jpeg"])

with label_col:
    scenario = Scenario(st.selectbox("Scenario", [s.value for s in Scenario]))
    sequence_id = st.text_input(
        "Sequence id",
        help="Photos replayed in order through one session share a sequence id, e.g. run_01.",
    )

    used = [lab.sequence_index for lab in labels if lab.sequence_id == sequence_id]
    sequence_index = st.number_input("Index in sequence", min_value=0, step=1, value=max(used, default=-1) + 1)

    step_titles = {s.id: s.title for s in procedure.steps}
    target = st.selectbox(
        "Target step",
        [NO_STEP] + list(step_titles),
        format_func=lambda sid: "None (closure or recheck)" if sid == NO_STEP else step_titles[sid],
    )

    outcomes = list(Outcome)
    expected = Outcome(
        st.selectbox(
            "Expected outcome",
            [o.value for o in outcomes],
            index=outcomes.index(SCENARIO_EXPECTED[scenario]),
        )
    )
    note = st.text_input("Note", help="Anything the label alone does not say, e.g. which item is covered.")

    if st.button("Save label", type="primary", disabled=image is None or not sequence_id):
        try:
            saved = save_capture(
                directory,
                image.getvalue(),
                split=split,
                scenario=scenario,
                sequence_id=sequence_id,
                sequence_index=int(sequence_index),
                target_step=None if target == NO_STEP else target,
                expected=expected,
                note=note,
            )
        except (ValueError, UnidentifiedImageError) as e:
            st.error(str(e))
        except OSError as e:
            st.error(f"Could not save the capture: {e}. Check the image, directory permissions and free disk space.")
        else:
            st.success(f"Saved {saved.capture_id}")
            labels = load_labels(directory)

st.subheader(f"Coverage: {split.value}")
counts = {s: {o: 0 for o in Outcome} for s in Scenario}
for lab in labels:
    counts[lab.scenario][lab.expected] += 1
st.dataframe(
    [{"scenario": s.value, **{o.value: n for o, n in by_outcome.items()}} for s, by_outcome in counts.items()],
    hide_index=True,
)

totals = {o: sum(by_outcome[o] for by_outcome in counts.values()) for o in Outcome}
for col, outcome in zip(st.columns(len(Outcome)), Outcome, strict=True):
    col.metric(outcome.value, totals[outcome])

st.subheader("Labels")
st.dataframe(
    [lab.model_dump(mode="json", exclude={"split", "image_file"}) for lab in reversed(labels)],
    hide_index=True,
)
