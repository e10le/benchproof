# BenchProof - agreed design decisions

Design document, revised October 26, 2026. Implementation started October 26 (offline core; no billable inference yet). See "Implementation status" at the end.

## Scope

- Target platform: Windows with Python 3.12. Use hosted model inference; local model installation is outside the initial scope.
- Code follows coding-rules.md (in this project).
- Stack: Python, Streamlit, Pydantic, OpenCV/Pillow. Repository name: benchproof. License for original application code: MIT; preserve separate model and dependency notices. If Baseten Spark Open is used, the README carries the Muse Model License attribution ("Built with Muse Spark Open").
- Track: Indie, solo.
- The baseline is uploaded recordings and explicit webcam snapshots (st.camera_input). Continuous browser video is a gated stretch.

## Initial procedure

Use an original mobile-desk kit: box, cardboard divider, notebook, thick marker, mouse and bundled cable. Substitute owned objects if necessary while preserving distinguishability. The actual objects and camera provide known-state test inputs, not presentation props. One fixed overhead camera, contrasting surfaces, objects separated enough to inspect individually.

1. Observe the open empty box.
2. Place the divider and confirm the approved setup (zone calibration happens here).
3. Place the notebook and marker in the left zone.
4. Place the mouse and bundled cable in the right zone.
5. Inspect the visible contents before allowing closure.

**Ordering.** Steps 3 and 4 are interchangeable: both depend only on step 2. Item order within a step does not matter. The only essential ordering restriction is that step 5 precedes closure, because closing hides the contents. Do not add arbitrary ordering to manufacture apparent complexity.

**Drafting.** Record narration with the video, transcribe the audio through Voice Transcribe (POST /v1/asr/transcribe), allow edits, and draft with first-party Muse Spark 1.3 using structured output. The author approves the graph and visible completion requirements. Turn timestamps suggest intervals; they do not identify exact action frames or establish a one-turn/one-step mapping. The author confirms step boundaries.

**Typed fallback.** If the microphone or transcription fails, the author types the narration. Drafting proceeds from typed text plus video. A microphone failure must never block a demonstration.

## Outcome semantics

Use pass, fail and insufficient consistently:

| Evidence | Condition result |
|---|---|
| A requirement is clearly violated under adequate inspection | fail |
| Visibility, identity, registration or evidence is inadequate; sources disagree without resolution | insufficient |
| The declared requirement is supported by adequate current evidence | pass |

For an aggregate checkpoint: an established violation yields fail; otherwise an unresolved applicable requirement yields insufficient; otherwise it passes. Do not convert uncertainty about one condition into certainty about another. When a checkpoint has both a failing and an insufficient condition, the recovery shown covers both. API errors and exhausted validation retries yield insufficient with an explicit processing-error reason.

**Human override.** An author may override a condition result. An override is recorded and displayed as an override, with who, when and why. It never counts as an automatic pass in the record or in evaluation metrics.

The current session replaces its latest judgment when applying an override, preserving original
outcomes, reasons and override metadata. A separate immutable event history is future work.

## Absence and exact inventory

Zero SAM detections alone do not establish absence. SAM contributes object grounding and geometry; its labels, counts and masks can be wrong. Whole-scene vision judgment is another check on the same visual evidence, not an independent guarantee of truth.

For an empty-box observation, require adequate interior visibility, a whole-interior assessment supporting emptiness, and no relevant detected object candidates. A clearly observed object makes the empty-box condition fail. Poor visibility or unresolved disagreement makes it insufficient.

For exact inventory, expected items should have detections. The negative condition concerns unexpected items or count excess within the declared test scope, not zero masks for all objects. Combine required-item checks with a whole-interior inventory assessment and supported extra/wrong-item checks. A detected second marker is a failure; a possibly present but obscured item is insufficient.

Text-prompted SAM is not a universal detector of every possible object. Bound the product claim to supported items, capture conditions and tested negatives. Do not certify that no arbitrary unrecognized object exists.

## Prompt selection

Compare candidate SAM prompts on development captures, record the selected prompt in the approved procedure, and freeze it before held-out evaluation. Two prompt sets are load-bearing:

- **Cable:** e.g. "coiled cable", "cable", "USB cable", "cable bundle". A bright, thick cable with a colored tie is the capture choice.
- **Lid / closed box:** e.g. "box lid", "closed box", "cardboard flaps". The winning prompt must distinguish a closed lid from another object lying on the box (a notebook on top is obscured, not closed).

The pen-for-marker case tests the full perception pipeline, not SAM alone.

## Zone calibration

The author draws and approves two polygons inside the box in a reference frame. Store the reference image, image dimensions, box geometry, orientation cue, polygons and calibration revision. A colored stripe on the divider helps locate it but does not define the zones.

Calibration stores oriented aspect ratios for the box and divider. Compare major-axis angles
modulo 180 degrees for non-square shapes. Use modulo 90 only when both the reference and current
oriented shape are near-square; the initial aspect-ratio tolerance is 2% and needs real-capture
tuning. Image-aligned width and height alone cannot distinguish diagonal quarter turns.

A box bounding-box drift threshold is useful but insufficient by itself. Rotation and perspective can change compartment locations without materially changing the bounds. Include an alignment/orientation check against the approved reference. If registration is uncertain or invalid, return insufficient and request recalibration. Do not silently shift or redraw the approved zones.

**Divider.** "Divider in approved position" is an invariant from step 2. If it moves, the zones are invalid: insufficient, request recalibration or repositioning. Test that this condition actually fires on a shifted-divider capture rather than assuming the registration check covers it.

An observed registration change revokes the previous final inspection. Insufficient describes
the current assessment; it does not preserve permission based on invalidated evidence. Mere
occlusion alone does not establish such a change.

Thresholds and acceptable lighting conditions must be selected from actual captures. They are design parameters to test, not guarantees that the interior is fully visible.

## Conditions and time

- An **observation** records a historical assessment at a named frame and remains part of the audit history. The initial empty-box observation need not remain true after packing starts.
- An **invariant** is activated when its establishing step succeeds. At later relevant checkpoints, assess it from fresh evidence while it remains applicable. "Item removed after an earlier pass" is caught by invariant recheck, not by a special case.
- The runtime state records active conditions, current stage, approvals, overrides and evidence. These facts are separate from the static procedure definition even if nested under the three public models.
- **Final inspection uses one capture** (or a short burst treated as one). Every applicable interior inventory invariant must pass on that capture. Evidence must not be combined across separate checkpoints, because an item could change between them. An earlier pass alone cannot authorize closure.
- Snapshot mode checks conditions at checkpoints; it does not establish continuous compliance between them.
- Evidence order comes from an acquisition sequence the capture source assigns when the photo is taken. A photo taken before the last judged photo is rejected. Wall-clock times are stored in UTC as audit metadata and never decide order.

## Closure and completion

Closure requires positive visual evidence of the lid covering the box. Loss of interior visibility alone is not closure.

| Procedure state and observation | Response |
|---|---|
| Box closed before final inspection | fail the premature-closure requirement; request reopening and inspection |
| Interior obscured by a hand or another object | insufficient; request a clear view; never infer closure |
| Inspection passed, but later visible inventory change occurs | revoke ready-to-close status and repeat affected checks |
| Inspection passed and a positively identified closed lid is subsequently observed | allow the terminal closure transition and retain the preclosure evidence |

Bind the final inspection and terminal record to the same session, procedure revision and calibration revision. An author edit or recalibration requires reassessing affected requirements before completion.

Do not require visible interior invariants in the closed-lid frame. Their assessed evidence is the final open-box inspection. Preserve the exact claim **contents observed before closure**, with separate inspection and closure times. The record does not claim functionality, and snapshot mode cannot establish that nothing changed in an unobserved interval.

## Data-model responsibilities

| Public model | Responsibilities |
|---|---|
| Procedure | Versioned steps, prerequisites, human actions, observation/invariant conditions, supported predicates, frozen SAM prompts, evidence requirements and state-dependent recovery. Its drafting shape supports first-party structured output. |
| Judgment | Per-condition and aggregate outcomes, grounded reasons, references to captured evidence, override marker, trusted request/model/provider metadata and capture/assessment times. |
| ZoneCalibration | Approved reference geometry and zones, orientation/alignment information, quality constraints and revision. |

Application validation must check references, graph consistency, allowed predicates, evidence ownership and author approval. Structural schema conformance does not establish a true or executable procedure. The model cannot approve its own draft.

Evidence identifiers must refer to actual captured artifacts. The application supplies trusted provider/model identifiers and request timing from the configured request and response; the model must not invent audit metadata. Preserve both frame-capture time and judgment time.

## Provider boundary

Put checkpoint judgment behind a provider-neutral interface. Start with Meta Spark 1.3. Baseten Spark Open (meta-models/muse-spark-open-1026) becomes an alternative when access is ready; Fireworks supports comparison and gated specialization.

The interface standardizes task inputs and validated outcomes. Provider adapters still handle differences in supported media, schemas, context limits, error handling and response formats. A configuration switch avoids a product rewrite but does not eliminate compatibility testing.

Use bounded retries. Invalid outcomes, invented evidence references or inconsistent condition identifiers must never advance a step. Missing provider-enforced JSON mode (Baseten Spark Open) is manageable through validation, not a non-issue.

## Milestone: correct / wrong / insufficient (Oct 27-28)

**Definition of done.** On new captures (not used to pick prompts or thresholds), the system distinguishes correct, wrong and insufficiently visible states, and a correction changes the result only after fresh evidence.

**Test matrix.**

| Deliberate condition | Required response |
|---|---|
| Missing cable | identify the missing item; withhold completion (fail) |
| Mouse in wrong compartment | identify the placement mismatch (fail) |
| Extra marker in left zone | identify the unexpected item (fail) |
| Pen in place of marker | identify the wrong item (fail), or insufficient if not distinguishable; never pass |
| Hand covering a required item | request another view (insufficient) |
| Lid closed before inspection | request reopening (fail premature closure) |
| Item removed after an earlier pass | invariant recheck fails the affected condition |
| Divider shifted after calibration | request recalibration (insufficient) |
| Correction after any fail | result changes only on a fresh capture |

Pen-for-marker is the first case to drop if time is short.

## Evaluation (held-out, later)

Separate development and held-out captures; held-out cases are separate attempts, not adjacent frames from tuning clips. Target: 36 checkpoint cases (12 correct, 12 planted errors, 12 ambiguous views) plus at least six complete procedure runs (correct, faulty, interrupted). Report false passes, fault detection, correct-state acceptance, ambiguity handling, recovery, drafting effort, latency and cost, with numerators and denominators. Baselines: ordinary checklist; single prompted vision decision; full approved-state evidence-gated loop. Overrides are reported separately and never counted as passes. A held-out case used to fix the system moves to development and is replaced.

## Verify on day one

- Audio extraction from the recording (format, sample rate) accepted by /v1/asr/transcribe.
- Size, length and format limits for video sent to Spark 1.3 (and whether upload via /v1/files is required).
- SAM 3.1 request shape through the Responses API and the meta-sam parser.
- Baseten model ID resolves with a real call, once access exists.

## Implementation status (Oct 26)

Implemented offline, with tests using synthetic masks and local app/file checks.
These tests establish rule behavior, not perception quality.

- Contracts: Procedure (graph validation, author approval), ZoneCalibration (valid, non-overlapping polygons), SceneAssessment (the judge model's structured-output schema), ConditionResult, Judgment, CompletionRecord. Reject non-finite geometry and coercion of fractional counts.
- Deterministic predicates: box_empty, divider_in_place (registration), item_in_zone, inventory_exact; two-source closure with lid coverage; mask de-duplication; box and divider rotation/extent checks.
- Session: prerequisites, interchangeable steps, invariants, single-capture final inspection, revocation on visible change or observed registration drift, recalibration revokes inspection, fresh-capture ID and image-digest enforcement, acquisition-sequence ordering, serialized session updates, marked overrides with original outcomes.
- Run history: an append-only event log per session, synced to disk before each state change. Ordered by index, not time. Rejected requests, revocations, overrides and cancellation are separate events; a check that crashes stays as an unanswered request. After a failed write the session stops accepting requests. No replay or resume.
- Streamlit capture-and-label page writing captures/<split>/labels.jsonl. Labels include normalized image digests. Reject cross-split sequence or identical-image leakage, serialize saves, and keep file failures visible.

Registration still cannot distinguish symmetric turns from mask geometry alone. Keep the
camera fixed and test an asymmetric orientation cue before claiming rotational robustness.

Deliberately not built yet: provider adapters (API shapes unverified), drafting and narration, zone-drawing UI, evaluation runner. Starting thresholds in code (ZONE_FRACTION 0.8, box shift 3%, angle 3 degrees) must be tuned on development captures.

## Next work

1. Kit and camera setup; capture and label development photos (labeling page).
2. Verify the day-one API details, then build the SAM 3.1 and Spark 1.3 perception adapters that produce CaptureObservation.
3. Run the prompt comparisons and freeze prompts.
4. Establish the correct/wrong/insufficient perception baseline against the test matrix before expanding the interface.
5. Zone-drawing calibration UI, narration capture with typed fallback, structured drafting with author approval.
6. Add automatic checkpoint proposals only after the core works; movement suggests a check after stability and visibility gates.
7. Consider Fireworks Glimmer SFT only after a working core and separate evaluation data exist. Compare untuned Glimmer, tuned Glimmer and Spark Open.
