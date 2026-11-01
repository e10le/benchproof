# BenchProof

BenchProof checks a physical procedure against captured visual evidence. This offline prototype
contains the rule engine, session state and photo-labeling page for a mobile-desk packing task.

Development began October 26, 2026 for the Meta Global AI Developer Hackathon (Indie track).

## Status

The rule engine runs offline. Perception calls SAM 3.1 on Meta Model API and Muse Spark Open on Baseten.

- Procedure, judgment and calibration contracts with validation (`benchproof/models.py`)
- Deterministic pass / fail / insufficient rules over one capture's perception output (`benchproof/predicates.py`)
- Session state: prerequisites, invariants, pre-closure inspection, overrides, completion record (`benchproof/session.py`)
- Append-only run history: every request, rejection, judgment, override, revocation and cancellation is synced to `runs/<session>.jsonl` before the session state changes (`benchproof/history.py`)
- Scene judge: Muse Spark Open (`meta-models/muse-spark-open-1026` on Baseten) reads a photo and reports visibility, lid state, item counts and unexpected objects; its reply is validated before use (`benchproof/scene.py`). Model calls share timeouts, limited retries and a per-session call budget (`benchproof/llm.py`)
- Object masks: SAM 3.1 on Meta Model API finds the box, divider, lid and each kit item, one short phrase per request, decoded with Meta's `meta-sam-parser` (`benchproof/segmenter.py`)
- Perception: every photo is oriented and resized once (`benchproof/photos.py`), then SAM 3.1 and the scene judge run on the same pixels in parallel to produce the observation the rules check (`benchproof/perceive.py`)
- Procedure drafting: Muse Spark Open turns narration and up to 8 demonstration photos into a draft procedure. The draft must pass the same validation as a hand-written procedure, cannot approve itself, and is retried with the reason when it is unusable (`benchproof/drafting.py`)
- A Streamlit page for capturing and hand-labeling test photos (`app.py`)

The tests drive the rules with synthetic masks. Those results check the logic only; they say nothing
about how well real perception works on real photos.

Planned next: a check page that runs perception on each photo and shows the result, Muse Voice Transcribe for narration, and an approval page for drafted procedures.

## The test task

Pack a mobile-desk kit: an open box, a cardboard divider, a notebook, a thick marker, a mouse and a
bundled cable. The approved procedure is in `procedures/mobile_desk_kit.json`:

1. Show the open, empty box.
2. Position the divider and approve the two zones.
3. Pack the notebook and marker in the left compartment.
4. Pack the mouse and cable in the right compartment. (Steps 3 and 4 can happen in either order.)
5. Inspect the contents in one photo before closing.

The record claims only "contents observed before closure". It does not claim the items work or that
nothing changed between photos.

The SAM prompts in the procedure file are first guesses. They are compared on development photos and
frozen before the held-out evaluation.

## Run it (Windows, Python 3.12)

```
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements-dev.txt
pytest
streamlit run app.py
```

Labeled photos are written to `captures/<split>/`, which git ignores.
Keep a complete recording sequence in one split. The labeling page rejects a sequence or identical
image content that was already saved in the other split.
Label saves use a shared lock so two app instances cannot write conflicting records. If a save
is interrupted and the app reports a stale `captures/.labels.lock`, close every app instance
before removing that lock and retrying. Keep saved images and their label log together.

## Try the scene judge on one photo

Set your Baseten API key in the terminal (never in a file in this repository), then run the judge on
any JPEG or PNG. It uses at most two model calls.

```
set BASETEN_API_KEY=your-key
python -m benchproof.scene path\to\photo.jpg
```

## Run both perception models on one photo

Set both keys in the terminal, then run perception on any JPEG or PNG. It prints what SAM 3.1 found
for each object and the scene judge's reading. It uses at most twelve model calls.

```
set MODEL_API_KEY=your-meta-key
set BASETEN_API_KEY=your-baseten-key
python -m benchproof.perceive path\to\photo.jpg
```

## Draft a procedure from your narration

Write what you do and why in a text file, then pass it with up to 8 photos of the demonstration in
order. The draft is written to `drafts/`, which git ignores. It uses at most three model calls.

```
set BASETEN_API_KEY=your-key
python -m benchproof.drafting narration.txt step1.jpg step2.jpg
```

## Development checks

```
python -m pytest
python -m ruff check .
python -m ruff format --check .
python -m pip check
```

## Limits

The image thresholds are starting values, not measured accuracy guarantees. Symmetric objects can
rotate without a distinguishable visual change; keep the camera and box fixed and use an asymmetric
orientation cue in real capture tests. The check page, narration, draft approval and zone-drawing UI
remain future work. No synthetic-mask test establishes real-world perception accuracy.

## License

Application code: MIT (see `LICENSE`). Model and dependency licenses are separate and are listed
here when each model is integrated.

SAM 3.1 is used through Meta Model API and its output is parsed with Meta's `meta-sam-parser`.

Built with Muse Spark Open, used under the
[Muse Model License](https://huggingface.co/meta-models/Muse-Spark-Open-816B-A42B/blob/main/LICENSE.md)
through Baseten's hosted Model API.
