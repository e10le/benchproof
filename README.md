# BenchProof

BenchProof checks a physical procedure against captured visual evidence. You take a photo after each
step; SAM 3.1 and Muse Spark look at it, and deterministic rules say pass, fail or insufficient
evidence, with the fix to make. The prototype covers a mobile-desk packing task.

Development began October 26, 2026 for the Meta Global AI Developer Hackathon (Indie track).

## Status

The rule engine runs offline. Perception calls SAM 3.1 on Meta Model API and Muse Spark Open on Baseten.

- Check page: start a session, take or upload a photo for a step, approve the compartments when the divider goes in, and see the result drawn on the photo with the fix for anything that failed (`pages/1_Check.py`, `benchproof/checking.py`, `benchproof/drawing.py`)

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

Planned next: a run history viewer, Muse Voice Transcribe for narration, and an approval page for drafted procedures.

## The test task

Pack a mobile-desk kit: an open cardboard box, a book as the divider, a mouse, a highlighter, a bundled
cable and a power bank. The procedure is in `procedures/mobile_desk_kit.json`:

1. Show the open, empty box.
2. Lay the book across the middle of the box and approve the two zones.
3. Pack the mouse and highlighter in the top compartment.
4. Pack the cable and power bank in the bottom compartment. (Steps 3 and 4 can happen in either order.)
5. Inspect the contents in one photo before closing.

The rule tests use a separate fixed kit in `tests/fixtures/test_kit.json`, so changing the real kit
does not change what the tests check.

The record claims only "contents observed before closure". It does not claim the items work or that
nothing changed between photos.

The SAM phrases in the procedure file were chosen with the phrase-testing tool on development photos.
Plain cardboard dividers matched the box walls, so a book is the divider. Phrases are frozen before the
held-out evaluation.

## Run it (Windows, Python 3.12)

```
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements-dev.txt
pytest
streamlit run app.py
```

`streamlit run app.py` opens the labeling page; the check page is in the sidebar.

Labeled photos are written to `captures/<split>/`, which git ignores.
Keep a complete recording sequence in one split. The labeling page rejects a sequence or identical
image content that was already saved in the other split.
Label saves use a shared lock so two app instances cannot write conflicting records. If a save
is interrupted and the app reports a stale `captures/.labels.lock`, close every app instance
before removing that lock and retrying. Keep saved images and their label log together.

## Check a packing session

Set both keys in the same terminal, then start the app and open **Check** in the sidebar:

```
set MODEL_API_KEY=your Meta Model API key
set BASETEN_API_KEY=your Baseten hackathon key
streamlit run app.py
```

Enter your name to approve the procedure, then follow the steps. At the divider step the page
proposes two compartments from the box and divider it found; adjust them if needed and approve.
Every later photo is checked against those compartments. Each photo costs at least 8 model calls, and a
session stops at 200. Every request and result is written to `runs/<session>.jsonl`, which git
ignores. When the box is closed after a passing inspection, the page offers the evidence record.

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

## Find SAM phrases that match your objects

Try several phrases on one photo. Each phrase is one SAM call. For every phrase it prints how many
objects matched, with confidence, and saves an image to `prompt_tests/` showing what was matched.

```
set MODEL_API_KEY=your-meta-key
python -m benchproof.prompt_test path\to\photo.jpg "black cable" "usb cable" "wire"
```

Put the winning phrase into `procedures/mobile_desk_kit.json` before the held-out evaluation.

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
orientation cue in real capture tests. Proposed zones are rectangles aligned with the photo; keep
the box square to the camera. Narration and draft approval remain future work. No synthetic-mask test establishes real-world perception accuracy.

## License

Application code: MIT (see `LICENSE`). Model and dependency licenses are separate and are listed
here when each model is integrated.

SAM 3.1 is used through Meta Model API and its output is parsed with Meta's `meta-sam-parser`.

Built with Muse Spark Open, used under the
[Muse Model License](https://huggingface.co/meta-models/Muse-Spark-Open-816B-A42B/blob/main/LICENSE.md)
through Baseten's hosted Model API.
