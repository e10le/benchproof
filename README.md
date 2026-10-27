# BenchProof

BenchProof checks a physical procedure against captured visual evidence. This offline prototype
contains the rule engine, session state and photo-labeling page for a mobile-desk packing task.

Development began October 26, 2026 for the Meta Global AI Developer Hackathon (Indie track).

## Status

This is the offline core. It makes no model calls yet.

- Procedure, judgment and calibration contracts with validation (`benchproof/models.py`)
- Deterministic pass / fail / insufficient rules over one capture's perception output (`benchproof/predicates.py`)
- Session state: prerequisites, invariants, pre-closure inspection, overrides, completion record (`benchproof/session.py`)
- A Streamlit page for capturing and hand-labeling test photos (`app.py`)

The tests drive the rules with synthetic masks. Those results check the logic only; they say nothing
about how well real perception works on real photos.

Planned next: Muse Voice Transcribe for narration, Muse Spark 1.3 for drafting the procedure and
reading each scene, and SAM 3.1 for object masks, all through the Meta Model API.

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
orientation cue in real capture tests. Model adapters, narration, drafting, and zone-drawing UI
remain future work. No synthetic-mask test establishes real-world perception accuracy.

## License

Application code: MIT (see `LICENSE`). Model and dependency licenses are separate and are listed
here when each model is integrated.
