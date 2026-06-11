# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

**fretmap** — a DSP-based guitar transcription tool. It takes an isolated guitar recording, detects notes/intervals/chords, and renders them as ASCII guitar tab and fretboard diagrams. Pure Python; the only runtime dependencies are `numpy` and `soundfile` (no ML, no librosa).

## Commands

```bash
pip install -e ".[dev]"            # install with pytest (runtime deps: numpy, soundfile)

python -m pytest tests             # full test suite
python -m pytest tests/test_pipeline.py -q                       # one file
python -m pytest tests/test_pipeline.py::test_chord_pitch_detection  # one test

python examples/make_demo.py       # synthesize demo wavs into examples/output/
fretmap examples/output/rhythm.wav # run the CLI on one (or: python -m fretmap …)
fretmap-gui                        # run the Tkinter GUI
```

There is no linter or formatter configured.

## Architecture

The whole app is a linear pipeline, orchestrated by `analyze()` in `fretmap/analyze.py`:

```
audio file → load_audio (audio.py)
           → detect_onsets via spectral flux (dsp.py)
           → per-segment polyphonic pitch detection (multipitch.py)
           → note/interval/chord classification (music.py)
           → fretboard position assignment (fretboard.py)
           → ASCII tab / fretboard diagrams / JSON (render.py)
```

`analyze()` returns a list of `Event` dataclasses (time, midis, kind `'note'|'interval'|'chord'`, label, fretboard positions). `Event.to_dict()` defines the `--json` output schema. Three front-ends sit on top of this core: `cli.py` (entry point `fretmap`), `gui.py` (entry point `fretmap-gui`), and the public API in `__init__.py` (`analyze`, `analyze_file`, `lead_filter`, `Event`).

Key domain conventions:

- Pitches are MIDI note numbers throughout; conversion helpers live in `music.py` (`name_to_midi`, `midi_to_name`) and `multipitch.py` (freq↔midi).
- `analyze()` is two-pass: pass 1 detects pitches per segment (search range derived from the tuning); pass 2 estimates a global tuning offset (`estimate_tuning_offset`), re-rounds, drops still-ringing notes from previous events (`harmonic_rise` in `dsp.py`), then classifies and assigns positions.
- `detect_pitches()` itself is greedy find-and-cancel plus a **hidden-note recovery/validation pass** (`_refine_polyphony`): octave/twelfth/double-octave doublings invisible to cancellation are recovered via envelope-excess evidence on a longer FFT; greedy notes sitting on another note's comb are re-validated. Per-note stiff-string stretch is fitted (`MAX_BETA`), and YIN arbitrates octaves on single notes. Set `FRETMAP_DEBUG=1` (or `=2` for per-harmonic traces) to debug recovery decisions.
- Chord naming (`music.py`) uses `(required, optional)` pitch-class templates — the perfect 5th is optional on most qualities, so shells are named with a `" (no 5th)"` label suffix. Known waiver: B3 inside open E-shape voicings is physically unrecoverable (2 cents from 3×E2) — see `tests/test_voicing.py`.
- Tunings are tuples of open-string MIDI numbers, **low string first** (e.g. `STANDARD_TUNING = (40, 45, 50, 55, 59, 64)`); fretboard positions are `(string_index, fret)` with string 0 = lowest/thickest string.
- `assign_positions` is stateful across events via a "hand position" float — single notes prefer staying near the current hand; chords are solved by backtracking for a playable shape (distinct strings, fret span ≤ 4).
- The README's "How it works" section documents the DSP algorithms (spectral-flux onsets, harmonically weighted multi-F0 with harmonic cancellation, sub-octave guard); keep it in sync when changing `dsp.py`/`multipitch.py`.

## Testing approach

Tests need no audio fixtures: `fretmap/synth.py` synthesizes plucked-string audio with known ground truth (`pluck_midi`, `render_sequence`, plus `inharmonicity`, `pluck_pos`/`pickup_pos` comb filtering and a `drive` soft-clip helper for realism), and tests assert the pipeline recovers the input. `tests/test_voicing.py` is the accuracy contract for the hidden-note recovery: false-positive guards (combed/driven single strings and dyads must stay phantom-free) are as load-bearing as the recovery assertions — change detection thresholds only with both halves green.

`tests/test_gui.py` enforces that `fretmap/gui.py` is importable **without tkinter installed** — all `tkinter` imports in `gui.py` (and in `cli.py` helpers) must stay lazy, inside functions/methods.

## Versioning and the self-updater

The version is declared in **two places that must be kept in sync**: `pyproject.toml` and `__version__` in `fretmap/__init__.py`. The self-updater (`fretmap/updater.py`) compares `__version__` against the latest GitHub release tag, so a release with an unbumped `__init__.py` breaks shipped executables' update detection.

Other updater constraints:

- `updater.py` deliberately uses **only the standard library** (urllib, not requests) to keep the PyInstaller bundle small. Keep it that way.
- Release asset names are generated by `updater.asset_name()` (e.g. `fretmap-gui-windows.exe`, `fretmap-linux`). The rename step in `.github/workflows/build.yml` must produce exactly those names or shipped executables can't find their update download.
- Self-update only works in frozen (PyInstaller) builds — `can_self_update()` checks `sys.frozen`.

## CI / releases

`.github/workflows/build.yml` runs on every push: installs, runs the test suite, builds single-file PyInstaller executables (CLI `--console` from `fretmap/__main__.py`, GUI `--windowed` from `packaging/gui_launcher.py`) on Windows/macOS/Linux, smoke-tests the CLI binary against a generated demo, and smoke-tests the GUI launch under xvfb on Linux. Pushing a `v*` tag (or `workflow_dispatch` with a `release_tag` input) additionally attaches the renamed binaries to a GitHub Release — which is what the self-updater follows.

## GUI threading model

`gui.py` runs analysis and update downloads on daemon `threading.Thread`s, passing results back through a `queue.Queue` polled with `root.after(100, …)` — never touch tk widgets from a worker thread. Update checks run on launch and every 6 hours (`RECHECK_MS`).
