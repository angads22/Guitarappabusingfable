"""End-to-end tests on synthesised guitar audio."""

import numpy as np
import pytest

from fretmap.analyze import analyze
from fretmap.dsp import detect_onsets
from fretmap.multipitch import detect_pitches
from fretmap.music import name_to_midi
from fretmap.render import render_summary_fretboard, render_tab
from fretmap.synth import pluck_midi, render_sequence

SR = 44100


def n(*names):
    return [name_to_midi(x) for x in names]


def test_single_pitch_detection():
    for name in ("E2", "A2", "D3", "G3", "B3", "E4", "C5"):
        midi = name_to_midi(name)
        tone = pluck_midi(midi, 0.6, sr=SR)
        pitches = detect_pitches(tone, SR)
        assert [p.midi for p in pitches] == [midi], f"failed for {name}"


def test_chord_pitch_detection():
    # Octave doublings (A3 = 2x A2) share every harmonic with the lower
    # note and may be absorbed by it; require full pitch-class coverage
    # and no spurious pitch classes.
    midis = n("A2", "E3", "A3", "C4", "E4")
    audio = render_sequence([(0.0, midis, 1.0)], sr=SR)
    pitches = detect_pitches(audio[: int(0.9 * SR)], SR)
    detected = {p.midi for p in pitches}
    assert {m % 12 for m in midis} == {m % 12 for m in detected}
    assert detected <= set(midis)


def test_onset_count():
    events = [(0.0, n("E2"), 0.4), (0.5, n("A2"), 0.4), (1.0, n("D3"), 0.4)]
    audio = render_sequence(events, sr=SR)
    onsets = detect_onsets(audio, SR)
    assert len(onsets) == 3
    for got, (want, _, _) in zip(onsets, events):
        assert abs(got / SR - want) < 0.05


def test_full_pipeline_mixed_track():
    events_in = [
        (0.0, n("E2"), 0.45),                                 # note
        (0.6, n("G2", "D3"), 0.45),                           # power chord interval
        (1.2, n("A2", "E3", "A3", "C4", "E4"), 0.9),          # Am chord
        (2.2, n("B3"), 0.4),                                  # note
    ]
    audio = render_sequence(events_in, sr=SR)
    events = analyze(audio, SR)

    assert len(events) == 4
    kinds = [e.kind for e in events]
    assert kinds == ["note", "interval", "chord", "note"]

    assert events[0].midis == n("E2")
    assert events[0].label == "E2"
    assert events[1].short == "G5"
    assert events[2].label == "Am"
    assert {m % 12 for m in events[2].midis} == {9, 0, 4}  # A, C, E
    assert events[3].midis == n("B3")

    # Every detected note has either a fretboard position or None.
    for ev in events:
        assert len(ev.positions) == len(ev.midis)

    # Rendering should not crash and should contain fret digits.
    tab = render_tab(events)
    assert "|" in tab and "0" in tab
    assert "Fretboard map" in render_summary_fretboard(events)


def test_silence_yields_no_events():
    audio = np.zeros(SR)
    assert analyze(audio, SR) == []


def test_event_to_dict():
    audio = render_sequence([(0.0, n("E2"), 0.5)], sr=SR)
    [ev] = analyze(audio, SR)
    d = ev.to_dict()
    assert d["kind"] == "note"
    assert d["notes"] == ["E2"]
    assert d["positions"] == [{"string": 0, "fret": 0}]


@pytest.mark.parametrize(
    "names,label",
    [
        (("E2", "B2", "E3", "G#3", "B3", "E4"), "E"),
        (("D3", "A3", "D4", "F#4"), "D"),
        (("G2", "B2", "D3", "F3"), "G7"),
    ],
)
def test_chord_labels_end_to_end(names, label):
    audio = render_sequence([(0.0, n(*names), 1.2)], sr=SR)
    events = analyze(audio, SR)
    assert len(events) == 1
    assert events[0].kind == "chord"
    assert events[0].label == label
