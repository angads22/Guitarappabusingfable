"""End-to-end tests on synthesised guitar audio."""

import numpy as np
import pytest

from fretmap.analyze import Event, analyze
from fretmap.dsp import detect_onsets
from fretmap.multipitch import detect_pitches, midi_to_freq
from fretmap.music import name_to_midi
from fretmap.render import render_summary_fretboard, render_tab
from fretmap.synth import pluck, pluck_midi, render_sequence

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
    # note; the recovery pass must bring them back, with no spurious
    # extras: the detected set is exactly the played voicing.
    midis = n("A2", "E3", "A3", "C4", "E4")
    audio = render_sequence([(0.0, midis, 1.0)], sr=SR)
    pitches = detect_pitches(audio[: int(0.9 * SR)], SR)
    assert {p.midi for p in pitches} == set(midis)


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


def test_tuning_extends_pitch_range():
    # Drop-C's low C2 (65.4 Hz) is below the default detector floor; the
    # search range must follow the requested tuning.
    dropc = tuple(n("C2", "G2", "C3", "F3", "A3", "D4"))
    audio = render_sequence([(0.0, n("C2"), 0.6), (0.7, n("G2"), 0.6)], sr=SR)
    events = analyze(audio, SR, tuning=dropc)
    assert [e.midis for e in events] == [n("C2"), n("G2")]


def test_high_fret_detection():
    # Fret 24 on the high E string (E6 = 1318.5 Hz) was above the old
    # 1200 Hz candidate ceiling.
    audio = render_sequence([(0.0, n("E6"), 0.6)], sr=SR)
    [ev] = analyze(audio, SR)
    assert ev.midis == n("E6")


def test_detuned_recording_rounds_consistently():
    # A guitar detuned ~45 cents with per-note intonation jitter straddles
    # the rounding boundary: without a global tuning-offset estimate the
    # -52c note flips to the semitone below.
    detune = {"E2": -52, "A2": -38, "D3": -49, "G3": -41}
    audio = np.zeros(int(2.5 * SR))
    for i, (name, cents) in enumerate(detune.items()):
        freq = midi_to_freq(name_to_midi(name)) * 2 ** (cents / 1200)
        tone = pluck(freq, 0.5, sr=SR)
        s = int(i * 0.6 * SR)
        audio[s : s + len(tone)] += tone
    events = analyze(audio, SR)
    assert [e.midis for e in events] == [n(name) for name in detune]


def test_ringing_chord_not_rereported():
    # Melody notes played while a chord is still ringing must come out as
    # plain notes, not chords polluted by the sustained pitches.
    audio = render_sequence(
        [
            (0.0, n("E2", "B2", "E3", "G#3"), 2.0),
            (0.5, n("C5"), 0.5),
            (1.1, n("B4"), 0.5),
        ],
        sr=SR,
    )
    events = analyze(audio, SR)
    assert [e.kind for e in events] == ["chord", "note", "note"]
    assert events[1].midis == n("C5")
    assert events[2].midis == n("B4")


def test_inharmonic_strings():
    # Real strings are stiff: partial h sits at h*f0*sqrt(1+B*h^2), sharp
    # of the ideal comb. Detection must not rely on perfectly harmonic
    # synthesis.
    for name in ("E2", "A2", "D3", "G3", "B3", "E4"):
        audio = render_sequence([(0.0, n(name), 0.6)], sr=SR, inharmonicity=3e-4)
        [ev] = analyze(audio, SR)
        assert ev.midis == n(name), f"failed for {name}"

    audio = render_sequence(
        [(0.0, n("A2", "E3", "A3", "C4", "E4"), 1.0)], sr=SR, inharmonicity=2e-4
    )
    [ev] = analyze(audio, SR)
    assert ev.kind == "chord"
    assert ev.label == "Am"


def _tab_event(time: float, duration: float = 0.25) -> Event:
    return Event(
        time=time,
        duration=duration,
        midis=[40],
        freqs=[82.41],
        kind="note",
        label="E2",
        short="E",
        positions=[(0, 0)],
    )


def test_tab_spacing_follows_rhythm():
    # Column spacing is proportional to the time until the next onset, in
    # units of the median inter-onset gap.
    even = [_tab_event(t) for t in (0.0, 0.25, 0.5, 0.75)]
    assert "-0---0---0---0--" in render_tab(even)

    # Third gap is 3x the tatum -> three units of trailing dashes.
    uneven = [_tab_event(t) for t in (0.0, 0.25, 0.5, 1.25)]
    assert "-0---0---0-------0--" in render_tab(uneven)
