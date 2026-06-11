"""Voicing-accuracy tests: the tab must reflect the strings actually played.

Octave/twelfth/double-octave doublings inside chords share every partial
with a lower note and are recovered by excess evidence against the lower
note's spectral envelope. These tests pin both the recoveries and — just
as important — the detector's silence on notes that are NOT there, across
realistic pluck-position/pickup comb filtering, string inharmonicity and
mild overdrive.

Known physical limit (documented waiver, not a bug): B3 inside an open
E-shape voicing is unrecoverable from one spectrum — its fundamental lies
2 cents from E2's 3rd harmonic (and B2's 2nd), so its energy cannot be
attributed safely. Full voicings still come out with 5 of 6 strings,
clearly distinct from 3-string shells.
"""

import pytest

from fretmap.analyze import analyze, lead_filter
from fretmap.multipitch import detect_pitches
from fretmap.music import classify, midi_to_name, name_to_midi
from fretmap.synth import drive, pluck_midi, render_sequence

SR = 44100
B_REAL = 2e-4  # typical electric-guitar stiff-string coefficient


def n(*names):
    return [name_to_midi(x) for x in names]


def detected(audio):
    return sorted(midi_to_name(p.midi) for p in detect_pitches(audio[: int(1.1 * SR)], SR))


# --- false-positive guards: the recovery pass must stay silent -------------


@pytest.mark.parametrize("name", ["E2", "A2", "D3", "G3", "B3", "E4"])
@pytest.mark.parametrize("pluck_pos", [0.2, 0.33, 0.4])
def test_single_string_no_phantoms(name, pluck_pos):
    # Pluck-position combs are the classic phantom-octave trap: 0.33 nulls
    # the 3rd harmonic, 0.4 systematically boosts even over odd harmonics.
    audio = pluck_midi(
        n(name)[0], 1.2, sr=SR, inharmonicity=3e-4, pluck_pos=pluck_pos, pickup_pos=0.2
    )
    assert detected(audio) == [name], f"{name} pluck@{pluck_pos}"


@pytest.mark.parametrize("names", [("G2", "D3"), ("A3", "C#4")])
def test_dyads_no_phantoms(names):
    # A fifth dyad shares every 3rd harmonic — no phantom G3/D4 allowed.
    for B in (0.0, B_REAL):
        audio = render_sequence([(0.0, n(*names), 1.0)], sr=SR, inharmonicity=B, pluck_pos=0.2)
        assert detected(audio) == sorted(names), f"{names} B={B}"


def test_driven_tones_no_phantoms():
    # tanh overdrive only creates harmonics of what is played.
    assert detected(drive(pluck_midi(n("E2")[0], 1.2, sr=SR, pluck_pos=0.2), 3.0)) == ["E2"]
    audio = drive(render_sequence([(0.0, n("G2", "D3"), 1.0)], sr=SR, pluck_pos=0.2), 2.0)
    assert detected(audio) == ["D3", "G2"]


# --- doubling recovery ------------------------------------------------------


@pytest.mark.parametrize("base", ["E2", "A2", "D3", "G3"])
def test_octave_pairs_recovered(base):
    for B in (0.0, B_REAL):
        m = n(base)[0]
        audio = pluck_midi(
            m, 1.2, sr=SR, inharmonicity=B, pluck_pos=0.2
        ) + pluck_midi(m + 12, 1.2, sr=SR, inharmonicity=B * 0.2, pluck_pos=0.2)
        assert detected(audio) == sorted([base, midi_to_name(m + 12)]), f"{base} B={B}"


def test_twelfth_and_double_octave_recovered():
    for names in (("G2", "D4"), ("A2", "E4"), ("E2", "E4")):
        audio = render_sequence([(0.0, n(*names), 1.0)], sr=SR, inharmonicity=B_REAL, pluck_pos=0.2)
        assert detected(audio) == sorted(names), str(names)


def test_power_chord_three_voices():
    # E5 with the doubled root: the E3 must come back, nothing else.
    audio = render_sequence(
        [(0.0, n("E2", "B2", "E3"), 1.0)], sr=SR, inharmonicity=B_REAL, pluck_pos=0.2
    )
    assert detected(audio) == ["B2", "E2", "E3"]


# --- the Em7 requirement: full voicing vs shell -----------------------------


def test_full_em7_vs_shell():
    full = render_sequence(
        [(0.0, n("E2", "B2", "D3", "G3", "B3", "E4"), 1.2)],
        sr=SR,
        inharmonicity=B_REAL,
        pluck_pos=0.2,
    )
    [ev_full] = analyze(full, SR)
    assert ev_full.short == "Em7"
    # B3 is the documented unrecoverable doubling; the other 5 strings are
    # exact, including the top E4.
    assert set(ev_full.note_names) == {"E2", "B2", "D3", "G3", "E4"}
    assert all(p is not None for p in ev_full.positions)

    shell = render_sequence(
        [(0.0, n("E2", "D3", "G3"), 1.2)], sr=SR, inharmonicity=B_REAL, pluck_pos=0.2
    )
    [ev_shell] = analyze(shell, SR)
    assert ev_shell.note_names == ["E2", "D3", "G3"]
    assert ev_shell.short == "Em7"
    assert "no 5th" in ev_shell.label

    # The tabs are distinct: a strum shows (most of) its strings, the
    # shell exactly three.
    assert len([p for p in ev_full.positions if p]) >= 5
    assert len([p for p in ev_shell.positions if p]) == 3


def test_open_chord_voicings():
    open_e = render_sequence(
        [(0.0, n("E2", "B2", "E3", "G#3", "B3", "E4"), 1.2)],
        sr=SR,
        inharmonicity=B_REAL,
        pluck_pos=0.2,
    )
    [ev] = analyze(open_e, SR)
    assert ev.label == "E"
    got = set(ev.note_names)
    assert {"E2", "B2", "E3", "G#3", "E4"} <= got  # B3: documented waiver
    assert {m % 12 for m in ev.midis} == {4, 8, 11}

    open_am = render_sequence(
        [(0.0, n("A2", "E3", "A3", "C4", "E4"), 1.2)],
        sr=SR,
        inharmonicity=B_REAL,
        pluck_pos=0.2,
    )
    [ev] = analyze(open_am, SR)
    assert ev.label == "Am"
    assert set(ev.note_names) == {"A2", "E3", "A3", "C4", "E4"}


# --- extended-chord naming ---------------------------------------------------


@pytest.mark.parametrize(
    "names,label,short",
    [
        (("E2", "G3", "D4"), "Em7 (no 5th)", "Em7"),
        (("G2", "B2", "D3", "F3"), "G7", "G7"),
        (("E2", "G2", "C3"), "C/E", "C"),
        (("C3", "E3", "G3", "D4"), "Cadd9", "Cadd9"),
        (("G2", "B2", "D3", "F3", "A3"), "G9", "G9"),
        (("E2", "G#2", "D3", "G3"), "E7#9 (no 5th)", "E7#9"),
        (("G2", "B2", "E3", "F3"), "G13 (no 5th)", "G13"),
        (("C3", "E3", "G3", "B3", "D4"), "Cmaj9", "Cmaj9"),
        (("A2", "C3", "G3", "D4"), "Am11 (no 5th)", "Am11"),
        (("C3", "E3", "A3", "D4"), "C6/9 (no 5th)", "C6/9"),
        (("B2", "D3", "F3", "A3"), "Bm7b5", "Bm7b5"),
    ],
)
def test_chord_naming(names, label, short):
    kind, got_label, got_short = classify(n(*names))
    assert kind == "chord"
    assert (got_label, got_short) == (label, short)


def test_extended_chord_end_to_end():
    audio = render_sequence(
        [(0.0, n("G2", "B2", "F3", "A3"), 1.2)], sr=SR, inharmonicity=B_REAL, pluck_pos=0.2
    )
    [ev] = analyze(audio, SR)
    assert ev.short == "G9"


# --- lead isolation ----------------------------------------------------------


def test_lead_filter_extracts_melody():
    rhythm = [(0.0, n("E2", "B2", "E3", "G#3"), 0.9), (1.0, n("A2", "E3", "A3", "C4"), 0.9)]
    melody = [
        (0.25, n("B4"), 0.2),
        (0.5, n("C#5"), 0.2),
        (0.75, n("E5"), 0.2),
        (1.25, n("A4"), 0.2),
        (1.5, n("B4"), 0.2),
        (1.75, n("C#5"), 0.2),
    ]
    audio = render_sequence(rhythm + melody, sr=SR, inharmonicity=B_REAL, pluck_pos=0.2)
    events = analyze(audio, SR)
    lead = lead_filter(events)
    assert [ev.label for ev in lead] == ["B4", "C#5", "E5", "A4", "B4", "C#5"]
    assert all(ev.kind == "note" for ev in lead)
