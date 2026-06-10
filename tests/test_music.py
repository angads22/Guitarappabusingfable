from fretmap.music import classify, midi_to_name, name_to_midi


def test_midi_name_roundtrip():
    assert midi_to_name(40) == "E2"
    assert midi_to_name(69) == "A4"
    assert name_to_midi("E2") == 40
    assert name_to_midi("F#3") == 54
    assert name_to_midi("Bb2") == 46


def test_single_note():
    kind, label, short = classify([name_to_midi("E2")])
    assert kind == "note"
    assert label == "E2"


def test_interval_perfect_fifth_is_power_chord():
    kind, label, short = classify([name_to_midi("E2"), name_to_midi("B2")])
    assert kind == "interval"
    assert short == "E5"


def test_interval_minor_third():
    kind, label, short = classify([name_to_midi("A3"), name_to_midi("C4")])
    assert kind == "interval"
    assert "minor 3rd" in label
    assert short == "m3"


def test_octave_dyad():
    kind, label, short = classify([name_to_midi("E2"), name_to_midi("E3")])
    assert kind == "interval"
    assert short == "P8"


def test_major_chord():
    kind, label, _ = classify([name_to_midi(x) for x in ("C3", "E3", "G3")])
    assert kind == "chord"
    assert label == "C"


def test_minor_chord_with_doublings():
    kind, label, _ = classify([name_to_midi(x) for x in ("A2", "E3", "A3", "C4", "E4")])
    assert kind == "chord"
    assert label == "Am"


def test_dominant_seventh():
    kind, label, _ = classify([name_to_midi(x) for x in ("G2", "B2", "D3", "F3")])
    assert kind == "chord"
    assert label == "G7"


def test_slash_chord():
    kind, label, _ = classify([name_to_midi(x) for x in ("E2", "G2", "C3")])
    assert kind == "chord"
    assert label == "C/E"


def test_power_chord_with_octave():
    kind, label, _ = classify([name_to_midi(x) for x in ("E2", "B2", "E3")])
    assert kind == "chord"
    assert label == "E5"
