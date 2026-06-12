from fretmap.fretboard import (
    STANDARD_TUNING,
    assign_positions,
    parse_tuning,
    positions_for_note,
)
from fretmap.music import name_to_midi


def test_positions_for_open_e2():
    assert positions_for_note(name_to_midi("E2")) == [(0, 0)]


def test_positions_for_e4_multiple_strings():
    pos = positions_for_note(name_to_midi("E4"))
    assert (5, 0) in pos and (4, 5) in pos and (3, 9) in pos


def test_single_note_prefers_open_or_near_hand():
    [pos] = assign_positions([name_to_midi("E2")])
    assert pos == (0, 0)


def test_chord_uses_distinct_strings_and_small_span():
    midis = [name_to_midi(x) for x in ("A2", "E3", "A3", "C4", "E4")]  # Am shape
    positions = assign_positions(midis)
    assert all(p is not None for p in positions)
    strings = [p[0] for p in positions]
    assert len(set(strings)) == len(strings)
    fretted = [p[1] for p in positions if p[1] > 0]
    assert max(fretted) - min(fretted) <= 4


def test_chord_prefers_idiomatic_shape():
    # Known catalogue voicings map to the fingering a guitarist would use.
    open_c = [name_to_midi(x) for x in ("C3", "E3", "G3", "C4", "E4")]
    assert assign_positions(open_c) == [(1, 3), (2, 2), (3, 0), (4, 1), (5, 0)]
    open_d = [name_to_midi(x) for x in ("D3", "A3", "D4", "F#4")]
    assert assign_positions(open_d) == [(2, 0), (3, 2), (4, 3), (5, 2)]


def test_out_of_range_note_dropped():
    [pos] = assign_positions([20])  # below low E
    assert pos is None


def test_parse_tuning():
    assert parse_tuning("E2,A2,D3,G3,B3,E4") == STANDARD_TUNING
    drop_d = parse_tuning("D2,A2,D3,G3,B3,E4")
    assert drop_d[0] == name_to_midi("D2")
