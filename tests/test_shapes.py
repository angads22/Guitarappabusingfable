"""Self-validating chord-shape catalogue.

Every seed shape must describe itself: its pitch content in standard
tuning classifies to its declared name, and it obeys the playability and
curation rules that the voicing-completion feature depends on.
"""

import time

import pytest

from fretmap.fretboard import (
    DEFAULT_MAX_FRET,
    MAX_CHORD_SPAN,
    STANDARD_TUNING,
    parse_tuning,
)
from fretmap.music import classify, name_to_midi
from fretmap.shapes import (
    SHAPES,
    build_catalogue,
    completion_candidates,
    shape_midis,
    shape_positions,
)


def n(*names):
    return [name_to_midi(x) for x in names]


@pytest.mark.parametrize(
    "name,frets", SHAPES, ids=[f"{i}-{s[0]}" for i, s in enumerate(SHAPES)]
)
def test_seed_self_describes(name, frets):
    midis = sorted(shape_midis(frets, STANDARD_TUNING))
    _kind, _label, short = classify(midis)
    assert short == name


def test_seed_playability_and_curation():
    for name, frets in SHAPES:
        played = [s for s, f in enumerate(frets) if f is not None]
        fretted = [f for f in frets if f]
        if fretted:
            assert max(fretted) - min(fretted) <= MAX_CHORD_SPAN, (name, frets)
        # No unison doublings in standard tuning.
        midis = shape_midis(frets, STANDARD_TUNING)
        assert len(set(midis)) == len(midis), (name, frets)
        # Curation rule: shapes sounding >= 5 strings must be contiguous,
        # or they exactly match "full strum minus the masked string" and
        # silently disable voicing completion.
        if len(played) >= 5:
            assert played == list(range(played[0], played[-1] + 1)), (name, frets)


def test_catalogue_builds_for_tunings():
    for spec in ("E2,A2,D3,G3,B3,E4", "D2,A2,D3,G3,B3,E4", "D2,A2,D3,G3,A3,D4"):
        tuning = parse_tuning(spec)
        catalogue = build_catalogue(tuning, DEFAULT_MAX_FRET)
        assert len(catalogue) > 200
        for key, fingerings in catalogue.items():
            assert list(key) == sorted(set(key))
            for positions in fingerings:
                assert [tuning[s] + f for s, f in positions] == list(key)


def test_catalogue_build_is_fast():
    build_catalogue.cache_clear()
    t0 = time.perf_counter()
    build_catalogue(STANDARD_TUNING, DEFAULT_MAX_FRET)
    assert time.perf_counter() - t0 < 0.5


def test_exact_match_open_am():
    matches = shape_positions(
        n("A2", "E3", "A3", "C4", "E4"), STANDARD_TUNING, DEFAULT_MAX_FRET
    )
    assert ((1, 0), (2, 2), (3, 2), (4, 1), (5, 0)) in matches


def test_completion_candidates_full_em7():
    # The full open Em7 minus its spectrally masked B3.
    detected = n("E2", "B2", "D3", "G3", "E4")
    cands = completion_candidates(detected, STANDARD_TUNING, DEFAULT_MAX_FRET)
    missings = {m for m, _ in cands}
    assert name_to_midi("B3") in missings
    b3_fingerings = [pos for m, pos in cands if m == name_to_midi("B3")]
    assert ((0, 0), (1, 2), (2, 0), (3, 0), (4, 0), (5, 0)) in b3_fingerings


def test_no_exact_match_blocks_completion_input():
    # The detected 5-note Em7 set must NOT be an exact catalogue entry
    # (that is what the contiguity curation rule guarantees).
    detected = n("E2", "B2", "D3", "G3", "E4")
    assert shape_positions(detected, STANDARD_TUNING, DEFAULT_MAX_FRET) == []
