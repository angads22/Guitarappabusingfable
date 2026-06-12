"""Catalogue of guitar chord shapes: idiomatic voicings plus fret offsets.

A shape is a per-string tuple, low string first. Entries: None = string
not played, 0 = open string (fixed under transposition), n >= 1 =
fretted (shifts with the transposition offset). Mixed shapes — fretted
strings against fixed open strings — therefore generate the whole family
of drone voicings as the fretted part moves up the neck; the chord
identity of each realization is whatever its pitch content says, never
stored.

Every seed declares the chord short-name its base form classifies to in
standard tuning; tests/test_shapes.py enforces it (self-validating
catalogue), along with the playability rules:

- fretted span <= MAX_CHORD_SPAN, strings distinct, no unison doublings
  (two strings sounding the same midi cannot be represented in the
  Event positions<->midis parallelism, so such realizations are skipped
  per tuning at build time);
- shapes sounding >= 5 strings must use contiguous strings. This is
  load-bearing for voicing completion: a seeded 5-note shape with a
  muted inner string (e.g. 0-2-0-0-x-0) would be an exact match for
  "full strum minus the spectrally masked string" and silently disable
  the inference forever.
"""

from functools import lru_cache

X = None  # muted string, for readable seed tables

# (declared short-name at base position in standard tuning, frets low->high)
SHAPES: list[tuple[str, tuple[int | None, ...]]] = [
    # --- open chords (CAGED) and common variants ---------------------------
    ("E",      (0, 2, 2, 1, 0, 0)),
    ("Em",     (0, 2, 2, 0, 0, 0)),
    ("E7",     (0, 2, 0, 1, 0, 0)),
    ("Em7",    (0, 2, 0, 0, 0, 0)),
    ("Em7",    (0, 2, 0, 0, 3, 0)),
    ("Emaj7",  (0, 2, 1, 1, 0, 0)),
    ("Esus4",  (0, 2, 2, 2, 0, 0)),
    ("Eadd9",  (0, 2, 2, 1, 0, 2)),
    ("Em9",    (0, 2, 0, 0, 0, 2)),
    ("A",      (X, 0, 2, 2, 2, 0)),
    ("Am",     (X, 0, 2, 2, 1, 0)),
    ("A7",     (X, 0, 2, 0, 2, 0)),
    ("Am7",    (X, 0, 2, 0, 1, 0)),
    ("Amaj7",  (X, 0, 2, 1, 2, 0)),
    ("Asus2",  (X, 0, 2, 2, 0, 0)),
    ("Asus4",  (X, 0, 2, 2, 3, 0)),
    ("Aadd9",  (X, 0, 2, 4, 2, 0)),
    ("A7sus4", (X, 0, 2, 0, 3, 0)),
    ("D",      (X, X, 0, 2, 3, 2)),
    ("Dm",     (X, X, 0, 2, 3, 1)),
    ("D7",     (X, X, 0, 2, 1, 2)),
    ("Dm7",    (X, X, 0, 2, 1, 1)),
    ("Dmaj7",  (X, X, 0, 2, 2, 2)),
    ("Dsus2",  (X, X, 0, 2, 3, 0)),
    ("Dsus4",  (X, X, 0, 2, 3, 3)),
    ("D6",     (X, X, 0, 2, 0, 2)),
    ("C",      (X, 3, 2, 0, 1, 0)),
    ("Cmaj7",  (X, 3, 2, 0, 0, 0)),
    ("C7",     (X, 3, 2, 3, 1, 0)),
    ("Cadd9",  (X, 3, 2, 0, 3, 0)),
    ("G",      (3, 2, 0, 0, 0, 3)),
    ("G",      (3, 2, 0, 0, 3, 3)),
    ("G6",     (3, 2, 0, 0, 0, 0)),
    ("G7",     (3, 2, 0, 0, 0, 1)),
    ("F",      (X, X, 3, 2, 1, 1)),
    ("Fmaj7",  (X, X, 3, 2, 1, 0)),
    ("B7",     (X, 2, 1, 2, 0, 2)),
    ("Em7",    (X, 2, 0, 0, 0, 0)),   # Em7/B — low E skipped
    # --- movable barre forms (E-shape at F, A-shape at A#) -----------------
    ("F",      (1, 3, 3, 2, 1, 1)),
    ("Fm",     (1, 3, 3, 1, 1, 1)),
    ("F7",     (1, 3, 1, 2, 1, 1)),
    ("Fm7",    (1, 3, 1, 1, 1, 1)),
    ("Fmaj7",  (1, X, 2, 2, 1, X)),
    ("A#",     (X, 1, 3, 3, 3, 1)),
    ("A#m",    (X, 1, 3, 3, 2, 1)),
    ("A#7",    (X, 1, 3, 1, 3, 1)),
    ("A#m7",   (X, 1, 3, 1, 2, 1)),
    ("A#maj7", (X, 1, 3, 2, 3, 1)),
    ("A#sus2", (X, 1, 3, 3, 1, 1)),
    ("A#sus4", (X, 1, 3, 3, 4, 1)),
    # --- power chords (movable and open) -----------------------------------
    ("F5",     (1, 3, X, X, X, X)),
    ("F5",     (1, 3, 3, X, X, X)),
    ("A#5",    (X, 1, 3, X, X, X)),
    ("A#5",    (X, 1, 3, 3, X, X)),
    ("D#5",    (X, X, 1, 3, X, X)),
    ("D#5",    (X, X, 1, 3, 4, X)),
    ("E5",     (0, 2, X, X, X, X)),
    ("E5",     (0, 2, 2, X, X, X)),
    ("A5",     (X, 0, 2, X, X, X)),
    ("A5",     (X, 0, 2, 2, X, X)),
    ("D5",     (X, X, 0, 2, X, X)),
    ("D5",     (X, X, 0, 2, 3, X)),
    # --- shell voicings: R-7-3 / R-3-7 grips on 6th- and 5th-string roots --
    ("Fm7",    (1, X, 1, 1, X, X)),
    ("F7",     (1, X, 1, 2, X, X)),
    ("Fmaj7",  (1, X, 2, 2, X, X)),
    ("A#m7",   (X, 1, X, 1, 2, X)),
    ("A#7",    (X, 1, X, 1, 3, X)),
    ("A#maj7", (X, 1, X, 2, 3, X)),
    # --- movable triads, all inversions, top-3 and middle-3 string sets ----
    ("C",      (X, X, X, 5, 5, 3)),
    ("G",      (X, X, X, 4, 3, 3)),
    ("C#",     (X, X, X, 1, 2, 1)),
    ("A#m",    (X, X, X, 3, 2, 1)),
    ("G#6",    (X, X, X, 1, 1, 1)),  # = Fm 1st inversion; bass-as-root names it 6
    ("Dm",     (X, X, X, 2, 3, 1)),
    ("F",      (X, X, 3, 2, 1, X)),
    ("C#",     (X, X, 3, 1, 2, X)),
    ("A",      (X, X, 2, 2, 2, X)),
    ("Fm",     (X, X, 3, 1, 1, X)),
    ("F6",     (X, X, 3, 2, 3, X)),  # = Dm 1st inversion; bass-as-root names it 6
    ("Am",     (X, X, 2, 2, 1, X)),
    # --- extended-chord grips (movable) -------------------------------------
    ("B9",     (X, 2, 1, 2, 2, X)),
    ("B7#9",   (X, 2, 1, 2, 3, X)),
    ("B13",    (X, 2, 1, 2, 2, 4)),
    ("D#dim7", (X, X, 1, 2, 1, 2)),
    ("Bm7b5",  (X, 2, 3, 2, 3, X)),
    ("C6/9",   (X, 3, 2, 2, 3, X)),
    # --- drone / jangle shapes: movable fretted strings + fixed open B,e ----
    ("Cmaj9",  (3, 5, 5, 5, 0, 0)),
    ("Cmaj7",  (X, 3, 5, 5, 0, 0)),
    ("Cmaj7",  (X, X, 5, 5, 0, 0)),
]


def shape_midis(frets: tuple[int | None, ...], tuning: tuple[int, ...]) -> list[int]:
    """Sounding midis of a shape (unsorted, low string first)."""
    return [t + f for t, f in zip(tuning, frets) if f is not None]


def _realizations(frets, tuning, max_fret):
    """Yield (sorted midi tuple, positions parallel to it) per offset."""
    fretted = [f for f in frets if f]
    k_lo = 1 - min(fretted) if fretted else 0
    k_hi = max_fret - max(fretted) if fretted else 0
    for k in range(k_lo, k_hi + 1):
        positions = [
            (s, f + k if f else 0) for s, f in enumerate(frets) if f is not None
        ]
        midis = [tuning[s] + f for s, f in positions]
        if len(set(midis)) != len(midis):
            continue  # unison doubling under this tuning: unrepresentable
        order = sorted(range(len(midis)), key=midis.__getitem__)
        yield tuple(midis[i] for i in order), tuple(positions[i] for i in order)


@lru_cache(maxsize=8)
def build_catalogue(
    tuning: tuple[int, ...], max_fret: int
) -> dict[tuple[int, ...], list[tuple[tuple[int, int], ...]]]:
    """sorted midi tuple -> known fingerings (each parallel to the key)."""
    catalogue: dict[tuple[int, ...], list[tuple[tuple[int, int], ...]]] = {}
    for _name, frets in SHAPES:
        if len(frets) != len(tuning):
            continue
        for midis, positions in _realizations(frets, tuning, max_fret):
            fingerings = catalogue.setdefault(midis, [])
            if positions not in fingerings:
                fingerings.append(positions)
    return catalogue


def shape_positions(
    midis: list[int], tuning: tuple[int, ...], max_fret: int
) -> list[tuple[tuple[int, int], ...]]:
    """Catalogue fingerings whose pitch content is exactly `midis`."""
    return build_catalogue(tuple(tuning), max_fret).get(tuple(sorted(midis)), [])


def completion_candidates(
    midis: list[int], tuning: tuple[int, ...], max_fret: int
) -> list[tuple[int, tuple[tuple[int, int], ...]]]:
    """Shapes that are exact one-note supersets of `midis`.

    Returns (missing_midi, fingering) pairs; the fingering is parallel to
    sorted(midis + [missing_midi]).
    """
    catalogue = build_catalogue(tuple(tuning), max_fret)
    want = set(midis)
    out = []
    for key, fingerings in catalogue.items():
        if len(key) != len(want) + 1 or not want < set(key):
            continue
        (missing,) = set(key) - want
        for positions in fingerings:
            out.append((missing, positions))
    return out
