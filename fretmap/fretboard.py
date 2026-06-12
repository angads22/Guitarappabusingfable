"""Mapping detected pitches onto guitar fretboard positions.

Positions are (string_index, fret) with string 0 = lowest (thickest)
string. Single notes prefer staying near the current hand position;
chords are solved by backtracking for a playable shape (distinct
strings, limited fret span).
"""

from fretmap.music import name_to_midi
from fretmap.shapes import shape_positions

STANDARD_TUNING = (40, 45, 50, 55, 59, 64)  # E2 A2 D3 G3 B3 E4
DEFAULT_MAX_FRET = 24
MAX_CHORD_SPAN = 4  # frets between lowest and highest fretted note
IDIOM_BONUS = 0.75  # cost edge given to known catalogue fingerings


def parse_tuning(spec: str) -> tuple[int, ...]:
    """Parse a tuning like 'E2,A2,D3,G3,B3,E4' (low string first)."""
    midis = tuple(name_to_midi(part) for part in spec.split(","))
    if len(midis) < 3:
        raise ValueError("tuning needs at least 3 strings")
    return midis


def positions_for_note(
    midi: int, tuning: tuple[int, ...] = STANDARD_TUNING, max_fret: int = DEFAULT_MAX_FRET
) -> list[tuple[int, int]]:
    return [
        (s, midi - open_midi)
        for s, open_midi in enumerate(tuning)
        if 0 <= midi - open_midi <= max_fret
    ]


def _single_cost(fret: int, hand: float) -> float:
    if fret == 0:
        return 0.5
    return abs(fret - hand) + 0.08 * fret


def _chord_cost(frets: list[int], hand: float) -> float:
    fretted = [f for f in frets if f > 0]
    if not fretted:
        return 0.0
    span = max(fretted) - min(fretted)
    return span * 2.0 + 0.3 * (sum(fretted) / len(fretted)) + 0.3 * abs(min(fretted) - hand)


def assign_positions(
    midis: list[int],
    tuning: tuple[int, ...] = STANDARD_TUNING,
    max_fret: int = DEFAULT_MAX_FRET,
    hand: float = 2.0,
) -> list[tuple[int, int] | None]:
    """Choose a fretboard position for each note (parallel to `midis`).

    Notes that cannot be placed (out of range, or more simultaneous notes
    than strings allow) map to None.
    """
    options = [positions_for_note(m, tuning, max_fret) for m in midis]

    if len(midis) == 1:
        if not options[0]:
            return [None]
        return [min(options[0], key=lambda p: _single_cost(p[1], hand))]

    # Backtracking search over note->string assignments for a chord shape.
    order = sorted(range(len(midis)), key=lambda i: midis[i])
    best: tuple[float, list[tuple[int, int] | None]] = (float("inf"), [None] * len(midis))

    def search(idx: int, used: set[int], chosen: list[tuple[int, int] | None]) -> None:
        nonlocal best
        if idx == len(order):
            frets = [p[1] for p in chosen if p is not None]
            fretted = [f for f in frets if f > 0]
            if fretted and max(fretted) - min(fretted) > MAX_CHORD_SPAN:
                return
            placed = sum(1 for p in chosen if p is not None)
            # Strummed chords occupy adjacent strings: penalise shapes
            # with skipped strings between the used ones.
            strings = sorted(p[0] for p in chosen if p is not None)
            gaps = sum(b - a - 1 for a, b in zip(strings, strings[1:]))
            cost = _chord_cost(frets, hand) + 0.35 * gaps - placed * 100.0
            if cost < best[0]:
                best = (cost, chosen.copy())
            return
        i = order[idx]
        for s, f in options[i]:
            if s in used:
                continue
            chosen[i] = (s, f)
            used.add(s)
            search(idx + 1, used, chosen)
            used.remove(s)
            chosen[i] = None
        # Allow dropping this note if it cannot be placed.
        search(idx + 1, used, chosen)

    search(0, set(), [None] * len(midis))

    # Idiomatic tie-break: when the pitch set is a known catalogue shape,
    # prefer the fingering a guitarist would actually use unless the
    # backtracker found something clearly cheaper.
    if len(midis) >= 3:
        by_pitch = sorted(range(len(midis)), key=lambda i: midis[i])
        for shape in shape_positions(midis, tuning, max_fret):
            frets = [f for _, f in shape]
            strings = sorted(s for s, _ in shape)
            gaps = sum(b - a - 1 for a, b in zip(strings, strings[1:]))
            cost = (
                _chord_cost(frets, hand)
                + 0.35 * gaps
                - len(shape) * 100.0
                - IDIOM_BONUS
            )
            if cost < best[0]:
                chosen: list[tuple[int, int] | None] = [None] * len(midis)
                for i, pos in zip(by_pitch, shape):
                    chosen[i] = pos
                best = (cost, chosen)

    return best[1]


def update_hand(positions: list[tuple[int, int] | None], hand: float) -> float:
    fretted = [f for p in positions if p is not None for f in [p[1]] if f > 0]
    if not fretted:
        return hand
    return sum(fretted) / len(fretted)
