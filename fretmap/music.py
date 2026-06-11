"""Music theory: note names, interval names, chord identification."""

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

INTERVAL_NAMES = {
    0: "unison", 1: "minor 2nd", 2: "major 2nd", 3: "minor 3rd",
    4: "major 3rd", 5: "perfect 4th", 6: "tritone", 7: "perfect 5th",
    8: "minor 6th", 9: "major 6th", 10: "minor 7th", 11: "major 7th",
    12: "octave",
}

INTERVAL_SHORT = {
    0: "P1", 1: "m2", 2: "M2", 3: "m3", 4: "M3", 5: "P4", 6: "TT",
    7: "P5", 8: "m6", 9: "M6", 10: "m7", 11: "M7", 12: "P8",
}

# Chord templates as (required, optional) pitch-class interval sets
# (relative to the root) -> suffix. A chord matches when its pitch classes
# cover every required tone and add nothing outside required+optional. The
# perfect 5th is optional on most qualities (guitar voicings — especially
# shells — routinely omit it; a missing b5 of dim/m7b5 is never waived,
# it's the identity of the chord). Listed roughly by how common the chord
# is; earlier entries win ties.
CHORD_TEMPLATES: list[tuple[frozenset[int], frozenset[int], str]] = [
    (frozenset({0, 4, 7}), frozenset(), ""),
    (frozenset({0, 3, 7}), frozenset(), "m"),
    (frozenset({0, 4, 10}), frozenset({7}), "7"),
    (frozenset({0, 3, 10}), frozenset({7}), "m7"),
    (frozenset({0, 4, 11}), frozenset({7}), "maj7"),
    (frozenset({0, 5, 7}), frozenset(), "sus4"),
    (frozenset({0, 2, 7}), frozenset(), "sus2"),
    (frozenset({0, 3, 6}), frozenset(), "dim"),
    (frozenset({0, 4, 8}), frozenset(), "aug"),
    (frozenset({0, 4, 9}), frozenset({7}), "6"),
    (frozenset({0, 3, 9}), frozenset({7}), "m6"),
    (frozenset({0, 3, 6, 9}), frozenset(), "dim7"),
    (frozenset({0, 3, 6, 10}), frozenset(), "m7b5"),
    (frozenset({0, 3, 11}), frozenset({7}), "m(maj7)"),
    (frozenset({0, 2, 4, 7}), frozenset(), "add9"),
    (frozenset({0, 2, 3, 7}), frozenset(), "m(add9)"),
    (frozenset({0, 4, 5, 7}), frozenset(), "add11"),
    (frozenset({0, 5, 10}), frozenset({7}), "7sus4"),
    (frozenset({0, 2, 4, 10}), frozenset({7}), "9"),
    (frozenset({0, 2, 3, 10}), frozenset({7}), "m9"),
    (frozenset({0, 2, 4, 11}), frozenset({7}), "maj9"),
    (frozenset({0, 3, 4, 10}), frozenset({7}), "7#9"),
    (frozenset({0, 1, 4, 10}), frozenset({7}), "7b9"),
    (frozenset({0, 4, 9, 10}), frozenset({7, 2}), "13"),
    (frozenset({0, 4, 5, 10}), frozenset({7, 2}), "11"),
    (frozenset({0, 3, 5, 10}), frozenset({7, 2}), "m11"),
    (frozenset({0, 2, 4, 9}), frozenset({7}), "6/9"),
]


def midi_to_name(midi: int) -> str:
    return f"{NOTE_NAMES[midi % 12]}{midi // 12 - 1}"


def name_to_midi(name: str) -> int:
    """Parse names like 'E2', 'F#3', 'Bb1' into MIDI numbers."""
    name = name.strip()
    pc = name[0].upper()
    rest = name[1:]
    semis = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}[pc]
    while rest and rest[0] in "#b♯♭":
        semis += 1 if rest[0] in "#♯" else -1
        rest = rest[1:]
    octave = int(rest)
    return (octave + 1) * 12 + semis % 12 if semis >= 0 else (octave + 1) * 12 + semis


def classify(midis: list[int]) -> tuple[str, str, str]:
    """Classify a set of simultaneous MIDI notes.

    Returns (kind, label, short_label) where kind is one of
    'note', 'interval', 'chord'.
    """
    midis = sorted(midis)
    if len(midis) == 1:
        name = midi_to_name(midis[0])
        return "note", name, NOTE_NAMES[midis[0] % 12]

    pcs = sorted({m % 12 for m in midis})

    if len(pcs) == 1:
        low, high = midis[0], midis[-1]
        label = f"{midi_to_name(low)}+{midi_to_name(high)} (octave)"
        return "interval", label, "P8"

    if len(pcs) == 2 and len(midis) == 2:
        low, high = midis
        diff = high - low
        reduced = diff if diff <= 12 else diff % 12 or 12
        if reduced == 7:
            # Root + perfect fifth is the classic power chord.
            label = f"{NOTE_NAMES[low % 12]}5 ({midi_to_name(low)}+{midi_to_name(high)})"
            return "interval", label, f"{NOTE_NAMES[low % 12]}5"
        label = (
            f"{midi_to_name(low)}+{midi_to_name(high)} ({INTERVAL_NAMES[reduced]})"
        )
        return "interval", label, INTERVAL_SHORT[reduced]

    # Three or more distinct pitch classes (or a doubled dyad): chord.
    # Tolerant template matching: required tones must all be present,
    # extras must be covered by the optional set. Bass-as-root wins, then
    # the most fully voiced template, then template order.
    bass_pc = midis[0] % 12
    matches: list[tuple[tuple[int, float, int], str, str]] = []
    for root in pcs:
        intervals = frozenset((p - root) % 12 for p in pcs)
        for priority, (required, optional, suffix) in enumerate(CHORD_TEMPLATES):
            if not (required <= intervals <= required | optional):
                continue
            score = len(intervals & required) + 0.5 * len(intervals & optional)
            short = NOTE_NAMES[root] + suffix
            label = short
            if root != bass_pc:
                label += f"/{NOTE_NAMES[bass_pc]}"
            if 7 in optional and 7 not in intervals and len(required) >= 3:
                label += " (no 5th)"
            key = (0 if root == bass_pc else 1, -score, priority)
            matches.append((key, label, short))

    if matches:
        matches.sort(key=lambda t: t[0])
        _key, label, short = matches[0]
        return "chord", label, short

    # Power chord with doubled root: e.g. E2 B2 E3.
    if {(p - bass_pc) % 12 for p in pcs} == {0, 7}:
        name = f"{NOTE_NAMES[bass_pc]}5"
        return "chord", name, name

    notes = ",".join(NOTE_NAMES[p] for p in pcs)
    return "chord", f"({notes})", "?"
