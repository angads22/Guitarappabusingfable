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

# Pitch-class interval sets (relative to the root) -> chord suffix.
# Listed roughly by how common the chord is; earlier entries win ties.
CHORD_TEMPLATES: list[tuple[frozenset[int], str]] = [
    (frozenset({0, 4, 7}), ""),
    (frozenset({0, 3, 7}), "m"),
    (frozenset({0, 4, 7, 10}), "7"),
    (frozenset({0, 3, 7, 10}), "m7"),
    (frozenset({0, 4, 7, 11}), "maj7"),
    (frozenset({0, 5, 7}), "sus4"),
    (frozenset({0, 2, 7}), "sus2"),
    (frozenset({0, 3, 6}), "dim"),
    (frozenset({0, 4, 8}), "aug"),
    (frozenset({0, 4, 7, 9}), "6"),
    (frozenset({0, 3, 7, 9}), "m6"),
    (frozenset({0, 3, 6, 9}), "dim7"),
    (frozenset({0, 3, 6, 10}), "m7b5"),
    (frozenset({0, 3, 7, 11}), "m(maj7)"),
    (frozenset({0, 2, 4, 7}), "add9"),
    (frozenset({0, 5, 7, 10}), "7sus4"),
    (frozenset({0, 2, 4, 7, 10}), "9"),
    (frozenset({0, 2, 3, 7, 10}), "m9"),
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
    bass_pc = midis[0] % 12
    matches: list[tuple[int, int, str]] = []  # (root!=bass, priority, name)
    for root in pcs:
        intervals = frozenset((p - root) % 12 for p in pcs)
        for priority, (template, suffix) in enumerate(CHORD_TEMPLATES):
            if intervals == template:
                name = NOTE_NAMES[root] + suffix
                if root != bass_pc:
                    name += f"/{NOTE_NAMES[bass_pc]}"
                matches.append((0 if root == bass_pc else 1, priority, name))

    if matches:
        matches.sort()
        name = matches[0][2]
        return "chord", name, name.split("/")[0]

    # Power chord with doubled root: e.g. E2 B2 E3.
    if {(p - bass_pc) % 12 for p in pcs} == {0, 7}:
        name = f"{NOTE_NAMES[bass_pc]}5"
        return "chord", name, name

    notes = ",".join(NOTE_NAMES[p] for p in pcs)
    return "chord", f"({notes})", "?"
