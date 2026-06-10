"""ASCII rendering: guitar tab, per-event fretboard diagrams, summary map."""

from fretmap.analyze import Event
from fretmap.fretboard import STANDARD_TUNING
from fretmap.music import NOTE_NAMES


def _string_labels(tuning: tuple[int, ...]) -> list[str]:
    """Display names per string, low to high; highest string lowercased."""
    labels = [NOTE_NAMES[m % 12] for m in tuning]
    labels[-1] = labels[-1].lower()
    return labels


def render_tab(
    events: list[Event],
    tuning: tuple[int, ...] = STANDARD_TUNING,
    width: int = 78,
) -> str:
    if not events:
        return "(no notes detected)"

    n_strings = len(tuning)
    labels = _string_labels(tuning)
    prefix_w = max(len(l) for l in labels) + 1

    columns: list[tuple[str, list[str]]] = []
    for ev in events:
        frets = [""] * n_strings
        for pos in ev.positions:
            if pos is not None:
                s, f = pos
                frets[s] = str(f)
        w = max(2, max((len(f) for f in frets), default=2), min(len(ev.short), 6))
        cells = ["-" * (w - len(f)) + f + "--" for f in frets]
        header = ev.short[: w + 2].ljust(w + 2)
        columns.append((header, cells))

    systems: list[str] = []
    i = 0
    while i < len(columns):
        line_len = prefix_w + 1
        j = i
        while j < len(columns) and line_len + len(columns[j][1][0]) + 1 <= width:
            line_len += len(columns[j][1][0])
            j += 1
        j = max(j, i + 1)
        chunk = columns[i:j]
        header = " " * (prefix_w + 1) + "".join(c[0] for c in chunk)
        lines = [header.rstrip()]
        for s in range(n_strings - 1, -1, -1):  # high string on top
            row = labels[s].ljust(prefix_w) + "|" + "".join(c[1][s] for c in chunk) + "|"
            lines.append(row)
        systems.append("\n".join(lines))
        i = j

    return "\n\n".join(systems)


def render_event_fretboard(ev: Event, tuning: tuple[int, ...] = STANDARD_TUNING) -> str:
    n_strings = len(tuning)
    labels = _string_labels(tuning)
    prefix_w = max(len(l) for l in labels)

    frets_used = [p[1] for p in ev.positions if p is not None and p[1] > 0]
    lo = min(frets_used) if frets_used else 1
    hi = max(frets_used) if frets_used else 1
    lo = max(1, min(lo, max(1, hi - 3)))
    hi = max(hi, lo + 3)

    by_string: dict[int, int] = {p[0]: p[1] for p in ev.positions if p is not None}
    note_by_string = {
        p[0]: NOTE_NAMES[m % 12]
        for m, p in zip(ev.midis, ev.positions)
        if p is not None
    }

    head = f"[{ev.time:7.2f}s] {ev.label}  ({' '.join(ev.note_names)})"
    lines = [head]
    for s in range(n_strings - 1, -1, -1):
        fret = by_string.get(s)
        nut = "o" if fret == 0 else " "
        cells = []
        for f in range(lo, hi + 1):
            if fret == f:
                name = note_by_string[s]
                cells.append(name.center(3, "-"))
            else:
                cells.append("---")
        lines.append(f"{labels[s].ljust(prefix_w)} {nut}|" + "|".join(cells) + "|")
    numbers = " " * (prefix_w + 2) + " ".join(str(f).center(3) for f in range(lo, hi + 1))
    lines.append(numbers.rstrip())
    return "\n".join(lines)


def render_summary_fretboard(
    events: list[Event], tuning: tuple[int, ...] = STANDARD_TUNING
) -> str:
    if not events:
        return "(no notes detected)"

    used: dict[tuple[int, int], str] = {}
    for ev in events:
        for m, p in zip(ev.midis, ev.positions):
            if p is not None:
                used[p] = NOTE_NAMES[m % 12]

    labels = _string_labels(tuning)
    prefix_w = max(len(l) for l in labels)
    max_fret = max((f for _, f in used), default=4)
    hi = max(4, max_fret)

    lines = ["Fretboard map of all detected positions:"]
    for s in range(len(tuning) - 1, -1, -1):
        nut = "o" if used.get((s, 0)) else " "
        cells = []
        for f in range(1, hi + 1):
            name = used.get((s, f))
            cells.append(name.center(3, "-") if name else "---")
        lines.append(f"{labels[s].ljust(prefix_w)} {nut}|" + "|".join(cells) + "|")
    numbers = " " * (prefix_w + 2) + " ".join(str(f).center(3) for f in range(1, hi + 1))
    lines.append(numbers.rstrip())
    return "\n".join(lines)


def render_event_list(events: list[Event]) -> str:
    if not events:
        return "(no notes detected)"
    lines = []
    for ev in events:
        kind = ev.kind.ljust(8)
        notes = " ".join(ev.note_names)
        lines.append(f"[{ev.time:7.2f}s] {kind} {ev.label:<24} notes: {notes}")
    return "\n".join(lines)
