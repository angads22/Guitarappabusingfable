"""Analysis pipeline: audio in, timed note/interval/chord events out."""

from dataclasses import dataclass, field

import numpy as np

from fretmap.audio import load_audio
from fretmap.dsp import detect_onsets, harmonic_rise, trim_to_sounding
from fretmap.fretboard import (
    DEFAULT_MAX_FRET,
    STANDARD_TUNING,
    assign_positions,
    update_hand,
)
from fretmap.multipitch import (
    detect_pitches,
    estimate_tuning_offset,
    freq_to_midi_float,
    midi_to_freq,
)
from fretmap.music import classify, midi_to_name


@dataclass
class Event:
    time: float                    # onset, seconds
    duration: float                # seconds
    midis: list[int]
    freqs: list[float]
    kind: str                      # 'note' | 'interval' | 'chord'
    label: str                     # e.g. 'E2', 'E2+B2 (perfect 5th)', 'Am7'
    short: str                     # compact label for the tab header
    positions: list[tuple[int, int] | None] = field(default_factory=list)
    saliences: list[float] = field(default_factory=list)  # parallel to midis

    @property
    def note_names(self) -> list[str]:
        return [midi_to_name(m) for m in self.midis]

    def to_dict(self) -> dict:
        return {
            "time": round(self.time, 4),
            "duration": round(self.duration, 4),
            "kind": self.kind,
            "label": self.label,
            "notes": self.note_names,
            "midis": self.midis,
            "freqs": [round(f, 2) for f in self.freqs],
            "saliences": [round(s, 3) for s in self.saliences],
            "positions": [
                {"string": p[0], "fret": p[1]} if p else None for p in self.positions
            ],
        }


def analyze(
    samples: np.ndarray,
    sr: int,
    tuning: tuple[int, ...] = STANDARD_TUNING,
    max_fret: int = DEFAULT_MAX_FRET,
    max_polyphony: int = 6,
) -> list[Event]:
    onsets = detect_onsets(samples, sr)

    # Search range follows the tuning: lowest open string to the highest
    # fret, with half-semitone margins.
    fmin = midi_to_freq(min(tuning)) * 0.97
    fmax = midi_to_freq(max(tuning) + max_fret) * 1.03

    # Pass 1: pitch detection per inter-onset segment.
    segments: list[tuple[int, np.ndarray, list]] = []
    for i, start in enumerate(onsets):
        end = onsets[i + 1] if i + 1 < len(onsets) else len(samples)
        seg = trim_to_sounding(samples[start:end], sr)
        pitches = detect_pitches(
            seg, sr, fmin=fmin, fmax=fmax, max_polyphony=max_polyphony
        )
        if pitches:
            segments.append((start, seg, pitches))

    # Pass 2: round pitches against the track's global tuning offset, drop
    # notes that are only still ringing from the previous event, classify,
    # and assign fretboard positions.
    offset = estimate_tuning_offset([p for _, _, ps in segments for p in ps])
    events: list[Event] = []
    hand = 2.0
    prev_detected: set[int] = set()

    for start, seg, pitches in segments:
        notes: dict[int, object] = {}
        for p in pitches:
            midi = int(round(freq_to_midi_float(p.freq) - offset / 100.0))
            if midi not in notes or p.salience > notes[midi].salience:
                notes[midi] = p
        detected = set(notes)
        kept = {
            midi: p
            for midi, p in notes.items()
            if midi not in prev_detected or harmonic_rise(samples, sr, start, p.freq)
        }
        prev_detected = detected
        if not kept:
            continue

        midis = sorted(kept)
        kind, label, short = classify(midis)
        positions = assign_positions(midis, tuning, max_fret, hand)
        hand = update_hand(positions, hand)
        events.append(
            Event(
                time=start / sr,
                duration=len(seg) / sr,
                midis=midis,
                freqs=[kept[m].freq for m in midis],
                kind=kind,
                label=label,
                short=short,
                positions=positions,
                saliences=[kept[m].salience for m in midis],
            )
        )
    return events


def lead_filter(
    events: list[Event],
    tuning: tuple[int, ...] = STANDARD_TUNING,
    max_fret: int = DEFAULT_MAX_FRET,
) -> list[Event]:
    """Keep only the lead line of a multi-guitar mix.

    Two passes so a chordal intro cannot hijack the register estimate:
    first the track-wide median of each event's most salient pitch fixes
    the initial register; then a forward pass picks, per event, the pitch
    maximising salience minus a soft distance-from-register penalty (so
    octave jumps survive), drops strums (chords / 4+ simultaneous pitches) and
    far-off low-salience winners, and tracks the register as an EMA of the
    kept notes.
    """
    def is_lead_candidate(ev: Event) -> bool:
        return 0 < len(ev.midis) < 4 and ev.kind != "chord"

    picks = [
        ev.midis[int(np.argmax(ev.saliences))] if ev.saliences else ev.midis[-1]
        for ev in events
        if is_lead_candidate(ev)
    ]
    if not picks:
        return []
    register = float(np.median(picks))

    out: list[Event] = []
    hand = 2.0
    for ev in events:
        if not is_lead_candidate(ev):
            continue
        sal = ev.saliences if len(ev.saliences) == len(ev.midis) else [1.0] * len(ev.midis)
        best = max(
            range(len(ev.midis)),
            key=lambda i: sal[i] - 0.08 * abs(ev.midis[i] - register),
        )
        midi = ev.midis[best]
        if abs(midi - register) > 12 and sal[best] < 0.5:
            continue
        register += 0.25 * (midi - register)
        kind, label, short = classify([midi])
        positions = assign_positions([midi], tuning, max_fret, hand)
        hand = update_hand(positions, hand)
        out.append(
            Event(
                time=ev.time,
                duration=ev.duration,
                midis=[midi],
                freqs=[ev.freqs[best]] if best < len(ev.freqs) else [],
                kind=kind,
                label=label,
                short=short,
                positions=positions,
                saliences=[sal[best]],
            )
        )
    return out


def analyze_file(
    path: str,
    tuning: tuple[int, ...] = STANDARD_TUNING,
    max_fret: int = DEFAULT_MAX_FRET,
    max_polyphony: int = 6,
    lead: bool = False,
) -> tuple[list[Event], int, float]:
    """Analyze an audio file. Returns (events, sample_rate, duration_sec)."""
    samples, sr = load_audio(path)
    events = analyze(samples, sr, tuning=tuning, max_fret=max_fret, max_polyphony=max_polyphony)
    if lead:
        events = lead_filter(events, tuning, max_fret)
    return events, sr, len(samples) / sr
