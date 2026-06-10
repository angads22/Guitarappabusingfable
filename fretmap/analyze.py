"""Analysis pipeline: audio in, timed note/interval/chord events out."""

from dataclasses import dataclass, field

import numpy as np

from fretmap.audio import load_audio
from fretmap.dsp import detect_onsets, trim_to_sounding
from fretmap.fretboard import (
    DEFAULT_MAX_FRET,
    STANDARD_TUNING,
    assign_positions,
    update_hand,
)
from fretmap.multipitch import detect_pitches
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
    events: list[Event] = []
    hand = 2.0

    for i, start in enumerate(onsets):
        end = onsets[i + 1] if i + 1 < len(onsets) else len(samples)
        seg = trim_to_sounding(samples[start:end], sr)
        pitches = detect_pitches(seg, sr, max_polyphony=max_polyphony)
        if not pitches:
            continue
        midis = [p.midi for p in pitches]
        kind, label, short = classify(midis)
        positions = assign_positions(midis, tuning, max_fret, hand)
        hand = update_hand(positions, hand)
        events.append(
            Event(
                time=start / sr,
                duration=len(seg) / sr,
                midis=midis,
                freqs=[p.freq for p in pitches],
                kind=kind,
                label=label,
                short=short,
                positions=positions,
            )
        )
    return events


def analyze_file(
    path: str,
    tuning: tuple[int, ...] = STANDARD_TUNING,
    max_fret: int = DEFAULT_MAX_FRET,
    max_polyphony: int = 6,
) -> tuple[list[Event], int, float]:
    """Analyze an audio file. Returns (events, sample_rate, duration_sec)."""
    samples, sr = load_audio(path)
    events = analyze(samples, sr, tuning=tuning, max_fret=max_fret, max_polyphony=max_polyphony)
    return events, sr, len(samples) / sr
