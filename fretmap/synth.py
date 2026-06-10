"""Simple plucked-string synthesis for demos and tests.

Additive synthesis with harmonically decaying partials and per-harmonic
exponential decay — a rough but pitch-exact stand-in for a plucked
guitar string.
"""

import numpy as np

from fretmap.multipitch import midi_to_freq


def pluck(
    freq: float,
    duration: float,
    sr: int = 44100,
    amp: float = 0.5,
    n_harmonics: int = 12,
) -> np.ndarray:
    t = np.arange(int(duration * sr)) / sr
    sig = np.zeros_like(t)
    for h in range(1, n_harmonics + 1):
        fh = h * freq
        if fh >= sr / 2 * 0.95:
            break
        decay = np.exp(-t * (2.0 + 0.8 * h))
        sig += (1.0 / h) * decay * np.sin(2 * np.pi * fh * t)
    attack = 1.0 - np.exp(-t / 0.004)
    sig *= attack
    # Raised-cosine release so the note doesn't end in a broadband click.
    rel = min(len(sig), max(2, int(0.015 * sr)))
    sig[-rel:] *= 0.5 * (1.0 + np.cos(np.linspace(0, np.pi, rel)))
    peak = np.max(np.abs(sig))
    return amp * sig / peak if peak > 0 else sig


def pluck_midi(midi: int, duration: float, sr: int = 44100, amp: float = 0.5) -> np.ndarray:
    return pluck(midi_to_freq(midi), duration, sr=sr, amp=amp)


def render_sequence(
    events: list[tuple[float, list[int], float]],
    sr: int = 44100,
    tail: float = 0.3,
) -> np.ndarray:
    """Render (start_time, [midi notes], duration) events into one track.

    Chord notes get a small strum stagger so onsets look realistic.
    """
    total = max(t + d for t, _, d in events) + tail
    out = np.zeros(int(total * sr))
    for start, midis, dur in events:
        for k, midi in enumerate(midis):
            stagger = int(k * 0.006 * sr)
            tone = pluck_midi(midi, dur, sr=sr, amp=0.5 / max(1, len(midis)) ** 0.5)
            i = int(start * sr) + stagger
            j = min(len(out), i + len(tone))
            out[i:j] += tone[: j - i]
    peak = np.max(np.abs(out))
    return out / peak * 0.9 if peak > 0 else out
