"""Polyphonic pitch detection on a single note/chord segment.

Greedy iterative multi-F0 estimation: score every candidate fundamental by
a harmonically weighted sum of spectral peak magnitudes, take the best
candidate, cancel its harmonics from the working spectrum, and repeat
until the score falls below a fraction of the strongest pitch.
"""

from dataclasses import dataclass

import numpy as np

GUITAR_FMIN = 73.0   # just below drop-D D2
GUITAR_FMAX = 1200.0  # just above fret 24 on the high E string


@dataclass
class Pitch:
    freq: float
    midi: int
    cents: float      # deviation from equal temperament, in cents
    salience: float   # relative harmonic score (1.0 for the strongest pitch)


def freq_to_midi_float(f: float) -> float:
    return 69.0 + 12.0 * np.log2(f / 440.0)


def midi_to_freq(m: float) -> float:
    return 440.0 * 2.0 ** ((m - 69.0) / 12.0)


def _band_max(
    spec: np.ndarray, bin_hz: float, f: float, res_hz: float, tol: float = 0.025
) -> tuple[float, int]:
    """Largest magnitude within +-tol (relative) of frequency f.

    The band is never narrower than the analysis-window resolution.
    """
    half_hz = max(tol * f, 1.2 * res_hz)
    lo = max(0, int((f - half_hz) / bin_hz))
    hi = min(len(spec) - 1, int((f + half_hz) / bin_hz) + 1)
    if lo > hi:
        return 0.0, lo
    k = lo + int(np.argmax(spec[lo : hi + 1]))
    return float(spec[k]), k


def _zero_band(spec: np.ndarray, bin_hz: float, f: float, res_hz: float, tol: float = 0.03) -> None:
    """Cancel a harmonic: wide enough to remove the window main lobe skirt."""
    half_hz = max(tol * f, 2.5 * res_hz)
    lo = max(0, int((f - half_hz) / bin_hz))
    hi = min(len(spec) - 1, int((f + half_hz) / bin_hz) + 1)
    spec[lo : hi + 1] = 0.0


def _parabolic_peak(spec: np.ndarray, k: int) -> float:
    """Refined peak position (fractional bin) by parabolic interpolation."""
    if k <= 0 or k >= len(spec) - 1:
        return float(k)
    a, b, c = spec[k - 1], spec[k], spec[k + 1]
    denom = a - 2 * b + c
    if abs(denom) < 1e-12:
        return float(k)
    return k + 0.5 * (a - c) / denom


def detect_pitches(
    seg: np.ndarray,
    sr: int,
    fmin: float = GUITAR_FMIN,
    fmax: float = GUITAR_FMAX,
    max_polyphony: int = 6,
    rel_floor: float = 0.14,
    n_harmonics: int = 10,
) -> list[Pitch]:
    """Detect one or more simultaneous pitches in a segment."""
    # Skip the noisy pluck transient, then analyse a window long enough to
    # resolve low-E harmonics (~0.2 s gives ~5 Hz raw resolution before
    # zero padding).
    skip = int(0.012 * sr)
    want = int(0.20 * sr)
    win = seg[skip : skip + want] if len(seg) > skip + int(0.05 * sr) else seg[:want]
    if len(win) < int(0.04 * sr):
        return []

    windowed = win * np.hanning(len(win))
    nfft = 1 << int(np.ceil(np.log2(len(win) * 4)))
    spec = np.abs(np.fft.rfft(windowed, nfft))
    bin_hz = sr / nfft
    res_hz = sr / len(win)  # analysis-window frequency resolution
    spec[: int(50.0 / bin_hz)] = 0.0  # kill DC / rumble
    spec_max = spec.max()
    if spec_max <= 1e-9:
        return []

    # Candidate fundamentals on a quarter-semitone grid.
    midi_lo = int(np.floor(freq_to_midi_float(fmin)))
    midi_hi = int(np.ceil(freq_to_midi_float(fmax)))
    cand_midis = np.arange(midi_lo, midi_hi + 0.001, 0.25)
    cand_freqs = midi_to_freq(cand_midis)
    harm_limit = min(sr / 2.0 * 0.95, 6000.0)
    weights = 1.0 / np.arange(1, n_harmonics + 1) ** 0.8

    work = spec.copy()
    found: list[tuple[float, float]] = []
    first_score = None

    for _ in range(max_polyphony):
        best_score, best_f = 0.0, 0.0
        for f in cand_freqs:
            score = 0.0
            for h in range(1, n_harmonics + 1):
                fh = h * f
                if fh > harm_limit:
                    break
                amp, k = _band_max(work, bin_hz, fh, res_hz)
                if amp <= 0.0:
                    continue
                # Weight by how exactly the captured peak sits on this
                # candidate's harmonic comb; otherwise low candidates with
                # wide bands hoover up other notes' harmonics.
                offset = k * bin_hz - fh
                sigma = max(0.004 * fh, 0.8 * res_hz)
                comb = np.exp(-0.5 * (offset / sigma) ** 2)
                score += weights[h - 1] * amp * comb
            if score > best_score:
                best_score, best_f = score, f

        if first_score is None:
            if best_score < 1e-6:
                break
            first_score = best_score
        elif best_score < rel_floor * first_score:
            break

        f0 = best_f

        # Half-pitch (sub-octave) guard: if odd harmonics of the chosen f0
        # are nearly absent but even ones are strong, the true pitch is 2*f0.
        odd = sum(_band_max(work, bin_hz, h * f0, res_hz)[0] for h in (1, 3, 5))
        even = sum(_band_max(work, bin_hz, h * f0, res_hz)[0] for h in (2, 4, 6))
        if even > 0 and odd < 0.15 * even and 2 * f0 <= fmax * 1.06:
            f0 *= 2.0

        # Require some energy at the fundamental itself.
        a0, _ = _band_max(work, bin_hz, f0, res_hz)
        if a0 < 0.01 * spec_max:
            break

        # Refine f0 from the first few harmonic peak positions.
        num = den = 0.0
        for h in range(1, 5):
            fh = h * f0
            if fh > harm_limit:
                break
            amp, k = _band_max(work, bin_hz, fh, res_hz)
            if amp <= 0:
                continue
            est = _parabolic_peak(work, k) * bin_hz / h
            if abs(freq_to_midi_float(max(est, 1.0)) - freq_to_midi_float(f0)) < 0.5:
                num += amp * est
                den += amp
        if den > 0:
            f0 = num / den

        found.append((f0, best_score))

        # Cancel this note's harmonics before searching for the next pitch.
        for h in range(1, n_harmonics + 1):
            fh = h * f0
            if fh > harm_limit:
                break
            _zero_band(work, bin_hz, fh, res_hz)

    if not found:
        return []

    top = max(s for _, s in found)
    pitches: dict[int, Pitch] = {}
    for f0, score in found:
        mf = freq_to_midi_float(f0)
        midi = int(round(mf))
        cents = (mf - midi) * 100.0
        p = Pitch(freq=f0, midi=midi, cents=cents, salience=score / top)
        if midi not in pitches or p.salience > pitches[midi].salience:
            pitches[midi] = p
    return sorted(pitches.values(), key=lambda p: p.midi)
