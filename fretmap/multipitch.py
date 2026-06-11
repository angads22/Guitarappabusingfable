"""Polyphonic pitch detection on a single note/chord segment.

Greedy iterative multi-F0 estimation: score every candidate fundamental by
a harmonically weighted sum of spectral peak magnitudes (with a fitted
stiff-string stretch), take the best candidate, cancel its harmonics from
the working spectrum, and repeat until the score falls below a fraction
of the strongest pitch. A recovery pass then re-examines the uncancelled
spectrum for notes hidden inside a found note's harmonic comb (octave and
twelfth doublings inside chords), and a time-domain YIN estimate
arbitrates octave errors on single notes.
"""

import os
from dataclasses import dataclass

import numpy as np

GUITAR_FMIN = 73.0   # just below drop-D D2
GUITAR_FMAX = 1350.0  # just above fret 24 on the high E string (E6 = 1318.5 Hz)

# Stiff-string inharmonicity: partial h sits at h*f0*(1 + (B/2)*h^2).
# beta below is B/2; electric-guitar B runs up to ~5e-4 on wound strings.
MAX_BETA = 2.5e-4


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


def _spectrum(win: np.ndarray, sr: int) -> tuple[np.ndarray, float, float]:
    """Hann-windowed, 4x zero-padded magnitude spectrum of a segment.

    Returns (spec, bin_hz, res_hz) with DC/rumble below 50 Hz removed.
    """
    windowed = win * np.hanning(len(win))
    nfft = 1 << int(np.ceil(np.log2(len(win) * 4)))
    spec = np.abs(np.fft.rfft(windowed, nfft))
    bin_hz = sr / nfft
    res_hz = sr / len(win)  # analysis-window frequency resolution
    spec[: int(50.0 / bin_hz)] = 0.0
    return spec, bin_hz, res_hz


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


def _eval_candidate(
    work: np.ndarray,
    bin_hz: float,
    res_hz: float,
    f: float,
    harm_limit: float,
    weights: np.ndarray,
    n_harmonics: int,
) -> tuple[float, float]:
    """Score a candidate fundamental; returns (score, fitted beta).

    Each captured peak is weighted by how exactly it sits on the
    candidate's harmonic comb (stops low candidates with wide bands from
    hoovering up other notes' harmonics). The comb is first stretched by a
    fitted stiff-string coefficient so genuinely sharp high partials of a
    real string are not discounted: offsets follow (B/2)*h^2, estimated
    robustly as a median so a stray foreign peak cannot bend the fit.
    """
    amps, offs, fhs, hs = [], [], [], []
    for h in range(1, n_harmonics + 1):
        fh = h * f
        if fh > harm_limit:
            break
        amp, k = _band_max(work, bin_hz, fh, res_hz)
        if amp <= 0.0:
            continue
        amps.append(amp)
        offs.append(k * bin_hz - fh)
        fhs.append(fh)
        hs.append(h)
    if not amps:
        return 0.0, 0.0
    amps = np.array(amps)
    offs = np.array(offs)
    fhs = np.array(fhs)
    hs = np.array(hs)

    beta = 0.0
    strong = (hs >= 4) & (amps >= 0.1 * amps.max())
    if strong.sum() >= 3:
        rel = np.clip(offs[strong] / fhs[strong], 0.0, None)
        beta = float(np.median(rel / hs[strong] ** 2))
        beta = min(max(beta, 0.0), MAX_BETA)

    resid = offs - beta * hs**2 * fhs
    sigma = np.maximum(0.004 * fhs, 0.8 * res_hz)
    comb = np.exp(-0.5 * (resid / sigma) ** 2)
    return float(np.sum(weights[hs - 1] * amps * comb)), beta


def _yin_pitch(seg: np.ndarray, sr: int, fmin: float, fmax: float) -> tuple[float, float] | None:
    """Time-domain pitch via YIN (CMND); returns (freq, dip depth) or None.

    Used only as an octave arbiter for single notes: the period estimate is
    immune to the weak-fundamental / strong-even-harmonic spectra that fool
    frequency-domain octave decisions (e.g. bridge-pickup tones).
    """
    skip = int(0.02 * sr)
    x = seg[skip : skip + int(0.25 * sr)]
    if len(x) < int(0.08 * sr):
        return None
    x = x - x.mean()
    n = len(x)
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    fx = np.fft.rfft(x, nfft)
    acf = np.fft.irfft(fx * np.conj(fx), nfft)[:n]
    cumsq = np.concatenate([[0.0], np.cumsum(x * x)])
    taus = np.arange(n)
    energy = (cumsq[n - taus] - cumsq[0]) + (cumsq[n] - cumsq[taus])
    d = np.maximum(energy - 2.0 * acf, 0.0)
    cmnd = np.ones(n)
    run = np.cumsum(d[1:])
    cmnd[1:] = d[1:] * taus[1:] / np.maximum(run, 1e-12)

    lag_lo = max(2, int(sr / fmax))
    lag_hi = min(n - 2, int(np.ceil(sr / fmin)))
    if lag_lo >= lag_hi:
        return None
    region = cmnd[lag_lo : lag_hi + 1]
    below = np.nonzero(region < 0.12)[0]
    if below.size:
        i = int(below[0])
        while i + 1 < len(region) and region[i + 1] < region[i]:
            i += 1
    else:
        i = int(np.argmin(region))
    lag = lag_lo + i
    a, b, c = cmnd[lag - 1], cmnd[lag], cmnd[lag + 1]
    den = a - 2 * b + c
    refined = lag + (0.5 * (a - c) / den if abs(den) > 1e-12 else 0.0)
    return sr / refined, float(region.min())


def detect_pitches(
    seg: np.ndarray,
    sr: int,
    fmin: float = GUITAR_FMIN,
    fmax: float = GUITAR_FMAX,
    max_polyphony: int = 6,
    rel_floor: float = 0.14,
    n_harmonics: int = 10,
    recover_hidden: bool = True,
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

    spec, bin_hz, res_hz = _spectrum(win, sr)
    spec_max = spec.max()
    if spec_max <= 1e-9:
        return []

    # Candidate fundamentals on a quarter-semitone grid.
    midi_lo = int(np.floor(freq_to_midi_float(fmin)))
    midi_hi = int(np.ceil(freq_to_midi_float(fmax)))
    cand_midis = np.arange(midi_lo, midi_hi + 0.001, 0.25)
    cand_freqs = midi_to_freq(cand_midis)
    harm_limit = min(sr / 2.0 * 0.95, 6000.0)
    weights = 1.0 / np.arange(1, max(n_harmonics, 16) + 1) ** 0.8

    work = spec.copy()
    found: list[tuple[float, float, float]] = []  # (f0, score, beta)
    first_score = None
    cand_alive = np.ones(len(cand_freqs), dtype=bool)

    for _ in range(3 * max_polyphony):
        if len(found) >= max_polyphony:
            break
        best_score, best_f, best_idx, best_beta = 0.0, 0.0, -1, 0.0
        for i, f in enumerate(cand_freqs):
            if not cand_alive[i]:
                continue
            score, beta = _eval_candidate(
                work, bin_hz, res_hz, f, harm_limit, weights, n_harmonics
            )
            if score > best_score:
                best_score, best_f, best_idx, best_beta = score, f, i, beta

        if best_idx < 0 or best_score < 1e-6:
            break
        if first_score is not None and best_score < rel_floor * first_score:
            break

        f0 = best_f

        # Half-pitch (sub-octave) guard: if odd harmonics of the chosen f0
        # are nearly absent but even ones are strong, the true pitch is 2*f0.
        odd = sum(_band_max(work, bin_hz, h * f0, res_hz)[0] for h in (1, 3, 5))
        even = sum(_band_max(work, bin_hz, h * f0, res_hz)[0] for h in (2, 4, 6))
        if even > 0 and odd < 0.15 * even and 2 * f0 <= fmax * 1.06:
            f0 *= 2.0

        # Require some energy at the fundamental itself. If it's gone (e.g.
        # an octave shadow whose energy was cancelled with an earlier note),
        # reject just this candidate and keep searching for other notes.
        a0, _ = _band_max(work, bin_hz, f0, res_hz)
        if a0 < 0.01 * spec_max:
            cand_alive[np.abs(cand_midis - freq_to_midi_float(best_f)) < 0.3] = False
            continue

        if first_score is None:
            first_score = best_score

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

        found.append((f0, best_score, best_beta))

        # Cancel this note's (stretched) harmonics before searching for the
        # next pitch.
        for h in range(1, n_harmonics + 1):
            fh = h * f0 * (1.0 + best_beta * h * h)
            if fh > harm_limit:
                break
            _zero_band(work, bin_hz, fh, res_hz)

    if not found:
        return []

    # Octave arbitration for single notes: a YIN period estimate is far
    # more reliable than spectral octave heuristics on real tones. Only an
    # exact octave move is ever allowed.
    if len(found) == 1:
        yin = _yin_pitch(seg, sr, fmin * 0.94, fmax * 1.06)
        if yin is not None:
            fy, depth = yin
            f0, score, beta = found[0]
            if depth < 0.15:
                my = freq_to_midi_float(fy)
                for target in (2.0 * f0, 0.5 * f0):
                    if (
                        abs(my - freq_to_midi_float(target)) < 0.6
                        and fmin * 0.94 <= fy <= fmax * 1.06
                    ):
                        found[0] = (fy, score, beta)
                        break

    # Validation + recovery on the original (uncancelled) spectrum:
    # greedy notes sitting on another note's harmonic comb are re-tested
    # (cancellation leftovers can masquerade as notes), and notes whose
    # every harmonic coincides with a found note's comb (octave/twelfth/
    # double-octave doublings inside chords) — invisible to greedy
    # cancellation — are hunted via envelope-excess evidence.
    if recover_hidden:
        found = _refine_polyphony(seg, sr, found, fmax, max_polyphony, rel_floor)

    top = max(s for _, s, _ in found)
    pitches: dict[int, Pitch] = {}
    for f0, score, _beta in found:
        mf = freq_to_midi_float(f0)
        midi = int(round(mf))
        cents = (mf - midi) * 100.0
        p = Pitch(freq=f0, midi=midi, cents=cents, salience=score / top)
        if midi not in pitches or p.salience > pitches[midi].salience:
            pitches[midi] = p
    return sorted(pitches.values(), key=lambda p: p.midi)


# ---------------------------------------------------------------------------
# Hidden-note recovery and nested-note validation
# ---------------------------------------------------------------------------

_N_RECOVERY_HARMONICS = 16

# Acceptance thresholds for the spectral excess ratios (measured level at
# the hidden note's partial positions over the base note's interpolated
# private-harmonic envelope). Calibrated on pluck-position/pickup
# comb-filtered synthesis: combed single notes stay near or below ~1.2;
# genuine doublings of comparable level sit >= ~1.9.
_SPEC_BOTH = 1.45     # n == 2 clean evidence harmonics: both must clear this
_SPEC_MED = 1.45      # n >= 3: median must clear this ...
_SPEC_SECOND = 1.25   # ... and the second largest this
_SPEC_ALONE = 2.4     # spectral-only bar when no temporal channel exists
_DECAY_VETO = 0.75    # evidence decaying clearly faster than anchors = not a note
_VALIDATE_MED = 1.3   # nested greedy notes below this excess are ghosts

# Evidence partial indices of the hidden note per multiplier. m=2 and m=3
# use odd j only so energy from a double octave (m=4) or an octave (whose
# partials land on the even multiples) can never support them.
_EVIDENCE_J = {2: (1, 3, 5, 7), 3: (1, 3, 5), 4: (1, 2, 3, 4)}


def _stretched_partials(f0: float, beta: float, n: int, limit: float) -> list[tuple[int, float]]:
    out = []
    for h in range(1, n + 1):
        fh = h * f0 * (1.0 + beta * h * h)
        if fh > limit:
            break
        out.append((h, fh))
    return out


def _refine_polyphony(
    seg: np.ndarray,
    sr: int,
    found: list[tuple[float, float, float]],
    fmax: float,
    max_polyphony: int,
    rel_floor: float,
) -> list[tuple[float, float, float]]:
    """Validate nested greedy notes and recover hidden doublings.

    Both jobs use the same physics: a real note at m*f0 adds energy at its
    partial positions j*m*f0 beyond what the base note's spectral envelope
    (log-log interpolated between its private harmonics) predicts.

    Validation: a greedy-found note whose fundamental sits on an integer
    multiple of another found note can be a cancellation leftover of that
    note's comb rather than a real string; if its excess evidence is weak
    it is dropped.

    Recovery: octave/twelfth/double-octave doublings inside chords share
    every partial with the base note and are invisible to greedy
    cancellation; they are accepted only on multi-harmonic excess evidence
    (and, for long segments, a decay veto: evidence that dies faster than
    the base's own envelope belongs to the base, not to a hidden string —
    while a static body/pickup resonance never shows excess on two widely
    separated partials at once).
    """
    debug = bool(os.environ.get("FRETMAP_DEBUG"))
    skip = int(0.012 * sr)
    win = seg[skip : skip + int(0.40 * sr)]
    if len(win) < int(0.18 * sr) or not found:
        return found
    spec, bin_hz, res_hz = _spectrum(win, sr)
    if spec.max() <= 1e-9:
        return found
    harm_limit = min(sr / 2.0 * 0.95, 6000.0)
    weights = 1.0 / np.arange(1, _N_RECOVERY_HARMONICS + 1) ** 0.8

    # Temporal channel: early/late short windows (only for long segments).
    n_short = int(0.20 * sr)
    early_start = skip + int(0.03 * sr)
    late_start = skip + int(0.25 * sr)
    have_temporal = len(seg) >= late_start + n_short
    if have_temporal:
        spec_e, bin_e, res_e = _spectrum(seg[early_start : early_start + n_short], sr)
        spec_l, bin_l, res_l = _spectrum(seg[late_start : late_start + n_short], sr)

    def decay(fh: float) -> float | None:
        ae, _ = _band_max(spec_e, bin_e, fh, res_e)
        al, _ = _band_max(spec_l, bin_l, fh, res_l)
        if ae <= 1e-12:
            return None
        return al / ae

    def collision_tol(fh: float) -> float:
        return max(0.005 * fh, 1.5 * res_hz)

    class _Note:
        def __init__(self, f0: float, score: float, beta: float):
            self.f0, self.score = f0, score
            # Refit the stiff-string stretch on the long window: collision
            # geometry between combs (e.g. B2's partials separating from
            # E2's at high h) needs the precision the short greedy window
            # cannot give.
            a0, _ = _band_max(spec, bin_hz, f0, res_hz, tol=0.005)
            rels = []
            for h in range(4, _N_RECOVERY_HARMONICS + 1):
                fh = h * f0
                if fh * (1.0 + MAX_BETA * h * h) > harm_limit:
                    break
                amp, k = _band_max(spec, bin_hz, fh * (1.0 + 0.5 * MAX_BETA * h * h),
                                   res_hz, tol=0.005 + 0.5 * MAX_BETA * h * h)
                if amp < 0.05 * max(a0, 1e-12):
                    continue
                rel = (k * bin_hz - fh) / fh
                if rel >= -0.002:
                    rels.append(rel / (h * h))
            self.fit_ok = len(rels) >= 3
            if self.fit_ok:
                beta = min(max(float(np.median(rels)), 0.0), MAX_BETA)
            self.beta = beta
            self.rebuild()

        def rebuild(self):
            self.partials = dict(
                _stretched_partials(self.f0, self.beta, _N_RECOVERY_HARMONICS, harm_limit)
            )
            self.amps = {}
            for h, fh in self.partials.items():
                self.amps[h], _ = _band_max(spec, bin_hz, fh, res_hz, tol=0.005)
            self.a_max = max(self.amps.values(), default=0.0)
            self.long_score = sum(
                weights[h - 1] * a for h, a in self.amps.items()
            )

    notes = sorted((_Note(*t) for t in found), key=lambda x: x.f0)
    if not notes or max(x.long_score for x in notes) <= 0:
        return found

    # String stiffness is a property of the instrument: when a note's own
    # stretch fit failed (its high partials are buried under the rest of
    # the chord), fall back to the median fit of the notes that could be
    # measured. Without this, an unstretched comb assumption keeps e.g.
    # E2's h9 "colliding" with B3's h3 forever.
    fitted = [x.beta for x in notes if x.fit_ok and x.beta > 1e-5]
    if debug:
        print("  betas:", [(round(x.f0, 1), round(x.beta * 1e6), x.fit_ok) for x in notes])
    if fitted:
        shared = float(np.median(fitted))
        for x in notes:
            # A fit landing at ~zero on a guitar whose other strings
            # measure real stiffness is contamination, not physics.
            if not x.fit_ok or x.beta < 1e-5:
                x.beta = shared
                x.rebuild()
    top_long = max(x.long_score for x in notes)
    top_greedy = max(x.score for x in notes)
    hidden: list[tuple[float, dict[int, float], float]] = []  # (freq, excess by j, score)

    def collides(f: float, exclude: tuple) -> bool:
        for other in notes:
            if other in exclude:
                continue
            for fg in other.partials.values():
                if abs(fg - f) < collision_tol(f):
                    return True
        return False

    def _note_envelope(note: "_Note", f: float) -> float | None:
        """Upper estimate of `note`'s spectral level at frequency f from
        its measured partial amplitudes (log-log interpolation between the
        nearest measured partials; nearest-amp beyond the ends)."""
        pts = sorted(
            (fh, note.amps[h]) for h, fh in note.partials.items() if note.amps[h] > 0
        )
        if not pts:
            return 0.0
        lower = [(pf, pa) for pf, pa in pts if pf < f]
        upper = [(pf, pa) for pf, pa in pts if pf > f]
        if not lower:
            return upper[0][1]
        if not upper:
            return lower[-1][1]
        (f1, a1), (f2, a2) = lower[-1], upper[0]
        slope = np.log(a2 / a1) / np.log(f2 / f1)
        slope = min(max(slope, -2.0), 1.0)
        return float(a1 * (f / f1) ** slope)

    def evidence_for(
        base: "_Note", f_hidden: float, js: tuple[int, ...], subject: "_Note | None" = None
    ) -> list[tuple[int, float, float, float]]:
        """Usable excess evidence (j, freq, amp, ratio) for a note at
        f_hidden judged against the base note's envelope. `subject` is the
        already-found note being validated, if any — its own comb must not
        count as a collision against its own evidence."""
        exclude = (base, subject) if subject is not None else (base,)
        anchors = sorted(
            h
            for h, fh in base.partials.items()
            if abs(round(fh / f_hidden) * f_hidden - fh) > collision_tol(fh)
            and base.amps[h] >= 0.03 * base.a_max
            and not collides(fh, exclude)
        )
        if len(anchors) < 2:
            return []

        def predict(f: float) -> float | None:
            lower = [h for h in anchors if base.partials[h] < f]
            upper = [h for h in anchors if base.partials[h] > f]
            if not lower:
                return None  # never extrapolate below the first anchor
            h1 = max(lower)
            f1, a1 = base.partials[h1], base.amps[h1]
            if upper:
                h2 = min(upper)
                f2, a2 = base.partials[h2], base.amps[h2]
            elif len(lower) >= 2:
                # Extrapolate a decaying envelope above the last anchor.
                h2 = h1
                h1 = max(h for h in lower if h != h2)
                f1, a1 = base.partials[h1], base.amps[h1]
                f2, a2 = base.partials[h2], base.amps[h2]
            else:
                return None
            if a1 <= 0 or a2 <= 0:
                return None
            slope = np.log(a2 / a1) / np.log(f2 / f1)
            slope = min(max(slope, -2.0), 1.0 if upper else 0.0)
            ref_f, ref_a = (f1, a1) if upper else (f2, a2)
            pred = ref_a * (f / ref_f) ** slope
            # Contribution of already-accepted hidden notes whose partials
            # land here (an accepted octave explains part of what a
            # double-octave hypothesis would see).
            for g, excess, _s in hidden:
                j = round(f / g)
                if j >= 1 and abs(j * g - f) < collision_tol(f):
                    pred += excess.get(j, excess.get(1, 0.0) / j)
            return pred

        def evidence_amp(f_ev: float) -> float | None:
            """Amplitude attributable to a partial at f_ev, or None when
            it cannot be separated from another note's partial.

            Another note's k-th partial may sit anywhere between k*g
            (unstretched) and its maximally stretched position. Where such
            a range comes near f_ev, the evidence is usable only if the
            spectrum shows two *resolved* peaks — one at f_ev (the hidden
            note is a plain string: negligible stretch) and a distinct one
            for the conflicting partial. A single merged peak is
            unattributable. This is decided from the measured spectrum, so
            it needs no per-note stiffness estimate at all."""
            tol = collision_tol(f_ev)
            conflicts = []
            for other in notes:
                if other in exclude:
                    continue
                for k in other.partials:
                    lo = k * other.f0
                    hi = lo * (1.0 + MAX_BETA * k * k)
                    if lo - tol < f_ev < hi + tol:
                        conflicts.append((other, lo, hi))
            if not conflicts:
                amp, _k = _band_max(spec, bin_hz, f_ev, res_hz, tol=0.005)
                return amp
            span_lo = min(f_ev, min(c[1] for c in conflicts)) - 3.0 * res_hz
            span_hi = max(f_ev, max(c[2] for c in conflicts)) + 3.0 * res_hz
            ilo = max(1, int(span_lo / bin_hz))
            ihi = min(len(spec) - 2, int(span_hi / bin_hz) + 1)
            sl = spec[ilo : ihi + 1]
            if sl.size < 3 or sl.max() <= 0:
                return None
            # Local maxima with simple prominence/separation suppression.
            cand = (
                np.nonzero((sl[1:-1] >= sl[:-2]) & (sl[1:-1] > sl[2:]))[0] + 1 + ilo
            )
            cand = [k for k in cand if spec[k] >= 0.08 * sl.max()]
            peaks: list[int] = []
            for k in sorted(cand, key=lambda k: -spec[k]):
                if all(abs(k - p) * bin_hz >= 1.2 * res_hz for p in peaks):
                    peaks.append(k)
            if not peaks:
                return None
            freqs = [(_parabolic_peak(spec, k) * bin_hz, spec[k]) for k in peaks]
            near = min(freqs, key=lambda t: abs(t[0] - f_ev))
            if abs(near[0] - f_ev) > 1.5 * res_hz:
                return None
            # Every conflicting partial must either be visible as its own
            # resolved peak, or be provably too weak (from its note's own
            # measured envelope) to explain ours — in which case its
            # estimated contribution is subtracted. Otherwise our peak may
            # *be* the conflictor.
            amp = float(near[1])
            for other, lo, hi in conflicts:
                if any(
                    lo - 1.5 * res_hz <= pf <= hi + 1.5 * res_hz
                    and abs(pf - near[0]) >= 1.5 * res_hz
                    for pf, _pa in freqs
                ):
                    continue
                est = _note_envelope(other, f_ev)
                if est is None or est >= 0.5 * float(near[1]):
                    return None
                amp -= est
            return max(amp, 0.0)

        usable = []
        trace = os.environ.get("FRETMAP_DEBUG") == "2"
        if trace:
            print(f"    [trace base={base.f0:.1f} hid={f_hidden:.1f} anchors={anchors}]")
        for j in js:
            f_ev = j * f_hidden
            if f_ev > harm_limit:
                break
            amp = evidence_amp(f_ev)
            if amp is None:
                if trace:
                    print(f"      j{j} {f_ev:.1f}: merged/unresolved")
                continue
            pred = predict(f_ev)
            if pred is None:
                if trace:
                    print(f"      j{j} {f_ev:.1f}: no envelope")
                continue
            if amp < 0.02 * base.a_max:
                if trace:
                    print(f"      j{j} {f_ev:.1f}: too weak {amp:.3g} < {0.02*base.a_max:.3g}")
                continue
            usable.append((j, f_ev, amp, amp / max(pred, 1e-12)))
        return usable

    # --- Validation: drop greedy notes that are cancellation ghosts -------
    keep = []
    for note in notes:
        ghost = False
        for base in notes:
            if base is note or base.f0 >= note.f0:
                continue
            k = note.f0 / base.f0
            if abs(k - round(k)) > 0.02 * k or round(k) < 2:
                continue
            usable = evidence_for(base, note.f0, tuple(range(1, 7)), subject=note)
            if len(usable) < 2:
                continue  # inconclusive: keep the note
            med = float(np.median([u[3] for u in usable]))
            if debug:
                print(
                    f"  validate {note.f0:.1f} vs {base.f0:.1f}: "
                    f"ev={[(u[0], round(u[3], 2)) for u in usable]} med={med:.2f}"
                )
            if med < _VALIDATE_MED:
                ghost = True
                break
        if not ghost:
            keep.append(note)
    notes = keep

    # --- Recovery: hypothesize hidden doublings ---------------------------
    def near_existing(f: float) -> bool:
        mf = freq_to_midi_float(f)
        return any(
            abs(freq_to_midi_float(g) - mf) < 0.6
            for g in [x.f0 for x in notes] + [a[0] for a in hidden]
        )

    for base in notes:
        for m in (2, 3, 4):
            if len(notes) + len(hidden) >= max_polyphony:
                break
            f_hidden = m * base.f0
            if f_hidden > fmax * 1.02 or near_existing(f_hidden):
                continue
            usable = evidence_for(base, f_hidden, _EVIDENCE_J[m])
            if len(usable) < 2:
                continue
            ratios = sorted((u[3] for u in usable), reverse=True)
            n = len(ratios)
            med = float(np.median(ratios))
            if debug:
                print(
                    f"  base={base.f0:.1f} m={m} "
                    f"ev={[(u[0], round(u[3], 2)) for u in usable]} med={med:.2f}"
                )
            if n == 2:
                # Either both harmonics clearly in excess, or one decisive
                # excess (a dominant hidden fundamental) plus a consistent
                # second.
                spectral = min(ratios) >= _SPEC_BOTH or (
                    ratios[0] >= 2.8 and ratios[1] >= 1.15
                )
            else:
                spectral = med >= _SPEC_MED and ratios[1] >= _SPEC_SECOND
            if not spectral:
                continue

            if have_temporal:
                ev_decays = [d for _j, f, _a, _r in usable if (d := decay(f)) is not None]
                anchor_fs = [
                    fh
                    for h, fh in base.partials.items()
                    if base.amps[h] >= 0.03 * base.a_max and not collides(fh, (base,))
                ]
                an_decays = [d for f in anchor_fs if (d := decay(f)) is not None]
                if ev_decays and an_decays:
                    ok = float(np.median(ev_decays)) >= _DECAY_VETO * float(
                        np.median(an_decays)
                    )
                    if debug and not ok:
                        print(
                            f"    decay veto ev={np.median(ev_decays):.3f} "
                            f"anch={np.median(an_decays):.3f}"
                        )
                    if not ok and med < _SPEC_ALONE:
                        continue
            elif med < _SPEC_ALONE and not (n == 2 and min(ratios) >= _SPEC_ALONE * 0.8):
                continue

            # Salience economy: the excess energy must be comparable to a
            # genuinely detected quiet note.
            excess = {}
            for j, f_ev, amp, ratio in usable:
                excess[j] = max(amp * (1.0 - 1.0 / max(ratio, 1.0)), 0.0)
            hidden_long = sum(weights[j - 1] * e for j, e in excess.items())
            # Overwhelming ratio evidence is its own economy: a note whose
            # fundamental is merged into a base harmonic only shows weak
            # absolute excess at its high partials.
            if med < 2.0 * _SPEC_MED and hidden_long < rel_floor * top_long * 0.5:
                if debug:
                    print(f"    economy reject {hidden_long:.3g}")
                continue

            # Refine the hidden frequency from its fundamental's peak when
            # that bin was usable; otherwise derive it from the base note.
            f_ref = f_hidden
            if usable[0][0] == 1:
                amp, k = _band_max(spec, bin_hz, f_hidden, res_hz, tol=0.005)
                if amp > 0:
                    cand = _parabolic_peak(spec, k) * bin_hz
                    if (
                        abs(freq_to_midi_float(max(cand, 1.0)) - freq_to_midi_float(f_hidden))
                        < 0.5
                    ):
                        f_ref = cand

            out_score = hidden_long / top_long * top_greedy
            hidden.append((f_ref, excess, out_score))

    return [(x.f0, x.score, x.beta) for x in notes] + [
        (f, s, 0.0) for f, _e, s in hidden
    ]


def estimate_tuning_offset(pitches: list[Pitch]) -> float:
    """Global tuning offset of a track in cents, from all detected pitches.

    A guitar tuned somewhat off A440 puts every note the same distance from
    the equal-tempered grid; rounding each note independently then flips
    notes near the +-50 cent boundary inconsistently. The circular mean of
    the per-note deviations (cents wrap at +-50) gives one consistent
    offset to subtract before rounding.
    """
    if not pitches:
        return 0.0
    angles = np.array([p.cents for p in pitches]) * (2 * np.pi / 100.0)
    weights = np.array([max(p.salience, 1e-3) for p in pitches])
    z = np.sum(weights * np.exp(1j * angles))
    if abs(z) < 1e-9:
        return 0.0
    return float(np.angle(z) * (100.0 / (2 * np.pi)))
