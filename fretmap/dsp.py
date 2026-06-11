"""Low-level DSP: STFT magnitudes and onset detection via spectral flux."""

import numpy as np


def stft_mag(x: np.ndarray, n_fft: int = 2048, hop: int = 512) -> np.ndarray:
    """Magnitude spectrogram, shape (frames, n_fft // 2 + 1)."""
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)))
    window = np.hanning(n_fft)
    frames = np.lib.stride_tricks.sliding_window_view(x, n_fft)[::hop]
    return np.abs(np.fft.rfft(frames * window, axis=1))


def _moving_average(x: np.ndarray, width: int) -> np.ndarray:
    width = max(1, width)
    kernel = np.ones(width) / width
    return np.convolve(x, kernel, mode="same")


def detect_onsets(
    x: np.ndarray,
    sr: int,
    n_fft: int = 2048,
    hop: int = 512,
    min_sep: float = 0.06,
) -> list[int]:
    """Return onset positions in samples, sorted ascending.

    Uses positive spectral flux of the log-magnitude spectrogram with an
    adaptive (local mean) threshold and minimum peak separation.
    """
    if len(x) < n_fft:
        return [0] if np.max(np.abs(x), initial=0.0) > 1e-4 else []

    # Prepend silence so an attack right at sample 0 still produces a
    # spectral-flux rise (flux needs a "before" frame).
    pad = n_fft
    x = np.concatenate([np.zeros(pad), x])

    mags = stft_mag(x, n_fft=n_fft, hop=hop)
    logm = np.log1p(10.0 * mags)
    diff = np.diff(logm, axis=0)
    flux = np.maximum(diff, 0.0).sum(axis=1)
    flux = np.concatenate([[0.0], flux])
    peak_flux = flux.max()
    if peak_flux <= 0:
        return []
    flux = flux / peak_flux
    flux = _moving_average(flux, 3)

    local = _moving_average(flux, max(3, int(0.5 * sr / hop)))
    thresh = 1.3 * local + 0.05

    # Frame RMS gate so noise-floor flux in near-silence is ignored.
    rms = np.sqrt((mags**2).mean(axis=1))
    rms_gate = 0.02 * rms.max()

    half = 3
    candidates = []
    for i in range(1, len(flux) - 1):
        lo, hi = max(0, i - half), min(len(flux), i + half + 1)
        if flux[i] >= flux[lo:hi].max() and flux[i] > thresh[i] and rms[i] > rms_gate:
            candidates.append(i)

    # Keep the strongest peak within any min_sep window.
    min_frames = max(1, int(min_sep * sr / hop))
    accepted: list[int] = []
    for i in sorted(candidates, key=lambda i: -flux[i]):
        if all(abs(i - j) >= min_frames for j in accepted):
            accepted.append(i)
    accepted.sort()
    return [max(0, i * hop - pad) for i in accepted]


def harmonic_rise(
    x: np.ndarray,
    sr: int,
    onset: int,
    freq: float,
    n_harmonics: int = 4,
    win: float = 0.06,
    ratio: float = 1.8,
) -> bool:
    """Did energy near `freq`'s harmonics rise at `onset` (samples)?

    Compares short windows just before and just after the onset. A string
    still ringing from an earlier event only decays across the onset
    (no rise); a re-struck or new note jumps well above the pre-onset
    level. Used to tell sustained notes from re-attacks.
    """
    n = int(win * sr)
    skip = int(0.015 * sr)  # let the broadband pluck transient pass
    if onset < n:
        return True  # nothing before the onset: necessarily a new note
    after = x[onset + skip : onset + skip + n]
    if len(after) < n // 2:
        return True
    before = x[onset - n : onset]
    m = min(len(before), len(after))
    w = np.hanning(m)
    fb = np.abs(np.fft.rfft(before[:m] * w))
    fa = np.abs(np.fft.rfft(after[:m] * w))
    bin_hz = sr / m

    def band_energy(spec: np.ndarray, fh: float) -> float:
        half_hz = max(0.03 * fh, 2.0 * bin_hz)
        lo = max(0, int((fh - half_hz) / bin_hz))
        hi = min(len(spec) - 1, int((fh + half_hz) / bin_hz) + 1)
        return float(spec[lo : hi + 1].max(initial=0.0))

    e_before = e_after = 0.0
    for h in range(1, n_harmonics + 1):
        fh = h * freq
        if fh >= sr / 2:
            break
        e_before += band_energy(fb, fh)
        e_after += band_energy(fa, fh)
    if e_before <= 1e-12:
        return True
    return e_after > ratio * e_before


def trim_to_sounding(seg: np.ndarray, sr: int, rel: float = 0.05) -> np.ndarray:
    """Trim trailing near-silence from a segment using an RMS envelope."""
    hop = max(1, int(0.01 * sr))
    n = len(seg) // hop
    if n < 2:
        return seg
    frames = seg[: n * hop].reshape(n, hop)
    rms = np.sqrt((frames**2).mean(axis=1))
    gate = rel * rms.max()
    sounding = np.nonzero(rms > gate)[0]
    if sounding.size == 0:
        return seg
    return seg[: (sounding[-1] + 1) * hop]
