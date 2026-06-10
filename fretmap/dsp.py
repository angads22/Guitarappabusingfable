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
