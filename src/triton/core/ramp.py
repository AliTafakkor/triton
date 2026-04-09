"""Audio ramp (fade-in / fade-out) utilities.

Supported ramp shapes
---------------------
linear      Uniform gain sweep from 0 → 1 (or 1 → 0).
exponential Gain accelerates toward the target — slow start, fast finish.
logarithmic Gain accelerates away from zero — fast start, slow finish.
cosine      Smooth S-shaped half-cosine (Hann-like) transition.
"""

from __future__ import annotations

import numpy as np

RAMP_SHAPES: tuple[str, ...] = ("linear", "exponential", "logarithmic", "cosine")


def _ramp_curve(n_samples: int, shape: str) -> np.ndarray:
    """Return a monotonically increasing ramp from 0 to 1 of length *n_samples*.

    Args:
        n_samples: Number of samples in the ramp.
        shape: One of ``linear``, ``exponential``, ``logarithmic``, ``cosine``.

    Returns:
        1-D float32 array of length *n_samples* with values in ``[0, 1]``.

    Raises:
        ValueError: If *shape* is not recognised.
    """
    if n_samples == 0:
        return np.empty(0, dtype=np.float32)

    t = np.linspace(0.0, 1.0, n_samples, dtype=np.float64)

    if shape == "linear":
        curve = t

    elif shape == "exponential":
        # Exponential: slow start, fast finish.
        # Use base-10: curve = (10^t - 1) / (10 - 1)
        k = 10.0
        curve = (np.power(k, t) - 1.0) / (k - 1.0)

    elif shape == "logarithmic":
        # Logarithmic: fast start, slow finish (inverse of exponential).
        # curve = log10(1 + t * (k-1)) / log10(k)
        k = 10.0
        curve = np.log10(1.0 + t * (k - 1.0)) / np.log10(k)

    elif shape == "cosine":
        # Half-cosine (raised-cosine): smooth S-curve transition.
        curve = 0.5 * (1.0 - np.cos(t * np.pi))

    else:
        raise ValueError(
            f"Unknown ramp shape '{shape}'. Choose from: {', '.join(RAMP_SHAPES)}."
        )

    return curve.astype(np.float32)


def apply_ramp(
    audio: np.ndarray,
    sr: int,
    *,
    ramp_start: float = 0.0,
    ramp_end: float = 0.0,
    shape: str = "cosine",
) -> np.ndarray:
    """Apply a fade-in and/or fade-out envelope to *audio*.

    The ramp shape is applied symmetrically: the fade-in uses the ascending
    curve and the fade-out uses the descending mirror.  Both mono (1-D) and
    multi-channel (2-D, shape ``(n_samples, channels)``) arrays are supported.

    Args:
        audio: Input waveform as a float32 NumPy array.
        sr: Sample rate in Hz.
        ramp_start: Duration of the fade-in in seconds (0 = no fade-in).
        ramp_end: Duration of the fade-out in seconds (0 = no fade-out).
        shape: Ramp shape — one of ``linear``, ``exponential``,
               ``logarithmic``, or ``cosine``.

    Returns:
        A copy of *audio* with the ramp envelope applied.

    Raises:
        ValueError: If *shape* is unrecognised, if either duration is negative,
                    or if the combined ramp duration exceeds the audio length.
    """
    if ramp_start < 0:
        raise ValueError("ramp_start must be >= 0.")
    if ramp_end < 0:
        raise ValueError("ramp_end must be >= 0.")

    audio = np.asarray(audio, dtype=np.float32)
    n_total = audio.shape[0]
    n_start = int(round(ramp_start * sr))
    n_end = int(round(ramp_end * sr))

    if n_start + n_end > n_total:
        raise ValueError(
            f"Combined ramp duration ({ramp_start + ramp_end:.3f}s) exceeds "
            f"audio duration ({n_total / sr:.3f}s)."
        )

    result = audio.copy()

    if n_start > 0:
        fade_in = _ramp_curve(n_start, shape)
        if result.ndim == 1:
            result[:n_start] *= fade_in
        else:
            result[:n_start] *= fade_in[:, np.newaxis]

    if n_end > 0:
        fade_out = _ramp_curve(n_end, shape)[::-1].copy()
        if result.ndim == 1:
            result[n_total - n_end:] *= fade_out
        else:
            result[n_total - n_end:] *= fade_out[:, np.newaxis]

    return result
