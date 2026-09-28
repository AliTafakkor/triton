"""Acoustic features: broadband amplitude envelope."""

from __future__ import annotations

from math import gcd

import numpy as np
from scipy.signal import butter, hilbert, resample_poly, sosfiltfilt

from triton.core.signal import to_mono_float32


def envelope_feature(
	audio: np.ndarray,
	sr: int,
	*,
	rate: int = 100,
	cutoff: float = 30.0,
	filter_order: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
	"""Compute the broadband amplitude envelope and resample it to a feature rate.

	Uses the magnitude of the analytic signal, ``|hilbert(x)|``, low-passed at
	``cutoff`` (zero-phase) and resampled to ``rate``: the standard envelope for
	EEG/MEG encoding models. This intentionally differs from
	``triton.core.signal.extract_envelope(method="hilbert")``, which half-wave
	rectifies first; the DC offset that rectification introduces makes the Hilbert
	envelope rise well before a sound actually starts.

	Args:
		audio: Mono or channel-first waveform.
		sr: Audio sample rate (Hz).
		rate: Output feature rate (Hz). ~100 Hz is typical for EEG/MEG.
		cutoff: Low-pass cutoff for envelope smoothing (Hz).
		filter_order: Butterworth order of the smoothing filter.

	Returns:
		(times, values): sample times in seconds and envelope values, both float32.
	"""
	if rate <= 0 or rate > sr:
		raise ValueError(f"Feature rate must be between 1 and the audio sample rate ({sr} Hz).")

	envelope = np.abs(hilbert(to_mono_float32(audio)))
	sos = butter(filter_order, min(cutoff, sr / 2 * 0.95), btype="lowpass", fs=sr, output="sos")
	envelope = sosfiltfilt(sos, envelope)

	divisor = gcd(int(rate), int(sr))
	values = resample_poly(envelope, int(rate) // divisor, int(sr) // divisor).astype(np.float32)
	times = (np.arange(values.size) / float(rate)).astype(np.float32)
	return times, values
