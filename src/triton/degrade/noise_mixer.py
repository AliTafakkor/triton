"""Backward-compatible noise mixing helpers.

This module preserves the historical `triton.degrade.noise_mixer` import path
while delegating the implementation to the canonical core mixer.
"""

from __future__ import annotations

import numpy as np

from triton.core.mixer import mix_at_snr


def add_noise(target: np.ndarray, noise: np.ndarray, *, snr_db: float) -> np.ndarray:
	"""Mix target audio with noise at the requested SNR in dB.

	Args:
		target: Target waveform array.
		noise: Noise waveform array.
		snr_db: Desired signal-to-noise ratio in dB.

	Returns:
		Normalized mixed waveform.
	"""
	return mix_at_snr(target, noise, snr_db)
