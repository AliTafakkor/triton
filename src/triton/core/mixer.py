"""Core audio mixing utilities.

Dependency-light math for speech-in-noise mixing.
"""

from __future__ import annotations

import librosa
import numpy as np


def _rms(signal: np.ndarray, axis: int = -1) -> np.ndarray:
	"""Compute root mean square along an axis."""
	signal = np.asarray(signal, dtype=np.float32)
	return np.sqrt(np.mean(np.square(signal), axis=axis))


def _match_length(noise: np.ndarray, target_length: int) -> np.ndarray:
	"""Tile or crop noise to match target length along the last axis."""
	noise = np.asarray(noise, dtype=np.float32)
	if noise.shape[-1] == 0:
		raise ValueError("Noise must have non-zero length.")

	if noise.shape[-1] < target_length:
		repeats = int(np.ceil(target_length / noise.shape[-1]))
		reps = [1] * noise.ndim
		reps[-1] = repeats
		noise = np.tile(noise, reps)

	return noise[..., :target_length]


def mix_at_snr(speech: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
	"""Mix speech and noise at a target SNR (dB) using RMS scaling.

	Args:
		speech: Speech waveform array.
		noise: Noise waveform array.
		snr_db: Target signal-to-noise ratio in dB.

	Returns:
		Mixed waveform normalized to max amplitude 1.0.
	"""
	speech = np.asarray(speech, dtype=np.float32)
	noise = np.asarray(noise, dtype=np.float32)

	if speech.shape[-1] == 0:
		raise ValueError("Speech must have non-zero length.")

	noise = _match_length(noise, speech.shape[-1])

	speech_rms = _rms(speech)
	noise_rms = _rms(noise)

	if np.any(noise_rms == 0):
		raise ValueError("Noise RMS is zero; cannot scale to target SNR.")

	target_noise_rms = speech_rms / (10 ** (snr_db / 20.0))
	scale = target_noise_rms / noise_rms

	scaled_noise = noise * scale
	mixed = speech + scaled_noise

	peak = np.max(np.abs(mixed))
	if peak > 0:
		return librosa.util.normalize(mixed, norm=np.inf)

	return mixed
