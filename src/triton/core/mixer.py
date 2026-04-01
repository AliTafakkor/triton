"""Core audio mixing utilities.

Dependency-light math for speech-in-noise mixing.
"""

from __future__ import annotations

import numpy as np

from triton.core.io import normalize_peak, normalize_rms, rms


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

	speech_rms = float(rms(speech))
	noise_rms = float(rms(noise))
	if noise_rms <= 0:
		raise ValueError("Noise RMS is zero; cannot scale to target SNR.")

	target_noise_rms = speech_rms / (10 ** (snr_db / 20.0))
	scale = target_noise_rms / noise_rms

	scaled_noise = noise * scale
	mixed = speech + scaled_noise

	return normalize_peak(mixed)


def mix_babble(
	talkers: list[np.ndarray],
	target_rms: float | None = None,
	*,
	peak_normalize: bool = True,
	normalize_talkers: bool = True,
) -> np.ndarray:
	"""Mix multiple talkers into babble speech.

	The talkers are matched to the same length by tiling shorter inputs. When
	normalize_talkers is enabled, each talker is RMS-normalized before mixing.

	Args:
		talkers: List of talker waveform arrays (1D or 2D for multi-channel).
		target_rms: Target RMS level for each talker before mixing. If None,
			the talkers are mixed at their current levels.
		peak_normalize: Whether to peak-normalize the mixed output.
		normalize_talkers: Whether to RMS-normalize each talker before mixing.

	Returns:
		Mixed babble waveform.

	Raises:
		ValueError: If talkers list is empty or any talker has zero length.
	"""
	if not talkers:
		raise ValueError("At least one talker is required.")

	talkers = [np.asarray(t, dtype=np.float32) for t in talkers]

	max_length = max(t.shape[-1] for t in talkers)
	if max_length == 0:
		raise ValueError("All talkers must have non-zero length.")

	talkers = [_match_length(t, max_length) for t in talkers]

	if normalize_talkers:
		if target_rms is None:
			target_rms = 0.1
		if target_rms <= 0:
			raise ValueError("Target RMS must be positive.")
		talkers = [normalize_rms(talker, target=float(target_rms)) for talker in talkers]

	mixed = np.sum(talkers, axis=0)
	return normalize_peak(mixed) if peak_normalize else mixed


def mix_babble_from_segments(
	talker_segments: list[list[np.ndarray]],
	target_rms: float | None = None,
	*,
	peak_normalize: bool = True,
) -> np.ndarray:
	"""Build and mix babble from grouped talker segments.

	Each segment is RMS-normalized before concatenating the segments for a talker.
	The resulting talker tracks are then mixed without re-normalizing the tracks.
	"""
	if not talker_segments:
		raise ValueError("At least one talker is required.")

	concatenated_talkers: list[np.ndarray] = []
	for segments in talker_segments:
		if not segments:
			raise ValueError("Each talker must contain at least one audio file.")

		normalized_segments: list[np.ndarray] = []
		for segment in segments:
			segment_array = np.asarray(segment, dtype=np.float32)
			if segment_array.shape[-1] == 0:
				raise ValueError("Talker segments must have non-zero length.")
			if target_rms is not None:
				if target_rms <= 0:
					raise ValueError("Target RMS must be positive.")
				segment_array = normalize_rms(segment_array, target=float(target_rms))
			normalized_segments.append(segment_array)

		concatenated_talkers.append(np.concatenate(normalized_segments, axis=-1))

	return mix_babble(
		concatenated_talkers,
		target_rms=None,
		peak_normalize=peak_normalize,
		normalize_talkers=False,
	)
