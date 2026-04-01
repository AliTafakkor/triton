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





def mix_at_snr(speech: np.ndarray, noise: np.ndarray, snr_db: float, target_rms: float | None = None) -> np.ndarray:
	"""Mix speech and noise at a target SNR (dB) using symmetric amplitude scaling.

	Implements the following procedure:
	1. Normalize both signals independently to the same RMS level before mixing.
	2. Split the target SNR symmetrically: boost signal by SNR/2 dB and attenuate
	   noise by SNR/2 dB. Use the SPL/amplitude convention: multiplier = 10^(dB/20).
	3. Mix as a weighted sum: mixed = signal * multS + noise * multN
	4. Re-normalize the result to the target RMS. This corrects for energy
	   accumulation without altering the SNR, since both components are scaled equally.
	5. The SNR is defined by the ratio of the two multipliers, not absolute levels.

	Args:
		speech: Speech waveform array.
		noise: Noise waveform array.
		snr_db: Target signal-to-noise ratio in dB.
		target_rms: Target RMS level for each signal before mixing. If None, uses
			the RMS of the speech signal.

	Returns:
		Mixed waveform normalized to target RMS.
	"""
	speech = np.asarray(speech, dtype=np.float32)
	noise = np.asarray(noise, dtype=np.float32)

	if speech.shape[-1] == 0:
		raise ValueError("Speech must have non-zero length.")

	noise = _match_length(noise, speech.shape[-1])

	# Step 1: Normalize both signals independently to the same RMS level
	speech_rms = float(rms(speech))
	noise_rms = float(rms(noise))

	if speech_rms <= 0:
		raise ValueError("Speech RMS is zero; cannot mix.")
	if noise_rms <= 0:
		raise ValueError("Noise RMS is zero; cannot mix.")

	# Use speech RMS as reference if no target specified
	if target_rms is None:
		target_rms = speech_rms

	normalized_speech = normalize_rms(speech, target_rms)
	normalized_noise = normalize_rms(noise, target_rms)

	# Step 2: Split the SNR symmetrically
	# SNR = 20 * log10(mult_signal / mult_noise)
	# For symmetric split: mult_signal = 10^(SNR/2/20) and mult_noise = 10^(-SNR/2/20)
	half_snr_linear = 10 ** (snr_db / 2 / 20)
	mult_speech = half_snr_linear
	mult_noise = 1 / half_snr_linear

	# Step 3: Mix as weighted sum
	mixed = normalized_speech * mult_speech + normalized_noise * mult_noise

	# Step 4: Re-normalize to target RMS
	# This corrects for energy accumulation without altering the SNR
	mixed = normalize_rms(mixed, target_rms)

	return mixed


def mix_at_snr_segmented(
	speech: np.ndarray,
	noise: np.ndarray,
	snr_db_array: np.ndarray | list[float],
	segment_samples: int,
	target_rms: float | None = None,
	smooth_boundaries: bool = True,
) -> np.ndarray:
	"""Mix speech and noise with varying SNR across segments, optionally smoothing boundaries.

	If multiple SNR levels are applied to consecutive segments, smooth the multiplier
	vectors across segment boundaries to avoid abrupt transitions.

	Args:
		speech: Speech waveform array.
		noise: Noise waveform array.
		snr_db_array: Array of target SNR values in dB for each segment.
		segment_samples: Number of samples per segment.
		target_rms: Target RMS level for each signal before mixing. If None, uses
			the RMS of the speech signal.
		smooth_boundaries: If True, apply linear interpolation across segment
			boundaries to smooth multiplier transitions.

	Returns:
		Mixed waveform with variable SNR, normalized to target RMS.
	"""
	speech = np.asarray(speech, dtype=np.float32)
	noise = np.asarray(noise, dtype=np.float32)
	snr_db_array = np.asarray(snr_db_array, dtype=np.float32)

	if speech.shape[-1] == 0:
		raise ValueError("Speech must have non-zero length.")

	noise = _match_length(noise, speech.shape[-1])

# Step 1: Normalize both signals independently to the same RMS level
	speech_rms = float(rms(speech))
	noise_rms = float(rms(noise))

	if speech_rms <= 0:
		raise ValueError("Speech RMS is zero; cannot mix.")
	if noise_rms <= 0:
		raise ValueError("Noise RMS is zero; cannot mix.")

	if target_rms is None:
		target_rms = speech_rms

	normalized_speech = normalize_rms(speech, target_rms)
	normalized_noise = normalize_rms(noise, target_rms)

	# Step 2: Build multiplier vectors for each segment
	num_samples = speech.shape[-1]
	num_segments = int(np.ceil(num_samples / segment_samples))

	# Ensure snr_db_array matches number of segments
	if len(snr_db_array) != num_segments:
		if len(snr_db_array) == 1:
			snr_db_array = np.repeat(snr_db_array, num_segments)
		else:
			raise ValueError(
				f"Length of snr_db_array ({len(snr_db_array)}) must match "
				f"number of segments ({num_segments})"
			)

	# Compute multipliers for each segment
	mult_speech_seg = np.zeros(num_samples, dtype=np.float32)
	mult_noise_seg = np.zeros(num_samples, dtype=np.float32)

	for i in range(num_segments):
		start_idx = i * segment_samples
		end_idx = min((i + 1) * segment_samples, num_samples)
		half_snr_linear = 10 ** (snr_db_array[i] / 2 / 20)
		mult_speech_seg[start_idx:end_idx] = half_snr_linear
		mult_noise_seg[start_idx:end_idx] = 1 / half_snr_linear

	# Step 3: Smooth boundaries if requested
	if smooth_boundaries and num_segments > 1:
		smooth_window = min(segment_samples // 4, 512)  # Smooth over ~1/4 of segment
		for i in range(num_segments - 1):
			boundary_idx = (i + 1) * segment_samples
			if boundary_idx < num_samples:
				# Linear blend across boundary
				start_blend = max(0, boundary_idx - smooth_window)
				end_blend = min(num_samples, boundary_idx + smooth_window)
				blend_len = end_blend - start_blend
				if blend_len > 0:
					blend_factor = np.linspace(0, 1, blend_len)
					before_val_s = mult_speech_seg[start_blend]
					after_val_s = mult_speech_seg[min(boundary_idx + smooth_window - 1, num_samples - 1)]
					before_val_n = mult_noise_seg[start_blend]
					after_val_n = mult_noise_seg[min(boundary_idx + smooth_window - 1, num_samples - 1)]
					mult_speech_seg[start_blend:end_blend] = (
						before_val_s * (1 - blend_factor) + after_val_s * blend_factor
					)
					mult_noise_seg[start_blend:end_blend] = (
						before_val_n * (1 - blend_factor) + after_val_n * blend_factor
					)

	# Step 4: Mix as weighted sum
	mixed = normalized_speech * mult_speech_seg + normalized_noise * mult_noise_seg

	# Step 5: Re-normalize to target RMS
	mixed = normalize_rms(mixed, target_rms)

	return mixed


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