"""Audio format conversion utilities."""

from __future__ import annotations

import numpy as np
import librosa


def to_mono(audio: np.ndarray, method: str = "mean") -> np.ndarray:
	"""Convert stereo to mono.

	Args:
		audio: Input waveform (mono or stereo).
		method: Conversion method - "mean", "left", or "right".

	Returns:
		Mono waveform.
	"""
	audio = np.asarray(audio, dtype=np.float32)
	
	if audio.ndim == 1:
		return audio
	
	if method == "mean":
		return np.mean(audio, axis=0)
	elif method == "left":
		return audio[0]
	elif method == "right":
		return audio[1] if audio.shape[0] > 1 else audio[0]
	else:
		raise ValueError(f"Unsupported mono conversion method: {method}")


def to_stereo(audio: np.ndarray, method: str = "duplicate") -> np.ndarray:
	"""Convert mono to stereo.

	Args:
		audio: Input waveform (mono or stereo).
		method: Conversion method - "duplicate" or "silence".

	Returns:
		Stereo waveform (2, N).
	"""
	audio = np.asarray(audio, dtype=np.float32)
	
	if audio.ndim == 2 and audio.shape[0] == 2:
		return audio
	
	if audio.ndim == 2:
		audio = to_mono(audio)
	
	if method == "duplicate":
		return np.stack([audio, audio], axis=0)
	elif method == "silence":
		return np.stack([audio, np.zeros_like(audio)], axis=0)
	else:
		raise ValueError(f"Unsupported stereo conversion method: {method}")


def resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
	"""Resample audio to a target sample rate.

	Args:
		audio: Input waveform.
		orig_sr: Original sample rate.
		target_sr: Target sample rate.

	Returns:
		Resampled waveform.
	"""
	if orig_sr == target_sr:
		return audio
	
	audio = np.asarray(audio, dtype=np.float32)
	return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)


def requantize(audio: np.ndarray, bit_depth: int) -> np.ndarray:
	"""Change bit depth (quantization) of audio.

	Args:
		audio: Input waveform (float32, range -1 to 1).
		bit_depth: Target bit depth (8, 16, 24, 32).

	Returns:
		Quantized waveform.
	"""
	audio = np.asarray(audio, dtype=np.float32)
	
	if bit_depth not in {8, 16, 24, 32}:
		raise ValueError(f"Unsupported bit depth: {bit_depth}")
	
	# Clip to valid range
	audio = np.clip(audio, -1.0, 1.0)
	
	if bit_depth == 32:
		return audio
	
	# Quantize
	max_val = 2 ** (bit_depth - 1) - 1
	quantized = np.round(audio * max_val)
	
	# Convert back to float
	return (quantized / max_val).astype(np.float32)
