"""Backward-compatible noise mixing helpers.

This module preserves the historical `triton.degrade.noise_mixer` import path
while delegating the implementation to the canonical core mixer.
"""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np

from triton.core.io import load_audio
from triton.core.mixer import mix_at_snr
from triton.degrade.speech_noise import speech_shaped_noise


BABBLE_MULTITALKER_RE = re.compile(r"^bab-t\d+$", re.IGNORECASE)


def _slice_noise_to_target(
	noise: np.ndarray,
	target_length: int,
	rng: np.random.Generator,
) -> np.ndarray:
	"""Match noise length to target length using a random crop when longer."""
	if noise.shape[-1] <= target_length:
		return noise

	start = int(rng.integers(0, noise.shape[-1] - target_length + 1))
	return noise[..., start : start + target_length]


def _resolve_noise(
	*,
	target: np.ndarray,
	noise: np.ndarray | None,
	noise_type: str,
	noise_file: Path | None,
	sample_rate: int | None,
	seed: int | None,
) -> np.ndarray:
	"""Resolve noise from an array, a file, or generated noise type."""
	target_length = int(target.shape[-1])
	if target_length <= 0:
		raise ValueError("Target must have non-zero length.")

	rng = np.random.default_rng(seed)
	selected_type = noise_type.strip().lower()
	if selected_type in {"", "auto"} and noise_file is not None:
		selected_type = "babble" if BABBLE_MULTITALKER_RE.match(noise_file.stem) else "white"
	elif selected_type in {"", "auto"}:
		selected_type = "white"

	if noise is not None:
		resolved = np.asarray(noise, dtype=np.float32)
	elif noise_file is not None:
		resolved, _ = load_audio(noise_file, sr=sample_rate, mono=True)
		resolved = np.asarray(resolved, dtype=np.float32)
	elif selected_type == "white":
		if sample_rate is None or sample_rate <= 0:
			raise ValueError("sample_rate must be provided and positive for generated noise.")
		resolved = speech_shaped_noise(target_length / float(sample_rate), sample_rate, spectrum="flat", normalize=True)
	elif selected_type in {"colored", "ssn"}:
		if sample_rate is None or sample_rate <= 0:
			raise ValueError("sample_rate must be provided and positive for generated noise.")
		resolved = speech_shaped_noise(target_length / float(sample_rate), sample_rate, spectrum="ltass", normalize=True)
	elif selected_type == "babble":
		raise ValueError("Babble noise must be provided via `noise` or `noise_file`; generation is not supported in add_noise.")
	else:
		raise ValueError(f"Unsupported noise_type: {noise_type}")

	if resolved.shape[-1] == 0:
		raise ValueError("Resolved noise has zero length.")

	return _slice_noise_to_target(np.asarray(resolved, dtype=np.float32), target_length, rng)


def add_noise(
	target: np.ndarray,
	noise: np.ndarray | None = None,
	*,
	snr_db: float,
	noise_type: str = "auto",
	noise_file: Path | None = None,
	sample_rate: int | None = None,
	seed: int | None = None,
) -> np.ndarray:
	"""Mix target audio with noise at the requested SNR in dB.

	Args:
		target: Target waveform array.
		noise: Optional noise waveform array.
		snr_db: Desired signal-to-noise ratio in dB.
		noise_type: Noise mode ("auto", "white", "colored", "ssn", "babble").
		noise_file: Optional path to an existing noise file.
		sample_rate: Sample rate used for loading/generating noise when needed.
		seed: Optional RNG seed used for random noise cropping.

	Returns:
		Normalized mixed waveform.
	"""
	target_audio = np.asarray(target, dtype=np.float32)
	resolved_noise = _resolve_noise(
		target=target_audio,
		noise=noise,
		noise_type=noise_type,
		noise_file=noise_file,
		sample_rate=sample_rate,
		seed=seed,
	)
	return mix_at_snr(target_audio, resolved_noise, snr_db)
