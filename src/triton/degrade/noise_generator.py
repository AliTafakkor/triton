"""Noise generation utilities for speech degradation experiments."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
from scipy import signal

from triton.core.io import iter_audio_files, iter_source_audio, load_audio, normalize_peak
from triton.core.signal import to_mono_float32
from triton.degrade.noise_mixer import mix_with_noise_at_snr


def compute_ltass(
	source: Path | np.ndarray | Iterable[Path | np.ndarray],
	*,
	sr: int,
	n_fft: int = 2048,
	hop_length: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
	"""Compute long-term average speech spectrum (LTASS).

	Args:
		source: Corpus path, single waveform, or iterable of paths/waveforms.
		sr: Sample rate used for loading/resampling source audio.
		n_fft: FFT size.
		hop_length: Hop size for STFT.

	Returns:
		(freqs_hz, mean_power_spectrum) arrays.
	"""
	if n_fft <= 0 or hop_length <= 0:
		raise ValueError("n_fft and hop_length must be positive.")

	accum_power = None
	n_used = 0

	for audio in iter_source_audio(source, sr=sr):
		if audio.size == 0:
			continue
		if audio.size < n_fft:
			audio = np.pad(audio, (0, n_fft - audio.size))

		_, _, stft = signal.stft(
			audio,
			fs=sr,
			nperseg=n_fft,
			noverlap=max(0, n_fft - hop_length),
			boundary=None,
			padded=False,
		)
		power = np.mean(np.abs(stft) ** 2, axis=1)

		if accum_power is None:
			accum_power = power
		else:
			accum_power += power
		n_used += 1

	if n_used == 0 or accum_power is None:
		raise ValueError("No valid audio found to compute LTASS.")

	mean_power = accum_power / n_used
	freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
	return freqs.astype(np.float32), mean_power.astype(np.float32)


def generate_ssn(
	*,
	shape_source: Path | np.ndarray | Iterable[Path | np.ndarray],
	length_samples: int,
	sr: int,
	n_fft: int = 2048,
	hop_length: int = 512,
	seed: int | None = None,
	normalize: bool = True,
) -> np.ndarray:
	"""Generate speech-shaped noise (SSN) from LTASS of source speech.

	Args:
		shape_source: Corpus path or exact speech file(s) to derive LTASS.
		length_samples: Desired output noise length.
		sr: Output sample rate.
		n_fft: FFT size for LTASS computation.
		hop_length: STFT hop size for LTASS computation.
		seed: Optional random seed.
		normalize: Normalize output to peak 0.99.

	Returns:
		Generated SSN waveform.
	"""
	if length_samples <= 0:
		raise ValueError("length_samples must be positive.")

	freqs, mean_power = compute_ltass(shape_source, sr=sr, n_fft=n_fft, hop_length=hop_length)

	rng = np.random.default_rng(seed)
	white = rng.standard_normal(length_samples).astype(np.float32)

	n_fft_shape = int(2 ** np.ceil(np.log2(max(length_samples, n_fft))))
	white_spec = np.fft.rfft(white, n=n_fft_shape)
	white_mag = np.abs(white_spec)

	shape_freqs = np.fft.rfftfreq(n_fft_shape, d=1.0 / sr)
	interp_power = np.interp(shape_freqs, freqs, mean_power, left=mean_power[0], right=mean_power[-1])
	target_mag = np.sqrt(np.maximum(interp_power, 1e-12))

	shaped_spec = (white_spec / (white_mag + 1e-12)) * target_mag
	shaped = np.fft.irfft(shaped_spec, n=n_fft_shape)[:length_samples].astype(np.float32)

	if normalize:
		shaped = normalize_peak(shaped)
	return shaped


def generate_babble(
	talker_root: Path,
	*,
	length_samples: int,
	sr: int,
	n_talkers: int = 8,
	seed: int | None = None,
	normalize: bool = True,
) -> np.ndarray:
	"""Generate babble noise from speech folders organized by talker.

	The expected layout is one subfolder per talker under `talker_root`, each
	containing one or more speech files.

	Args:
		talker_root: Folder containing talker subfolders.
		length_samples: Desired output length in samples.
		sr: Output sample rate.
		n_talkers: Number of talkers to combine.
		seed: Optional random seed.
		normalize: Normalize output to peak 0.99.

	Returns:
		Babble-noise waveform.
	"""
	if length_samples <= 0:
		raise ValueError("length_samples must be positive.")
	if n_talkers <= 0:
		raise ValueError("n_talkers must be positive.")

	talker_root = talker_root.expanduser().resolve()
	if not talker_root.exists() or not talker_root.is_dir():
		raise ValueError(f"Invalid talker_root: {talker_root}")

	talker_dirs = sorted([path for path in talker_root.iterdir() if path.is_dir()])
	if not talker_dirs:
		raise ValueError("No talker subdirectories found for babble generation.")

	rng = np.random.default_rng(seed)
	selected_dirs = list(rng.choice(talker_dirs, size=min(n_talkers, len(talker_dirs)), replace=False))

	babble = np.zeros(length_samples, dtype=np.float32)
	n_used = 0

	for talker_dir in selected_dirs:
		files = list(iter_audio_files(talker_dir))
		if not files:
			continue
		audio_path = files[int(rng.integers(0, len(files)))]
		audio, _ = load_audio(audio_path, sr=sr, mono=True)
		audio = to_mono_float32(audio)
		if audio.size == 0:
			continue

		if audio.size < length_samples:
			repeats = int(np.ceil(length_samples / audio.size))
			audio = np.tile(audio, repeats)

		start = int(rng.integers(0, max(1, audio.size - length_samples + 1)))
		segment = audio[start : start + length_samples]
		babble += segment
		n_used += 1

	if n_used == 0:
		raise ValueError("No usable speech files found in selected talker folders.")

	babble /= float(n_used)
	babble -= np.mean(babble)

	if normalize:
		babble = normalize_peak(babble)
	return babble.astype(np.float32)
