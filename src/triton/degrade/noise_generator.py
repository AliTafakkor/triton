"""Noise generation utilities for speech degradation experiments."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from collections.abc import Iterable
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import soundfile as sf
from scipy import signal

from triton.core.io import iter_audio_files, iter_source_audio, load_audio, normalize_peak
from triton.core.mixer import mix_babble_from_segments
from triton.core.project import BabbleTalkerGroup, select_babble_talker_groups
from triton.core.signal import to_mono_float32


@dataclass(slots=True)
class ProjectBabbleResult:
	"""Result metadata for babble generated from project-labeled talkers."""

	audio: np.ndarray
	sample_rate: int
	selected_groups: list[BabbleTalkerGroup]
	planned_group_files: list[list[Path]]
	short_source_labels: list[str]
	unknown_duration_labels: list[str]
	repeat_counts_by_label: dict[str, int]


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


def _estimate_audio_duration_seconds(file_path: Path) -> float | None:
	"""Estimate duration from file metadata without decoding the full file."""
	try:
		info = sf.info(str(file_path))
		if info.samplerate > 0 and info.frames > 0:
			return float(info.frames) / float(info.samplerate)
	except Exception:
		return None
	return None


def _select_files_for_intended_length(
	group_files: list[Path],
	intended_length_seconds: float,
) -> tuple[list[Path], float, bool]:
	"""Select only as many files as needed to reach intended duration."""
	if intended_length_seconds <= 0:
		return list(group_files), 0.0, True

	target_seconds = float(intended_length_seconds)
	cumulative = 0.0
	selected: list[Path] = []
	has_unknown = False

	for file_path in group_files:
		selected.append(file_path)
		estimated = _estimate_audio_duration_seconds(file_path)
		if estimated is None:
			has_unknown = True
		else:
			cumulative += estimated

		if not has_unknown and cumulative >= target_seconds:
			break

	return selected, cumulative, has_unknown


def _fit_segments_to_target_length(
	segments: list[np.ndarray],
	target_samples: int,
	rng: np.random.Generator,
) -> tuple[list[np.ndarray], int, bool]:
	"""Trim or randomly repeat segments to match target length."""
	if target_samples <= 0:
		return segments, 0, False
	if not segments:
		raise ValueError("Each talker must contain at least one loaded segment.")

	base_segments = [np.asarray(segment, dtype=np.float32) for segment in segments]
	base_total = int(sum(segment.shape[-1] for segment in base_segments))

	if base_total >= target_samples:
		trimmed: list[np.ndarray] = []
		acc = 0
		for segment in base_segments:
			remaining = target_samples - acc
			if remaining <= 0:
				break
			if segment.shape[-1] <= remaining:
				trimmed.append(segment)
				acc += int(segment.shape[-1])
			else:
				trimmed.append(segment[..., :remaining])
				acc += remaining
		return trimmed, 0, False

	fitted = list(base_segments)
	acc = base_total
	repeats_added = 0
	while acc < target_samples:
		segment = base_segments[int(rng.integers(0, len(base_segments)))]
		remaining = target_samples - acc
		if segment.shape[-1] > remaining:
			fitted.append(segment[..., :remaining])
			acc += remaining
		else:
			fitted.append(segment)
			acc += int(segment.shape[-1])
		repeats_added += 1

	return fitted, repeats_added, True


def _load_project_audio(file_path: Path, sr: int, channel_mode: Literal["mono", "stereo"]) -> np.ndarray:
	"""Load project audio as mono or stereo float32 with sample axis last."""
	audio, _ = load_audio(file_path, sr=sr, mono=(channel_mode == "mono"))
	audio = np.asarray(audio, dtype=np.float32)

	if channel_mode == "mono":
		if audio.ndim > 1:
			return np.mean(audio, axis=0, dtype=np.float32)
		return audio

	if audio.ndim == 1:
		return np.stack([audio, audio], axis=0).astype(np.float32)
	if audio.shape[0] == 1:
		return np.repeat(audio, 2, axis=0).astype(np.float32)
	if audio.shape[0] >= 2:
		return audio[:2, :].astype(np.float32)

	raise ValueError(f"Unsupported audio shape for {file_path}: {audio.shape}")


def generate_project_babble(
	project_dir: Path,
	*,
	sr: int,
	channel_mode: Literal["mono", "stereo"],
	num_talkers: int,
	num_female_talkers: int | None = None,
	num_male_talkers: int | None = None,
	intended_length_seconds: float = 30.0,
	target_rms: float = 0.1,
	peak_normalize: bool = True,
	seed: int | None = None,
	max_workers: int | None = None,
	progress_callback: Callable[[str, int | None], None] | None = None,
) -> ProjectBabbleResult:
	"""Generate babble from project files grouped by ``bab-fN`` / ``bab-mN`` labels."""
	project_dir = project_dir.expanduser().resolve()

	def _progress(message: str, pct: int | None = None) -> None:
		if progress_callback is not None:
			progress_callback(message, pct)

	_progress("Selecting talker groups...", 5)
	selected_groups = select_babble_talker_groups(
		project_dir,
		num_talkers=int(num_talkers),
		num_female_talkers=num_female_talkers,
		num_male_talkers=num_male_talkers,
	)

	total_files = sum(len(group.files) for group in selected_groups)
	_progress(f"Selected {len(selected_groups)} groups with {total_files} files.", 12)

	target_samples = int(round(float(intended_length_seconds) * float(sr)))
	_progress(f"Planning files for intended length: {float(intended_length_seconds):.1f}s per talker...", 18)

	planned_group_files: list[list[Path]] = []
	short_source_labels: list[str] = []
	unknown_duration_labels: list[str] = []
	for group in selected_groups:
		selected_files, estimated_seconds, has_unknown = _select_files_for_intended_length(
			group.files,
			float(intended_length_seconds),
		)
		planned_group_files.append(selected_files)
		if has_unknown:
			unknown_duration_labels.append(group.label)
		elif estimated_seconds < float(intended_length_seconds):
			short_source_labels.append(group.label)

	total_planned_files = sum(len(files) for files in planned_group_files)
	_progress(f"Loading {total_planned_files}/{total_files} selected files...", 25)

	talker_segments_slots: list[list[np.ndarray | None]] = [
		[None] * len(files) for files in planned_group_files
	]

	if total_planned_files == 0:
		raise ValueError("No files available in selected talker groups.")

	if max_workers is None:
		max_workers = min(max(2, (os.cpu_count() or 4) - 1), max(1, total_planned_files), 8)

	loaded_files = 0
	log_interval = max(1, total_planned_files // 20)
	with ThreadPoolExecutor(max_workers=max_workers) as executor:
		future_map = {}
		for group_idx, files in enumerate(planned_group_files):
			for file_idx, file_path in enumerate(files):
				future = executor.submit(_load_project_audio, file_path, sr, channel_mode)
				future_map[future] = (group_idx, file_idx, file_path)

		for future in as_completed(future_map):
			group_idx, file_idx, file_path = future_map[future]
			talker_segments_slots[group_idx][file_idx] = future.result()
			loaded_files += 1
			if loaded_files % log_interval == 0 or loaded_files == total_planned_files:
				load_progress = 25 + int(40 * (loaded_files / max(1, total_planned_files)))
				_progress(f"Loaded {loaded_files}/{total_planned_files} files (latest: {file_path.name})", load_progress)

	talker_segments: list[list[np.ndarray]] = []
	for slots in talker_segments_slots:
		segments = [segment for segment in slots if segment is not None]
		if len(segments) != len(slots):
			raise RuntimeError("Failed to load one or more selected files.")
		talker_segments.append(segments)

	rng = np.random.default_rng(seed)
	repeat_counts_by_label: dict[str, int] = {}
	fitted_talker_segments: list[list[np.ndarray]] = []
	for group, segments in zip(selected_groups, talker_segments, strict=False):
		fitted_segments, repeats_added, repeated = _fit_segments_to_target_length(
			segments,
			target_samples=target_samples,
			rng=rng,
		)
		fitted_talker_segments.append(fitted_segments)
		if repeated:
			repeat_counts_by_label[group.label] = repeats_added

	_progress("Normalizing segments, concatenating talkers, and mixing...", 75)
	babble = mix_babble_from_segments(
		fitted_talker_segments,
		target_rms=float(target_rms),
		peak_normalize=bool(peak_normalize),
	)
	_progress("Babble generation complete.", 100)

	return ProjectBabbleResult(
		audio=babble,
		sample_rate=int(sr),
		selected_groups=selected_groups,
		planned_group_files=planned_group_files,
		short_source_labels=short_source_labels,
		unknown_duration_labels=unknown_duration_labels,
		repeat_counts_by_label=repeat_counts_by_label,
	)
