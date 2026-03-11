"""Audio I/O and file utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import librosa
import numpy as np
import soundfile as sf


SUPPORTED_EXTS = {".wav", ".flac", ".ogg", ".mp3", ".m4a"}


def is_audio_file(path: Path) -> bool:
	"""Check if a path is a supported audio file."""
	return path.is_file() and path.suffix.lower() in SUPPORTED_EXTS


def iter_audio_files(path: Path) -> Iterable[Path]:
	"""Iterate over validated audio files in a path (file or directory).

	Raises:
		ValueError: If path does not exist or no supported audio files are found.
	"""
	path = path.expanduser().resolve()
	if not path.exists():
		raise ValueError(f"Path does not exist: {path}")

	files: list[Path] = []
	if path.is_file():
		if is_audio_file(path):
			files = [path]
	else:
		for ext in SUPPORTED_EXTS:
			files.extend(path.rglob(f"*{ext}"))

	files = sorted(files)
	if not files:
		raise ValueError(f"No audio files found at: {path}")

	yield from files


def load_audio(path: Path, sr: int | None = None, mono: bool = True) -> tuple[np.ndarray, int]:
	"""Load an audio file.

	Args:
		path: Audio file path.
		sr: Target sample rate (None preserves original).
		mono: Convert to mono.

	Returns:
		(audio, sample_rate) tuple.
	"""
	audio, sample_rate = librosa.load(path, sr=sr, mono=mono)
	return audio, sample_rate


def save_audio(path: Path, audio: np.ndarray, sr: int) -> None:
	"""Save an audio file.

	Args:
		path: Output path.
		audio: Audio waveform.
		sr: Sample rate.
	"""
	path.parent.mkdir(parents=True, exist_ok=True)
	sf.write(path, audio, sr)


def normalize_peak(audio: np.ndarray, target: float = 0.99) -> np.ndarray:
	"""Normalize audio to a target peak amplitude.

	Args:
		audio: Input waveform.
		target: Target peak amplitude (0-1).

	Returns:
		Normalized waveform.
	"""
	audio = np.asarray(audio, dtype=np.float32)
	peak = np.max(np.abs(audio))
	if peak > 0:
		return target * audio / peak
	return audio


def rms(signal: np.ndarray, axis: int = -1) -> np.ndarray:
	"""Compute root mean square along an axis.

	Args:
		signal: Input signal.
		axis: Axis along which to compute RMS.

	Returns:
		RMS value(s).
	"""
	signal = np.asarray(signal, dtype=np.float32)
	return np.sqrt(np.mean(np.square(signal), axis=axis))
