"""Audio I/O and file utilities."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timezone
import json
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from triton.core.signal import to_mono_float32


SUPPORTED_EXTS = {".wav", ".flac", ".ogg", ".mp3", ".m4a"}


def sidecar_path(path: Path) -> Path:
	"""Return sidecar JSON path for an output file."""
	return path.with_suffix(path.suffix + ".json")


def write_sidecar(
	path: Path,
	*,
	source: dict[str, object] | None = None,
	actions: list[dict[str, object]] | None = None,
	extra: dict[str, object] | None = None,
) -> Path:
	"""Write provenance sidecar JSON next to a generated file.

	Args:
		path: Generated file path.
		source: Source metadata dictionary.
		actions: Ordered list of applied actions with details.
		extra: Optional extra metadata.

	Returns:
		Path to written sidecar JSON.
	"""
	meta_path = sidecar_path(path)
	payload: dict[str, object] = {
		"triton": {
			"schema_version": 1,
			"generated_at": datetime.now(timezone.utc).isoformat(),
		},
		"artifact": {
			"path": str(path.resolve()),
			"name": path.name,
			"suffix": path.suffix.lower(),
		},
		"source": source or {},
		"actions": actions or [],
	}
	if extra:
		payload["extra"] = extra

	meta_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
	return meta_path


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


def iter_source_audio(
	source: Path | np.ndarray | Iterable[Path | np.ndarray],
	*,
	sr: int,
) -> Iterable[np.ndarray]:
	"""Iterate mono float32 audio from path/array source inputs.

	Args:
		source: Audio source as a path, waveform, or iterable of paths/waveforms.
		sr: Sample rate used for loading/resampling path-based audio.

	Yields:
		Mono float32 audio waveforms.
	"""
	if isinstance(source, np.ndarray):
		yield to_mono_float32(source)
		return

	if isinstance(source, Path):
		for audio_path in iter_audio_files(source):
			audio, _ = load_audio(audio_path, sr=sr, mono=True)
			yield to_mono_float32(audio)
		return

	for item in source:
		if isinstance(item, Path):
			audio, _ = load_audio(item, sr=sr, mono=True)
			yield to_mono_float32(audio)
		elif isinstance(item, np.ndarray):
			yield to_mono_float32(item)
		else:
			raise TypeError("Source items must be Path or numpy.ndarray.")


def save_audio(
	path: Path,
	audio: np.ndarray,
	sr: int,
	*,
	source: dict[str, object] | None = None,
	actions: list[dict[str, object]] | None = None,
	extra: dict[str, object] | None = None,
) -> None:
	"""Save an audio file.

	Args:
		path: Output path.
		audio: Audio waveform.
		sr: Sample rate.
		source: Optional source metadata for sidecar provenance.
		actions: Optional ordered action history for sidecar provenance.
		extra: Optional extra metadata for sidecar provenance.
	"""
	path.parent.mkdir(parents=True, exist_ok=True)
	audio_arr = np.asarray(audio)
	sf.write(path, audio_arr, sr)

	channels = 1 if audio_arr.ndim == 1 else int(audio_arr.shape[1])
	artifact_extra = {
		"audio": {
			"sample_rate": int(sr),
			"samples": int(audio_arr.shape[0]),
			"channels": channels,
		}
	}
	if extra:
		artifact_extra.update(extra)

	write_sidecar(path, source=source, actions=actions, extra=artifact_extra)


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


def normalize_rms(audio: np.ndarray, target: float = 0.1) -> np.ndarray:
	"""Normalize audio to a target RMS (root mean square) amplitude.

	Args:
		audio: Input waveform.
		target: Target RMS amplitude (0-1). Default 0.1 (-20dB) is a safe loudness level.

	Returns:
		Normalized waveform.
	"""
	audio = np.asarray(audio, dtype=np.float32)
	if audio.ndim > 1:
		# For multi-channel, compute RMS across all channels and samples
		audio_rms = np.sqrt(np.mean(np.square(audio)))
	else:
		audio_rms = np.sqrt(np.mean(np.square(audio)))

	if audio_rms > 0:
		return target * audio / audio_rms
	return audio
