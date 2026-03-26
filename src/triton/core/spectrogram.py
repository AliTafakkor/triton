"""Spectrogram computation and persistence helpers."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import librosa
import numpy as np


SPECTROGRAM_TYPES = {"stft", "mel", "cqt"}


@dataclass(slots=True)
class SpectrogramResult:
	kind: str
	values: np.ndarray
	freqs: np.ndarray
	times: np.ndarray


DEFAULT_SPECTROGRAM_SETTINGS: dict[str, object] = {
	"type": "stft",
	"n_fft": 1024,
	"hop_length": 256,
	"win_length": 1024,
	"window": "hann",
	"n_mels": 128,
	"fmin": 32.7,
	"fmax": 8000.0,
	"power": 2.0,
}


def normalize_spectrogram_settings(settings: dict[str, object]) -> dict[str, object]:
	merged = dict(DEFAULT_SPECTROGRAM_SETTINGS)
	for key in DEFAULT_SPECTROGRAM_SETTINGS:
		if key in settings:
			merged[key] = settings[key]

	kind = str(merged["type"]).lower().strip()
	if kind not in SPECTROGRAM_TYPES:
		raise ValueError(f"Unsupported spectrogram type: {kind}")

	merged["type"] = kind
	merged["n_fft"] = int(merged["n_fft"])
	merged["hop_length"] = int(merged["hop_length"])
	merged["win_length"] = int(merged["win_length"])
	merged["window"] = str(merged["window"])
	merged["n_mels"] = int(merged["n_mels"])
	merged["fmin"] = float(merged["fmin"])
	merged["fmax"] = float(merged["fmax"])
	merged["power"] = float(merged["power"])
	if kind == "cqt" and float(merged["fmin"]) <= 0:
		merged["fmin"] = 32.7
	return merged


def compute_spectrogram(audio: np.ndarray, sr: int, settings: dict[str, object]) -> SpectrogramResult:
	cfg = normalize_spectrogram_settings(settings)
	mono = np.asarray(audio, dtype=np.float32)
	if mono.ndim > 1:
		mono = np.mean(mono, axis=1, dtype=np.float32)

	kind = str(cfg["type"])
	if kind == "stft":
		stft = librosa.stft(
			mono,
			n_fft=int(cfg["n_fft"]),
			hop_length=int(cfg["hop_length"]),
			win_length=int(cfg["win_length"]),
			window=str(cfg["window"]),
		)
		power = np.abs(stft) ** float(cfg["power"])
		values = librosa.power_to_db(np.maximum(power, 1e-12), ref=np.max)
		freqs = librosa.fft_frequencies(sr=sr, n_fft=int(cfg["n_fft"]))
		times = librosa.frames_to_time(np.arange(values.shape[1]), sr=sr, hop_length=int(cfg["hop_length"]))
		return SpectrogramResult(kind=kind, values=values.astype(np.float32), freqs=freqs.astype(np.float32), times=times.astype(np.float32))

	if kind == "mel":
		mel = librosa.feature.melspectrogram(
			y=mono,
			sr=sr,
			n_fft=int(cfg["n_fft"]),
			hop_length=int(cfg["hop_length"]),
			win_length=int(cfg["win_length"]),
			window=str(cfg["window"]),
			n_mels=int(cfg["n_mels"]),
			fmin=float(cfg["fmin"]),
			fmax=float(cfg["fmax"]),
			power=float(cfg["power"]),
		)
		values = librosa.power_to_db(np.maximum(mel, 1e-12), ref=np.max)
		freqs = librosa.mel_frequencies(
			n_mels=int(cfg["n_mels"]),
			fmin=float(cfg["fmin"]),
			fmax=float(cfg["fmax"]),
		)
		times = librosa.frames_to_time(np.arange(values.shape[1]), sr=sr, hop_length=int(cfg["hop_length"]))
		return SpectrogramResult(kind=kind, values=values.astype(np.float32), freqs=freqs.astype(np.float32), times=times.astype(np.float32))

	cqt = librosa.cqt(
		mono,
		sr=sr,
		hop_length=int(cfg["hop_length"]),
		fmin=float(cfg["fmin"]),
	)
	amp = np.abs(cqt)
	values = librosa.amplitude_to_db(np.maximum(amp, 1e-12), ref=np.max)
	freqs = librosa.cqt_frequencies(values.shape[0], fmin=float(cfg["fmin"]))
	times = librosa.frames_to_time(np.arange(values.shape[1]), sr=sr, hop_length=int(cfg["hop_length"]))
	return SpectrogramResult(kind=kind, values=values.astype(np.float32), freqs=freqs.astype(np.float32), times=times.astype(np.float32))


def save_spectrogram(path: Path, result: SpectrogramResult, settings: dict[str, object]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	np.savez_compressed(
		path,
		kind=np.array(result.kind),
		values=result.values.astype(np.float32),
		freqs=result.freqs.astype(np.float32),
		times=result.times.astype(np.float32),
		settings_json=np.array(json.dumps(normalize_spectrogram_settings(settings), sort_keys=True)),
	)


def load_spectrogram(path: Path) -> tuple[SpectrogramResult, dict[str, object]]:
	with np.load(path, allow_pickle=False) as data:
		kind = str(data["kind"].item())
		values = np.asarray(data["values"], dtype=np.float32)
		freqs = np.asarray(data["freqs"], dtype=np.float32)
		times = np.asarray(data["times"], dtype=np.float32)
		settings = json.loads(str(data["settings_json"].item()))
	return SpectrogramResult(kind=kind, values=values, freqs=freqs, times=times), settings
