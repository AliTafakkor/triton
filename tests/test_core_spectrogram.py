from __future__ import annotations

import numpy as np

from triton.core.spectrogram import compute_spectrogram, load_spectrogram, save_spectrogram


def _tone(sr: int = 16000, duration: float = 0.25) -> np.ndarray:
	t = np.arange(int(sr * duration), dtype=np.float32) / sr
	return np.sin(2 * np.pi * 440.0 * t).astype(np.float32)


def test_compute_spectrogram_types() -> None:
	audio = _tone()
	sr = 16000

	for kind in ["stft", "mel", "cqt"]:
		result = compute_spectrogram(audio, sr, {"type": kind})
		assert result.kind == kind
		assert result.values.ndim == 2
		assert result.freqs.ndim == 1
		assert result.times.ndim == 1
		assert result.values.shape[1] == result.times.shape[0]


def test_save_and_load_spectrogram(tmp_path) -> None:
	audio = _tone()
	result = compute_spectrogram(audio, 16000, {"type": "mel", "n_mels": 64})
	path = tmp_path / "test.spectrogram.npz"
	save_spectrogram(path, result, {"type": "mel", "n_mels": 64})

	loaded_result, settings = load_spectrogram(path)
	assert loaded_result.kind == "mel"
	assert loaded_result.values.shape == result.values.shape
	assert int(settings["n_mels"]) == 64
