"""Linguistic timing features: word onsets from Whisper word-level timestamps."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Word:
	text: str
	start: float
	end: float


def load_whisper(model_size: str = "small"):
	"""Load a Whisper model on CPU. Call once and reuse across files."""
	try:
		import whisper
	except ImportError as exc:
		raise ImportError("Word onsets need openai-whisper: pip install 'conch-triton[features]'") from exc

	return whisper.load_model(model_size, device="cpu")


def transcribe_words(path: str | Path, *, model, language: str | None = None) -> list[Word]:
	"""Transcribe a file and return every word with its start/end time in seconds."""
	result = model.transcribe(str(path), language=language, word_timestamps=True, fp16=False)
	return [
		Word(text=str(word["word"]).strip(), start=float(word["start"]), end=float(word["end"]))
		for segment in result.get("segments", [])
		for word in segment.get("words", [])
	]


def onset_train(onsets: np.ndarray, n_samples: int, sr: int, rate: int) -> tuple[np.ndarray, np.ndarray]:
	"""Build an impulse train with a 1 at each onset, sampled at ``rate`` Hz.

	This is the usual regressor for word onsets in encoding models. Its length is
	``ceil(n_samples * rate / sr)``, the same as ``envelope_feature`` for the same
	audio, so the two can be stacked directly.

	Args:
		onsets: Onset times in seconds.
		n_samples: Length of the source audio in samples.
		sr: Sample rate of the source audio (Hz).
		rate: Output rate (Hz).

	Returns:
		(times, train): sample times in seconds and the 0/1 impulse train, float32.
	"""
	n_out = -(-int(n_samples) * int(rate) // int(sr))  # integer ceil, matches resample_poly
	train = np.zeros(n_out, dtype=np.float32)
	indices = np.round(np.asarray(onsets, dtype=np.float64) * rate).astype(int)
	train[indices[(indices >= 0) & (indices < n_out)]] = 1.0
	times = (np.arange(n_out) / float(rate)).astype(np.float32)
	return times, train
