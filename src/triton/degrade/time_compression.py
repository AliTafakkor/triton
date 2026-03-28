"""Time-scale modification using Parselmouth/Praat and librosa."""

from __future__ import annotations

import numpy as np

try:
	import parselmouth
except ImportError:
	parselmouth = None

try:
	import librosa
except ImportError:
	librosa = None


def compress_time(
	audio: np.ndarray,
	sr: int,
	*,
	factor: float = 1.0,
) -> np.ndarray:
	"""Apply time compression to audio without changing pitch.

	Uses Praat/Parselmouth when available, otherwise falls back to librosa.

	Args:
		audio: Input waveform (mono); will be converted to mono if stereo.
		sr: Sample rate.
		factor: Compression factor (0.5 = half duration, 1.0 = no change, 2.0 = double duration).

	Returns:
		Time-compressed waveform at the same sample rate.

	Raises:
		ValueError: If factor is non-positive.
		ImportError: If neither parselmouth nor librosa are installed.
	"""
	if factor <= 0:
		raise ValueError("Compression factor must be positive.")

	# Convert to mono if needed
	if audio.ndim > 1:
		audio = np.mean(audio, axis=1, dtype=np.float32)

	audio = np.asarray(audio, dtype=np.float32)

	# Try Praat/Parselmouth first.
	if parselmouth is not None:
		try:
			sound = parselmouth.Sound(audio, sampling_frequency=sr)
			manipulation = parselmouth.praat.call(sound, "To Manipulation", 0.01, 75, 600)
			scaled = parselmouth.praat.call(manipulation, "Scale times", factor)
			resynthesized = parselmouth.praat.call(scaled, "Get resynthesis (overlap-add)")
			return np.asarray(resynthesized.values[0], dtype=np.float32)
		except Exception:
			pass

	# Fall back to librosa if Praat is unavailable or fails.
	if librosa is not None:
		rate = 1.0 / factor
		try:
			stretched = librosa.effects.time_stretch(audio, rate)
		except TypeError:
			stretched = librosa.effects.time_stretch(audio, rate=rate)
		return np.asarray(stretched, dtype=np.float32)

	raise ImportError(
		"No suitable time compression backend available. Install praat-parselmouth or librosa."
	)

