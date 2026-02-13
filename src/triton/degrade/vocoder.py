"""Channel vocoder degradation following Shannon et al. (1995)."""

from __future__ import annotations

import numpy as np

from triton.core.signal import extract_envelope, bandpass_filter
from triton.core.io import normalize_peak


def noise_vocode(
	audio: np.ndarray,
	sr: int,
	*,
	n_bands: int = 8,
	freq_range: tuple[float, float] = (200, 8000),
	envelope_cutoff: float = 160.0,
	vocoder_type: str = "noise",
	filter_order: int = 3,
) -> np.ndarray:
	"""Apply channel vocoding following Shannon et al. (1995).

	Implements logarithmic band spacing, Butterworth filters, half-wave
	rectified Hilbert envelopes, low-pass smoothing, and carrier modulation.

	Args:
		audio: Input waveform (mono).
		sr: Sample rate.
		n_bands: Number of spectral channels (degradation level).
		freq_range: (low_hz, high_hz) frequency range.
		envelope_cutoff: Low-pass cutoff for envelope extraction (Hz).
		vocoder_type: Either "noise" or "sine".
		filter_order: Butterworth filter order.

	Returns:
		Vocoded waveform.
	"""
	if n_bands <= 0:
		raise ValueError("Number of bands must be positive.")
	if vocoder_type not in {"noise", "sine"}:
		raise ValueError(f"Unsupported vocoder type: {vocoder_type}")

	audio = np.asarray(audio, dtype=np.float32)
	low_freq, high_freq = freq_range

	nyquist = sr / 2.0
	high_freq = min(high_freq, nyquist * 0.999)
	if low_freq >= high_freq:
		raise ValueError("Invalid frequency range for vocoding.")

	# Logarithmic band spacing (Shannon et al., 1995)
	band_edges = np.logspace(np.log10(low_freq), np.log10(high_freq), n_bands + 1)

	vocoded = np.zeros_like(audio)
	time = np.arange(len(audio)) / sr

	# Generate white noise carrier once for noise vocoding
	if vocoder_type == "noise":
		noise = np.random.randn(len(audio)).astype(np.float32)

	for low, high in zip(band_edges[:-1], band_edges[1:]):
		if high <= low:
			continue

		# Band-pass filter the signal
		band_signal = bandpass_filter(audio, low, high, sr, order=filter_order)

		# Extract envelope (half-wave rectified Hilbert)
		envelope = extract_envelope(
			band_signal, sr, method="hilbert", cutoff=envelope_cutoff, filter_order=filter_order
		)

		# Generate carrier
		if vocoder_type == "noise":
			carrier = bandpass_filter(noise, low, high, sr, order=filter_order)
		elif vocoder_type == "sine":
			center_freq = np.sqrt(low * high)
			carrier = np.sin(2 * np.pi * center_freq * time)

		# Modulate carrier with envelope
		vocoded += carrier * envelope

	return normalize_peak(vocoded)
