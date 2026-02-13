"""Core signal processing utilities."""

from __future__ import annotations

import numpy as np
from scipy import signal
from scipy.signal import hilbert


def extract_envelope(
	audio: np.ndarray,
	sr: int,
	*,
	method: str = "hilbert",
	cutoff: float = 160.0,
	filter_order: int = 4,
) -> np.ndarray:
	"""Extract the amplitude envelope of an audio signal.

	Args:
		audio: Input waveform.
		sr: Sample rate.
		method: Envelope method - "hilbert" (half-wave rectified) or "rms".
		cutoff: Low-pass cutoff frequency for envelope smoothing (Hz).
		filter_order: Butterworth filter order for smoothing.

	Returns:
		Envelope signal.
	"""
	audio = np.asarray(audio, dtype=np.float32)

	if method == "hilbert":
		# Half-wave rectification
		rectified = np.maximum(0.0, audio)
		# Hilbert transform
		analytic = hilbert(rectified)
		envelope = np.abs(analytic)
	elif method == "rms":
		# Simple rectification
		envelope = np.abs(audio)
	else:
		raise ValueError(f"Unsupported envelope method: {method}")

	# Low-pass smooth the envelope
	nyquist = sr / 2.0
	cutoff = min(cutoff, nyquist * 0.95)
	sos = signal.butter(filter_order, cutoff, btype="lowpass", fs=sr, output="sos")
	envelope = signal.sosfiltfilt(sos, envelope)

	return envelope


def bandpass_filter(
	audio: np.ndarray,
	low: float,
	high: float,
	sr: int,
	order: int = 3,
) -> np.ndarray:
	"""Apply a band-pass filter.

	Args:
		audio: Input waveform.
		low: Low frequency cutoff (Hz).
		high: High frequency cutoff (Hz).
		sr: Sample rate.
		order: Butterworth filter order.

	Returns:
		Filtered signal.
	"""
	audio = np.asarray(audio, dtype=np.float32)
	sos = signal.butter(order, [low, high], btype="bandpass", fs=sr, output="sos")
	return signal.sosfiltfilt(sos, audio)
