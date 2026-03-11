from __future__ import annotations

import numpy as np

from triton.core.io import save_audio
from triton.degrade.noise_generator import (
	compute_ltass,
	generate_babble,
	generate_ssn,
)
from triton.degrade.noise_mixer import add_noise


def test_compute_ltass_from_arrays() -> None:
	sr = 16000
	t = np.arange(sr, dtype=np.float32) / sr
	speech_a = np.sin(2 * np.pi * 220 * t).astype(np.float32)
	speech_b = np.sin(2 * np.pi * 440 * t).astype(np.float32)

	freqs, power = compute_ltass([speech_a, speech_b], sr=sr)

	assert freqs.ndim == 1
	assert power.ndim == 1
	assert freqs.shape == power.shape
	assert np.all(power >= 0)


def test_generate_ssn_and_mix() -> None:
	sr = 16000
	target = np.random.default_rng(0).standard_normal(sr).astype(np.float32) * 0.1
	shape_source = np.random.default_rng(1).standard_normal(sr).astype(np.float32) * 0.1

	noise = generate_ssn(shape_source=shape_source, length_samples=sr, sr=sr, seed=123)
	assert noise.shape == target.shape
	assert np.isfinite(noise).all()

	mixed = add_noise(target, noise, snr_db=0.0)
	assert mixed.shape == target.shape
	assert np.max(np.abs(mixed)) <= 1.0


def test_generate_babble_from_talker_dirs(tmp_path) -> None:
	sr = 16000
	duration = sr // 2
	t = np.arange(duration, dtype=np.float32) / sr

	talker_a = tmp_path / "talker_a"
	talker_b = tmp_path / "talker_b"
	talker_a.mkdir()
	talker_b.mkdir()

	save_audio(talker_a / "utt1.wav", np.sin(2 * np.pi * 180 * t).astype(np.float32), sr)
	save_audio(talker_b / "utt1.wav", np.sin(2 * np.pi * 260 * t).astype(np.float32), sr)

	babble = generate_babble(tmp_path, length_samples=sr, sr=sr, n_talkers=2, seed=0)

	assert babble.shape == (sr,)
	assert np.isfinite(babble).all()
	assert np.max(np.abs(babble)) <= 1.0
