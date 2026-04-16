from __future__ import annotations

import numpy as np

from triton.core.io import save_audio
from triton.core.mixer import mix_at_snr
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


def test_add_noise_generated_types() -> None:
	sr = 16000
	target = np.random.default_rng(3).standard_normal(sr).astype(np.float32) * 0.05

	white_mixed = add_noise(target, snr_db=0.0, noise_type="white", sample_rate=sr, seed=7)
	colored_mixed = add_noise(target, snr_db=0.0, noise_type="colored", sample_rate=sr, seed=7)

	assert white_mixed.shape == target.shape
	assert colored_mixed.shape == target.shape
	assert np.isfinite(white_mixed).all()
	assert np.isfinite(colored_mixed).all()


def test_add_noise_babble_requires_external_noise() -> None:
	target = np.ones(1000, dtype=np.float32) * 0.1

	try:
		add_noise(target, snr_db=0.0, noise_type="babble", sample_rate=16000)
		assert False, "Expected ValueError for babble without noise source"
	except ValueError as exc:
		assert "not supported" in str(exc).lower()


def test_add_noise_auto_recognizes_bab_t_file(tmp_path) -> None:
	sr = 16000
	target = np.random.default_rng(11).standard_normal(sr).astype(np.float32) * 0.05
	noise = np.random.default_rng(12).standard_normal(sr * 2).astype(np.float32) * 0.05
	noise_path = tmp_path / "bab-t8.wav"
	save_audio(noise_path, noise, sr)

	auto_mixed = add_noise(
		target,
		snr_db=-3.0,
		noise_type="auto",
		noise_file=noise_path,
		sample_rate=sr,
		seed=17,
	)
	explicit_mixed = add_noise(
		target,
		snr_db=-3.0,
		noise_type="babble",
		noise_file=noise_path,
		sample_rate=sr,
		seed=17,
	)

	assert auto_mixed.shape == target.shape
	assert np.allclose(auto_mixed, explicit_mixed)


def test_add_noise_random_crop_when_noise_is_longer() -> None:
	target = np.ones(50, dtype=np.float32) * 0.1
	noise = np.arange(200, dtype=np.float32)
	seed = 5

	rng = np.random.default_rng(seed)
	start = int(rng.integers(0, len(noise) - len(target) + 1))
	expected = mix_at_snr(target, noise[start : start + len(target)], 0.0)

	mixed = add_noise(target, noise, snr_db=0.0, seed=seed)
	assert np.allclose(mixed, expected)


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
