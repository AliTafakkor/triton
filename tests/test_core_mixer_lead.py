"""Noise that starts before the target and ends after it (lead / lag), and the pad_silence step."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from triton.core.io import rms
from triton.core.mixer import mix_at_snr, mix_at_snr_with_lead
from triton.core.pipeline_runtime import apply_pipeline_step, default_step_options
from triton.core.project import Project
from triton.degrade.noise_mixer import add_noise

SR = 16000


def _speech(seconds: float = 1.0, seed: int = 0) -> np.ndarray:
	t = np.arange(int(seconds * SR)) / SR
	rng = np.random.default_rng(seed)
	envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)
	return (0.2 * envelope * np.sin(2 * np.pi * 220 * t) + 0.01 * rng.standard_normal(t.size)).astype(np.float32)


def _noise(seconds: float, seed: int = 1) -> np.ndarray:
	return np.random.default_rng(seed).standard_normal(int(seconds * SR)).astype(np.float32) * 0.3


def test_without_lead_or_lag_it_is_mix_at_snr() -> None:
	speech, noise = _speech(), _noise(1.0)
	np.testing.assert_allclose(mix_at_snr_with_lead(speech, noise, 3.0), mix_at_snr(speech, noise, 3.0), atol=1e-7)


@pytest.mark.parametrize("snr_db", [-6.0, 0.0, 12.0])
def test_speech_span_matches_mix_at_snr_and_noise_fills_lead_and_lag(snr_db: float) -> None:
	speech = _speech()
	lead, lag = SR // 2, SR // 4
	noise = _noise(2.0)[: lead + speech.size + lag]
	mixed = mix_at_snr_with_lead(speech, noise, snr_db, lead_samples=lead, lag_samples=lag)

	assert mixed.size == lead + speech.size + lag
	span = slice(lead, lead + speech.size)
	# Over the speech, the result is exactly what mix_at_snr gives with the noise under the speech.
	np.testing.assert_allclose(mixed[span], mix_at_snr(speech, noise[span], snr_db), atol=1e-6)
	assert float(rms(mixed[span])) == pytest.approx(float(rms(speech)), rel=1e-5)
	# Before and after the speech there is only noise, at the level it has under the speech.
	scale = mixed[:lead] / noise[:lead]
	np.testing.assert_allclose(scale, scale[0], rtol=1e-4)
	np.testing.assert_allclose(mixed[-lag:] / noise[-lag:], scale[0], rtol=1e-4)


def test_target_rms_sets_the_level_over_the_speech() -> None:
	speech = _speech()
	mixed = mix_at_snr_with_lead(speech, _noise(2.0), 0.0, lead_samples=SR // 2, lag_samples=0, target_rms=0.05)
	assert float(rms(mixed[SR // 2:])) == pytest.approx(0.05, rel=1e-5)


def test_negative_lead_is_rejected() -> None:
	with pytest.raises(ValueError, match="negative"):
		mix_at_snr_with_lead(_speech(), _noise(2.0), 0.0, lead_samples=-1)


def test_add_noise_with_lead_and_lag_is_longer_and_repeatable() -> None:
	speech, noise = _speech(), _noise(10.0)
	a = add_noise(speech, noise, snr_db=0.0, sample_rate=SR, seed=7, lead_s=1.0, lag_s=0.5)
	b = add_noise(speech, noise, snr_db=0.0, sample_rate=SR, seed=7, lead_s=1.0, lag_s=0.5)
	assert a.size == speech.size + int(1.5 * SR)
	np.testing.assert_array_equal(a, b)


def test_add_noise_without_lead_is_unchanged() -> None:
	speech, noise = _speech(), _noise(10.0)
	old_style = add_noise(speech, noise, snr_db=2.0, sample_rate=SR, seed=3)
	explicit_zero = add_noise(speech, noise, snr_db=2.0, sample_rate=SR, seed=3, lead_s=0.0, lag_s=0.0)
	np.testing.assert_array_equal(old_style, explicit_zero)
	assert old_style.size == speech.size


def test_add_noise_lead_needs_a_sample_rate() -> None:
	with pytest.raises(ValueError, match="sample_rate"):
		add_noise(_speech(), _noise(10.0), snr_db=0.0, lead_s=1.0)


def test_pipeline_add_noise_step_passes_lead_and_lag(tmp_path: Path) -> None:
	project = Project(name="t", path=tmp_path, sample_rate=SR, channel_mode="mono")
	noise_path = tmp_path / "noise.wav"
	import soundfile as sf

	sf.write(noise_path, _noise(10.0), SR)
	options = {**default_step_options("add_noise", SR), "noise_file": str(noise_path), "snr_db": 0.0,
	           "seed": 1, "lead_s": 1.0, "lag_s": 0.5}
	out, sr = apply_pipeline_step(_speech(), SR, "add_noise", project, options)
	assert sr == SR and out.size == SR + int(1.5 * SR)
	assert default_step_options("add_noise", SR)["lead_s"] == 0.0


def test_pad_silence_step_adds_silence_without_changing_the_signal(tmp_path: Path) -> None:
	project = Project(name="t", path=tmp_path, sample_rate=SR, channel_mode="mono")
	speech = _speech()
	out, _ = apply_pipeline_step(speech, SR, "pad_silence", project, {"lead_s": 1.0, "lag_s": 0.5})
	assert out.size == speech.size + int(1.5 * SR)
	assert not np.any(out[:SR]) and not np.any(out[-SR // 2:])
	np.testing.assert_array_equal(out[SR:SR + speech.size], speech)

	stereo = np.stack([speech, speech], axis=1)
	out2, _ = apply_pipeline_step(stereo, SR, "pad_silence", project, {"lead_s": 0.25, "lag_s": 0.0})
	assert out2.shape == (speech.size + SR // 4, 2)
	assert default_step_options("pad_silence", SR) == {"lead_s": 0.0, "lag_s": 0.0}
