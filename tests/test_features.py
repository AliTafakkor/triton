from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from triton.core.project import create_project, delete_project_file, rename_project_file
from triton.features.acoustic import envelope_feature
from triton.features.storage import feature_path, list_feature_files, load_feature, save_feature
from triton.features.words import onset_train


def test_envelope_feature_resamples_to_rate_and_tracks_amplitude() -> None:
	sr = 16000
	t = np.arange(2 * sr) / sr
	# 200 Hz tone that is silent for the first second and loud for the second.
	audio = (np.sin(2 * np.pi * 200 * t) * (t >= 1.0)).astype(np.float32)

	times, values = envelope_feature(audio, sr, rate=100, cutoff=10.0)

	assert values.shape == (200,)
	assert times[1] - times[0] == pytest.approx(0.01)
	# Standard |hilbert| envelope: near zero before onset (no DC leakage), ~1 during the tone.
	assert values[:80].mean() < 0.02
	assert values[120:180].mean() == pytest.approx(1.0, abs=0.05)


def test_envelope_feature_rejects_rate_above_sample_rate() -> None:
	with pytest.raises(ValueError):
		envelope_feature(np.zeros(100, dtype=np.float32), 50, rate=100)


def test_onset_train_places_impulses_on_grid() -> None:
	times, train = onset_train(np.array([0.0, 0.5, 0.504, 5.0]), duration=1.0, rate=100)

	assert train.shape == times.shape == (100,)
	# 0.5 and 0.504 round to the same sample; 5.0 is past the end and dropped.
	assert np.flatnonzero(train).tolist() == [0, 50]


def _fake_speech_model(num_layers: int = 3, hidden: int = 4):
	torch = pytest.importorskip("torch")

	def extractor(audio, sampling_rate, return_tensors):
		return {"input_values": torch.tensor(audio)[None, :]}

	def model(input_values, output_hidden_states):
		# Mimic the wav2vec2 conv encoder: one frame per 320 samples, 400-sample receptive field.
		n_frames = (input_values.shape[1] - 400) // 320 + 1
		states = tuple(torch.full((1, n_frames, hidden), float(layer)) for layer in range(num_layers + 1))
		return SimpleNamespace(hidden_states=states)

	model.config = SimpleNamespace(num_hidden_layers=num_layers)
	return extractor, model


def test_extract_hidden_states_selects_layers_and_chunks() -> None:
	from triton.features.neural import extract_hidden_states

	extractor, model = _fake_speech_model()
	audio = np.zeros(16000 * 3, dtype=np.float32)

	times, states, layers = extract_hidden_states(audio, extractor=extractor, model=model, layers=[3, 1], chunk_seconds=1.0)

	assert layers == [1, 3]
	# Three 1 s chunks of 49 frames each (one frame lost at each chunk edge).
	assert states.shape == (2, 147, 4)
	assert np.all(states[0] == 1.0) and np.all(states[1] == 3.0)
	assert times[0] == pytest.approx(0.0125)
	assert times[49] == pytest.approx(1.0125)
	assert np.all(np.diff(times) > 0)


def test_extract_hidden_states_validates_input() -> None:
	from triton.features.neural import extract_hidden_states

	extractor, model = _fake_speech_model()
	with pytest.raises(ValueError, match="out of range"):
		extract_hidden_states(np.zeros(16000, dtype=np.float32), extractor=extractor, model=model, layers=[9])
	with pytest.raises(ValueError, match="too short"):
		extract_hidden_states(np.zeros(100, dtype=np.float32), extractor=extractor, model=model)


def _project_with_file(tmp_path):
	project = create_project(tmp_path / "demo", sample_rate=16000, channel_mode="mono")
	audio_path = project.path / "data" / "normalized" / "talker1.wav"
	sf.write(audio_path, np.zeros(1600, dtype=np.float32), 16000)
	return project, audio_path


def test_save_and_load_feature_roundtrip_with_sidecar(tmp_path) -> None:
	project, audio_path = _project_with_file(tmp_path)

	path = save_feature(
		project.path, audio_path, "envelope",
		{"times": np.arange(3, dtype=np.float32), "envelope": np.ones(3, dtype=np.float32)},
		options={"rate": 100},
	)

	assert path == feature_path(project.path, audio_path, "envelope")
	assert path.parent.name == "talker1"
	arrays, meta = load_feature(path)
	assert set(arrays) == {"times", "envelope"}
	assert meta["options"] == {"rate": 100}
	sidecar = json.loads(path.with_suffix(".npz.json").read_text())
	assert sidecar["actions"][0]["step"] == "features.envelope"
	assert sidecar["extra"]["features"]["envelope"] == [3]
	assert list_feature_files(project.path) == [path]


def test_rename_and_delete_carry_features_along(tmp_path) -> None:
	project, audio_path = _project_with_file(tmp_path)
	save_feature(project.path, audio_path, "envelope", {"times": np.zeros(1)}, options={})

	renamed = rename_project_file(audio_path, "talker2.wav")

	moved = feature_path(project.path, renamed, "envelope")
	assert moved.exists()
	assert moved.with_suffix(".npz.json").exists()
	assert not feature_path(project.path, audio_path, "envelope").parent.exists()

	delete_project_file(renamed)
	assert list_feature_files(project.path) == []
