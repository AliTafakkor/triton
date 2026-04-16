from __future__ import annotations

import json

import numpy as np

from triton.core.io import save_audio, sidecar_path
from triton.core.project import Pipeline, create_project
from triton.gui.app import _pipeline_run_dir, _run_pipeline_on_file


def test_pipeline_run_writes_step_folders_and_sidecars(tmp_path) -> None:
	project = create_project(tmp_path / "demo", sample_rate=16000, channel_mode="mono")
	source_path = project.path / "data" / "raw" / "sample.wav"

	t = np.arange(2400, dtype=np.float32) / 16000.0
	audio = np.stack([
		np.sin(2 * np.pi * 220 * t),
		np.sin(2 * np.pi * 440 * t),
	], axis=1).astype(np.float32)
	save_audio(source_path, audio, 16000)

	pipeline = Pipeline(
		name="demo_pipeline",
		steps=["normalize", "to_mono", "add_noise"],
		step_options={
			"0": {"target_peak": 0.8},
			"2": {"noise_type": "white", "snr_db": 0.0, "seed": 0},
		},
	)
	run_dir = _pipeline_run_dir(project, pipeline.name, "unit_test")
	final_path = _run_pipeline_on_file(source_path, project, pipeline, run_dir)

	step1_path = run_dir / "step_01_normalize" / "sample.wav"
	step2_path = run_dir / "step_02_to_mono" / "sample.wav"
	step3_path = run_dir / "step_03_add_noise" / "sample.wav"
	assert step1_path.exists()
	assert step2_path.exists()
	assert step3_path.exists()
	assert final_path == step3_path

	step1_meta = json.loads(sidecar_path(step1_path).read_text(encoding="utf-8"))
	step2_meta = json.loads(sidecar_path(step2_path).read_text(encoding="utf-8"))
	step3_meta = json.loads(sidecar_path(step3_path).read_text(encoding="utf-8"))

	assert step1_meta["source"]["path"] == str(source_path.resolve())
	assert len(step1_meta["actions"]) == 1
	assert step1_meta["actions"][0]["step"] == "normalize"

	assert len(step2_meta["actions"]) == 2
	assert step2_meta["actions"][1]["step"] == "to_mono"
	assert step2_meta["extra"]["pipeline"]["name"] == "demo_pipeline"

	assert len(step3_meta["actions"]) == 3
	assert step3_meta["actions"][2]["step"] == "add_noise"


def test_pipeline_add_noise_uses_project_noise_selection(tmp_path) -> None:
	project = create_project(tmp_path / "demo_select_noise", sample_rate=16000, channel_mode="mono")
	source_path = project.path / "data" / "raw" / "speech.wav"
	noise_path = project.path / "data" / "normalized" / "bab-t8.wav"

	t = np.arange(3200, dtype=np.float32) / 16000.0
	speech = np.sin(2 * np.pi * 220 * t).astype(np.float32)
	noise = np.sin(2 * np.pi * 90 * np.arange(6400, dtype=np.float32) / 16000.0).astype(np.float32)
	save_audio(source_path, speech, 16000)
	save_audio(noise_path, noise, 16000)

	pipeline = Pipeline(
		name="noise_selector_pipeline",
		steps=["add_noise"],
		step_options={
			"0": {
				"noise_type": "auto",
				"noise_project_file": "data/normalized/bab-t8.wav",
				"snr_db": -5.0,
				"seed": 3,
			}
		},
	)
	run_dir = _pipeline_run_dir(project, pipeline.name, "unit_test_selector")
	final_path = _run_pipeline_on_file(source_path, project, pipeline, run_dir)

	assert final_path.exists()
	meta = json.loads(sidecar_path(final_path).read_text(encoding="utf-8"))
	assert meta["actions"][0]["step"] == "add_noise"
	assert meta["actions"][0]["options"]["noise_project_file"] == "data/normalized/bab-t8.wav"
