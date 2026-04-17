from __future__ import annotations

import csv

import numpy as np

from triton.core.io import save_audio
from triton.core.pipeline_matrix import generate_matrix_csv, run_matrix_csv
from triton.core.pipeline_runtime import new_pipeline_run_id
from triton.core.project import Pipeline, create_project, project_normalized_dir


def test_generate_matrix_csv_builds_cartesian_product(tmp_path) -> None:
	project = create_project(tmp_path / "demo", sample_rate=16000, channel_mode="mono")
	norm_dir = project_normalized_dir(project.path)
	norm_dir.mkdir(parents=True, exist_ok=True)
	source_a = norm_dir / "a.wav"
	source_b = norm_dir / "b.wav"
	audio = np.zeros(1600, dtype=np.float32)
	save_audio(source_a, audio, 16000)
	save_audio(source_b, audio, 16000)

	pipeline = Pipeline(name="normalize_only", steps=["normalize"])
	matrix_csv = tmp_path / "matrix.csv"
	rows = generate_matrix_csv(
		project,
		pipeline,
		output_csv=matrix_csv,
		parameter_specs=["0.target_peak=0.6,0.9"],
	)

	assert rows == 4
	with matrix_csv.open("r", encoding="utf-8", newline="") as handle:
		reader = csv.DictReader(handle)
		items = list(reader)

	assert len(items) == 4
	assert set(row["file"] for row in items) == {"a.wav", "b.wav"}
	assert set(row["0.target_peak"] for row in items) == {"0.6", "0.9"}


def test_run_matrix_csv_creates_row_subfolders(tmp_path) -> None:
	project = create_project(tmp_path / "demo", sample_rate=16000, channel_mode="mono")
	norm_dir = project_normalized_dir(project.path)
	norm_dir.mkdir(parents=True, exist_ok=True)
	source_path = norm_dir / "sample.wav"
	t = np.arange(1600, dtype=np.float32) / 16000.0
	audio = np.sin(2 * np.pi * 220 * t).astype(np.float32)
	save_audio(source_path, audio, 16000)

	pipeline = Pipeline(name="normalize_only", steps=["normalize"])
	matrix_csv = tmp_path / "matrix.csv"
	matrix_csv.write_text("file,0.target_peak\nsample.wav,0.5\nsample.wav,0.8\n", encoding="utf-8")

	successes, errors, base_run_dir = run_matrix_csv(
		project,
		pipeline,
		matrix_csv=matrix_csv,
		run_id=new_pipeline_run_id(),
	)

	assert errors == []
	assert len(successes) == 2
	assert (base_run_dir / "row_0001" / "step_01_normalize" / "sample.wav").exists()
	assert (base_run_dir / "row_0002" / "step_01_normalize" / "sample.wav").exists()


def test_run_matrix_csv_groups_finals_by_parameter_set(tmp_path) -> None:
	project = create_project(tmp_path / "demo", sample_rate=16000, channel_mode="mono")
	norm_dir = project_normalized_dir(project.path)
	norm_dir.mkdir(parents=True, exist_ok=True)
	base_t = np.arange(1600, dtype=np.float32) / 16000.0
	save_audio(norm_dir / "a.wav", np.sin(2 * np.pi * 220 * base_t).astype(np.float32), 16000)
	save_audio(norm_dir / "b.wav", np.sin(2 * np.pi * 330 * base_t).astype(np.float32), 16000)

	pipeline = Pipeline(name="normalize_only", steps=["normalize"])
	matrix_csv = tmp_path / "matrix.csv"
	matrix_csv.write_text(
		"file,0.target_peak\n"
		"a.wav,0.5\n"
		"b.wav,0.5\n"
		"a.wav,0.8\n"
		"b.wav,0.8\n",
		encoding="utf-8",
	)

	successes, errors, base_run_dir = run_matrix_csv(
		project,
		pipeline,
		matrix_csv=matrix_csv,
		run_id=new_pipeline_run_id(),
		collect_finals_by_params=True,
	)

	assert errors == []
	assert len(successes) == 4
	finals_root = base_run_dir / "final_by_params"
	set_dirs = sorted(path for path in finals_root.iterdir() if path.is_dir())
	assert len(set_dirs) == 2
	for set_dir in set_dirs:
		assert (set_dir / "a.wav").exists()
		assert (set_dir / "b.wav").exists()
