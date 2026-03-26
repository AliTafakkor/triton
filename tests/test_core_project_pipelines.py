from __future__ import annotations

from triton.core.project import (
	Pipeline,
	create_project,
	load_project_pipelines,
	save_project_pipelines,
)


def test_project_pipeline_round_trip(tmp_path) -> None:
	project_dir = tmp_path / "demo"
	create_project(project_dir, sample_rate=16000, channel_mode="mono")

	pipelines = [
		Pipeline(name="degrade_and_normalize", steps=["vocode_noise", "normalize"]),
		Pipeline(name="canonicalize", steps=["resample_project", "to_mono", "requantize_16"]),
	]

	save_project_pipelines(project_dir, pipelines)
	loaded = load_project_pipelines(project_dir)

	assert [item.name for item in loaded] == ["degrade_and_normalize", "canonicalize"]
	assert loaded[0].steps == ["vocode_noise", "normalize"]
	assert loaded[1].steps == ["resample_project", "to_mono", "requantize_16"]


def test_save_empty_pipelines_clears_config_section(tmp_path) -> None:
	project_dir = tmp_path / "demo"
	create_project(project_dir, sample_rate=22050, channel_mode="stereo")

	save_project_pipelines(project_dir, [Pipeline(name="tmp", steps=["normalize"])])
	save_project_pipelines(project_dir, [])

	loaded = load_project_pipelines(project_dir)
	assert loaded == []
