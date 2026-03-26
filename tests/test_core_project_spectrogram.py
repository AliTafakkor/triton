from __future__ import annotations

import tomllib

from triton.core.project import create_project, load_project_spectrogram_settings, project_config_path


def test_project_config_contains_spectrogram_defaults(tmp_path) -> None:
	project = create_project(tmp_path / "demo", sample_rate=16000, channel_mode="mono")
	config = tomllib.loads(project_config_path(project.path).read_text(encoding="utf-8"))

	assert "spectrogram" in config
	assert config["spectrogram"]["type"] in {"stft", "mel", "cqt"}
	assert int(config["spectrogram"]["n_fft"]) > 0


def test_load_project_spectrogram_settings_applies_defaults(tmp_path) -> None:
	project = create_project(tmp_path / "demo", sample_rate=22050, channel_mode="stereo")
	settings = load_project_spectrogram_settings(project.path)

	assert settings["type"] == "stft"
	assert int(settings["n_fft"]) == 1024
	assert int(settings["n_mels"]) == 128
