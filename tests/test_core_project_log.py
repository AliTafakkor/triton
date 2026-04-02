from __future__ import annotations

import json

import numpy as np

from triton.core.project import (
	add_project_file,
	create_project,
	delete_project_file,
	load_file_labels,
	log_project_event,
	read_project_log,
	rename_project_file,
	save_project_generated_audio,
	set_project_file_labels,
)
from triton.core.io import sidecar_path


def test_project_log_records_creation(tmp_path) -> None:
	project = create_project(tmp_path / "demo", sample_rate=16000, channel_mode="mono")
	events = read_project_log(project.path)

	assert events
	assert any(event.get("event") == "project_created" for event in events)


def test_project_log_records_file_lifecycle(tmp_path) -> None:
	project = create_project(tmp_path / "demo", sample_rate=16000, channel_mode="mono")
	first = add_project_file(project.path, "sample.wav", b"RIFF")
	second = rename_project_file(first, "sample2.wav")
	delete_project_file(second)

	events = read_project_log(project.path)
	event_names = [str(item.get("event")) for item in events]

	assert "file_imported" in event_names
	assert "file_renamed" in event_names
	assert "file_deleted" in event_names


def test_project_log_manual_event(tmp_path) -> None:
	project = create_project(tmp_path / "demo", sample_rate=16000, channel_mode="mono")
	log_project_event(project.path, "custom_event", {"ok": True})

	events = read_project_log(project.path)
	assert any(item.get("event") == "custom_event" for item in events)


def test_bulk_file_label_assignment(tmp_path) -> None:
	project = create_project(tmp_path / "demo", sample_rate=16000, channel_mode="mono")
	first = add_project_file(project.path, "one.wav", b"RIFF")
	second = add_project_file(project.path, "two.wav", b"RIFF")

	set_project_file_labels(project.path, [first, second], "bab-f1")

	labels = load_file_labels(project.path)
	assert labels == {"one.wav": "bab-f1", "two.wav": "bab-f1"}


def test_project_babble_artifact_writes_sidecar_and_label(tmp_path) -> None:
	project = create_project(tmp_path / "demo", sample_rate=16000, channel_mode="mono")
	audio = np.zeros(1600, dtype=np.float32)
	out_path = save_project_generated_audio(
		project.path,
		"babble.wav",
		audio,
		16000,
		label="bab-t4",
		source={"type": "project_babble", "project_path": str(project.path.resolve())},
		actions=[{"step": "generate_project_babble", "options": {"num_talkers": 4}}],
		extra={"babble": {"num_talkers": 4, "target_rms": 0.1}},
	)

	labels = load_file_labels(project.path)
	assert labels[out_path.name] == "bab-t4"

	payload = json.loads(sidecar_path(out_path).read_text(encoding="utf-8"))
	assert payload["source"]["type"] == "project_babble"
	assert payload["actions"][0]["step"] == "generate_project_babble"
	assert payload["extra"]["babble"]["num_talkers"] == 4
