from __future__ import annotations

from triton.core.project import (
	add_project_file,
	create_project,
	delete_project_file,
	load_file_labels,
	log_project_event,
	read_project_log,
	rename_project_file,
	set_project_file_labels,
)


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
