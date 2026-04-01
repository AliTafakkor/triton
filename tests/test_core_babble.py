from __future__ import annotations

import numpy as np

from triton.core.mixer import mix_babble_from_segments
from triton.core.project import (
	add_project_file,
	create_project,
	select_babble_talker_groups,
	set_file_label,
)


def test_select_babble_talker_groups_balances_by_sex(tmp_path) -> None:
	project = create_project(tmp_path / "demo", sample_rate=16000, channel_mode="mono")

	for filename, label in [
		("f1.wav", "bab-f1"),
		("f2.wav", "bab-f2"),
		("m1.wav", "bab-m1"),
		("m2.wav", "bab-m2"),
		("m3.wav", "bab-m3"),
	]:
		path = add_project_file(project.path, filename, b"RIFF")
		set_file_label(project.path, path, label)

	selected = select_babble_talker_groups(project.path, num_talkers=3)

	assert [group.label for group in selected] == ["bab-f1", "bab-m1", "bab-m2"]
	assert sum(group.sex == "f" for group in selected) == 1
	assert sum(group.sex == "m" for group in selected) == 2


def test_select_babble_talker_groups_explicit_split(tmp_path) -> None:
	project = create_project(tmp_path / "demo", sample_rate=16000, channel_mode="mono")

	for filename, label in [
		("f1.wav", "bab-f1"),
		("f2.wav", "bab-f2"),
		("m1.wav", "bab-m1"),
		("m2.wav", "bab-m2"),
	]:
		path = add_project_file(project.path, filename, b"RIFF")
		set_file_label(project.path, path, label)

	selected = select_babble_talker_groups(
		project.path,
		num_talkers=4,
		num_female_talkers=2,
		num_male_talkers=2,
	)

	assert [group.label for group in selected] == ["bab-f1", "bab-f2", "bab-m1", "bab-m2"]


def test_mix_babble_from_segments_concatenates_and_normalizes_segments() -> None:
	talker_segments = [
		[
			np.array([1.0, 1.0], dtype=np.float32),
			np.array([1.0, 1.0], dtype=np.float32),
		],
		[
			np.array([2.0, 2.0], dtype=np.float32),
		],
	]

	mixed = mix_babble_from_segments(talker_segments, target_rms=1.0, peak_normalize=False)

	assert mixed.shape == (4,)
	assert np.allclose(mixed, np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32))