"""Babble speech generation commands."""

from __future__ import annotations

from pathlib import Path

import typer
import numpy as np

from triton.core.io import load_audio, save_audio
from triton.core.mixer import mix_babble_from_segments
from triton.core.project import (
	load_project_config,
	log_project_event,
	select_babble_talker_groups,
)


babble_app = typer.Typer(add_completion=False, help="Generate babble speech")


@babble_app.command("mix")
def generate_babble(
	project_dir: Path = typer.Argument(
		..., help="Path to project directory"
	),
	num_talkers: int = typer.Option(
		..., "--num-talkers", help="Number of talkers to mix"
	),
	num_female_talkers: int | None = typer.Option(
		None, "--num-female-talkers", help="Number of female talkers to use"
	),
	num_male_talkers: int | None = typer.Option(
		None, "--num-male-talkers", help="Number of male talkers to use"
	),
	output_path: Path = typer.Option(
		Path("outputs/babble.wav"), help="Output audio file path"
	),
	target_rms: float = typer.Option(
		0.1, help="Target RMS level for each talker before mixing"
	),
	peak_normalize: bool = typer.Option(
		True, help="Peak normalize the final babble mix"
	),
):
	"""Generate babble speech from project files labeled as babble talkers.

	Talker files are selected from labels of the form ``bab-f1`` and ``bab-m1``.
	When female and male counts are omitted, the requested number of talkers is
	balanced as evenly as possible across the two sexes.
	"""
	project_dir = project_dir.expanduser().resolve()
	output_path = output_path.expanduser().resolve()
	output_path.parent.mkdir(parents=True, exist_ok=True)

	# Load project to get sample rate
	try:
		project = load_project_config(project_dir)
	except FileNotFoundError as exc:
		raise typer.BadParameter(f"Project not found: {project_dir}") from exc

	try:
		selected_groups = select_babble_talker_groups(
			project_dir,
			num_talkers=num_talkers,
			num_female_talkers=num_female_talkers,
			num_male_talkers=num_male_talkers,
		)
	except ValueError as exc:
		raise typer.BadParameter(str(exc)) from exc

	talker_segments: list[list[np.ndarray]] = []
	selected_files: list[str] = []
	for group in selected_groups:
		segments = []
		for file_path in group.files:
			audio, _ = load_audio(
				file_path,
				sr=project.sample_rate,
				mono=(project.channel_mode == "mono"),
			)
			segments.append(audio)
			selected_files.append(file_path.name)
		talker_segments.append(segments)

	try:
		babble = mix_babble_from_segments(
			talker_segments,
			target_rms=target_rms,
			peak_normalize=peak_normalize,
		)
	except ValueError as exc:
		raise typer.BadParameter(str(exc)) from exc

	save_audio(output_path, babble, project.sample_rate)

	log_project_event(
		project_dir,
		"babble_generated",
		{
			"output_path": str(output_path.resolve()),
			"talker_count": len(selected_groups),
			"talker_labels": [group.label for group in selected_groups],
			"talker_files": selected_files,
			"female_talkers": sum(1 for group in selected_groups if group.sex == "f"),
			"male_talkers": sum(1 for group in selected_groups if group.sex == "m"),
			"target_rms": target_rms,
			"peak_normalize": peak_normalize,
			"output_duration_seconds": float(len(babble) / project.sample_rate),
		},
	)

	typer.echo(
		f"Generated babble with {len(selected_groups)} talkers "
		f"({sum(1 for group in selected_groups if group.sex == 'f')} female, "
		f"{sum(1 for group in selected_groups if group.sex == 'm')} male) at {output_path}"
	)
