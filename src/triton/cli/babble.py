"""Babble speech generation commands."""

from __future__ import annotations

from pathlib import Path

import typer

from triton.core.io import save_audio
from triton.core.project import (
	load_project_config,
	log_project_event,
)
from triton.degrade.noise_generator import generate_project_babble


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
	intended_length_seconds: float = typer.Option(
		30.0,
		"--intended-length-seconds",
		help="Per-talker target length; files are repeated randomly when short.",
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
		result = generate_project_babble(
			project_dir,
			sr=project.sample_rate,
			channel_mode=project.channel_mode,
			num_talkers=num_talkers,
			num_female_talkers=num_female_talkers,
			num_male_talkers=num_male_talkers,
			intended_length_seconds=float(intended_length_seconds),
			target_rms=target_rms,
			peak_normalize=peak_normalize,
			progress_callback=lambda message, _: typer.echo(message),
		)
	except (ValueError, RuntimeError) as exc:
		raise typer.BadParameter(str(exc)) from exc

	save_audio(output_path, result.audio, project.sample_rate)

	log_project_event(
		project_dir,
		"babble_generated",
		{
			"output_path": str(output_path.resolve()),
			"talker_count": len(result.selected_groups),
			"talker_labels": [group.label for group in result.selected_groups],
			"talker_files": [
				file_path.name
				for files in result.planned_group_files
				for file_path in files
			],
			"female_talkers": sum(1 for group in result.selected_groups if group.sex == "f"),
			"male_talkers": sum(1 for group in result.selected_groups if group.sex == "m"),
			"short_source_labels": result.short_source_labels,
			"unknown_duration_labels": result.unknown_duration_labels,
			"talker_repeats": result.repeat_counts_by_label,
			"intended_length_seconds": float(intended_length_seconds),
			"target_rms": target_rms,
			"peak_normalize": peak_normalize,
			"output_duration_seconds": float(result.audio.shape[-1] / project.sample_rate),
		},
	)

	typer.echo(
		f"Generated babble with {len(result.selected_groups)} talkers "
		f"({sum(1 for group in result.selected_groups if group.sex == 'f')} female, "
		f"{sum(1 for group in result.selected_groups if group.sex == 'm')} male) at {output_path}"
	)
