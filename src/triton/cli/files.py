"""File management commands."""

from __future__ import annotations

from pathlib import Path

import typer

from triton.core.project import (
	load_project_config,
	list_project_files,
	set_file_label,
	get_file_label,
	load_file_labels,
	project_raw_dir,
)


files_app = typer.Typer(add_completion=False, help="Manage project files")


@files_app.command("label")
def label_file(
	project_dir: Path = typer.Argument(..., help="Path to project directory"),
	filename: str = typer.Argument(..., help="Name of the file to label"),
	label: str = typer.Argument(..., help="Label to assign (empty string to remove)"),
):
	"""Set or update a label for a project file.
	
	Labels help organize and filter files. For example, you can label
	talker audio files as 'talker1', 'talker2', etc., for easy identification
	in other commands.
	"""
	project_dir = project_dir.expanduser().resolve()

	try:
		project = load_project_config(project_dir)
	except FileNotFoundError as exc:
		raise typer.BadParameter(f"Project not found: {project_dir}") from exc

	# Verify file exists in project
	raw_dir = project_raw_dir(project_dir)
	file_path = raw_dir / filename

	if not file_path.exists():
		raise typer.BadParameter(f"File not found in project: {filename}")

	try:
		set_file_label(project_dir, file_path, label)
	except ValueError as exc:
		raise typer.BadParameter(str(exc)) from exc

	if label.strip():
		typer.echo(f"Labeled '{filename}' as '{label}'")
	else:
		typer.echo(f"Removed label from '{filename}'")


@files_app.command("list")
def list_files(
	project_dir: Path = typer.Argument(..., help="Path to project directory"),
	label: str = typer.Option(None, help="Filter by label"),
):
	"""List audio files in the project, optionally filtered by label.
	
	Shows all audio files in the project's raw data directory. If a label
	is specified, only files with that label are shown.
	"""
	project_dir = project_dir.expanduser().resolve()

	try:
		project = load_project_config(project_dir)
	except FileNotFoundError as exc:
		raise typer.BadParameter(f"Project not found: {project_dir}") from exc

	files = list_project_files(project_dir, label=label)

	if not files:
		if label:
			typer.echo(f"No files found with label '{label}'")
		else:
			typer.echo("No files found in project")
		return

	# Load all labels for display
	all_labels = load_file_labels(project_dir)

	if label:
		typer.echo(f"Files with label '{label}':\n")
	else:
		typer.echo("Project files:\n")

	for file_path in files:
		file_label = all_labels.get(file_path.name)
		label_str = f" [{file_label}]" if file_label else ""
		typer.echo(f"  {file_path.name}{label_str}")

	typer.echo(f"\nTotal: {len(files)} file(s)")


@files_app.command("show-labels")
def show_labels(
	project_dir: Path = typer.Argument(..., help="Path to project directory"),
):
	"""Display all file labels in the project.
	
	Shows a summary of which files have labels and their assigned labels.
	"""
	project_dir = project_dir.expanduser().resolve()

	try:
		project = load_project_config(project_dir)
	except FileNotFoundError as exc:
		raise typer.BadParameter(f"Project not found: {project_dir}") from exc

	all_labels = load_file_labels(project_dir)

	if not all_labels:
		typer.echo("No labels assigned to any files")
		return

	typer.echo("File labels:\n")
	for filename, label in sorted(all_labels.items()):
		typer.echo(f"  {filename}: {label}")

	typer.echo(f"\nTotal labeled files: {len(all_labels)}")
