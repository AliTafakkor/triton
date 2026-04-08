"""File management commands."""

from __future__ import annotations

from pathlib import Path

import typer

from triton.core.project import (
	delete_project_files_by_label,
	load_project_config,
	list_project_files,
	set_file_label,
	get_file_label,
	load_file_labels,
	project_normalized_dir,
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

	# Verify file exists in project (normalized dir is the canonical location)
	norm_dir = project_normalized_dir(project_dir)
	file_path = norm_dir / filename

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
		file_labels = all_labels.get(file_path.stem, [])
		label_str = f" [{', '.join(file_labels)}]" if file_labels else ""
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
	for filename, file_labels in sorted(all_labels.items()):
		typer.echo(f"  {filename}: {', '.join(file_labels)}")

	typer.echo(f"\nTotal labeled files: {len(all_labels)}")


@files_app.command("delete-label")
def delete_files_by_label(
	project_dir: Path = typer.Argument(..., help="Path to project directory"),
	label: str = typer.Argument(..., help="Label to match for deletion"),
	yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
):
	"""Delete all project files that carry the given label.

	This removes matching normalized files and their raw counterparts.
	"""
	project_dir = project_dir.expanduser().resolve()

	try:
		project = load_project_config(project_dir)
	except FileNotFoundError as exc:
		raise typer.BadParameter(f"Project not found: {project_dir}") from exc

	clean_label = label.strip()
	if not clean_label:
		raise typer.BadParameter("Label cannot be empty")

	files = list_project_files(project_dir, label=clean_label)
	if not files:
		typer.echo(f"No files found with label '{clean_label}'")
		return

	if not yes:
		confirmed = typer.confirm(
			f"Delete {len(files)} file(s) with label '{clean_label}'?",
			default=False,
		)
		if not confirmed:
			typer.echo("Cancelled")
			return

	deleted = delete_project_files_by_label(project_dir, clean_label)
	typer.echo(f"Deleted {len(deleted)} file(s) with label '{clean_label}'")
	for file_path in deleted:
		typer.echo(f"  {file_path.name}")
