"""Pipeline matrix CLI commands."""

from __future__ import annotations

from pathlib import Path

import typer

from triton.core.pipeline_matrix import generate_matrix_csv, run_matrix_csv
from triton.core.pipeline_runtime import new_pipeline_run_id, run_pipeline_on_file
from triton.core.project import log_project_event, load_project_config, load_project_pipelines, list_project_files


matrix_app = typer.Typer(add_completion=False, help="Generate and run pipeline parameter matrices")


def _load_named_pipeline(project_dir: Path, pipeline_name: str):
	pipelines = load_project_pipelines(project_dir)
	for pipeline in pipelines:
		if pipeline.name == pipeline_name:
			return pipeline
	raise typer.BadParameter(f"Pipeline '{pipeline_name}' not found in project config.")


@matrix_app.command("generate")
def matrix_generate(
	project_dir: Path = typer.Argument(..., help="Path to Triton project"),
	pipeline_name: str = typer.Argument(..., help="Pipeline name defined in triton.toml"),
	output_csv: Path = typer.Argument(..., help="Output CSV path"),
	param: list[str] = typer.Option(None, "--param", help="Parameter spec: step.option=v1,v2"),
	file: list[str] = typer.Option(None, "--file", help="Input file (relative to data/raw or absolute); repeatable"),
	label: str | None = typer.Option(None, "--label", help="Include all files with this label"),
) -> None:
	"""Generate a CSV matrix from files and parameter combinations."""
	project_dir = project_dir.expanduser().resolve()
	project = load_project_config(project_dir)
	pipeline = _load_named_pipeline(project_dir, pipeline_name)

	# Collect files from --file and --label options
	files_to_process = list(file or [])
	if label:
		labeled_files = list_project_files(project_dir, label=label)
		files_to_process.extend([str(f) for f in labeled_files])

	rows = generate_matrix_csv(
		project,
		pipeline,
		output_csv=output_csv.expanduser().resolve(),
		parameter_specs=param or [],
		files=files_to_process or None,
	)

	typer.echo(f"Wrote {rows} matrix row(s) to {output_csv}")
	log_project_event(
		project.path,
		"pipeline_matrix_generated",
		{
			"pipeline": pipeline.name,
			"output_csv": str(output_csv),
			"rows": int(rows),
			"params": list(param or []),
		},
	)


@matrix_app.command("run")
def matrix_run(
	project_dir: Path = typer.Argument(..., help="Path to Triton project"),
	pipeline_name: str = typer.Argument(..., help="Pipeline name defined in triton.toml"),
	matrix_csv: Path = typer.Argument(..., help="CSV matrix path"),
	run_id: str | None = typer.Option(None, help="Optional run id"),
	group_finals_by_params: bool = typer.Option(False, "--group-finals-by-params", help="Copy final outputs into folders grouped by parameter set"),
) -> None:
	"""Run a pipeline for every row in a matrix CSV."""
	project_dir = project_dir.expanduser().resolve()
	project = load_project_config(project_dir)
	pipeline = _load_named_pipeline(project_dir, pipeline_name)
	resolved_run_id = run_id or new_pipeline_run_id()

	successes, errors, base_run_dir = run_matrix_csv(
		project,
		pipeline,
		matrix_csv=matrix_csv.expanduser().resolve(),
		run_id=resolved_run_id,
		collect_finals_by_params=group_finals_by_params,
	)

	typer.echo(f"Run folder: {base_run_dir}")
	typer.echo(f"Succeeded: {len(successes)}")
	typer.echo(f"Failed: {len(errors)}")
	for error in errors:
		typer.echo(error)

	log_project_event(
		project.path,
		"pipeline_matrix_run_completed",
		{
			"pipeline": pipeline.name,
			"run_id": resolved_run_id,
			"matrix_csv": str(matrix_csv),
			"run_dir": str(base_run_dir),
			"group_finals_by_params": bool(group_finals_by_params),
			"succeeded": len(successes),
			"failed": len(errors),
		},
	)

	if errors:
		raise typer.Exit(code=1)


@matrix_app.command("run-label")
def matrix_run_label(
	project_dir: Path = typer.Argument(..., help="Path to Triton project"),
	pipeline_name: str = typer.Argument(..., help="Pipeline name defined in triton.toml"),
	label: str = typer.Argument(..., help="File label to run on"),
	run_id: str | None = typer.Option(None, help="Optional run id"),
) -> None:
	"""Run a pipeline on all files with a specific label."""
	project_dir = project_dir.expanduser().resolve()
	project = load_project_config(project_dir)
	pipeline = _load_named_pipeline(project_dir, pipeline_name)
	resolved_run_id = run_id or new_pipeline_run_id()

	# Get all files with the specified label
	labeled_files = list_project_files(project_dir, label=label)

	if not labeled_files:
		typer.echo(f"No files found with label '{label}'")
		raise typer.Exit(code=1)

	from triton.gui.shared import _pipeline_run_dir

	successes: list[Path] = []
	errors: list[str] = []
	run_dir = _pipeline_run_dir(project, pipeline.name, resolved_run_id)

	with typer.progressbar(labeled_files, label=f"Running {pipeline.name}") as progress:
		for file_path in progress:
			try:
				output_path = run_pipeline_on_file(file_path, project, pipeline, run_dir)
				successes.append(output_path)
			except Exception as exc:
				errors.append(f"{file_path.name}: {exc}")

	typer.echo(f"Run folder: {run_dir}")
	typer.echo(f"Succeeded: {len(successes)}")
	typer.echo(f"Failed: {len(errors)}")
	for error in errors:
		typer.echo(error)

	log_project_event(
		project.path,
		"pipeline_run_label_completed",
		{
			"pipeline": pipeline.name,
			"run_id": resolved_run_id,
			"label": label,
			"run_dir": str(run_dir),
			"requested_files": len(labeled_files),
			"succeeded": len(successes),
			"failed": len(errors),
		},
	)

	if errors:
		raise typer.Exit(code=1)

