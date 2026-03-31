"""Pipeline matrix CLI commands."""

from __future__ import annotations

from pathlib import Path

import typer

from triton.core.pipeline_matrix import generate_matrix_csv, run_matrix_csv
from triton.core.pipeline_runtime import new_pipeline_run_id
from triton.core.project import log_project_event, load_project_config, load_project_pipelines


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
) -> None:
	"""Generate a CSV matrix from files and parameter combinations."""
	project_dir = project_dir.expanduser().resolve()
	project = load_project_config(project_dir)
	pipeline = _load_named_pipeline(project_dir, pipeline_name)

	rows = generate_matrix_csv(
		project,
		pipeline,
		output_csv=output_csv.expanduser().resolve(),
		parameter_specs=param or [],
		files=file or None,
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
			"succeeded": len(successes),
			"failed": len(errors),
		},
	)

	if errors:
		raise typer.Exit(code=1)
