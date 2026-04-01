"""Pipeline matrix generation and CSV-driven execution helpers."""

from __future__ import annotations

from collections.abc import Iterable
import csv
import itertools
from pathlib import Path

from triton.core.pipeline_runtime import pipeline_run_dir, run_pipeline_on_file
from triton.core.project import Pipeline, Project


MATRIX_FILE_COLUMN = "file"


def _parse_value(raw: str) -> object:
	text = raw.strip()
	if text == "":
		return ""
	lower = text.lower()
	if lower in {"true", "false"}:
		return lower == "true"
	try:
		if "." in text:
			return float(text)
		return int(text)
	except ValueError:
		return text


def parse_parameter_specs(parameter_specs: Iterable[str]) -> dict[str, list[object]]:
	parsed: dict[str, list[object]] = {}
	for spec in parameter_specs:
		text = spec.strip()
		if not text:
			continue
		if "=" not in text:
			raise ValueError(f"Invalid parameter spec '{spec}'. Use step.option=v1,v2")
		key, raw_values = text.split("=", 1)
		clean_key = key.strip()
		if clean_key.count(".") < 1:
			raise ValueError(f"Invalid parameter key '{clean_key}'. Use step.option")
		values = [_parse_value(item) for item in raw_values.split(",") if item.strip()]
		if not values:
			raise ValueError(f"No values provided for parameter '{clean_key}'.")
		parsed[clean_key] = values
	return parsed


def _resolve_file(project: Project, file_value: str) -> Path:
	candidate = Path(file_value).expanduser()
	if candidate.is_absolute():
		path = candidate
	else:
		raw_path = project.raw_dir / candidate
		if raw_path.exists():
			path = raw_path
		else:
			path = project.path / candidate
	if not path.exists() or not path.is_file():
		raise FileNotFoundError(f"Matrix row file not found: {file_value}")
	return path.resolve()


def _parse_option_key(key: str) -> tuple[str, str]:
	if "." not in key:
		raise ValueError(f"Invalid option column '{key}'. Use step.option")
	step_ref, option_name = key.split(".", 1)
	step_ref = step_ref.strip()
	option_name = option_name.strip()
	if not step_ref or not option_name:
		raise ValueError(f"Invalid option column '{key}'. Use step.option")
	return step_ref, option_name


def _collect_row_overrides(row: dict[str, str]) -> dict[str, dict[str, object]]:
	overrides: dict[str, dict[str, object]] = {}
	for key, value in row.items():
		if key == MATRIX_FILE_COLUMN:
			continue
		if value is None or str(value).strip() == "":
			continue
		step_ref, option_name = _parse_option_key(str(key))
		overrides.setdefault(step_ref, {})[option_name] = _parse_value(str(value))
	return overrides


def _merge_step_options(base: dict[str, dict[str, object]], row_overrides: dict[str, dict[str, object]], steps: list[str]) -> dict[str, dict[str, object]]:
	merged = {str(key): dict(value) for key, value in base.items()}
	for step_ref, options in row_overrides.items():
		target_key = step_ref
		if step_ref.isdigit():
			target_key = str(int(step_ref))
		elif step_ref in steps:
			target_key = step_ref
		merged.setdefault(target_key, {})
		merged[target_key].update(options)
	return merged


def generate_matrix_csv(
	project: Project,
	pipeline: Pipeline,
	output_csv: Path,
	parameter_specs: Iterable[str],
	files: Iterable[str] | None = None,
) -> int:
	file_values = list(files or [path.name for path in project.list_files()])
	if not file_values:
		raise ValueError("No input files available to include in matrix CSV.")

	parameter_grid = parse_parameter_specs(parameter_specs)
	parameter_keys = list(parameter_grid.keys())

	if parameter_keys:
		combinations = list(itertools.product(*(parameter_grid[key] for key in parameter_keys)))
	else:
		combinations = [tuple()]

	rows: list[dict[str, object]] = []
	for file_value in file_values:
		for combo in combinations:
			row: dict[str, object] = {MATRIX_FILE_COLUMN: file_value}
			for index, key in enumerate(parameter_keys):
				row[key] = combo[index]
			rows.append(row)

	fieldnames = [MATRIX_FILE_COLUMN, *parameter_keys]
	output_csv.parent.mkdir(parents=True, exist_ok=True)
	with output_csv.open("w", encoding="utf-8", newline="") as handle:
		writer = csv.DictWriter(handle, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(rows)

	return len(rows)


def run_matrix_csv(
	project: Project,
	pipeline: Pipeline,
	matrix_csv: Path,
	run_id: str,
) -> tuple[list[Path], list[str], Path]:
	if not matrix_csv.exists():
		raise FileNotFoundError(f"Matrix CSV does not exist: {matrix_csv}")

	base_run_dir = pipeline_run_dir(project, pipeline.name, run_id)
	successes: list[Path] = []
	errors: list[str] = []

	with matrix_csv.open("r", encoding="utf-8", newline="") as handle:
		reader = csv.DictReader(handle)
		if not reader.fieldnames or MATRIX_FILE_COLUMN not in set(reader.fieldnames):
			raise ValueError(f"Matrix CSV must include '{MATRIX_FILE_COLUMN}' column.")

		for row_index, row in enumerate(reader, start=1):
			file_value = str(row.get(MATRIX_FILE_COLUMN, "")).strip()
			if not file_value:
				errors.append(f"row {row_index}: missing file value")
				continue

			try:
				file_path = _resolve_file(project, file_value)
				row_overrides = _collect_row_overrides(row)
				merged_options = _merge_step_options(pipeline.step_options, row_overrides, pipeline.steps)
				row_pipeline = Pipeline(name=pipeline.name, steps=list(pipeline.steps), step_options=merged_options)
				row_run_dir = base_run_dir / f"row_{row_index:04d}"
				final_path = run_pipeline_on_file(file_path, project, row_pipeline, row_run_dir)
			except Exception as exc:
				errors.append(f"row {row_index}: {file_value}: {exc}")
			else:
				successes.append(final_path)

	return successes, errors, base_run_dir
