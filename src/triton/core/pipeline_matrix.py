"""Pipeline matrix generation and CSV-driven execution helpers."""

from __future__ import annotations

from collections.abc import Callable, Iterable
import csv
import itertools
from pathlib import Path
import re
import shutil

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
		norm_path = project.normalized_dir / candidate
		if norm_path.exists():
			path = norm_path
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


def _row_override_signature(row_overrides: dict[str, dict[str, object]]) -> tuple[tuple[str, str], ...]:
	flattened: list[tuple[str, str]] = []
	for step_ref in sorted(row_overrides.keys()):
		for option_name in sorted(row_overrides[step_ref].keys()):
			flattened.append((f"{step_ref}.{option_name}", str(row_overrides[step_ref][option_name])))
	return tuple(flattened)


def _sanitize_slug(text: str) -> str:
	clean = re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-._")
	return clean or "set"


def _signature_label(signature: tuple[tuple[str, str], ...]) -> str:
	if not signature:
		return "default"
	parts = [f"{key}={value}" for key, value in signature[:3]]
	label = "__".join(parts)
	return _sanitize_slug(label)[:72]


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
	collect_finals_by_params: bool = False,
	progress_callback: Callable[[int, int, str], None] | None = None,
) -> tuple[list[Path], list[str], Path]:
	if not matrix_csv.exists():
		raise FileNotFoundError(f"Matrix CSV does not exist: {matrix_csv}")

	base_run_dir = pipeline_run_dir(project, pipeline.name, run_id)
	successes: list[Path] = []
	errors: list[str] = []
	param_set_dirs: dict[tuple[tuple[str, str], ...], Path] = {}
	finals_root = base_run_dir / "final_by_params"
	if collect_finals_by_params:
		finals_root.mkdir(parents=True, exist_ok=True)

	with matrix_csv.open("r", encoding="utf-8", newline="") as handle:
		reader = csv.DictReader(handle)
		if not reader.fieldnames or MATRIX_FILE_COLUMN not in set(reader.fieldnames):
			raise ValueError(f"Matrix CSV must include '{MATRIX_FILE_COLUMN}' column.")
		rows = list(reader)
		total_rows = len(rows)

		for row_index, row in enumerate(rows, start=1):
			file_value = str(row.get(MATRIX_FILE_COLUMN, "")).strip()
			if progress_callback is not None:
				progress_callback(row_index, total_rows, file_value)
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
				if collect_finals_by_params:
					signature = _row_override_signature(row_overrides)
					if signature not in param_set_dirs:
						set_index = len(param_set_dirs) + 1
						set_label = _signature_label(signature)
						set_dir = finals_root / f"set_{set_index:04d}_{set_label}"
						set_dir.mkdir(parents=True, exist_ok=True)
						param_set_dirs[signature] = set_dir
					set_dir = param_set_dirs[signature]
					dest_path = set_dir / file_path.name
					if dest_path.exists():
						dest_path = set_dir / f"row_{row_index:04d}_{file_path.name}"
					shutil.copy2(final_path, dest_path)
					sidecar = final_path.with_suffix(f"{final_path.suffix}.json")
					if sidecar.exists():
						sidecar_dest = dest_path.with_suffix(f"{dest_path.suffix}.json")
						shutil.copy2(sidecar, sidecar_dest)

	return successes, errors, base_run_dir
