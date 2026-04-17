"""Pipeline tab for the Triton GUI."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from triton.core.project import Pipeline, Project, load_file_labels, log_project_event, list_project_files
from triton.core.pipeline_runtime import PIPELINE_ACTIONS, PIPELINE_DEFAULT_STEP, PIPELINE_STEP_ORDER
from triton.core.ramp import RAMP_SHAPES
from triton.gui.shared import (
	_apply_pipeline_step,
	_default_step_options,
	_load_pipelines,
	_new_pipeline_run_id,
	_pipeline_action_label,
	_pipeline_key,
	_pipeline_run_dir,
	_run_pipeline_on_file,
	_save_pipelines,
)


def _close_pipeline_editor() -> None:
	st.session_state.pop("pipeline_editor_mode", None)
	st.session_state.pop("pipeline_editor_original_name", None)


def _delete_pipeline_step(step_index: int) -> None:
	step_count = int(st.session_state.get("pipeline_editor_step_count", 1))
	if step_count <= 1 or step_index < 0 or step_index >= step_count:
		return

	key_prefixes = [
		"pipeline_editor_step_",
		"pipeline_step_target_peak_",
		"pipeline_step_target_rms_",
		"pipeline_step_target_mode_",
		"pipeline_step_custom_sr_",
		"pipeline_step_bit_depth_",
		"pipeline_step_n_bands_",
		"pipeline_step_vocoder_type_",
		"pipeline_step_envelope_cutoff_",
		"pipeline_step_noise_type_",
		"pipeline_step_snr_db_",
		"pipeline_step_noise_project_file_",
		"pipeline_step_noise_file_",
		"pipeline_step_noise_seed_",
		"pipeline_step_compress_factor_",
		"pipeline_step_ramp_start_",
		"pipeline_step_ramp_end_",
		"pipeline_step_ramp_shape_",
	]

	for index in range(step_index, step_count - 1):
		for prefix in key_prefixes:
			next_key = f"{prefix}{index + 1}"
			current_key = f"{prefix}{index}"
			if next_key in st.session_state:
				st.session_state[current_key] = st.session_state[next_key]
			else:
				st.session_state.pop(current_key, None)

	for prefix in key_prefixes:
		st.session_state.pop(f"{prefix}{step_count - 1}", None)

	st.session_state["pipeline_editor_step_count"] = step_count - 1


def _request_delete_pipeline_step(step_index: int) -> None:
	st.session_state["pipeline_editor_delete_request"] = int(step_index)


def _open_pipeline_editor(mode: str, project: Project, pipeline: Pipeline | None = None) -> None:
	steps = list(pipeline.steps) if pipeline and pipeline.steps else [PIPELINE_DEFAULT_STEP]
	step_options = pipeline.step_options if pipeline else {}

	st.session_state["pipeline_editor_mode"] = mode
	st.session_state["pipeline_editor_original_name"] = pipeline.name if pipeline else ""
	st.session_state["pipeline_editor_name"] = pipeline.name if pipeline else ""
	st.session_state["pipeline_editor_step_count"] = len(steps)

	for index, action in enumerate(steps):
		action_key = action if action in PIPELINE_ACTIONS else PIPELINE_DEFAULT_STEP
		st.session_state[f"pipeline_editor_step_{index}"] = action_key

		defaults = _default_step_options(action_key, int(project.sample_rate))
		stored_by_index = step_options.get(str(index), {}) if isinstance(step_options.get(str(index), {}), dict) else {}
		stored_by_action = step_options.get(action_key, {}) if isinstance(step_options.get(action_key, {}), dict) else {}
		stored = stored_by_index or stored_by_action
		options = {**defaults, **stored}

		st.session_state[f"pipeline_step_target_peak_{index}"] = float(options.get("target_peak", 0.99))
		st.session_state[f"pipeline_step_target_rms_{index}"] = float(options.get("target_rms", 0.1))
		st.session_state[f"pipeline_step_target_mode_{index}"] = str(options.get("target_mode", "project"))
		st.session_state[f"pipeline_step_custom_sr_{index}"] = int(options.get("custom_sr", int(project.sample_rate)))
		st.session_state[f"pipeline_step_bit_depth_{index}"] = int(options.get("bit_depth", 16))
		st.session_state[f"pipeline_step_n_bands_{index}"] = int(options.get("n_bands", 8))
		st.session_state[f"pipeline_step_vocoder_type_{index}"] = str(options.get("vocoder_type", "noise"))
		st.session_state[f"pipeline_step_envelope_cutoff_{index}"] = float(options.get("envelope_cutoff", 160.0))
		st.session_state[f"pipeline_step_noise_type_{index}"] = str(options.get("noise_type", "auto"))
		st.session_state[f"pipeline_step_snr_db_{index}"] = float(options.get("snr_db", 0.0))
		st.session_state[f"pipeline_step_noise_project_file_{index}"] = str(options.get("noise_project_file", ""))
		st.session_state[f"pipeline_step_noise_file_{index}"] = str(options.get("noise_file", ""))
		seed_value = options.get("seed")
		st.session_state[f"pipeline_step_noise_seed_{index}"] = int(seed_value) if seed_value is not None and str(seed_value).strip() != "" else -1
		st.session_state[f"pipeline_step_compress_factor_{index}"] = float(options.get("factor", 1.0))
		st.session_state[f"pipeline_step_ramp_start_{index}"] = float(options.get("ramp_start", 0.05))
		st.session_state[f"pipeline_step_ramp_end_{index}"] = float(options.get("ramp_end", 0.05))
		st.session_state[f"pipeline_step_ramp_shape_{index}"] = str(options.get("shape", "cosine"))


def _close_pipeline_editor() -> None:
	st.session_state.pop("pipeline_editor_mode", None)
	st.session_state.pop("pipeline_editor_original_name", None)


def _delete_pipeline_step(step_index: int) -> None:
	step_count = int(st.session_state.get("pipeline_editor_step_count", 1))
	if step_count <= 1 or step_index < 0 or step_index >= step_count:
		return

	key_prefixes = [
		"pipeline_editor_step_",
		"pipeline_step_target_peak_",
		"pipeline_step_target_rms_",
		"pipeline_step_target_mode_",
		"pipeline_step_custom_sr_",
		"pipeline_step_bit_depth_",
		"pipeline_step_n_bands_",
		"pipeline_step_vocoder_type_",
		"pipeline_step_envelope_cutoff_",
		"pipeline_step_noise_type_",
		"pipeline_step_snr_db_",
		"pipeline_step_noise_project_file_",
		"pipeline_step_noise_file_",
		"pipeline_step_noise_seed_",
		"pipeline_step_compress_factor_",
		"pipeline_step_ramp_start_",
		"pipeline_step_ramp_end_",
		"pipeline_step_ramp_shape_",
	]

	for index in range(step_index, step_count - 1):
		for prefix in key_prefixes:
			next_key = f"{prefix}{index + 1}"
			current_key = f"{prefix}{index}"
			if next_key in st.session_state:
				st.session_state[current_key] = st.session_state[next_key]
			else:
				st.session_state.pop(current_key, None)

	for prefix in key_prefixes:
		st.session_state.pop(f"{prefix}{step_count - 1}", None)

	st.session_state["pipeline_editor_step_count"] = step_count - 1


def _request_delete_pipeline_step(step_index: int) -> None:
	st.session_state["pipeline_editor_delete_request"] = int(step_index)


def _render_step_options_editor(step: str, index: int, project: Project, project_files: list[Path]) -> dict[str, object]:
	if step == "normalize":
		target_peak = st.slider("Target peak", min_value=0.10, max_value=1.0, value=float(st.session_state.get(f"pipeline_step_target_peak_{index}", 0.99)), step=0.01, key=f"pipeline_step_target_peak_{index}")
		return {"target_peak": float(target_peak)}

	if step == "normalize_rms":
		target_rms = st.slider("Target RMS amplitude", min_value=0.01, max_value=1.0, value=float(st.session_state.get(f"pipeline_step_target_rms_{index}", 0.1)), step=0.01, key=f"pipeline_step_target_rms_{index}")
		return {"target_rms": float(target_rms)}

	if step == "resample_project":
		target_mode = st.selectbox("Target sample rate", options=["project", "custom"], format_func=lambda value: "Project sample rate" if value == "project" else "Custom sample rate", key=f"pipeline_step_target_mode_{index}")
		options: dict[str, object] = {"target_mode": target_mode}
		if target_mode == "custom":
			custom_sr = st.number_input("Custom sample rate (Hz)", min_value=8000, max_value=192000, value=int(st.session_state.get(f"pipeline_step_custom_sr_{index}", int(project.sample_rate))), step=1000, key=f"pipeline_step_custom_sr_{index}")
			options["custom_sr"] = int(custom_sr)
		else:
			options["custom_sr"] = int(project.sample_rate)
		return options

	if step == "requantize_16":
		bit_depth = st.selectbox("Bit depth", options=[8, 16, 24, 32], key=f"pipeline_step_bit_depth_{index}")
		return {"bit_depth": int(bit_depth)}

	if step == "vocode_noise":
		n_bands = st.slider("Number of bands", min_value=2, max_value=24, value=int(st.session_state.get(f"pipeline_step_n_bands_{index}", 8)), step=1, key=f"pipeline_step_n_bands_{index}")
		vocoder_type = st.selectbox("Carrier", options=["noise", "sine"], key=f"pipeline_step_vocoder_type_{index}")
		envelope_cutoff = st.slider("Envelope cutoff (Hz)", min_value=20.0, max_value=400.0, value=float(st.session_state.get(f"pipeline_step_envelope_cutoff_{index}", 160.0)), step=5.0, key=f"pipeline_step_envelope_cutoff_{index}")
		return {"n_bands": int(n_bands), "vocoder_type": str(vocoder_type), "envelope_cutoff": float(envelope_cutoff)}

	if step == "add_noise":
		project_noise_options = [""] + [str(path.relative_to(project.path)) for path in project_files]
		stored_project_noise = str(st.session_state.get(f"pipeline_step_noise_project_file_{index}", ""))
		if stored_project_noise not in project_noise_options:
			project_noise_options.append(stored_project_noise)
		noise_type = st.selectbox(
			"Noise type",
			options=["auto", "babble", "white", "colored", "ssn"],
			key=f"pipeline_step_noise_type_{index}",
		)
		snr_db = st.slider(
			"Target SNR (dB)",
			min_value=-40.0,
			max_value=40.0,
			value=float(st.session_state.get(f"pipeline_step_snr_db_{index}", 0.0)),
			step=0.5,
			key=f"pipeline_step_snr_db_{index}",
		)
		noise_project_file = st.selectbox(
			"Noise source (project file)",
			options=project_noise_options,
			key=f"pipeline_step_noise_project_file_{index}",
			format_func=lambda value: "(none)" if value == "" else value,
			help="Select an existing project audio file as the noise source.",
		)
		seed_value = st.number_input(
			"Random seed (-1 for random)",
			min_value=-1,
			max_value=2_147_483_647,
			value=int(st.session_state.get(f"pipeline_step_noise_seed_{index}", -1)),
			step=1,
			key=f"pipeline_step_noise_seed_{index}",
		)
		options: dict[str, object] = {
			"noise_type": str(noise_type),
			"snr_db": float(snr_db),
			"noise_project_file": str(noise_project_file).strip(),
		}
		if int(seed_value) >= 0:
			options["seed"] = int(seed_value)
		return options

	if step == "time_compress":
		factor = st.slider("Compression factor", min_value=0.1, max_value=10.0, value=float(st.session_state.get(f"pipeline_step_compress_factor_{index}", 1.0)), step=0.1, key=f"pipeline_step_compress_factor_{index}", help="< 1.0 = faster (compression), > 1.0 = slower (expansion)")
		return {"factor": float(factor)}

	if step == "ramp":
		shape = st.selectbox("Ramp shape", options=list(RAMP_SHAPES), key=f"pipeline_step_ramp_shape_{index}", help="linear: uniform sweep · cosine: smooth S-curve · exponential: slow start fast finish · logarithmic: fast start slow finish")
		ramp_start = st.slider("Fade-in duration (s)", min_value=0.0, max_value=5.0, value=float(st.session_state.get(f"pipeline_step_ramp_start_{index}", 0.05)), step=0.01, key=f"pipeline_step_ramp_start_{index}")
		ramp_end = st.slider("Fade-out duration (s)", min_value=0.0, max_value=5.0, value=float(st.session_state.get(f"pipeline_step_ramp_end_{index}", 0.05)), step=0.01, key=f"pipeline_step_ramp_end_{index}")
		return {"ramp_start": float(ramp_start), "ramp_end": float(ramp_end), "shape": str(shape)}

	st.caption("No options for this step.")
	return {}


def _render_pipelines_tab(project: Project, project_files: list[Path], mode: str = "all") -> None:
	show_designer = mode in {"all", "design"}
	show_runner = mode in {"all", "run"}
	if mode not in {"all", "design", "run"}:
		raise ValueError(f"Unsupported pipelines tab mode: {mode}")

	project_key = _pipeline_key(str(project.path))
	selected_pipeline_state_key = f"selected_pipeline_name_{project_key}_{mode}"
	selection_mode_key = f"pipeline_selection_mode_{project_key}_{mode}"

	pipelines = _load_pipelines(project)
	pipeline_names = [item.name for item in pipelines]

	pending_selected_pipeline = st.session_state.pop("pending_selected_pipeline_name", None)
	if pending_selected_pipeline is not None:
		if pending_selected_pipeline == "__clear__":
			st.session_state.pop(selected_pipeline_state_key, None)
		elif pending_selected_pipeline in set(pipeline_names):
			st.session_state[selected_pipeline_state_key] = pending_selected_pipeline

	if pipeline_names and st.session_state.get(selected_pipeline_state_key) not in set(pipeline_names):
		st.session_state[selected_pipeline_state_key] = pipeline_names[0]

	if show_designer and show_runner:
		st.markdown("### Pipelines")
		st.write("Design, run, and maintain project pipelines.")
	elif show_designer:
		st.markdown("### Pipeline Designer")
		st.write("Create and edit pipeline definitions and step options.")
	else:
		st.markdown("### Pipeline Runner")
		st.write("Choose an existing pipeline and run it on selected project files.")

	if show_designer:
		main_col, editor_col = st.columns([1.9, 1.1], gap="large")
	else:
		main_col = st.container()
		editor_col = None

	with main_col:
		if show_designer:
			action_col1, action_col2 = st.columns(2)
			if action_col1.button("New pipeline", type="primary"):
				_open_pipeline_editor("create", project)
				st.rerun()

		selected_pipeline: Pipeline | None = None
		if pipeline_names:
			selected_name = st.radio("Created pipelines", options=pipeline_names, key=selected_pipeline_state_key)
			selected_pipeline = next((item for item in pipelines if item.name == selected_name), None)

			if show_designer and action_col2.button("Edit selected"):
				if selected_pipeline is not None:
					_open_pipeline_editor("edit", project, selected_pipeline)
					st.rerun()
		else:
			if show_designer:
				action_col2.button("Edit selected", disabled=True)
			st.info("No pipelines yet. Click New pipeline to create your first one.")

		if selected_pipeline is not None:
			st.markdown(f"#### {selected_pipeline.name}")
			for index, step in enumerate(selected_pipeline.steps, start=1):
				st.caption(f"{index}. {_pipeline_action_label(step)} ({step})")

			if show_runner:
				# File selection mode
				selection_mode = st.radio(
					"Select files by:",
					options=["Files", "Label(s)"],
					horizontal=True,
					key=selection_mode_key
				)

				selected_paths: list[Path] = []

				if selection_mode == "Files":
					run_files = st.multiselect(
						"Files to run",
						options=[path.name for path in project_files],
						help="Choose one or more project files for this pipeline."
					)
					if run_files:
						selected_paths = [path for path in project_files if path.name in set(run_files)]

				else:  # selection_mode == "Label(s)"
					# Load available labels
					all_labels = load_file_labels(project.path)
					available_labels = sorted(set(lbl for lbls in all_labels.values() for lbl in lbls))

					if available_labels:
						selected_labels = st.multiselect(
							"Labels to run",
							options=available_labels,
							help="Choose one or more labels to run the pipeline on all files with those labels."
						)
						# Collect all files that have any of the selected labels
						if selected_labels:
							for label in selected_labels:
								label_files = list_project_files(project.path, label=label)
								selected_paths.extend(label_files)
							# Remove duplicates while preserving order
							seen = set()
							selected_paths = [p for p in selected_paths if not (p in seen or seen.add(p))]
					else:
						st.info("No labels found. Use the File Library tab to add labels to your files.")

				run_col, delete_col = st.columns(2)
				if run_col.button("Run selected pipeline", type="primary"):
					if not selected_paths:
						st.error("Select at least one file or label.")
					else:
						successes: list[Path] = []
						errors: list[str] = []
						run_id = _new_pipeline_run_id()
						run_dir = _pipeline_run_dir(project, selected_pipeline.name, run_id)

						with st.spinner(f"Running {selected_pipeline.name} on {len(selected_paths)} file(s)..."):
							for file_path in selected_paths:
								try:
									output_path = _run_pipeline_on_file(file_path, project, selected_pipeline, run_dir)
								except Exception as exc:
									errors.append(f"{file_path.name}: {exc}")
								else:
									successes.append(output_path)

						if successes:
							st.success(f"Processed {len(successes)} file(s).")
							st.caption(f"Run output folder: {run_dir}")
							for output in successes:
								st.caption(str(output))
							log_project_event(project.path, "pipeline_run_completed", {"pipeline": selected_pipeline.name, "run_id": run_id, "requested_files": len(selected_paths), "succeeded": len(successes), "failed": len(errors)})
						if errors:
							for error in errors:
								st.error(error)

				if show_designer and delete_col.button("Delete selected"):
					remaining = [item for item in pipelines if item.name != selected_pipeline.name]
					_save_pipelines(project, remaining)
					if remaining:
						st.session_state["pending_selected_pipeline_name"] = remaining[0].name
					else:
						st.session_state["pending_selected_pipeline_name"] = "__clear__"
					_close_pipeline_editor()
					st.rerun()

	if show_designer and editor_col is not None:
		with editor_col:
			st.markdown("### Editor")
			editor_mode = st.session_state.get("pipeline_editor_mode")
			if editor_mode not in {"create", "edit"}:
				st.caption("Choose New pipeline or Edit selected to open the right-side editor.")
			else:
				with st.container(border=True):
					delete_request = st.session_state.pop("pipeline_editor_delete_request", None)
					if isinstance(delete_request, int):
						_delete_pipeline_step(delete_request)
						st.rerun()

					head_col1, head_col2 = st.columns([3, 2])
					with head_col1:
						pipeline_name = st.text_input("Pipeline name", key="pipeline_editor_name", placeholder="speech_cleanup")
					with head_col2:
						step_count = int(st.number_input("Number of steps", min_value=1, max_value=12, value=int(st.session_state.get("pipeline_editor_step_count", 1)), step=1, key="pipeline_editor_step_count"))

					st.caption("Choose an action for each step, then configure options under it.")

					steps: list[str] = []
					step_options_by_index: dict[str, dict[str, object]] = {}

					for order_index in range(step_count):
						with st.container(border=True):
							step_col, delete_col = st.columns([5, 1])
							with step_col:
								step = st.selectbox(f"Step {order_index + 1}", options=PIPELINE_STEP_ORDER, format_func=lambda action: f"{_pipeline_action_label(action)} ({action})", key=f"pipeline_editor_step_{order_index}")
							with delete_col:
								st.button("✕", key=f"pipeline_editor_delete_step_{order_index}", disabled=step_count <= 1, on_click=_request_delete_pipeline_step, args=(order_index,))
							steps.append(step)
							st.caption("Step options")
							options = _render_step_options_editor(step, order_index, project, project_files)
							if options:
								step_options_by_index[str(order_index)] = options

					save_col, cancel_col = st.columns(2)
					if save_col.button("Save pipeline", type="primary"):
						clean_name = pipeline_name.strip()
						if not clean_name:
							st.error("Pipeline name is required.")
						else:
							existing_names = {item.name for item in pipelines}
							original_name = str(st.session_state.get("pipeline_editor_original_name", ""))
							if editor_mode == "create" and clean_name in existing_names:
								st.error("A pipeline with this name already exists.")
							elif editor_mode == "edit" and clean_name != original_name and clean_name in existing_names:
								st.error("A pipeline with this name already exists.")
							else:
								updated = Pipeline(name=clean_name, steps=steps, step_options=step_options_by_index)
								if editor_mode == "create":
									pipelines.append(updated)
								else:
									pipelines = [updated if item.name == original_name else item for item in pipelines]

								_save_pipelines(project, pipelines)
								st.session_state["pending_selected_pipeline_name"] = clean_name
								_close_pipeline_editor()
								st.rerun()

					if cancel_col.button("Cancel"):
						_close_pipeline_editor()
						st.rerun()
