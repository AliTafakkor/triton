"""Triton Streamlit GUI."""

from __future__ import annotations
from datetime import date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from io import BytesIO
from pathlib import Path
from importlib import resources

from triton.gui.project_views import _hero, _render_project_launcher, _render_matrix_tab
from triton.gui.tabs.file_library import _render_file_library as _render_file_library_tab
from triton.gui.tabs.pipelines import _render_pipelines_tab as _render_pipelines_tab_module
from triton.gui.tabs.rss import _render_rss_ingest_tab as _render_rss_ingest_tab_module

import librosa
import numpy as np
import plotly.graph_objects as go
import soundfile as sf
import streamlit as st

from triton.core.mixer import mix_at_snr
from triton.core.pipeline_runtime import (
	PIPELINE_ACTIONS,
	PIPELINE_DEFAULT_STEP,
	PIPELINE_STEP_ORDER,
	apply_pipeline_step,
	default_step_options,
	new_pipeline_run_id,
	pipeline_action_label,
	pipeline_key,
	pipeline_output_dir,
	pipeline_run_dir,
	run_pipeline_on_file,
)
from triton.core.project import (
	ChannelMode,
	Pipeline,
	Project,
	add_project_file,
	create_project,
	delete_project_file,
	log_project_event,
	list_project_files,
	load_project_config,
	load_project_pipelines,
	read_project_log,
	load_project_spectrogram_settings,
	load_recent_projects,
	project_raw_dir,
	rename_project_file,
	register_recent_project,
	save_project_pipelines,
	save_project_generated_audio,
	update_project_spectrogram_settings,
	load_babble_talker_groups,
	load_file_labels,
	set_file_labels,
	set_project_file_labels,
)
from triton.core.io import load_audio, write_sidecar
from triton.core.spectrogram import compute_spectrogram, load_spectrogram, save_spectrogram
from triton.degrade.noise_generator import generate_project_babble
from triton.ingest.rss import RssSource


def _load_app_css() -> str:
	css_path = resources.files(__package__) / "assets" / "app.css"
	return f"<style>{css_path.read_text(encoding='utf-8')}</style>"


APP_CSS = _load_app_css()


def _pipeline_action_label(action: str) -> str:
	return pipeline_action_label(action)


def _default_step_options(step: str, project_sr: int) -> dict[str, object]:
	return default_step_options(step, project_sr)


def _set_active_project(project_dir: Path) -> Project:
	project = load_project_config(project_dir)
	st.session_state["active_project"] = project
	register_recent_project(project_dir, project.name)
	log_project_event(project_dir, "project_opened_gui", {"project_name": project.name})
	return project


def _create_project(project_dir: Path, sample_rate: int, channel_mode: ChannelMode) -> Project:
	project = create_project(project_dir, sample_rate=sample_rate, channel_mode=channel_mode)
	st.session_state["active_project"] = project
	register_recent_project(project_dir, project.name)
	return project


def _babble_generation_signature(
	project: Project,
	*,
	num_talkers: int,
	num_female_talkers: int | None,
	num_male_talkers: int | None,
	intended_length_seconds: float,
	target_rms: float,
	peak_normalize: bool,
) -> dict[str, object]:
	return {
		"project_path": str(project.path.resolve()),
		"num_talkers": int(num_talkers),
		"num_female_talkers": None if num_female_talkers is None else int(num_female_talkers),
		"num_male_talkers": None if num_male_talkers is None else int(num_male_talkers),
		"intended_length_seconds": float(intended_length_seconds),
		"target_rms": float(target_rms),
		"peak_normalize": bool(peak_normalize),
	}


def _render_saved_babble_output(project: Project, babble_state: dict[str, object]) -> None:
	babble_bytes = babble_state["babble_bytes"]
	selected_groups = babble_state["selected_groups"]
	planned_group_files = babble_state["planned_group_files"]
	short_source_labels = babble_state["short_source_labels"]
	unknown_duration_labels = babble_state["unknown_duration_labels"]
	repeat_counts_by_label = babble_state["repeat_counts_by_label"]
	intended_length_seconds = float(babble_state["intended_length_seconds"])
	target_rms = float(babble_state["target_rms"])
	peak_norm = bool(babble_state["peak_normalize"])
	num_talkers = int(babble_state["num_talkers"])
	num_female_talkers = babble_state["num_female_talkers"]
	num_male_talkers = babble_state["num_male_talkers"]
	sample_rate = int(babble_state["sample_rate"])
	babble_label = f"bab-t{num_talkers}"

	if short_source_labels:
		st.warning(
			"Some talkers do not have enough unique source duration for the intended length: "
			+ ", ".join(short_source_labels)
			+ ". Their files were repeated randomly."
		)
	if unknown_duration_labels:
		st.warning(
			"Could not estimate duration for some files while planning: "
			+ ", ".join(unknown_duration_labels)
			+ ". Full selected files were loaded for those talkers."
		)
	if repeat_counts_by_label:
		st.info(
			"Random repeats applied for short talkers: "
			+ ", ".join(f"{label} (+{count})" for label, count in sorted(repeat_counts_by_label.items()))
		)

	st.success(
		f"Babble generated from {len(selected_groups)} talker groups "
		f"({sum(1 for group in selected_groups if group['sex'] == 'f')} female, "
		f"{sum(1 for group in selected_groups if group['sex'] == 'm')} male)."
	)

	st.markdown("#### Selected Talkers")
	for group, files in zip(selected_groups, planned_group_files, strict=False):
		st.caption(f"{group['label']}: {', '.join(Path(file_path).name for file_path in files)}")

	st.markdown("### Babble Output")
	st.audio(babble_bytes, format="audio/wav")
	download_col, add_col = st.columns(2)
	with download_col:
		st.download_button(
			label="Download babble",
			data=babble_bytes,
			file_name="triton_babble.wav",
			mime="audio/wav",
		)
	with add_col:
		if st.button("Add to project", type="secondary", key="babble_add_to_project"):
			filename = f"babble_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.wav"
			generation_options = {
				"num_talkers": num_talkers,
				"female_talkers": num_female_talkers,
				"male_talkers": num_male_talkers,
				"intended_length_seconds": intended_length_seconds,
				"target_rms": target_rms,
				"peak_normalize": peak_norm,
			}
			saved_path = save_project_generated_audio(
				project.path,
				filename,
				babble_state["audio"],
				sample_rate,
				label=babble_label,
				source={
					"type": "project_babble",
					"project_name": project.name,
					"project_path": str(project.path.resolve()),
					"talkers": [
						{
							"label": group["label"],
							"sex": group["sex"],
							"index": int(group["index"]),
							"files": [str(Path(file_path).resolve()) for file_path in files],
						}
						for group, files in zip(selected_groups, planned_group_files, strict=False)
					],
				},
				actions=[
					{
						"step": "generate_project_babble",
						"options": generation_options,
					},
					{
						"step": "add_project_generated_audio",
						"options": {
							"filename": filename,
							"label": babble_label,
						},
					},
				],
				extra={
					"babble": {
						"num_talkers": num_talkers,
						"female_talkers": num_female_talkers,
						"male_talkers": num_male_talkers,
						"intended_length_seconds": intended_length_seconds,
						"target_rms": target_rms,
						"peak_normalize": peak_norm,
						"selected_groups": selected_groups,
						"planned_group_files": [[str(Path(file_path).resolve()) for file_path in files] for files in planned_group_files],
						"short_source_labels": short_source_labels,
						"unknown_duration_labels": unknown_duration_labels,
						"repeat_counts_by_label": repeat_counts_by_label,
					},
				},
			)
			st.success(f"Added babble to project as {saved_path.name} with label {babble_label}.")


def _collect_spectrogram_settings(prefix: str, defaults: dict[str, object] | None = None) -> dict[str, object]:
	base = defaults or {}
	type_key = f"{prefix}_spec_type"
	n_fft_key = f"{prefix}_spec_n_fft"
	hop_key = f"{prefix}_spec_hop_length"
	win_key = f"{prefix}_spec_win_length"
	window_key = f"{prefix}_spec_window"
	n_mels_key = f"{prefix}_spec_n_mels"
	fmin_key = f"{prefix}_spec_fmin"
	fmax_key = f"{prefix}_spec_fmax"
	power_key = f"{prefix}_spec_power"

	default_type = str(base.get("type", "stft"))
	if default_type not in {"stft", "mel", "cqt"}:
		default_type = "stft"

	spec_type = st.selectbox(
		"Spectrogram type",
		options=["stft", "mel", "cqt"],
		index=["stft", "mel", "cqt"].index(default_type),
		key=type_key,
	)
	col1, col2, col3 = st.columns(3)
	n_fft = col1.number_input("n_fft", min_value=128, max_value=8192, value=int(base.get("n_fft", 1024)), step=128, key=n_fft_key)
	hop_length = col2.number_input("hop_length", min_value=32, max_value=4096, value=int(base.get("hop_length", 256)), step=32, key=hop_key)
	win_length = col3.number_input("win_length", min_value=128, max_value=8192, value=int(base.get("win_length", 1024)), step=128, key=win_key)

	default_window = str(base.get("window", "hann"))
	window_options = ["hann", "hamming", "blackman"]
	if default_window not in window_options:
		default_window = "hann"
	window = st.selectbox("Window", options=window_options, index=window_options.index(default_window), key=window_key)

	n_mels = int(base.get("n_mels", 128))
	fmin = float(base.get("fmin", 32.7))
	fmax = float(base.get("fmax", 8000.0))
	power = float(base.get("power", 2.0))

	if spec_type == "mel":
		m_col1, m_col2, m_col3 = st.columns(3)
		n_mels = int(m_col1.number_input("n_mels", min_value=16, max_value=512, value=n_mels, step=8, key=n_mels_key))
		fmin = float(m_col2.number_input("fmin (Hz)", min_value=0.0, max_value=20000.0, value=fmin, step=1.0, key=fmin_key))
		fmax = float(m_col3.number_input("fmax (Hz)", min_value=100.0, max_value=48000.0, value=fmax, step=10.0, key=fmax_key))
		power = float(st.number_input("power", min_value=1.0, max_value=4.0, value=power, step=0.1, key=power_key))
	elif spec_type == "cqt":
		c_col1, c_col2 = st.columns(2)
		fmin = float(c_col1.number_input("fmin (Hz)", min_value=1.0, max_value=20000.0, value=max(fmin, 1.0), step=1.0, key=fmin_key))
		power = float(c_col2.number_input("power", min_value=1.0, max_value=4.0, value=power, step=0.1, key=power_key))

	return {
		"type": str(spec_type),
		"n_fft": int(n_fft),
		"hop_length": int(hop_length),
		"win_length": int(win_length),
		"window": str(window),
		"n_mels": int(n_mels),
		"fmin": float(fmin),
		"fmax": float(fmax),
		"power": float(power),
	}


def _clear_active_project() -> None:
	st.session_state.pop("active_project", None)


def _format_file_size(size_bytes: int) -> str:
	units = ["B", "KB", "MB", "GB"]
	size = float(size_bytes)
	for unit in units:
		if size < 1024 or unit == units[-1]:
			return f"{size:.1f} {unit}"
		size /= 1024
	return f"{size_bytes} B"


def _spectrogram_path(audio_path: Path) -> Path:
	return audio_path.with_suffix(audio_path.suffix + ".spectrogram.npz")


def _generate_file_spectrogram(audio_path: Path, project: Project) -> Path:
	settings = load_project_spectrogram_settings(project.path)
	audio, sr = load_audio(audio_path, sr=None, mono=False)
	result = compute_spectrogram(audio, sr, settings)
	out_path = _spectrogram_path(audio_path)
	save_spectrogram(out_path, result, settings)
	write_sidecar(
		out_path,
		source={"path": str(audio_path.resolve())},
		actions=[{"step": "spectrogram", "options": settings}],
		extra={
			"spectrogram": {
				"type": result.kind,
				"shape": [int(result.values.shape[0]), int(result.values.shape[1])],
			}
		},
	)
	return out_path


def _save_uploaded_project_files(
	project: Project,
	uploaded_files: list[object],
	batch_label: str = "",
) -> list[Path]:
	saved_paths: list[Path] = []
	with st.status("Importing files...", expanded=True) as status:
		progress_bar = st.progress(0.0)
		for idx, uploaded_file in enumerate(uploaded_files):
			status.write(f"Importing: {uploaded_file.name}")
			path = add_project_file(project.path, uploaded_file.name, uploaded_file.getvalue())
			saved_paths.append(path)
			_generate_file_spectrogram(path, project)
			progress = (idx + 1) / len(uploaded_files)
			status.update(label=f"Importing files... ({idx + 1}/{len(uploaded_files)})", state="running")
			progress_bar.progress(progress)
		status.update(label=f"Imported {len(saved_paths)} file(s)", state="complete")
	
	if batch_label.strip() and saved_paths:
		set_project_file_labels(project.path, saved_paths, batch_label)
		log_project_event(
			project.path,
			"files_labeled_batch",
			{
				"count": len(saved_paths),
				"label": batch_label.strip(),
				"files": [path.name for path in saved_paths],
			},
		)

	if saved_paths:
		log_project_event(
			project.path,
			"files_imported_batch",
			{"count": len(saved_paths), "files": [path.name for path in saved_paths]},
		)

	return saved_paths


def _regenerate_all_project_spectrograms(project: Project, project_files: list[Path]) -> tuple[int, list[str]]:
	updated = 0
	errors: list[str] = []
	with st.status("Regenerating spectrograms...", expanded=True) as status:
		progress_bar = st.progress(0.0)
		for idx, file_path in enumerate(project_files):
			status.write(f"Processing: {file_path.name}")
			try:
				_generate_file_spectrogram(file_path, project)
			except Exception as exc:
				errors.append(f"{file_path.name}: {exc}")
			else:
				updated += 1
			progress = (idx + 1) / len(project_files)
			status.update(label=f"Regenerating spectrograms... ({idx + 1}/{len(project_files)})", state="running")
			progress_bar.progress(progress)
		status.update(label=f"Regenerated {updated} file(s)", state="complete")

	log_project_event(
		project.path,
		"spectrogram_regenerated_all",
		{"updated_files": updated, "failed_files": len(errors)},
	)
	return updated, errors


def _delete_project_file(file_path: Path) -> None:
	delete_project_file(file_path)


def _rename_project_file(file_path: Path, new_name: str) -> Path:
	return rename_project_file(file_path, new_name)


def _read_uploaded_audio(uploaded_file) -> tuple[np.ndarray, int]:
	audio, sr = sf.read(BytesIO(uploaded_file.getvalue()), dtype="float32", always_2d=True)
	return audio, sr


def _resample_audio(audio: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
	if source_sr == target_sr:
		return audio

	channels = [
		librosa.resample(audio[:, channel], orig_sr=source_sr, target_sr=target_sr)
		for channel in range(audio.shape[1])
	]
	return np.stack(channels, axis=1).astype(np.float32)


def _convert_channels(audio: np.ndarray, channel_mode: ChannelMode) -> np.ndarray:
	if channel_mode == "mono":
		return np.mean(audio, axis=1, dtype=np.float32)

	if audio.shape[1] == 1:
		return np.repeat(audio, 2, axis=1).astype(np.float32)

	if audio.shape[1] >= 2:
		return audio[:, :2].astype(np.float32)

	raise ValueError("Audio must contain at least one channel.")


def _load_uploaded_audio(
	uploaded_file,
	target_sr: int,
	channel_mode: str,
) -> tuple[np.ndarray, int, dict[str, int | str]]:
	raw_audio, source_sr = _read_uploaded_audio(uploaded_file)
	resampled_audio = _resample_audio(raw_audio, source_sr=source_sr, target_sr=target_sr)
	converted_audio = _convert_channels(resampled_audio, channel_mode=channel_mode)

	metadata = {
		"filename": uploaded_file.name,
		"source_sr": source_sr,
		"target_sr": target_sr,
		"source_channels": raw_audio.shape[1],
		"target_channels": 1 if channel_mode == "mono" else 2,
	}
	return converted_audio, target_sr, metadata


def _audio_bytes(audio: np.ndarray, sr: int) -> bytes:
	buffer = BytesIO()
	frames = audio if audio.ndim == 1 else np.asarray(audio, dtype=np.float32)
	sf.write(buffer, frames, sr, format="WAV")
	buffer.seek(0)
	return buffer.read()


def _display_audio_summary(title: str, metadata: dict[str, int | str]) -> None:
	st.markdown(f"#### {title}")
	st.caption(
		f"{metadata['filename']} | {metadata['source_sr']} Hz -> {metadata['target_sr']} Hz | "
		f"{metadata['source_channels']} ch -> {metadata['target_channels']} ch"
	)


def _render_styles() -> None:
	st.markdown(APP_CSS, unsafe_allow_html=True)

def _render_file_library(project: Project, project_files: list[Path]) -> None:
	_render_file_library_tab(project, project_files)


def _parse_episode_published_date(published: str | None) -> date | None:
	if not published:
		return None

	text = published.strip()
	if not text:
		return None

	try:
		if text.endswith("Z"):
			parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
		else:
			parsed = datetime.fromisoformat(text)
		return parsed.date()
	except ValueError:
		pass

	try:
		return parsedate_to_datetime(text).date()
	except (TypeError, ValueError, IndexError):
		return None


def _render_rss_ingest_tab(project: Project) -> None:
	_render_rss_ingest_tab_module(project)


def _pipeline_key(name: str) -> str:
	return pipeline_key(name)


def _pipeline_output_dir(project: Project, pipeline_name: str) -> Path:
	return pipeline_output_dir(project, pipeline_name)


def _pipeline_run_dir(project: Project, pipeline_name: str, run_id: str) -> Path:
	return pipeline_run_dir(project, pipeline_name, run_id)


def _new_pipeline_run_id() -> str:
	return new_pipeline_run_id()


def _apply_pipeline_step(
	audio: np.ndarray,
	sr: int,
	step: str,
	project: Project,
	step_options: dict[str, object] | None = None,
) -> tuple[np.ndarray, int]:
	return apply_pipeline_step(audio, sr, step, project, step_options)


def _run_pipeline_on_file(file_path: Path, project: Project, pipeline: Pipeline, run_dir: Path) -> Path:
	return run_pipeline_on_file(file_path, project, pipeline, run_dir)


def _save_pipelines(project: Project, pipelines: list[Pipeline]) -> None:
	save_project_pipelines(project.path, pipelines)


def _load_pipelines(project: Project) -> list[Pipeline]:
	return load_project_pipelines(project.path)


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
		st.session_state[f"pipeline_step_target_mode_{index}"] = str(options.get("target_mode", "project"))
		st.session_state[f"pipeline_step_custom_sr_{index}"] = int(options.get("custom_sr", int(project.sample_rate)))
		st.session_state[f"pipeline_step_bit_depth_{index}"] = int(options.get("bit_depth", 16))
		st.session_state[f"pipeline_step_n_bands_{index}"] = int(options.get("n_bands", 8))
		st.session_state[f"pipeline_step_vocoder_type_{index}"] = str(options.get("vocoder_type", "noise"))
		st.session_state[f"pipeline_step_envelope_cutoff_{index}"] = float(options.get("envelope_cutoff", 160.0))
		st.session_state[f"pipeline_step_compress_factor_{index}"] = float(options.get("factor", 1.0))


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
		"pipeline_step_target_mode_",
		"pipeline_step_custom_sr_",
		"pipeline_step_bit_depth_",
		"pipeline_step_n_bands_",
		"pipeline_step_vocoder_type_",
		"pipeline_step_envelope_cutoff_",
		"pipeline_step_compress_factor_",
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


def _render_step_options_editor(step: str, index: int, project: Project) -> dict[str, object]:
	if step == "normalize":
		target_peak = st.slider(
			"Target peak",
			min_value=0.10,
			max_value=1.0,
			value=float(st.session_state.get(f"pipeline_step_target_peak_{index}", 0.99)),
			step=0.01,
			key=f"pipeline_step_target_peak_{index}",
		)
		return {"target_peak": float(target_peak)}

	if step == "normalize_rms":
		target_rms = st.slider(
			"Target RMS amplitude",
			min_value=0.01,
			max_value=1.0,
			value=float(st.session_state.get(f"pipeline_step_target_rms_{index}", 0.1)),
			step=0.01,
			key=f"pipeline_step_target_rms_{index}",
		)
		return {"target_rms": float(target_rms)}

	if step == "resample_project":
		target_mode = st.selectbox(
			"Target sample rate",
			options=["project", "custom"],
			format_func=lambda value: "Project sample rate" if value == "project" else "Custom sample rate",
			key=f"pipeline_step_target_mode_{index}",
		)
		options: dict[str, object] = {"target_mode": target_mode}
		if target_mode == "custom":
			custom_sr = st.number_input(
				"Custom sample rate (Hz)",
				min_value=8000,
				max_value=192000,
				value=int(st.session_state.get(f"pipeline_step_custom_sr_{index}", int(project.sample_rate))),
				step=1000,
				key=f"pipeline_step_custom_sr_{index}",
			)
			options["custom_sr"] = int(custom_sr)
		else:
			options["custom_sr"] = int(project.sample_rate)
		return options

	if step == "requantize_16":
		bit_depth = st.selectbox(
			"Bit depth",
			options=[8, 16, 24, 32],
			key=f"pipeline_step_bit_depth_{index}",
		)
		return {"bit_depth": int(bit_depth)}

	if step == "vocode_noise":
		n_bands = st.slider(
			"Number of bands",
			min_value=2,
			max_value=24,
			value=int(st.session_state.get(f"pipeline_step_n_bands_{index}", 8)),
			step=1,
			key=f"pipeline_step_n_bands_{index}",
		)
		vocoder_type = st.selectbox(
			"Carrier",
			options=["noise", "sine"],
			key=f"pipeline_step_vocoder_type_{index}",
		)
		envelope_cutoff = st.slider(
			"Envelope cutoff (Hz)",
			min_value=20.0,
			max_value=400.0,
			value=float(st.session_state.get(f"pipeline_step_envelope_cutoff_{index}", 160.0)),
			step=5.0,
			key=f"pipeline_step_envelope_cutoff_{index}",
		)
		return {
			"n_bands": int(n_bands),
			"vocoder_type": str(vocoder_type),
			"envelope_cutoff": float(envelope_cutoff),
		}

	if step == "time_compress":
		factor = st.slider(
			"Compression factor",
			min_value=0.1,
			max_value=10.0,
			value=float(st.session_state.get(f"pipeline_step_compress_factor_{index}", 1.0)),
			step=0.1,
			key=f"pipeline_step_compress_factor_{index}",
			help="< 1.0 = faster (compression), > 1.0 = slower (expansion)",
		)
		return {"factor": float(factor)}

	st.caption("No options for this step.")
	return {}


def _render_pipelines_tab(project: Project, project_files: list[Path]) -> None:
	_render_pipelines_tab_module(project, project_files)


def _render_project_workspace(project: Project) -> None:
	target_sr = int(project.sample_rate)
	channel_mode = str(project.channel_mode)
	project_dir = project.path.expanduser()
	project_files = list_project_files(project_dir)

	with st.sidebar:
		st.header("Active Project")
		st.caption(str(project.path))
		st.metric("Sample rate", f"{target_sr} Hz")
		st.metric("Channels", channel_mode)
		st.metric("Stored files", str(len(project_files)))

		current_spec = load_project_spectrogram_settings(project.path)
		sidebar_prefix = f"sidebar_{_pipeline_key(str(project.path))}"
		with st.expander("Spectrogram defaults", expanded=False):
			edited_spec = _collect_spectrogram_settings(sidebar_prefix, defaults=current_spec)
			if st.button("Update spectrogram defaults", key=f"update_spec_{sidebar_prefix}"):
				if edited_spec == current_spec:
					st.info("No changes detected.")
				else:
					st.session_state["pending_spectrogram_update"] = {
						"project": str(project.path.resolve()),
						"settings": edited_spec,
					}

		pending_update = st.session_state.get("pending_spectrogram_update")
		if isinstance(pending_update, dict) and pending_update.get("project") == str(project.path.resolve()):
			st.warning(
				"Spectrogram settings changed. All imported files must be re-generated with the new settings."
			)
			confirm_col, cancel_col = st.columns(2)
			if confirm_col.button("Accept + Update", key=f"confirm_spec_update_{sidebar_prefix}", type="primary"):
				with st.spinner("Updating spectrogram settings and regenerating files..."):
					new_settings = pending_update.get("settings", current_spec)
					update_project_spectrogram_settings(project.path, dict(new_settings))
					updated, errors = _regenerate_all_project_spectrograms(project, project_files)

				log_project_event(
					project.path,
					"spectrogram_update_accepted",
					{"updated_files": int(updated), "failed_files": len(errors)},
				)
				st.session_state.pop("pending_spectrogram_update", None)
				if errors:
					st.error(f"Updated {updated} file(s), {len(errors)} failed.")
					for item in errors:
						st.caption(item)
				else:
					st.success(f"Updated settings and regenerated spectrograms for {updated} file(s).")
				st.rerun()

			if cancel_col.button("Cancel", key=f"cancel_spec_update_{sidebar_prefix}"):
				log_project_event(project.path, "spectrogram_update_cancelled", {})
				st.session_state.pop("pending_spectrogram_update", None)
				st.rerun()

		with st.expander("Project activity log", expanded=False):
			events = read_project_log(project.path, limit=25)
			if not events:
				st.caption("No project activity recorded yet.")
			else:
				for event in reversed(events):
					timestamp = str(event.get("timestamp", ""))
					event_name = str(event.get("event", ""))
					st.caption(f"{timestamp} | {event_name}")

		if st.button("Close project"):
			_clear_active_project()
			st.rerun()

	import_tab, ingest_tab, pipelines_tab, mix_tab, babble_tab, transcribe_tab, roadmap_tab = st.tabs(["Manage and Explore Files", "Ingest RSS", "Pipelines", "Mix", "Babble", "Transcribe", "Roadmap"])

	with import_tab:
		metric_col1, metric_col2, metric_col3 = st.columns(3)
		metric_col1.metric("Project sample rate", f"{target_sr} Hz")
		metric_col2.metric("Channel policy", channel_mode)
		metric_col3.metric("Stored files", str(len(project_files)))

		st.markdown("### Import Workspace")
		st.write(
			"Import source files into the project, preview them, and inspect precomputed spectrograms built from project defaults."
		)
		st.write(
			"Imported files are stored in project raw storage and get spectrogram artifacts generated automatically."
		)

		# Label rename section
		with st.expander("📋 Manage Labels", expanded=False):
			st.write("Rename a label to apply the new name to all files that currently have that label.")

			all_labels_dict = load_file_labels(project.path)
			label_counts: dict[str, int] = {}
			for file_labels in all_labels_dict.values():
				for lbl in file_labels:
					label_counts[lbl] = label_counts.get(lbl, 0) + 1
			existing_labels = sorted(label_counts)

			if not existing_labels:
				st.info("No labels found. Label files using the table below or batch import labels.")
			else:
				col1, col2, col3 = st.columns([1.5, 1.5, 1])

				with col1:
					old_label = st.selectbox(
						"Select label to rename",
						options=existing_labels,
						format_func=lambda x: f"{x}  ({label_counts[x]} file(s))",
						key="rename_old_label"
					)

				with col2:
					new_label = st.text_input(
						"New label name",
						placeholder="e.g., bab-m1",
						key="rename_new_label"
					)

				with col3:
					if st.button("Rename", key="apply_rename_label", use_container_width=True, type="primary"):
						if new_label.strip() and new_label.strip() != old_label:
							files_to_update = [
								file_path for file_path in project_files
								if old_label in all_labels_dict.get(file_path.stem, [])
							]

							for file_path in files_to_update:
								current = all_labels_dict.get(file_path.stem, [])
								updated = [new_label.strip() if lbl == old_label else lbl for lbl in current]
								set_file_labels(project.path, file_path, updated)

							log_project_event(
								project.path,
								"label_renamed",
								{"old_label": old_label, "new_label": new_label.strip(), "affected_files": len(files_to_update)},
							)
							st.success(f"Renamed label '{old_label}' to '{new_label.strip()}' for {len(files_to_update)} file(s)")
							st.rerun()
						elif new_label.strip() == old_label:
							st.warning("New label is the same as the old label")
						else:
							st.error("Enter a new label name")

				st.divider()
				st.write("Apply a label to all files that currently have no labels.")
				unlabeled_files = [f for f in project_files if not all_labels_dict.get(f.stem)]
				if not unlabeled_files:
					st.info("All files already have at least one label.")
				else:
					bulk_col1, bulk_col2 = st.columns([2, 1])
					with bulk_col1:
						bulk_label = st.text_input(
							"Label for unlabeled files",
							placeholder="e.g., untagged",
							key="bulk_unlabeled_label_input",
						)
					with bulk_col2:
						if st.button(
							f"Apply to {len(unlabeled_files)} file(s)",
							key="apply_bulk_unlabeled_label",
							type="primary",
							use_container_width=True,
						):
							if bulk_label.strip():
								for file_path in unlabeled_files:
									set_file_labels(project.path, file_path, [bulk_label.strip()])
								log_project_event(
									project.path,
									"files_labeled_batch",
									{"count": len(unlabeled_files), "label": bulk_label.strip(), "files": [f.name for f in unlabeled_files]},
								)
								st.success(f"Applied label '{bulk_label.strip()}' to {len(unlabeled_files)} file(s)")
								st.rerun()
							else:
								st.error("Enter a label name")

		_render_file_library(project, project_files)

	with ingest_tab:
		_render_rss_ingest_tab(project)

	with pipelines_tab:
		run_subtab, matrix_subtab = st.tabs(["Run Pipeline", "Pipeline Matrix"])
		
		with run_subtab:
			_render_pipelines_tab(project, project_files)
		
		with matrix_subtab:
			_render_matrix_tab(
				project,
				project_files,
				load_pipelines=_load_pipelines,
				log_event=log_project_event,
				new_run_id=_new_pipeline_run_id,
			)

	with mix_tab:
		col1, col2 = st.columns(2)
		with col1:
			speech_file = st.file_uploader("Upload speech audio", type=["wav", "flac", "ogg", "mp3", "m4a"], key="speech")
		with col2:
			noise_file = st.file_uploader("Upload noise audio", type=["wav", "flac", "ogg", "mp3", "m4a"], key="noise")

		snr_db = st.slider("Target SNR (dB)", min_value=-30.0, max_value=30.0, value=-5.0, step=0.5)
		mix_button = st.button("Normalize and Mix", type="primary")

		if mix_button:
			if not speech_file or not noise_file:
				st.error("Please upload both speech and noise files.")
			else:
				with st.spinner("Normalizing inputs and mixing..."):
					speech_audio, speech_sr, speech_meta = _load_uploaded_audio(
						speech_file,
						target_sr=target_sr,
						channel_mode=channel_mode,
					)
					noise_audio, _, noise_meta = _load_uploaded_audio(
						noise_file,
						target_sr=target_sr,
						channel_mode=channel_mode,
					)

					mixed = mix_at_snr(speech_audio, noise_audio, snr_db)
					mixed_bytes = _audio_bytes(mixed, speech_sr)

				st.success("Mix complete.")

				preview_col1, preview_col2 = st.columns(2)
				with preview_col1:
					_display_audio_summary("Speech import", speech_meta)
					st.audio(_audio_bytes(speech_audio, speech_sr), format="audio/wav")
				with preview_col2:
					_display_audio_summary("Noise import", noise_meta)
					st.audio(_audio_bytes(noise_audio, speech_sr), format="audio/wav")

				st.markdown("### Mixed output")
				st.audio(mixed_bytes, format="audio/wav")
				st.download_button(
					label="Download mix",
					data=mixed_bytes,
					file_name="triton_mix.wav",
					mime="audio/wav",
				)
				log_project_event(
					project.path,
					"mix_preview_generated",
					{
						"snr_db": float(snr_db),
						"speech_file": speech_meta["filename"],
						"noise_file": noise_meta["filename"],
					},
				)

	with babble_tab:
		st.markdown("### Babble Speech Generation")
		st.write(
			"Label files as bab-f1, bab-f2, bab-m1, bab-m2, etc. Files that share a label are concatenated into one talker before mixing."
		)

		babble_groups = load_babble_talker_groups(project.path)
		if len(babble_groups) < 2:
			st.info("Label at least two imported files with bab-f1, bab-m1, and similar labels to generate babble.")
		else:
			group_cols = st.columns(2)
			with group_cols[0]:
				st.markdown("#### Available Talker Groups")
				for group in babble_groups:
					st.caption(f"{group.label} ({group.sex}, {len(group.files)} file(s))")
			with group_cols[1]:
				st.markdown("#### Talker Counts")
				num_talkers_default = min(len(babble_groups), max(2, len(babble_groups)))
				# Guard against stale widget state from older runs where the minimum differed.
				if "babble_num_talkers" in st.session_state:
					st.session_state["babble_num_talkers"] = int(
						max(2, min(len(babble_groups), int(st.session_state["babble_num_talkers"])))
					)
				num_talkers = st.number_input(
					"Number of talkers",
					min_value=2,
					max_value=len(babble_groups),
					value=int(num_talkers_default),
					step=1,
					key="babble_num_talkers",
				)
				use_split = st.checkbox(
					"Set female and male talkers separately",
					value=False,
					key="babble_use_sex_split",
				)
				num_female_talkers = None
				num_male_talkers = None
				if use_split:
					female_groups = [group for group in babble_groups if group.sex == "f"]
					male_groups = [group for group in babble_groups if group.sex == "m"]
					split_cols = st.columns(2)
					with split_cols[0]:
						num_female_talkers = st.number_input(
							"Female talkers",
							min_value=0,
							max_value=len(female_groups),
							value=min(len(female_groups), int(num_talkers) // 2),
							step=1,
							key="babble_num_female_talkers",
						)
					with split_cols[1]:
						num_male_talkers = st.number_input(
							"Male talkers",
							min_value=0,
							max_value=len(male_groups),
							value=min(len(male_groups), int(num_talkers) - int(num_talkers) // 2),
							step=1,
							key="babble_num_male_talkers",
						)

				st.markdown("#### Mix Settings")
				set1, set2, set3 = st.columns(3)
				with set1:
					target_rms = st.slider(
						"Target RMS level",
						min_value=0.01,
						max_value=0.5,
						value=0.1,
						step=0.01,
						help="RMS level to normalize each source file to before concatenating and mixing.",
					)
				with set2:
					intended_length_seconds = st.number_input(
						"Intended length (seconds)",
						min_value=1.0,
						max_value=3600.0,
						value=30.0,
						step=1.0,
						help="Per-talker's concatenated length target. Extra files are skipped once reached.",
					)
				with set3:
					peak_norm = st.checkbox(
						"Peak normalize output",
						value=True,
						help="Rescale the final babble output to prevent clipping.",
					)

					babble_generation_signature = _babble_generation_signature(
						project,
						num_talkers=int(num_talkers),
						num_female_talkers=None if num_female_talkers is None else int(num_female_talkers),
						num_male_talkers=None if num_male_talkers is None else int(num_male_talkers),
						intended_length_seconds=float(intended_length_seconds),
						target_rms=float(target_rms),
						peak_normalize=bool(peak_norm),
					)
					stored_babble_output = st.session_state.get("babble_generated_output")
					if stored_babble_output and stored_babble_output.get("signature") != babble_generation_signature:
						st.session_state.pop("babble_generated_output", None)
						stored_babble_output = None

				mix_babble_button = st.button("Generate Babble", type="primary", key="mix_babble_button")

				if mix_babble_button:
					st.session_state.pop("babble_generated_output", None)
					progress_bar = st.progress(0)
					status_line = st.empty()
					console_box = st.empty()
					console_lines: list[str] = []

					def _append_console(message: str, progress: int | None = None) -> None:
						console_lines.append(message)
						console_box.code("\n".join(console_lines[-12:]), language="text")
						if progress is not None:
							progress_bar.progress(max(0, min(100, int(progress))))
						status_line.caption(message)

					try:
						result = generate_project_babble(
							project.path,
							sr=target_sr,
							channel_mode=channel_mode,
							num_talkers=int(num_talkers),
							num_female_talkers=None if num_female_talkers is None else int(num_female_talkers),
							num_male_talkers=None if num_male_talkers is None else int(num_male_talkers),
							intended_length_seconds=float(intended_length_seconds),
							target_rms=float(target_rms),
							peak_normalize=bool(peak_norm),
							progress_callback=_append_console,
						)

						if result.repeat_counts_by_label:
							_append_console(
								"Random repeats applied for short talkers: "
								+ ", ".join(
									f"{label} (+{count})" for label, count in sorted(result.repeat_counts_by_label.items())
								),
								progress=70,
							)

						_append_console("Encoding output WAV...", progress=90)
						babble_bytes = _audio_bytes(result.audio, target_sr)
						_append_console("Babble generation complete.", progress=100)
						log_project_event(
							project.path,
							"babble_generated",
							{
								"num_talkers": int(num_talkers),
								"female_talkers": None if num_female_talkers is None else int(num_female_talkers),
								"male_talkers": None if num_male_talkers is None else int(num_male_talkers),
								"intended_length_seconds": float(intended_length_seconds),
								"talker_labels": [group.label for group in result.selected_groups],
								"talker_files": [
									file_path.name
									for files in result.planned_group_files
									for file_path in files
								],
								"talker_repeats": result.repeat_counts_by_label,
								"short_source_labels": result.short_source_labels,
								"unknown_duration_labels": result.unknown_duration_labels,
								"target_rms": float(target_rms),
								"peak_normalize": bool(peak_norm),
							},
						)

						st.session_state["babble_generated_output"] = {
							"signature": babble_generation_signature,
							"audio": result.audio,
							"babble_bytes": babble_bytes,
							"sample_rate": target_sr,
							"num_talkers": int(num_talkers),
							"num_female_talkers": None if num_female_talkers is None else int(num_female_talkers),
							"num_male_talkers": None if num_male_talkers is None else int(num_male_talkers),
							"intended_length_seconds": float(intended_length_seconds),
							"target_rms": float(target_rms),
							"peak_normalize": bool(peak_norm),
							"selected_groups": [
								{"label": group.label, "sex": group.sex, "index": int(group.index)}
								for group in result.selected_groups
							],
							"planned_group_files": [
								[str(file_path.resolve()) for file_path in files] for files in result.planned_group_files
							],
							"short_source_labels": list(result.short_source_labels),
							"unknown_duration_labels": list(result.unknown_duration_labels),
							"repeat_counts_by_label": dict(result.repeat_counts_by_label),
						}

					except Exception as exc:
						_append_console(f"Babble generation failed: {exc}", progress=100)
						st.error(f"Babble generation failed: {exc}")

				stored_babble_output = st.session_state.get("babble_generated_output")
				if stored_babble_output and stored_babble_output.get("signature") == babble_generation_signature:
					_render_saved_babble_output(project, stored_babble_output)

	with transcribe_tab:
		st.markdown("### Transcribe")
		st.write("Run speech-to-text on project files using OpenAI Whisper.")

		if not project_files:
			st.info("Import some audio files first to transcribe them.")
		else:
			tcol1, tcol2 = st.columns([2, 1])
			with tcol1:
				selected_file = st.selectbox(
					"Select file to transcribe",
					options=project_files,
					format_func=lambda p: Path(p).name,
					key="transcribe_file_select",
				)
			with tcol2:
				model_size = st.selectbox(
					"Whisper model",
					options=["tiny", "base", "small", "medium", "large"],
					index=2,
					help="Larger models are more accurate but slower. 'small' is a good default.",
					key="transcribe_model",
				)

			language = st.text_input(
				"Language code (optional)",
				placeholder="e.g. en, fr, de — leave blank for auto-detect",
				key="transcribe_lang",
			)

			run_transcribe = st.button("Transcribe", type="primary", key="run_transcribe")

			if run_transcribe and selected_file:
				with st.spinner(f"Loading Whisper '{model_size}' model and transcribing..."):
					try:
						from triton.transcribe.whisper import transcribe_file

						result = transcribe_file(
							selected_file,
							model_size=model_size,
							language=language if language else None,
						)

						st.success(f"Transcription complete — detected language: **{result.language or 'unknown'}**")

						st.markdown("#### Full text")
						st.text_area(
							"Transcript",
							value=result.text,
							height=150,
							key="transcript_output",
							label_visibility="collapsed",
						)

						if result.segments:
							st.markdown("#### Segments")
							segment_data = [
								{
									"Start (s)": f"{seg.start:.1f}",
									"End (s)": f"{seg.end:.1f}",
									"Text": seg.text.strip(),
								}
								for seg in result.segments
							]
							st.dataframe(segment_data, use_container_width=True)

						log_project_event(
							project.path,
							"transcription_completed",
							{
								"file": Path(selected_file).name,
								"model": model_size,
								"language": result.language,
							},
						)
					except Exception as exc:
						st.error(f"Transcription failed: {exc}")

	with roadmap_tab:
		st.markdown("### Next GUI milestones")
		st.write("Add batch transcription across multiple files and compare intelligibility scores.")
		st.write("Integrate time compression into the Mix tab for combined degradation experiments.")
		st.write("Add an asset browser and recent-run history backed by project metadata.")


def render_app() -> None:
	st.set_page_config(page_title="Triton", page_icon="🐚", layout="wide")
	_render_styles()

	active_project = st.session_state.get("active_project")
	_hero(active_project)

	if active_project is None:
		_render_project_launcher(
			set_active_project=_set_active_project,
			collect_spectrogram_settings=_collect_spectrogram_settings,
			create_project_fn=create_project,
			register_recent_project_fn=register_recent_project,
			load_recent_projects_fn=load_recent_projects,
		)
	else:
		_render_project_workspace(active_project)