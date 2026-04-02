"""Shared GUI helpers for Triton."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import streamlit as st

from triton.core.io import load_audio, write_sidecar
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
	set_file_label,
	set_project_file_labels,
)
from triton.core.spectrogram import compute_spectrogram, load_spectrogram, save_spectrogram


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
		extra={"spectrogram": {"type": result.kind, "shape": [int(result.values.shape[0]), int(result.values.shape[1])] }},
	)
	return out_path


def _save_uploaded_project_files(project: Project, uploaded_files: list[object], batch_label: str = "") -> list[Path]:
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
		log_project_event(project.path, "files_labeled_batch", {"count": len(saved_paths), "label": batch_label.strip(), "files": [path.name for path in saved_paths]})

	if saved_paths:
		log_project_event(project.path, "files_imported_batch", {"count": len(saved_paths), "files": [path.name for path in saved_paths]})

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

	log_project_event(project.path, "spectrogram_regenerated_all", {"updated_files": updated, "failed_files": len(errors)})
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

	channels = [librosa.resample(audio[:, channel], orig_sr=source_sr, target_sr=target_sr) for channel in range(audio.shape[1])]
	return np.stack(channels, axis=1).astype(np.float32)


def _convert_channels(audio: np.ndarray, channel_mode: ChannelMode) -> np.ndarray:
	if channel_mode == "mono":
		return np.mean(audio, axis=1, dtype=np.float32)

	if audio.shape[1] == 1:
		return np.repeat(audio, 2, axis=1).astype(np.float32)

	if audio.shape[1] >= 2:
		return audio[:, :2].astype(np.float32)

	raise ValueError("Audio must contain at least one channel.")


def _load_uploaded_audio(uploaded_file, target_sr: int, channel_mode: str) -> tuple[np.ndarray, int, dict[str, int | str]]:
	raw_audio, source_sr = _read_uploaded_audio(uploaded_file)
	resampled_audio = _resample_audio(raw_audio, source_sr=source_sr, target_sr=target_sr)
	converted_audio = _convert_channels(resampled_audio, channel_mode=channel_mode)
	metadata = {"filename": uploaded_file.name, "source_sr": source_sr, "target_sr": target_sr, "source_channels": raw_audio.shape[1], "target_channels": 1 if channel_mode == "mono" else 2}
	return converted_audio, target_sr, metadata


def _audio_bytes(audio: np.ndarray, sr: int) -> bytes:
	buffer = BytesIO()
	frames = audio if audio.ndim == 1 else np.asarray(audio, dtype=np.float32)
	sf.write(buffer, frames, sr, format="WAV")
	buffer.seek(0)
	return buffer.read()


def _display_audio_summary(title: str, metadata: dict[str, int | str]) -> None:
	st.markdown(f"#### {title}")
	st.caption(f"{metadata['filename']} | {metadata['source_sr']} Hz -> {metadata['target_sr']} Hz | {metadata['source_channels']} ch -> {metadata['target_channels']} ch")


def _pipeline_key(name: str) -> str:
	return pipeline_key(name)


def _pipeline_output_dir(project: Project, pipeline_name: str) -> Path:
	return pipeline_output_dir(project, pipeline_name)


def _pipeline_run_dir(project: Project, pipeline_name: str, run_id: str) -> Path:
	return pipeline_run_dir(project, pipeline_name, run_id)


def _new_pipeline_run_id() -> str:
	return new_pipeline_run_id()


def _apply_pipeline_step(audio: np.ndarray, sr: int, step: str, project: Project, step_options: dict[str, object] | None = None) -> tuple[np.ndarray, int]:
	return apply_pipeline_step(audio, sr, step, project, step_options)


def _run_pipeline_on_file(file_path: Path, project: Project, pipeline: Pipeline, run_dir: Path) -> Path:
	return run_pipeline_on_file(file_path, project, pipeline, run_dir)


def _save_pipelines(project: Project, pipelines: list[Pipeline]) -> None:
	save_project_pipelines(project.path, pipelines)


def _load_pipelines(project: Project) -> list[Pipeline]:
	return load_project_pipelines(project.path)
