"""Triton Streamlit GUI."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import streamlit as st

from triton.core.mixer import mix_at_snr
from triton.core.project import (
	ChannelMode,
	Project,
	add_project_file,
	create_project,
	delete_project_file,
	list_project_files,
	load_project_config,
	load_recent_projects,
	project_raw_dir,
	rename_project_file,
	register_recent_project,
)


def _set_active_project(project_dir: Path) -> Project:
	project = load_project_config(project_dir)
	st.session_state["active_project"] = project
	register_recent_project(project_dir, project.name)
	return project


def _create_project(project_dir: Path, sample_rate: int, channel_mode: ChannelMode) -> Project:
	project = create_project(project_dir, sample_rate=sample_rate, channel_mode=channel_mode)
	st.session_state["active_project"] = project
	register_recent_project(project_dir, project.name)
	return project


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


def _save_uploaded_project_files(project_dir: Path, uploaded_files: list[object]) -> int:
	saved_count = 0
	for uploaded_file in uploaded_files:
		add_project_file(project_dir, uploaded_file.name, uploaded_file.getvalue())
		saved_count += 1

	return saved_count


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
	st.markdown(
		"""
		<style>
		:root {
			--bg-top: #06141f;
			--bg-mid: #0e3040;
			--bg-bottom: #d7e8ea;
			--panel: rgba(7, 30, 43, 0.74);
			--panel-border: rgba(143, 205, 206, 0.22);
			--panel-soft: rgba(250, 252, 252, 0.06);
			--accent: #ff9f1c;
			--accent-soft: #ffd6a0;
			--ink: #ebf6f7;
			--muted: #b7d0d4;
		}

		.stApp {
			background:
				radial-gradient(circle at top left, rgba(255, 159, 28, 0.16), transparent 30%),
				linear-gradient(180deg, var(--bg-top) 0%, var(--bg-mid) 45%, #153e50 72%, var(--bg-bottom) 140%);
		}

		[data-testid="stHeader"] {
			background: transparent;
		}

		[data-testid="stSidebar"] {
			background: rgba(5, 20, 33, 0.9);
			border-right: 1px solid rgba(143, 205, 206, 0.18);
		}

		h1, h2, h3, h4, h5, h6, p, label, div, span {
			color: var(--ink) !important;
		}

		div[data-testid="stMetric"] {
			background: linear-gradient(180deg, rgba(9, 40, 55, 0.95), rgba(9, 40, 55, 0.7));
			border: 1px solid var(--panel-border);
			padding: 14px;
			border-radius: 18px;
			box-shadow: 0 18px 40px rgba(2, 10, 16, 0.22);
		}

		.stButton > button, .stDownloadButton > button, button[kind="primary"] {
			background: linear-gradient(135deg, #ff9f1c, #ffbf69) !important;
			color: #08202c !important;
			border: none !important;
			border-radius: 999px !important;
			font-weight: 700 !important;
			box-shadow: 0 12px 24px rgba(255, 159, 28, 0.28);
		}

		.stTextInput input, .stSelectbox div[data-baseweb="select"], .stTextArea textarea {
			background: rgba(255, 255, 255, 0.06) !important;
			border-radius: 14px !important;
		}

		.stFileUploader, .stAudio, .stForm {
			background: var(--panel);
			border: 1px solid var(--panel-border);
			border-radius: 20px;
			padding: 12px;
			box-shadow: 0 18px 40px rgba(2, 10, 16, 0.16);
		}

		div[data-baseweb="tab-list"] {
			gap: 8px;
		}

		button[data-baseweb="tab"] {
			background: rgba(255, 255, 255, 0.06);
			border-radius: 999px;
			padding: 8px 16px;
		}

		button[data-baseweb="tab"][aria-selected="true"] {
			background: rgba(255, 159, 28, 0.2);
		}
		</style>
		""",
		unsafe_allow_html=True,
	)


def _hero(project: Project | None) -> None:
	status = "No active project" if project is None else f"Active project: {project.name}"
	st.markdown(
		f"""
		<div style="padding: 28px; border-radius: 28px; background: linear-gradient(135deg, rgba(4, 27, 39, 0.92), rgba(14, 61, 78, 0.88)); border: 1px solid rgba(143, 205, 206, 0.2); box-shadow: 0 30px 60px rgba(2, 10, 16, 0.24); margin-bottom: 18px;">
		  <div style="font-size: 13px; letter-spacing: 0.14em; text-transform: uppercase; color: #ffd6a0; margin-bottom: 10px;">Triton workspace</div>
		  <div style="font-size: 42px; font-weight: 800; line-height: 1.05; margin-bottom: 12px;">Project-shaped audio workflows</div>
		  <div style="max-width: 760px; color: #cbe2e5; font-size: 17px; line-height: 1.6; margin-bottom: 14px;">Start from a project anywhere on disk, keep its audio rules consistent, and let Triton handle import, storage, and processing inside that boundary.</div>
		  <div style="display: inline-block; padding: 8px 14px; border-radius: 999px; background: rgba(255, 159, 28, 0.16); color: #ffe9c7; font-size: 13px; font-weight: 700;">{status}</div>
		</div>
		""",
		unsafe_allow_html=True,
	)


def _render_project_launcher() -> None:
	st.subheader("Open or create a project")
	st.write(
		"Pick an existing project anywhere on disk or create a new one with the canonical audio settings you want Triton to enforce."
	)

	create_tab, open_tab, recent_tab = st.tabs(["Create", "Open", "Recent"])

	with create_tab:
		with st.form("create_project_form"):
			project_name = st.text_input("Project name", value="my-project")
			project_root = st.text_input("Project folder", value=str(Path.home() / "Projects" / "triton" / "my-project"))
			sample_rate = st.selectbox("Sample rate", options=[8000, 16000, 22050, 24000, 32000, 44100, 48000], index=1, key="create_sr")
			channel_mode = st.radio("Channel mode", options=["mono", "stereo"], horizontal=True, key="create_channels")
			create_submitted = st.form_submit_button("Create project", type="primary")

		if create_submitted:
			project_dir = Path(project_root).expanduser()
			if project_dir.name != project_name and project_name.strip():
				project_dir = project_dir.parent / project_name.strip()

			try:
				project = _create_project(project_dir, sample_rate=sample_rate, channel_mode=channel_mode)
			except Exception as exc:
				st.error(f"Could not create project: {exc}")
			else:
				st.success(f"Project ready: {project.name}")
				st.rerun()

	with open_tab:
		with st.form("open_project_form"):
			project_root = st.text_input("Existing project folder", value=str(Path.home()))
			open_submitted = st.form_submit_button("Open project", type="primary")

		if open_submitted:
			project_dir = Path(project_root).expanduser()
			try:
				_set_active_project(project_dir)
			except Exception as exc:
				st.error(f"Could not open project: {exc}")
			else:
				st.success(f"Opened {project_dir.name}")
				st.rerun()

	with recent_tab:
		recent_projects = load_recent_projects()
		if not recent_projects:
			st.info("No recent projects yet.")
		else:
			for index, project in enumerate(recent_projects):
				col1, col2 = st.columns([5, 1])
				with col1:
					st.markdown(f"#### {project['name']}")
					st.caption(project["path"])
				with col2:
					if st.button("Open", key=f"recent_project_{index}"):
						try:
							_set_active_project(Path(project["path"]).expanduser())
						except Exception as exc:
							st.error(f"Could not open project: {exc}")
						else:
							st.rerun()


def _render_file_library(project_dir: Path, project_files: list[Path]) -> None:
	st.markdown("### Project library")
	st.caption(f"Raw project assets live in {project_raw_dir(project_dir)}")

	add_col, count_col = st.columns([2, 3])
	with add_col:
		with st.form("project_file_upload_form"):
			uploaded_files = st.file_uploader(
				"Add files to project",
				type=["wav", "flac", "ogg", "mp3", "m4a"],
				accept_multiple_files=True,
				key="project_file_upload",
			)
			upload_submitted = st.form_submit_button("Add selected files", type="primary")

		if upload_submitted:
			if not uploaded_files:
				st.error("Choose at least one file to add.")
			else:
				saved_count = _save_uploaded_project_files(project_dir, list(uploaded_files))
				st.success(f"Added {saved_count} file(s) to the project.")
				st.rerun()

	with count_col:
		st.markdown(
			f"""
			<div style="height: 100%; min-height: 136px; padding: 18px 20px; border-radius: 22px; background: rgba(255, 255, 255, 0.06); border: 1px solid rgba(143, 205, 206, 0.18);">
			  <div style="font-size: 12px; text-transform: uppercase; letter-spacing: 0.14em; color: #ffd6a0; margin-bottom: 8px;">Library status</div>
			  <div style="font-size: 36px; font-weight: 800; margin-bottom: 6px;">{len(project_files)}</div>
			  <div style="color: #cbe2e5;">audio file(s) currently stored in this project.</div>
			</div>
			""",
			unsafe_allow_html=True,
		)

	if not project_files:
		st.info("No files have been added to this project yet.")
		return

	for index, file_path in enumerate(project_files):
		row_background = "rgba(7, 30, 43, 0.72)" if index % 2 == 0 else "rgba(12, 40, 56, 0.64)"
		row_border = "rgba(143, 205, 206, 0.20)" if index % 2 == 0 else "rgba(143, 205, 206, 0.14)"
		st.markdown(
			f"""
			<div style="height: 0; border-top: 22px solid {row_background}; border-left: 1px solid {row_border}; border-right: 1px solid {row_border}; border-radius: 8px 8px 0 0; margin-top: 4px;"></div>
			""",
			unsafe_allow_html=True,
		)

		with st.form(f"rename_file_{index}"):
			row_col1, row_col2, row_col3, row_col4 = st.columns([6.0, 1.1, 1.1, 2.0])
			with row_col1:
				new_name = st.text_input("File name", value=file_path.name, key=f"rename_input_{index}", label_visibility="collapsed")
				st.caption(str(file_path))
			with row_col2:
				st.caption(f"{file_path.suffix.lower()}")
			with row_col3:
				st.caption(_format_file_size(file_path.stat().st_size))
			with row_col4:
				action_col1, action_col2 = st.columns(2)
				rename_submitted = action_col1.form_submit_button("✏️", help="Rename file")
				remove_submitted = action_col2.form_submit_button("🗑️", help="Remove file")

			if rename_submitted:
				try:
					_rename_project_file(file_path, new_name)
				except Exception as exc:
					st.error(f"Could not rename {file_path.name}: {exc}")
				else:
					st.success(f"Renamed to {new_name}")
					st.rerun()

			if remove_submitted:
				_delete_project_file(file_path)
				st.rerun()


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
		if st.button("Close project"):
			_clear_active_project()
			st.rerun()

	overview_tab, mix_tab, roadmap_tab = st.tabs(["Overview", "Mix", "Roadmap"])

	with overview_tab:
		metric_col1, metric_col2, metric_col3 = st.columns(3)
		metric_col1.metric("Project sample rate", f"{target_sr} Hz")
		metric_col2.metric("Channel policy", channel_mode)
		metric_col3.metric("Stored files", str(len(project_files)))

		st.markdown("### Active project")
		st.write(
			"This workspace starts from a concrete project on disk. The project owns the canonical audio settings, and stored assets are managed inside that boundary."
		)
		st.write(
			"The file library below is the first step toward project-native import, ingest, and processing instead of ad hoc one-off uploads."
		)

		_render_file_library(project_dir, project_files)

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

	with roadmap_tab:
		st.markdown("### Next GUI milestones")
		st.write("Turn file adds into true project import that normalizes assets into canonical storage.")
		st.write("Route RSS ingest through the same import pipeline so external audio becomes project-managed immediately.")
		st.write("Add an asset browser and recent-run history backed by project metadata.")


def render_app() -> None:
	st.set_page_config(page_title="Triton", page_icon="🐚", layout="wide")
	_render_styles()

	active_project = st.session_state.get("active_project")
	_hero(active_project)

	if active_project is None:
		_render_project_launcher()
	else:
		_render_project_workspace(active_project)