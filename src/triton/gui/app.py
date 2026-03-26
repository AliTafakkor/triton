"""Triton Streamlit GUI."""

from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from uuid import uuid4

import librosa
import numpy as np
import plotly.graph_objects as go
import soundfile as sf
import streamlit as st

from triton.core.mixer import mix_at_snr
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
	update_project_spectrogram_settings,
)
from triton.core.io import load_audio, normalize_peak, save_audio, write_sidecar
from triton.core.conversion import requantize
from triton.core.spectrogram import compute_spectrogram, load_spectrogram, save_spectrogram
from triton.degrade.vocoder import noise_vocode


PIPELINE_ACTIONS: dict[str, str] = {
	"normalize": "Peak normalize",
	"resample_project": "Resample to project sample rate",
	"to_mono": "Convert to mono",
	"to_stereo": "Convert to stereo",
	"requantize_16": "Requantize to 16-bit",
	"vocode_noise": "Noise vocoder degradation",
}

PIPELINE_STEP_ORDER = list(PIPELINE_ACTIONS.keys())
PIPELINE_DEFAULT_STEP = "normalize"


def _pipeline_action_label(action: str) -> str:
	return PIPELINE_ACTIONS.get(action, action)


def _default_step_options(step: str, project_sr: int) -> dict[str, object]:
	if step == "normalize":
		return {"target_peak": 0.99}
	if step == "resample_project":
		return {"target_mode": "project", "custom_sr": int(project_sr)}
	if step == "requantize_16":
		return {"bit_depth": 16}
	if step == "vocode_noise":
		return {"n_bands": 8, "vocoder_type": "noise", "envelope_cutoff": 160.0}
	return {}


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


def _save_uploaded_project_files(project: Project, uploaded_files: list[object]) -> list[Path]:
	saved_paths: list[Path] = []
	for uploaded_file in uploaded_files:
		path = add_project_file(project.path, uploaded_file.name, uploaded_file.getvalue())
		saved_paths.append(path)
		_generate_file_spectrogram(path, project)

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
	for file_path in project_files:
		try:
			_generate_file_spectrogram(file_path, project)
		except Exception as exc:
			errors.append(f"{file_path.name}: {exc}")
		else:
			updated += 1

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
			--hero-start: rgba(4, 27, 39, 0.92);
			--hero-end: rgba(14, 61, 78, 0.88);
			--hero-kicker: #ffd6a0;
			--hero-body: #cbe2e5;
			--hero-pill-bg: rgba(255, 159, 28, 0.16);
			--hero-pill-text: #ffe9c7;
			--card-top-bg: rgba(255, 255, 255, 0.06);
			--card-subtle-text: #cbe2e5;
			--metric-start: rgba(9, 40, 55, 0.95);
			--metric-end: rgba(9, 40, 55, 0.7);
		}

		html[data-theme="light"], body[data-theme="light"], [data-theme="light"] {
			--bg-top: #f6efe4;
			--bg-mid: #f7fbff;
			--bg-bottom: #e7f1f3;
			--panel: rgba(255, 255, 255, 0.92);
			--panel-border: rgba(16, 41, 53, 0.16);
			--panel-soft: rgba(10, 37, 50, 0.05);
			--ink: #102935;
			--muted: #385767;
			--hero-start: rgba(255, 255, 255, 0.96);
			--hero-end: rgba(235, 247, 250, 0.98);
			--hero-kicker: #8f4b00;
			--hero-body: #26485a;
			--hero-pill-bg: rgba(255, 159, 28, 0.22);
			--hero-pill-text: #5f3400;
			--card-top-bg: rgba(255, 255, 255, 0.9);
			--card-subtle-text: #2e4f5f;
			--metric-start: rgba(255, 255, 255, 0.97);
			--metric-end: rgba(240, 248, 251, 0.97);
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

		html[data-theme="light"] [data-testid="stSidebar"],
		body[data-theme="light"] [data-testid="stSidebar"],
		[data-theme="light"] [data-testid="stSidebar"] {
			background: rgba(255, 255, 255, 0.94);
			border-right: 1px solid rgba(16, 41, 53, 0.12);
		}

		h1, h2, h3, h4, h5, h6, p, label, div, span {
			color: var(--ink) !important;
		}

		div[data-testid="stMetric"] {
			background: linear-gradient(180deg, var(--metric-start), var(--metric-end));
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

		.stButton > button *, .stDownloadButton > button *, button[kind="primary"] * {
			color: #08202c !important;
		}

		html[data-theme="light"] .stButton > button[kind="secondary"],
		html[data-theme="light"] .stButton > button[kind="tertiary"],
		html[data-theme="light"] .stDownloadButton > button[kind="secondary"],
		html[data-theme="light"] .stDownloadButton > button[kind="tertiary"],
		body[data-theme="light"] .stButton > button[kind="secondary"],
		body[data-theme="light"] .stButton > button[kind="tertiary"],
		body[data-theme="light"] .stDownloadButton > button[kind="secondary"],
		body[data-theme="light"] .stDownloadButton > button[kind="tertiary"],
		[data-theme="light"] .stButton > button[kind="secondary"],
		[data-theme="light"] .stButton > button[kind="tertiary"],
		[data-theme="light"] .stDownloadButton > button[kind="secondary"],
		[data-theme="light"] .stDownloadButton > button[kind="tertiary"] {
			color: #102935 !important;
			border: 1px solid rgba(16, 41, 53, 0.22) !important;
			background: rgba(255, 255, 255, 0.96) !important;
		}

		html[data-theme="light"] .stButton > button[kind="secondary"] *,
		html[data-theme="light"] .stButton > button[kind="tertiary"] *,
		html[data-theme="light"] .stDownloadButton > button[kind="secondary"] *,
		html[data-theme="light"] .stDownloadButton > button[kind="tertiary"] *,
		body[data-theme="light"] .stButton > button[kind="secondary"] *,
		body[data-theme="light"] .stButton > button[kind="tertiary"] *,
		body[data-theme="light"] .stDownloadButton > button[kind="secondary"] *,
		body[data-theme="light"] .stDownloadButton > button[kind="tertiary"] *,
		[data-theme="light"] .stButton > button[kind="secondary"] *,
		[data-theme="light"] .stButton > button[kind="tertiary"] *,
		[data-theme="light"] .stDownloadButton > button[kind="secondary"] *,
		[data-theme="light"] .stDownloadButton > button[kind="tertiary"] * {
			color: #102935 !important;
		}

		@media (prefers-color-scheme: light) {
			.stButton > button[kind="secondary"],
			.stButton > button[kind="tertiary"],
			.stDownloadButton > button[kind="secondary"],
			.stDownloadButton > button[kind="tertiary"] {
				color: #102935 !important;
				border: 1px solid rgba(16, 41, 53, 0.22) !important;
				background: rgba(255, 255, 255, 0.96) !important;
			}

			.stButton > button[kind="secondary"] *,
			.stButton > button[kind="tertiary"] *,
			.stDownloadButton > button[kind="secondary"] *,
			.stDownloadButton > button[kind="tertiary"] * {
				color: #102935 !important;
			}
		}

		.stTextInput input, .stSelectbox div[data-baseweb="select"], .stTextArea textarea {
			background: rgba(255, 255, 255, 0.06) !important;
			border-radius: 14px !important;
		}

		html[data-theme="light"] .stTextInput input,
		html[data-theme="light"] .stSelectbox div[data-baseweb="select"],
		html[data-theme="light"] .stTextArea textarea,
		body[data-theme="light"] .stTextInput input,
		body[data-theme="light"] .stSelectbox div[data-baseweb="select"],
		body[data-theme="light"] .stTextArea textarea,
		[data-theme="light"] .stTextInput input,
		[data-theme="light"] .stSelectbox div[data-baseweb="select"],
		[data-theme="light"] .stTextArea textarea {
			background: rgba(255, 255, 255, 0.92) !important;
			border: 1px solid rgba(16, 41, 53, 0.18) !important;
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

		html[data-theme="light"] button[data-baseweb="tab"],
		body[data-theme="light"] button[data-baseweb="tab"],
		[data-theme="light"] button[data-baseweb="tab"] {
			background: rgba(255, 255, 255, 0.85);
			border: 1px solid rgba(16, 41, 53, 0.14);
		}

		/* Force dark theme even when user/browser selects light mode. */
		html[data-theme="light"],
		body[data-theme="light"],
		[data-theme="light"] {
			color-scheme: dark !important;
			--bg-top: #06141f !important;
			--bg-mid: #0e3040 !important;
			--bg-bottom: #d7e8ea !important;
			--panel: rgba(7, 30, 43, 0.74) !important;
			--panel-border: rgba(143, 205, 206, 0.22) !important;
			--panel-soft: rgba(250, 252, 252, 0.06) !important;
			--ink: #ebf6f7 !important;
			--muted: #b7d0d4 !important;
			--hero-start: rgba(4, 27, 39, 0.92) !important;
			--hero-end: rgba(14, 61, 78, 0.88) !important;
			--hero-kicker: #ffd6a0 !important;
			--hero-body: #cbe2e5 !important;
			--hero-pill-bg: rgba(255, 159, 28, 0.16) !important;
			--hero-pill-text: #ffe9c7 !important;
			--card-top-bg: rgba(255, 255, 255, 0.06) !important;
			--card-subtle-text: #cbe2e5 !important;
			--metric-start: rgba(9, 40, 55, 0.95) !important;
			--metric-end: rgba(9, 40, 55, 0.7) !important;
		}

		html[data-theme="light"] [data-testid="stSidebar"],
		body[data-theme="light"] [data-testid="stSidebar"],
		[data-theme="light"] [data-testid="stSidebar"] {
			background: rgba(5, 20, 33, 0.9) !important;
			border-right: 1px solid rgba(143, 205, 206, 0.18) !important;
		}

		html[data-theme="light"] .stTextInput input,
		html[data-theme="light"] .stSelectbox div[data-baseweb="select"],
		html[data-theme="light"] .stTextArea textarea,
		body[data-theme="light"] .stTextInput input,
		body[data-theme="light"] .stSelectbox div[data-baseweb="select"],
		body[data-theme="light"] .stTextArea textarea,
		[data-theme="light"] .stTextInput input,
		[data-theme="light"] .stSelectbox div[data-baseweb="select"],
		[data-theme="light"] .stTextArea textarea {
			background: rgba(255, 255, 255, 0.06) !important;
			border: 1px solid rgba(143, 205, 206, 0.22) !important;
		}

		html[data-theme="light"] .stButton > button[kind="secondary"],
		html[data-theme="light"] .stButton > button[kind="tertiary"],
		html[data-theme="light"] .stDownloadButton > button[kind="secondary"],
		html[data-theme="light"] .stDownloadButton > button[kind="tertiary"],
		body[data-theme="light"] .stButton > button[kind="secondary"],
		body[data-theme="light"] .stButton > button[kind="tertiary"],
		body[data-theme="light"] .stDownloadButton > button[kind="secondary"],
		body[data-theme="light"] .stDownloadButton > button[kind="tertiary"],
		[data-theme="light"] .stButton > button[kind="secondary"],
		[data-theme="light"] .stButton > button[kind="tertiary"],
		[data-theme="light"] .stDownloadButton > button[kind="secondary"],
		[data-theme="light"] .stDownloadButton > button[kind="tertiary"] {
			color: #ebf6f7 !important;
			border: 1px solid rgba(143, 205, 206, 0.22) !important;
			background: rgba(255, 255, 255, 0.06) !important;
		}

		html[data-theme="light"] .stButton > button[kind="secondary"] *,
		html[data-theme="light"] .stButton > button[kind="tertiary"] *,
		html[data-theme="light"] .stDownloadButton > button[kind="secondary"] *,
		html[data-theme="light"] .stDownloadButton > button[kind="tertiary"] *,
		body[data-theme="light"] .stButton > button[kind="secondary"] *,
		body[data-theme="light"] .stButton > button[kind="tertiary"] *,
		body[data-theme="light"] .stDownloadButton > button[kind="secondary"] *,
		body[data-theme="light"] .stDownloadButton > button[kind="tertiary"] *,
		[data-theme="light"] .stButton > button[kind="secondary"] *,
		[data-theme="light"] .stButton > button[kind="tertiary"] *,
		[data-theme="light"] .stDownloadButton > button[kind="secondary"] *,
		[data-theme="light"] .stDownloadButton > button[kind="tertiary"] * {
			color: #ebf6f7 !important;
		}

		html[data-theme="light"] button[data-baseweb="tab"],
		body[data-theme="light"] button[data-baseweb="tab"],
		[data-theme="light"] button[data-baseweb="tab"] {
			background: rgba(255, 255, 255, 0.06) !important;
			border: 1px solid rgba(143, 205, 206, 0.18) !important;
		}
		</style>
		""",
		unsafe_allow_html=True,
	)


def _hero(project: Project | None) -> None:
	status = "No active project" if project is None else f"Active project: {project.name}"
	st.markdown(
		f"""
		<div style="padding: 28px; border-radius: 28px; background: linear-gradient(135deg, var(--hero-start), var(--hero-end)); border: 1px solid var(--panel-border); box-shadow: 0 30px 60px rgba(2, 10, 16, 0.24); margin-bottom: 18px;">
		  <div style="font-size: 13px; letter-spacing: 0.14em; text-transform: uppercase; color: var(--hero-kicker); margin-bottom: 10px;">Triton workspace</div>
		  <div style="font-size: 42px; font-weight: 800; line-height: 1.05; margin-bottom: 12px;">Project-shaped audio workflows</div>
		  <div style="max-width: 760px; color: var(--hero-body); font-size: 17px; line-height: 1.6; margin-bottom: 14px;">Start from a project anywhere on disk, keep its audio rules consistent, and let Triton handle import, storage, and processing inside that boundary.</div>
		  <div style="display: inline-block; padding: 8px 14px; border-radius: 999px; background: var(--hero-pill-bg); color: var(--hero-pill-text); font-size: 13px; font-weight: 700;">{status}</div>
		</div>
		""",
		unsafe_allow_html=True,
	)


def _render_project_launcher() -> None:
	st.subheader("Open or create a project")
	st.write(
		"Pick an existing project anywhere on disk or create a new one with the canonical audio settings you want Triton to enforce."
	)

	recent_tab, open_tab, create_tab = st.tabs(["Recent", "Open", "Create"])

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

	with create_tab:
		with st.form("create_project_form"):
			project_name = st.text_input("Project name", value="my-project")
			project_root = st.text_input("Project folder", value=str(Path.home() / "Projects" / "triton" / "my-project"))
			sample_rate = st.selectbox("Sample rate", options=[8000, 16000, 22050, 24000, 32000, 44100, 48000], index=1, key="create_sr")
			channel_mode = st.radio("Channel mode", options=["mono", "stereo"], horizontal=True, key="create_channels")
			with st.expander("Spectrogram defaults", expanded=False):
				spectrogram_settings = _collect_spectrogram_settings("create")
			create_submitted = st.form_submit_button("Create project", type="primary")

		if create_submitted:
			project_dir = Path(project_root).expanduser()
			if project_dir.name != project_name and project_name.strip():
				project_dir = project_dir.parent / project_name.strip()

			try:
				project = create_project(
					project_dir,
					sample_rate=sample_rate,
					channel_mode=channel_mode,
					spectrogram_settings=spectrogram_settings,
				)
				st.session_state["active_project"] = project
				register_recent_project(project_dir, project.name)
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


def _render_file_library(project: Project, project_files: list[Path]) -> None:
	st.markdown("### Import Files")
	st.caption(f"Imported source files are stored in {project_raw_dir(project.path)}")

	add_col, count_col = st.columns([2, 3])
	with add_col:
		with st.form("project_file_upload_form"):
			uploaded_files = st.file_uploader(
				"Import audio files",
				type=["wav", "flac", "ogg", "mp3", "m4a"],
				accept_multiple_files=True,
				key="project_file_upload",
			)
			upload_submitted = st.form_submit_button("Import selected files", type="primary")

		if upload_submitted:
			if not uploaded_files:
				st.error("Choose at least one file to import.")
			else:
				saved_paths = _save_uploaded_project_files(project, list(uploaded_files))
				st.success(f"Imported {len(saved_paths)} file(s) and precomputed spectrograms.")
				st.rerun()

	with count_col:
		st.markdown(
			f"""
			<div style="height: 100%; min-height: 136px; padding: 18px 20px; border-radius: 22px; background: var(--card-top-bg); border: 1px solid var(--panel-border);">
			  <div style="font-size: 12px; text-transform: uppercase; letter-spacing: 0.14em; color: var(--hero-kicker); margin-bottom: 8px;">Import status</div>
			  <div style="font-size: 36px; font-weight: 800; margin-bottom: 6px;">{len(project_files)}</div>
			  <div style="color: var(--card-subtle-text);">file(s) available for playback, spectrogram, and pipelines.</div>
			</div>
			""",
			unsafe_allow_html=True,
		)

	if not project_files:
		st.info("No files have been imported to this project yet.")
		return

	selected_spectrogram = st.session_state.get("selected_spectrogram_file")
	selected_count = 0

	st.markdown("### Imported Files")
	list_col, panel_col = st.columns([1.9, 1.1], gap="large")

	with list_col:
		search_col, sort_col = st.columns([3, 2])
		search_text = search_col.text_input("Search files", value="", placeholder="Type a file name...")
		sort_mode = sort_col.selectbox("Sort", options=["name", "size_desc", "size_asc"], format_func=lambda value: {
			"name": "Name (A-Z)",
			"size_desc": "Size (largest first)",
			"size_asc": "Size (smallest first)",
		}[value])

		visible_files = [path for path in project_files if search_text.strip().lower() in path.name.lower()]
		if sort_mode == "size_desc":
			visible_files = sorted(visible_files, key=lambda path: path.stat().st_size, reverse=True)
		elif sort_mode == "size_asc":
			visible_files = sorted(visible_files, key=lambda path: path.stat().st_size)

		if not visible_files:
			st.info("No files match your search.")
		else:
			for index, file_path in enumerate(visible_files):
				spec_path = _spectrogram_path(file_path)
				check_key = f"import_checked_{_pipeline_key(str(file_path))}"
				with st.container(border=True):
					line1_check_col, line1_name_col, line1_player_col, line1_buttons_col = st.columns([0.8, 3.0, 2.8, 3.4])
					with line1_check_col:
						checked = st.checkbox("Select file", key=check_key, label_visibility="collapsed")
						if checked:
							selected_count += 1
					with line1_name_col:
						new_name = st.text_input(
							"Rename file",
							value=file_path.name,
							key=f"rename_input_{index}",
							label_visibility="collapsed",
						)
					with line1_player_col:
						st.audio(str(file_path), format="audio/wav")
					with line1_buttons_col:
						spect_col, rename_col, remove_col = st.columns(3)
						if spect_col.button("Spec", key=f"list_spec_{index}"):
							if not spec_path.exists():
								try:
									_generate_file_spectrogram(file_path, project)
								except Exception as exc:
									st.error(f"Could not generate spectrogram for {file_path.name}: {exc}")
								else:
									st.session_state["selected_spectrogram_file"] = str(file_path)
									st.rerun()
							else:
								st.session_state["selected_spectrogram_file"] = str(file_path)
								st.rerun()
						if rename_col.button("Rename", key=f"rename_btn_{index}"):
							try:
								renamed = _rename_project_file(file_path, new_name)
							except Exception as exc:
								st.error(f"Could not rename {file_path.name}: {exc}")
							else:
								if spec_path.exists():
									_spectrogram_path(file_path).rename(_spectrogram_path(renamed))
								if st.session_state.get("selected_spectrogram_file") == str(file_path):
									st.session_state["selected_spectrogram_file"] = str(renamed)
								st.rerun()
						if remove_col.button("Remove", key=f"remove_file_{index}"):
							_delete_project_file(file_path)
							if spec_path.exists():
								spec_path.unlink()
							if st.session_state.get("selected_spectrogram_file") == str(file_path):
								st.session_state.pop("selected_spectrogram_file", None)
							st.session_state.pop(check_key, None)
							st.rerun()

					line2_path_col, line2_details_col = st.columns([7, 3])
					with line2_path_col:
						st.caption(str(file_path))
					with line2_details_col:
						st.caption(f"{_format_file_size(file_path.stat().st_size)} | {file_path.suffix.lower()}")

	with panel_col:
		with st.container(border=True):
			st.markdown("### Spectrogram")
			selected_path = next((path for path in project_files if str(path) == str(selected_spectrogram)), None)
			if selected_path is None:
				st.caption("Click Spec on a file to open its spectrogram here.")
			else:
				spec_path = _spectrogram_path(selected_path)
				st.caption(selected_path.name)
				if not spec_path.exists():
					st.warning("No spectrogram found for this file. Click Spec again to generate it.")
				else:
					try:
						result, _ = load_spectrogram(spec_path)
					except Exception as exc:
						st.error(f"Could not load spectrogram for {selected_path.name}: {exc}")
					else:
						times = np.asarray(result.times, dtype=np.float32)
						freqs = np.asarray(result.freqs, dtype=np.float32)
						if result.values.size == 0 or times.size == 0 or freqs.size == 0:
							st.warning("Spectrogram data is empty.")
						else:
							values = np.asarray(result.values, dtype=np.float32)
							# Keep the interactive chart responsive by capping rendered cells.
							max_time_bins = 700
							max_freq_bins = 256
							time_step = max(1, int(np.ceil(values.shape[1] / max_time_bins)))
							freq_step = max(1, int(np.ceil(values.shape[0] / max_freq_bins)))

							if time_step > 1 or freq_step > 1:
								plot_values = values[::freq_step, ::time_step]
								plot_times = times[::time_step]
								plot_freqs = freqs[::freq_step]
								st.caption(
									f"Interactive preview downsampled: time x{time_step}, frequency x{freq_step}"
								)
							else:
								plot_values = values
								plot_times = times
								plot_freqs = freqs

							fig = go.Figure(
								data=go.Heatmap(
									z=plot_values,
									x=plot_times,
									y=plot_freqs,
									colorscale="Gray",
									zmin=-80.0,
									zmax=0.0,
									colorbar={"title": "dB"},
								)
							)
							fig.update_layout(
								title=f"{result.kind.upper()} Spectrogram",
								xaxis_title="Time (s)",
								yaxis_title="Frequency (Hz)",
								plot_bgcolor="rgba(16, 41, 53, 1)",
								paper_bgcolor="rgba(0, 0, 0, 0)",
								font={"color": "#e6f2f2"},
								margin={"l": 60, "r": 20, "t": 40, "b": 50},
							)
							fig.update_xaxes(showgrid=False)
							fig.update_yaxes(showgrid=False)
							st.plotly_chart(
								fig,
								width="stretch",
								config={"scrollZoom": True, "displaylogo": False},
							)

	st.caption(f"Selected in list: {selected_count}")


def _pipeline_key(name: str) -> str:
	return "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in name.strip().lower())


def _pipeline_output_dir(project: Project, pipeline_name: str) -> Path:
	key = _pipeline_key(pipeline_name)
	if not key:
		key = "pipeline"
	return project.path / "data" / "derived" / "pipelines" / key


def _pipeline_run_dir(project: Project, pipeline_name: str, run_id: str) -> Path:
	return _pipeline_output_dir(project, pipeline_name) / f"run_{run_id}"


def _new_pipeline_run_id() -> str:
	timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
	return f"{timestamp}_{uuid4().hex[:8]}"


def _ensure_2d(audio: np.ndarray) -> np.ndarray:
	if audio.ndim == 1:
		return audio[:, np.newaxis]
	return audio


def _to_sample_major(audio: np.ndarray) -> np.ndarray:
	arr = np.asarray(audio, dtype=np.float32)
	if arr.ndim == 2 and arr.shape[0] <= 8 and arr.shape[1] > arr.shape[0]:
		return arr.T
	return arr


def _apply_pipeline_step(
	audio: np.ndarray,
	sr: int,
	step: str,
	project: Project,
	step_options: dict[str, object] | None = None,
) -> tuple[np.ndarray, int]:
	options = step_options or {}

	if step == "normalize":
		target_peak = float(options.get("target_peak", 0.99))
		target_peak = min(max(target_peak, 0.01), 1.0)
		return normalize_peak(audio, target=target_peak), sr

	if step == "resample_project":
		target_mode = str(options.get("target_mode", "project"))
		if target_mode == "custom":
			target_sr = int(options.get("custom_sr", int(project.sample_rate)))
		else:
			target_sr = int(project.sample_rate)
		if target_sr <= 0:
			raise ValueError("Resample target sample rate must be positive.")

		audio_2d = _ensure_2d(audio)
		resampled = _resample_audio(audio_2d, source_sr=sr, target_sr=target_sr)
		if audio.ndim == 1:
			return resampled[:, 0], target_sr
		return resampled, target_sr

	if step == "to_mono":
		audio_2d = _ensure_2d(audio)
		return _convert_channels(audio_2d, channel_mode="mono"), sr

	if step == "to_stereo":
		if audio.ndim == 1:
			input_audio = audio[:, np.newaxis]
		else:
			input_audio = audio
		return _convert_channels(input_audio, channel_mode="stereo"), sr

	if step == "requantize_16":
		bit_depth = int(options.get("bit_depth", 16))
		if bit_depth not in {8, 16, 24, 32}:
			raise ValueError(f"Unsupported bit depth: {bit_depth}")
		return requantize(audio, bit_depth=bit_depth), sr

	if step == "vocode_noise":
		n_bands = int(options.get("n_bands", 8))
		envelope_cutoff = float(options.get("envelope_cutoff", 160.0))
		vocoder_type = str(options.get("vocoder_type", "noise"))
		mono = audio if audio.ndim == 1 else np.mean(audio, axis=1, dtype=np.float32)
		return noise_vocode(
			mono,
			sr,
			n_bands=n_bands,
			envelope_cutoff=envelope_cutoff,
			vocoder_type=vocoder_type,
		), sr

	raise ValueError(f"Unsupported pipeline step: {step}")


def _run_pipeline_on_file(file_path: Path, project: Project, pipeline: Pipeline, run_dir: Path) -> Path:
	audio, sr = load_audio(file_path, sr=None, mono=False)
	processed = _to_sample_major(audio)
	current_sr = int(sr)
	action_history: list[dict[str, object]] = []
	final_output: Path | None = None

	if not pipeline.steps:
		raise ValueError("Pipeline contains no steps.")

	for step_index, step in enumerate(pipeline.steps):
		step_options = pipeline.step_options.get(str(step_index), pipeline.step_options.get(step, {}))
		input_sr = current_sr
		processed, current_sr = _apply_pipeline_step(processed, current_sr, step, project, step_options)

		step_dir = run_dir / f"step_{step_index + 1:02d}_{_pipeline_key(step) or 'step'}"
		output_path = step_dir / f"{file_path.stem}.wav"
		action_history.append(
			{
				"step_index": step_index + 1,
				"step": step,
				"options": step_options,
				"input_sample_rate": int(input_sr),
				"output_sample_rate": int(current_sr),
			}
		)
		save_audio(
			output_path,
			processed,
			current_sr,
			source={"path": str(file_path.resolve())},
			actions=list(action_history),
			extra={
				"pipeline": {
					"name": pipeline.name,
					"run_id": run_dir.name,
				},
			},
		)
		final_output = output_path

	if final_output is None:
		raise RuntimeError("Pipeline did not produce an output file.")
	return final_output


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

	st.caption("No options for this step.")
	return {}


def _render_pipelines_tab(project: Project, project_files: list[Path]) -> None:
	pipelines = _load_pipelines(project)
	pipeline_names = [item.name for item in pipelines]

	pending_selected_pipeline = st.session_state.pop("pending_selected_pipeline_name", None)
	if pending_selected_pipeline is not None:
		if pending_selected_pipeline == "__clear__":
			st.session_state.pop("selected_pipeline_name", None)
		elif pending_selected_pipeline in set(pipeline_names):
			st.session_state["selected_pipeline_name"] = pending_selected_pipeline

	if pipeline_names and st.session_state.get("selected_pipeline_name") not in set(pipeline_names):
		st.session_state["selected_pipeline_name"] = pipeline_names[0]

	st.markdown("### Pipelines")
	st.write("Default view shows your pipeline list. Pick one to run or edit, or create a new pipeline from the right panel.")

	main_col, editor_col = st.columns([1.9, 1.1], gap="large")

	with main_col:
		action_col1, action_col2 = st.columns(2)
		if action_col1.button("New pipeline", type="primary"):
			_open_pipeline_editor("create", project)
			st.rerun()

		selected_pipeline: Pipeline | None = None
		if pipeline_names:
			selected_name = st.radio(
				"Created pipelines",
				options=pipeline_names,
				key="selected_pipeline_name",
			)
			selected_pipeline = next((item for item in pipelines if item.name == selected_name), None)

			if action_col2.button("Edit selected"):
				if selected_pipeline is not None:
					_open_pipeline_editor("edit", project, selected_pipeline)
					st.rerun()
		else:
			action_col2.button("Edit selected", disabled=True)
			st.info("No pipelines yet. Click New pipeline to create your first one.")

		if selected_pipeline is not None:
			st.markdown(f"#### {selected_pipeline.name}")
			for index, step in enumerate(selected_pipeline.steps, start=1):
				st.caption(f"{index}. {_pipeline_action_label(step)} ({step})")

			run_files = st.multiselect(
				"Files to run",
				options=[path.name for path in project_files],
				help="Choose one or more project files for this pipeline.",
			)

			run_col, delete_col = st.columns(2)
			if run_col.button("Run selected pipeline", type="primary"):
				if not run_files:
					st.error("Select at least one file.")
				else:
					selected_paths = [path for path in project_files if path.name in set(run_files)]
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
						log_project_event(
							project.path,
							"pipeline_run_completed",
							{
								"pipeline": selected_pipeline.name,
								"run_id": run_id,
								"requested_files": len(selected_paths),
								"succeeded": len(successes),
								"failed": len(errors),
							},
						)
					if errors:
						for error in errors:
							st.error(error)

			if delete_col.button("Delete selected"):
				remaining = [item for item in pipelines if item.name != selected_pipeline.name]
				_save_pipelines(project, remaining)
				if remaining:
					st.session_state["pending_selected_pipeline_name"] = remaining[0].name
				else:
					st.session_state["pending_selected_pipeline_name"] = "__clear__"
				_close_pipeline_editor()
				st.rerun()

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
					step_count = int(
						st.number_input(
							"Number of steps",
							min_value=1,
							max_value=12,
							value=int(st.session_state.get("pipeline_editor_step_count", 1)),
							step=1,
							key="pipeline_editor_step_count",
						)
					)

				st.caption("Choose an action for each step, then configure options under it.")

				steps: list[str] = []
				step_options_by_index: dict[str, dict[str, object]] = {}

				for order_index in range(step_count):
					with st.container(border=True):
						step_col, delete_col = st.columns([4, 1])
						with step_col:
							step = st.selectbox(
								f"Step {order_index + 1}",
								options=PIPELINE_STEP_ORDER,
								format_func=lambda action: f"{_pipeline_action_label(action)} ({action})",
								key=f"pipeline_editor_step_{order_index}",
							)
						with delete_col:
							st.button(
								"Delete",
								key=f"pipeline_editor_delete_step_{order_index}",
								disabled=step_count <= 1,
								on_click=_request_delete_pipeline_step,
								args=(order_index,),
								width="stretch",
							)
						steps.append(step)
						st.caption("Step options")
						options = _render_step_options_editor(step, order_index, project)
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

	import_tab, pipelines_tab, mix_tab, roadmap_tab = st.tabs(["Import", "Pipelines", "Mix", "Roadmap"])

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

		_render_file_library(project, project_files)

	with pipelines_tab:
		_render_pipelines_tab(project, project_files)

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