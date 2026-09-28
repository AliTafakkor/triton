"""Feature extraction tab for the Triton GUI."""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from triton.core.project import Project, log_project_event
from triton.features.storage import feature_path, list_feature_files, load_feature, save_feature


@st.cache_resource(show_spinner="Loading speech model...")
def _load_neural_model(key: str):
	from triton.features.neural import load_model

	return load_model(key)


@st.cache_resource(show_spinner="Loading Whisper model...")
def _load_whisper(model_size: str):
	from triton.features.words import load_whisper

	return load_whisper(model_size)


def render_features_tab(project: Project, project_files: list[Path]) -> None:
	from triton.features.neural import FRAME_RATE, MODELS

	st.markdown("### Extract Features")
	st.write(
		"Compute time-aligned features for relating audio to EEG/MEG recordings. "
		"Each feature is saved as an `.npz` file (with a `times` array in seconds) under "
		"`data/derived/features/<file>/`, with a provenance sidecar next to it."
	)

	if not project_files:
		st.info("Import some audio files first to extract features.")
		return

	selected_names = st.multiselect(
		"Files",
		options=[p.name for p in project_files],
		key="features_files",
	)

	st.markdown("#### Features")
	env_col, words_col, neural_col = st.columns(3)

	with env_col:
		with st.container(border=True):
			do_envelope = st.checkbox("Amplitude envelope", value=True, key="features_do_envelope")
			env_rate = int(st.number_input(
				"Output rate (Hz)", min_value=10, max_value=1000, value=100, step=10,
				key="features_env_rate", disabled=not do_envelope,
				help="Rate of the saved envelope. 100 Hz is typical for EEG/MEG analysis.",
			))
			env_cutoff = float(st.number_input(
				"Low-pass cutoff (Hz)", min_value=1.0, max_value=500.0, value=30.0,
				step=1.0, key="features_env_cutoff", disabled=not do_envelope,
				help="Smoothing applied before resampling. Keep it below half the output rate.",
			))

	with words_col:
		with st.container(border=True):
			do_words = st.checkbox("Word onsets", value=False, key="features_do_words")
			whisper_size = st.selectbox(
				"Whisper model", options=["tiny", "base", "small", "medium"], index=2,
				key="features_whisper_size", disabled=not do_words,
				help="Word timestamps come from Whisper. Larger models are more accurate but slower.",
			)
			words_language = st.text_input(
				"Language (optional)", placeholder="e.g. en", key="features_words_lang", disabled=not do_words,
			)
			st.caption("Saves each word's start/end time plus an onset impulse train at the envelope rate.")

	with neural_col:
		with st.container(border=True):
			do_neural = st.checkbox("Neural network layers", value=False, key="features_do_neural")
			model_key = st.selectbox(
				"Model", options=list(MODELS), format_func=lambda key: MODELS[key].label,
				key="features_model", disabled=not do_neural,
			)
			# All offered models are "base" size: 12 transformer layers + the input (layer 0).
			layers = st.multiselect(
				"Layers", options=list(range(13)), default=list(range(13)),
				key="features_layers", disabled=not do_neural,
				help="0 is the input to the first transformer layer; 1-12 are the transformer layer outputs.",
			)
			st.caption(f"One 768-dim vector per layer every {1000 / FRAME_RATE:.0f} ms (~{FRAME_RATE * 768 * 4 / 1e6:.2f} MB per layer per second of audio).")

	nothing_chosen = not (do_envelope or do_words or (do_neural and layers))
	run = st.button("Extract", type="primary", disabled=not selected_names or nothing_chosen, key="features_run")

	if run:
		selected_paths = [p for p in project_files if p.name in set(selected_names)]
		_run_extraction(
			project,
			selected_paths,
			envelope={"rate": env_rate, "cutoff": env_cutoff} if do_envelope else None,
			words={"model_size": whisper_size, "language": words_language.strip() or None, "rate": env_rate} if do_words else None,
			neural={"model": model_key, "layers": sorted(layers)} if do_neural and layers else None,
		)

	_render_saved_features(project)


def _run_extraction(
	project: Project,
	paths: list[Path],
	*,
	envelope: dict | None,
	words: dict | None,
	neural: dict | None,
) -> None:
	from triton.features.acoustic import envelope_feature
	from triton.features.neural import MODELS, TARGET_SR, extract_hidden_states
	from triton.features.words import onset_train, transcribe_words

	saved: list[Path] = []
	errors: list[str] = []
	progress = st.progress(0.0, text="Starting...")

	for index, path in enumerate(paths):
		progress.progress(index / len(paths), text=f"Processing {path.name} ({index + 1}/{len(paths)})")
		try:
			audio, sr = librosa.load(str(path), sr=None, mono=True)
			duration = audio.size / sr

			if envelope:
				times, values = envelope_feature(audio, sr, rate=envelope["rate"], cutoff=envelope["cutoff"])
				saved.append(save_feature(
					project.path, path, "envelope", {"times": times, "envelope": values},
					options={**envelope, "method": "abs(hilbert(x))"},
				))

			if words:
				model = _load_whisper(words["model_size"])
				found = transcribe_words(path, model=model, language=words["language"])
				onsets = np.array([w.start for w in found], dtype=np.float32)
				times, train = onset_train(onsets, duration, words["rate"])
				saved.append(save_feature(
					project.path, path, "word_onsets",
					{
						"words": np.array([w.text for w in found], dtype=str),
						"onsets": onsets,
						"offsets": np.array([w.end for w in found], dtype=np.float32),
						"times": times,
						"onset_train": train,
					},
					options=words,
				))

			if neural:
				extractor, model = _load_neural_model(neural["model"])
				audio16 = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR) if sr != TARGET_SR else audio
				times, states, layer_ids = extract_hidden_states(audio16, extractor=extractor, model=model, layers=neural["layers"])
				saved.append(save_feature(
					project.path, path, neural["model"],
					{"times": times, "hidden_states": states, "layers": np.array(layer_ids, dtype=np.int16)},
					options=neural,
					extra={"model_id": MODELS[neural["model"]].model_id},
				))
		except Exception as exc:
			errors.append(f"{path.name}: {exc}")

	progress.progress(1.0, text="Done")

	if saved:
		log_project_event(
			project.path,
			"features_extracted",
			{
				"files": [p.name for p in paths],
				"envelope": envelope,
				"words": words,
				"neural": neural,
				"outputs": [p.name for p in saved],
			},
		)
		st.success(f"Saved {len(saved)} feature file(s).")
		st.session_state["features_preview_file"] = str(paths[0])
	for error in errors:
		st.error(error)


def _render_saved_features(project: Project) -> None:
	feature_files = list_feature_files(project.path)
	if not feature_files:
		return

	st.markdown("#### Saved features")
	rows = []
	for path in feature_files:
		rows.append({
			"Audio file": path.parent.name,
			"Feature": path.name[len(path.parent.name) + 1:-len(".npz")],
			"Size": f"{path.stat().st_size / 1e6:.2f} MB",
			"Path": str(path.relative_to(project.path)),
		})
	st.dataframe(rows, width="stretch", hide_index=True)

	stems = sorted({path.parent.name for path in feature_files})
	default_stem = Path(st.session_state.get("features_preview_file", "")).stem
	preview_stem = st.selectbox(
		"Preview", options=stems, index=stems.index(default_stem) if default_stem in stems else 0,
		key="features_preview_select",
	)
	_render_preview(project, preview_stem)


def _render_preview(project: Project, stem: str) -> None:
	stand_in = Path(f"{stem}.wav")
	env_path = feature_path(project.path, stand_in, "envelope")
	words_path = feature_path(project.path, stand_in, "word_onsets")

	if env_path.exists() or words_path.exists():
		fig = go.Figure()
		if env_path.exists():
			arrays, _ = load_feature(env_path)
			fig.add_trace(go.Scatter(x=arrays["times"], y=arrays["envelope"], name="Envelope", line={"width": 1.5}))
		if words_path.exists():
			arrays, _ = load_feature(words_path)
			for word, onset in zip(arrays["words"], arrays["onsets"]):
				fig.add_vline(x=float(onset), line={"width": 1, "dash": "dot", "color": "#f4a340"})
				fig.add_annotation(x=float(onset), y=1, yref="paper", text=str(word), showarrow=False, textangle=-60, xanchor="left", yanchor="bottom", font={"size": 10})
		fig.update_layout(
			title="Envelope and word onsets", xaxis_title="Time (s)", yaxis_title="Amplitude",
			paper_bgcolor="rgba(0, 0, 0, 0)", margin={"l": 60, "r": 20, "t": 80, "b": 50}, showlegend=False,
		)
		st.plotly_chart(fig, width="stretch")

	neural_paths = [
		path for path in sorted(env_path.parent.glob(f"{stem}.*.npz"))
		if path.name[len(stem) + 1:-len(".npz")] not in {"envelope", "word_onsets"}
	]
	# Activations can be hundreds of MB, so only load them when asked.
	if not neural_paths or not st.checkbox("Show neural layer activity", key=f"features_show_neural_{stem}"):
		return

	for neural_path in neural_paths:
		kind = neural_path.name[len(stem) + 1:-len(".npz")]
		arrays, meta = load_feature(neural_path)
		states, times, layers = arrays["hidden_states"], arrays["times"], arrays["layers"]
		# Layer-by-time view: how strongly each layer responds over time (L2 norm per frame).
		norms = np.linalg.norm(states, axis=2)
		norms = (norms - norms.mean(axis=1, keepdims=True)) / (norms.std(axis=1, keepdims=True) + 1e-8)
		fig = go.Figure(go.Heatmap(z=norms, x=times, y=[str(int(layer)) for layer in layers], colorscale="Viridis", colorbar={"title": "z"}))
		fig.update_layout(
			title=f"{meta.get('model_id', kind)}: activation strength per layer (z-scored)",
			xaxis_title="Time (s)", yaxis_title="Layer", paper_bgcolor="rgba(0, 0, 0, 0)",
			margin={"l": 60, "r": 20, "t": 50, "b": 50},
		)
		st.plotly_chart(fig, width="stretch")
		st.caption(f"Saved shape: {states.shape[0]} layers × {states.shape[1]} frames × {states.shape[2]} dims")
