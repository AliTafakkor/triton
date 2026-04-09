"""Shared pipeline runtime helpers for GUI and CLI."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import librosa
import numpy as np

from triton.core.conversion import requantize
from triton.core.io import load_audio, normalize_peak, normalize_rms, save_audio
from triton.core.project import Pipeline, Project
from triton.core.ramp import apply_ramp
from triton.degrade.time_compression import compress_time
from triton.degrade.vocoder import noise_vocode


PIPELINE_ACTIONS: dict[str, str] = {
	"normalize": "Peak normalize",
	"normalize_rms": "RMS normalize",
	"resample_project": "Resample to project sample rate",
	"to_mono": "Convert to mono",
	"to_stereo": "Convert to stereo",
	"requantize_16": "Requantize to 16-bit",
	"vocode_noise": "Noise vocoder degradation",
	"time_compress": "Time compression (Praat/Parselmouth)",
	"ramp": "Fade in / Fade out (ramp)",
}

PIPELINE_STEP_ORDER = list(PIPELINE_ACTIONS.keys())
PIPELINE_DEFAULT_STEP = "normalize"


def pipeline_action_label(action: str) -> str:
	return PIPELINE_ACTIONS.get(action, action)


def default_step_options(step: str, project_sr: int) -> dict[str, object]:
	if step == "normalize":
		return {"target_peak": 0.99}
	if step == "normalize_rms":
		return {"target_rms": 0.1}
	if step == "resample_project":
		return {"target_mode": "project", "custom_sr": int(project_sr)}
	if step == "requantize_16":
		return {"bit_depth": 16}
	if step == "vocode_noise":
		return {"n_bands": 8, "vocoder_type": "noise", "envelope_cutoff": 160.0}
	if step == "time_compress":
		return {"factor": 1.0}
	if step == "ramp":
		return {"ramp_start": 0.05, "ramp_end": 0.05, "shape": "cosine"}
	return {}


def pipeline_key(name: str) -> str:
	return "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in name.strip().lower())


def pipeline_output_dir(project: Project, pipeline_name: str) -> Path:
	key = pipeline_key(pipeline_name)
	if not key:
		key = "pipeline"
	return project.path / "data" / "derived" / "pipelines" / key


def pipeline_run_dir(project: Project, pipeline_name: str, run_id: str) -> Path:
	return pipeline_output_dir(project, pipeline_name) / f"run_{run_id}"


def new_pipeline_run_id() -> str:
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


def _resample_audio(audio: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
	if source_sr == target_sr:
		return audio

	channels = [
		librosa.resample(audio[:, channel], orig_sr=source_sr, target_sr=target_sr)
		for channel in range(audio.shape[1])
	]
	return np.stack(channels, axis=1).astype(np.float32)


def _convert_channels(audio: np.ndarray, channel_mode: str) -> np.ndarray:
	if channel_mode == "mono":
		return np.mean(audio, axis=1, dtype=np.float32)

	if audio.shape[1] == 1:
		return np.repeat(audio, 2, axis=1).astype(np.float32)

	return audio.astype(np.float32)


def apply_pipeline_step(
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

	if step == "normalize_rms":
		target_rms = float(options.get("target_rms", 0.1))
		target_rms = min(max(target_rms, 0.001), 1.0)
		return normalize_rms(audio, target=target_rms), sr

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

	if step == "time_compress":
		factor = float(options.get("factor", 1.0))
		if factor <= 0:
			raise ValueError("Compression factor must be positive.")
		mono = audio if audio.ndim == 1 else np.mean(audio, axis=1, dtype=np.float32)
		return compress_time(mono, sr, factor=factor), sr

	if step == "ramp":
		ramp_start = float(options.get("ramp_start", 0.05))
		ramp_end = float(options.get("ramp_end", 0.05))
		shape = str(options.get("shape", "cosine"))
		return apply_ramp(audio, sr, ramp_start=ramp_start, ramp_end=ramp_end, shape=shape), sr

	raise ValueError(f"Unsupported pipeline step: {step}")


def run_pipeline_on_file(file_path: Path, project: Project, pipeline: Pipeline, run_dir: Path) -> Path:
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
		processed, current_sr = apply_pipeline_step(processed, current_sr, step, project, step_options)

		step_dir = run_dir / f"step_{step_index + 1:02d}_{pipeline_key(step) or 'step'}"
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
