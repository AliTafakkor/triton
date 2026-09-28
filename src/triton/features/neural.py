"""Layer-wise activations from pretrained self-supervised speech models.

All supported models share the wav2vec2-style architecture: a convolutional
encoder that emits one frame every 20 ms (320 samples at 16 kHz) followed by a
stack of transformer layers. Hidden state 0 is the input to the first
transformer layer; hidden state ``i`` is the output of transformer layer ``i``.
Comparing layers against brain responses is the approach of Kell et al. (2018)
and Millet et al. (2022).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TARGET_SR = 16000
HOP_SAMPLES = 320
RECEPTIVE_FIELD_SAMPLES = 400
FRAME_RATE = TARGET_SR / HOP_SAMPLES  # 50 Hz


@dataclass(frozen=True)
class NeuralModel:
	label: str
	model_id: str


MODELS: dict[str, NeuralModel] = {
	"wav2vec2": NeuralModel("Wav2Vec2 base (LibriSpeech 960h)", "facebook/wav2vec2-base-960h"),
	"hubert": NeuralModel("HuBERT base (LibriSpeech 960h)", "facebook/hubert-base-ls960"),
	"wavlm": NeuralModel("WavLM base plus", "microsoft/wavlm-base-plus"),
}


def load_model(key: str):
	"""Load the feature extractor and base model for ``key``. Call once and cache."""
	if key not in MODELS:
		raise ValueError(f"Unknown model '{key}'. Choose from: {', '.join(MODELS)}")

	try:
		from triton._torch_compat import patch_torch_compiler

		patch_torch_compiler()
		from transformers import AutoFeatureExtractor, AutoModel
	except ImportError as exc:
		raise ImportError("Neural features need torch and transformers: pip install 'conch-triton[features]'") from exc

	model_id = MODELS[key].model_id
	extractor = AutoFeatureExtractor.from_pretrained(model_id)
	model = AutoModel.from_pretrained(model_id)
	model.eval()
	return extractor, model


def num_hidden_states(model) -> int:
	"""Number of selectable layers (transformer layers + the pre-transformer input)."""
	return int(model.config.num_hidden_layers) + 1


def estimate_size_bytes(duration_s: float, n_layers: int, hidden_size: int) -> int:
	"""Approximate size of the float32 activations saved for one file."""
	return int(duration_s * FRAME_RATE) * n_layers * hidden_size * 4


def extract_hidden_states(
	audio: np.ndarray,
	*,
	extractor,
	model,
	layers: list[int] | None = None,
	chunk_seconds: float = 20.0,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
	"""Run 16 kHz mono ``audio`` through ``model`` and collect hidden states.

	Audio is processed in chunks of ``chunk_seconds`` (a multiple of the 20 ms hop)
	so memory stays bounded on long recordings; self-attention does not see across
	chunk boundaries. A trailing piece shorter than the 25 ms receptive field is
	dropped.

	Returns:
		(times, states, layers): frame-centre times in seconds ``(n_frames,)``,
		activations ``(n_layers, n_frames, hidden_size)`` as float32, and the layer
		indices in the order they appear in ``states``.
	"""
	import torch

	audio = np.asarray(audio, dtype=np.float32)
	if audio.ndim != 1:
		raise ValueError("Audio must be mono (1D) and sampled at 16 kHz.")
	if audio.size < RECEPTIVE_FIELD_SAMPLES:
		raise ValueError("Audio is too short: need at least 25 ms at 16 kHz.")

	total = num_hidden_states(model)
	selected = list(range(total)) if not layers else sorted(set(int(i) for i in layers))
	invalid = [i for i in selected if i < 0 or i >= total]
	if invalid:
		raise ValueError(f"Layer indices {invalid} out of range; model has layers 0-{total - 1}.")

	chunk_samples = max(1, int(round(chunk_seconds * FRAME_RATE))) * HOP_SAMPLES
	time_chunks: list[np.ndarray] = []
	state_chunks: list[np.ndarray] = []

	for start in range(0, audio.size, chunk_samples):
		piece = audio[start:start + chunk_samples]
		if piece.size < RECEPTIVE_FIELD_SAMPLES:
			break
		inputs = extractor(piece, sampling_rate=TARGET_SR, return_tensors="pt")
		with torch.no_grad():
			outputs = model(inputs["input_values"], output_hidden_states=True)
		stacked = torch.stack([outputs.hidden_states[i][0] for i in selected]).cpu().numpy()
		n_frames = stacked.shape[1]
		centres = start + np.arange(n_frames) * HOP_SAMPLES + RECEPTIVE_FIELD_SAMPLES / 2
		time_chunks.append((centres / TARGET_SR).astype(np.float32))
		state_chunks.append(stacked.astype(np.float32))

	return np.concatenate(time_chunks), np.concatenate(state_chunks, axis=1), selected
