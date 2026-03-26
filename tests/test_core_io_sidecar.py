from __future__ import annotations

import json

import numpy as np

from triton.core.io import save_audio, sidecar_path


def test_save_audio_writes_sidecar(tmp_path) -> None:
	out_path = tmp_path / "artifact.wav"
	audio = (0.1 * np.sin(2 * np.pi * 440 * np.arange(1600, dtype=np.float32) / 16000)).astype(np.float32)

	save_audio(
		out_path,
		audio,
		16000,
		source={"path": "/tmp/source.wav"},
		actions=[{"step": "normalize", "options": {"target_peak": 0.9}}],
	)

	meta_path = sidecar_path(out_path)
	assert out_path.exists()
	assert meta_path.exists()

	payload = json.loads(meta_path.read_text(encoding="utf-8"))
	assert payload["artifact"]["name"] == "artifact.wav"
	assert payload["source"]["path"] == "/tmp/source.wav"
	assert payload["actions"][0]["step"] == "normalize"
	assert payload["extra"]["audio"]["sample_rate"] == 16000
