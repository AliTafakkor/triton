"""Local transcription using openai-whisper."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import whisper


@dataclass(frozen=True)
class TranscriptSegment:
	start: float
	end: float
	text: str


@dataclass(frozen=True)
class TranscriptResult:
	text: str
	language: str | None
	segments: list[TranscriptSegment]


def transcribe_file(
	path: str | Path,
	*,
	model_size: str = "small",
	device: str = "auto",
	compute_type: str = "auto",
	language: str | None = None,
	beam_size: int = 5,
	vad_filter: bool = True,
) -> TranscriptResult:
	"""Transcribe a single audio file using openai-whisper."""
	use_device = "cpu" if device == "auto" else device
	model = whisper.load_model(model_size, device=use_device)

	fp16 = use_device != "cpu" and compute_type in {"auto", "float16", "fp16"}
	result = model.transcribe(
		str(path),
		language=language,
		beam_size=beam_size,
		fp16=fp16,
	)

	segments: list[TranscriptSegment] = []
	for seg in result.get("segments", []):
		segments.append(
			TranscriptSegment(start=seg["start"], end=seg["end"], text=seg["text"])
		)

	return TranscriptResult(
		text=(result.get("text") or "").strip(),
		language=result.get("language"),
		segments=segments,
	)
