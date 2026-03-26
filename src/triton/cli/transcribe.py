"""Transcription commands."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from triton.transcribe.whisper import transcribe_file
from triton.core.io import iter_audio_files, write_sidecar


transcribe_app = typer.Typer(add_completion=False, help="Transcribe audio locally")


@transcribe_app.command("local")
def transcribe_local(
	input_path: Path = typer.Argument(..., help="Audio file or directory"),
	output_dir: Path = typer.Option(Path("outputs/transcripts"), help="Output directory"),
	model: str = typer.Option("small", help="Whisper model size"),
	device: str = typer.Option("auto", help="Device: auto|cpu|cuda"),
	compute_type: str = typer.Option("auto", help="Compute type for Whisper"),
	language: str | None = typer.Option(None, help="Language code (e.g., en)"),
	beam_size: int = typer.Option(5, help="Beam size"),
	vad_filter: bool = typer.Option(True, help="Enable VAD filter"),
	write_json: bool = typer.Option(False, help="Write JSON with segments"),
):
	"""Transcribe audio using faster-whisper locally."""
	input_path = input_path.expanduser().resolve()
	output_dir = output_dir.expanduser().resolve()
	output_dir.mkdir(parents=True, exist_ok=True)

	try:
		files = list(iter_audio_files(input_path))
	except ValueError as exc:
		raise typer.BadParameter(str(exc)) from exc

	for audio_path in files:
		result = transcribe_file(
			audio_path,
			model_size=model,
			device=device,
			compute_type=compute_type,
			language=language,
			beam_size=beam_size,
			vad_filter=vad_filter,
		)

		rel_name = audio_path.name if input_path.is_file() else audio_path.relative_to(input_path)
		out_path = output_dir / rel_name
		out_path = out_path.with_suffix(".txt")
		out_path.parent.mkdir(parents=True, exist_ok=True)

		out_path.write_text(result.text, encoding="utf-8")
		write_sidecar(
			out_path,
			source={"path": str(audio_path.resolve())},
			actions=[
				{
					"step": "transcribe_local",
					"options": {
						"model": model,
						"device": device,
						"compute_type": compute_type,
						"language": language,
						"beam_size": beam_size,
						"vad_filter": vad_filter,
					},
				}
			],
		)
		typer.echo(f"Wrote {out_path}")

		if write_json:
			json_path = out_path.with_suffix(".json")
			payload = {
				"language": result.language,
				"text": result.text,
				"segments": [
					{"start": s.start, "end": s.end, "text": s.text}
					for s in result.segments
				],
			}
			json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
			write_sidecar(
				json_path,
				source={"path": str(audio_path.resolve())},
				actions=[
					{
						"step": "transcribe_local_json",
						"options": {
							"model": model,
							"device": device,
							"compute_type": compute_type,
							"language": language,
							"beam_size": beam_size,
							"vad_filter": vad_filter,
						},
					}
				],
			)
			typer.echo(f"Wrote {json_path}")
