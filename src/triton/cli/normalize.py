"""Audio normalization commands."""

from __future__ import annotations

from pathlib import Path

import typer

from triton.core.io import (
	iter_audio_files,
	load_audio,
	save_audio,
	normalize_peak,
	normalize_rms,
)


normalize_app = typer.Typer(add_completion=False, help="Normalize audio amplitude")


@normalize_app.command("peak")
def normalize_peak_cmd(
	input_path: Path = typer.Argument(..., help="Audio file or directory"),
	output_dir: Path = typer.Option(Path("outputs/normalized"), help="Output directory"),
	target: float = typer.Option(
		0.99,
		help="Target peak amplitude (0.0-1.0)",
		min=0.0,
		max=1.0,
	),
):
	"""Normalize audio to target peak amplitude.
	
	Preserves dynamics while preventing clipping. Useful for safety headroom
	before mixing or processing.
	"""
	input_path = input_path.expanduser().resolve()
	output_dir = output_dir.expanduser().resolve()
	output_dir.mkdir(parents=True, exist_ok=True)

	try:
		files = list(iter_audio_files(input_path))
	except ValueError as exc:
		raise typer.BadParameter(str(exc)) from exc

	for audio_path in files:
		audio, sr = load_audio(audio_path, sr=None, mono=False)
		normalized = normalize_peak(audio, target=target)

		rel_name = audio_path.name if input_path.is_file() else audio_path.relative_to(input_path)
		out_path = output_dir / rel_name
		save_audio(out_path, normalized, sr)
		typer.echo(f"Wrote {out_path}")


@normalize_app.command("rms")
def normalize_rms_cmd(
	input_path: Path = typer.Argument(..., help="Audio file or directory"),
	output_dir: Path = typer.Option(Path("outputs/normalized"), help="Output directory"),
	target: float = typer.Option(
		0.1,
		help="Target RMS amplitude (0.0-1.0)",
		min=0.0,
		max=1.0,
	),
):
	"""Normalize audio to target RMS (energy) level.
	
	RMS normalization adjusts audio loudness by normalizing the root mean square
	amplitude. This is duration-independent (works equally well for short and long
	files) and is ideal for ensuring consistent loudness across files before
	degradation, mixing, or transcription.
	
	Common use cases:
	- Preparing audio for noise mixing (control SNR consistently)
	- Equalizing loudness across files before processing
	- Speech processing pipelines
	"""
	input_path = input_path.expanduser().resolve()
	output_dir = output_dir.expanduser().resolve()
	output_dir.mkdir(parents=True, exist_ok=True)

	try:
		files = list(iter_audio_files(input_path))
	except ValueError as exc:
		raise typer.BadParameter(str(exc)) from exc

	for audio_path in files:
		audio, sr = load_audio(audio_path, sr=None, mono=False)
		normalized = normalize_rms(audio, target=target)

		rel_name = audio_path.name if input_path.is_file() else audio_path.relative_to(input_path)
		out_path = output_dir / rel_name
		save_audio(out_path, normalized, sr)
		typer.echo(f"Wrote {out_path}")
