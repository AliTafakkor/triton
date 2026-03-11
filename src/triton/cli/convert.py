"""Audio format conversion commands."""

from __future__ import annotations

from pathlib import Path

import typer

from triton.core.io import iter_audio_files, load_audio, save_audio
from triton.core.conversion import to_mono, to_stereo, resample, requantize


convert_app = typer.Typer(add_completion=False, help="Convert audio formats")


@convert_app.command("mono")
def convert_mono(
	input_path: Path = typer.Argument(..., help="Audio file or directory"),
	output_dir: Path = typer.Option(Path("outputs/mono"), help="Output directory"),
	method: str = typer.Option("mean", help="Conversion method: mean, left, right"),
):
	"""Convert stereo audio to mono."""
	input_path = input_path.expanduser().resolve()
	output_dir = output_dir.expanduser().resolve()
	output_dir.mkdir(parents=True, exist_ok=True)

	try:
		files = list(iter_audio_files(input_path))
	except ValueError as exc:
		raise typer.BadParameter(str(exc)) from exc

	for audio_path in files:
		audio, sr = load_audio(audio_path, sr=None, mono=False)
		mono_audio = to_mono(audio, method=method)

		rel_name = audio_path.name if input_path.is_file() else audio_path.relative_to(input_path)
		out_path = output_dir / rel_name

		save_audio(out_path, mono_audio, sr)
		typer.echo(f"Wrote {out_path}")


@convert_app.command("stereo")
def convert_stereo(
	input_path: Path = typer.Argument(..., help="Audio file or directory"),
	output_dir: Path = typer.Option(Path("outputs/stereo"), help="Output directory"),
	method: str = typer.Option("duplicate", help="Conversion method: duplicate, silence"),
):
	"""Convert mono audio to stereo."""
	input_path = input_path.expanduser().resolve()
	output_dir = output_dir.expanduser().resolve()
	output_dir.mkdir(parents=True, exist_ok=True)

	try:
		files = list(iter_audio_files(input_path))
	except ValueError as exc:
		raise typer.BadParameter(str(exc)) from exc

	for audio_path in files:
		audio, sr = load_audio(audio_path, sr=None, mono=False)
		stereo_audio = to_stereo(audio, method=method)

		rel_name = audio_path.name if input_path.is_file() else audio_path.relative_to(input_path)
		out_path = output_dir / rel_name

		save_audio(out_path, stereo_audio, sr)
		typer.echo(f"Wrote {out_path}")


@convert_app.command("resample")
def convert_resample(
	input_path: Path = typer.Argument(..., help="Audio file or directory"),
	target_sr: int = typer.Option(..., help="Target sample rate (Hz)"),
	output_dir: Path = typer.Option(Path("outputs/resampled"), help="Output directory"),
):
	"""Resample audio to a target sample rate."""
	input_path = input_path.expanduser().resolve()
	output_dir = output_dir.expanduser().resolve()
	output_dir.mkdir(parents=True, exist_ok=True)

	try:
		files = list(iter_audio_files(input_path))
	except ValueError as exc:
		raise typer.BadParameter(str(exc)) from exc

	for audio_path in files:
		audio, orig_sr = load_audio(audio_path, sr=None, mono=True)
		resampled_audio = resample(audio, orig_sr, target_sr)

		rel_name = audio_path.name if input_path.is_file() else audio_path.relative_to(input_path)
		out_path = output_dir / rel_name

		save_audio(out_path, resampled_audio, target_sr)
		typer.echo(f"Wrote {out_path}")


@convert_app.command("quantize")
def convert_quantize(
	input_path: Path = typer.Argument(..., help="Audio file or directory"),
	bit_depth: int = typer.Option(..., help="Target bit depth: 8, 16, 24, 32"),
	output_dir: Path = typer.Option(Path("outputs/quantized"), help="Output directory"),
):
	"""Change bit depth (quantization) of audio."""
	input_path = input_path.expanduser().resolve()
	output_dir = output_dir.expanduser().resolve()
	output_dir.mkdir(parents=True, exist_ok=True)

	try:
		files = list(iter_audio_files(input_path))
	except ValueError as exc:
		raise typer.BadParameter(str(exc)) from exc

	for audio_path in files:
		audio, sr = load_audio(audio_path, sr=None, mono=True)
		quantized_audio = requantize(audio, bit_depth)

		rel_name = audio_path.name if input_path.is_file() else audio_path.relative_to(input_path)
		out_path = output_dir / rel_name

		save_audio(out_path, quantized_audio, sr)
		typer.echo(f"Wrote {out_path}")
