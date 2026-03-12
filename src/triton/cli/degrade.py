"""Degradation commands."""

from __future__ import annotations

from pathlib import Path

import typer

from triton.degrade.vocoder import noise_vocode
from triton.degrade.speech_noise import (
	speech_shaped_noise,
	speech_correlated_noise,
)
from triton.core.io import iter_audio_files, load_audio, save_audio


degrade_app = typer.Typer(add_completion=False, help="Apply audio degradations")


@degrade_app.command("vocode")
def vocode(
	input_path: Path = typer.Argument(..., help="Audio file or directory"),
	output_dir: Path = typer.Option(Path("outputs/vocoded"), help="Output directory"),
	vocoder_type: str = typer.Option("noise", help="Vocoder type: noise or sine"),
	n_bands: int = typer.Option(8, help="Number of frequency bands"),
	freq_low: float = typer.Option(200.0, help="Low frequency cutoff (Hz)"),
	freq_high: float = typer.Option(8000.0, help="High frequency cutoff (Hz)"),
	envelope_cutoff: float = typer.Option(160.0, help="Envelope cutoff (Hz)"),
	filter_order: int = typer.Option(3, help="Butterworth filter order"),
):
	"""Apply channel vocoding (Shannon et al., 1995) to audio files."""
	input_path = input_path.expanduser().resolve()
	output_dir = output_dir.expanduser().resolve()
	output_dir.mkdir(parents=True, exist_ok=True)

	try:
		files = list(iter_audio_files(input_path))
	except ValueError as exc:
		raise typer.BadParameter(str(exc)) from exc

	for audio_path in files:
		audio, sr = load_audio(audio_path, sr=None, mono=True)
		vocoded = noise_vocode(
			audio,
			sr,
			n_bands=n_bands,
			freq_range=(freq_low, freq_high),
			envelope_cutoff=envelope_cutoff,
			vocoder_type=vocoder_type,
			filter_order=filter_order,
		)

		rel_name = audio_path.name if input_path.is_file() else audio_path.relative_to(input_path)
		out_path = output_dir / rel_name
		save_audio(out_path, vocoded, sr)
		typer.echo(f"Wrote {out_path}")


@degrade_app.command("ssn")
def gen_speech_shaped_noise(
	duration: float = typer.Argument(..., help="Duration in seconds"),
	output_path: Path = typer.Argument(..., help="Output audio file path"),
	sample_rate: int = typer.Option(16000, "--sr", help="Sample rate (Hz)"),
	spectrum: str = typer.Option("ltass", help="Spectrum type: ltass or flat"),
):
	"""Generate speech-shaped noise (SSN) - Byrne et al., 1994."""
	output_path = output_path.expanduser().resolve()
	output_path.parent.mkdir(parents=True, exist_ok=True)

	noise = speech_shaped_noise(duration, sample_rate, spectrum=spectrum, normalize=True)
	save_audio(output_path, noise, sample_rate)
	typer.echo(f"Generated {duration}s speech-shaped noise at {output_path}")


@degrade_app.command("scn")
def gen_speech_correlated_noise(
	input_path: Path = typer.Argument(..., help="Speech audio file or directory"),
	output_dir: Path = typer.Option(
		Path("outputs/speech_noise"), help="Output directory"
	),
	method: str = typer.Option(
		"spectrum_match",
		help="Method: spectrum_match or modulation",
	),
	frame_length: int = typer.Option(2048, help="FFT frame length"),
	hop_length: int | None = typer.Option(None, help="Hop length (default: frame_length/4)"),
):
	"""Generate speech-correlated noise (SCN) from speech signal."""
	input_path = input_path.expanduser().resolve()
	output_dir = output_dir.expanduser().resolve()
	output_dir.mkdir(parents=True, exist_ok=True)

	if not input_path.exists():
		raise typer.BadParameter("Input path does not exist.")

	files = list(iter_audio_files(input_path))
	if not files:
		raise typer.BadParameter("No audio files found.")

	for audio_path in files:
		speech, sr = load_audio(audio_path, sr=None, mono=True)
		noise = speech_correlated_noise(
			speech,
			sr,
			method=method,
			frame_length=frame_length,
			hop_length=hop_length,
			normalize=True,
		)

		# Save with _noise suffix
		rel_name = audio_path.name if input_path.is_file() else audio_path.relative_to(input_path)
		stem = rel_name.stem if hasattr(rel_name, "stem") else Path(str(rel_name)).stem
		suffix = rel_name.suffix if hasattr(rel_name, "suffix") else Path(str(rel_name)).suffix
		out_name = f"{stem}_noise{suffix}"
		out_path = output_dir / out_name
		save_audio(out_path, noise, sr)
		typer.echo(f"Wrote {out_path}")

