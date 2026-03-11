"""Degradation commands."""

from __future__ import annotations

from pathlib import Path

import typer

from triton.degrade.vocoder import noise_vocode
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
