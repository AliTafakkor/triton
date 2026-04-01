"""Triton CLI entrypoint."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import typer

from triton.core.mixer import mix_at_snr
from triton.core.io import is_audio_file, iter_audio_files, load_audio, save_audio
from triton.cli.ingest import ingest_app
from triton.cli.transcribe import transcribe_app
from triton.cli.degrade import degrade_app
from triton.cli.convert import convert_app
from triton.cli.matrix import matrix_app
from triton.cli.normalize import normalize_app


app = typer.Typer(add_completion=False, help="Triton audio processing CLI")
app.add_typer(ingest_app, name="ingest")
app.add_typer(transcribe_app, name="transcribe")
app.add_typer(degrade_app, name="degrade")
app.add_typer(convert_app, name="convert")
app.add_typer(matrix_app, name="matrix")
app.add_typer(normalize_app, name="normalize")


@app.command()
def mix(
	speech_path: Path = typer.Argument(..., help="Speech file or directory"),
	noise_path: Path = typer.Argument(..., help="Noise file"),
	snr_db: float = typer.Option(-5.0, help="Target SNR in dB"),
	output_dir: Path = typer.Option(
		Path("outputs"), help="Directory to write mixed audio"
	),
	sample_rate: int | None = typer.Option(
		None, help="Resample rate. If omitted, use speech sample rate."
	),
):
	"""Mix speech with noise at a target SNR (single file or directory)."""
	speech_path = speech_path.expanduser().resolve()
	noise_path = noise_path.expanduser().resolve()
	output_dir = output_dir.expanduser().resolve()
	output_dir.mkdir(parents=True, exist_ok=True)

	if not noise_path.exists() or not is_audio_file(noise_path):
		raise typer.BadParameter("Noise path must be an audio file.")

	try:
		speech_files = list(iter_audio_files(speech_path))
	except ValueError as exc:
		raise typer.BadParameter(str(exc)) from exc

	noise_audio, noise_sr = load_audio(noise_path, sr=sample_rate)

	for speech_file in speech_files:
		speech_audio, speech_sr = load_audio(speech_file, sr=sample_rate)
		if sample_rate is None and noise_sr != speech_sr:
			noise_audio, _ = load_audio(noise_path, sr=speech_sr)

		mixed = mix_at_snr(speech_audio, noise_audio, snr_db)

		rel_name = (
			speech_file.name if speech_path.is_file() else speech_file.relative_to(speech_path)
		)
		out_path = output_dir / rel_name
		save_audio(out_path, mixed, speech_sr)
		typer.echo(f"Wrote {out_path}")


def run() -> None:
	app()


if __name__ == "__main__":
	run()
