"""Triton CLI entrypoint."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import librosa
import numpy as np
import soundfile as sf
import typer

from triton.core.mixer import mix_at_snr
from triton.cli.ingest import ingest_app


app = typer.Typer(add_completion=False, help="Triton audio processing CLI")
app.add_typer(ingest_app, name="ingest")


SUPPORTED_EXTS = {".wav", ".flac", ".ogg", ".mp3", ".m4a"}


def _is_audio_file(path: Path) -> bool:
	return path.is_file() and path.suffix.lower() in SUPPORTED_EXTS


def _iter_audio_files(path: Path) -> Iterable[Path]:
	if path.is_file():
		if _is_audio_file(path):
			yield path
		return

	for ext in SUPPORTED_EXTS:
		yield from path.rglob(f"*{ext}")


def _load_audio(path: Path, target_sr: int | None) -> Tuple[np.ndarray, int]:
	audio, sr = librosa.load(path, sr=target_sr, mono=True)
	return audio, sr


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

	if not noise_path.exists() or not _is_audio_file(noise_path):
		raise typer.BadParameter("Noise path must be an audio file.")

	if not speech_path.exists():
		raise typer.BadParameter("Speech path does not exist.")

	speech_files = list(_iter_audio_files(speech_path))
	if not speech_files:
		raise typer.BadParameter("No audio files found in speech path.")

	noise_audio, noise_sr = _load_audio(noise_path, sample_rate)

	for speech_file in speech_files:
		speech_audio, speech_sr = _load_audio(speech_file, sample_rate)
		if sample_rate is None and noise_sr != speech_sr:
			noise_audio, _ = _load_audio(noise_path, speech_sr)

		mixed = mix_at_snr(speech_audio, noise_audio, snr_db)

		rel_name = (
			speech_file.name if speech_path.is_file() else speech_file.relative_to(speech_path)
		)
		out_path = output_dir / rel_name
		out_path.parent.mkdir(parents=True, exist_ok=True)

		sf.write(out_path, mixed, speech_sr)
		typer.echo(f"Wrote {out_path}")


def run() -> None:
	app()


if __name__ == "__main__":
	run()
