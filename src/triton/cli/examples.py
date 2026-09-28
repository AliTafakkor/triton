"""Example audio commands."""

from __future__ import annotations

from pathlib import Path

import typer

from triton.examples import CITATION, DEFAULT_DIR, EXAMPLES, LICENSE, ChecksumError, download_examples


examples_app = typer.Typer(add_completion=False, help="Download verified example audio (HARVARD corpus)")


@examples_app.command("list")
def list_cmd():
	"""List the example files and where they come from."""
	for example in EXAMPLES:
		typer.echo(f"{example.name:36s} {example.size / 1e6:5.1f} MB  {example.description}")
	typer.echo(f"\nSource: {CITATION}\nLicense: {LICENSE}")


@examples_app.command("download")
def download_cmd(
	dest: Path = typer.Option(DEFAULT_DIR, help="Directory to store the files"),
	name: list[str] = typer.Option(None, "--name", help="Only this file (repeatable). Default: all."),
):
	"""Download the example files and verify each one's SHA-256.

	Exits with a non-zero status if any file is truncated or does not match its
	pinned checksum, so a partial download is never mistaken for a complete one.
	"""

	def report(example, status):
		labels = {"cached": "already downloaded, checksum OK", "downloading": "downloading...", "done": "checksum OK"}
		typer.echo(f"  {example.name}: {labels[status]}")

	try:
		paths = download_examples(dest, names=name or None, on_progress=report)
	except ValueError as exc:
		raise typer.BadParameter(str(exc)) from exc
	except (ChecksumError, OSError) as exc:
		typer.secho(f"Download failed: {exc}", fg=typer.colors.RED, err=True)
		raise typer.Exit(code=1) from exc

	typer.echo(f"\n{len(paths)} verified file(s) in {Path(dest).expanduser()}")
	typer.echo(f"Please cite: {CITATION}\nLicense: {LICENSE}")
