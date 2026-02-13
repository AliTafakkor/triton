"""Triton ingestion commands."""

from __future__ import annotations

from pathlib import Path

import typer

from triton.ingest.rss import RssSource


ingest_app = typer.Typer(add_completion=False, help="Ingest audio datasets")


@ingest_app.command("rss")
def ingest_rss(
	feed: str = typer.Option(..., help="RSS feed URL"),
	output_dir: Path = typer.Option(Path("data/ingest"), help="Output directory"),
	limit: int | None = typer.Option(None, help="Limit number of episodes"),
	overwrite: bool = typer.Option(False, help="Overwrite existing files"),
	dry_run: bool = typer.Option(False, help="List episodes without downloading"),
):
	"""Download audio enclosures from an RSS feed."""
	source = RssSource(feed)
	entries = source.list_entries()

	if limit is not None:
		entries = entries[: max(0, limit)]

	if dry_run:
		for entry in entries:
			typer.echo(f"{entry.title} -> {entry.url}")
		return

	paths = source.download(entries, output_dir, overwrite=overwrite)
	for path in paths:
		typer.echo(f"Wrote {path}")
