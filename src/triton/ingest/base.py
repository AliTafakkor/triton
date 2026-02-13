"""Shared ingestion types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Episode:
	"""Audio episode metadata."""

	title: str
	url: str
	published: str | None
	guid: str | None
	filename: str


class Source(Protocol):
	"""Protocol for ingestion sources."""

	def list_entries(self) -> list[Episode]:
		"""Return available entries for this source."""
		raise NotImplementedError

	def download(
		self,
		entries: list[Episode],
		output_dir: str,
		*,
		overwrite: bool = False,
	) -> list[str]:
		"""Download entries to output_dir and return file paths."""
		raise NotImplementedError
