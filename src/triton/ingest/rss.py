"""RSS ingestion source."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from urllib.parse import urlparse

import feedparser
import requests

from triton.ingest.base import Episode
from triton.core.io import write_sidecar


@dataclass(frozen=True)
class RssSource:
	"""RSS feed source for audio enclosures."""

	feed_url: str

	def list_entries(self) -> list[Episode]:
		feed = feedparser.parse(self.feed_url)
		entries: list[Episode] = []
		for item in feed.entries:
			url = _select_audio_url(item)
			if not url:
				continue
			title = _get(item, "title", "audio")
			filename = _filename_from_url_or_title(url, title)
			entries.append(
				Episode(
					title=title or "Untitled",
					url=url,
					published=_get(item, "published", None),
					guid=_get(item, "id", None),
					filename=filename,
				)
			)

		return entries

	def download(
		self,
		entries: list[Episode],
		output_dir: str | Path,
		*,
		overwrite: bool = False,
	) -> list[str]:
		output_path = Path(output_dir).expanduser().resolve()
		output_path.mkdir(parents=True, exist_ok=True)
		paths: list[str] = []

		for episode in entries:
			out_file = output_path / episode.filename
			if out_file.exists() and not overwrite:
				paths.append(str(out_file))
				continue

			_download_file(episode.url, out_file)
			write_sidecar(
				out_file,
				source={"url": episode.url, "feed_url": self.feed_url},
				actions=[
					{
						"step": "rss_download",
						"options": {
							"title": episode.title,
							"published": episode.published,
							"guid": episode.guid,
						},
					}
				],
			)
			paths.append(str(out_file))

		return paths


AUDIO_EXTS = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aac"}


def _get(item, key: str, default=None):
	if isinstance(item, dict):
		return item.get(key, default)
	return getattr(item, key, default)


def _select_audio_url(item) -> str | None:
	enclosures = _get(item, "enclosures", [])
	for enclosure in enclosures:
		url = enclosure.get("href") or enclosure.get("url")
		if not url:
			continue
		content_type = (enclosure.get("type") or "").lower()
		if content_type.startswith("audio/"):
			return url
		path = urlparse(url).path
		if Path(path).suffix.lower() in AUDIO_EXTS:
			return url

	return None


def _filename_from_url_or_title(url: str, title: str) -> str:
	path = urlparse(url).path
	name = Path(path).name
	if name:
		ext = Path(name).suffix.lower()
		if ext in AUDIO_EXTS:
			return name

	slug = _slugify(title)
	return f"{slug}.mp3"


def _slugify(text: str) -> str:
	text = text.strip().lower()
	text = re.sub(r"[^a-z0-9]+", "-", text)
	text = re.sub(r"-+", "-", text).strip("-")
	return text or "audio"


def _download_file(url: str, output_path: Path) -> None:
	with requests.get(url, stream=True, timeout=60) as response:
		response.raise_for_status()
		with open(output_path, "wb") as handle:
			for chunk in response.iter_content(chunk_size=1024 * 1024):
				if chunk:
					handle.write(chunk)
