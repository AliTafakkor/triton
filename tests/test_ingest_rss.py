import unittest
from pathlib import Path
import tempfile
from unittest.mock import MagicMock, patch

from triton.ingest.rss import RssSource


class TestRssIngest(unittest.TestCase):
	@patch("triton.ingest.rss.feedparser.parse")
	def test_list_entries_selects_audio(self, mock_parse):
		mock_parse.return_value = MagicMock(
			entries=[
				{
					"title": "Episode One",
					"id": "guid-1",
					"published": "2025-01-01",
					"enclosures": [
						{"href": "https://example.com/ep1.mp3", "type": "audio/mpeg"}
					],
				},
				{
					"title": "Episode Two",
					"id": "guid-2",
					"published": "2025-01-02",
					"enclosures": [
						{"href": "https://example.com/ep2.txt", "type": "text/plain"}
					],
				},
			]
		)

		source = RssSource("https://example.com/feed")
		entries = source.list_entries()

		self.assertEqual(len(entries), 1)
		self.assertEqual(entries[0].title, "Episode One")
		self.assertEqual(entries[0].url, "https://example.com/ep1.mp3")
		self.assertEqual(entries[0].filename, "ep1.mp3")

	@patch("triton.ingest.rss.requests.get")
	def test_download_writes_files(self, mock_get):
		response = MagicMock()
		response.__enter__.return_value = response
		response.__exit__.return_value = None
		response.iter_content.return_value = [b"abc", b"def"]
		response.raise_for_status.return_value = None
		mock_get.return_value = response

		source = RssSource("https://example.com/feed")

		with tempfile.TemporaryDirectory() as tmp_dir:
			out_dir = Path(tmp_dir)
			file_path = out_dir / "episode.mp3"

			episodes = [
				MagicMock(
					title="Episode",
					url="https://example.com/episode.mp3",
					published=None,
					guid=None,
					filename="episode.mp3",
				)
			]

			paths = source.download(episodes, out_dir, overwrite=True)
			self.assertEqual(Path(paths[0]).resolve(), file_path.resolve())
			self.assertTrue(file_path.exists())

			with open(file_path, "rb") as handle:
				self.assertEqual(handle.read(), b"abcdef")


if __name__ == "__main__":
	unittest.main()
