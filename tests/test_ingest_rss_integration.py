import os
import tempfile
import unittest
from pathlib import Path

from triton.ingest.rss import RssSource


class TestRssIngestIntegration(unittest.TestCase):
	def test_downloads_moth_episode(self):
		if os.getenv("TRITON_INTEGRATION") != "1":
			self.skipTest("Set TRITON_INTEGRATION=1 to run integration test")

		source = RssSource("https://feeds.megaphone.fm/the-moth")
		entries = source.list_entries()
		self.assertTrue(entries, "No entries found in feed")

		batch = entries[:10]
		with tempfile.TemporaryDirectory() as tmp_dir:
			paths = source.download(batch, Path(tmp_dir), overwrite=True)
			self.assertEqual(len(paths), len(batch))
			for path in paths:
				out_path = Path(path)
				self.assertTrue(out_path.exists())
				self.assertGreater(out_path.stat().st_size, 0)


if __name__ == "__main__":
	unittest.main()
