# ingest

Download audio from external sources.

## RSS

- `pixi run triton ingest rss --feed <rss-url> --output-dir data/ingest --limit 10`

### Options

- `--feed`: RSS feed URL
- `--output-dir`: output directory
- `--limit`: limit number of episodes
- `--dry-run`: list episodes without downloading
- `--overwrite`: overwrite existing files
