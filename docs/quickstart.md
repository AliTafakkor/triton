# Quickstart

## 1) Mix speech with noise

- `pixi run triton mix path/to/speech.wav path/to/noise.wav --snr-db -5`

## 2) Ingest from RSS

- `pixi run triton ingest rss --feed https://feeds.megaphone.fm/the-moth --output-dir data/moth --limit 10`

## 3) Transcribe locally

- `pixi run triton transcribe local data/moth/example.mp3 --output-dir outputs/transcripts --model tiny`
