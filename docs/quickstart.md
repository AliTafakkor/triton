# Quickstart

## 1) Mix speech with noise

- `pixi run triton mix path/to/speech.wav path/to/noise.wav --snr-db -5`

## 2) Ingest from RSS

- `pixi run triton ingest rss --feed https://feeds.megaphone.fm/the-moth --output-dir data/moth --limit 10`

## 3) Transcribe locally (optional feature)

Requires the `transcribe` feature (see [Install](install.md#optional-features)):

- `pixi run --feature transcribe triton transcribe local data/moth/example.mp3 --output-dir outputs/transcripts --model tiny`

## 4) Apply degradations

- `pixi run triton degrade vocode data/speech --vocoder-type noise --n-bands 8`

## 5) Convert formats

- `pixi run triton convert resample data/audio --target-sr 16000`
- `pixi run triton convert mono data/stereo`
