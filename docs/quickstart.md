# Quickstart

## 1) Launch the GUI

- `pixi run gui`

## 2) Create or open a project

From the GUI dashboard:

- Create a project with sample rate and channel settings (mono/stereo), or
- Open an existing project from anywhere on disk, or
- Re-open from recent projects.

## 3) Add and manage project files

In the **Overview** tab:

- Add audio files to the project library
- Rename files
- Remove files

## 4) Mix speech with noise in GUI

In the **Mix** tab:

- Upload speech and noise
- Set target SNR
- Play and download the mixed output

## 5) Mix speech with noise in CLI (path-based)

- `pixi run triton mix path/to/speech.wav path/to/noise.wav --snr-db -5`

## 6) Ingest from RSS

- `pixi run triton ingest rss --feed https://feeds.megaphone.fm/the-moth --output-dir data/moth --limit 10`

## 7) Transcribe locally (optional feature)

Requires the `transcribe` feature (see [Install](install.md#optional-features)):

- `pixi run --feature transcribe triton transcribe local data/moth/example.mp3 --output-dir outputs/transcripts --model tiny`

## 8) Apply degradations

- `pixi run triton degrade vocode data/speech --vocoder-type noise --n-bands 8`

## 9) Convert formats

- `pixi run triton convert resample data/audio --target-sr 16000`
- `pixi run triton convert mono data/stereo`

## Notes

- Project lifecycle and storage logic now live in `triton.core.project` so GUI and future CLI project commands can share the same implementation.
- Current CLI commands remain path-oriented; project-first CLI commands can be layered on top of the shared core module.
