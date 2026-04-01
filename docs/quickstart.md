# Quickstart

## 1) Launch the GUI

- `pixi run gui`

## 2) Create or open a project

From the GUI dashboard:

- Create a project with sample rate and channel settings (mono/stereo), or
- Open an existing project from anywhere on disk, or
- Re-open from recent projects.

## 3) Import and inspect project files

In the **Import** tab:

- Import audio files into project raw storage
- Rename files
- Remove files
- Play files inline
- View precomputed spectrograms

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

## 10) Run a pipeline matrix (parameter sweep)

Once you've defined a pipeline in your project:

```bash
# Generate a CSV of file × parameter combinations
pixi run triton matrix generate my-project my-pipeline matrix.csv \n  --param '0.target_peak=0.5,0.8' \n  --file normalized/file1.wav \n  --file normalized/file2.wav

# Run all rows in the matrix
pixi run triton matrix run my-project my-pipeline matrix.csv
```

Each row produces isolated outputs so results are easy to compare and aggregate. See [Pipeline Matrix](cli/matrix.md) for details.

## Notes

- Project lifecycle and storage logic now live in `triton.core.project` so GUI and future CLI project commands can share the same implementation.
- Current CLI commands remain path-oriented; project-first CLI commands can be layered on top of the shared core module.
- Pipeline runs are grouped by run folder, and each step writes to its own step folder under `data/derived/pipelines/...`.
- Generated artifacts include sidecar provenance JSON files (`<artifact>.<suffix>.json`) to record source and action history.
- Imported files get spectrogram artifacts generated from project defaults in `triton.toml` (`[spectrogram]`).
- Pipeline Matrix allows reproducible batch processing by combining multiple files and parameter combinations in a single run.
