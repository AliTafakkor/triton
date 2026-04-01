# Quickstart

## 1) Launch the GUI

- `pixi run gui`

## 2) Create or open a project

From the GUI dashboard:

- Create a project with sample rate and channel settings (mono/stereo), or
- Open an existing project from anywhere on disk, or
- Re-open from recent projects.

## 3) Manage and explore project files

In the **Manage and Explore Files** tab:

- Import audio files into project raw storage
- Apply one label to all files uploaded together
- Rename files
- **Label files** (e.g., `bab-f1`, `bab-m1`, `background`) for organization
- **Rename labels** to apply a new label to all files with the old label (useful for bulk corrections or preparations)
- **Filter by label** to view only files with a specific tag
- Remove files
- Play files inline
- View precomputed spectrograms

## 4) Generate babble speech (mix multiple talkers)

In the **Babble** tab:

- Label talker-group files as `bab-f1`, `bab-f2`, `bab-m1`, etc., or assign that label during import to a whole batch
- Set the total number of talker groups to mix
- Optionally set female and male counts separately
- Normalize each source file to the same RMS, concatenate files for each talker, and optionally peak-normalize the output
- Play and download the mixed babble output

**Or from CLI:**

```bash
# Label talker groups
pixi run triton files label my-project speaker_f1.wav "bab-f1"
pixi run triton files label my-project speaker_f2.wav "bab-f2"
pixi run triton files label my-project speaker_m1.wav "bab-m1"

# Mix 3 talker groups with RMS normalization
pixi run triton babble mix my-project \
  --num-talkers 3 \
  --output-path outputs/babble.wav
```

## 5) Mix speech with noise in GUI

In the **Mix** tab:

- Upload speech and noise
- Set target SNR
- Play and download the mixed output

## 6) Mix speech with noise in CLI (path-based)

- `pixi run triton mix path/to/speech.wav path/to/noise.wav --snr-db -5`

## 7) Ingest from RSS

- `pixi run triton ingest rss --feed https://feeds.megaphone.fm/the-moth --output-dir data/moth --limit 10`

## 8) Transcribe locally (optional feature)

Requires the `transcribe` feature (see [Install](install.md#optional-features)):

- `pixi run --feature transcribe triton transcribe local data/moth/example.mp3 --output-dir outputs/transcripts --model tiny`

## 9) Apply degradations

- `pixi run triton degrade vocode data/speech --vocoder-type noise --n-bands 8`

## 10) Convert formats

- `pixi run triton convert resample data/audio --target-sr 16000`
- `pixi run triton convert mono data/stereo`

## 11) Run a pipeline matrix (parameter sweep)

Once you've defined a pipeline in your project:

```bash
# Generate a CSV of file × parameter combinations
pixi run triton matrix generate my-project my-pipeline matrix.csv \n  --param '0.target_peak=0.5,0.8' \n  --file normalized/file1.wav \n  --file normalized/file2.wav

# Run all rows in the matrix
pixi run triton matrix run my-project my-pipeline matrix.csv
```

Each row produces isolated outputs so results are easy to compare and aggregate. See [Pipeline Matrix](cli/matrix.md) for details.

## File Labeling & Organization

Organize your project files with labels for easier workflows:

```bash
# Label files
pixi run triton files label my-project speaker_f.wav "bab-f1"
pixi run triton files label my-project background.wav "noise"

# List files by label
pixi run triton files list my-project --label "bab-f1"

# View all labels
pixi run triton files show-labels my-project
```

See [Files](cli/files.md) for more details.

## Babble Speech Workflow

Generate babble by mixing labeled talker groups with RMS normalization:

```bash
# Mix talker groups
pixi run triton babble mix my-project \
  --num-talkers 3 \
  --output-path outputs/babble_3talkers.wav

# Or specify female and male counts explicitly
pixi run triton babble mix my-project \
  --num-talkers 4 \
  --num-female-talkers 2 \
  --num-male-talkers 2 \
  --output-path outputs/babble_4talkers.wav

# Add noise to the babble at a target SNR
pixi run triton mix outputs/babble.wav noise.wav --snr-db -5
```

See [Babble](cli/babble.md) for full command reference.

## Notes

- Project lifecycle and storage logic now live in `triton.core.project` so GUI and future CLI project commands can share the same implementation.
- Current CLI commands remain path-oriented; project-first CLI commands can be layered on top of the shared core module.
- Pipeline runs are grouped by run folder, and each step writes to its own step folder under `data/derived/pipelines/...`.
- Generated artifacts include sidecar provenance JSON files (`<artifact>.<suffix>.json`) to record source and action history.
- Imported files get spectrogram artifacts generated from project defaults in `triton.toml` (`[spectrogram]`).
- Pipeline Matrix allows reproducible batch processing by combining multiple files and parameter combinations in a single run.
- File labels persist in `metadata/file_labels.json` and work across CLI and GUI interfaces.
