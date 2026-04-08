# Triton 🔱🐚
An audio signal processing toolkit for speech-in-noise research.

Triton is a modular audio utility designed to standardize stimuli preparation and signal degradation. It provides a project-centric workflow where every audio asset lives inside a project that enforces a single canonical format from the moment a file is imported.

## Features

**Project-based audio management**: Create a project and define its full audio spec — sample rate, channel mode (mono/stereo), bit depth (8/16/24/32), and output file format (wav/flac/ogg). Every file imported into the project is automatically normalized to that spec.

**Automatic normalization on import**: Raw files land in `data/raw/` untouched. A converted copy (matching the project spec) is written to `data/normalized/` automatically, including resampling, channel conversion, and requantization.

**Filename prefix on import**: In the GUI import form, you can set an optional filename prefix (for example, `feedA_`) that is prepended to every uploaded filename to avoid collisions between same-named files from different sources.

**Multi-label file organization**: Assign one or more labels to any file (e.g. `bab-f1`, `studio`). Apply a batch label at import, edit labels per file in the GUI using comma-separated values, filter files by label, delete all files with a specific label from the CLI, and use the Manage Labels panel to rename labels or bulk-label all unlabeled files. Labels are stored in `metadata/file_labels.json` as lists.

**Babble Speech Generation**: Mix labeled talker groups (`bab-f1`, `bab-m1`, …), normalize each source file to a common RMS, concatenate per talker, and generate cocktail-party scenarios with balanced male/female selection. Generated babble is added back to the project as a labeled derivative with a provenance sidecar.

**Pipeline Matrix**: Define a pipeline once, then sweep across multiple files and parameter combinations in a single reproducible batch. Generate a CSV of file × parameter combinations and run them all at once, with isolated outputs per row for easy comparison and aggregation.

**Spectrogram viewer**: Import-time spectrogram generation (STFT / Mel / CQT) with per-project defaults stored in `triton.toml`. The GUI renders interactive Plotly spectrograms with zoom/pan/scroll in the file browser.

## Project Structure

```text
my-project/
  triton.toml           ← project config (audio spec, pipelines, spectrogram defaults)
  data/
    raw/                ← original imported files (any supported format)
    normalized/         ← auto-converted to project spec on import
    derived/            ← pipeline outputs and generated artifacts
  metadata/
    file_labels.json    ← multi-label store per file
    project.log.jsonl   ← append-only audit log of all events
```

## Project Config (`triton.toml`)

```toml
[project]
name = "my-project"
root = "/path/to/my-project"

[audio]
sample_rate = 16000
channels = "mono"
bit_depth = 16
format = "wav"

[storage]
raw = "data/raw"
normalized = "data/normalized"
derived = "data/derived"
metadata = "metadata"

[spectrogram]
type = "stft"
n_fft = 1024
hop_length = 256
win_length = 1024
window = "hann"
n_mels = 128
fmin = 32.7
fmax = 8000.0
power = 2.0
```

## Workflow

1. **Create a project** — choose sample rate, channels, bit depth, file format.
2. **Import files** — optionally set a filename prefix, then import; raw originals go to `data/raw/` and normalized copies go to `data/normalized/` automatically.
3. **Label files** — assign one or more labels per file; filter, rename, and bulk-label from the GUI or CLI.
4. **Run pipelines** — degrade, convert, mix, and transcribe against project assets.
5. **Generate babble** — select labeled talker groups and mix cocktail-party noise.
6. **Inspect outputs** — browse spectrograms, review the audit log, download artifacts.

## Audio Mixing

Triton uses symmetric RMS-based SNR scaling for mixing:
- Both signals normalized independently to the same RMS level before mixing
- Target SNR split symmetrically: signal boosted by SNR/2 dB, noise attenuated by SNR/2 dB
- Result re-normalized to target RMS after mixing, controlling loudness independently of SNR
- Optional boundary smoothing for multi-segment mixing to avoid abrupt transitions

## Pipeline Output Layout

```text
<project>/
  data/
    derived/
      pipelines/
        <pipeline_name_key>/
          run_<UTC timestamp>_<id>/
            step_01_<step_key>/
              <input_stem>.wav
              <input_stem>.wav.json     ← provenance sidecar
            step_02_<step_key>/
              ...
```

## Sidecar Provenance

Every generated file has a JSON sidecar (`<file>.<ext>.json`) capturing:
- artifact identity (path, name, suffix)
- source reference (raw path or ingest URL/feed)
- ordered action history with parameters
- generation timestamp and schema version

## Spectrograms

Import-time spectrograms are stored as `<file>.spectrogram.npz` with a `.json` sidecar. The spectrogram panel in the GUI renders the precomputed artifact without recomputing on every page render.

## Why Pixi?

Triton uses Pixi to lock the full environment — including heavy native dependencies like `ffmpeg`, `librosa`, and `soundfile` — ensuring identical behavior across macOS (Apple Silicon) and Linux via a single `pixi.lock` file.

## Core Components

- **`/core`** — foundational Python API: RMS mixing, vocoding, filtering, conversion, spectrogram, project model
- **CLI** — batch processing; file management, babble generation, pipeline runs, matrix sweeps
- **GUI** — Streamlit dashboard: drag-and-drop import, spectrogram viewer, pipeline builder, label manager, babble mixer
