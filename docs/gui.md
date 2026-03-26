# GUI

Triton includes a Streamlit GUI for project-oriented workflows.

## Run

- `pixi run gui`

## Project Dashboard

When the app opens, you start from a project launcher:

- **Recent** (default): quickly reopen recently used projects
- **Open**: open an existing project directory from anywhere on disk
- **Create**: create a new project with canonical audio settings (sample rate and channel mode)

When creating a project, you can also define spectrogram defaults from the UI (type and parameters). Those values are persisted into `triton.toml` under `[spectrogram]`.

## Project Structure

Each project stores config and data inside its own directory:

```text
<project>/
  triton.toml
  data/
    raw/
    normalized/
    derived/
  metadata/
```

## Import Tab

The Import tab is the default project workspace entry point for source files.

- import one or more audio files into project raw storage (`data/raw`)
- view files in a compact list with metadata
- play audio inline per file
- rename or remove files
- open a precomputed spectrogram per file in the right-side spectrogram panel

Imported files automatically trigger spectrogram computation using project defaults from `triton.toml` (`[spectrogram]` section).

Spectrogram artifacts are stored next to source files:

- `example.wav` -> `example.wav.spectrogram.npz`
- sidecar metadata: `example.wav.spectrogram.npz.json`

The `Spec` button selects the file and displays this precomputed artifact in the right panel.

## Mix Tab

Mix speech and noise interactively:

- upload speech and noise audio
- set target SNR
- preview result and download mixed output

Inputs are normalized in-session to project audio settings before mixing.

## Pipelines Tab

The Pipelines tab is the project processing workspace.

- select an existing pipeline
- run it on one or more project files
- create/edit pipelines from the right-side editor
- define ordered steps and step-specific options

### Pipeline Run Storage

Each pipeline execution creates a dedicated run folder and each step writes to its own subfolder:

```text
data/derived/pipelines/<pipeline_key>/run_<timestamp>_<id>/
  step_01_<step_key>/
  step_02_<step_key>/
  ...
```

For each processed input file, every step emits a step artifact (`.wav`) and a sidecar provenance JSON (`.wav.json`).

### Sidecar Provenance

Step sidecars include:

- source file path
- ordered action history up to that step
- step options and sample-rate transitions
- pipeline name/run identifier

## Project Spectrogram Defaults

Project spectrogram defaults are persisted in `triton.toml`:

- `type` (`stft`, `mel`, `cqt`)
- `n_fft`
- `hop_length`
- `win_length`
- `window`
- `n_mels`
- `fmin`
- `fmax`
- `power`

These defaults are used when importing files to precompute spectrograms.

## Roadmap Tab

The roadmap tab tracks next GUI steps:

- project-native import pipeline
- ingest routed through project storage
- asset browser and run history backed by metadata
