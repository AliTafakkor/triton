# GUI

Triton includes a Streamlit GUI for project-oriented workflows.

## Run

- `pixi run gui`
- UI theme is forced to dark mode via project Streamlit config (`.streamlit/config.toml`).

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

## Manage and Explore Files Tab

The Manage and Explore Files tab is the default project workspace entry point for managing source files and their labels.

### Import Files

- import one or more audio files into project raw storage (`data/raw`)
- optionally assign one label to the full upload batch during import
- view files in a compact list with metadata
- play audio inline per file
- rename or remove files
- open a precomputed spectrogram per file in the right-side spectrogram panel

After importing files, the file browser selection and upload widget are automatically reset to allow for a clean next import.

### Rename Labels

Rename an existing label to apply the new name to all files that currently have that label. This is useful for:

- correcting label typos across multiple files
- reorganizing labels (e.g., renaming all `talker` labels to `bab-f1`)
- preparing files for babble generation with consistent naming (e.g., `bab-f1`, `bab-m1`)

Imported files automatically trigger spectrogram computation using project defaults from `triton.toml` (`[spectrogram]` section).

Spectrogram artifacts are stored next to source files:

- `example.wav` -> `example.wav.spectrogram.npz`
- sidecar metadata: `example.wav.spectrogram.npz.json`

The `Spec` button selects the file and displays this precomputed artifact in the right panel.

Spectrogram viewer behavior:

- rendered as an interactive Plotly heatmap (zoom/pan/scroll)
- time/frequency axes are shown directly in the chart
- inferno colormap is used for display
- large spectrograms are downsampled for responsive interaction (shown in the panel caption)

Accessibility and deprecation cleanup in the Import list:

- hidden labels are non-empty to avoid Streamlit accessibility warnings
- deprecated `use_container_width` usage has been replaced with `width="stretch"`

## Mix Tab

Mix speech and noise interactively:

- upload speech and noise audio
- set target SNR
- preview result and download mixed output

Inputs are normalized in-session to project audio settings before mixing.

## Babble Tab

Generate babble speech from labeled talker groups:

- select the total number of talker groups to mix
- optionally set female and male counts separately
- optionally assign the same label to an uploaded batch in the Import tab
- use labels such as `bab-f1`, `bab-f2`, `bab-m1`, and `bab-m2`
- normalize each source file to the target RMS before concatenating files for the same talker
- optionally peak-normalize the mixed output to prevent clipping
- download the generated babble for use in experiments

If no sex split is provided, Triton balances female and male talkers as evenly as possible. When multiple files share a babble label, they are concatenated in filename order after RMS normalization.

Typical workflow:

1. Import audio files in the Manage and Explore Files tab, optionally assigning a batch label
2. Edit labels individually in the file table, or rename multiple labels at once using the "Rename Labels" panel
3. Go to the Babble tab
4. Review the available babble talker groups (labeled with `bab-f1`, `bab-m1`, etc.)
5. Set the total number of talkers, then optionally set female and male counts
6. Adjust target RMS and peak normalization settings
7. Click "Generate Babble" and download the output

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
