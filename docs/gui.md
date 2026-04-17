# GUI

Triton includes a Streamlit GUI for project-oriented workflows.

## Run

- `pixi run gui`
- UI theme is forced to dark mode via project Streamlit config (`.streamlit/config.toml`).

## Developer Layout

The GUI is now split by responsibility so `app.py` stays focused on page-level orchestration:

- `triton.gui.app`: app bootstrap, page setup, and top-level routing
- `triton.gui.shared`: shared helpers used across multiple tabs
- `triton.gui.tabs.file_library`: Manage and Explore Files tab UI
- `triton.gui.tabs.pipelines`: Pipelines tab UI and controls
- `triton.gui.tabs.rss`: RSS ingest tab UI
- `triton.gui.assets.app.css`: GUI stylesheet loaded at app startup

When adding a new tab, place tab-specific rendering logic in `triton.gui.tabs.<name>` and keep cross-tab helpers in `triton.gui.shared`.

## Project Dashboard

When the app opens, you start from a project launcher:

- **Recent** (default): quickly reopen recently used projects
- **Open**: open an existing project directory from anywhere on disk
- **Create**: create a new project with canonical audio settings (sample rate and channel mode)

When creating a project, you define the full audio spec: sample rate, channel mode (mono/stereo), bit depth (8/16/24/32), and file format (wav/flac/ogg). You can also set spectrogram defaults (type and parameters). All settings are persisted into `triton.toml`.

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

- import one or more audio files; originals are saved to `data/raw/` and a normalized copy (matching the project spec) is written to `data/normalized/` automatically
- optionally set a filename prefix to prepend to all imported files (useful when different sources share the same original filenames)
- optionally assign one label to the full upload batch during import
- view files in a compact list with metadata (name, size, format, labels)
- play audio inline per file
- rename or remove files (also removes the matching raw counterpart)
- open a precomputed spectrogram per file in the right-side spectrogram panel (spectrogram is generated from the normalized file)

After importing, the file browser selection and upload widget are automatically reset to allow for a clean next import.

### Label Files

Each file in the list has a **Label** text field. Enter one or more comma-separated labels (e.g. `bab-f1, studio`). Labels are saved immediately when you commit the field (press Enter or tab away).

- The **Filter by label** dropdown filters the list; select `(None)` to show only files with no labels.
- Label counts are shown in the Manage Labels expander alongside each label name.

### Manage Labels

The **Manage Labels** expander (above the file list) provides:

#### Rename a label

Rename an existing label to apply the new name to all files that currently have that label. Useful for:

- correcting label typos across multiple files
- reorganizing labels (e.g., renaming all `talker` labels to `bab-f1`)
- preparing files for babble generation with consistent naming (e.g., `bab-f1`, `bab-m1`)

On multi-label files, only the target label is replaced; other labels are preserved.

#### Bulk label unlabeled files

Enter a label and click **Apply** to assign it to every file that currently has no labels. Useful for quickly categorizing a freshly imported batch.

### Bulk Delete by Label

You can remove all imported files associated with a label directly in the Manage and Explore Files tab.

- in **Bulk Actions**, select a label in **Delete all files with label**
- check **Confirm**
- click **Delete Label Files**

This removes all normalized files with that label, removes matching raw-source files, and updates label metadata.

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
- set an intended per-talker output length
- normalize each source file to the target RMS before concatenating files for the same talker
- optionally peak-normalize the mixed output to prevent clipping
- monitor a progress bar and a live status console while babble is generated
- download the generated babble for use in experiments
- add the generated babble back into the project as a derivative artifact labeled `bab-t#`, where `#` is the number of talkers used

When you add a generated babble to the project, Triton stores a provenance sidecar alongside the audio. The sidecar records the source files, babble parameters, intended length, RMS settings, and other generation details so the derivative can be traced and reproduced later.

If no sex split is provided, Triton balances female and male talkers as evenly as possible. When multiple files share a babble label, they are concatenated in filename order after RMS normalization.

If source material for a selected talker is shorter than the intended length, Triton warns and randomly repeats that talker's files until target length is reached.

Babble generation in the GUI and CLI both use the same shared core function in `triton.degrade.noise_generator`, so file selection, loading, and mixing behavior are consistent.

Typical workflow:

1. Import audio files in the Manage and Explore Files tab, optionally assigning a batch label
2. Edit labels individually in the file table, or rename multiple labels at once using the "Rename Labels" panel
3. Go to the Babble tab
4. Review the available babble talker groups (labeled with `bab-f1`, `bab-m1`, etc.)
5. Set the total number of talkers, then optionally set female and male counts
6. Set intended length, then adjust target RMS and peak normalization settings
7. Click "Generate Babble"
8. Use Download to save the output locally or Add to project to persist it as a labeled derivative with provenance metadata

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

The Run Pipelines tab shows a live progress bar while files are being processed.

### Matrix CSV Storage

Matrix CSVs generated in the GUI are saved under:

```text
metadata/matrices/<pipeline_name>/<pipeline_name>.matrix.csv
```

In **Pipeline Matrix → Generate Matrix**, selecting a pipeline immediately refreshes the displayed step/option controls for that pipeline.

In **Pipeline Matrix → Run Matrix**, you can choose a source mode:

- **Saved**: browse and select matrix CSVs already stored for the selected pipeline
- **Upload**: upload a CSV from your local machine
- **Path**: provide an absolute or project-relative CSV path

In **Pipeline Matrix → Run Matrix**, selecting a pipeline immediately refreshes the saved matrix list for that pipeline.

The Matrix Run panel also shows a live row-by-row progress bar while rows are executed, and it can optionally collect only final outputs into `final_by_params/set_####_*` folders for each parameter combination.

### Add Noise Step

The pipeline editor includes an `add_noise` step with these options:

- `noise_type`: `auto`, `babble`, `white`, `colored`, `ssn`
- `snr_db`: target SNR in dB
- `noise_project_file`: select a noise source from existing project audio files
- `seed`: optional random seed

Notes:

- `babble` in this step is file-backed only and is not generated.
- With `noise_type=auto`, files named like `bab-t8.wav` are auto-recognized as multitalker babble sources.
- If the provided noise is longer than the speech signal, Triton randomly selects a same-length segment before mixing.

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

## RSS Ingest Tab

Fetch and download podcast episodes directly into a project from an RSS feed.

- enter a feed URL and set a max episode limit
- optionally filter by publish date (preset ranges: last 7 / 30 / 90 days, or custom dates)
- preview matching episodes before downloading
- download episodes; each file is normalized to the project spec and a spectrogram is generated
- downloaded episodes land in `data/raw/` first, then normalized copies are written to `data/normalized/`
