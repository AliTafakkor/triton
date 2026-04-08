# core.project

Project lifecycle, storage, and normalization helpers shared by GUI and CLI.

## Project

Typed project model loaded from `triton.toml`. Defines the complete audio contract for everything inside the project.

### Fields

| Field | Type | Description |
|---|---|---|
| `name` | `str` | Project name |
| `path` | `Path` | Project root directory |
| `sample_rate` | `int` | Target sample rate in Hz (e.g. 16000) |
| `channel_mode` | `"mono" \| "stereo"` | Target channel format |
| `bit_depth` | `int` | Target bit depth: 8, 16, 24, or 32 (default 16) |
| `file_format` | `str` | Target output format: `"wav"`, `"flac"`, or `"ogg"` (default `"wav"`) |

### Properties

- `raw_dir` — path to `data/raw/`
- `normalized_dir` — path to `data/normalized/`

### Methods

- `Project.load(project_dir)` — load from `triton.toml`
- `Project.create(project_dir, sample_rate, channel_mode, bit_depth, file_format)` — initialize + write config
- `register_recent()` — add to recent projects list
- `list_files()` — list normalized audio files (from `data/normalized/`)
- `add_file(filename, content)` — write raw bytes to `data/raw/`
- `to_dict()` — serialize to plain dict

## Project config (`triton.toml`)

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
...
```

## Pipeline

Typed pipeline model persisted in `triton.toml`.

### Fields

- `name`
- `steps` (ordered action keys)
- `step_options` (per-step settings keyed by step index)

### Pipeline helpers

- `load_project_pipelines(project_dir)`
- `save_project_pipelines(project_dir, pipelines)`

## Spectrogram settings

Project config includes a `[spectrogram]` section with defaults used at import time.

### Keys

- `type` (`stft`, `mel`, `cqt`)
- `n_fft`, `hop_length`, `win_length`, `window`
- `n_mels`, `fmin`, `fmax`, `power`

### Helpers

- `load_project_spectrogram_settings(project_dir)`
- `update_project_spectrogram_settings(project_dir, settings)`

## Lifecycle helpers

- `create_project(project_dir, sample_rate, channel_mode, bit_depth, file_format)` — create a new project
- `load_project_config(project_dir)` — load existing project
- `initialize_project_tree(project_dir)` — create directory layout
- `write_project_config(project_dir, sample_rate, channel_mode, bit_depth, file_format)` — write `triton.toml`

## Path helpers

- `project_config_path(project_dir)`
- `project_raw_dir(project_dir)` — `data/raw/`
- `project_normalized_dir(project_dir)` — `data/normalized/`

## Recent projects

- `load_recent_projects()`
- `save_recent_projects(projects)`
- `register_recent_project(project_dir, project_name)`

## File management

- `list_project_files(project_dir, label=None)` — list raw audio files, optionally filtered by label
- `list_normalized_project_files(project_dir)` — list normalized audio files in `data/normalized/`
- `add_project_file(project_dir, filename, content)` — write raw bytes to `data/raw/`
- `normalize_project_file(project_dir, raw_path, project)` — convert a raw file to the project spec and save to `data/normalized/`
- `rename_project_file(file_path, new_name)`
- `delete_project_file(file_path)`
- `delete_project_files_by_label(project_dir, label)` — delete all normalized files with a given label (and matching raw stems)
- `sanitize_filename(name)`

## File labels

Labels are stored in `metadata/file_labels.json` keyed by file stem (no extension), e.g. `{ "filename": ["label1", "label2"] }`.
Files can have multiple labels. Use comma-separated values in the GUI label field.
Label keys use the stem so that the raw file (`file.mp3`) and normalized file (`file.wav`) share the same label entry.

- `load_file_labels(project_dir)` → `dict[str, list[str]]`
- `save_file_labels(project_dir, labels)`
- `set_file_labels(project_dir, file_path, labels)` — set all labels for a file (replaces existing)
- `set_file_label(project_dir, file_path, label)` — set a single label (backward-compat wrapper)
- `get_file_labels(project_dir, file_path)` → `list[str]`
- `get_file_label(project_dir, file_path)` → `str | None` (first label)
- `set_project_file_labels(project_dir, file_paths, label)` — batch apply one label

## Constants

- `PROJECT_CONFIG_NAME` — `"triton.toml"`
- `SUPPORTED_AUDIO_SUFFIXES` — `{".wav", ".flac", ".ogg", ".mp3", ".m4a"}`
