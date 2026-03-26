# GUI

Triton includes a Streamlit GUI for project-oriented workflows.

## Run

- `pixi run gui`

## Project Dashboard

When the app opens, you start from a project launcher:

- **Create**: create a new project with canonical audio settings (sample rate and channel mode)
- **Open**: open an existing project directory from anywhere on disk
- **Recent**: quickly reopen recently used projects

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

## Overview Tab

The Overview tab provides basic file management for project assets:

- list stored files
- add files to the project library
- rename files
- remove files

Current file operations target the raw storage area (`data/raw`).

## Mix Tab

Mix speech and noise interactively:

- upload speech and noise audio
- set target SNR
- preview result and download mixed output

Inputs are normalized in-session to project audio settings before mixing.

## Pipelines Tab

The Pipelines tab is the default workspace view for project processing.

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

## Roadmap Tab

The roadmap tab tracks next GUI steps:

- project-native import pipeline
- ingest routed through project storage
- asset browser and run history backed by metadata
