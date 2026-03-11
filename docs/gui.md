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

## Roadmap Tab

The roadmap tab tracks next GUI steps:

- project-native import pipeline
- ingest routed through project storage
- asset browser and run history backed by metadata
