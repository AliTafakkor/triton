# Triton 🐚

**Triton** is an audio signal processing toolkit for speech-in-noise research in the CONCH Lab.

Triton is moving to a **project-first** workflow: you create/open a project, define canonical audio settings, add files into that project, and run processing inside that boundary.

<div style="display:flex; gap:16px; flex-wrap:wrap;">
  <div style="flex:1; min-width:240px; background:#0f2d3f; padding:16px; border-radius:8px;">
    <h3>Core</h3>
    <p>Dependency-light math for SNR mixing, babble generation, vocoding, and filtering.</p>
  </div>
  <div style="flex:1; min-width:240px; background:#0f2d3f; padding:16px; border-radius:8px;">
    <h3>CLI</h3>
    <p>Batch processing for large datasets with reproducible settings. Label and organize files for cleaner workflows.</p>
  </div>
  <div style="flex:1; min-width:240px; background:#0f2d3f; padding:16px; border-radius:8px;">
    <h3>GUI</h3>
    <p>Project dashboard for create/open, file management with labeling, and interactive mixing.</p>
  </div>
</div>

## Key Features

**Pipelines & Reproducibility**: Define processing pipelines in your project config and run them on any file with consistent settings. Use [Pipeline Matrix](cli/matrix.md) to sweep across multiple files and parameter combinations in a single batch.

**Project-first workflow**: Centralized settings, file management, and reproducible outputs.

**File Labeling**: Organize your project files with custom labels (e.g., `bab-f1`, `bab-m1`, `background`), apply one label to a whole upload batch during import, and filter by label in CLI and GUI for easier asset management.

**Babble Speech Generation**: Mix labeled talker groups with per-file RMS normalization, concatenation per talker, intended-length planning, and balanced male/female selection when counts are not specified. If a talker is short, files repeat randomly to reach target length. GUI and CLI share the same core babble generator, and generated babble can be added back into the project as a labeled derivative (`bab-t#`) with provenance metadata.

## Quickstart

- Launch GUI and create/open a project.
- Add files, label them for organization, and manage project assets.
- Generate babble from labeled talker groups, or mix at target SNR in GUI or CLI.
- For batch processing, define a pipeline and run [Pipeline Matrix](cli/matrix.md) for parameter sweeps.

See the [Quickstart](quickstart.md) for commands.
