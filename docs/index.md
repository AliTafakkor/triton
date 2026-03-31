# Triton 🐚

**Triton** is an audio signal processing toolkit for speech-in-noise research in the CONCH Lab.

Triton is moving to a **project-first** workflow: you create/open a project, define canonical audio settings, add files into that project, and run processing inside that boundary.

<div style="display:flex; gap:16px; flex-wrap:wrap;">
  <div style="flex:1; min-width:240px; background:#0f2d3f; padding:16px; border-radius:8px;">
    <h3>Core</h3>
    <p>Dependency-light math for SNR mixing, vocoding, and filtering.</p>
  </div>
  <div style="flex:1; min-width:240px; background:#0f2d3f; padding:16px; border-radius:8px;">
    <h3>CLI</h3>
    <p>Batch processing for large datasets with reproducible settings.</p>
  </div>
  <div style="flex:1; min-width:240px; background:#0f2d3f; padding:16px; border-radius:8px;">
    <h3>GUI</h3>
    <p>Project dashboard for create/open, file management, and interactive mixing.</p>
  </div>
</div>

## Key Features

**Pipelines & Reproducibility**: Define processing pipelines in your project config and run them on any file with consistent settings. Use [Pipeline Matrix](cli/matrix.md) to sweep across multiple files and parameter combinations in a single batch.

**Project-first workflow**: Centralized settings, file management, and reproducible outputs.

## Quickstart

- Launch GUI and create/open a project.
- Add files and manage project assets.
- Mix at target SNR in GUI or run CLI commands on project files.
- For batch processing, define a pipeline and run [Pipeline Matrix](cli/matrix.md) for parameter sweeps.

See the [Quickstart](quickstart.md) for commands.
