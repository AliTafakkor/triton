# core.pipeline_runtime

Shared runtime abstraction for executing pipelines on audio files.

`triton.core.pipeline_runtime` provides the core execution engine for applying pipelines. It handles:

- Run ID generation and directory organization
- Step execution and output isolation
- Pipeline file writing and patching
- Run metadata and logging

This module is used by both the CLI and GUI to ensure consistent pipeline behavior across interfaces.

## Key Functions

### `new_pipeline_run_id() -> str`

Generate a new unique run ID based on current timestamp.

Returns a string like `run_20260331_143022_abc123`.

### `pipeline_run_dir(project: Project, pipeline_name: str, run_id: str) -> Path`

Get the output directory for a pipeline run.

Returns `project.path / data/derived/pipelines/<pipeline_name>/<run_id>`.

### `run_pipeline_on_file(project: Project, pipeline: Pipeline, input_file: Path, run_id: str, row_id: str | None = None, overrides: dict[str, dict[str, object]] | None = None) -> dict`

Execute a pipeline on a single audio file with optional parameter overrides.

**Parameters:**
- `project`: Loaded project config
- `pipeline`: Pipeline definition
- `input_file`: Path to input audio file
- `run_id`: Run identifier
- `row_id`: Optional row label (e.g., `row_0`, `row_1`) for matrix runs
- `overrides`: Step-level parameter overrides (`{step_name: {option: value}}`)

**Returns:** Dict with execution metadata including outputs and step logs.

## Usage Example

```python
from pathlib import Path
from triton.core import new_pipeline_run_id, run_pipeline_on_file
from triton.core.project import load_project_config, load_project_pipelines

project = load_project_config("my-project")
pipelines = load_project_pipelines("my-project")
pipeline = pipelines[0]

run_id = new_pipeline_run_id()
result = run_pipeline_on_file(
    project, 
    pipeline, 
    input_file=Path("my-project/data/raw/audio.wav"),
    run_id=run_id,
    overrides={"0": {"target_peak": 0.8}},  # Override step 0's target_peak
)

print(f"Run completed: {result}")
```

## Design Notes

- Each run is organized into isolated subdirectories to prevent output collisions.
- Parameter overrides are step-level (by index or name) for cleaner CLI/UI design.
- Run metadata is logged in `run.log.jsonl` for reproducibility and debugging.
