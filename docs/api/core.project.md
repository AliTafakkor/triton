# core.project

Project lifecycle and storage helpers shared by GUI and CLI.

## Project

Typed project model with lifecycle and convenience methods.

### Fields

- `name`
- `path`
- `sample_rate`
- `channel_mode`

### Methods

- `Project.load(project_dir)`
- `Project.create(project_dir, sample_rate, channel_mode)`
- `register_recent()`
- `list_files()`
- `add_file(filename, content)`
- `to_dict()`

## Pipeline

Typed pipeline model persisted in `triton.toml`.

### Fields

- `name`
- `steps` (ordered action keys)
- `step_options` (per-step settings keyed by step index)

### Pipeline helpers

- `load_project_pipelines(project_dir)`
- `save_project_pipelines(project_dir, pipelines)`

## Lifecycle helpers

- `create_project(project_dir, sample_rate, channel_mode)`
- `load_project_config(project_dir)`
- `initialize_project_tree(project_dir)`
- `write_project_config(project_dir, sample_rate, channel_mode)`

## Path helpers

- `project_config_path(project_dir)`
- `project_raw_dir(project_dir)`

## Recent projects

- `load_recent_projects()`
- `save_recent_projects(projects)`
- `register_recent_project(project_dir, project_name)`

## File management

- `list_project_files(project_dir)`
- `add_project_file(project_dir, filename, content)`
- `rename_project_file(file_path, new_name)`
- `delete_project_file(file_path)`
- `sanitize_filename(name)`

## Constants

- `PROJECT_CONFIG_NAME`
- `SUPPORTED_AUDIO_SUFFIXES`
