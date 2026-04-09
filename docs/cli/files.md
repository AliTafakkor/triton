# files

Manage project audio files with labeling, filtering, and normalization.

## Overview

The `files` command provides project-aware file management:

- **Label files** with one or more custom tags (e.g., `"bab-f1"`, `"background"`)
- **Filter files** by label in both CLI and GUI
- **View labels** across all project files
- **Organize assets** for easier downstream processing

Labels persist in `metadata/file_labels.json`. A file can have multiple labels, stored as a list.

## Commands

### Label a file

```bash
pixi run triton files label <project_dir> <filename> "<label>"
```

Assign a label to a file. Labels are appended to the existing label set.

**Examples:**

```bash
pixi run triton files label my-project speaker_female.wav "bab-f1"
pixi run triton files label my-project speaker_female.wav "studio"    # adds second label
pixi run triton files label my-project office_noise.wav "background"
```

### List files

```bash
pixi run triton files list <project_dir> [--label "<label>"]
```

List all project files, optionally filtered by label.

**Examples:**

```bash
# All files
pixi run triton files list my-project

# Only babble talker groups
pixi run triton files list my-project --label "bab-f1"
```

### Show all labels

```bash
pixi run triton files show-labels <project_dir>
```

Display all file-to-label mappings in the project.

**Output:**

```
File labels:

  speaker_female.wav: bab-f1, studio
  speaker_male.wav: bab-m1
  office_noise.wav: background

Total labeled files: 3
```

### Delete all files with a label

```bash
pixi run triton files delete-label <project_dir> "<label>" [--yes]
```

Delete every file that has the given label.

- Removes matching files from `data/normalized/` and `data/derived/`
- Removes matching raw counterparts from `data/raw/` (same stem)
- Removes provenance sidecars (`.json`) and spectrogram files (`.spectrogram.npz`) for each deleted file
- Removes label entries for deleted files from `metadata/file_labels.json`

**Examples:**

```bash
# Prompt for confirmation
pixi run triton files delete-label my-project "background"

# Non-interactive (CI/scripts)
pixi run triton files delete-label my-project "bab-f1" --yes
```

## GUI File Management

In the **Manage and Explore Files** tab:

### Import and Normalize

When files are imported:
1. The original file is saved as-is to `data/raw/`.
2. A normalized copy is automatically generated to `data/normalized/` matching the project spec:
   - Resampled to the project sample rate
   - Converted to the project channel mode (mono/stereo)
   - Saved with the project bit depth and file format
3. A spectrogram artifact is precomputed for the normalized file.

Generated artifacts (e.g. babble mixes saved via **Add to project**) are written to `data/derived/` and also appear in the file list.

### Label Files

- **Batch label on import**: Apply one label to all files uploaded together.
- **Per-file label**: The Label column accepts comma-separated labels, e.g. `talker1, bab-f1`.
- **Filter by label**: Use the dropdown above the file list. Select `(None)` to show only unlabeled files.
- **Bulk label unlabeled files**: In the **Manage Labels** expander, apply a label to all files that have no labels.

### Manage Labels

The **Manage Labels** expander (above the file list) provides:

- **Rename**: Select an existing label (shown with its file count) and rename it across all files that carry it. On multi-label files only the target label is replaced; other labels are preserved.
- **Bulk label unlabeled files**: Enter a label name and apply it to all files that currently have no labels.

## Python API

```python
from triton.core.project import (
  delete_project_files_by_label,
    set_file_labels,
    get_file_labels,
    list_project_files,
    load_file_labels,
    normalize_project_file,
    list_normalized_project_files,
)

# Assign multiple labels
set_file_labels(project_dir, file_path, ["bab-f1", "studio"])

# Get all labels for a file
labels = get_file_labels(project_dir, file_path)  # → ["bab-f1", "studio"]

# Filter files by label
talker_files = list_project_files(project_dir, label="bab-f1")

# Delete all files with a label
deleted = delete_project_files_by_label(project_dir, "background")

# Load all labels (keyed by stem, not filename)
labels = load_file_labels(project_dir)
# → {"speaker": ["bab-f1", "studio"], "noise": ["background"]}

# List normalized files
norm_files = list_normalized_project_files(project_dir)
```

## Storage

```
my-project/
  data/
    raw/              ← original imported files (any format)
    normalized/       ← converted to project spec on import
    derived/          ← pipeline outputs
  metadata/
    file_labels.json  ← label store
```

`file_labels.json` format (keys are file stems, no extension):

```json
{
  "speaker_female": ["bab-f1", "studio"],
  "speaker_male": ["bab-m1"],
  "office_noise": ["background"]
}
```

## Notes

- Labels are optional — files can exist without a label.
- Labels are case-sensitive.
- A file can have any number of labels; use comma-separated values in the GUI.
- The `(None)` filter in the GUI shows only files with no labels.
- Babble generation scans all labels on each file for the `bab-f<n>` / `bab-m<n>` pattern.
- Normalization runs automatically on import; existing raw files are not retroactively normalized.
- Deleting a file removes its raw source, its normalized copy, and all companion files (sidecar `.json`, spectrogram `.spectrogram.npz`). Label entries are not automatically pruned by `delete_project_file` but are cleaned up by `delete_project_files_by_label`.
