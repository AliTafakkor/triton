# files

Manage project audio files with labeling and filtering.

## Overview

The files command provides project-aware file management including:

- **Label files** with custom tags (e.g., "bab-f1", "bab-m1", "background")
- **Filter files** by label in both CLI and GUI
- **View labels** across all project files
- **Organize assets** for easier downstream processing

Labels persist in `metadata/file_labels.json` and work across CLI and GUI interfaces.

## Commands

### Label a file

```bash
pixi run triton files label <project_dir> <filename> "<label>"
```

Assign a label to a file. Use meaningful labels like:
- `"bab-f1"`, `"bab-m1"` - for babble talker groups
- `"speaker_f"`, `"speaker_m"` - for general speech talkers
- `"noise"`, `"traffic"`, `"background"` - for background sounds
- `"reference"` - for anchors or standard files

**Example:**

```bash
pixi run triton files label my-project speaker_female.wav "talker"
pixi run triton files label my-project speaker_male.wav "talker"
pixi run triton files label my-project office_noise.wav "background"
```

### List files

```bash
pixi run triton files list <project_dir> [--label "<label>"]
```

List all project files, optionally filtered by label.

**Example:**

```bash
# All files
pixi run triton files list my-project

# Only babble talker groups
pixi run triton files list my-project --label "bab-f1"

# Only background noise
pixi run triton files list my-project --label "background"
```

### Show all labels

```bash
pixi run triton files show-labels <project_dir>
```

Display a summary of all labels and the files they tag.

**Example:**

```bash
pixi run triton files show-labels my-project
```

Output:
```
bab-f1 (2 files):
  - speaker_female_part1.wav
  - speaker_female_part2.wav
background (1 files):
  - office_noise.wav
(no label) (1 files):
  - mixed_speech.wav
```

## GUI File Management

In the **Manage and Explore Files** tab:

### Import and Label Files

1. **Apply a batch label during import**: When you upload multiple files, you can assign the same label to all of them at once.
2. **View labels**: A "Label" column shows the assigned label for each file.
3. **Add labels**: Click a file's label cell to edit and assign a new label.
4. **Filter by label**: Use the dropdown filter above the file list to show only files with a specific label.

### Rename Labels

Use the **Rename Labels** expander to rename a label across all files that have it:

1. Select an existing label from the dropdown
2. Enter a new label name
3. Click "Rename"
4. All files with the old label are updated to the new label

This is useful for:
- Correcting label typos across multiple files
- Reorganizing labels (e.g., from generic `talker` to `bab-f1`)
- Preparing files for babble generation with consistent naming

Labels update instantly and persist to disk.

## Use Cases

### Speech-in-noise mixing

```bash
# Label babble talker groups
triton files label my-project talker1_part1.wav "bab-f1"
triton files label my-project talker1_part2.wav "bab-f1"
triton files label my-project talker2.wav "bab-m1"
triton files label my-project talker3.wav "bab-m2"

triton files label my-project office_noise.wav "noise"

# Later: mix all talker groups together
triton babble mix my-project \
  --num-talkers 3 \
  --output-path outputs/babble.wav

# Or use in GUI: filter by "bab-f1" or "bab-m1" to inspect a talker group
# or apply one label to an uploaded batch during import
```

### Multi-format ingest

```bash
# Label files by source
triton files label my-project podcast_ep1.wav "podcast"
triton files label my-project podcast_ep2.wav "podcast"
triton files label my-project audiobook.wav "audiobook"

# Process only podcasts
triton files list my-project --label "podcast"
```

### Reference and comparison

```bash
# Tag artifacts for comparison runs
triton files label my-project baseline_output.wav "reference"
triton files label my-project experiment_output.wav "experiment"
```

## Python API

```python
from triton.core.project import (
    set_file_label,
    get_file_label,
    list_project_files,
    load_file_labels,
)

# Assign a label
set_file_label(project_dir, "speaker.wav", "bab-f1")

# Get label for a file
label = get_file_label(project_dir, "speaker.wav")  # → "bab-f1"

# List files with a specific label
talker_files = list_project_files(project_dir, label="bab-f1")
# → ["speaker.wav", ...]

# Load all labels as a dict
labels = load_file_labels(project_dir)
# → {"speaker.wav": "bab-f1", "noise.wav": "background", ...}
```

## Storage

Labels are stored in `metadata/file_labels.json` under your project:

```json
{
  "speaker_female.wav": "bab-f1",
  "speaker_male.wav": "bab-m1",
  "office_noise.wav": "background"
}
```

This file is managed automatically; edits via CLI or GUI update it immediately.

## Notes

- Labels are optional—files can exist without a label.
- Labels are case-sensitive.
- Labels persist across CLI and GUI sessions.
- Removing a file from the project does not automatically remove its label entry.
- Babble generation looks for labels in the form `bab-f<number>` and `bab-m<number>`.
- Multiple files can share the same label; babble treats them as one talker group and concatenates them after RMS normalization.
