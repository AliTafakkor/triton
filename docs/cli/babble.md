# babble

Generate babble speech from project files that use the babble label convention.

## Overview

Babble generation now works on labeled talker groups instead of arbitrary file picks.
Use labels like `bab-f1`, `bab-f2`, `bab-m1`, and `bab-m2` to mark files that belong to a given talker group.

The babble workflow is:

- select the number of talker groups to mix
- optionally set female and male counts separately
- normalize each source file to the target RMS
- concatenate all files for each selected talker group
- mix the resulting talker tracks and optionally peak-normalize the output

This is intended for speech-in-noise research, cocktail-party scenarios, and transcription stress testing.

## Label Convention

Babble looks for file labels that match:

- `bab-f<number>` for female talker groups
- `bab-m<number>` for male talker groups

Examples:

- `bab-f1`
- `bab-f2`
- `bab-m1`
- `bab-m2`

If multiple files share the same label, they are treated as one talker and concatenated in filename order after per-file RMS normalization.

## CLI

```bash
pixi run triton babble mix <project_dir> \
  --num-talkers <N> \
  [--num-female-talkers <F>] \
  [--num-male-talkers <M>] \
  [--target-rms 0.1] \
  [--peak-normalize / --no-peak-normalize] \
  [--output-path outputs/babble.wav]
```

### Options

- `--num-talkers`: Total number of talker groups to mix
- `--num-female-talkers`: Number of female talker groups to use
- `--num-male-talkers`: Number of male talker groups to use
- `--target-rms`: RMS applied to each source file before concatenation
- `--peak-normalize`: Normalize the final mix to safe headroom
- `--output-path`: Output file path, default `outputs/babble.wav`

If neither female nor male counts are provided, Triton balances them as evenly as possible. For an odd total, one sex gets the extra talker group.

### Example: balanced mix

```bash
pixi run triton files label my-project speaker_f1.wav "bab-f1"
pixi run triton files label my-project speaker_f2.wav "bab-f2"
pixi run triton files label my-project speaker_m1.wav "bab-m1"
pixi run triton files label my-project speaker_m2.wav "bab-m2"

pixi run triton babble mix my-project \
  --num-talkers 3 \
  --output-path outputs/babble_3talkers.wav
```

### Example: explicit female/male split

```bash
pixi run triton babble mix my-project \
  --num-talkers 4 \
  --num-female-talkers 2 \
  --num-male-talkers 2 \
  --target-rms 0.1 \
  --output-path outputs/babble_4talkers.wav
```

## Workflow

1. Import the source files into a project, optionally assigning a single label to the full upload batch.
2. Label talker-group files with `bab-f1`, `bab-m1`, and similar labels.
3. Run `triton babble mix` with a total talker count.
4. Optionally set female and male counts explicitly.
5. Download the mixed babble output.

## Python API

```python
from pathlib import Path

from triton.core.io import load_audio
from triton.core.mixer import mix_babble_from_segments
from triton.core.project import select_babble_talker_groups

selected_groups = select_babble_talker_groups("my-project", num_talkers=3)

# Load the files for each selected group, then pass a list of lists of arrays
talker_segments = []
for group in selected_groups:
  segments = []
  for file_path in group.files:
    audio, _ = load_audio(Path(file_path), sr=16000, mono=True)
    segments.append(audio)
  talker_segments.append(segments)

babble = mix_babble_from_segments(talker_segments, target_rms=0.1, peak_normalize=True)
```

## Notes

- Per-file RMS normalization happens before concatenation.
- Multiple files can share a babble label; they are normalized, concatenated, and treated as a single talker group.
- Talker groups are selected from the project metadata, so files do not need to be copied anywhere else.
- The GUI uses the same label convention and balancing rules as the CLI.

## See Also

- [Files](files.md) - Label and organize project files
- [GUI](../gui.md) - Use babble generation from the Streamlit interface
- [Core API reference](../api/core.mixer.md)
