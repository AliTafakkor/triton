# babble

Generate babble speech from project files that use the babble label convention.

## Overview

Babble generation now works on labeled talker groups instead of arbitrary file picks.
Use labels like `bab-f1`, `bab-f2`, `bab-m1`, and `bab-m2` to mark files that belong to a given talker group.

Both CLI and GUI call the same core function in `triton.degrade.noise_generator` so behavior stays consistent across interfaces.

The babble workflow is:

- select the number of talker groups to mix
- optionally set female and male counts separately
- normalize each source file to the target RMS
- concatenate all files for each selected talker group
- honor an intended per-talker length so extra files are skipped when not needed
- if a talker is short, repeat that talker's files randomly until target length is reached
- mix the resulting talker tracks and optionally peak-normalize the output

When Babble output is added back into a project from the GUI, Triton stores it as a derivative artifact labeled `bab-t#`, where `#` is the number of talkers used. The derivative is written with a provenance sidecar that records the source files and babble generation parameters.

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
  [--intended-length-seconds 30] \
  [--target-rms 0.1] \
  [--peak-normalize / --no-peak-normalize] \
  [--output-path outputs/babble.wav]
```

### Options

- `--num-talkers`: Total number of talker groups to mix
- `--num-female-talkers`: Number of female talker groups to use
- `--num-male-talkers`: Number of male talker groups to use
- `--intended-length-seconds`: Per-talker target duration; extra files are skipped once target is reached
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
  --intended-length-seconds 45 \
  --target-rms 0.1 \
  --output-path outputs/babble_4talkers.wav
```

## Workflow

1. Import the source files into a project, optionally assigning a single label to the full upload batch.
2. Label talker-group files with `bab-f1`, `bab-m1`, and similar labels.
3. Run `triton babble mix` with a total talker count.
4. Optionally set female and male counts explicitly.
5. Optionally set an intended per-talker duration.
6. Download the mixed babble output.

## Python API

```python
from pathlib import Path

from triton.degrade.noise_generator import generate_project_babble

result = generate_project_babble(
    Path("my-project"),
    sr=16000,
    channel_mode="mono",
    num_talkers=3,
    intended_length_seconds=30.0,
    target_rms=0.1,
    peak_normalize=True,
)

babble = result.audio
labels_used = [g.label for g in result.selected_groups]
```

## Notes

- Per-file RMS normalization happens before concatenation.
- Multiple files can share a babble label; they are normalized, concatenated, and treated as a single talker group.
- Intended-length planning avoids loading extra files when enough material already exists.
- If a talker does not have enough unique source material, files are repeated randomly to reach target length.
- Talker groups are selected from the project metadata, so files do not need to be copied anywhere else.
- GUI and CLI use the same core babble generation function and therefore share identical selection and mixing behavior.
- GUI babble generation can also persist the output back into the project with a sidecar so the derivative remains traceable.

## See Also

- [Files](files.md) - Label and organize project files
- [GUI](../gui.md) - Use babble generation from the Streamlit interface
- [Noise Generator API reference](../api/degrade.noise_generator.md)
