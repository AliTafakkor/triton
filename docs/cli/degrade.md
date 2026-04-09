# degrade

Apply audio degradations for psychoacoustic research.

## Python API additions

Triton also includes degradation utilities for generating and mixing noise in Python:

- `triton.degrade.noise_generator` for LTASS, SSN, and babble generation
- `triton.degrade.noise_mixer` for SNR-based noise addition

## ramp

Apply a fade-in and/or fade-out ramp envelope to audio files.

- `pixi run triton degrade ramp <audio-or-dir> --ramp-start 0.05 --ramp-end 0.05 --shape cosine`

### Options

- `--ramp-start`: Fade-in duration in seconds (default `0.05`).
- `--ramp-end`: Fade-out duration in seconds (default `0.05`).
- `--shape`: Ramp shape — `linear`, `exponential`, `logarithmic`, or `cosine` (default `cosine`).
- `--output-dir`: Output directory (default `outputs/ramped`).

### Ramp shapes

| Shape | Behaviour |
|---|---|
| `linear` | Uniform gain sweep. |
| `exponential` | Slow start, fast finish. |
| `logarithmic` | Fast start, slow finish. |
| `cosine` | Smooth S-shaped half-cosine transition (default). |

## vocode

Apply channel vocoding (Shannon et al., 1995).

- `pixi run triton degrade vocode <audio-or-dir> --vocoder-type noise --n-bands 8`

### Options

- `--vocoder-type`: noise or sine
- `--n-bands`: number of frequency bands (degradation level)
- `--freq-low`: low frequency cutoff (Hz, default 200)
- `--freq-high`: high frequency cutoff (Hz, default 8000)
- `--envelope-cutoff`: envelope low-pass cutoff (Hz, default 160)
- `--filter-order`: Butterworth filter order (default 3)
- `--output-dir`: output directory
