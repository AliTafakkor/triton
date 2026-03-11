# degrade

Apply audio degradations for psychoacoustic research.

## Python API additions

Triton also includes degradation utilities for generating and mixing noise in Python:

- `triton.degrade.noise_generator` for LTASS, SSN, and babble generation
- `triton.degrade.noise_mixer` for SNR-based noise addition

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
