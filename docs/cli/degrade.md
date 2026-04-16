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

## add-noise

Add noise to speech files/directories at a target SNR.

- `pixi run triton degrade add-noise <audio-or-dir> --noise-type auto --snr-db -5 --noise-file outputs/babble.wav`

### Options

- `--noise-type`: `auto`, `babble`, `white`, `colored`, or `ssn` (default `auto`).
- `--noise-file`: Optional noise file. Required when `--noise-type babble`.
- `--snr-db`: Target signal-to-noise ratio in dB.
- `--seed`: Optional seed for deterministic random segment selection.
- `--sr`: Optional resample rate for both speech and file-backed noise.
- `--output-dir`: Output directory (default `outputs/noisy`).

### Behavior notes

- Babble is file-backed only in `add-noise` and is never generated there.
- With `--noise-type auto`, file stems matching `bab-tN` (for example `bab-t8.wav`) are auto-recognized as multitalker babble sources.
- If noise is longer than speech, Triton randomly crops an equal-length noise segment before SNR mixing.
