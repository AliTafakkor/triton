# degrade.noise_generator

Speech-noise generation utilities for degradation workflows.

## compute_ltass

Compute the long-term average speech spectrum (LTASS) from a corpus path, waveform, or iterable of paths/waveforms.

**Signature**

`compute_ltass(source, sr, n_fft=2048, hop_length=512)`

**Args**

- `source`: Corpus path, single waveform, or iterable of paths/waveforms
- `sr`: Sample rate used for loading/resampling
- `n_fft`: FFT size
- `hop_length`: STFT hop size

**Returns**

- `(freqs_hz, mean_power_spectrum)`

## generate_ssn

Generate speech-shaped noise (SSN) by shaping white noise to the LTASS of `shape_source`.

**Signature**

`generate_ssn(shape_source, length_samples, sr, n_fft=2048, hop_length=512, seed=None, normalize=True)`

**Args**

- `shape_source`: Corpus path or exact speech source(s) used for spectral shaping
- `length_samples`: Output noise length in samples
- `sr`: Output sample rate
- `n_fft`: FFT size for LTASS estimation
- `hop_length`: STFT hop size for LTASS estimation
- `seed`: Optional random seed
- `normalize`: Peak-normalize output to 0.99

## generate_babble

Generate babble noise from a folder containing one subfolder per talker.

**Signature**

`generate_babble(talker_root, length_samples, sr, n_talkers=8, seed=None, normalize=True)`

**Args**

- `talker_root`: Directory with talker subdirectories
- `length_samples`: Output babble length in samples
- `sr`: Output sample rate
- `n_talkers`: Number of talkers to combine
- `seed`: Optional random seed
- `normalize`: Peak-normalize output to 0.99
