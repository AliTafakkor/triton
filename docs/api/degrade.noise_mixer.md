# degrade.noise_mixer

Noise-mixing utilities for applying SNR-controlled degradation.

## add_noise

Mix speech with noise at a target SNR in dB.

**Signature**

`add_noise(speech, noise, snr_db)`

**Args**

- `speech`: Speech waveform
- `noise`: Noise waveform (tiled/cropped to speech length as needed)
- `snr_db`: Target signal-to-noise ratio in dB

**Returns**

- Peak-normalized mixed waveform
