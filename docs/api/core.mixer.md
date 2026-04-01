# core.mixer

Mix audio signals at a target Signal-to-Noise Ratio (SNR) using symmetric amplitude scaling and RMS normalization.

The mixer follows a robust 5-step procedure:

1. **Pre-mixing normalization**: Normalize speech and noise independently to the same RMS level
2. **Symmetric SNR splitting**: Boost signal by SNR/2 dB and attenuate noise by SNR/2 dB
3. **Weighted sum**: Mix as `mixed = speech × mult_signal + noise × mult_noise`
4. **Post-mixing RMS normalization**: Re-normalize result to target RMS to correct energy accumulation
5. **SNR preservation**: SNR is defined by the ratio of multipliers, independent of absolute signal level

## mix_at_snr

Mix speech and noise at a target SNR using symmetric RMS scaling.

**Signature**

```python
mix_at_snr(speech: np.ndarray, noise: np.ndarray, snr_db: float, 
           target_rms: float = 0.1) -> np.ndarray
```

**Parameters**

- `speech`: Speech signal (1D array)
- `noise`: Noise signal (1D array)
- `snr_db`: Target SNR in dB (symmetric split: signal boosted by SNR/2 dB, noise attenuated by SNR/2 dB)
- `target_rms`: Target RMS level for output normalization (default: 0.1, range 0.01–0.5 recommended)

**Returns**

Mixed audio normalized to `target_rms`.

**Details**

The multiplier applied to each component is computed as:

- `mult_signal = 10^(snr_db / 2 / 20)`  (boost by SNR/2 dB)
- `mult_noise = 10^(-snr_db / 2 / 20)`  (attenuate by noise/2 dB)

Both signal and noise are normalized independently before mixing, ensuring consistent energy levels regardless of input loudness. After mixing, the result is re-normalized to `target_rms`, which controls final loudness without affecting SNR.

**Example**

Mix a 16 kHz speech signal with noise at +10 dB SNR:

```python
from triton.core.mixer import mix_at_snr
import numpy as np

speech = np.random.randn(16000)  # 1 second at 16 kHz
noise = np.random.randn(16000)
mixed = mix_at_snr(speech, noise, snr_db=10)
```

---

## mix_at_snr_segmented

Mix multiple segments of audio at varying SNR levels with optional boundary smoothing.

**Signature**

`mix_at_snr(speech: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray`

```python
mix_at_snr_segmented(speech_segments: list[np.ndarray], 
                     noise_segments: list[np.ndarray],
                     snr_levels: list[float],
                     target_rms: float = 0.1,
                     smooth_transitions: bool = False,
                     transition_samples: int = 100) -> list[np.ndarray]
```

**Parameters**

- `speech_segments`: List of speech signal segments (1D arrays)
- `noise_segments`: List of noise signal segments (1D arrays, must match `speech_segments` length)
- `snr_levels`: List of SNR values in dB (must match segment count)
- `target_rms`: Target RMS level for each segment's output
- `smooth_transitions`: If `True`, smoothly interpolate amplitude multipliers across segment boundaries to avoid clicks or abrupt changes
- `transition_samples`: Number of samples to smooth across boundaries (default: 100)

**Returns**

List of mixed segments, each normalized to `target_rms`.

**Details**

This function is useful for processing long audio divided into chunks with varying degradation levels (e.g., different SNR per sentence in a speech corpus). When `smooth_transitions=True`, multiplier vectors are linearly interpolated across boundaries, creating a continuous amplitude envelope without audible discontinuities.

**Example**

Mix three sentences at different SNR levels with smooth transitions:

```python
from triton.core.mixer import mix_at_snr_segmented

speech_segments = [segment1, segment2, segment3]
noise_segments = [noise1, noise2, noise3]
snr_levels = [5, 10, 15]  # Progressively cleaner

mixed = mix_at_snr_segmented(
    speech_segments, 
    noise_segments,
    snr_levels,
    smooth_transitions=True
)
```

## mix_babble

Mix multiple talker waveforms into a babble signal.

**Signature**

`mix_babble(talkers: list[np.ndarray], target_rms: float | None = None, peak_normalize: bool = True, normalize_talkers: bool = True) -> np.ndarray`

**Args**

- `talkers`: List of talker arrays
- `target_rms`: Optional RMS target for each talker before mixing
- `peak_normalize`: Peak-normalize final mix
- `normalize_talkers`: Whether to RMS-normalize talkers before mixing

## mix_babble_from_segments

Normalize talker segments, concatenate per talker, then mix talkers.

**Signature**

`mix_babble_from_segments(talker_segments: list[list[np.ndarray]], target_rms: float | None = None, peak_normalize: bool = True) -> np.ndarray`

**Args**

- `talker_segments`: Nested list where each inner list is one talker's segments
- `target_rms`: Optional RMS target applied per segment before concatenation
- `peak_normalize`: Peak-normalize final mix