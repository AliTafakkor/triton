# core.mixer

## mix_at_snr

Mix speech and noise at a target SNR using RMS scaling.

**Signature**

`mix_at_snr(speech: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray`

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
