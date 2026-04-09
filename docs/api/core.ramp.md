# core.ramp

Fade-in / fade-out envelope utilities.

## apply_ramp

Apply a ramp envelope to the beginning and/or end of an audio signal.

**Signature**

`apply_ramp(audio, sr, *, ramp_start=0.0, ramp_end=0.0, shape="cosine")`

**Args**

- `audio`: Input waveform as a float32 NumPy array — mono (1-D) or multi-channel `(n_samples, channels)`.
- `sr`: Sample rate in Hz.
- `ramp_start`: Fade-in duration in seconds (`0` = no fade-in).
- `ramp_end`: Fade-out duration in seconds (`0` = no fade-out).
- `shape`: Ramp shape — one of `linear`, `exponential`, `logarithmic`, or `cosine`.

**Returns**

A copy of `audio` with the ramp envelope applied.

**Raises**

- `ValueError`: If `shape` is unrecognised, if either duration is negative, or if the combined ramp duration exceeds the audio length.

## RAMP_SHAPES

Tuple of valid shape name strings: `("linear", "exponential", "logarithmic", "cosine")`.

## Ramp shapes

| Shape | Behaviour |
|---|---|
| `linear` | Uniform gain sweep — ramp gain increases proportionally with time. |
| `exponential` | Slow start, fast finish — gain accelerates toward the target level. |
| `logarithmic` | Fast start, slow finish — gain accelerates away from silence. |
| `cosine` | Smooth S-shaped half-cosine transition (default) — perceptually smooth. |
