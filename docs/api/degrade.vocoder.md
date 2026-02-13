# degrade.vocoder

Channel vocoder degradation.

## noise_vocode

Apply channel vocoding following Shannon et al. (1995).

Implements logarithmic band spacing, Butterworth filters, half-wave rectified Hilbert envelopes, low-pass smoothing, and carrier modulation.

**Signature**

`noise_vocode(audio, sr, n_bands=8, freq_range=(200, 8000), envelope_cutoff=160.0, vocoder_type="noise", filter_order=3)`

**Args**

- `vocoder_type`: "noise" or "sine"
- `n_bands`: Number of spectral channels
- `freq_range`: (low_hz, high_hz)
- `envelope_cutoff`: Low-pass cutoff for envelope (Hz)
- `filter_order`: Butterworth filter order
