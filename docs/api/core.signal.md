# core.signal

Signal processing utilities.

## extract_envelope

Extract amplitude envelope from audio.

**Signature**

`extract_envelope(audio, sr, method="hilbert", cutoff=160.0, filter_order=4)`

## bandpass_filter

Apply band-pass filter.

**Signature**

`bandpass_filter(audio, low, high, sr, order=3)`
