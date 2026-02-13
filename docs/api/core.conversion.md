# core.conversion

Audio format conversion utilities.

## to_mono

Convert stereo to mono.

**Signature**

`to_mono(audio, method="mean")`

## to_stereo

Convert mono to stereo.

**Signature**

`to_stereo(audio, method="duplicate")`

## resample

Resample audio to target sample rate.

**Signature**

`resample(audio, orig_sr, target_sr)`

## requantize

Change bit depth (quantization).

**Signature**

`requantize(audio, bit_depth)`
