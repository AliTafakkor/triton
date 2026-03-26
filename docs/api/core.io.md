# core.io

Audio I/O and file utilities.

## load_audio

Load an audio file.

**Signature**

`load_audio(path, sr=None, mono=True) -> (audio, sample_rate)`

## save_audio

Save an audio file.

**Signature**

`save_audio(path, audio, sr, source=None, actions=None, extra=None)`

Also writes a provenance sidecar JSON at `<path><suffix>.json`.

## write_sidecar

Write provenance sidecar metadata for any Triton-generated artifact.

**Signature**

`write_sidecar(path, source=None, actions=None, extra=None) -> Path`

## sidecar_path

Return sidecar path for a generated artifact.

**Signature**

`sidecar_path(path) -> Path`

## normalize_peak

Normalize audio to target peak amplitude.

**Signature**

`normalize_peak(audio, target=0.99)`

## rms

Compute root mean square.

**Signature**

`rms(signal, axis=-1)`

## is_audio_file

Check if path is supported audio file.

## iter_audio_files

Iterate over audio files in path.

## SUPPORTED_EXTS

Set of supported audio extensions.
