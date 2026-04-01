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

`normalize_peak(audio, target=0.99) -> audio`

Scales audio so the maximum absolute sample value equals the target. Preserves dynamics.

**Parameters:**

- `audio` (np.ndarray): Audio signal
- `target` (float): Target peak amplitude, 0.0–1.0. Default: 0.99

**Returns:** Normalized audio array

**Use when:** You need a safety headroom before mixing, or consistent peak levels across files.

---

## normalize_rms

Normalize audio to target RMS (energy/loudness) level.

**Signature**

`normalize_rms(audio, target=0.1) -> audio`

Scales audio so the root mean square (RMS) amplitude equals the target. Duration-independent and perceptually meaningful.

**Parameters:**

- `audio` (np.ndarray): Audio signal
- `target` (float): Target RMS amplitude, 0.0–1.0. Default: 0.1

**Returns:** Normalized audio array

**Use when:** You need consistent loudness across files of different lengths, or preparing audio for noise mixing (control SNR reliably).

**Key difference from peak normalization:**

RMS normalization is **duration-independent**. A 1-second file and a 10-second file with the same loudness pattern will normalize to the same amplitude regardless of length. This makes it ideal for datasets with variable-length files and for controlling signal-to-noise ratio consistently in mixing workflows.

---

## rms

Compute root mean square.

**Signature**

`rms(signal, axis=-1) -> float`

## is_audio_file

Check if path is supported audio file.

## iter_audio_files

Iterate over audio files in path.

## SUPPORTED_EXTS

Set of supported audio extensions.
