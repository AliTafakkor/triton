# core.spectrogram

Spectrogram computation and persistence helpers.

## SpectrogramResult

Typed spectrogram result container.

### Fields

- `kind`
- `values`
- `freqs`
- `times`

## compute_spectrogram

Compute a spectrogram from waveform and settings.

**Signature**

`compute_spectrogram(audio, sr, settings) -> SpectrogramResult`

### Supported types

- `stft`
- `mel`
- `cqt`

## normalize_spectrogram_settings

Merge and validate spectrogram settings.

**Signature**

`normalize_spectrogram_settings(settings) -> dict`

## save_spectrogram

Persist a computed spectrogram to compressed `npz`.

**Signature**

`save_spectrogram(path, result, settings)`

## load_spectrogram

Load persisted spectrogram artifact and settings.

**Signature**

`load_spectrogram(path) -> (SpectrogramResult, settings)`

## Constants

- `SPECTROGRAM_TYPES`
- `DEFAULT_SPECTROGRAM_SETTINGS`
