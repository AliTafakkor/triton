# Normalize

Normalize audio amplitude to target levels.

## Commands

### normalize peak

Normalize audio to target peak amplitude.

```bash
triton normalize peak <input> [OPTIONS]
triton normalize peak ./audio.wav --target 0.95
triton normalize peak ./audio_dir --output-dir ./normalized --target 0.98
```

**Options:**

- `--target`: Target peak amplitude (0.0–1.0). Default: `0.99`
  - Higher values preserve more of the original signal but risk clipping
  - Lower values enable more headroom for subsequent processing
  - Common values: 0.95–0.99

**Use cases:**

- Safety headroom before mixing
- Consistent peak levels across files
- Preventing clipping in processing chains

---

### normalize rms

Normalize audio to target RMS (energy/loudness) level.

```bash
triton normalize rms <input> [OPTIONS]
triton normalize rms ./audio.wav --target 0.1
triton normalize rms ./audio_dir --output-dir ./normalized --target 0.15
```

**Options:**

- `--target`: Target RMS amplitude (0.0–1.0). Default: `0.1`
  - Lower values = quieter normalization
  - Higher values = louder normalization
  - Common values: 0.05–0.2

**Use cases:**

- Preparing audio for noise mixing (control SNR consistently)
- Equalizing loudness across files before processing
- Speech processing pipelines (transcription, vocoding)
- Ensuring energetically-consistent datasets

**Why RMS over Peak?**

RMS normalization is **duration-independent** — a 1-second file and 10-second file with similar dynamics normalize to the same loudness regardless of length. This makes it ideal for:

- Datasets with variable-length files
- Noise mixing workflows (consistent SNR regardless of file length)
- ML model training (uniform loudness distribution)

## Examples

**Normalize all audio in a directory to peak:**

```bash
triton normalize peak ./raw_audio --output-dir ./normalized_peak --target 0.98
```

**Normalize to RMS for speech processing:**

```bash
triton normalize rms ./speech_files --output-dir ./normalized_speech --target 0.1
```

**Use in pipelines:**

In your `triton.toml`, include normalization as a pipeline step:

```toml
[[pipelines]]
name = "preprocess"
steps = [
    "normalize_rms",
    "resample_project",
    "to_mono"
]

[pipelines.step_options."0"]
target_rms = 0.1
```

Then run the pipeline on your project:

```bash
triton matrix run ./my_project preprocess matrix.csv
```
