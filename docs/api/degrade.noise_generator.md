# degrade.noise_generator

Speech-noise generation utilities for degradation workflows.

## compute_ltass

Compute the long-term average speech spectrum (LTASS) from a corpus path, waveform, or iterable of paths/waveforms.

**Signature**

`compute_ltass(source, sr, n_fft=2048, hop_length=512)`

**Args**

- `source`: Corpus path, single waveform, or iterable of paths/waveforms
- `sr`: Sample rate used for loading/resampling
- `n_fft`: FFT size
- `hop_length`: STFT hop size

**Returns**

- `(freqs_hz, mean_power_spectrum)`

## generate_ssn

Generate speech-shaped noise (SSN) by shaping white noise to the LTASS of `shape_source`.

**Signature**

`generate_ssn(shape_source, length_samples, sr, n_fft=2048, hop_length=512, seed=None, normalize=True)`

**Args**

- `shape_source`: Corpus path or exact speech source(s) used for spectral shaping
- `length_samples`: Output noise length in samples
- `sr`: Output sample rate
- `n_fft`: FFT size for LTASS estimation
- `hop_length`: STFT hop size for LTASS estimation
- `seed`: Optional random seed
- `normalize`: Peak-normalize output to 0.99

## generate_babble

Generate babble noise from a folder containing one subfolder per talker.

**Signature**

`generate_babble(talker_root, length_samples, sr, n_talkers=8, seed=None, normalize=True)`

**Args**

- `talker_root`: Directory with talker subdirectories
- `length_samples`: Output babble length in samples
- `sr`: Output sample rate
- `n_talkers`: Number of talkers to combine
- `seed`: Optional random seed
- `normalize`: Peak-normalize output to 0.99

## ProjectBabbleResult

Result metadata returned by `generate_project_babble`.

**Fields**

- `audio`: Generated babble waveform
- `sample_rate`: Output sample rate
- `selected_groups`: Selected babble talker groups from project labels
- `planned_group_files`: Files actually used per talker after intended-length planning
- `short_source_labels`: Talker labels that lacked enough unique duration
- `unknown_duration_labels`: Talker labels where duration estimation failed
- `repeat_counts_by_label`: Random repeat counts added per short talker

## generate_project_babble

Generate babble from project files labeled with `bab-fN` / `bab-mN` and return audio plus planning metadata.

**Signature**

`generate_project_babble(project_dir, sr, channel_mode, num_talkers, num_female_talkers=None, num_male_talkers=None, intended_length_seconds=30.0, target_rms=0.1, peak_normalize=True, seed=None, max_workers=None, progress_callback=None)`

**Args**

- `project_dir`: Project directory containing label metadata
- `sr`: Output sample rate
- `channel_mode`: `mono` or `stereo`
- `num_talkers`: Total number of talker groups to mix
- `num_female_talkers`: Optional female group count override
- `num_male_talkers`: Optional male group count override
- `intended_length_seconds`: Target per-talker concatenated duration
- `target_rms`: RMS target applied per source segment before concatenation
- `peak_normalize`: Peak-normalize final output to safe headroom
- `seed`: Optional seed used for repeat/randomization decisions
- `max_workers`: Optional parallel loading worker limit
- `progress_callback`: Optional callback receiving `(message, percent)` updates

**Returns**

- `ProjectBabbleResult`

**Notes**

- This is the shared implementation used by both GUI and CLI babble workflows.
- When source duration is short, segments are repeated randomly to meet intended length.
- When intended length is already satisfied, extra files are skipped to reduce load time.
