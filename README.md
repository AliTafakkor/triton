# Triton 🔱🐚
An audio signal processing toolkit for speech-in-noise research.

Triton is a modular audio utility designed to standardize stimuli preparation and signal degradation. It provides a robust, reproducible "vessel" for audio manipulation, bridging the gap between raw signal math and accessible lab tools.

## Product Direction
Triton should evolve toward a project-centric workflow. Instead of treating every command as a one-off file transform, the system should organize work around a project that defines the audio contract for everything inside it.

In that model, a user starts by creating a project and choosing settings such as target sampling rate, channel format (mono or stereo), storage paths, and default processing behavior. After that, any audio brought into the project is normalized to the project specification automatically. The goal is to make downstream operations predictable: degradation, transcription, conversion, mixing, and ingest should all operate on a consistent internal representation.

## Proposed Workflow
1. Create a project.
2. Save project settings, including sample rate, mono/stereo policy, and output layout.
3. Import local files or ingest external sources into that project.
4. Convert imported media automatically to the project specification.
5. Run downstream tasks such as degrade, transcribe, convert, and mix against project-managed assets.
6. Persist outputs and metadata back into the project so runs are reproducible.

## Development Plan
### Phase 1: Define the Project Model
- Introduce a first-class project concept with a project root and a machine-readable config file.
- Store canonical settings such as sample rate, channel mode, preferred file format, and directory layout.
- Add a lightweight metadata layer for tracking imported assets and derived outputs.

Possible project structure:

```text
my-project/
	triton.toml
	files/
		raw/
		normalized/
		derived/
	metadata/
		assets.json
		runs.json
```

The config should answer a simple question for every command: what is the canonical representation of audio in this project?

### Phase 2: Add Project Lifecycle Commands
- Add CLI commands to create, inspect, and update projects.
- Example direction:
	- `triton project init my-project --sr 16000 --channels mono`
	- `triton project show my-project`
	- `triton project set my-project --sr 24000`
- Ensure all file-oriented commands can resolve the active project explicitly or from the current working directory.

### Phase 3: Make Import the Entry Point
- Add an import workflow for local media.
- Imported files should be copied or linked into project storage, validated, and converted to canonical project settings.
- Preserve provenance metadata such as original path, original sample rate, channel count, and conversion history.

This is the step that turns Triton from a toolbox into a managed pipeline.

### Phase 4: Treat Ingest as a Project-Aware Import Source
- Extend ingest so RSS or other external sources feed directly into a project.
- Downloaded audio should land in raw storage first, then be normalized into project-managed assets.
- Attach source metadata such as feed URL, episode title, publication date, and retrieval timestamp.

In practice, ingest should become one way to add files to a project rather than a separate, disconnected feature.

### Phase 5: Make Processing Commands Project-Native
- Update degrade, transcribe, convert, and mix so they can target project assets by logical identifiers instead of only filesystem paths.
- Standardize where derived outputs are written and how they are named.
- Record processing parameters so results can be reproduced later.

Example direction:

```bash
triton degrade vocode --project my-project --input asset:sentence_001 --preset noise8
triton transcribe local --project my-project --input asset:moth_episode_03
```

### Phase 6: Add Reproducibility and UX Improvements
- Support presets at the project level for common transformations.
- Add manifest or run-history views so users can see what has been imported and produced.
- Introduce safe overwrite rules, duplicate handling, and validation warnings when settings change.
- Consider a future GUI around the same project abstraction instead of building separate logic for the interface layer.

## Design Principles
- One project defines one canonical audio specification.
- Import and ingest should normalize data at the boundary, not leave format cleanup to later commands.
- Downstream processing should operate on stable project assets, not ad hoc file paths whenever possible.
- Metadata should be captured automatically so provenance and reproducibility do not depend on user memory.
- CLI and future GUI should share the same project model and storage conventions.

## Immediate Implementation Priorities
1. Define `triton.toml` and the on-disk project layout.
2. Implement `triton project init` and project resolution helpers.
3. Add a project-aware import command that converts files to canonical settings.
4. Refactor ingest to write into projects through the same import pipeline.
5. Update degrade and transcribe commands to accept `--project` and operate on normalized assets.

## Core Components
The Engine (/core): The foundational Python API for audio math. Contains logic for RMS-based SNR mixing, vocoding, and filtering. Designed to be imported directly into other simulation or modeling projects to ensure consistent signal processing.

The CLI: Built for high-volume batch processing. Allows for rapid transformation of entire audio directories (e.g., degrading a full sentence set to -5dB SNR) via the terminal.

The HTML GUI: A Streamlit-based web dashboard. Offers a drag-and-drop interface for lab members to test degradations, visualize waveforms, and download processed files without writing code.

Why Pixi?
Reproducibility is the priority. Triton uses Pixi to lock the environment, ensuring that heavy dependencies—like ffmpeg and librosa—behave identically across different operating systems and machines. A single `pixi.lock` file works seamlessly on both macOS (Apple Silicon) and Linux, enabling smooth collaboration and cross-platform development.

## Pipeline Output Layout

Pipeline runs are written under project-derived storage with explicit run and step boundaries:

```text
<project>/
	data/
		derived/
			pipelines/
				<pipeline_name_key>/
					run_<UTC timestamp>_<id>/
						step_01_<step_key>/
							<input_stem>.wav
							<input_stem>.wav.json
						step_02_<step_key>/
							<input_stem>.wav
							<input_stem>.wav.json
						...
```

- Each click of "Run selected pipeline" creates a distinct `run_<...>` folder.
- Each step writes outputs into its own `step_<index>_<name>` folder.
- The final output for a file is the artifact from the last step folder.

## Sidecar Provenance Metadata

Any file generated by Triton should include a JSON sidecar next to the artifact. Sidecars use `<artifact_name><artifact_suffix>.json`, for example:

- `output.wav` -> `output.wav.json`
- `transcript.txt` -> `transcript.txt.json`

Each sidecar captures:

- artifact identity (path/name/suffix)
- source reference (source path or source URL/feed)
- ordered action history with options/details
- generation timestamp and schema version

This makes it possible to trace what file was produced from which source and by what sequence of operations.

## Spectrograms

Triton now supports project-level default spectrogram settings and import-time spectrogram generation.

Project defaults live in `triton.toml` under `[spectrogram]`:

- `type` (`stft`, `mel`, `cqt`)
- `n_fft`, `hop_length`, `win_length`, `window`
- `n_mels`, `fmin`, `fmax`, `power`

When audio is imported in the GUI Import tab, Triton computes and stores a spectrogram artifact per file using those defaults.

Artifacts:

- `<file>.spectrogram.npz`
- `<file>.spectrogram.npz.json` (sidecar provenance)

The `Spec` action in the Import list opens the precomputed artifact in the right-side spectrogram panel rather than recomputing every render.
