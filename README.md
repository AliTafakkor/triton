# Triton 🔱🐚
An audio signal processing toolkit for speech-in-noise research.

Triton is a modular audio utility designed to standardize stimuli preparation and signal degradation. It provides a robust, reproducible "vessel" for audio manipulation, bridging the gap between raw signal math and accessible lab tools.

## Core Components
The Engine (/core): The foundational Python API for audio math. Contains logic for RMS-based SNR mixing, vocoding, and filtering. Designed to be imported directly into other simulation or modeling projects to ensure consistent signal processing.

The CLI: Built for high-volume batch processing. Allows for rapid transformation of entire audio directories (e.g., degrading a full sentence set to -5dB SNR) via the terminal.

The HTML GUI: A Streamlit-based web dashboard. Offers a drag-and-drop interface for lab members to test degradations, visualize waveforms, and download processed files without writing code.

Why Pixi?
Reproducibility is the priority. Triton uses Pixi to lock the environment, ensuring that heavy dependencies—like ffmpeg and librosa—behave identically across different operating systems and machines.
