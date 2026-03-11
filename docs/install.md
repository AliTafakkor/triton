# Install

## Pixi (recommended)

[Pixi](https://pixi.sh) locks dependencies to ensure reproducibility across all platforms (macOS, Linux, Windows).

### Setup

1. Install Pixi: https://pixi.sh
2. Clone Triton and navigate to the repo
3. Create the default environment:
   ```bash
   pixi install
   ```

### Optional Features

The default environment includes core audio utilities and testing. Optional features can be installed separately:

- **Transcription** (`transcribe` feature): Adds `openai-whisper` for local audio transcription
  ```bash
  pixi install --feature transcribe
  ```

### Cross-Platform Development

Triton supports both **macOS (Apple Silicon)** and **Linux** in a single environment. When developing on either platform, Pixi automatically selects the correct dependencies.

## Verify Installation

```bash
pixi run triton --help
```

This should display the Triton CLI help text.

You can also verify the GUI:

```bash
pixi run gui
```

This should start Streamlit and print a local URL (usually `http://localhost:8501`).

