# Install

## pip

```bash
pip install "conch-triton[all]"
```

The package is published as **`conch-triton`** (the name `triton` is taken on PyPI by
NVIDIA's GPU compiler), but it is still imported as `triton` and the command is `triton`.

Optional extras keep the base install small. Pick what you need:

| Extra | Adds | Needed for |
|---|---|---|
| `transcribe` | `openai-whisper` | Transcribe tab, `triton transcribe` |
| `classify` | `torch`, `transformers` | Classify tab |
| `features` | `torch`, `transformers`, `openai-whisper` | Features tab (word onsets, neural layers) |
| `all` | all of the above | everything |

Without an extra, the rest of Triton still works; using a feature whose extra is
missing shows the exact `pip install` command to run.

`ffmpeg` is a system program and must be installed separately
(`brew install ffmpeg` on macOS, `sudo apt install ffmpeg` on Debian/Ubuntu).

## Pixi (for development)

[Pixi](https://pixi.sh) locks every dependency, including `ffmpeg`, to ensure
reproducibility across macOS, Linux and Windows.

1. Install Pixi: https://pixi.sh
2. Clone Triton and `cd` into the repo
3. Create the environment (includes all extras, tests and docs):
   ```bash
   pixi install
   ```

## Verify Installation

```bash
triton --help          # or: pixi run triton --help
triton gui             # or: pixi run gui
```

The GUI opens in your browser (usually `http://localhost:8501`). To try it without your own
recordings, run `triton examples download` or use **Add example files** in the GUI.

The first use of Transcribe, Classify or neural Features downloads the model weights
(~140 MB to ~1.5 GB depending on the model); later runs use the local cache.

## Publishing a release (maintainers)

Releases are published to PyPI by `.github/workflows/publish.yml` when a version tag is pushed,
using PyPI trusted publishing (no API token stored in the repo). One-time setup:

1. On pypi.org, create the `conch-triton` project's trusted publisher: owner `AliTafakkor`,
   repository `triton`, workflow `publish.yml`, environment `pypi`.
2. In the GitHub repo settings, create an environment named `pypi`.

Then, for each release: bump `version` in `pyproject.toml`, commit, and run
`git tag v0.1.0 && git push --tags`.
