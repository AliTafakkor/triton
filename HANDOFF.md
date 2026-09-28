# Handoff: summer 2026

Work by Kamyar Modabber (June–September 2026), handed back to Ali Tafakkor. Everything
below is merged into `main`; every change went through an issue and a PR.

## State of `main`

- `pixi run pytest`: 59 passed, 1 skipped. `ruff check --select F`: clean.
  `mkdocs build --strict`: clean.
- Every GUI tab was driven end to end on the HARVARD example files before handoff (import,
  spectrogram + playback, RSS preview, a pipeline using all 10 step types, a 2-row matrix,
  babble → add to project, transcribe, classify, features). Mix uses file-upload widgets that
  can't be scripted, so its mixing functions were checked directly: SNR accurate to 0.01 dB.
- The wheel was installed with `pip` into a clean environment: the CLI, `triton gui` and all
  tabs work without the optional extras, and missing extras show the `pip install` command.

## What changed

| Area | PRs | Summary |
|---|---|---|
| Bug fixes | #18, #20, #22, #26, #34 | Stereo spectrogram axis; RSS partial downloads + URL-encoded names; rename orphaning spectrograms; m4a crash on macOS; labeling files in `data/derived` (babble) crashed |
| Classify tab | #24 | AST (AudioSet 527 classes) tagging; top label added to the file's labels |
| Packaging | #31 | `conch-triton` on PyPI (import name stays `triton`), extras, trusted-publishing workflow |
| GUI fixes | #32, #33 | Audio player in the file library; pipeline-matrix label filter |
| Cleanup | #34 | ~30 unused imports/variables removed |
| Feature extraction | #35 | Features tab: envelope, word onsets, Wav2Vec2/HuBERT/WavLM layers (#30) |
| Example audio | #36 | `triton examples download` + GUI button; HARVARD corpus, SHA-256 verified (#28) |
| Final pass | this PR | `triton gui` command; CLI no longer crashes without Whisper; Classify keeps existing labels; docs |

## Decisions for you

1. **`core.signal.extract_envelope(method="hilbert")` leaks energy before sound onsets.** It
   half-wave rectifies *before* the Hilbert transform; the DC offset that creates makes the
   envelope rise ahead of the sound (≈18 % of the loud level in the silence before a tone
   burst, vs < 1 % for the standard `|hilbert(x)|`). The Features tab uses the standard
   method (`triton/features/acoustic.py`). I did **not** change the core function because the
   vocoder depends on it, so whether to change it is your call.
2. **The `triton` import name still clashes with NVIDIA's `triton`.** PyTorch detects the GPU
   compiler with a bare `import triton`, finds this package, and crashes. The workaround
   (`triton/_torch_compat.py`) makes that check look for `triton.language` and is applied
   before loading transformers. The permanent fix is renaming the package to `conch_triton`,
   which touches every import in the repo.
3. **PyPI is configured but nothing is published yet.** One-time setup is in
   [docs/install.md → Publishing a release](docs/install.md#publishing-a-release-maintainers).
   After that, `git tag v0.1.0 && git push --tags` publishes.
4. **Example audio is CC BY-NC 4.0** (non-commercial), so it is downloaded from Figshare, not
   bundled with the MIT-licensed package. Citation is shown wherever it's used.

## Next steps

- **#30 (open): rest of feature extraction.** Surprisal (word predictability from a language
  model, e.g. GPT-2, aligned to the word onsets), phoneme and syllable onsets (needs forced
  alignment, e.g. Montreal Forced Aligner or WhisperX), and more models: YAMNet (TensorFlow)
  and AST. AST works on 2D spectrogram patches and only reads 10 s, so it needs a different
  time mapping than the wav2vec2-family models. `triton/features/` is organized so each is a
  new module + a checkbox in `gui/tabs/features.py`.
- **#27 (open): finer-grained speech labels** (speaker sex, clean vs. noisy, …).
- **Features on long files:** all 13 layers of a 5-minute file is ~600 MB. Options: store
  float16, or add an optional resample of activations to the envelope rate.
- **Features CLI:** extraction is GUI-only; a `triton features` command would allow batch
  runs on a cluster. The core functions in `triton/features/` are ready for it.
- **Harvard sentences as separate files:** `Harvard_speech100.wav` is 100 sentences in one
  file; splitting on silence would give per-sentence stimuli.

## Where things live

```text
src/triton/
  _torch_compat.py        NVIDIA-triton name-clash workaround (used by classify + features)
  classify/ast.py         AST classification
  features/               acoustic.py, words.py, neural.py, storage.py
  examples.py             example-audio manifest + verified downloader
  cli/examples.py         `triton examples`
  gui/tabs/classify.py    Classify tab
  gui/tabs/features.py    Features tab
tests/test_features.py, tests/test_examples.py
```
