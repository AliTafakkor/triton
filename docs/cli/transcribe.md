# transcribe

Local transcription using Whisper.

## Usage

- `pixi run triton transcribe local <audio-or-dir> --output-dir outputs/transcripts --model tiny`

## Options

- `--model`: whisper model size (tiny, base, small, medium, large)
- `--device`: auto|cpu|cuda
- `--compute-type`: auto|float16|float32
- `--language`: language code (e.g., en)
- `--write-json`: write segments to JSON
