# convert

Format conversion utilities.

## mono

Convert stereo to mono.

- `pixi run triton convert mono <audio-or-dir> --method mean`

### Options

- `--method`: mean, left, or right
- `--output-dir`: output directory

## stereo

Convert mono to stereo.

- `pixi run triton convert stereo <audio-or-dir> --method duplicate`

### Options

- `--method`: duplicate or silence
- `--output-dir`: output directory

## resample

Change sample rate.

- `pixi run triton convert resample <audio-or-dir> --target-sr 16000`

### Options

- `--target-sr`: target sample rate in Hz
- `--output-dir`: output directory

## quantize

Change bit depth.

- `pixi run triton convert quantize <audio-or-dir> --bit-depth 16`

### Options

- `--bit-depth`: 8, 16, 24, or 32
- `--output-dir`: output directory
