# Examples

Download example audio to try Triton without sourcing your own recordings: the
HARVARD speech-in-noise corpus from the University of Salford.

| File | Contents |
|---|---|
| `Harvard_speech100.wav` | 100 HARVARD sentences, one talker |
| `HARVARD100_SSN_52nd_order_lpc.wav` | Speech-shaped noise (same long-term spectrum as the speech) |
| `HARVARD100_SMN_52nd_order_lpc.wav` | Speech-modulated noise (speech-shaped noise following the speech envelope) |
| `white_noise_263s_Matlab10.wav` | White noise |

All files are 48 kHz mono 16-bit PCM, 263.3 s, ~25 MB each.

In the GUI, the same files can be added to a project from **Manage and Explore
Files → No audio handy? Add example files**. They are labeled `harvard-speech` and
`harvard-noise` so they can be selected by label in Mix, Babble and the pipeline
matrix.

## Commands

### examples list

List the example files, their source and license.

```bash
triton examples list
```

---

### examples download

Download the files and verify each one against a pinned size and SHA-256.

```bash
triton examples download
triton examples download --dest ./examples/audio
triton examples download --name Harvard_speech100.wav --name white_noise_263s_Matlab10.wav
```

**Options:**

- `--dest`: Directory to store the files. Default: `~/.cache/triton/examples`
- `--name`: Download only this file (repeatable). Default: all four.

**Behavior:**

- Files already present with the correct checksum are reused, not downloaded again.
- A download that is truncated or does not match its checksum is retried, then
  the command exits with status `1`. A partial file is never left in place, so a
  failed download can't be mistaken for a complete one.
- A file that is present but corrupted is replaced.

## Source and license

The files are downloaded from a frozen Figshare archive rather than bundled with
Triton, because they are licensed separately from Triton's MIT license:

> Demonte, P. (2019). HARVARD corpus Speech Shaped Noise and Speech Modulated
> Noise for SIN test. University of Salford.
> <https://doi.org/10.17866/rd.salford.c.4700054>

License: [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/): free to
share and adapt with attribution, **non-commercial use only**. Cite the source if
you use the files in published work.
