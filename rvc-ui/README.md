# rvc-ui

Qt6 Widgets desktop front-end for the RVC-GGML toolchain.

The first tab is the **Song Cover** one-click flow:

```
song.wav
  -> rvc-sep <vocal separator>                        (models/roformers/vocal_separator)
  -> [optional] rvc-sep <background separator>        (models/roformers/backup_separator)
  -> RVC-GGML conversion of the vocals (and background) with the target voice
  -> [optional] pitch-shift the instrumental (sox, or ffmpeg as fallback)
  -> mix (main / instrumental / background gains) -> output WAV
```

Model folders are discovered from the repository `models/` tree:

| Folder | Used for |
|---|---|
| `models/voice_models/<name>/<name>.gguf` (+ `.index.gguf`) | target voice (RVC character model) |
| `models/roformers/vocal_separator/*.gguf` | separating vocals from the song |
| `models/roformers/backup_separator/*.gguf` | optional background-singer split (Male-Female) |
| `models/extractors/*.gguf` | RVC feature extractor (default: contentvec) |
| `models/f0/*.gguf` | RVC f0 extractor (default: rmvpe) |

## Multi-step tab

Every generation step can be run in isolation: separate (vocals / instrumental),
split background singers, convert (RVC), pitch-shift, and mix (up to three
tracks with gains).

- Inputs: any step's input can be a file via `<browse file...>` or **any other
  step's output** (the shared source list) — arbitrary piping between steps.
- Outputs: every step writes to an editable path and offers **Save as...** to
  copy a result anywhere.
- Intermediate files default to `<repo>/work/multistep/`.

## Build

Requires Qt 6 (Widgets), CMake, and the `third_party/miniaudio` submodule
(`git submodule update --init third_party/miniaudio`).

```bash
cmake -S rvc-ui -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

The tool/model directories can be baked into the binary (used when the
`RVC_ROOT` / `RVC_TOOLS_DIR` environment variables are unset):

```bash
cmake -S rvc-ui -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DRVC_UI_DEFAULT_TOOLS_DIR=/usr/lib/rvc.cpp \
    -DRVC_UI_DEFAULT_ROOT=/usr/share/rvc.cpp
```

Resolution order per directory: environment variable → baked default →
auto-detection (tools: the executable's directory; models: walking up from the
executable / working directory for a `models/` directory).

Windows: configure the same way with Qt6 installed (MSVC + Ninja), or add the
component to your own meta-build.

## Run

The UI locates its dependencies at runtime:

- `RVC_ROOT` — repository root containing `models/` (auto-detected by walking up
  from the executable / working directory when unset).
- `RVC_TOOLS_DIR` — directory containing `rvc-sep` and `RVC-GGML` (defaults to
  the executable's directory; in a packaged build all executables sit together).
- `sox` or `ffmpeg` on `PATH` are used for instrumental pitch-shifting
  (only needed when the pitch shift is not 0).

## Notes

- Input formats: WAV / FLAC / MP3 / Ogg Vorbis (decoded with miniaudio).
  Output: WAV natively; FLAC / MP3 / OGG are encoded with ffmpeg.
- Mix output is peak-normalised to -1 dBFS when it would clip or when the peak
  is below -5 dBFS; otherwise it is left untouched.
- "Auto pitch (proposed)": detects the median voiced f0 of the vocals and adds
  the shift that moves it to the target (155 Hz male / 255 Hz female, clamped
  to ±12 semitones) via RVC-GGML's `--proposed-pitch`. Only the converted voice
  is affected — background vocals and the instrumental keep the manual shift.
- Defaults favour natural output: protect 0, RMS mix 0, auto-pitch disabled at
  a 255 Hz target, F0 extractor prefers `rmvpe.gguf`.
- The full log lives in a separate window ("Log" button, bottom left); the main
  view shows a step progress bar and a status line instead.
- Generated outputs can be played in-app (play/pause/stop/seek, miniaudio); the
  position slider is a placeholder for a future waveform widget (that widget
  should draw precomputed peaks, not scan the audio per frame).
- The background split converts both stems with RVC and mixes them with the
  separately adjustable "background" gain (stem 0 is treated as the main vocals,
  stem 1 as the background).
- Intermediate files are cached in `<output dir>/<song>_work/`.
- Multi-step and training tabs are planned but not implemented yet.
