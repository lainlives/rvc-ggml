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

## Build

Requires Qt 6 (Widgets) and CMake.

```bash
cmake -S rvc-ui -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

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

- The background split converts both stems with RVC and mixes them with the
  separately adjustable "background" gain (stem 0 is treated as the main vocals,
  stem 1 as the background).
- Intermediate files are cached in `<output dir>/<song>_work/`.
- Multi-step and training tabs are planned but not implemented yet.
