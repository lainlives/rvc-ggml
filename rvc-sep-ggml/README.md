# rvc-sep-ggml

Roformer-based audio source separation (vocals / instrumental) built on ggml,
adapted into the RVC-GGML monorepo.

- Upstream: **BSRoformer.cpp** — MIT License, Copyright (c) 2026 沉默の金 <cmzj@cmzj.org>
  (adapted copy kept as a reference at `work/Roformer.cpp`; mirror:
  `https://lain.lives.moe/lainlives/Roformer.cpp`).
- Supports **Mel-Band Roformer** and **BS Roformer** GGUF models
  (see `models/roformers/`), CPU and Vulkan GPU backends.

Differences from upstream:

- Audio I/O uses `dr_wav` (WAV in/out, float32, automatic resample to 44.1 kHz)
  instead of FFmpeg/libav; convert other formats with ffmpeg first.
- When built as part of `rvc-train-ggml`, it reuses the `ggml` target provided by
  `rvc-ggml` (single ggml/Vulkan backend per build tree); standalone builds
  resolve ggml the same way `rvc-ggml` does.

## Build

```bash
cmake -S rvc-sep-ggml -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

On Windows use the regular MSVC/Ninja flow (see the repository README); the
training-side build (`rvc-train-ggml`) cascades this component automatically.

## Usage

```bash
rvc-sep "models/roformers/MelBand Roformer | Vocals by Kimberley Jensen-q4_1.gguf" \
        input/song.wav output/vocals.wav
```

- GPU (Vulkan) is used by default; pass `--device cpu` to force the CPU backend.
- Multi-stem models write `<output>_stem_0.wav`, `<output>_stem_1.wav`, ...
  (stem 0 is usually the vocals).
- `--chunk-size <N>` / `--overlap <N>` override the model's defaults.
