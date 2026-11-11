# AGENTS.md — RVC-GGML monorepo notes for agents

Pure C++ / [ggml](https://github.com/ggml-org/ggml) reimplementation of
[RVC (Retrieval-based-Voice-Conversion)](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI):
**training + inference, no CUDA required** (Vulkan acceleration for AMD/Intel/NVIDIA, CPU fallback).

- `rvc-ggml/` — inference (`RVC-GGML.exe` + Python model-conversion scripts), backend = ggml.
- `rvc-sep-ggml/` — Roformer vocal/source separation (`rvc-sep.exe`; adapted from BSRoformer.cpp).
- `rvc-train-ggml/` — training (`rvc-prep.exe` preprocessing + `rvc-train.exe` training/export),
  backend = [train.cpp](https://github.com/Chui-Las/train.cpp) submodule (`third_party/train.cpp`).
- `rvc-ui/` — Qt6 desktop UI (Song Cover one-click flow; multi-step/training tabs planned).
- Primary platform: **Windows 10/11 x64, MSVC + Ninja**. There is **no CI and no unit-test
  framework** — verification is via built-in self-tests and the Python check scripts.
- Docs and source strings were translated from Chinese to English (Nov 2026). Git *history*
  (commit messages) is still Chinese; do not rewrite it.

## Golden rules

1. **Never modify `third_party/*` submodules** (`train.cpp`, `huggingface-hub-cpp`, `miniaudio`) — third-party checkouts.
2. Keep PowerShell scripts (`*.ps1`) **ASCII-only** — PowerShell 5.1 mis-parses UTF-8 without BOM.
3. Keep C++ console/log/error strings **ASCII** and keep `printf`/`cout` format specifiers and
   argument order exactly (some buffers are sized for the worst case).
4. Keep docs in sync with behavior; the math spec lives in
   `rvc-ggml/docs/technical_03_unified_implementation_python_to_ggml.md`.
5. Do not commit build trees, models, or audio — see `.gitignore` (`build*/`, `models/`,
   `work/`, `output/`, `*.gguf`, `*.wav`, `.pth`, ...).
6. Four files intentionally carry a UTF-8 BOM; preserve it when rewriting:
   `rvc-ggml/CMakeLists.txt`, `rvc-ggml/src/main.cpp`, `rvc-ggml/src/hubert/hubert.cpp`,
   `rvc-train-ggml/src/prep/feat.cpp`. All other text files are UTF-8, LF, no BOM.
7. Intentional non-English remnants (do not "fix" blindly):
   - The wizards accept `"是"` as a "yes" answer alias
     (`rvc-train-ggml/src/cli/{prep_wizard,train_wizard}.cpp`).
   - `THIRD-PARTY-NOTICES.md` keeps original copyright holder names inside the verbatim
     license texts (RVC, BSRoformer.cpp).
8. Vendored third-party headers (`dr_wav.h`, `pocketfft_hdronly.h`) are not ours — do not restyle.

## Build (Windows)

```powershell
git clone --recursive https://github.com/Chui-Las/RVC-GGML.git RVC-GGML
cd RVC-GGML

powershell -ExecutionPolicy Bypass -File rvc-train-ggml\scripts\dev_build.ps1            # configure + build
powershell -ExecutionPolicy Bypass -File rvc-train-ggml\scripts\dev_build.ps1 -Action run # run self-tests
```

- The build cascades: traincpp (submodule) → `rvc_core` + `RVC-GGML.exe` → `rvc-prep.exe` / `rvc-train.exe` /
  `rvc-sep.exe` (Roformer separation; reuses the same ggml target).
- Outputs land in `rvc-train-ggml\build\bin\`; libraries in `rvc-train-ggml\build\lib\`.
- Manual CMake: `cmake -S rvc-train-ggml -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build`.
- Dependency path overrides: `RVCTRAIN_TRAINCPP_DIR`, `RVCTRAIN_RVCGGML_DIR`, `RVCTRAIN_RVCSEP_DIR`,
  `RVCTRAIN_GGML_DIR`; inference side: `RVC_GGML_DIR` (local ggml source), `RVC_VULKAN`, `RVC_CUDA`;
  separator side: `RVC_SEP_GGML_DIR`, `RVC_SEP_VULKAN`, `RVC_SEP_CUDA`.
- ggml resolution order in `rvc-ggml/CMakeLists.txt`: `RVC_GGML_DIR` → installed package →
  sibling `../ggml-master` / `../ggml` → FetchContent of official **v0.23.0** with this repo's
  patch auto-applied.
- Toolchain overrides for `dev_build.ps1`: `RVCT_VCVARS`, `RVCT_CMAKE`, `RVCT_NINJA`.
- Language level: C++20; MSVC `/utf-8` everywhere; `/W3` for inference targets, `/W4` for
  training targets; MSVC static runtime for `rvc-ggml`.
- The Qt6 UI is a separate build: `cmake -S rvc-ui -B build -G Ninja -DCMAKE_BUILD_TYPE=Release`
  (needs the `third_party/miniaudio` submodule). It locates the CLIs via `RVC_TOOLS_DIR` (else
  next to its executable) and the model tree via `RVC_ROOT` (else by walking up for `models/`);
  both support build-time fallbacks via `-DRVC_UI_DEFAULT_TOOLS_DIR` / `-DRVC_UI_DEFAULT_ROOT`.

### MinGW cross-check (Windows target from Linux, optional)

A MinGW cross-build of both projects works (verified 2026-11):

```bash
cmake -S rvc-ggml -B /tmp/rvc-build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=<mingw-toolchain.cmake> -DRVC_VULKAN=OFF -DRVC_CUDA=OFF
cmake --build /tmp/rvc-build -j
```

The training side additionally needs a one-line shim for the traincpp submodule
(`std::aligned_alloc` is missing from MinGW's libstdc++): put a header on the include path
that defines it via `__mingw_aligned_malloc` and pass `-DCMAKE_CXX_FLAGS="-include <shim>.h"`.

## Linux support (implemented 2026-11)

Both projects build and run natively on Linux x86-64 (GCC), CPU and Vulkan; Windows behavior is
unchanged (every Windows-only island is behind `#ifdef _WIN32`, and the MinGW cross-build still
passes). Verified 2026-11 with Mesa RADV (RX 9070): `RVC-GGML --list-devices`, `rvc-prep
--selftest` and `rvc-train --selftest` pass on both CPU and Vulkan.

```bash
# Vulkan is picked up automatically when an SDK is installed; force CPU with
# -DRVC_VULKAN=OFF / -DTRC_VULKAN=OFF. Binaries have no .exe suffix on Linux.
cmake -S rvc-ggml -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

cmake -S rvc-train-ggml -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Port notes:
- Shared helpers: `rvc-train-ggml/src/common/platform.h` (`env_present`/`env_get`/`env_set`,
  `path_sep`); inference uses `utils::set_env`. Path joins use the native separator.
- `rvc-train-ggml/CMakeLists.txt` wraps the traincpp backend archives in
  `$<LINK_GROUP:RESCAN,...>` — GNU ld is single-pass and needs it for the core↔backend cycle;
  MSVC is unaffected.
- POSIX: the wizard locates `rvc-prep` (no `.exe`) and spawns it with `std::system`; Windows
  keeps `CreateProcessW` + `rvc-prep.exe`.
- ggml's `vulkan-shaders-gen` nested configure inherits `CC`/`CXX` from the environment; export
  working compilers (or configure with `-DCMAKE_C_COMPILER`/`CXX`) if a shell sets broken ones.
- Still Windows-first: `scripts/dev_build.ps1` (MSVC/vcvars) and `scripts/train_pipeline.ps1`
  (PowerShell; may work under `pwsh`, but a bash driver is not written yet).
- The traincpp submodule still prints some Chinese log lines (device names/memory tables) —
  upstream cosmetic issue, do not translate third_party.

## Running

### Inference — `RVC-GGML.exe`

Model files (converted from the original RVC assets with `rvc-ggml/scripts/export_gguf.py`,
which needs `torch` + `gguf`, plus `faiss-cpu` for index conversion):
`models/hubert.gguf` — feature extractor; the `hubert`/`wavlm`/`spin` subcommands cover
fairseq HuBERT and HF-format ContentVec/HuBERT/WavLM/SPIN (keys are remapped automatically;
WavLM gated relative-position tensors are dropped, matching the ecosystem's plain-HuBERT
loading) — plus `models/rmvpe.gguf`, `models/<character>.gguf`, `models/<character>.index.gguf`.

```powershell
RVC-GGML.exe --input in.wav --model models\<character>.gguf --index models\<character>.index.gguf `
             --index-rate 0.75 --model-dir models --device 0 --output out.wav
RVC-GGML.exe --list-devices
RVC-GGML.exe            # no args -> interactive wizard
```

- Device IDs (ggml convention): `>=0` = Nth GPU-class device, `99` = CPU.
- `--quant <level>` loads `hubert.<level>.gguf` / `rmvpe.<level>.gguf` / `<character>.<level>.gguf`;
  levels: `fp16/q8_0/q5_1/q4_1/q4_0`.
- `--hubert <gguf>` / `--rmvpe <gguf>` override the feature / f0 extractor paths
  (defaults `<model-dir>/hubert.gguf` / `<model-dir>/rmvpe.gguf`).
- `--proposed-pitch <hz>` auto-shifts the vocals so the median voiced f0 targets
  `<hz>` (155 male / 255 female, clamped ±12 semitones) and logs
  `proposed pitch: median f0=..., offset=..., total=...`; the UI parses that line
  to shift the instrumental by the same total.
- Pipeline: WAV → 16 kHz mono → 48 Hz high-pass → silent split points → HuBERT features →
  RMVPE f0 → FAISS retrieval mix (`--index-rate`) → VITS synthesis → optional resample.
- Model-level parameters (pitch shift, retrieval rate, protect, ...) are embedded in the
  character GGUF; CLI flags override them.

### Separation — `rvc-sep.exe`

```powershell
rvc-sep.exe "models\roformers\<model>.gguf" input\song.wav output\vocals.wav   # --device cpu to force CPU
```

- Roformer (Mel-Band / BS) GGUF models live in `models/roformers/`; WAV in/out, any input
  rate is auto-resampled to 44.1 kHz; multi-stem models write `<output>_stem_N.wav` (0 = vocals).
- Built standalone (`rvc-sep-ggml/`) or as part of the training-side cascade.

### UI — `rvc-ui`

Qt6 Widgets app; the first tab is the Song Cover one-click flow (vocal/instrumental separation,
optional background-singer split, RVC conversion, pitch shift, mix). See `rvc-ui/README.md`.
Requires Qt6 Widgets; `sox` (or ffmpeg) only when pitch != 0.
Input: WAV/FLAC/MP3/OGG (miniaudio); output: WAV natively, FLAC/MP3/OGG via ffmpeg.
The mix is peak-normalised to -1 dBFS when clipping or below -5 dBFS.
The log lives in a pop-out window ("Log" button); the main view shows step progress/status.
In-app playback uses miniaudio (`rvc-ui/src/audio_player.*`); the seek slider is a
placeholder for a future waveform widget — keep waveform drawing cheap (precompute
peaks once, never scan the full buffer per repaint).
The "Multi-step" tab runs each step in isolation with a shared source registry:
inputs can be files or any step's output; outputs can be saved anywhere.
Its Roformer card aggregates vocal separators, background splitters and de-reverb
models in one dropdown (single-stem models also emit `input - output`).
Auto pitch (`--proposed-pitch`) affects only the converted voice.
The Song Cover flow can optionally de-reverb the main vocals after separation
(`models/roformers/de-reverb/`), and both separations have a "Flip stems" option
for models that swap the stem order.

### Training — `rvc-prep.exe` + `rvc-train.exe` (two processes by design)

Project rule **R28**: the two Vulkan backends (ggml and traincpp) must never coexist in one
process, so preprocessing and training are separate executables, chained by
`rvc-train-ggml/scripts/train_pipeline.ps1` or the interactive wizards.

```powershell
# Full pipeline: prep -> train -> export -> closed-loop inference
powershell -ExecutionPolicy Bypass -File rvc-train-ggml\scripts\train_pipeline.ps1 `
    -Audio "D:\myvoice" -Mode finetune `
    -G work\f0G40k.train.gguf -D work\f0D40k.train.gguf `
    -Epochs 50 -ExportEvery 5 -Export -Infer -InferInput test.wav

rvc-prep.exe  --interactive      # preprocessing wizard
rvc-train.exe --interactive      # training wizard (also the no-args default)
```

- One-time conversion of official pretrained G/D `.pth` → training-state GGUF (needs `torch`):
  `rvc-train-ggml/tools/pth_to_train_gguf.py`, verify with `tools/verify_train_gguf.py`.
- `rvc-prep` extracts f0 (RMVPE), features (HuBERT, v1=256 / v2=768 dims), spectrograms and
  writes the training cache (`dataset.json`, `spec.json`, `samples/*.bin`); `--validate-cache`
  checks it. `rvc-prep --index ...` builds retrieval `.index.gguf` (IVF/k-means).
- Device IDs (ggml convention for `rvc-prep`): `>=0` GPU, `99` CPU. For `rvc-train`, `--device N`
  is a **traincpp device index** from `rvc-train --list-devices` (default: auto, prefers
  GPU → IGPU → CPU).
- `rvc-train` modes: `--train <config.json>`, `--scratch` (from scratch), `--resume <ckpt>`,
  `--export-gguf` (training-state → inference GGUF); config templates live in
  `rvc-train-ggml/configs/` (+ `configs/README.md` for the schema).
- Exit codes (rvc-train): `0` success, `1` failure, `2` argument error.

### Self-tests / verification

- `RVC-GGML.exe`: `--audio-selftest`, `--hubert-selftest`, `--rmvpe-selftest`,
  `--vits-selftest` (`--ref-dir`, `--noise-dir`, `--f0-inject` for deterministic replay
  against reference dumps).
- `rvc-prep.exe`: `--selftest`, `--validate-cache <dir>`, `--validate-index <gguf>`,
  `--search-index <gguf> --queries <npy> --search-out <raw>`.
- `rvc-train.exe`: `--selftest`, `--selftest-forward/backward/loss/train-step/train-loop/trainer/
  data/scratch/resume/wizard`, `--validate-gguf`, `--dump-hashes`.
- Python checks: `rvc-ggml/scripts/verify_gguf.py`, `verify_tensor.py`, `verify_audio.py`,
  `rvc-train-ggml/tools/verify_train_gguf.py`.
- Golden/reference data is **not stored in the repo** (`output/ref`, `--golden` dirs come from
  the original Python pipeline); tests that need them are skipped/must be given the data.
- Correctness bar (from the original port): exported model self-check `corr=1.0000`,
  end-to-end `corr >= 0.999` vs the Python reference.

## Environment variables

| Var | Where | Meaning |
|---|---|---|
| `RVC_VK_ENABLE_INTDOT=1` | inference | re-enable the integer dot-product path (disabled by default due to a defect on some old Intel drivers: NaN/all-zero) |
| `RVC_VK_ENABLE_F16=1` | inference | re-enable the fp16 compute path (disabled by default) |
| `RVC_HUBERT_DUMP`, `RVC_RMVPE_DEBUG`, `RVC_DEBUG_OP`, `RVC_DEBUG_CONV`, `RVC_DEBUG_LINEAR` | inference | debug dumps for graph/operator troubleshooting |
| `TRC_VK_HOST_MEMORY=1` | training | fall back to host-visible Vulkan buffers (default is device-local) |
| `TRC_VK_CHUNK_NODES=64` | training | batch node submission; recommended for long runs to mitigate TDR/device loss |
| `TRC_OPT_HOST=1` | training | force the host optimizer (A/B comparisons only) |
| `RVCT_VCVARS`, `RVCT_CMAKE`, `RVCT_NINJA` | build | toolchain overrides for `dev_build.ps1` |

## Architecture quick map

| Area | Where |
|---|---|
| CLI / selftests (inference) | `rvc-ggml/src/main.cpp` |
| ggml backend, device enumeration, graph exec | `rvc-ggml/src/engine.{h,cpp}` |
| Audio I/O, DSP, mel | `rvc-ggml/src/audio/` |
| GGUF loading, weight_norm folding | `rvc-ggml/src/gguf/gguf_model.{h,cpp}` |
| HuBERT / RMVPE / VITS | `rvc-ggml/src/{hubert,rmvpe,vits}/` |
| Pipeline, FAISS retrieval, sine gen | `rvc-ggml/src/rvc/` |
| Preprocessing (slicer, f0, feat, spec, cache) | `rvc-train-ggml/src/prep/` |
| Data loader (host-side batching/bucketing) | `rvc-train-ggml/src/data/` |
| Model (enc_p/enc_q/flow/dec/D), config, selftests | `rvc-train-ggml/src/model/` |
| Train step, trainer, losses, resume, selftests | `rvc-train-ggml/src/train/` |
| Export (training GGUF, inference GGUF, index, kmeans, npy) | `rvc-train-ggml/src/export/` |
| Wizards, JSON parsing | `rvc-train-ggml/src/cli/`, `src/common/` |
| Roformer separation (rvc-sep) | `rvc-sep-ggml/` (model/inference/audio, adapted from BSRoformer.cpp) |
| Song-cover UI (Qt6) | `rvc-ui/` (Song Cover page; QProcess orchestration of the CLIs) |

### Docs (all English now)

| Doc | Contents |
|---|---|
| `rvc-ggml/docs/usage_release_vs_original.md` | Inference usage, parameter mapping, common errors |
| `rvc-ggml/docs/technical_01_port_overview.md` | Port overview, feature comparison |
| `rvc-ggml/docs/technical_02_ggml_vulkan_details.md` | GGUF layout, quantization, Vulkan details, N-split patch |
| `rvc-ggml/docs/technical_03_unified_implementation_python_to_ggml.md` | Operator mapping + pipeline math (authoritative spec) |
| `rvc-ggml/docs/technical_04_source_map_and_module_manual.md` | Inference source map / interface manual |
| `rvc-train-ggml/docs/usage_guide.md` | Full training workflow |
| `rvc-train-ggml/docs/numbers_handbook.md` | How to read training parameters/log numbers |
| `rvc-train-ggml/docs/architecture.md` | Training architecture, data flow, GGUF/cache schema |
| `rvc-sep-ggml/README.md` | Roformer separation tool: build, usage, attribution |
| `rvc-ui/README.md` | Qt6 UI: build, model folders, one-click song-cover flow |

Some source comments reference internal docs that were **never committed** (e.g.
`docs/handover.md`, `docs/interfaces_and_formats.md`, `docs/dev_progress.md`,
`docs/feedback-*-Q22/23.md`) — treat those as historical notes, not links.

## Pitfalls to know

- **ggml Vulkan N-split patch is mandatory** for long audio (single chunk > 43.7 s) and 48 kHz
  models: `rvc-ggml/patches/ggml-vulkan-matmul-n-split.patch` (auto-applied only on the
  FetchContent path; apply manually with local ggml sources). See `patches/README.md` and
  `technical_02_ggml_vulkan_details.md` §5.
- Default training params (`batch=1`, `bucket=384`, `disc_len=12800`, `TRC_VK_CHUNK_NODES=64`)
  are tuned conservatively for ~4 GB VRAM/shared memory; scale up only on bigger GPUs
  (see `rvc-train-ggml/configs/README.md`).
- `spec.json` cache metadata strings (`note`, `layout`, ...) are informational English text;
  nothing parses them — only `dataset.json` and the `samples/*.bin` files are consumed.
- Device-ID conventions differ between backends (ggml: `99`=CPU; traincpp: index from its own
  device list, no `99`).
- `rvc-train-ggml/src/prep/{feat,cache}.cpp` implement exact reproductions of upstream Python
  quirks — do not "clean up" numerical code without cross-checking
  `technical_03_unified_implementation_python_to_ggml.md` and the training docs.
- The audio resampler is triplicated (`rvc-ggml/src/audio/audio_io.cpp`, `rvc-sep-ggml/src/audio.cpp`,
  `rvc-ui/src/audio_utils.cpp`) — keep the three copies in sync. An older phase-table version
  produced sign-flipped spikes for some ratios (e.g. 32k → 44.1k); `rvc-ui` carries a regression
  test (`-DRVC_UI_BUILD_TESTS=ON`, target `resample_test`).
- Feature extractors: ContentVec/HuBERT also serve v1 models (they ship `final_proj`);
  WavLM/SPIN extractors are v2-only (no `final_proj`; the loader returns a clear error for v1).

## Workflow expectations for agents

- Read the module and its doc before changing it; keep changes minimal and consistent with
  surrounding style.
- After C++ changes: build (Windows preferred; MinGW CPU cross-build at least for the
  inference side). After Python changes: `python3 -m py_compile <file>`.
- If behavior/format changes: update the matching doc and, if GGUF keys are affected, the
  verify scripts (`rvc-ggml/scripts/verify_*.py`, `rvc-train-ggml/tools/verify_train_gguf.py`).
- Do not commit unless asked; never force-push; keep commit messages in English going forward.
