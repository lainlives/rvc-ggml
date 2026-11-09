# RVC-GGML

A pure C++ / [ggml](https://github.com/ggml-org/ggml) implementation of [RVC (Retrieval-based-Voice-Conversion)](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) **training + inference**, no CUDA required — **AMD / Intel / NVIDIA GPUs can all be accelerated through Vulkan**, and it also runs on CPU with no discrete GPU.

- **rvc-ggml**: inference. A complete port of the Python/PyTorch RVC inference chain (HuBERT features → RMVPE pitch → VITS synthesis → FAISS retrieval) to C++ + ggml; models are distributed in GGUF format.
- **rvc-train-ggml**: training. Based on [train.cpp](https://github.com/Chui-Las/train.cpp) v0.4.0 (a C++ deep-learning training library with autograd + Vulkan backend, on-device optimizer and device-local training buffers), it implements RVC preprocessing, the GAN training loop and inference-model export, forming a "train → export GGUF → inference" closed loop with rvc-ggml.

> Design goal: output numerically and perceptually consistent with the original Python version (exported models pass strict self-tests with corr=1.0000; end-to-end corr ≥ 0.999).

## Repository structure

```
├── rvc-ggml/          inference (RVC-GGML.exe + Python model-conversion scripts)
├── rvc-train-ggml/    training (rvc-prep.exe preprocessing + rvc-train.exe training/export)
└── third_party/
    └── train.cpp/     training library (git submodule)
```

## System requirements

- Windows 10/11 x64 (primary release platform) or Linux x86-64 (native builds verified on both; see `AGENTS.md`)
- CMake ≥ 3.20, Ninja, and either MSVC (Visual Studio C++ toolset) or GCC/Clang
- [Vulkan SDK](https://vulkan.lunarg.com/) (including glslc; without it the build automatically falls back to pure CPU)
- A GPU supporting Vulkan 1.2 (4 GB of VRAM/shared memory as a starting point); with no GPU the CPU backend can be used

## Build

```powershell
git clone --recursive https://github.com/Chui-Las/RVC-GGML.git RVC-GGML
cd RVC-GGML

# Training side (cascade-builds rvc-prep / rvc-train / RVC-GGML)
powershell -ExecutionPolicy Bypass -File rvc-train-ggml\scripts\dev_build.ps1
# Artifacts land in rvc-train-ggml\build\bin\
# When cmake/ninja are not on PATH, the script automatically uses the versions bundled with Visual Studio
```

Or manually with CMake (Windows):

```powershell
cmake -S rvc-train-ggml -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

On Linux (Vulkan is detected automatically; add `-DRVC_VULKAN=OFF -DTRC_VULKAN=OFF` for CPU-only;
binaries land in `rvc-train-ggml/build/bin/` and have no `.exe` suffix):

```bash
cmake -S rvc-train-ggml -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

The ggml dependency is resolved automatically: FetchContent pulls official ggml v0.23.0 and then applies this project's patch (see `rvc-ggml/patches/`); you can also point at a local copy with `-DRVC_GGML_DIR=<local ggml source>`.

## Quick start

### 1. Model conversion (.pth → GGUF)

Requires model files from the original RVC ecosystem (hubert_base.pt, rmvpe.pt, character .pth/.index):

```powershell
cd rvc-ggml
python scripts\export_gguf.py hubert --input <RVC dir>\assets\hubert\hubert_base.pt --out models\hubert.gguf --dtype fp16
python scripts\export_gguf.py rmvpe  --input <RVC dir>\assets\rmvpe\rmvpe.pt       --out models\rmvpe.gguf  --dtype q8_0
python scripts\export_gguf.py model  --input <character.pth>                        --out models\<character>.gguf --dtype q4_0
python scripts\export_gguf.py index  --input <character.index>                      --out models\<character>.index.gguf
# Or batch-convert everything from the original RVC root in one command:
python scripts\export_gguf.py all --rvc-root <RVC dir> --out models --dtype fp16
```

### 2. Inference

```powershell
RVC-GGML.exe --input input.wav --model models\<character>.gguf \
             --index models\<character>.index.gguf --index-rate 0.75 \
             --model-dir models --device 0 --output output.wav
# --device 0 = Vulkan (lists devices first, then prompts for selection); 99 = CPU
RVC-GGML.exe --list-devices
# Launching without arguments enters the interactive wizard; model training parameters (pitch shift / retrieval rate, etc.) are embedded in the character GGUF
```

### 3. Training

Fine-tune the official pretrained models (minimal closed loop):

```powershell
cd ..\rvc-train-ggml
# 0) Official pretrained G/D .pth -> training-state GGUF (one-time, see tools\README.md)
python tools\pth_to_train_gguf.py --pth <RVC dir>\assets\pretrained\f0G40k.pth --config configs\v1-40k-f0.json --out work\f0G40k.train.gguf
python tools\pth_to_train_gguf.py --pth <RVC dir>\assets\pretrained\f0D40k.pth --config configs\v1-40k-f0.json --out work\f0D40k.train.gguf

# 1) Preprocess + train + export + closed-loop inference in one command:
powershell -ExecutionPolicy Bypass -File scripts\train_pipeline.ps1 `
    -Audio "D:\myvoice" -Mode finetune `
    -G work\f0G40k.train.gguf -D work\f0D40k.train.gguf `
    -Epochs 50 -ExportEvery 5 -Export -Infer -InferInput test.wav

# Or the interactive wizard (double-click the exe / run without arguments):
rvc-train.exe --interactive
# Train from scratch: choose "scratch" in the wizard, or rvc-train --scratch --train configs\v1-40k-f0.json
```

> Note: preprocessing (rvc-prep, ggml backend) and training (rvc-train, traincpp backend) are two separate processes, chained automatically by train_pipeline.ps1 / the wizard; a single exe cannot complete all the steps.

## Documentation index

| Document | Contents |
|---|---|
| [`rvc-ggml/docs/usage_release_vs_original.md`](rvc-ggml/docs/usage_release_vs_original.md) | Inference usage instructions, parameter mapping, common errors |
| [`rvc-ggml/docs/technical_04_source_map_and_module_manual.md`](rvc-ggml/docs/technical_04_source_map_and_module_manual.md) | Inference-side source structure and interface manual |
| [`rvc-ggml/docs/technical_03_unified_implementation_python_to_ggml.md`](rvc-ggml/docs/technical_03_unified_implementation_python_to_ggml.md) | Operator mapping and pipeline math definitions (self-contained) |
| [`rvc-ggml/docs/technical_02_ggml_vulkan_details.md`](rvc-ggml/docs/technical_02_ggml_vulkan_details.md) | GGUF layout, quantization, Vulkan adaptation and pitfalls |
| [`rvc-ggml/docs/technical_01_port_overview.md`](rvc-ggml/docs/technical_01_port_overview.md) | Inference port overview and feature comparison |
| [`rvc-train-ggml/docs/usage_guide.md`](rvc-train-ggml/docs/usage_guide.md) | Full training workflow guide |
| [`rvc-train-ggml/docs/numbers_handbook.md`](rvc-train-ggml/docs/numbers_handbook.md) | Reading the numbers in training parameters/logs |
| [`rvc-train-ggml/docs/architecture.md`](rvc-train-ggml/docs/architecture.md) | Training-side architecture, data flow, GGUF schema, cache format |

## Compatibility notes

- Some older Intel drivers have a defective integer dot-product path (NaN/all-zero) when a quantized model's graph is executed repeatedly; the program avoids it by default, and it can be restored with `RVC_VK_ENABLE_INTDOT=1`; the fp16 compute path is likewise disabled by default (`RVC_VK_ENABLE_F16=1` to restore).
- Official ggml's Vulkan matrix multiply trips the `maxComputeWorkGroupCount` assertion when the N dimension exceeds the limit (long audio / 48 kHz models) — this project's patch fixes it, and the FetchContent path applies it automatically.
- On the training side, since train.cpp v0.4.0, Vulkan compute buffers prefer device-local by default (true VRAM on discrete GPUs + shared staging / mappable device-local / host-visible fallback); with older drivers or for troubleshooting, set `TRC_VK_HOST_MEMORY=1` to fall back to host-visible.
- For long training runs it is recommended to set `TRC_VK_CHUNK_NODES=64` (batched submission, mitigating TDR / device loss on some discrete-GPU drivers); `TRC_OPT_HOST=1` is only for forcing the host optimizer in A/B comparisons.

## License

This project is open source under the [MIT](LICENSE) license.

It depends on / references the following excellent open-source projects; see [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md) for their respective licenses:

- [ggml](https://github.com/ggml-org/ggml) (MIT) — inference backend
- [train.cpp](https://github.com/Chui-Las/train.cpp) (MIT) — C++ deep-learning training library (v0.4.0, submodule)
- [RVC](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) (MIT) — semantic baseline and model ecosystem
- [dr_wav](https://github.com/mackron/dr_libs) (MIT-0 / Unlicense) and
  [pocketfft](https://github.com/mreineck/pocketfft) (BSD-3) — vendored single-header libraries
