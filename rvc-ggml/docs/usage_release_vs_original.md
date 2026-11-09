# RVC-GGML Usage Guide: Release Version vs. Original (v0.2.0)

> This document takes the **RVC-GGML v0.2.0 Windows release** as the reference and explains the complete usage from model conversion
> to application invocation, and compares parameters, errors and system requirements item by item against the original RVC (WebUI/CLI).

---

## 1. System Requirements (Runtime)

| Item | Requirement | Notes |
|---|---|---|
| OS | Windows 10 / 11 64-bit | The release provides a win64 build only |
| CPU | x86-64 with **AVX2** support | The release is compiled for the AVX2 instruction set; basically supported on Intel CPUs from 2013 and AMD CPUs from 2015 onward |
| Memory | 8 GB or more recommended | Conversion depends on host memory and VRAM |
| GPU (optional) | A GPU and driver supporting **Vulkan 1.2** (16-bit storage required; the vast majority of drivers from 2017 onward satisfy this) | Intel integrated/discrete, AMD and NVIDIA all work; no CUDA needed |
| VRAM | 4 GB or more recommended | 5 seconds of audio takes roughly "weights ~450 MB (fp16) + graph buffers ~0.4 GB"; longer audio takes more |
| Disk | Runtime package ~1.7 GB (including all quant levels); output counted separately | Smaller if only a few models are bundled |
| Other | System-provided `vulkan-1.dll` (from the driver); the package bundles `vcomp140.dll` (OpenMP runtime) | No need to install Python, CUDA or the VC++ runtime |

**CPU mode**: `--device 99`. Old machines whose CPUs lack AVX2 are not supported.

---

## 2. Release Package Structure

```
RVC-GGML-v0.2.0-win64\
├─ RVC-GGML.exe            main program
├─ vcomp140.dll            OpenMP runtime (must be in the same directory as the exe)
├─ README.md               quick instructions
├─ interactive_mode.bat    double-click to enter the interactive wizard
├─ list_devices.bat        list Vulkan devices
├─ example_convert.bat     one example command
├─ input\                  input audio samples (wav)
├─ output\                 output directory (conversion results / quantization test results)
├─ models\                 models (see the table below)
└─ scripts\                model conversion and validation scripts (Python)
```

Contents of `models\` (each type is provided in 5 levels: fp16 / q8_0 / q5_1 / q4_1 / q4_0):

| File | Purpose | fp16 size | Size reference per level |
|---|---|---|---|
| `hubert.gguf` (+`.q*.gguf`) | Content features (required) | 190.5 MB | 110.5 / 83.8 / 73.1 / 67.8 MB |
| `rmvpe.gguf` (+`.q*.gguf`) | Pitch extraction (required for f0 models) | 178.9 MB | 100.9 / 74.9 / 64.5 / 59.3 MB |
| `guanguanV1.gguf` (+`.q*.gguf`) | Character model (v1 / 40 kHz / f0) | 70.2 MB | 54.0 / 48.7 / 46.5 / 45.4 MB |
| `youzhanv2-xi.gguf` (+`.q*.gguf`) | Character model (v2 / 48 kHz / f0) | 75.1 MB | 58.9 / 53.5 / 51.4 / 50.3 MB |
| `guanguanV1.index.gguf` | Retrieval index (optional, used with `--index-rate`) | 33.2 MB | Index is not quantized |

> Meaning of the quant levels: q8_0 ≈ near lossless; q5_1/q4_1 are compromises; q4_0 has the smallest size and its audio-quality difference is relatively audible (see "Known Limitations").

---

## 3. Quick Start

### 3.1 Interactive Mode (Recommended)

Double-click `interactive_mode.bat`, or run `RVC-GGML.exe` from the command line. The wizard asks, in order:

1. **Character model**: lists all character models under `models\` (automatically reads version / sample rate / f0 / speaker count / quant level);
2. **Quant level**: lists the available unified levels (fp16/q8_0/…), applied to hubert + rmvpe + character model together;
3. **Input audio**: lists `input\*.wav`; any path can also be entered;
4. **Device**: lists Vulkan devices and CPU (GPU 0 is preferred by default);
5. **Conversion parameters**: pitch shift, speaker (asked only for multi-speaker models), index mix rate (asked only when a same-named index exists), RMS mix rate, unvoiced protection, optional resampling;
6. After confirmation, conversion begins; the output is `output\<input name>_<model name>[_<level>].wav`; press Enter to exit when done.

### 3.2 Command-Line Mode

```powershell
# Basic conversion (fp16, GPU 0)
RVC-GGML.exe --input input\test_5s_16k.wav --model models\guanguanV1.gguf ^
    --output output\demo.wav --rms-mix-rate 0.75 --device 0

# Switch the quant level of the whole model set in one go
RVC-GGML.exe --input input\test_5s_16k.wav --model-dir models --quant q4_0 ^
    --output output\demo_q4_0.wav --rms-mix-rate 0.75 --device 0

# With retrieval (requires a same-named or explicitly specified index)
RVC-GGML.exe --input input\test_5s_16k.wav --model models\guanguanV1.gguf ^
    --index models\guanguanV1.index.gguf --index-rate 0.75 ^
    --output output\demo_idx.wav --device 0

# List devices
RVC-GGML.exe --list-devices
```

---

## 4. Command-Line Argument Quick Reference (vs. Original)

| Argument | Value / Default | Meaning | Corresponding original RVC concept |
|---|---|---|---|
| `--input <wav>` | required (conversion mode) | Input audio; any sample rate/channel count, automatically converted to 16k mono | WebUI's "audio to convert" |
| `--output <wav>` | `output/converted.wav` | Output 16-bit PCM mono WAV | Output file |
| `--model <gguf>` | `models/guanguanV1.gguf` | Character model | WebUI .pth selection |
| `--model-dir <dir>` | `models` | Model directory (where hubert/rmvpe/index live) | — |
| `--quant <level>` | empty = each file's actual level | `fp16/q8_0/q5_1/q4_1/q4_0`, uniformly replaces hubert/rmvpe/character model | Not in the original (this project's quantization feature) |
| `--pitch <n>` | `0` | Pitch shift in semitones: `+12` up one octave, `-12` down one octave | Pitch shift (integer) |
| `--sid <n>` | `0` | Speaker ID (multi-speaker models) | Speaker ID |
| `--index <gguf>` | not used | Retrieval index (`.index.gguf` in this project) | `*.index` retrieval file |
| `--index-rate <r>` | `0` | Retrieval feature mix rate 0–1; 0.5–0.75 commonly used | Retrieval feature proportion |
| `--protect <p>` | `0.33` | Unvoiced protection 0–0.5: higher values preserve more original unvoiced features | Protect unvoiced consonants |
| `--rms-mix-rate <r>` | `1.0` | Loudness mix 0–1: 1 = fully use the converted result's loudness, 0 = fully use the original audio's loudness | Loudness factor (rms_mix_rate) |
| `--resample-sr <n>` | `0` | Output resample rate; 0 = model sample rate (40k/48k) | Output sample rate (audio resampling) |
| `--device <id>` | `99` | `0+` = Vulkan device index, `99` = CPU | The original offers "select GPU/inference device" |
| `--x-pad` / `--x-query` / `--x-center` / `--x-max` | `1/6/38/41` (seconds) | Long-audio chunking parameters: leading/trailing padding, split-point search radius, chunk interval, duration threshold for enabling chunking | `configs/config.py` device_config |
| `--list-devices` | — | Lists available Vulkan devices | — |
| `--interactive, -i` | — | Force entry into the interactive wizard | — |
| `--noise-dir <dir>` / `--f0-inject` | — | **For verification only**: injects reference noise/f0 so C++ and Python can be compared sample by sample | None |
| `--audio-selftest` / `--hubert-selftest` / `--rmvpe-selftest` / `--vits-selftest` | — | Developer self-test modes (require reference data/models) | None |

**Original parameters that are fixed / not provided**:

- The f0 algorithm is fixed to **RMVPE** (the original offers pm/harvest/crepe); `filter_radius` is fixed at 3 (unused by RMVPE);
- UVR5 vocal separation, training/finetuning, real-time voice conversion and TTS are not provided;
- Advanced options such as custom f0 curves and harvest parameters are not supported.

---

## 5. Model Conversion (.pth / .pt / .index → GGUF)

The runtime cannot read original-format models directly; convert them with `scripts\export_gguf.py`.

### 5.1 Conversion Environment

| Component | Requirement | Purpose |
|---|---|---|
| Python | 3.9–3.13 (tested with 3.9.13 / 3.13) | Runs the scripts |
| PyTorch (CPU version is enough) | 1.x/2.x (tested with 2.0.0+cpu / 2.12.1+cpu) | Reads .pt/.pth |
| numpy | 1.21+ | Numerical processing |
| `gguf` (pip package) | tested with 0.16.2 | Writes GGUF |
| `faiss-cpu` | only needed to convert `.index` | Reads FAISS indexes |
| `librosa` (optional) | Used for the RMVPE mel filterbank; when missing, the script uses a built-in equivalent implementation | — |

Installation (Tsinghua mirror example):

```powershell
pip install numpy torch gguf -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install faiss-cpu -i https://pypi.tuna.tsinghua.edu.cn/simple   # only needed to convert indexes
```

### 5.2 Conversion Commands

Run from the project/release package root directory (**mind the current directory**: the script is under `scripts\`, and input/output paths are resolved relative to the current directory):

```powershell
python scripts\export_gguf.py hubert --input <hubert_base.pt> --out models\hubert.gguf --dtype fp16
python scripts\export_gguf.py rmvpe  --input <rmvpe.pt>       --out models\rmvpe.gguf  --dtype q8_0
python scripts\export_gguf.py model  --input <character.pth>  --out models\<character>.gguf --dtype q4_0
python scripts\export_gguf.py index  --input <character.index> --out models\<character>.index.gguf
# Batch from the RVC root directory (hubert fixed at fp16; character models per --dtype; automatically converts logs/*.index)
python scripts\export_gguf.py all --rvc-root <RVC upstream repo root> --out models --dtype fp16
```

`--dtype` options: `f32 / fp16 / q8_0 / q5_0 / q5_1 / q4_0 / q4_1`.

**Naming convention (relied upon by `--quant`)**:

- fp16 baseline: `hubert.gguf`, `rmvpe.gguf`, `<character>.gguf`
- Quant levels: `hubert.q4_0.gguf`, `rmvpe.q4_0.gguf`, `<character>.q4_0.gguf`
- Index: `<character>.index.gguf` (not level-specific)

Sign of a successful conversion: it prints `quantized x / F16 y / F32 z, file xxx MB`.

### 5.3 Common Conversion Errors

| Symptom | Cause | Fix |
|---|---|---|
| `can't open file '...\export_gguf.py'` | Wrong script path (not in the current directory) | Use `python scripts\export_gguf.py ...` and make sure you run it from the release package root directory |
| `FileNotFoundError: models\xxx.pth` | The input path does not exist relative to the current directory | Use a full path, or `cd` to the directory containing the .pth first |
| `Missing gguf package` | gguf is not installed | `pip install gguf` |
| `RVC model is missing the 'weight' field` | The input is not an RVC character model .pth | Verify the file's source and format |
| `Only IndexIVFFlat is supported (currently …)` | The index is not a FAISS IVF Flat | Original RVC indexes generally satisfy this; other types are not yet supported |
| `[WARN] xxx: quantization failed… falling back to F16` | The tensor does not satisfy the quantization conditions | Normal warning; does not affect usage |
| Conversion is very slow | Reading the .pt + tensor-by-tensor quantization | A 50s model takes about 1 minute; just wait patiently |

---

## 6. Runtime Error Reference

| Error message | Meaning | Fix |
|---|---|---|
| `Failed to open GGUF file: <path>` / `gguf_init_from_file: failed ... (No such file or directory)` | The path does not exist **or contains Chinese/non-ASCII characters** (ggml opens files as UTF-8, while Windows command-line arguments use the native codepage) | First confirm the file is present before using `--list-devices`; **use pure English for model paths**; audio paths are not subject to this limitation |
| `GGUF is missing the general.architecture key` | Not a GGUF exported by this tool | Re-convert with `scripts\export_gguf.py` |
| `Weight buffer allocation failed (out of VRAM?)` | Not enough VRAM/memory to hold the model | Switch to a lower quant level; close other programs using VRAM; or use `--device 99` to run from system memory |
| `GPU device ID N out of range (M available), falling back to CPU` | The `--device` index does not exist | Run `--list-devices` first and use the actual index (0-based) |
| `GPU device initialization failed, falling back to CPU` | Vulkan driver/device problem (driver too old, device in use) | Update the graphics driver; or keep using CPU |
| `GGML compute failed: <status>` | Graph execution failed (out of VRAM / driver problem / unsupported operator) | Switch to a lower level or CPU; update the driver; report the full output |
| `Quant level file does not exist: <path>` | `--quant q4_0` was given but the corresponding level file is missing | Generate that level with `scripts\export_gguf.py`, or drop `--quant` |
| `VITS missing weight: <name>` / `RMVPE missing tensor` | The model does not match the expected structure (old/corrupt file, wrong type used) | Re-export the model with the current script; make sure `--model` is a character model and not hubert/rmvpe |
| `linear: dimension mismatch` | The character model does not match the feature dimensions (e.g. mixing a v2 model with v1 data) | Make sure the model matches the pipeline |
| `Input audio is empty` / read failure | The WAV is empty/corrupt/not a WAV | Use another file; supported: WAV (PCM 8/16/24/32-bit, 32-bit float, A/μ-law) |
| `f0 injection file missing` / `noise file missing` | `--f0-inject`/`--noise-dir` was used but the reference data is incomplete | Only needed by the verification flow; do not add these two options for normal use |
| `Model not found: models\hubert.gguf` (self-test) | A file required by self-test mode is missing | Before self-testing, make sure `models\` and `output\ref\` are complete |
| The program crashes/exits with no output | The path contains Chinese characters (see the first entry) or the model file is corrupt | Retry with an English path; verify the model size |

---

## 7. Frequently Asked Questions (FAQ)

**Q1: Which input formats are supported?**
WAV (PCM 8/16/24/32-bit, 32-bit float, A/μ-law); multi-channel is automatically downmixed; any sample rate is automatically resampled to 16 kHz.

**Q2: What is the output format?**
16-bit PCM mono WAV at the model's sample rate (40 kHz / 48 kHz), which can be specified with `--resample-sr` (e.g. 44100/48000).

**Q3: Is the conversion result the same as the original RVC?**
With the same model, noise and f0, the waveform correlation coefficient is 0.999+ (perceptually consistent). Due to floating-point implementation and quantization differences,
it is not bit-identical; the original's random noise on every inference also causes differences by itself.

**Q4: Why are two runs with the same input not exactly identical?**
The program uses a fixed random seed by default and is reproducible under normal conditions. However, f0 estimation in very rare frames (when the salience top1/top2 are extremely close)
has floating-point sensitivity, which may cause phase-level differences (the perceptual impact is very small).

**Q5: How is the pitch shift set?**
`--pitch 12` raises by one octave, `--pitch -12` lowers by one octave, `--pitch 2` raises by two semitones. Corresponds to the "pitch shift" setting in the original WebUI.

**Q6: What should `--index-rate` be set to?**
0.5–0.75 is the common range (higher values get closer to the training set's timbre, but the content may be more "sticky"); 0 = no retrieval at all.

**Q7: What about `--protect` and `--rms-mix-rate`?**
`--protect` (default 0.33): higher values preserve more of the original audio's unvoiced (breathy/sibilant) features;
`--rms-mix-rate` controls loudness mixing: 1 = fully adopt the converted result's loudness, 0 = fully adopt the original audio's loudness.

**Q8: What if I run out of VRAM?**
Lower the quant level (e.g. q4_0) → reduce the chunk size (decrease `--x-center`; chunk length is about `x_center + 2×x_pad` seconds) → use CPU (`--device 99`).

**Q9: Can model paths contain Chinese characters?**
**No (limitation of the current version)**. GGUF loading requires a UTF-8 path, while Windows command-line arguments use the native codepage;
place models/directories under a pure-English path. Audio paths may contain Chinese characters.

**Q10: Where are the original's UVR5 vocal separation / training / real-time features?**
This version only performs offline conversion and does not include those features; separate the vocals first with the original or another tool before feeding in the input audio.

**Q11: Why can't the pitch algorithm be selected?**
This version is fixed to RMVPE (a trade-off between accuracy and speed); pm/harvest/crepe are not available.

**Q12: Does model conversion degrade the audio quality?**
fp16 is imperceptible; q8_0 is near lossless; q5_1/q4_1 show slight differences; q4_0 differences are relatively audible (especially timbre details).
Using q8_0 or q5_1 is recommended to balance size and quality; the actual listening test is the final word.

**Q13: Can long audio (several minutes) and 48 kHz character models run stably?**
Yes (in builds since 2026-09-17). Earlier versions would abort on Vulkan when encountering longer chunks because the matmul workgroup count exceeded the limit
(especially with 48 kHz models). The fix requires **modifying the ggml source** (Vulkan matmul N-dimension splitting); the patch is in the source repository at
`patches/ggml-vulkan-matmul-n-split.patch`, technical details in §5.5 of `technical_02_ggml_vulkan_details.md`; be sure to apply it when building from source,
otherwise compiling against official ggml will reproduce the issue.

---

## 8. Differences from the Original at a Glance (Usage Perspective)

| Aspect | Original RVC | RVC-GGML v0.2.0 |
|---|---|---|
| UI | WebUI (browser) + CLI | Command line + interactive wizard (no WebUI) |
| Runtime environment | Python + torch + fairseq… | Single exe + driver |
| Model usage | Directly select .pth/.pt/.index | Must convert to GGUF first (including quant levels) |
| Hardware | Depends on the PyTorch backend (CUDA/DML…) | Vulkan, universal (cross-vendor) |
| Quantization | No official inference quantization | 5-level mixed quantization, VRAM/size adjustable |
| Pitch algorithm | 4 choices | RMVPE only |
| Vocal separation / training / real-time | Supported | Not supported |
| Long audio | Automatic chunking | Automatic chunking (same parameters as the original) |
| Non-ASCII paths | Supported | **Not supported for model paths** (audio is supported) |

---

## 9. Version Information

- Version: RVC-GGML v0.2.0 (Windows x64); 2026-09-17 fix build (long-audio / 48 kHz crash fix)
- Model format: GGUF layout v2 (`rvc.conv_flat`; the old layout still loads, but quantization requires re-export with this version's script)
- Random seed: fixed (unless reference data is injected)
- **Build note (important)**: This program links against a **modified local ggml source tree** — the Vulkan backend matmul gained
  N-dimension splitting, fixing the crash assertion with 48 kHz character models and very long audio (single chunk > 43.7 s) on Vulkan
  (`maxComputeWorkGroupCount`). The patch is in the source repository at `patches/ggml-vulkan-matmul-n-split.patch`;
  it must be applied when building from source, otherwise the issue reproduces. See §5.5 of `technical_02_ggml_vulkan_details.md` for details.
- Companion docs: `technical_01_port_overview.md` (Project Port and Implementation Overview), `technical_02_ggml_vulkan_details.md` (ggml/Vulkan Implementation Details)
