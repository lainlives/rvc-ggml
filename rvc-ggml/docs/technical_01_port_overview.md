# RVC-GGML Technical Document 1: Project Port and Implementation Overview

> Intended readership: readers who may not be familiar with C++ but understand the RVC project and the principles of voice conversion (VC).
> This document answers three questions: **how this rewrite project was advanced, with what approach each part was implemented, and how the C++ version differs from the original Python version**.
> It does not show large code excerpts; see `technical_02_ggml_vulkan_details.md` for implementation details.

---

## 1. Project Goal and Scope

### 1.1 Goal in One Sentence

Port the **offline inference** capability of RVC (Retrieval-based Voice Conversion; the original is a Python + PyTorch/WebUI implementation)
into a standalone executable written in **pure C++ + ggml**, accelerated via **Vulkan** on Intel / AMD / NVIDIA GPUs (with a CPU fallback),
producing converted speech that is **perceptually consistent** with the original.

### 1.2 Feature Comparison with the Original

|Feature|Original RVC|RVC-GGML (this project)|
|-|-|-|
|Offline voice conversion|✅ WebUI / CLI|✅ CLI + interactive wizard|
|Feature extraction|fairseq HuBERT / ContentVec|✅ Self-implemented equivalent structure|
|Pitch extraction|pm / harvest / rmvpe / crepe selectable|✅ RMVPE only|
|Retrieval (feature mixing)|FAISS IVF index|✅ Equivalent implementation (reads the same index data)|
|Synthesizer|VITS (SynthesizerTrnMs 256/768)|✅ v1(256) / v2(768), both f0 and non-f0 structures implemented|
|Model format|.pth / .pt / .index|GGUF (with mixed quantization)|
|Runtime dependencies|Python + torch + fairseq + faiss + librosa…|Single exe + system Vulkan driver + vcomp140.dll|
|Training / finetuning|✅|❌ Not supported|
|Real-time voice conversion (GUI/VST/ASIO)|✅|❌ Not supported|
|UVR5 vocal separation|✅|❌ Not supported (input must be already-separated vocals)|
|TTS|✅|❌ Not supported|

### 1.3 Technology Choices

* **Language**: C++20 (MSVC / Visual Studio 2026 toolchain, static CRT)
* **Inference framework**: ggml 0.23 (local source tree), serving both as the tensor/graph runtime and as the Vulkan / CPU backend.
  **Note: this project modifies the ggml source** — the Vulkan backend matmul gains N-dimension splitting (fixing long-audio/48 kHz model crashes;
  see 6.4 and §5.5 of `technical_02_ggml_vulkan_details.md`); the patch file is at `patches/ggml-vulkan-matmul-n-split.patch`;
  when building from source you must use a patched ggml, otherwise the runtime assertion reproduces
* **Backends**: Vulkan (default GPU) + CPU fallback; CUDA build options are kept but are not a target
* **Model format**: GGUF (metadata + weights), supporting fp32/fp16/q8_0/q5_1/q4_1/q4_0 mixed quantization
* **Delivery target**: a single-directory portable package for Windows 10/11 x64 (exe + models + dependency DLLs)

**Why ggml**: it is the low-level tensor library of the llama.cpp ecosystem, shipping with Vulkan/CPU backends, quantization kernels and cross-GPU-vendor
shaders; it fits the goal of "offline inference + quantization + cross-vendor GPUs" and avoids distributing a large runtime with the package.

---

## 2. The Original RVC Inference Pipeline (Align First on "What to Reproduce")

The core flow of one conversion in the original RVC (`infer/modules/vc/pipeline.py` etc.):

1. **Audio preprocessing**: any format/sample rate → mono 16 kHz; 48 Hz high-pass via `scipy.signal.filtfilt`;
long audio is chunked at "low-energy points" (`x_pad/x_query/x_center/x_max` parameters).
2. **Content features (HuBERT)**: the 16k waveform goes into fairseq HuBERT/ContentVec; v1 takes the layer-9 output projected to 256 dims via final_proj,
v2 takes the raw 768 dims of layer 12; the output is a feature sequence at 50 fps.
3. **Retrieval mixing (optional)**: uses a FAISS IVF index to find the top-k nearest neighbors among the training-set features, and
performs weighted mixing with the current features by `index_rate` (improves timbre).
4. **Pitch (f0)**: RMVPE estimates the fundamental frequency (Hz) from the 16k audio, scales it by the semitone pitch-shift factor,
and quantizes it into 1–255 pitch bins for the synthesizer's table lookup.
5. **Synthesis (VITS)**:

   * `enc_p`: features + pitch lookup → relative-position attention encoding → mean/variance (m_p/logs_p);
   * sample z_p, obtain z via the `flow` inverse transform;
   * `dec` (HiFiGAN NSF): uses SineGen to generate harmonic excitation from f0 (f0 models) and synthesizes the waveform together with z;
   * **Unvoiced protection (protect)**: unvoiced frames are replaced with the original features; **RMS mixing**: mixes the original audio's loudness by `rms_mix_rate`.
6. **Post-processing**: concatenate the chunks, optionally resample to `resample_sr`, normalize the result and write the WAV.

Runtime dependencies: Python, PyTorch, fairseq, faiss, librosa/scipy, etc.

---

## 3. Overall Port Progress

The project advanced in stages following "**split the pipeline first, align each model next, integrate the full pipeline last**", with an automatable
numerical acceptance criterion for each stage (compared against the Python reference implementation, correlation coefficient corr ≥ 0.999):

|Stage|Content|Acceptance|
|-|-|-|
|P0|Audio pipeline: resampling / high-pass / RMS / silence chunking|Sample-by-sample alignment with Python|
|P1|Model conversion tool: four GGUF types (hubert/rmvpe/character/index)|Tensor-by-tensor match with the source weights|
|P2|HuBERT graph|10 intermediate stages corr ≥ 0.99999|
|P3|RMVPE graph|mel/salience/f0 all aligned|
|P4|VITS graph (enc_p/flow/dec + SineGen)|audio corr ≥ 0.999999|
|P5|Full pipeline + retrieval + long audio|end-to-end corr ≥ 0.999, both backends|
|P6|Interactive CLI, mixed quantization, convolution quantization, release packaging|all levels usable on Vulkan/CPU|

**Why "per-stage numerical alignment"**: the biggest fear in neural-network ports is that "the whole sounds fine while a part is already wrong". The approach is to have the Python reference
dump tensors to disk at intermediate layers (mel, convolution outputs, attention outputs, m\_p/z, audio…), have C++ run the same inputs to produce the same layer's results,
and judge by correlation coefficient / SNR. This way any mis-implemented operator is caught at the earliest stage where it appears.

**Reference implementations consulted**: besides the original RVC, the graph organization of an existing community pure-ggml RVC implementation (0xShug0/audio.cpp, MIT)
was also consulted; but this project takes the original Python behavior as the standard.

---

## 4. How Each Part Is Implemented (at the Design Level)

### 4.1 Audio Pipeline

* **Polyphase FIR resampling**: replaces librosa (which uses soxr internally) for arbitrary sample rate → 16k; quality-level match (not sample-by-sample).
* **48 Hz high-pass**: hand-implemented zero-phase filtfilt (odd/even extension + forward/backward filtering); after the fix, **bit-identical** to scipy.
* **Silence chunking**: computes minimum points on the energy curve according to `x_pad/x_query/x_center/x_max` (seconds); same formula as Python.
* **RMS curve/mixing**: computes the energy envelope per frame; `rms_mix_rate` controls the mix ratio between the converted result's and the original audio's loudness.

### 4.2 Model Format (GGUF) and Mixed Quantization

* A Python script converts `.pth/.pt/.index` to GGUF: metadata (model version, sample rate, layer count, speaker count…)
plus all weights; PyTorch tensor shapes agree with ggml's `ne` semantics, with **no transposition**.
* **Mixed quantization**: large weights that are insensitive to quantization (attention/feed-forward/convolution) are quantized; numerically sensitive parts (bias, LayerNorm/
BatchNorm, embedding, GRU, flow coupling layers, transposed convolution, etc.) stay F32/F16.
* Key constraint and solution: a ggml quantized tensor's "row width" must be a multiple of 32, but convolution kernel widths (3/7/10/11…) do not satisfy this,
so **convolutions originally could not be quantized**. This project "flattens" convolution weights at export time into matrices with row width = input channels × kernel area,
and implements convolution at runtime with im2col + quantized matmul (see `technical_02_ggml_vulkan_details.md` for details).

### 4.3 HuBERT (Content Features)

* Structure reproduction: convolutional frontend (7 layers, downsampling at each layer) → LayerNorm → post\_extract\_proj → positional convolution (with GELU) →
LayerNorm → 12-layer Transformer (**Post-Norm** structure) → v1 takes layer 9 + final\_proj / v2 takes layer 12.
* Attention is implemented as "per-head small matmuls" (split the heads, then do a matmul on each head), consistent with PyTorch semantics and friendly to ggml.
* Features are output as \[frame count, 256/768], row-major consistent with Python.

### 4.4 RMVPE (Pitch)

* Composed of four subgraphs: **mel frontend** (host-side FFT; the filterbank directly uses the precomputed tables in the GGUF, consistent with librosa),
**DeepUnet feature network** (encode/middle/decode, with transposed-convolution upsampling), **bidirectional GRU** (forward/reverse),
and a **360-class salience head**.
* In the static graph the GRU is "unrolled" into a fixed-length sequence; streaming semantics across chunks is achieved by "state relay" (retain mask + final hidden state).
* f0 decoding: argmax + neighborhood weighted average (cents→Hz) + threshold-based unvoiced decision.
* Long audio: a two-tier strategy of whole-segment inference (≤ 16352 frames, about 163.5 seconds) and windowed inference with context.
  The threshold is back-derived from the Vulkan matmul N-dimension limit (it was set to 65536 frames before the fix, which would trigger the assertion at >163.8 seconds;
  see §3.6 and §5.5 of `technical_02_ggml_vulkan_details.md` for details).

### 4.5 Retrieval (FAISS IVF Equivalent Implementation)

* The training index (`.index`) is converted to GGUF: centroids, each inverted list's vectors/offsets/lengths and nprobe are all stored together.
* At runtime, following FAISS's IVF semantics: use the centroids to select the nprobe nearest lists → brute-force L2² distances within the lists → top-k →
  mix features with weights normalized as `(1/score)²`, then weight with the original features by `index_rate`.
* Results are element-wise identical to the FAISS original index (the nprobe parameter must be stored with the index — this was fixed).

### 4.6 VITS (Synthesizer)

* **enc_p**: `emb_phone` (linear) + `emb_pitch` (lookup by pitch bin) → scale + LeakyReLU →
  6 layers of \[relative-position multi-head attention + LayerNorm + FFN(k=3) + LayerNorm] → proj yields \[2×inter, T].
* **flow**: 4 groups of coupling-layer inverse transforms, with "channel flips" interleaved. The flip is implemented as a matmul with a 192×192 permutation matrix (avoiding extensive concatenation).
* **dec (HiFiGAN NSF)**: conv_pre → 4 levels of upsampling (transposed convolution) → each level superimposes "excitation noise convolution" and 3 residual blocks
→ LeakyReLU → conv_post → tanh. Outputs 40 kHz/48 kHz waveforms.
* **SineGen**: generates sinusoidal harmonics on the host side from f0 (random phase controlled by a fixed seed/injected noise, guaranteeing reproducibility),
  merged via tanh(linear) to serve as excitation.
* Layout: the encoder/flow parts use \[channels, time] (convenient for consistency with linear layers/attention), while the generator part uses \[time, channels] (the natural convolution order),
  with one transpose at the entry/exit.

### 4.7 Pipeline Orchestration and Long Audio

* Same order as Python: high-pass → chunking → per chunk \[HuBERT → retrieval → ×2 nearest-neighbor upsampling → protect → VITS → trimming] →
  concatenation → RMS mixing → optional resampling → int16 normalization.
* Graphs are **cached** by input length: a graph is built once per length; a length change triggers automatic rebuild (consistent with Python behavior).
* Generated random noise uses a fixed seed by default (results are reproducible across runs), which facilitates comparison.

### 4.8 Devices and Backends

* Enumerates Vulkan devices at startup (`--list-devices`); `--device 0/1/...` can be specified; `--device 99` forces CPU.
* If Vulkan device initialization fails or the device is unavailable, it automatically falls back to CPU and prints a warning.
* Workarounds for two known Intel driver issues are enabled by default (see `technical_02_ggml_vulkan_details.md`).

### 4.9 Interactive Mode and Command Line

* Starting with no action argument → a single wizard: character model → quant level → input audio → device → parameters → confirm conversion.
* The complete command-line argument system (`--input/--model/--quant/--pitch/--index...`) is retained for scripted invocation.

---

## 5. "In What Form the Original Is Reproduced"

Core principle: **reproduce at the formula and structure level, replace the runtime wholesale; reuse the weights directly**.

1. **Weights unchanged**: all neural-network weights are exported directly from the original .pth/.pt to GGUF, numerically identical (except fp16/quant levels).
2. **Formulas unchanged**: each layer's operation order, normalization placement, activation function (including the exact erf-based GELU), attention scaling,
convolution padding/trimming, and noise injection method are all reproduced from the original.
3. **Flow unchanged**: chunking strategy, retrieval mixing position (before ×2 interpolation), protect semantics, RMS mixing formula and normalization method are identical.
4. **Runtime replacement**:

   * torch/fairseq's autodiff framework → ggml static compute graphs;
   * FAISS → a self-implemented IVF list scan;
   * librosa/scipy audio utilities → self-implemented DSP (resampling/high-pass/FFT);
   * Python runtime → a single C++ process.
5. **How "reproduction" is verified**: with the same input, noise and f0, the C++ and Python outputs reach corr ≥ 0.999 at the audio level
(measured 0.9991\~0.9999 across pipelines, SNR 27–33 dB); the differences come mainly from floating-point accumulation order and quantization, not the algorithm.

---

## 6. Main Differences Between the C++/ggml Version and the Original

### 6.1 Runtime and Deployment

|Aspect|Original|This project|
|-|-|-|
|Dependencies|Python + torch + multiple C extensions|Single exe (+ system Vulkan driver + vcomp140.dll)|
|Startup cost|Interpreter + model loading (seconds)|Process start + model loading (hundreds of milliseconds)|
|GPU support|Depends on PyTorch (CUDA/ROCm/DirectML…)|Uniformly via Vulkan, cross-vendor|
|Quantization|No official inference quantization in the original (mostly fp32/fp16)|GGUF mixed quantization (fp16/q8_0/q5_1/q4_1/q4_0)|

### 6.2 Numerical and Implementation-Detail Differences

* **Precision**: the Python reference uses float64 for some operators on CPU, while this project is uniformly float32;
some convolutions internally use fp16 im2col, giving errors on the order of \~1e-3 (no perceptual impact; the verification thresholds were set accordingly).
* **f0 numerical sensitivity**: when the salience top1/top2 are extremely close, the argmax can flip (<0.1% of frames). A single wrong f0 frame causes a
permanent SineGen phase shift, dropping the correlation at the waveform level, but **perceptually it is only a 10ms phase difference** and is not an algorithmic error.
* **Resampling**: the original's librosa(soxr) and this project's polyphase FIR are not sample-by-sample identical (quality-level match); this has no effect when resampling is not used (the default).
* **Random noise**: the original uses random noise on every inference; this project uses a fixed seed by default (reproducible), and the verification mode can inject reference noise.
* **Retrieval**: FAISS's SIMD distance computation and the self-implemented loop may differ in the last bits under float32; the mixing weights are identical.

### 6.3 Functional Differences

* The pitch algorithm is RMVPE only (pm/harvest/crepe are not supported).
* Training/finetuning, real-time voice conversion, UVR5 separation and TTS are not supported.
* Input requirements: local WAV files; output is 16-bit PCM mono WAV.
* Aside from the original WebUI's "automatic model version detection", some peripheral configurations are not supported (e.g. custom f0 curves, harvest parameters, etc.).

### 6.4 Modifications to the ggml Source (Important)

This project **does not use official ggml directly**: the local ggml source has one necessary modification, without which long audio would crash.

* **What was changed**: the Vulkan backend matmul `ggml_vk_mul_mat` gains **N-dimension splitting** (the official implementation only splits very large M;
  the N dimension is never split). When `N > maxComputeWorkGroupCount[1] × 32` (= 2,097,120), computation is tiled by column;
  split_k is disabled within a tile, and the existing M-splitting logic was extracted into a shared function for reuse.
* **Why it must be changed**: Vulkan limits the number of workgroups in the Y direction to ≤ 65535, and matmul dispatches by `N/32`.
  The typical trigger is the last-level convolution of the VITS generator: `N = VITS frame count × hop`
  — a 48 kHz model chunk > 43.7 s, 40 kHz > 52.4 s, or RMVPE whole-segment > 163.8 s triggers the
  `ggml-vulkan.cpp:8617` assertion (reproducible on both Intel integrated and Arc discrete GPUs).
* **Patch and rebuild**: the patch is at `patches/ggml-vulkan-matmul-n-split.patch` (see the README in that directory for application instructions).
  When FetchContent pulls official ggml, the build system applies the patch automatically; for a local ggml source tree, the build checks for it,
  prints a WARNING if it is not found, and requires manual application; after upgrading the ggml source, it must be re-applied.
* **Impact**: purely dispatch partitioning; numerical results and behavior are unchanged for short audio/small models; it takes effect on the Vulkan path, while the CPU path is unaffected.
* Technical details and verification records are in §5.5 of `technical_02_ggml_vulkan_details.md`.

---

## 7. Per-Stage Acceptance Results (Summary)

|Stage|Result|
|-|-|
|P0 Audio pipeline|Resampling/high-pass/RMS/chunking aligned with Python (high-pass bit-identical)|
|P1 Model conversion|All four types (hubert/rmvpe/character/index) pass tensor-by-tensor validation|
|P2 HuBERT|10 intermediate results corr ≥ 0.999995|
|P3 RMVPE|mel corr=1.0, all voiced f0 frames consistent|
|P4 VITS|v1/v2 aligned at all stages, audio corr ≥ 0.999999, SNR ≥ 60 dB|
|P5 Full pipeline|end-to-end corr 0.9991\~0.9999 (including retrieval, 90s long audio), on both CPU and Vulkan backends|
|P6 Quantization|all five levels run on both CPU/Vulkan; q8_0 is near lossless (corr 0.9995), q4_0 has audible differences|

---

## 8. Known Limitations and Future Directions

1. **q4_0 quality drop is fairly noticeable**: the current strategy also quantizes the generator convolutions; this can be adjusted later based on listening tests (e.g. keep the generator at F16).
2. **Long-audio VRAM**: the VITS graph buffers for ~40s chunks can reach the GB level; finer chunk strategies/reuse are left for later.
3. **Non-ASCII paths**: GGUF loading fails when the model path contains Chinese characters (ggml opens files as UTF-8, while Windows command-line
parameters use the machine's ANSI codepage); audio paths are not affected. Using an English model directory is recommended; a later version will fix this.
4. **32k / non-f0 models**: supported structurally, but no corresponding models are currently available, so they have not been tested.
5. Further VRAM/quantization optimization (e.g. changing the im2col cache of quantized convolutions to F16) is left to the next stage.

