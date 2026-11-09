# RVC-GGML Source Structure and Module Manual

> For secondary development / code maintenance. Division of labor with Tech Doc 3: Tech Doc 3 covers "how the algorithms compute" (independent of the source code),
> while this manual covers "how the code is organized" — directory and file responsibilities, key classes and interfaces, object lifetimes, call chains and data flow.
> Complex sub-algorithms (flow/generator, GRU state handoff, resamplers, etc.) are given only their trunk and entry points here;
> dedicated documents can be produced when line-by-line breakdowns are needed (see §11).

---

## 1. Repository Structure and Build

### 1.1 Directory Overview

| Path | Contents | Purpose |
|---|---|---|
| `CMakeLists.txt` | Build script | Resolves the ggml dependency, generates the `rvc_core` static library and `RVC-GGML.exe`, detects ggml patches |
| `README.md` (repo root) | Project description | Overview, build and quick start |
| `docs/` | Docs 1/2/3/4 | Overview, ggml details, unified algorithm description, this source manual |
| `patches/` | ggml patches and notes | Vulkan matmul N-dimension splitting; must be reapplied after upgrading ggml |
| `include/` | `dr_wav.h`, `pocketfft_hdronly.h` | Third-party single-header libraries (WAV I/O, real FFT), unmodified |
| `src/` | All C++ implementation | See §3–§7 |
| `scripts/` | Python toolchain | Model conversion (export), numerical verification (verify) |
| `models/` | GGUF models | User-generated `hubert.gguf`, `rmvpe.gguf`, `<character>.gguf`, `<character>.index.gguf` and the various quant levels |
| `build/` | CMake build tree | Contains `RVC-GGML.exe` and the ggml subprojects |

### 1.2 CMakeLists.txt

1. **ggml dependency resolution (priority)**: `RVC_GGML_DIR` (cache variable) → `find_package(ggml)` →
   sibling directory `../ggml-master` → `../ggml` → FetchContent pulls official v0.23.0;
2. **Patches**: if Vulkan is enabled, official ggml lacks the N-dimension split patch; on the FetchContent path the build system
   automatically applies `patches/ggml-vulkan-matmul-n-split.patch` (idempotent; skipped if the source already contains the patch);
   for a local ggml source path it only detects — it reads `src/ggml-vulkan/ggml-vulkan.cpp` looking for the marker
   `ggml_vk_mul_mat_q_f16_m_split`, and emits `message(WARNING)` if missing (long audio / 48kHz models will assert at runtime,
   so the patch must be applied manually);
3. **Targets**: `rvc_core` (static library, all inference code) + `RVC-GGML` (`main.cpp`, links `rvc_core`);
   under MSVC, `/utf-8` and the static CRT are applied uniformly.

### 1.3 Layered Call Relationships

```
main.cpp
 ├─ engine        (GgmlDevice / GgmlGraphRunner)
 ├─ gguf_model    (metadata + weight upload)
 ├─ rvc/pipeline  (full-pipeline orchestration)
 │   ├─ hubert/     HubertEncoder
 │   ├─ rmvpe/      RmvpeExtractor
 │   ├─ vits/       Synthesizer ── text_encoder / flow / generator
 │   ├─ rvc/retrieval  RetrievalIndex
 │   └─ rvc/sinegen    SineGen
 ├─ audio/        (WAV, resampling, DSP, mel)
 └─ vits/modules  (operator library shared by the three graph builders hubert/rmvpe/vits)
```

---

## 2. Common Infrastructure

### 2.1 `src/utils.{h,cpp}`

- `Timer`: millisecond timing and formatting (logs elapsed time);
- `Logger`: `INFO/WARN/ERROR` three levels, output to stderr; `progress()` simple progress bar;
- `utils::` file and string helpers: `file_exists`, `list_files(dir, ext)`, `stem`, `filename`,
  `ensure_directory`, `trim/split`. Used by the CLI wizard and the model loading paths.

### 2.2 `src/engine.{h,cpp}`

**`GgmlDevice`** (RAII holder of `ggml_backend_t`):
- `device_id >= 0`: enumerate `GGML_BACKEND_DEVICE_TYPE_GPU/IGPU` devices and initialize the Nth one;
  on initialization failure or out-of-range, print a warning and fall back to CPU;
- `device_id == 99`: initialize the CPU backend directly;
- `name()` for log and interactive UI display.

**`GgmlGraphRunner`** (execution container for all static graphs):
- Constructor parameters: `ctx_mem_size` (tensor metadata pool), `max_nodes` (graph node limit);
- Lifecycle: `ctx()` build graph → `expand(node)` record outputs → `allocate()` allocate → `compute()` →
  `set_input/get_output` read/write;
- Key implementation: `allocate()` = `ggml_gallocr_new(default_buffer_type)` +
  `reserve` + `alloc_graph`; records `buffer_size()`/`node_count()` for logging and VRAM troubleshooting;
- Input/output: `ggml_backend_tensor_set/get`, must exactly match the tensor byte count (assertion-checked).

```
runner = GgmlGraphRunner(backend, ctx_MB, max_nodes)
build_graph(runner.ctx(), ...)            # each model module's responsibility
for out in outputs: runner.expand(out)
runner.allocate()
repeat: set_input(...); runner.compute(); get_output(...)
```

### 2.3 `src/gguf/gguf_model.{h,cpp}`

**Interface**:

| Member | Description |
|---|---|
| `load(path, backend)` | Parse metadata + upload all weights to the backend |
| `load_meta_only(path)` | Metadata only (the interactive wizard scans models without consuming VRAM) |
| `weight(name)` / `has_weight` | Get a tensor pointer (exact name match) |
| `meta_u32/f32/bool/str/array_str/array_i32` | Metadata access (with fallback) |
| `read_f32(name)` | Copy a weight back to the host and convert to F32 (supports F32/F16; quantized tensors unsupported) |
| `tensor_count()` | Count |

**Weight upload flow** (`load_weights`):

```
buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx_weights, default_buft(backend))
file   = _wfopen(path)                          # wide-char path, supports non-ASCII filesystem paths
for t in ctx_weights:
    tid = gguf_find_tensor(name)
    offset = data_offset + tensor_offset(tid)
    read exact byte count → ggml_backend_tensor_set(t, buf, 0, ggml_nbytes(t))
```

Note: tensor data is transferred without conversion (GGUF's quantized block layout matches ggml's memory layout); the
F32/F16 distinction is made at load time, and dequantization happens during compute. `read_f32` is used for the small number of weights whose values are needed on the host
(e.g. the RMVPE mel filterbank, SineGen's `l_linear`, relative position embeddings).

---

## 3. Audio and DSP (Host Side)

### 3.1 `src/audio/audio_io.{h,cpp}`

- `AudioData`: `samples` (interleaved float), `sample_rate`, `num_channels`, `total_samples`;
- `load_wav`: dr_wav reads any PCM/float format, uniformly converting to interleaved float;
- `save_wav` / `save_wav_i16`: write 16-bit PCM; the pipeline uses the latter to write already-quantized int16 directly to disk;
- Resampling core `windowed_sinc_resample`: Kaiser window (beta≈7.865) windowed sinc, polyphase structure,
  cutoff frequency `0.95*min(src,dst)`; `resample_mono` is the mono wrapper;
- `to_mono_16k`: first downmix by averaging channels, then resample to 16k.

Call sites: `main::run_convert` reads the input, the pipeline optionally resamples the output, `--audio-selftest`.

### 3.2 `src/audio/dsp.{h,cpp}`

| Function | Implementation notes | Called by |
|---|---|---|
| `high_pass_48hz_in_place` | 5th-order Butterworth coefficients + `lfilter_zi` initial state + odd-extension boundaries, filtered once forward and once backward (filtfilt); bit-identical to scipy | pipeline §1, audio-selftest |
| `reflect_pad` | numpy `mode='reflect'` semantics (no endpoint duplication) | split-point curve, pipeline padding, RMS |
| `frame_rms` | window length `sr/2*2`, hop `sr/2`, center alignment, boundary reflection (aligned with librosa) | RMS mixing |
| `interpolate_linear` | linear interpolation to the target frame count (aligned with `F.interpolate(mode='linear', align_corners=True)`) | RMS mixing |
| `apply_rms_mix` | `out *= rms_src^(1-r) * rms_out^(r-1)` | pipeline §6 |
| `audio_sum_curve` | 160-point sliding `|x|` absolute sum (curve used for split-point search) | self-test, chunking |
| `quiet_split_points` | search for the curve minimum within ±`t_query` around `t_center` intervals | **for self-test comparison**; the pipeline main loop inlines an equivalent implementation |
| `coarse_pitch_bin` | `1 + round(12*log2(f0/50))`, range 1..255, unvoiced 0 | pipeline (whole-segment and per-chunk) |
| `median_filter_f0` / `interpolate_unvoiced` | f0 median filtering / unvoiced interpolation | currently retained utilities, not called by the main pipeline |

### 3.3 `src/audio/mel.{h,cpp}`

- `make_mel_filterbank`: librosa HTK-style triangular filterbank (needed only for reference/self-test; the pipeline uses the GGUF table directly);
- `log_mel_spectrogram`: pocketfft real FFT (Hann window, center mode) → power spectrum → filterbank →
  log → clamp; output `[n_mels, frames]` row-major. Sole consumer: RMVPE.

---

## 4. Operator Wrapper Layer `src/vits/modules.{h,cpp}`

Positioning: a **functional operator library** (takes a `ggml_context*` and tensors, returns a new tensor), does not manage graph lifetime;
shared by the three graph builders HuBERT / RMVPE / VITS. All operators output F32 (quantized weights are dequantized internally by `mul_mat`).

| Group | Interface | Notes |
|---|---|---|
| Context | `GraphContext` (ctx+graph container) | Lightweight wrapper; the three model builders actually use `GgmlGraphRunner` |
| Basic | `linear` | `mul_mat(W,x)` + bias broadcast; `W=[K,N]` |
| Convolution | `conv1d` (`flat` switch) | flat: `im2col + mul_mat` (quantization-friendly); non-flat: `ggml_conv_1d` |
| | `conv1d_grouped` | per-group `view+cont+conv2d` (H=1) then `concat` |
| | `conv1d_ct` / `conv1x1_ct` | `[C,T]` layout wrapper (internally transposes to `[T,C]`) |
| Normalization | `layer_norm` / `group_norm` | along `ne0` / channel dimension `ne2` |
| Indexing | `get_rows` | pitch embedding, speaker embedding lookup (F32 tables) |
| Relative position | `relative_to_absolute` / `absolute_to_relative` | enc_p relative logits ↔ attention weight shape transforms |
| Utility | `sliding_window` | `unfold` semantics (not used in the main pipeline) |
| | `make_const_f32` | register constant inputs (written once after allocation) |
| | `to_f32` / `permute_cont` / `slice_channels` | type promotion, permutation + contiguity, `[C,T]` channel slicing |
| | `mul_checked` / `add_checked` / `check_dim*` | broadcast/dimension validation, fail early at build time |

**flat convolution implementation notes** (inside `conv1d`): the kernel parameter tensor only provides the shape (im2col does not read its data);
column vector type = weights F16→F16, otherwise→F32; the output `[O, OW*B]` is converted back to `[OW, O, B]` via `permute_cont`.
This corresponds one-to-one with the pseudocode in Tech Doc 3 §4.2.

---

## 5. Model Modules

### 5.1 HuBERT `src/hubert/hubert.{h,cpp}`

**Responsibility**: 16k waveform → content features (v1 256-dim / v2 768-dim, 50 fps).

| Component | Description |
|---|---|
| `Config` / `read_config` | Conv channels/kernels/strides, hidden, heads, pos_conv groups, `conv_flat` |
| `feature_extractor` | 7 layers of Conv1d+GELU; layer 0 adds per-channel GroupNorm (reshape `[T,1,C,1]` then groups=C) |
| `encoder_layer` | Post-Norm Transformer: per-head `mul_mat` attention (q/k/v head slicing→`cont`), residual, LayerNorm, FFN (GELU erf) |
| `build_graph` | conv frontend → LayerNorm → `post_extract_proj` → `pos_conv` (pad + grouped conv + GELU + length trim) → `encoder.layer_norm` → 12 layers; outputs `layer9` (index 8, v1), `layer12`, `final_proj` |
| `HubertEncoder` | Reuse entry point: caches the graph by input sample count; `extract(audio, v2)` uploads the waveform, executes, fetches outputs |

```
HubertEncoder.extract(audio, v2):
    if built_samples != audio.size():        # rebuild only if the shape changed
        build(samples)                       # build graph + allocate + cache input/output tensors
    runner.set_input(in, audio); runner.compute()
    return get_output(v2 ? out_l12 : out_proj)
```

Debug: with `RVC_HUBERT_DUMP=1`, `debug_tensors()` collects intermediate tensor names and pointers; the self-test program
`set_output + expand`s them and writes them to disk (`cpp_dbg_*.raw`) for stage-by-stage numerical comparison.

### 5.2 RMVPE `src/rmvpe/rmvpe.{h,cpp}`

**Responsibility**: whole 16k waveform → f0 (Hz). Four static subgraphs + host-side windowing/chunk scheduling.

| Subgraph | Build function | Input/output |
|---|---|---|
| feature (DeepUnet) | `build_feature_graph` | mel `[128, T]` → features `[384, T]` |
| GRU forward | `build_gru_graph(reverse=false)` | `[384, 512]` + keep + h0 → sequence `[256,512]` + final hidden state |
| GRU reverse | `build_gru_graph(reverse=true)` | same as above (expanded in reverse order within the chunk) |
| head | `build_head_graph` | `[fwd;rev]` → salience `[360,512]` |

**`RmvpeExtractor::Impl` state**: four `GgmlGraphRunner`s, their respective persistent input/output tensors,
the `feature_frames` cache, the mel filterbank (host).

**`infer()` main flow** (trunk):

```
mel = compute_mel(audio)                       # pocketfft frontend
padded = 32 * ceil(frames/32)                  # time-dimension padding
single_shot = padded <= kFeatureSingleShotLimit
ensure_feature_graph(single_shot ? padded : 8192)     # rebuild only on shape change
for each window (whole segment or 4096 active + 2×2048 context):
    copy_mel_window(...); feature.compute(); copy the active region into features
for each 512-frame chunk (forward): fill_gru_chunk → fwd.compute → save sequence + final-state handoff
for each 512-frame chunk (reverse, from the tail backwards): rev.compute → head.compute → write salience
f0 = decode_salience(salience, frames, threshold)     # argmax + neighborhood weighting + threshold
```

**Key constants** (top of the same file): `kFeatureActiveFrames=4096`, `kFeatureOverlapFrames=2048`,
`kFeatureSingleShotLimit=16352` (Vulkan safe limit; derivation in Tech Doc 3 §6.2), `kGruChunkFrames=512`.

**Helper functions**: `copy_mel_window` (window alignment/zero padding), `copy_feature_rows` (take only active rows),
`fill_gru_chunk` (zero-pad when frames < 512 + keep mask), `decode_salience`.

### 5.3 VITS `src/vits/*`

**Config**: `synthesizer.h::Config`, read from GGUF metadata by `read_config`; `hop_samples =
prod(upsample_rates)`; `flow_n_layers` detected from weight names.

**Submodules and build functions**:

| File | Function | Responsibility |
|---|---|---|
| `text_encoder.cpp` | `expand_relative_embedding` | expand the `[2W+1, D]` embedding window into a `[2T-1, D]` or transposed `[D, 2T-1]` constant |
| | `attention_layer` | per layer: q/k/v (1×1 conv) → per-head score + relative logits → softmax → context + relative value term → output projection |
| | `ffn_layer` | two-layer convolution with k=3 same padding + ReLU |
| | `build_text_encoder_stats` | feature projection + pitch lookup + `√H` scaling + LeakyReLU → 6 layers → proj → `[2*inter, T]` |
| `flow.cpp` | `build_wn` | WaveNet residual block: `cond_layer` computes the conditioning for all layers at once, sliced per layer and `repeat`ed; `in_layers` + dual gate (tanh*sigmoid) + `res_skip_layers` branching |
| | `coupling_reverse` | mean-only inverse affine: `x1 -= post(WN(pre(x0), g))` |
| | `build_flow_reverse` | order `{6,4,2,0}`: each step first flips channels with `mul_mat(flip_matrix, z)`, then applies the inverse coupling |
| `generator.cpp` | `conv_transpose1d_pytorch` | `ggml_conv_transpose_1d(p0=0)` then trim `(k-s)/2` on both sides |
| | `resblock1` | 3 groups (dilation 1/3/5): `convs1` with dilation + `convs2` kernel 3, residuals summed |
| | `build_generator` | `conv_pre` → conditioning term → 4×[LeakyReLU + upsampling + noise conv + mean of 3× residual blocks] → LeakyReLU(0.01) → `conv_post` → tanh |

Noise conv rule (inside `build_generator`): `stride = product of the remaining upsampling rates`; kernel length `stride==1 ? 1 : 2*stride`.

**`synthesizer.cpp` orchestration**:

```
Synthesizer.build(frames, feature_dim, sid):
    runner = GgmlGraphRunner(1024MB, 65536)
    in_features [feature_dim, frames], in_noise [inter, frames]
    (f0) in_pitch [frames] I32, in_source [frames*hop]
    sid constant tensor → get_rows(emb_g) → g
    flip constant (192×192 anti-diagonal set to 1)
    stats = build_text_encoder_stats(...)      # relative-position constants registered at build time
    split m/logs; z_p = m + exp(logs)*noise*0.66666
    z = build_flow_reverse(...); audio = build_generator(...)
    five outputs set_output + expand + allocate
    constants and sid written only once
Synthesizer.infer(input):
    (rebuild if frames/feature_dim/sid change)
    upload features/pitch/noise/source → compute → fetch audio and m/logs/z_p/z (validation fields)
```

Layout: enc_p/flow use `[C,T]` throughout (`conv1d_ct` wrapper); the generator internally uses `[T,C]`; one `permute_cont` at the entry and one at the exit
(see the first line of `generator.cpp`).

---

## 6. Full Pipeline and Retrieval

### 6.1 `src/rvc/pipeline.{h,cpp}`

**Responsibility**: wire the components above into a single conversion equivalent to Python `pipeline.py`.

| Structure | Description |
|---|---|
| `PipelineConfig` | `x_pad/x_query/x_center/x_max` (seconds) |
| `ConvertOptions` | pitch shift, sid, index_rate, protect, rms_mix_rate, resample_sr, optional index pointer, validation mode (`noise_dir`/`f0_inject`) |
| `RvcPipeline::Impl` | Holds the three components (`HubertEncoder`/`RmvpeExtractor`/`Synthesizer`), the `l_linear` coefficient cache, `call_index` (call sequence number for noise/file) |
| `vc(...)` | single chunk: HuBERT → optional retrieval → ×2 nearest neighbor → protect → noise/SineGen → `synth.infer` → trim `x_pad*tgt_sr` |
| `run(...)` | high-pass → split-point search (inline) → reflect pad → whole-segment f0 → per-chunk `vc` → concatenate → RMS → optional resample → int16 |

**Call-order pseudocode** (simplified):

```
run(audio):
    high_pass_48hz(audio)
    opt_ts = (n+160 > x_max*16000) ? search low-energy points : []
    audio_pad = reflect_pad(audio, x_pad)
    f0 = rmvpe.infer(audio_pad)  # pitch shift / coarse quantization done once
    s = 0
    for t in opt_ts:  emit vc(audio_pad[s : t+2*x_pad+160], f0[s/160 : (t+2*x_pad)/160]); s = t
    emit vc(audio_pad[s : ], f0[s/160 : ])
    concatenate → apply_rms_mix → optional resample_mono → peak normalization → int16
```

Details of `vc()` (×2 nearest neighbor, protect formula, `pf = f0>=1 ? 1 : protect`) are in Tech Doc 3 §5.3.

### 6.2 `src/rvc/retrieval.{h,cpp}`

- `load()`: read the index with `GgufModel` (`index.dim/nlist/ntotal/nprobe` metadata +
  the four F32 tensors `centroids/vectors/list_offsets/list_lengths`); copy back to host and the model can then be released;
- `search()`: take the `nprobe` lists by centroid L2² → full scan within each list maintaining a top-k max-heap → ascending output (ties by position, ensuring determinism);
- `blend()`: for each frame take top-k (k=8), weights `(1/score)²` normalized and weighted, then
  `index_rate*mix + (1-index_rate)*feat`. Called in pipeline `vc()` before the ×2 interpolation.

### 6.3 `src/rvc/sinegen.{h,cpp}`

Host-side per-sample excitation generation: phase accumulation + `align_corners` linear interpolation + wrap-around compensation; unvoiced frames output noise;
noise can be injected externally (validation mode) or generated from a fixed seed. The output is then passed through `tanh(l_linear*x+b)` by the pipeline.

---

## 7. Program Entry Point and Interaction `src/main.cpp`

### 7.1 Startup Sequence

```
main:
    default-set GGML_VK_DISABLE_INTEGER_DOT_PRODUCT=1 / GGML_VK_DISABLE_F16=1
        (unless RVC_VK_ENABLE_INTDOT / RVC_VK_ENABLE_F16 explicitly enabled)
    parse command line (see usage doc)
    action dispatch:
        --list-devices        → list_gpu_devices()
        --audio-selftest      → run_audio_selftest()
        --hubert-selftest     → run_hubert_selftest()
        --rmvpe-selftest      → run_rmvpe_selftest()
        --vits-selftest       → run_vits_selftest()
        with --input          → run_convert()
        no action argument / --interactive → run_interactive()
```

### 7.2 `run_convert` (CLI Conversion)

```
load_wav(input) → to_mono_16k → GgmlDevice(device_id)
GgufModel: hubert, rmvpe, character (file suffix selected by --quant)
optional RetrievalIndex.load(index)
RvcPipeline pipeline(...); result = pipeline.convert(audio, opt)
save_wav_i16(output, result.pcm, result.sample_rate)
```

### 7.3 Self-Test Entry Points

| Entry point | Required data | Artifacts |
|---|---|---|
| `--audio-selftest` | any WAV | high-pass/RMS/split-point outputs written to disk, compared by `verify_audio.py` |
| `--hubert-selftest` | `output/ref` (Python reference) | `cpp_hubert_*.raw` (optionally dumps intermediate tensors) |
| `--rmvpe-selftest` | `output/ref` or `--input` + `--ref-dir` | `cpp_rmvpe_*.raw`; long audio written as `cpp_rmvpe_long_*` |
| `--vits-selftest` | `--model` + `--ref-dir` | `cpp_model_*.raw` (including SineGen inputs/outputs) |

Self-test modes reuse the production inference code (the same `GgmlGraphRunner`/model classes), only adding input injection and result dumping.

### 7.4 `run_interactive` (Wizard)

1. `utils::list_files(models, ".gguf")` excluding `.index.`, use `GgufModel::load_meta_only`
   to read each model's version/sample rate/f0/speaker count (`VoiceModelEntry`);
2. scan available quant levels (requires both `hubert.<q>.gguf` and `rmvpe.<q>.gguf` to exist); the default level is
   detected from the character filename (`detect_quant`), automatically switching to the character file at the same level;
3. list `input/*.wav` or accept a custom path;
4. list devices (`list_gpu_devices`);
5. collect pitch shift, sid, index (only ask when a same-named `.index.gguf` exists), index_rate, RMS, protect,
   optional resampling and chunking parameters;
6. print a summary for confirmation and then call the same `run_convert` path, output default
   `output/<input-name>_<character-name>[_<quant>].wav`.

### 7.5 Debug Environment Variables

| Variable | Effect |
|---|---|
| `RVC_DEBUG_OP` / `RVC_DEBUG_CONV` / `RVC_DEBUG_LINEAR` | print broadcast/convolution/linear shapes |
| `RVC_HUBERT_DUMP` | register and dump HuBERT intermediate tensors |
| `RVC_RMVPE_DEBUG` | RMVPE subgraph node counts/buffer sizes |
| `RVC_VK_ENABLE_INTDOT` / `RVC_VK_ENABLE_F16` | restore Vulkan acceleration paths disabled by default |

---

## 8. Python Toolchain `scripts/`

### 8.1 `export_gguf.py` (Model Conversion)

Subcommands: `hubert | rmvpe | model | index | all`. Core components:

| Component | Description |
|---|---|
| `fold_weight_norm` | equivalently fold a `.pth` with weight_norm into plain weights |
| `flatten_conv1d/2d` | flatten convolution weights to `(O, I*K)` / `(O, I*KH*KW)` (layout v2) |
| `*_should_flatten` / `*_policy` | decide by tensor name and shape: mixed strategy of flatten, quantize, F32, F16 |
| `write_tensor_mixed` | quantize per level; fall back to F16 on failure or infeasibility; stats output `quantized x / F16 y / F32 z` |
| `make_rmvpe_mel_basis` | generate a librosa HTK-consistent mel filterbank and store it in GGUF |
| `export_index` | split a FAISS IVF index into four F32 tensors + `index.nprobe` metadata |
| `export_all` | batch-convert hubert/character/index from the RVC root directory |

### 8.2 `verify_*.py` (Comparison)

| Script | Comparison target |
|---|---|
| `verify_gguf.py` | GGUF vs `.pth/.pt/.index`, tensor by tensor |
| `verify_tensor.py` | generic raw tensor comparison (corr/SNR/max_err) |
| `verify_audio.py` | high-pass/RMS/split points recomputed against scipy/librosa |

---

## 9. Full Call Chain of One Conversion (Function Level)

```
main
 └─ run_convert
     ├─ load_wav / to_mono_16k                       (audio_io)
     ├─ GgmlDevice(device_id)                        (engine)
     ├─ GgufModel::load ×3 (hubert/rmvpe/character) + optional RetrievalIndex::load
     ├─ RvcPipeline::convert                         (pipeline)
     │   ├─ dsp::high_pass_48hz_in_place
     │   ├─ dsp::audio_sum_curve / split-point search
     │   ├─ dsp::reflect_pad
     │   ├─ RmvpeExtractor::infer                    (rmvpe)
     │   │   ├─ compute_mel → ensure_feature_graph → feature.compute
     │   │   ├─ fwd/rev chunked handoff + head.compute
     │   │   └─ decode_salience
     │   └─ per-chunk vc:
     │       ├─ HubertEncoder::extract               (hubert)
     │       ├─ RetrievalIndex::blend                (retrieval)
     │       ├─ sinegen::generate                    (sinegen)
     │       └─ Synthesizer::infer                   (vits)
     │           ├─ runner.set_input / compute / get_output
     │           └─ the same graph built by text_encoder/flow/generator
     ├─ dsp::apply_rms_mix → resample_mono (optional)
     └─ peak normalization → int16 → save_wav_i16
```

---

## 10. Key Design Decisions and Conventions

| Design | Rationale |
|---|---|
| Each model class internally owns a `GgmlGraphRunner` and caches the graph by input shape | Avoid repeated graph building/allocation; automatically rebuild on length changes (long audio chunking) |
| Constant inputs (relative position embeddings, flip matrix, kernel placeholders) are written once after allocation | These tensors are invariant per frame, avoiding repeated uploads on every inference |
| Model weight tensors reside permanently in backend buffers; graph buffers are planned centrally by the graph allocator | Weights are reused across inferences; intermediate buffers are allocated according to the graph's peak demand |
| `mul_mat` is the only matrix multiplication entry point; quantized weights must be in `src0` | Compatible with ggml's quantization path (a hard constraint on CPU/Vulkan) |
| Convolution uniformly goes through `im2col + mul_mat` (new layout) | Makes convolution weights quantizable; the old layout can still use native convolution |
| Tensors in the model uniformly output F32; `[C,T]` and `[T,C]` are converted only at boundaries | Aligned with upstream operator semantics (norm/softmax along `ne0`) and the quantization path |
| Validation mode achieves per-sample comparison by injecting reference noise/reference f0 | Isolates randomness and RMVPE argmax flips; `--noise-dir`/`--f0-inject` |
| Self-test and production inference reuse the same code path | Guarantees that what is verified is the released behavior |

---

## 11. Complex Parts and Suggested Follow-Up Dedicated Documents

The following parts currently only get their trunk in this manual / Tech Doc 3; if line-level descriptions are needed, each is suggested to become its own document:

1. **VITS flow / WaveNet / generator**: conditioning slice broadcasting, `res_skip` channel branching, noise convolution stride rules,
   `conv_transpose1d` trimming and output alignment;
2. **RMVPE windowing and GRU state handoff**: window alignment functions, semantics of the keep mask at the first frame/padding frames,
   interleaving order of reverse chunks and head;
3. **Vulkan matmul N-dimension split patch** (`patches/`): view offsets, rationale for disabling split_k,
   compatibility with official M splitting;
4. **Audio resampler**: Kaiser window parameters, fractional-delay polyphase structure, sources of differences from soxr;
5. **Relative position attention**: index derivation of `relative_to_absolute/absolute_to_relative` and
   embedding window expansion (`expand_relative_embedding`);
6. **Quantization strategy**: complete rule tables and fallback logic for each `*_policy` in the export script;
7. **FAISS index conversion**: IVF list layout, nprobe semantics and determinism guarantees of search order.
