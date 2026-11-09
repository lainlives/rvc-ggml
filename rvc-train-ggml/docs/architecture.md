# RVC-Train-GGML Architecture Design

> Intended audience: developers/contributors. After reading, you should be able to answer: what parts the system
> consists of, how data flows, how weights travel, how graphs and memory are managed, what format is exported,
> and why it is done this way.

---

## 1. Overview

```
┌────────────────────────────────────────┐        ┌─────────────────────────────────────────────┐
│ rvc-prep.exe                           │        │ rvc-train.exe                               │
│ links rvc_core (ggml Vulkan)           │        │ links traincpp (Vulkan / CPU debug)         │
│                                        │        │                                             │
│ raw audio ──slice/resample/high-pass───┼──cache──▶ dataset (bucket)──dual-graph G/D──AdamW──▶ │
│ f0(RMVPE) / HuBERT / linear spectrogram│ files  │ checkpoint (GGUF, resumable)                │
└────────────────────────────────────────┘        │ exporter ──▶ inference GGUF (rvc.*)         │
        (separate processes, no linking)          └────────────────────────────┬────────────────┘
                                                                               │
                                                                       ┌───────▼────────┐
                                                                       │ RVC-GGML.exe   │ inference
                                                                       │ ggml Vulkan    │
                                                                       └────────────────┘

Tools (Python, one-shot conversion/validation):
  tools/pth_to_train_gguf.py  official .pth ──▶ training-state GGUF (finetune/resume input)
  tools/verify_train_gguf.py  compare/validate training-state GGUF against source .pth
```

**Why two exes (key decision)**

- `rvc_core` depends on ggml; `traincpp` has its own Vulkan backend. Two Vulkan backends in the same process
  create device/VRAM/driver-level sharing and conflict risks, and neither was designed for resource coordination with the other.
- Hence **process-level isolation**: preprocessing and training are two exes that exchange data through a **file cache**; at the build level they share one CMake project.
- Consequences: `rvc-prep` does not link traincpp; `rvc-train` does not link ggml/rvc_core.

---

## 2. Directory and Module Responsibilities

```
rvc-train-ggml/
├── CMakeLists.txt            top level: dependency resolution + two exes
├── docs/                     usage guide, numbers handbook, architecture design (this document)
├── src/
│   ├── common/               config (JSON)/logging/paths/byte utilities (in-house, no third party)
│   │   └── json.*            minimal JSON parser (config/run section/manifest)
│   ├── model/                RVC model assembly (training state)
│   │   ├── weights.*         training-state GGUF loading (by name/shape validation)
│   │   ├── config.*          ModelConfig (TrainMeta promotion + validation/derivation)
│   │   ├── model.*           parameter container ModelParams (param creation/grouping/expected manifest)
│   │   ├── emb.*             emb_g (speaker embedding)
│   │   ├── wn.*              WN gated encoder (in/cond/res_skip)
│   │   ├── resblock.*        ResBlock1 (k=3/7/11, d=1/3/5)
│   │   ├── selftest.*        forward comparison harness (golden manifest/graph building/comparison)
│   │   ├── enc_p.*           TextEncoder (relative-position MHA + EncoderLayer merged implementation)
│   │   ├── enc_q.*           PosteriorEncoder (pre+WN+proj+z sampling)
│   │   ├── flow.*            ResidualCouplingBlock forward
│   │   ├── dec.*             GeneratorNSF
│   │   ├── sinegen.*         host-side SineGen (pure C++, no ggml dependency)
│   │   ├── discriminator.*   DiscriminatorP/S + MPD aggregation (S grouping/P folding/fmap)
│   │   ├── scratch.*         random initialization for training from scratch (PyTorch default init spec)
│   │   └── synthesizer.*     SynthesizerTrn training forward (enc_p→enc_q→flow→dec two paths)
│   ├── prep/                 rvc-prep preprocessing (load/highpass/resample/slicer/norm/pipeline/f0/feat/spec/cache)
│   ├── data/                 cache read-only, bucketed sampling, batch assembly (official TextDataset assembly + bucket pad/mask + Rng injection)
│   ├── train/                G/D dual graph, losses, optimizer orchestration, training loop, logging, resume metadata, self-test
│   │   ├── stft_mel.*        differentiable STFT/mel (DFT kernel conv_1d + in-house slaney filterbank)
│   │   ├── losses.*          LSGAN D/G, feature matching, KL, c_mel·mel L1
│   │   ├── train_step.*      single-ctx shared-parameter dual-graph TrainStep + memory diagnostics
│   │   ├── trainer.*         training loop (epoch shuffle/ExponentialLR/logging/checkpoint)
│   │   ├── resume_meta.*     resume metadata (sidecar <ckpt>.meta.json + config/sample fingerprint validation)
│   │   └── *_selftest.*      numerical comparison self-tests (--selftest-*, for development-time validation)
│   ├── export/               inference GGUF export (model_gguf) + index GGUF write/read (gguf_writer/npy/kmeans/index_gguf)
│   ├── cli/                  interactive wizard (train_wizard / prep_wizard)
│   ├── main_prep.cpp         rvc-prep entry point
│   └── main_train.cpp        rvc-train entry point
├── tools/                    Python: pth → training-state GGUF conversion and validation (not a runtime dependency)
├── configs/                  training config templates (own schema + official field semantics)
├── scripts/                  dev_build.ps1 (one-command build), train_pipeline.ps1 (full training pipeline driver)
└── reference/                reference material placeholder (read-only)
```

---

## 3. Data Flow

### 3.1 Preprocessing (rvc-prep)

Aligned with the original RVC's `train/preprocess.py` + `dataset/extract_f0.py` + `extract_hubert_feature.py`:

1. Read WAV (dr_wav) → mono → target sample rate (load branch = in-house Kaiser-window sinc `resample_sinc`;
   16k branch = bit-level emulation of resampy `kaiser_best` `resample_kaiser_best`; the official pipeline uses ffmpeg swresample,
   so the two cannot be sample-for-sample identical → statistical gate)
2. 48Hz high-pass: **official causal `scipy.signal.lfilter` (butter order 5, fs=target sample rate; filtfilt has been commented out/disabled upstream)**
   → in-house on the integration side (double; b/a from golden comparison); **do not use** `rvc_core::dsp`'s 16k filtfilt
3. Slicer2 slicing (`threshold=-42/min_length=1500/min_interval=400/hop_size=15/max_sil_kept=500`,
   exact replication including banker's rounding) → 3.7s/3.4s chunks + peak normalization → 0_gt (F32)
4. 16k branch (3.7s chunk resampling; official uses librosa kaiser_best/resampy → statistical gate):
   - RMVPE (`rvc_core::rmvpe`) → f0 Hz → **UV linear interpolation on unvoiced frames** (official `np.interp`; no median filtering);
     coarse-quantized bins (mel scale 1..255, `np.rint`) → f0bin
   - HuBERT (`rvc_core::hubert` graph structure; this project builds its own executor `src/prep/feat.*`, bypassing the inference-side executor's
     duplicate-compute drift defect) → v1 final_proj(256) or v2 layer12(768); input is raw 16k samples
     (no normalization, no padding; frame count `(L-400)//320+1`)
5. Linear spectrogram cache (pocketfft host side, center=False + reflect pad `(n_fft-hop)/2`, periodic Hann,
   magnitude `sqrt(re²+im²+2e-7)`; training convention, see §5.3)
6. Write the training cache (custom format, see §5.2) and sample manifest
7. Retrieval index: features (`.npy`/cache) → `index.gguf` (`--index` **defaults to full IVF**:
   permutation + optional MiniBatchKMeans + IVF k-means + list grouping; `--index-mvp` falls back to a single list; `--search-index`
   search dump), loaded by RVC-GGML with zero changes

### 3.2 Training (rvc-train)

Each step (aligned with the original `train/train.py:425-527`):

1. Fetch batch (phone/pitch/pitchf/spec/wave/sid + mask)
2. G forward: `enc_p(phone,pitch)` → m_p/logs_p → `enc_q(spec)` → z → `flow(z)` → `rand_slice(z)` → `dec(z_slice, pitchf, g)`
3. `y_mel` (cached spectrogram slice, no gradient) and `y_hat_mel` (differentiable STFT, backprop into G)
4. D loss: `D(real), D(fake.detach())` → LSGAN → D graph backward → `optim_d.step()`
5. G loss: `D(real), D(fake)` (fmap) → `loss_gen + fm + 45·mel + 1·KL` → G graph backward → `optim_g.step()`
6. Every epoch: `ExponentialLR.step()`; checkpoint; export inference GGUF per policy

The training loop (`src/train/trainer.*`) uses a single-bucket graph (see §8), epoch shuffling (seed+epoch)/tail dropping,
G/D each stepping their own ExponentialLR once per epoch, `train.log`+`loss.csv`, checkpoint v2 dual partitions + shared RNG.
The save→load round trip has been verified: after A/B archiving, the 6 losses at the same step are bit-identical.

### 3.3 Export

Implementation: `src/export/model_gguf.{h,cpp}` (goes through train.cpp `trc_gguf`, host-only, compiled into `rvc-train`)
+ CLI `rvc-train --export-gguf <in.train.gguf> --out <out.gguf>`. Inference GGUF (RVC-GGML schema):
- Export only `enc_p / flow / dec / emb_g` (filter out `enc_q.*`/`discriminators.*`)
- `weight_g/weight_v` → after folding `w = g·v/‖v‖`, write `weight` (standard dim0 convention)
- Conv1d `[O,I,K]` → flatten to `(O, I*K)`; `dec.ups.*` (ConvT1d) is not flattened
- Write `rvc.*` metadata (version/sample_rate/has_f0/n_spk/inter_channels/.../conv_flat=true; `upsample_*` must be written)

Export correctness has been comparison-verified: per-tensor max_abs ≤ 3.6e-7 versus the official reference implementation (all F32).

### 3.4 Inference Closed Loop

`rvc-train --export-gguf` → `RVC-GGML.exe --input ... --model export.gguf --noise-dir <deterministic noise>`.

Measured reference (exported model F32, RVC-GGML with zero changes): strict `--vits-selftest` corr=1.00000000 / SNR~100dB
(CPU+Vulkan); end-to-end CPU corr 0.99906, Vulkan 0.99897; the exported artifact compared directly against the official inference model gives
corr ≥ 0.9999. Acceptance criteria: ① strict self-test corr≥0.999 ② export end-to-end ≡ official ≥0.9999 ③ CPU end-to-end ≥0.999.

### 3.5 Training↔Inference Compatibility Boundary

- **Model semantic baseline = original RVC runtime `infer/lib/infer_pack/{models,modules,attentions,commons}.py`**
  (RVC-GGML's validation baseline). The official training repo serves only as a reference for training-loop/loss semantics; its only mathematical difference from the runtime
  is SineGen (the training side uses the old `cumsum_shift` method).
- **Main path (release)**: the exporter writes "folded weight_norm + flattened Conv1d (except `dec.ups.*`) + `rvc.*` metadata",
  and RVC-GGML loads it directly with **zero changes**; `rvc.upsample_rates/kernel_sizes` must be written.
- **Optional small change (training-time preview, already implemented)**: RVC-GGML reads training-state GGUF directly, touching only the "file reading layer"
  (`RVC-GGML/src/gguf/gguf_model.{h,cpp}`):
  - End of `load_weights`: when `X.weight` is missing but `X.weight_g/weight_v` exist, allocate F32 `X.weight` in a separate ggml context +
    backing buffer (same shape as v, not flattened) and fold; `weight()` merges lookups across the main and folded contexts.
  - `meta_{u32,f32,bool,str,array_str,array_i32}`: when `rvc.*` misses, fall back to `rvc.train.*`.
  - **Do not touch**: inference graph/numerical path, SineGen, quantization layout, architecture-name checks, non-flattened convolution path
    (training-state 3D weights are already supported).
- **Forbidden**: modifying RVC-GGML's inference graph/numerical path for any reason (including switching SineGen to the new method, etc.).

---

## 4. Weight Chain

```
official .pth (G/D/pretrained)
   │  tools/pth_to_train_gguf.py (one-shot; keeps weight_g/v, enc_q, discriminators)
   ▼
training-state GGUF (traincpp ne order; key names = torch state_dict)
   │  src/model/weights.* (load by name + shape validation)
   ▼
parameter tensors in the train.cpp Context ──first-round forward/backward comparison & training──▶ checkpoint (GGUF)
   │
   └─ src/export/ ──▶ inference GGUF ──▶ RVC-GGML inference
```

- Finetune = load pretrained G/D and continue training (`--pretrain-g/--pretrain-d`); the mechanism is the same as resume (only the initial values differ).
- Train from scratch = no pretrained loading; initialize per PyTorch defaults (train.cpp `nn_init_*` already matches PyTorch).
- Training state and inference state are **two separate GGUF sets**: the former keeps `weight_g/weight_v`, the latter folds/flattens; do not mix them.

### 4.1 Training-State GGUF Schema (`rvc.train.*`)

Written by `tools/pth_to_train_gguf.py` (hand-rolled GGUF v3, standard library + numpy/torch only); read by
`src/model/weights.{h,cpp}` (TrainWeights).

- Tensors: name = torch `state_dict` key; `ne = reversed(torch.shape)` (scalars recorded as `[1]`);
  type F32 (current) or F16; data 32-byte aligned, little-endian.
- Metadata (missing fields read as 0/empty):

| key | type | description |
|---|---|---|
| `general.architecture` | string | always `"rvc-train"` (distinct from inference `rvc.*`) |
| `general.name` | string | output file stem |
| `rvc.train.kind` | string | `G` / `D` |
| `rvc.train.version` | string | `v1` / `v2` |
| `rvc.train.has_f0` | bool | f0 / non-f0 |
| `rvc.train.form` | string | source form: `train` (model keys)/`small` (weight+config) |
| `rvc.train.source_pth` | string | absolute path of source .pth (traceability) |
| `rvc.train.out_dtype` | string | `F32` / `F16` |
| `rvc.train.tensor_count` | u32 | number of tensors |
| `rvc.train.sample_rate` | u32 | 32000/40000/48000 |
| `rvc.train.spec_channels` | u32 | `filter_length/2+1` |
| `rvc.train.filter_length` / `hop_length` / `win_length` | u32 | STFT parameters |
| `rvc.train.n_mel_channels` / `mel_fmin` / `mel_fmax` | u32/f32 | mel parameters (write sr/2 when fmax is null) |
| `rvc.train.segment_size` | u32 | training segment length (may be 0 in the small-model embedded config) |
| `rvc.train.inter_channels` / `hidden_channels` / `filter_channels` | u32 | model channels |
| `rvc.train.n_heads` / `n_layers` / `kernel_size` | u32 | Transformer parameters |
| `rvc.train.p_dropout` | f32 | dropout |
| `rvc.train.gin_channels` / `spk_embed_dim` / `upsample_initial_channel` | u32 | remaining model parameters |
| `rvc.train.resblock` | string | `"1"` |
| `rvc.train.resblock_kernel_sizes` | i32[] | `[3,7,11]` |
| `rvc.train.resblock_dilation_sizes` | i32[] | flattened `[1,3,5,1,3,5,1,3,5]` |
| `rvc.train.upsample_rates` / `upsample_kernel_sizes` | i32[] | consistent with config |
| `rvc.train.d_periods` | i32[] | D only: official periods v1 `[2,3,5,7,11,17]`, v2 `[...] ,23,37` |

> Note: `spk_embed_dim` follows the actual value of `emb_g.weight.shape[0]` (warn if it disagrees with config).
> Inference small models (.pth extracted by WebUI) have no `enc_q` (finetuning requires official pretrained G/D weights).

---

## 5. Key Design Details

### 5.1 Graphs and Memory

- Static-shape graphs: one graph set per bucket (sequence length); one set each for G and D, per bucket.
- Dual-graph protocol (the shape verified in train.cpp):
  - G graph: G forward → feed D forward A (D frozen) → G loss; before building backward, call `param_list_set_param(d, false)`
  - D graph: D forward real + D forward fake (**newly created nodes**; must not reuse A's nodes) → D loss; freeze G before building
  - Step order: `graph_reset(gg)` → compute → `optim_step(g)`; `graph_reset(dg)` → compute → `optim_step(d)`
- Memory: prefer `buffer_alloc_graph_tensors` (graph-level liveness reuse); allocate the G graph first and the D graph after (shared tensors within the same ctx are skipped);
  if the peak does not fit, split the ctx or `buffer_free` + `context_release_graph_tensors` (metadata is not reclaimed).
- Constraints (known train.cpp semantics):
  - The same tensor must not appear in two graphs whose backward has already been built (building aborts)
  - Shared intermediates may be referenced across graphs, but on the frozen side they must not participate in backward (not included in needed propagation)
  - Intermediates are unreadable after compute (graph-level reuse overwrites them); use the old API for values that need reading
- **Landed form (`src/train/train_step.*`)**: **single ctx, shared-parameter dual graph**
  (D frozen in the G graph, trained in the D graph) — it has been proven that a single ctx can allocate in the G graph → build the D graph → allocate a second time;
  `set_batch`/`run_step` four-phase step order. Host-only leaves (Ids/pitchf/noise) do not create tensors and read files directly;
  scalars that must be read after compute are marked `TENSOR_FLAG_OUTPUT`.
- **Fallback (`src/train/train_step_selftest.cpp` keeps the implementation)**: two independent Contexts (ctx_g/ctx_d);
  D parameters in ctx_g are a frozen copy (same weight names, different objects → the multi-graph sharing guard is not triggered); release ctx_d immediately after the D step
  (after first snapshotting ΔW/gradient/loss), and allocate G optimizer state after release — smaller peak, runs on 8GB unified-memory-class
  iGPUs.

### 5.2 Training Cache Format

Principles: simple, stream-readable, decoupled from bucket sampling, F32 little-endian, row-major; Python format compatibility is not a goal.
Implementation in `src/prep/cache.{h,cpp}` (`write_cache_index`/`validate_cache`), `src/prep/spec.{h,cpp}`;
CLI = `rvc-prep --exp DIR`, `--validate-cache DIR`.

```
<exp_dir>/
  dataset.json          # cache metadata + sample manifest (see below)
  spec.json             # field dictionary (dtype/byte order/layout/derivation formula)
  samples/
    <id>.gt.bin         F32 [gt_samples]        # target_sr, after lfilter + peak normalization (= official 0_gt contents)
    <id>.16k.bin        F32 [frames16k]         # 16000Hz
    <id>.f0bin.bin      I32 [f0_frames]         # coarse-quantized bins 1..255
    <id>.f0hz.bin       F32 [f0_frames]         # continuous f0 Hz after UV interpolation (all-silent input skipped)
    <id>.feat.bin       F32 [feat_frames, dim]  # HuBERT; dim=256(v1)/768(v2), row-major (torch [T,C])
    <id>.spec.bin       F32 [bins, spec_frames] # row-major (torch [bins,T]); v1-40k bins=1025
```

- `dataset.json` key fields: `format="rvc-train-cache"`, `format_version=1`, `target_sr`, `model_version`,
  `has_f0`, `per=3.7`, `overlap=0.3`, `slicer{threshold,min_length,min_interval,hop_size,max_sil_kept}`,
  `spec{n_fft,hop,win,bins}`, `f0{f0_bin,f0_min,f0_max}`, `samples[]` each item:
  `{id, source, chunk, gt_samples, frames16k, f0_frames, feat_frames, feat_dim, spec_frames, spec_bins,
  files{gt,16k,f0hz,f0bin,feat,spec}}`; `id = <case>_<official idx1>`.
- Frame-count derivation (for self-tests): `feat_frames = (samples16k - 400)//320 + 1` (the HuBERT conv frontend has no padding,
  consistent with fairseq, **not** `ceil(samples16k/320)`);
  `f0_frames = samples16k//160 + 1` (RMVPE `torch.stft(center=True)` convention);
  **`spec_frames = 1+(padded-n_fft)//hop`, `padded = gt_samples + 2*int((n_fft-hop)/2)`**
  (reflect pad + center=False; measured for v1-40k = `gt_samples//hop`, 370 for a 3.7s chunk).
- Linear spectrogram semantics = official `spectrogram_torch` (2e-7 magnitude, periodic Hann; same convention as the §5.3 training mel);
  implementation `compute_linear_spec` uses pocketfft r2c (host side, not the training graph).
- Training consumption (`src/data/dataset.*`) matches official `data_utils.py:84-97`: phone ×2,
  truncated to `min(2*feat_frames,900)`, then truncated again to align with the spec length; the cache stores the full amount.

### 5.3 Differentiable STFT / mel and Losses

- `pad_ext(reflect)` (both sides (n_fft-hop)/2) → `conv_1d` folding the window into the DFT kernel (kernel `[n_fft,1,2*BINS]`,
  cos/sin blocks) → split re/im → `sqr/add/sqrt` → `mul_mat(mel_basis)` → `clamp/log`
- The DFT kernel/filterbank/ε are all constant graph tensors (not PARAM; filled by `stft_mel_fill_*` after allocation);
  complexity O(n_fft²·frames)
- **Numerical convention = official `train/mel_processing.py`** (training side): periodic Hann,
  `center=False` + explicit reflect pad, magnitude `sqrt(re²+im²+2e-7)`, `log(clamp(·, min=2e-6))`;
  **different** from the runtime infer_pack's 1e-6/1e-5 convention (do not mix). Comparison tolerances: forward ≤1e-4, backward ≤1e-3
- Landed: `src/train/stft_mel.{h,cpp}` (forward) + `src/train/losses.{h,cpp}` (LSGAN D/G,
  feature matching×2, KL, c_mel·mel L1) + `src/train/loss_selftest.{h,cpp}` (`--selftest-loss`).
  Measured reference (v1-40k-f0, CPU+Vulkan, 81 comparisons each with 0 failures): spec 9.5e-7, mel 7.2e-7,
  in-house slaney filterbank vs librosa 1.9e-9, loss/gradient rel ≤3.5e-6

### 5.4 Relative-Position Attention

Following the combination in train.cpp's "RVC Integration Guide" §5.2: `conv_1d(1×1)` + `mul_mat` (batch broadcast) +
`pad_ext/reshape/view/concat` to construct the relative table and the relative↔absolute transform; the mask uses `scores·M+(M−1)·1e4`
to replicate `masked_fill`, then ordinary `soft_max` (`soft_max_ext` with `max_bias>0` triggers ALiBi and is disabled).

### 5.5 Precision and Tolerances

- F32 throughout; golden comparison tolerances: forward ≤1e-4 (relative), backward ≤1e-3, one update step ≤1e-4.
- **Multi-step comparison of the training loop**: first-step hard gate ≤1e-4 (measured 5.3e-7); later steps amplify chaotically, so switch to a perturbation-probe ensemble
  envelope as the tolerance upper bound (loss must remain finite); end-to-end trajectory gate ≤1e-2.
- The inference closed loop is judged by "perceptually consistent": corr ≥0.999 under the same noise (aligned with RVC-GGML's acceptance criteria).

### 5.6 Forward Shape Table (v1-40k-f0)

> Model semantic baseline = original RVC runtime `infer/lib/infer_pack/{models,modules,attentions,commons}.py`
> (= RVC-GGML validation baseline; the only mathematical difference is SineGen, where the training side uses the old method).
> Convention: torch shapes below are `[B,C,T]`; when writing to GGUF, `ne = reversed(shape)`.
> Config: `inter=hidden=192, filter=768, heads=2, kernel=3, window=10, gin=256, up_init=512`;
> `upsample_rates=[10,10,2,2]` (product 400), `spec_channels=1025`;
> **in-model `segment_size` = train.segment_size//hop = 12800/400 = 32 frames** (train.py:207/215).

- `emb_g.weight [109,256]`; `g = emb_g(sid)[B,256]→unsqueeze(-1) [B,256,1]`.
- **enc_p** (TextEncoder):
  - `phone [B,T,256]` (the Linear input is the last dim 256) → `emb_phone [B,T,192]`;
    `pitch [B,T]` I32 → `emb_pitch [B,T,192]`; add ×√192 → LeakyReLU(0.1) → transpose `[B,192,T]`;
  - `x_mask [B,1,T]`; 6 layers: relative-position MHA (q/k/v/o 1×1; heads view `[B,2,96,T]→[B,2,T,96]`,
    `scores/rel_logits [B,2,T,T]`, `emb_rel_k/v [1,21,96]`, window=10) + Post-LN +
    FFN (conv1 `[768,192,3]`, conv2 `[192,768,3]`, padding left 1 right 1, **ReLU**) + Post-LN;
  - `proj [384,192,1]` → `stats [B,384,T] × x_mask` → `m_p/logs_p [B,192,T]`.
  - Note: `modules.LayerNorm` parameter names are **gamma/beta** (not weight/bias).
- **enc_q** (PosteriorEncoder):
  - `spec [B,1025,T]` → `pre [192,1025,1]` × `y_mask [B,1,T]` → `[B,192,T]`;
  - WN: 16 layers k5 d1; `cond_layer [6144,256,1]` (2·192·16=6144) → split `[B,384,T]` per layer;
    `in_layers.i [384,192,5]`, `res_skip_layers.i [384,192,1]` (192 for the last layer); all weight_g/v;
    `acts = tanh(x_in[:, :192]) * sigmoid(x_in[:, 192:])`; `x=(x+res[:192])*mask`, `output+=res[192:]`;
  - `proj [384,192,1]` × mask → `m_q/logs_q`; `z = (m_q + N(0,1)·exp(logs_q)) * y_mask` (noise injected on host).
- **flow** (training **forward** `reverse=False`, flows order `[RCL0, Flip, RCL1, Flip, RCL2, Flip, RCL3, Flip]`):
  - each RCL: `x0,x1=split(96)`; `pre [192,96,1]`; WN 3 layers; `post [96,192,1]` (mean_only, m is the stats);
    `x1 = m + x1*exp(0)*x_mask` (training forward is identity scaling); concat; Flip = **along the channel dim** `torch.flip(x,[1])`;
  - shapes stay `[B,192,T]` at every step.
- **rand_slice** (training): `rand_slice_segments(z, y_lengths, 32)` → `z_slice [B,192,32]`, `ids_slice [B]`;
  `slice_segments2(pitchf, ids_slice, 32)` → `pitchf_slice [B,32]` (**pitchf frame rate = 100fps, aligned with z frames**).
- **dec** (GeneratorNSF, f0 path; `upp=400`):
  - SineGen (**old `cumsum_shift` method**, consistent with RVC-GGML `src/rvc/sinegen.cpp`; noise injected on host) →
    `har_source [B,1,12800]` (32×400); `l_linear [1,1]`+tanh trainable inside the graph;
  - `conv_pre [512,192,7]` → `[B,512,32]`; `+cond(g) [512,256,1]`;
  - 4 levels: LeakyReLU(0.1) → `ups.i` ConvT1d (weight_g/v, `[I,O,K]`):
    `[512→256,k16,s10,p3]→320`, `[256→128,k16,s10,p3]→3200`, `[128→64,k4,s2,p1]→6400`,
    `[64→32,k4,s2,p1]→12800`; add `noise_convs.i` (conv_1d acting on har_source):
    `[256,1,80] s40 p20`, `[128,1,8] s4 p2`, `[64,1,4] s2 p1`, `[32,1,1]`;
  - 3×ResBlock1 per level (ch=256/128/64/32, k=3/7/11, d=1/3/5; convs1/convs2 all weight_g/v), averaged;
  - final LeakyReLU(**0.01**, `F.leaky_relu` default slope) → `conv_post [1,32,7]` (no bias) → tanh → `o [B,1,12800]`.
- **Discriminator** (v1 MPD, f0D40k weights): S: k15s1p7 1→16; k41s4p20 g4 16→64; g16 64→256; g64 256→1024;
  g256 1024→1024; k5s1p2 →1024; conv_post k3s1p1 →1 (7 fmap entries);
  P(period): `view [B,1,T/p,p]` (reflect-pad on the right if short) → 4×Conv2d k(5,1) s(3,1) p(2,0) 1→32→128→512→1024
  + 1×Conv2d s(1,1) → conv_post k(3,1) p(1,0) →1 (6 fmap entries); flatten(1,-1).

---

## 6. Config and CLI Conventions

```jsonc
// configs/v1-40k-f0.json (own schema: top-level train/data/model + optional run section)
{
  "run":   { "exp": "<cache dir>", "out": "<output dir>",
             "pretrain_g": "<training-state G.gguf>", "pretrain_d": "<training-state D.gguf>",
             "device": 1, "resume": "", "export_every": 0, "scratch": false,
             "export_run_json": "" },
  "data":  { "sampling_rate": 40000, "filter_length": 2048, "hop_length": 400,
             "win_length": 2048, "n_mel_channels": 125, "mel_fmin": 0.0, "mel_fmax": null },
  "model": { "version": "v1", "has_f0": true, "inter_channels": 192, "hidden_channels": 192,
             "filter_channels": 768, "n_heads": 2, "n_layers": 6, "kernel_size": 3,
             "p_dropout": 0.0, "resblock": "1", "resblock_kernel_sizes": [3,7,11],
             "resblock_dilation_sizes": [[1,3,5],[1,3,5],[1,3,5]],
             "upsample_rates": [10,10,2,2], "upsample_kernel_sizes": [16,16,4,4],
             "upsample_initial_channel": 512, "spk_embed_dim": 109, "gin_channels": 256 },
  "train": { "batch_size": 1, "learning_rate": 0.0001, "betas": [0.8,0.99], "eps": 1e-9,
             "weight_decay": 0.01, "lr_decay": 0.999875, "segment_size": 12800,
             "c_mel": 45, "c_kl": 1.0, "epochs": 20, "seed": 1234,
              "grad_accum": 1, "buckets": [128,192,256,384,512,768,900] }
}
```
> - The `run` section is an in-house extension (optional); non-empty CLI values override the `run` section. `--train <config>` can supply exp/out/pretrain_g/pretrain_d/device from the `run` section alone (the four CLI parameters are no longer mandatory). Old configs without a `run` section remain fully compatible.
> - Serialization/equivalence: `trainer_config_to_json`/`trainer_config_write_json`/`trainer_config_equivalent`
>   (`src/train/trainer.*`).
> - Training from scratch: when `run.scratch=true` (or CLI `--scratch`), the `ModelConfig` derived from the `model/data` sections
>   is used by `src/model/scratch.*` to generate G/D specs and initialize them; `src/export/mem_to_train_gguf.*` writes
>   `<out>/scratch/{g,d}.train.gguf`, then the existing TrainStep path is reused (zero changes to the training core).
> - Epoch snapshots/export: `Trainer::on_epoch_end` writes `epochs.csv`+`train.log` summary; `save_interval`→
>   `train.ep{N}.ckpt.gguf` (keep count `run.snapshot_keep`); `export_every`→ real-time G-parameter export of
>   training-state GGUF → reuse the exporter to produce `export.ep{N}.gguf`.

CLI:
```
rvc-prep  --input <wav/dir> --exp <dir> --sr 40000 --version v1 --has-f0 1 --device 0
          [--hubert models\hubert.gguf] [--rmvpe models\rmvpe.gguf]
          [--per 3.7] [--debug-dump <dir>] [--load-raw <f32> --load-raw-sr <sr>]
          [--no-16k] [--no-f0] [--no-feat] [--no-spec] [--limit N]
rvc-prep  --validate-cache <exp_dir>
rvc-prep  --index [--features <dir|file.npy>|--exp <cache_dir>] --out <index.gguf> [--version v1|v2]
          # full IVF by default; --index-mvp falls back to a single list; --nprobe/--seed/--kmeans-iters/--mini-centroids/--mini-epochs
rvc-prep  --validate-index <index.gguf>
rvc-prep  --search-index <index.gguf> --queries <q.npy> --search-out <hits.raw> [--k 8]
rvc-train --train <config.json> [--exp DIR] [--weights G.gguf] [--d-weights D.gguf] [--out DIR]
          [--device N] [--steps N] [--bucket T] [--batch N] [--epochs N] [--disc-len N]
          [--grad-accum N] [--save-interval N] [--export-every N] [--snapshot-keep N]
          [--resume <ckpt.gguf>] [--export-run-json <path>]
rvc-train --interactive [--config <base.json>] [--answers <file>]
          [--out DIR] [--device N] [--exp DIR] [--weights G] [--d-weights D] [--steps N] ...
rvc-train --scratch | --pretrain-g <train.gguf> --pretrain-d <train.gguf>   # from scratch / finetune (with --train/--interactive)
rvc-train --export-gguf <in.train.gguf> --out <infer.gguf>                  # export inference GGUF (host-only; fold/flatten/write rvc.*)
rvc-train --list-devices | --selftest-forward | --selftest-backward
rvc-train --selftest-* ...     # development-time numerical comparison self-tests (require golden data, not distributed with this repo, see src/train/*_selftest.*)
```

---

## 7. Build and Dependency Resolution

- `add_subdirectory(../third_party/train.cpp)` (submodule inside the monorepo; override with
  `-DRVCTRAIN_TRAINCPP_DIR=`): first set `TRC_BUILD_TESTS/EXAMPLES/INSTALL=OFF`;
  `rvc-train` links `traincpp::traincpp` (aggregate target, ensuring backend registration is not dropped by `/OPT:REF`).
- The training library is pinned to **train.cpp v0.4.0** (commit `e99d74b`): Vulkan compute buffers prefer
  device-local by default (discrete-GPU real VRAM + shared staging / mappable device-local / host-visible fallback),
  and the optimizer / gradient zeroing / clipping / WeightNorm sync transparently go device-side; `TRC_VK_HOST_MEMORY=1` falls back to
  host-visible, `TRC_OPT_HOST=1` forces the host optimizer (debug/A-B only).
- `add_subdirectory(../rvc-ggml)` (override with `-DRVCTRAIN_RVCGGML_DIR=`): it resolves ggml itself
  (official v0.23.0 + automatic patches, or local `RVC_GGML_DIR`); `rvc-prep` links `rvc_core`.
  Side effect: it also builds `RVC-GGML.exe` (harmless, and convenient for the inference closed loop).
- Runtime directory: `build/bin/`.
- Toolchain: MSVC (VS C++ toolset) + CMake ≥3.20 + Ninja; `scripts/dev_build.ps1` auto-detects
  vcvars64/cmake/ninja, and environment variables `RVCT_VCVARS/RVCT_CMAKE/RVCT_NINJA` can override.

## 8. Design Limitations and Future Directions

- **Single-bucket graph**: the training loop uses a single bucket per graph (currently default `bucket=384`; samples exceeding it are skipped);
  multi-bucket graph caching would require TrainStep to support rebuilding with variable T and is deferred; gradient accumulation (`train.grad_accum`) is already implemented.
- **Export precision**: exported GGUF is F32 for now; F16/quantization-level policy is an optional follow-up.
- **Long-T backward**: resblock1 backward at very large T (≥8192) has F32 cross-implementation kink divergence (not a library defect,
  but a numerical boundary); long-T comparison uses statistical criteria; when the whole D backward graph allocation exceeds 4 GiB, rely on graph-level liveness
  allocation (`buffer_alloc_graph_tensors`) and chunking control.
- **Wizard interaction**: the interactive wizard (`--interactive`) is the recommended main UX and will be adjusted based on user feedback.
