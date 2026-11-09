# RVC Inference: Unified Implementation Details from Python/PyTorch to C++/ggml

> This document is self-contained: it describes only implementation mechanisms, operator mappings, pipeline, and backend constraints.
> Readers need not consult the reference implementation or the original project; this document alone suffices to reproduce the overall flow and key algorithms.
> The content is primarily pseudocode and mathematical definitions; it does not cover engineering progress, release, or verification records.

---

## 1. Overall data flow

A single conversion is completed by one fixed pipeline, entirely static compute graph + host-side DSP:

```
WAV (any sample rate/channels)
  │ downmix + polyphase FIR resampling
  ▼
mono 16 kHz
  │ 48 Hz zero-phase high-pass
  ▼
whole-segment f0 estimation (RMVPE, on the whole audio after reflection padding)
  │
  ├─ split into chunks at low-energy points (x_pad / x_query / x_center / x_max, unit: seconds)
  │
  ▼  per chunk:
  HuBERT features (v1: layer 9 + final_proj→256; v2: layer 12→768)
  → (optional) FAISS IVF retrieval feature blending
  → ×2 nearest-neighbor upsampling
  → protect (unvoiced frames backfilled proportionally with original features)
  → pitch-shifted pitch bin + RMVPE f0
  → SineGen excitation (f0 model) → tanh(l_linear(·))
  → VITS (enc_p → inverse flow → HiFiGAN NSF decode)
  → trim head/tail by x_pad (target sample rate)
  │
  ▼ concatenate all chunks
RMS loudness mixing → (optional resample) → peak normalization → int16 WAV
```

Data space conventions:

| Stage | Sample/frame rate | Dimension notation |
|---|---|---|
| Waveform (model domain) | 40 kHz / 48 kHz | 1×N |
| Waveform (feature domain) | 16 kHz | 1×N |
| HuBERT features | 50 fps | `[T, C]`, C=256/768 |
| VITS frames | 100 fps | `[T, 192]` semantics (after ×2) |
| pitch bin / f0 | 100 fps | `[T]` int32 / `[T]` float |

---

## 2. Model representation (GGUF)

### 2.1 Dimension conventions

GGUF stores dimensions in ggml's `ne` order, where `ne[0]` equals the last dimension of the PyTorch shape. No transposition is performed at export:

| PyTorch shape | ggml `ne` |
|---|---|
| Linear `[N, K]` | `[K, N]` |
| Conv1d `[O, I, K]` | `[K, I, O]` |
| ConvTranspose1d `[I, O, K]` | `[K, O, I]` |
| Conv2d `[O, I, KH, KW]` | `[KW, KH, I, O]` |
| ConvTranspose2d `[I, O, KH, KW]` | `[KW, KH, O, I]` |

Given row-major ordering: a contiguous tensor `[C, T]` (PyTorch) and ggml's `[T, C]` share the same memory layout, so both sides can reshape the raw bytes directly for comparison.

### 2.2 Convolution weight flattening (quantization prerequisite)

ggml quantized tensors require `ne[0]` to be an integer multiple of the block size (32), while the `ne[0]` of Conv1d/Conv2d weights is the kernel width (3/7/10/11/16…), so convolution weights cannot be quantized directly; at the same time, quantized matrix multiplication only allows the quantized weight to be in the first operand `src0`, whereas native convolution places the weight in the second operand. The solution is to flatten convolution weights into 2D matrices at export time:

```
Conv1d [O, I, K]       → (O, I*K)        row order: i slow, k fast
Conv2d [O, I, KH, KW]  → (O, I*KH*KW)    row order: i slow, (kh, kw) fast
```

This row order matches the column-vector index order produced by im2col; flattening requires no data rearrangement. At runtime, convolution is instead implemented as `im2col + matrix multiply` (see 3.2). ConvTranspose weights are not flattened (the native transposed-convolution operator only accepts F32/F16 weights). A metadata boolean key marks this layout; models with the old layout take the native convolution path.

### 2.3 Mixed quantization rules

| Category | Handling |
|---|---|
| Large weights whose flattened row width % 32 == 0 (convolution, attention/FFN linear) | quantized by quant level (q8_0 / q5_1 / q4_1 / q4_0 etc.) |
| bias, LayerNorm/BatchNorm/GroupNorm parameters, embedding, GRU, `dec.m_source.l_linear`, flow coupling layers, generator upsampling/excitation noise convolutions | kept F32 |
| Tensors that are quantizable but whose row width does not satisfy the requirement or that fail quantization | fall back to F16 |

Metadata includes at minimum: network version (v1/v2), sample rate, whether f0 is used, speaker count, `inter_channels/hidden_channels/filter_channels/n_heads/n_layers`, upsampling rates and kernels, residual block kernels and dilations, `gin_channels`, `spk_embed_dim`, convolution flattening flag; RMVPE additionally includes `n_fft/hop_length/win_length/n_mels/n_classes/hidden_dim/feature_dim/mel` boundaries and clamp, and the mel filterbank tensor.

---

## 3. Graph execution framework

### 3.1 Static graph container

Each subnetwork (HuBERT / RMVPE / VITS) runs in three steps: build → allocate → execute:

```
graph_runner:
    ctx  = init_context()                     # metadata context
    build(ctx)                                # build graph, expand output nodes
    allocate()                                # graph-level memory planning (allocate input/intermediate/output buffers as a whole)
    set_input(name, host_data) / compute() / get_output(name, host_data)
```

Key points:

- Execution uses **single-backend direct computation** (no cross-backend scheduler), so every operator that appears must be supported by the target backend;
- Graphs are cached by input shape: the same length is built only once; a length change triggers a rebuild;
- Constant inputs (relative position embeddings, permutation matrices, kernel placeholder tensors, etc.) are uploaded once after graph allocation;
- Host-side outputs are uniformly F32 row-major arrays, for easy comparison with any reference implementation.

### 3.2 Tensor layout

| Region | Layout | Reason |
|---|---|---|
| enc_p / flow (linear, attention, coupling) | `[C, T]`, C at `ne0` | matches the natural input of `mul_mat(W, x)`; normalization/slicing along `ne0` |
| dec / audio frontend convolution | `[T, C]`, C at `ne1` | matches the natural input of im2col, upsampling occurs in the T dimension |
| Swap points | enc_p→dec, dec→output | each does one `permute + cont` |

---

## 4. Operator mapping and pseudocode

### 4.1 Linear layers / embeddings

```
linear(x, W, b):                      # x: [K, T], W: [K, N]
    y = mul_mat(W, x)                 # [N, T] (i.e. Wᵀx)
    if b: y = y + broadcast(b)        # b: [N] → [N, 1]
    return y

get_rows(table, idx):                 # integer-index table lookup (emb_pitch)
    return table[idx]                 # [C, T]
```

### 4.2 Conv1d: flattened path

```
conv1d_flat(x[T, I, B], w[K*I, O], bias[O], stride, pad, dilation):
    kernel = tensor(K, I, 1)                                     # provides shape only; im2col does not read its data
    cols   = im2col(kernel, x, stride, pad, dilation,
                    dst_type = (w.type == F16 ? F16 : F32))      # [K*I, OW, B]
    out    = mul_mat(w, reshape2d(cols, K*I, OW*B))              # [O, OW*B]
    out    = permute_cont(reshape3d(out, O, OW, B), (1,0,2,3))   # [OW, O, B]
    if bias: out += reshape(bias, (1, O, 1))
    return out
```

- `permute` argument semantics: source dim i goes to target axis aᵢ; `[O,OW,B]→[OW,O,B]` is `(1,0,2,3)`;
- Column vector type: F16 weights→F16 columns; quantized or F32 weights→F32 columns (the CPU backend requires `src1` to be F32 on the quantized `src0` path).

### 4.3 Conv2d: flattened path

```
conv2d_flat(x[W, H, IC, N], w[KW*KH*IC, OC], bias, stride, pad):
    kernel = tensor(KW, KH, IC, 1)
    cols   = im2col(kernel, x, stride, pad, 1, 1, is_2d=true, dst_type=…)  # [KW*KH*IC, OW, OH, N]
    out    = mul_mat(w, reshape2d(cols, KW*KH*IC, OW*OH*N))                # [OC, OW*OH*N]
    out    = reshape4d(out, OC, OW, OH, N)
    out    = permute_cont(out, (2,0,1,3))                                  # [OW, OH, OC, N]
    if bias: out += reshape(bias, (1,1,OC,1))                              # channel dim is ne2
    return out
```

### 4.4 Grouped convolution (HuBERT position conv, groups=16)

```
grouped_conv(x[C, T], w, g):
    for i in 0..g-1:
        xi = cont(slice_channels(x, i*C/g, (i+1)*C/g))
        wi = cont(slice_channels(w, ...))
        yi = conv2d(xi, wi, …)          # degenerates to 2D conv with height=1
    return concat(yi)
```

### 4.5 Transposed convolution

```
conv_transpose1d_pytorch(x, w, b, stride, k):
    raw     = conv_transpose_1d(w, x, stride, p0=0)     # output has pad=(k-stride)/2 extra on each side
    trimmed = view_cont(raw, len = raw.len - 2*pad, offset = pad)
    return trimmed + b

conv_transpose2d_2x(x, w):                              # k=3, s=2, p=1, output_padding=1
    full = conv_transpose_2d_p0(w, x, stride=2)         # output 2*in+1
    return cont(view(full, size = 2*in, offset = 1))    # take full[1 : 1+2*in]
```

### 4.6 Normalization and activation

```
layer_norm(x[C,T]):       # along ne0
    (x - mean_C) / sqrt(var_C + eps) * γ + β

group_norm_hubert(x):     # per-channel group normalization (groups=C)
    reshape x → [T, 1, C, 1], run group_norm along the channel dim (ne2)

batch_norm2d_eval(x):     # RMVPE
    (x - running_mean) / sqrt(running_var + eps) * γ + β     # parameters broadcast to ne2

gelu(x)      = 0.5x(1+erf(x/√2))        # exact erf version
leaky_relu   = max(x, 0) + slope*min(x, 0)   # slope: 0.1 normal, 0.01 at generator end
```

### 4.7 Multi-head attention (per-head small matrix multiplies)

```
function mha(x[C, T], Wq, Wk, Wv, Wo, H):
    q = linear(x, Wq); k = linear(x, Wk); v = linear(x, Wv)     # [C, T]
    D = C / H
    for h in 0..H-1:
        qh = cont(view(q, D, T, col_offset = h*D))
        kh = cont(view(k, …)); vh = cont(view(v, …))
        s  = mul_mat(kh, qh) * (1/sqrt(D))                       # [T, T], softmax along ne0 (key dim)
        p  = softmax(s)
        oh = mul_mat(transpose(vh), p)                           # [D, T]
    y = linear(concat(oh), Wo)
```

HuBERT uses **Post-Norm** Transformer layers (the order is not commutative):

```
residual = x
x = mha(x) + residual
x = layer_norm1(x)
residual = x
x = fc2(gelu(fc1(x))) + residual
x = layer_norm2(x)
```

enc_p uses **relative position attention** (window 10):

```
Relative position embeddings are expanded on the host side from the weights into two constant matrices: rel_k [D, 2T-1] and rel_v [2T-1, D].
s = qᵀk / sqrt(D) + relative_to_absolute( matmul(emb_rel_k, q) )     # [T, T]
p = softmax(s)
ctx = matmul(emb_rel_v, absolute_to_relative(p))                     # [D, T]
```

### 4.8 Flow inverse transform

Order (inverse inference direction): `Flip, flow6, Flip, flow4, Flip, flow2, Flip, flow0`. Each flow group consists of 3 WaveNet layers (kernel 5) performing inverse affine coupling:

```
x0, x1 = split(x)
h = WN(x0, cond)
x1 = (x1 - h_pos) * exp(-h_log)      # or additive coupling: x1 = x1 - h
```

Channel reversal is implemented as a single matrix multiply of the features with a **192×192 permutation matrix**, replacing per-channel slice-and-concatenate.

### 4.9 RMVPE (f0)

**mel frontend (host side)**: real FFT → |X|² → mel filterbank (carried by the model) → log → clamp.

**DeepUnet (feature network)**: 5-level encoder (each level 4 ConvBlockRes + 2×2 average pooling) → 4 intermediate levels → 5-level decoder (transposed-conv upsampling + skip concatenation + 4 ConvBlockRes) → output layer 3×3 convolution → 384 dims. ConvBlockRes is `conv3x3+BN+ReLU` twice, with a 1×1 shortcut for the residual when channels change.

**Whole-segment vs windowed**: whole-segment inference builds the graph by padded frame count; for very long input it falls back to windowed inference — window valid frames 4096, context 2048 on each side (graph fixed at 8192 frames), and concatenation takes only the valid region. Under Vulkan the whole-segment limit is 16352 frames (derivation in §6.2).

**Bidirectional GRU (512-step static unroll + state relay)**:

```
function gru_chunk(x[384, 512], keep[512], h0[256]):
    xp = linear(x, W_ih) + b_ih                  # [768, 512]
    h = h0
    for t in 0..511:
        g = xp[:, t] + linear(h, W_hh) + b_hh    # [768]
        r, z, n = split(g, [256, 256, 256])      # gate order r/z/n matches PyTorch
        n = tanh(n)
        h = (1 - z) * n + z * h
        seq[t] = h
    return seq, h
```

- The forward pass is chunked at 512 frames, with the hidden state at the end of a chunk passed to the next; the reverse pass chunks from the end backwards (unrolled in reverse order within a chunk), also relaying state;
- `keep` marks valid frames (padding frames do not participate in state updates).

**Salience head**: `[fwd; rev]` (512 dims) → Linear(360) → Sigmoid.

**Decoding**:

```
idx(t) = argmax_c salience[c, t]
cents(t) = Σ_{k=-4..4} w_k * (20*(idx(t)+k) + 1997.3794084376191)   # local averaging
f0(t) = 10 * 2^(cents(t)/1200)
if salience[idx(t), t] < 0.03: f0(t) = 0                            # unvoiced
```

### 4.10 Retrieval (FAISS IVF equivalent semantics)

```
function ivf_blend(feat, index, index_rate, k=8):
    lists = nprobe nearest inverted lists by L2² centroid distance        # nprobe stored with the index
    cands = concatenate vectors from each list
    topk  = k candidates with smallest L2² distance
    w     = (1 / score)², then normalized
    mix   = Σ w_i * vec_i
    return index_rate * mix + (1 - index_rate) * feat
```

The index is exported as four tensors: centroids, all vectors, per-list offsets, per-list lengths; blending happens before ×2 nearest-neighbor upsampling.

### 4.11 Generator (HiFiGAN NSF) and SineGen

```
dec(z[C, T], g, source):
    x = permute_cont(z) → [T, C]
    x = conv1d(x, dec.conv_pre) + broadcast(dec.cond(g))     # conditional term
    for i in 0..n_ups-1:
        x = leaky_relu(x)
        x = conv_transpose1d_pytorch(x, dec.ups[i])
        if f0: x += noise_conv_i(source)                     # see below
        x = (Σ_j resblock_j(x)) / n_resblock                 # 3 residual blocks per level, kernel/dilation {k,1},{k,3},{k,5}
    x = leaky_relu(x, 0.01)
    x = conv1d(x, dec.conv_post)                             # 1 channel
    return tanh(x)
```

- The excitation noise convolution stride after each upsampling level = product of the remaining upsampling rates (1×1 special case: kernel 1, stride 1), kernel length is `2*stride` (1 at the last level);
- Within a residual block: `convs1` has dilation (pad=(k*d-d)/2), `convs2` has kernel 3 without dilation.

SineGen (host side, per sample):

```
rad(t)   = (f0(t)/sr) mod 1
cum(t)   = cumsum(rad)
phase(i) = cumsum_shift(linear interpolation(cum*upp, align_corners=True))      # phase wrap compensation
voiced   = f0 > 0
noise_amp = voiced ? 0.003 : 0.1/3
out(i)   = voiced ? [sin(2π*phase(i)) * 0.1 + noise_amp*noise(i)]
                  : [noise_amp * noise(i)]
```

where `upp = sr / 100`, `noise` can be generated from a fixed seed or injected externally; the output then passes through `tanh(l_linear(·)+b)` as the excitation source.

---

## 5. Full pipeline

### 5.1 Preprocessing

1. **Downmix + resample**: any input → mono 16 kHz, using polyphase FIR (rational-ratio resampling).
2. **48 Hz zero-phase high-pass**: second-order IIR, forward/reverse filtering once each (filtfilt), odd extension at boundaries, `padlen = 3*max(len(a),len(b))`.
3. **Whole-segment f0**: run RMVPE on the reflection-padded audio; multiply f0 by the pitch-shift factor `2^(pitch/12)`, and quantize to 1..255 as
   `pitch_bin = 1 + round(12*log2(f0/50))` (0 for unvoiced).

### 5.2 Chunking and concatenation (consistent with the Python reference)

```
function split_points(audio, x_query, x_center, x_max):      # unit: seconds
    if len(audio)/16000 <= x_max: return []
    curve = Σ_{i=0..159} |audio_pad[t+i]|                    # sliding 160-point absolute sum, reflection pad 80
    for t = x_center*16000 step x_center*16000:
        lo = max(0, t - x_query*16000); hi = min(len, t + x_query*16000)
        emit(t - x_query*16000 + argmin(curve[lo:hi]))
```

Concatenation rules:

```
pad = x_pad*16000
for t in split_points:
    t = align_to_10ms(t)
    chunk = padded[s : min(t + 2*pad + 160, len)]
    f0_slice = f0[s/160 : (t + 2*pad)/160]
    emit(vc(chunk, f0_slice))
    s = t
emit(vc(padded[s:], f0[s/160:]))
```

### 5.3 Single-chunk conversion (vc)

```
T_tok, C = hubert(audio_chunk)                # 50 fps
feats0   = feats                              # keep a copy when protect < 0.5
if index: feats = ivf_blend(feats, index_rate)
feats = upsample2_nearest(feats)              # 100 fps (×2, nearest neighbor)
p_len = min(len(audio_chunk)/160, feats.frames, len(f0_slice))

for t in 0..p_len-1:                          # protect (unvoiced protection)
    pf = (f0[t] >= 1 Hz) ? 1.0 : protect
    feats[t] = pf * feats[t] + (1-pf) * feats0[t]

source = tanh(l_linear * sinegen(f0_slice) + b)
audio  = vits(feats, pitch_bin_slice, sid, source, noise)
trim   = x_pad * target_sample_rate
return audio[trim : len-trim]
```

### 5.4 Post-processing

```
# RMS loudness mixing
rms_src = 160-point frame RMS(original 16k), rms_conv = 160-point frame RMS(converted result, target sample rate)
factor(i) = rms_src(i)^(1-r) * rms_conv(i)^(r-1)
out *= factor (linear interpolation between frames down to per-sample)

# peak normalization and int16
peak  = max|out|
scale = 32768; if peak/0.99 > 1: scale /= peak/0.99
pcm   = int16(out * scale)
```

---

## 6. Backend constraints and adaptations

### 6.1 Hard constraints of ggml quantization

1. Quantized tensor `ne[0]` must be an integer multiple of 32;
2. In `mul_mat(A, B)`, quantization may only appear in `A` (`src0`);
3. The CPU backend requires `src1` to be F32 when `src0` is quantized;
4. The Vulkan backend does not support quantized `B` (asserts when `src0` is not F16/F32 and `src1` is a quantized type), so quantized weights must be in `src0`.

### 6.2 Vulkan grid limits and N-dimension splitting of matmul

Vulkan's compute dispatch is limited per dimension by `maxComputeWorkGroupCount` (minimum guaranteed 65535 per dimension). Matrix multiplication
generates Y-dimension workgroups as `ceil(N / wg_denoms[1])`; the relevant paths in this project use `wg_denoms[1] = 32`, so `N > 65535×32`
triggers a backend assertion. Upstream ggml only splits oversized **M** (limited by `maxStorageBufferRange`) and does not split N; the implementation adds N splitting:

```
function mul_mat(dst, a, b):
    max_n = device.maxComputeWorkGroupCount[1] * 32        # 32 = smallest wg_denoms[1] among all matmul pipelines
    if (dst.ne2 == 1 and dst.ne3 == 1 and dst.N > max_n):
        for n0 in 0 .. dst.N step max_n:
            cn   = min(max_n, dst.N - n0)
            dst' = view(dst, N=cn, offset += n0 * nb1)
            b'   = view(b,   N=cn, offset += n0 * nb1)
            matmul_shader(a, b', dst', disable_split_k = true)
        return
    ...original path (including row-wise splitting for oversized M)
```

Key points: the block stride uses the smallest `wg_denoms[1]` so that no pipeline (s/m/l, F16/F32/quantized) exceeds the limit; split-k is disabled within a block to prevent its reduction dispatch element count `m×n×batch` from exceeding the limit again; only effective when batch=1.

Two implementation thresholds are derived from this constraint:

| Stage | Composition of N | Vulkan safe limit |
|---|---|---|
| VITS generator last-stage convolution | `VITS frames × hop` (hop=400/480) | at hop=480, single chunk ≤ 4369 frames; at hop=400 ≤ 5242 frames |
| RMVPE whole-segment convolution | `128 × frames` | frames ≤ 16352 (about 163.5 s); beyond this use the windowed path |

### 6.3 Vulkan operator support notes

| Operator | Constraint |
|---|---|
| `IM2COL` | Supports F32/F16 output; works with `mul_mat` to implement flattened convolution |
| `CONV_TRANSPOSE_1D` | F32 weights and output only (generator upsampling forced to F32) |
| `CONV_2D / CONV_TRANSPOSE_2D` | weights/input/output must be contiguous, channel counts must match; used only by the old-layout path |
| `CONCAT` | both sides same type, contiguous (`cont` first after slicing) |
| `MUL_MAT` | quantized `src0` supports q4_0/q4_1/q5_1/q8_0; when N is too large, uses §6.2 splitting |

Two acceleration paths are disabled by default to avoid driver issues: integer dot product (Intel drivers may output NaN/zero after multiple graph executions with quantization) and fp16 compute (error amplification; audio SNR drops from about 60 dB to about 33 dB). Both can be re-enabled via environment variables.

### 6.4 Other adaptation points

- `concat/repeat` and others are sensitive to contiguity: make contiguous before any slice/view participates;
- Shaders need SPIR-V compilation on first run (several seconds); startup is faster after driver caching;
- Weights reside in VRAM; graph buffers are allocated by graph size; larger chunks use more VRAM.

---

## 7. Numerics and determinism

| Item | Description |
|---|---|
| Precision | float32 throughout; im2col inside some native convolutions is fp16, error on the order of 1e-3 |
| f0 sensitivity | when salience top1/top2 are close, argmax may flip (<0.1% of frames); a single frame error causes a SineGen phase offset and must be suppressed by frame-level correction/median filtering |
| Random noise | fixed seed by default (chunk index participates in derivation) for reproducibility; external noise can be injected for per-sample comparison |
| Resampling | polyphase FIR is quality-equivalent to soxr, not bit-identical; unaffected when output resampling is not performed by default |
| Retrieval distance | may differ in the last bits from FAISS SIMD implementations; blend weights agree |
| Alignment baseline | `mul_mat(A,B) = AᵀB`, `softmax/norm` along `ne0`, `group_norm` channel dim is `ne2`, GELU uses the exact erf version — these semantics determine axis choice and ordering during porting |
