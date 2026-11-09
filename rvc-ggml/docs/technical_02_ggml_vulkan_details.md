# RVC-GGML Tech Doc 2: ggml / Vulkan Implementation Details

> Audience: readers familiar with ggml and Vulkan.
> This document focuses on **how the parts that differ from the original are actually implemented**: model layout, graph construction techniques, the quantization scheme, Vulkan adaptation and pitfalls.
> Pseudocode is given only where needed; no C++ source is reproduced.
>
> **Important**: This project modifies the local ggml source (Vulkan matmul N-dimension splitting, see §5.5).
> The patch is in `patches/ggml-vulkan-matmul-n-split.patch`; it must be reapplied after upgrading ggml.

---

## 1. Model Format and Layout Conventions

### 1.1 Basic conventions

In GGUF, the dims order of each tensor is exactly ggml's `ne` order (`ne[0]` is the last dimension in PyTorch):

| PyTorch | ggml ne |
|---|---|
| Linear `[N, K]` | `[K, N]` |
| Conv1d `[O, I, K]` | `[K, I, O]` |
| ConvTranspose1d `[I, O, K]` | `[K, O, I]` |
| Conv2d `[O, I, KH, KW]` | `[KW, KH, I, O]` |
| ConvTranspose2d `[I, O, KH, KW]` | `[KW, KH, O, I]` |

On export, **no transposition is performed**; `ggml_mul_mat(W, x)` is `Wᵀx`, exactly PyTorch's `Linear(x)` semantics.
A contiguous tensor `[C, T]` has the same memory layout in ggml as PyTorch's row-major `[T, C]`, so reference data can be reshaped and compared directly.

### 1.2 Layout v2: flattened convolution weights (`rvc.conv_flat=true`)

**Problem**: ggml quantized tensors require `ne[0] % 32 == 0` (row-wise blocking), while the `ne[0]` of Conv1d/Conv2d weights is the
kernel width (3/7/10/11/16…), so convolution weights cannot be quantized. The old scheme could only fall back to F32 (larger than fp16),
and ggml's `ggml_conv_*` combination puts the weight as the **second operand** of `mul_mat`, whereas ggml's quantized matmul only supports
a quantized weight as the **first operand**.

**Solution**: at export time, flatten convolution weights into 2-D matrices with "row width = input channels × kernel area"; at runtime, stop using `ggml_conv_*`
and use `im2col + mul_mat(quantized weight, cols)` instead:

```
Conv1d [O, I, K]      -> stored as (O, I*K)        # row width I*K, quantizable when a multiple of 32
Conv2d [O, I, KH, KW] -> stored as (O, I*KH*KW)    # row order (i, kh, kw), kw fastest — matches im2col column order
ConvTranspose1d/2d    -> not flattened (native operators only support F32/F16 weights)
```

Memory-order correspondence: after row-major flattening, PyTorch naturally satisfies `i slow, k fast` (Conv1d) and `i slow, (kh,kw) fast` (Conv2d),
matching the index order used by ggml im2col when generating column vectors, so flattening requires no data rearrangement.

Model metadata gains a boolean key `rvc.conv_flat`; old models lack this key → they take the old path (automatic compatibility).

---

## 2. Graph Execution Framework

### 2.1 GgmlGraphRunner

A generic "build → allocate → execute" container:

- `ggml_init` (a standalone metadata context, 16 MiB class by default) + `ggml_new_graph_custom`
- during construction, `expand(node)` expands the compute graph
- `allocate()`: `ggml_gallocr_new/reserve/alloc_graph`, records graph buffer size and node count
- `compute()`: single-backend `ggml_backend_graph_compute` (no scheduler, so **every operator must be supported by the target backend**,
  otherwise it errors out — which is also why an exact knowledge of the Vulkan support matrix is required)
- inputs/outputs are read and written via `ggml_backend_tensor_set/get`; constant inputs (relative position embeddings, permutation matrices, etc.)
  are uploaded once after graph allocation

**Graph caching strategy**:
- HuBERT: one graph cached per number of input samples;
- VITS: cached by `frames/feature_dim/sid`;
- RMVPE: the feature graph is cached by frame count; the GRU/head graphs are fixed at 512 frames and kept resident;
- pipeline long audio has a different chunk length per chunk → rebuilt per chunk (same as Python); `--x-*` can be used to tune chunk length to control VRAM.

**Measured buffer sizes** (5 s audio): HuBERT ≈ 63 MiB; RMVPE feature ≈ 53 MiB (512-frame window);
VITS ≈ 305 MiB (fp16, 698-frame input for the end-to-end pipeline); at quantized levels, since im2col columns use F32, VITS ≈ 492 MiB.

### 2.2 Two tensor layouts

| Region | Layout | Reason |
|---|---|---|
| enc_p / flow (linear, attention, coupling layers) | `[C, T]` (C in ne0) | consistent with the natural input of `mul_mat(W, x)`; relative positions/masks are all sliced along the C dimension |
| dec (HiFiGAN) and audio front-end convolutions | `[T, C]` | consistent with the natural input of `ggml_conv_*`; progressive upsampling is along T |
| exchange points | enc_p output → dec input, dec output | one `permute+cont` |

---

## 3. Operator Mapping and Implementation Techniques

### 3.1 Linear layers

```
linear(x, W, b): y = mul_mat(W, x)        # W is src0, may be a quantized tensor directly
                 y = y + broadcast(b)      # bias stays F32/F16
```

### 3.2 Convolutions: native path and flattened path

```
conv1d_native(w3d, x):                    # old layout
    return ggml_conv_1d(w3d, x, stride, pad, dil)

conv1d_flat(w2d, x, stride, pad, dil):    # new layout (rvc.conv_flat)
    I = x->ne[1]; O = w2d->ne[1]; K = w2d->ne[0] / I
    kernel = new_tensor_3d(F32, K, I, 1)   # shape placeholder: im2col reads only the shape, not the data
    cols   = im2col(kernel, x, stride, pad, dil, dst_type = w2d is F16 ? F16 : F32)
    out    = mul_mat(w2d, reshape_2d(cols, K*I, OW*B))   # [O, OW*B]
    out    = permute_cont(reshape_3d(out, O, OW, B), 1,0,2,3)  # [OW, O, B]
    return out + bias
```

Key points and pitfalls:

1. **The src0 of `ggml_im2col` (the kernel parameter tensor) is only used to infer shapes**; neither the CPU nor the Vulkan implementation reads its data;
   a tiny placeholder tensor therefore suffices (marked as input, left to gallocr for allocation). This is also the key that makes the flattening scheme viable.
2. **Column vector type**: if the weight is F16 → columns use F16 (saves memory); if the weight is quantized/F32 → columns must use **F32**,
   because ggml-cpu's "src0 quantized" path requires `src1` to be F32 (internally converted to the Q8_0-family vec_dot type).
3. **The output must be converted back to convolution order**: the quantized matmul output is `[O, ...]`, while the convolution ecosystem expects `[OW, O, B]`.
   Note that the argument semantics of `ggml_permute(x, a0,a1,a2,a3)` are "source dimension i goes to target axis aᵢ":
   `[O,OW,B]→[OW,O,B]` is `(1,0,2,3)`; `[OC,OW,OH,N]→[OW,OH,OC,N]` is `(2,0,1,3)`.
4. **Grouped convolution** (HuBERT pos_conv, groups=16): ggml's conv does not support groups.
   The approach is to slice the input/weight into views per group (after slicing, `cont`, since im2col needs contiguous memory), run `ggml_conv_2d` per group (a degenerate case with height dimension = 1),
   and finally `concat` along the channel dimension. That weight stays F32/F16 and does not go through quantization.
5. **ConvTranspose1d** (generator upsampling; Vulkan only supports F32 weights): `ggml_conv_transpose_1d(p0=0)` outputs an extra
   `pad=(k-s)/2` on each side, trimmed with `view_2d + cont`.
6. **ConvTranspose2d** (RMVPE decoder): `ggml_conv_transpose_2d_p0` outputs `2W+1`;
   its relation to PyTorch `(k=3,s=2,p=1,op=1)` is to take `full[1 : 1+2W]`, trimmed via a `view_4d` offset.
7. **Conv2d flattened path**: the output `[OC, OW, OH, N]` goes through `permute_cont(2,0,1,3)` to become `[OW, OH, OC, N]`;
   batch-norm's channel dimension is ne[2], consistent with native conv output.

### 3.3 Normalization and activations

- `LayerNorm`: `ggml_norm` (along ne0) → `×γ + β`;
- `GroupNorm`: the channel dimension of `ggml_group_norm` is ne[2]; HuBERT's front-end per-channel normalization is implemented with `reshape [T,1,C,1]`;
- `BatchNorm2d` (RMVPE, eval mode): `(x - mean) / sqrt(var+eps) * γ + β`, parameters reshaped to `[1,1,C,1]` for broadcasting;
- GELU uses the exact erf version `ggml_gelu_erf` (fairseq `approximate='none'`);
- the LeakyReLU slope inside the generator is 0.1, and **the last layer is 0.01** (the PyTorch default) — an easy pitfall.

### 3.4 Attention (HuBERT / enc_p)

Multi-head attention does not use `ggml_flash_attn`; instead it uses "per-head small matmuls":

```
for h in heads:
    q_h,k_h,v_h = view2d(each head, D channels, T) + cont      # [D, T]
    score = mul_mat(k_h, q_h) * scale                   # [Tk, Tq], softmax along ne0=Tk
    out_h = mul_mat(transpose(v_h), softmax(score))     # [D, Tq]
concat(out_h)  # head order matches hidden indices
```

- HuBERT: absolute-position attention (Post-Norm structure; note the placement of residual and LayerNorm);
- RVC enc_p: absolute scores are added to "relative position logits"; the latter are aligned via the following two transforms:
  - `relative_to_absolute`: relative logits `[2T-1, T]` go through pad/reshape/pad to produce `[T, T]` (details omitted in pseudocode);
  - `absolute_to_relative`: the inverse transform of attention weights;
  - the relative position embeddings are expanded host-side from the exported weights into two constant matrices, `[D, 2T-1]` / `[2T-1, D]` (distinguishing transpose order).

### 3.5 Channel flip in Flow

In `flow`, each coupling layer is preceded by a channel-order flip (swapping odd/even indices). The naive approach is 192 slice-concatenations;
here a **192×192 permutation matrix** multiplies the features (`mul_mat`), completing it in a single matmul at negligible cost.

### 3.6 RMVPE static graph and state handoff

- **feature subgraph**: DeepUnet performs multi-level encoding/decoding on mel. For windowed inference, the graph is built for
  `valid frames 4096 + context 2×2048 = 8192 frames`; for **≤ 16352 frames (about 163.5 seconds)**, the whole segment is inferred in one pass,
  strictly matching Python's full-mel behavior.
  This limit is derived backwards from the Vulkan matmul N-dimension constraint (`N = 128×frames ≤ maxComputeWorkGroupCount[1]×32`, see §5.5);
  before the fix it was set to 65536 frames, and in practice the assertion at `ggml-vulkan.cpp:8617` was guaranteed to trigger beyond 163.8 seconds;
  now over-limit input automatically takes the windowed path (window boundaries differ very slightly from Python; perceptually consistent).
- **GRU subgraph**: PyTorch's `nn.GRU` (bidirectional) recurses step by step over time and cannot be turned directly into a static graph.
  The implementation **statically unrolls 512 time steps**, computing the gates at each step from the previous hidden state:

```
h = hidden0
for t in 0..511:
    gates = linear(x[t]; W_ih, b_ih) + linear(h; W_hh, b_hh)
    r, z, n = split(gates)                      # gate order matches PyTorch
    n = tanh(n)                                 # note the order: compute n before mixing
    h = (1 - z) * n + z * h
    seq[t] = h
```

For chunked inference, "state handoff": at the start of a chunk, apply the same "keep mask" as the reference implementation (clear on the first chunk, carry the previous chunk's hidden state on subsequent chunks);
at the end of a chunk, read out the final hidden state and use it as the next chunk's initial hidden state. Forward and backward directions are handled separately and concatenated at the end before being sent to the 360-class head.

### 3.7 Retrieval (equivalent semantics of FAISS IVF)

```
List selection: centroid distance L2², take the nearest nprobe (nprobe read from index metadata; must match the training index)
Within a list: full L2² over vectors → global top-k (k=8)
Weighting:      w_i = (1/score_i)² / Σ(1/score_j)²
Mixing:         feats = index_rate * Σ w_i * vec_i + (1-index_rate) * feats
```

Mixing happens before ×2 nearest-neighbor interpolation (same as Python). The original index is exported by the conversion script as
four F32 tensors `index.centroids / index.vectors / index.list_offsets / index.list_lengths` + metadata.

---

## 4. Quantization Implementation (the core difference in this project)

### 4.1 Four hard constraints of ggml quantization

1. The `ne[0]` of a quantized tensor must be an integer multiple of the block size (32);
2. In `ggml_mul_mat(A, B)`, quantization may only appear in **A (src0)**;
3. The CPU backend requires src1 to be **F32** when src0 is quantized (internally converted to the vec_dot type);
4. The Vulkan backend hits
   `GGML_ASSERT(y_non_contig || !qy_needs_dequant)` when src0 is not F16/F32 and src1 is a quantized type
   (i.e. "quantized B" is unsupported),
   so quantized weights must go in src0 and activation columns use F32/F16.

### 4.2 Flattened convolutions (satisfying constraints 1+2 simultaneously)

The algorithm is in the pseudocode in 3.2. The accompanying export rules:

```
Quantizability test: after flattening, row width % 32 == 0 (and the exporter itself can quantize that value range)
Sensitive tensors: bias / norm / embedding / GRU / 1D parameters etc. -> F32
Quantizable but failing tensors -> F16 (no longer fall back to F32, avoiding "larger after quantization")
ConvTranspose weights -> not flattened, F32 (required by Vulkan native operators)
```

### 4.3 Mixed quantization strategy (per model)

| Model | Kept F32 | Quantized |
|---|---|---|
| HuBERT | LayerNorm / final_proj / pos_conv / GroupNorm parameters / all 1D parameters | front-end convolutions (flattened) + attention and FFN linear layers |
| RMVPE | BatchNorm/GRU/salience head/1x1 shortcut/ConvTranspose/1D parameters | 3×3 convolutions at each DeepUnet level (flattened; layers whose row width does not satisfy the constraint, e.g. 16-channel layers, fall back to F16) |
| Character model | bias, `emb_g/emb_pitch/emb_phone`, `dec.m_source.l_linear`, `flow.*`, `dec.ups.*`, `dec.noise_convs.*` | enc_p convolutions (flattened), dec residual blocks/pre-processing convolutions (flattened) |

Model sizes for each level are given in the usage doc (`usage_release_vs_original.md`). Quantized inference is validated on both CPU and Vulkan (q8_0 end-to-end corr ≈ 0.9995).

---

## 5. Vulkan Adaptation and Known Pitfalls

### 5.1 Device requirements

- Instance level: **Vulkan 1.2** (ggml errors out directly when `vk::enumerateInstanceVersion() < 1.2`);
- Device level: must support `storageBuffer16BitAccess` (a Vulkan 1.1 core feature or the `VK_KHR_16bit_storage` extension);
- `VK_KHR_shader_float16_int8` is only required when the device enables the fp16 compute path; this project defaults to
  `GGML_VK_DISABLE_F16=1`, so that extension is not mandatory (but current official drivers basically all support it).

### 5.2 Two default mitigations (set automatically at program startup)

| Environment variable | Default | Reason |
|---|---|---|
| `GGML_VK_DISABLE_INTEGER_DOT_PRODUCT=1` | disable integer dot product | Intel drivers output NaN/all-zeros after multiple graph executions with quantized models (measured on Intel integrated GPUs; reproduced on ggml 0.18/0.23) |
| `GGML_VK_DISABLE_F16=1` | disable fp16 compute path | the Vulkan fp16 accumulation/dequantization path significantly amplifies error (measured audio SNR drops from ~60 dB to ~33 dB), and there is almost no speed difference on integrated GPUs |

Can be restored with `RVC_VK_ENABLE_INTDOT=1` / `RVC_VK_ENABLE_F16=1` (not recommended unless the driver is confirmed to be fine).

### 5.3 Vulkan support matrix relevant to this project (ggml 0.23)

| Operator/path | Vulkan support | Usage in this project |
|---|---|---|
| `MUL_MAT` (src0 quantized, src1 F32/F16) | ✅ including q4_0/q4_1/q5_1/q8_0; split by this project's patch when N is very large (§5.5) | linear layers, flattened convolutions |
| `IM2COL` (src1 F32 → dst F16/F32) | ✅ | convolution column extraction (placeholder kernel) |
| `CONV_TRANSPOSE_1D` | ⚠️ F32 weights/output only | generator upsampling (export forces F32) |
| `CONV_2D / CONV_TRANSPOSE_2D` | ⚠️ weights F32/F16, requires contiguous, `cin==b->ne[2]` | RMVPE (old layout) and ConvT2d |
| `GET_ROWS` (quantized table) | ✅ | emb_pitch (this project's strategy keeps it F32 to avoid lookup precision loss) |
| `CONCAT` | requires both sides same type/contiguous | grouped convolution, skip connections |
| Others (add/mul/norm/softmax/pool/permute/cont…) | ✅ (F32/F16) | used extensively in graphs |

### 5.4 Other adaptation points

- **Graph construction order**: operations like `concat`/`repeat` are sensitive to contiguity; call `cont` first after slicing;
- **Large tensors**: `ggml_mul_mat` shards very large **A (M dimension)** by rows (official implementation, `maxStorageBufferRange` limit);
  splitting very large **B (N dimension)** is this project's modification to ggml (see §5.5);
- **Shader compilation**: on first run each operator configuration requires compiling a SPIR-V shader, taking several seconds; the driver caches them,
  so subsequent starts are noticeably faster;
- **Memory**: weights stay resident in VRAM (hubert+rmvpe+character ≈ 440 MB fp16); graph buffers are on the order of 2.1;
  the larger the long-audio chunk, the more VRAM is used.

### 5.5 This project's modification to the ggml source: matmul N-dimension splitting (important)

**This is the only source modification in this project relative to official ggml**, involving the file
`src/ggml-vulkan/ggml-vulkan.cpp`; the patch is in `patches/ggml-vulkan-matmul-n-split.patch`.

**Problem**: ggml's Vulkan matmul `ggml_vk_matmul` only splits the batch/z dimension; **the N dimension is never split**.
At dispatch, `wg1 = ceil(N / wg_denoms[1])`, which must satisfy `wg1 ≤ maxComputeWorkGroupCount[1]` (Vulkan's minimum guarantee is 65535).
In this project's relevant path (flattened convolution) `wg_denoms[1] = 32`, so when `N > 65535×32 = 2,097,120` it triggers:

```
ggml_vulkan.cpp: GGML_ASSERT(wg0 <= ... maxComputeWorkGroupCount[0] &&
                            wg1 <= ... maxComputeWorkGroupCount[1] &&
                            wg2 <= ... maxComputeWorkGroupCount[2]) failed
```

Official ggml (including current master) only handles very large **M** (`needs_split`, `maxStorageBufferRange` limit) and has no N splitting.

**Trigger scenarios** (guaranteed crash before the fix):

| Scenario | Composition of N | Over-limit threshold |
|---|---|---|
| VITS generator final-stage convolution (48 kHz, hop=480) | `VITS frames × 480` | single chunk > 4369 frames (about 43.7 s) |
| Same as above (40 kHz, hop=400) | `VITS frames × 400` | single chunk > 5242 frames (about 52.4 s; close to the chunking limit, escaped by luck) |
| Flattened convolution in RMVPE whole-segment (non-windowed) inference | `128 × frames` | frames > 16384 (about 163.8 s) |

**Modification** (logic; see patch):

```
ggml_vk_mul_mat(dst, src0, src1):
    max_n = maxComputeWorkGroupCount[1] × 32          # 32 = the smallest wg_denoms[1] among all matmul pipelines
    if single batch (ne2==ne3==1) and dst.N > max_n:
        for n_offset, cur_n in split into max_n-column blocks(dst.N):
            dst_view  = view(dst,  ne1=cur_n, offset += n_offset × nb1)
            src1_view = view(src1, ne1=cur_n, offset += n_offset × nb1)
            per-block matmul (disable split_k)                    # avoid the split_k reduction grid m×n exceeding the limit again
        return
    ...original logic (the large-M split is factored out into ggml_vk_mul_mat_q_f16_m_split for reuse in both places)...
```

Key points:
1. Use 32 (the smallest `wg_denoms[1]`) as the step, ensuring that no matter which pipeline the backend picks (s/m/l, fp16/f32/quantized) the limit is not exceeded;
2. Within a block `disable_split_k=true`: the split_k reduction dispatch has `m×n×batch` elements, also constrained by the workgroup limit;
3. It only applies to `ne2==ne3==1` (batch=1), consistent with the official M-split restriction; other cases take the original path;
4. Pure dispatch partitioning with no change to the math; short audio/small models never trigger it (behavior matches official).

**Validation** (2026-09-17):
- With forced splitting (threshold temporarily lowered to 65535 columns so that even the 5 s sample goes through splitting), VITS v1/v2 self-test corr/SNR are **bit-identical** to the unsplit case;
- Short-sample regression: RMVPE 5 s, VITS v1/v2, e2e (Vulkan) all pass;
- GPU measurement: 48 kHz model + 204 s long audio completes full conversion (before the fix, the second chunk asserted).

**Rebuild notes**:
- After upgrading/replacing the ggml source the patch must be reapplied (under the FetchContent path the build system applies it automatically;
  with a local ggml source path CMake detects the patch marker and WARNINGs, requiring manual application);
- Using unpatched official ggml directly reproduces the crash on long audio;
- See `patches/README.md` for how to apply it.

---

## 6. Host-side DSP and Numerical Details

| Module | Implementation | Difference from the original |
|---|---|---|
| Resampling | polyphase FIR (rational ratio) | same quality class as librosa(soxr), not sample-by-sample |
| 48 Hz high-pass | hand-written filtfilt (odd/even extension + double filtering) | bit-identical to scipy after the fix (the earlier padlen/extension-order bug has been fixed) |
| mel front-end | pocketfft real FFT + filterbank table in GGUF | filterbank consistent with librosa htk=True |
| Silent split | sliding-window minimum of the energy curve | formula same as Python |
| SineGen | host-side per-sample sine generation + fixed-seed noise/injected noise | phase strictly tied to f0; a single-frame f0 difference causes phase drift |
| RMS mixing | frame energy envelope | same formula as Python |

---

## 7. Validation and Debugging System

- **Reference data**: Python scripts dump each intermediate layer during inference (four categories: hubert/rmvpe/vits/e2e);
- **Criteria**: correlation coefficient corr + SNR (audio additionally checks max_err); stage thresholds are in Tech Doc 1;
- **Determinism**: validation mode injects reference noise and reference f0 (`--noise-dir`, `--f0-inject`) to isolate randomness and
  RMVPE tie-breaking flips;
- **Debug environment variables**: `RVC_DEBUG_OP` (broadcast shapes), `RVC_DEBUG_CONV` (convolution shapes/layout),
  `RVC_DEBUG_LINEAR`, `RVC_HUBERT_DUMP` (dump intermediate tensors), `RVC_RMVPE_DEBUG` (subgraph nodes/buffers),
  `RVC_VK_ENABLE_INTDOT`, `RVC_VK_ENABLE_F16`.

---

## 8. Performance and Future Optimization Directions

- Reference magnitudes (Intel integrated GPU, 5 s audio): HuBERT ~0.2–1 s, RMVPE ~1.2 s (CPU)/~6.5 s (Vulkan),
  VITS ~1.7–2.2 s (Vulkan), full Vulkan pipeline about 4.5–6 s (quantized levels similar or slightly faster);
  entry-level discrete GPUs (Arc B570 class) are expected to be significantly faster than real time.
- **To optimize**:
  1. The im2col columns of quantized convolutions can switch to F16 on Vulkan (saves VRAM/bandwidth); CPU still uses F32;
  2. Tune the q4_0 mixed strategy by listening (keep the generator in F16, etc.);
  3. Long-audio VRAM: finer chunking or cross-chunk graph buffer reuse;
  4. Non-ASCII paths: convert command-line arguments to UTF-8 before handing them to the ggml file interface.
