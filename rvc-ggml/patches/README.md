# RVC-GGML modifications to the ggml source (local patch)

The ggml used by this project is **not the official upstream**: to keep **48 kHz character models**
and **very long audio** from crashing on Vulkan, one necessary change to the official ggml source
is required. After upgrading/replacing the ggml source you **must re-apply this patch**, or long
audio will trigger the `maxComputeWorkGroupCount` assertion again.

> When ggml is fetched automatically via FetchContent (see the dependency resolution in the root
> `CMakeLists.txt`), the build system applies this patch automatically, no manual step needed; the
> manual method below applies to the scenario of using a local ggml source tree.

## Patch file

- `ggml-vulkan-matmul-n-split.patch`
  - target file: `src/ggml-vulkan/ggml-vulkan.cpp` (function `ggml_vk_mul_mat`, adds `ggml_vk_mul_mat_q_f16_m_split`)
  - baseline: official ggml v0.23.0 (20349 lines in that file; 20382 after the patch, a net increase of 33 lines)

## How to apply

Run from the root of the ggml source tree (adjust the patch path to its actual location):

```powershell
# If you have git
git apply -p1 <path>\RVC-GGML\patches\ggml-vulkan-matmul-n-split.patch
# Or GNU patch
patch -p1 < <path>\RVC-GGML\patches\ggml-vulkan-matmul-n-split.patch
```

If a ggml version upgrade prevents the patch from applying directly, merge the changes manually as
described in Change details below (using the patch context as reference: insert the new function
before `ggml_vk_mul_mat`, and replace the inline M-split code in the original `needs_split` branch with the new call).

## Background

ggml's Vulkan matmul `ggml_vk_matmul` only splits the batch/z dimension and **never splits the N dimension**;
the number of workgroups in the shader's Y direction is limited by `maxComputeWorkGroupCount[1]` (Vulkan minimum guarantee 65535).
Matmul is dispatched by `ceil(N / wg_denoms[1])` (this project's relevant path has `wg_denoms[1] = 32`), so when

```
N > 65535 × 32 = 2,097,120
```

it triggers the following at `ggml_vulkan.cpp:8617`:

```
GGML_ASSERT(wg0 <= ... maxComputeWorkGroupCount[0] &&
            wg1 <= ... maxComputeWorkGroupCount[1] &&
            wg2 <= ... maxComputeWorkGroupCount[2]) failed
```

Official ggml (including current master) only handles oversized **M** (the `maxStorageBufferRange` limit) and has no N split.

## Trigger scenarios (guaranteed crash before the fix)

| Scenario | N composition | Overflow threshold |
|---|---|---|
| VITS generator final-stage convolution (48 kHz, hop=480) | `VITS frames × 480` | single chunk > 4369 frames (~43.7 s) |
| VITS generator final-stage convolution (40 kHz, hop=400) | `VITS frames × 400` | single chunk > 5242 frames (~52.4 s, close to the chunk limit, survived by luck) |
| Flattened convolution of RMVPE whole-segment (non-windowed) inference | `128 × frames` | frames > 16384 (~163.8 s) |

## Change details (`ggml_vk_mul_mat`)

1. N split added at the entry point: when `dst->ne[2]==1 && dst->ne[3]==1 && dst->ne[1] > maxWGC[1]*32`,
   split into chunks of `maxWGC[1]*32` columns; `dst` / `src1` views are built with `view_offs += n_offset * nb[1]` and computed chunk by chunk.
   (32 is the smallest `wg_denoms[1]` among all matmul pipelines; taking it guarantees no pipeline exceeds the limit.)
2. split_k is disabled inside each chunk (`disable_split_k=true`), to avoid the split_k reduction grid `m*n*batch` exceeding the limit again.
3. The existing inline M-split code is extracted into `ggml_vk_mul_mat_q_f16_m_split`, shared by the N-split chunks and the normal path.
4. Pure dispatch partitioning; no operator math is changed; it takes effect only when N exceeds the limit (short-audio / small-model behavior is completely unchanged).

## Validation (2026-09-17)

- With **forced splitting** (threshold temporarily lowered to 65535 columns), VITS v1/v2 self-test corr/SNR is **bit-identical** to the unsplit path; RMVPE 302 s passes
- Short-sample regression: RMVPE 5 s, VITS v1/v2 and e2e (Vulkan) all pass
- Measured on GPU: 48 kHz model + 204 s long audio full conversion passes (before the fix, the second chunk already tripped the assertion)

See Section 5 (5.5) of `docs/technical_02_ggml_vulkan_details.md` for detailed technical notes.
