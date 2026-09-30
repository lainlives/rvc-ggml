// f0.h —— M4-2 RMVPE f0 与 coarse bins（官方 81eed5e extract_f0 语义）
//
// 语义基准：81eed5e `train/dataset/extract_f0.py::FeatureInput`（行 91-123）：
//   f0 = RMVPE(assets/rmvpe/rmvpe.pt, is_half=False, device="cpu").infer_from_audio(x, thred=0.03)
//   uv = f0==0; f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])   # 仅 UV 插值
//   全零 → np.interp 抛异常 → 官方跳过该样本（本实现 valid=false）
//   coarse_f0：1127·ln(1+f0/700) 缩放到 [1,255] → np.rint(half-even) → int
//
// C++：RMVPE 复用 rvc_core `rvc::rmvpe::RmvpeExtractor`；插值复用
// `rvc::dsp::interpolate_unvoiced`（语义 = np.interp：内部线性 + 两端端点值）；
// bins 复用 `rvc::dsp::coarse_pitch_bin`（与 RVC-GGML 推理同函数，保证训练/推理一致）。

#pragma once

#include <cstdint>
#include <vector>

#include "rmvpe/rmvpe.h"

namespace rvc_train::prep {

struct F0Result {
    bool valid = false;           // 有发声帧（全零 → false，对应官方跳过该 chunk）
    std::vector<float> raw;       // RMVPE 原始 f0（Hz，0=无声；诊断/对拍用）
    std::vector<float> f0_hz;     // UV 插值后 f0（valid=false 时为空）
    std::vector<int32_t> f0_bin;  // coarse bins 1..255（valid=false 时为空）
};

// 输入：归一化后的 16k chunk（与官方 1_16k_wavs 内容一致）
F0Result compute_f0_rmvpe(const std::vector<float>& audio16k, rvc::rmvpe::RmvpeExtractor& rmvpe);

} // namespace rvc_train::prep
