// sinegen.h
// RVC SineGen（harmonic_num=0）主机端实现，用于生成 HiFiGAN NSF 的激励源。
// 与 infer/lib/infer_pack/models.py 的 SineGen.forward 算法一致（含噪声混合）。

#pragma once

#include <cstdint>
#include <vector>

namespace rvc {
namespace sinegen {

    // 生成 raw sine waves（SineGen 输出，未经 m_source.l_linear/tanh）
    //   f0:   [frames] Hz（0 = 清音）
    //   upp:  每帧上采样倍数（= prod(upsample_rates)）
    //   noise: [frames*upp] 高斯噪声（nullptr 时用 seed 内部生成）
    //   返回: [frames*upp]
    std::vector<float> generate(const std::vector<float>& f0, int sample_rate, int upp,
        const std::vector<float>* noise = nullptr, uint32_t seed = 1234);

} // namespace sinegen
} // namespace rvc
