// sinegen.h
// Host-side RVC SineGen (harmonic_num=0) implementation, used to generate the excitation source for HiFiGAN NSF.
// Algorithm matches SineGen.forward in infer/lib/infer_pack/models.py (including noise mixing).

#pragma once

#include <cstdint>
#include <vector>

namespace rvc {
namespace sinegen {

    // Generate raw sine waves (SineGen output, before m_source.l_linear/tanh)
    //   f0:   [frames] Hz (0 = unvoiced)
    //   upp:  per-frame upsampling factor (= prod(upsample_rates))
    //   noise: [frames*upp] Gaussian noise (generated internally from seed when nullptr)
    //   returns: [frames*upp]
    std::vector<float> generate(const std::vector<float>& f0, int sample_rate, int upp,
        const std::vector<float>* noise = nullptr, uint32_t seed = 1234);

} // namespace sinegen
} // namespace rvc
