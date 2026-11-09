// RVC-Train-GGML - host-side SineGen (src/model/sinegen.h)
//
// Ported from RVC-GGML `src/rvc/sinegen.cpp:24-95` (old `cumsum_shift` method, decided in Q11),
// same source as runtime infer_pack `models.py:353-407` and RVC-GGML `scripts/make_ref.py:160-199`.
// Pure host implementation, not part of the graph; noise is injected by the caller (comparison uses the same noise dumped by golden; generated uniformly by the training RNG during training).
#pragma once

#include <cstdint>
#include <vector>

namespace rvc_train {
namespace model {

// Generate raw sine waves (SineGen output, before m_source.l_linear/tanh):
//   f0:    [frames] Hz (<=0 treated as unvoiced)
//   noise: [frames*upp] Gaussian noise; when empty (nullptr or too short) generated internally from seed
//   seed:  only effective when noise is absent
// Returns [frames*upp]; returns an empty vector for invalid arguments.
std::vector<float> sinegen_generate(const float* f0, int64_t frames, int sample_rate, int upp,
                                    const float* noise, int64_t noise_len, uint32_t seed = 1234);

} // namespace model
} // namespace rvc_train
