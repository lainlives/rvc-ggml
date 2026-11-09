// RVC-Train-GGML - M7-3 gradient accumulation equivalence self-test (src/train/grad_accum_selftest.h)
//
// Gate G-7.3 (user decision: gradient-level equivalence is the primary criterion):
//   With identical weights and data, arm A (batch=B, grad_accum=K, run_step_accum) and
//   arm B (batch=B×K, grad_accum=1, run_step) must have equivalent **parameter gradients**
//   (in-graph accumulated losses are scaled by 1/K -> accumulated gradient == full-batch mean gradient).
// Small-scale Vulkan/CPU both fine; on this machine use the small Vulkan profile (bucket=384/B=1/K=2/disc=1024).
#pragma once

#include <cstdint>
#include <string>

namespace traincpp {
class Device;
}

namespace rvc_train {
namespace train {

// weights_g/weights_d = training-state GGUF; exp_dir = M4 cache.
// bucket = graph T; batch = samples per micro-batch B (>=1); grad_accum = accumulation steps K (>=2);
// disc_len = D input length (<=0 = official full segment). Returns 0 = gate passed; 1 = failed; 2 = argument/environment error.
int run_grad_accum_selftest(const std::string& weights_g, const std::string& weights_d,
                            const std::string& exp_dir, traincpp::Device* dev, int64_t bucket,
                            int64_t batch, int64_t grad_accum, int64_t disc_len);

} // namespace train
} // namespace rvc_train
