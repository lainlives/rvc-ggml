// RVC-Train-GGML - M5-3 training loop self-test entry (src/train/trainer_selftest.h)
//
// G-5.3 verification (Trainer + checkpoint round trip):
//   A = Trainer (real cache data) runs steps steps -> check the 6 losses are finite -> save checkpoint ->
//   checkpoint_verify (file matches A's in-memory state) -> A runs 1 more step recording loss/step/lr/RNG;
//   B = fresh Trainer loads the same checkpoint -> state check + checkpoint_verify ->
//   B runs 1 more step; B's next 6 losses must be bit-identical to A's (step/lr/RNG as well).
// Vulkan or CPU both fine (small scale on this machine uses Vulkan; large parameters are controlled by the CLI).
#pragma once

#include <cstdint>
#include <string>

namespace traincpp {
class Device;
}

namespace rvc_train {
namespace train {

// weights_g/weights_d = training-state GGUF; exp_dir = M4 cache; out_dir = log/checkpoint output.
// bucket/batch/disc_len are small-scale test parameters (disc_len=0 means the official full segment_samples).
// config_path (since V2-6): when non-empty, use it as the base training config (e.g. configs/v2-48k-f0.json), then override the test parameters above;
//   for v2 caches (hop/sr/feature dims differ from the v1 defaults). Empty = old behavior (built-in v1 defaults).
// Returns 0 = all passed; 1 = failed; 2 = argument/environment error.
int run_trainer_selftest(const std::string& weights_g, const std::string& weights_d,
                         const std::string& exp_dir, const std::string& out_dir,
                         traincpp::Device* dev, int64_t steps, int64_t bucket, int64_t batch,
                         int64_t disc_len, int64_t grad_accum = 1,
                         const std::string& config_path = std::string());

} // namespace train
} // namespace rvc_train
