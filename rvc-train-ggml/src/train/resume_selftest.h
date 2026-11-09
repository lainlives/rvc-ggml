// RVC-Train-GGML - M7-5 resume self-test entry (src/train/resume_selftest.h)
//
// Gate G-7.5: resuming matches uninterrupted training (bit-identical) + changing the config errors out explicitly.
// Protocol:
//   A  = fresh Trainer runs total_steps steps (crossing epoch boundaries), recording the 6 losses and state trace per step;
//   B1 = fresh Trainer runs pre_steps steps (state must match A) -> save checkpoint (with meta);
//   B2 = fresh Trainer loads that checkpoint (state must match A at pre) -> resumes to total_steps;
//        the resumed segment's 6 losses and state must be **bit-identical** to the corresponding steps of A;
//   Negative: resuming after changing batch_size/bucket/seed/grad_accum/learning_rate -> must error out explicitly;
//             resuming after tampering with the meta sample fingerprint -> must error out explicitly.
// Recommend a very small bucket (e.g. bucket=200: cache_v1 has only 2 samples -> 2 steps/epoch) so epoch boundaries are crossed within a few steps.
#pragma once

#include <cstdint>
#include <string>

namespace traincpp {
class Device;
}

namespace rvc_train {
namespace train {

// weights_g/weights_d = training-state GGUF; exp_dir = M4 cache; out_dir = log/checkpoint output.
// total_steps >= 2; pre_steps in (0, total_steps).
// config_path (since V2-6): when non-empty, use it as the base training config (e.g. configs/v2-48k-f0.json), then override test parameters;
//   for v2 caches (hop/sr/feature dims differ from the v1 defaults). Empty = old behavior (built-in v1 defaults).
// Returns 0 = all passed; 1 = failed; 2 = argument/environment error.
int run_resume_selftest(const std::string& weights_g, const std::string& weights_d,
                        const std::string& exp_dir, const std::string& out_dir,
                        traincpp::Device* dev, int64_t total_steps, int64_t pre_steps,
                        int64_t bucket, int64_t batch, int64_t disc_len, int64_t grad_accum,
                        const std::string& config_path = std::string());

} // namespace train
} // namespace rvc_train
