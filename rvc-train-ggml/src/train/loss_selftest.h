// RVC-Train-GGML - M3-2 loss/frontend golden comparison-test entry (src/train/loss_selftest.h)
//
// Golden mechanism: tools/gen_golden_loss.py -> work\golden\m3 (manifest.loss.<module>.json +
// loss.<module>.*.raw, shape = torch order). Semantic baseline = official 81eed5e training code.
// Criteria: forward rel ≤1e-4; backward rel ≤1e-3 (fallback: rms ≤2e-3 or max_abs ≤1e-5).
#pragma once

#include "traincpp/traincpp.h"

#include <string>

namespace rvc_train {
namespace train {

// module: stft | mel | lsgan_d | lsgan_g | fmap | kl | mel_l1 | all
// Returns: 0 = all passed; 1 = some failed; 2 = argument/environment error
int run_loss_selftest(const std::string& golden_dir, const std::string& module,
                      traincpp::Device* dev);

} // namespace train
} // namespace rvc_train
