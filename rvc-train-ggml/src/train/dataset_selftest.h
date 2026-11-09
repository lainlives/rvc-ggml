// RVC-Train-GGML - M5-1 data loader comparison-test entry (src/train/dataset_selftest.h)
#pragma once

#include <string>

namespace rvc_train {
namespace train {

// Comparison test for the M5-1 data loader (host only, no device needed):
//   --golden <dir> (manifest.dataset.json + raw) --exp <M4 cache directory>
// Returns 0 = all passed; 1 = failed; 2 = argument/environment error.
int run_dataset_selftest(const std::string& golden_dir, const std::string& exp_dir);

} // namespace train
} // namespace rvc_train
