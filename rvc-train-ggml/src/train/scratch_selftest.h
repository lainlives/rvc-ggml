// RVC-Train-GGML - M7-2 scratch spec comparison self-test (src/train/scratch_selftest.h)
//
// Part of gate G-7.2: the from-scratch specs must match the existing fixtures tensor by tensor (v1-40k G 560 / v1 D 129),
// and the generated training-state GGUF must pass the structural self-test and TrainWeights::read_meta/ModelConfig::validate.
#pragma once

#include <string>

namespace rvc_train {
namespace train {

// config_path: training config supplying the model/data sections (e.g. configs/v1-40k-f0.json)
// g_hashes/d_hashes: M1 hash manifests (name ne0 ne1 ne2 ne3 nbytes crc32)
// out_dir: directory where the scratch G/D GGUF files are generated
// Returns 0 = PASS, non-zero = FAIL.
int run_scratch_selftest(const std::string& config_path, const std::string& g_hashes,
                         const std::string& d_hashes, const std::string& out_dir);

} // namespace train
} // namespace rvc_train
