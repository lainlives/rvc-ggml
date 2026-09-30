// RVC-Train-GGML - M7-2 scratch 规格对拍自检（src/train/scratch_selftest.h）
//
// 门 G-7.2 的一部分：从零规格必须与既有夹具逐张量一致（v1-40k G 560 / v1 D 129），
// 且生成的训练态 GGUF 可被结构自检与 TrainWeights::read_meta/ModelConfig::validate 通过。
#pragma once

#include <string>

namespace rvc_train {
namespace train {

// config_path：提供 model/data 段的训练配置（如 configs/v1-40k-f0.json）
// g_hashes/d_hashes：M1 哈希清单（name ne0 ne1 ne2 ne3 nbytes crc32）
// out_dir：生成 scratch G/D GGUF 的目录
// 返回 0 = PASS，非 0 = FAIL。
int run_scratch_selftest(const std::string& config_path, const std::string& g_hashes,
                         const std::string& d_hashes, const std::string& out_dir);

} // namespace train
} // namespace rvc_train
