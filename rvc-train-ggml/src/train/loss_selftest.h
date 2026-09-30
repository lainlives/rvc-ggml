// RVC-Train-GGML - M3-2 损失/前端 golden 对拍入口（src/train/loss_selftest.h）
//
// golden 机制：tools/gen_golden_loss.py → work\golden\m3（manifest.loss.<module>.json +
// loss.<module>.*.raw，shape = torch 顺序）。语义基准 = 官方 81eed5e 训练代码。
// 判定：前向 rel ≤1e-4；反向 rel ≤1e-3（rms ≤2e-3 或 max_abs ≤1e-5 兜底）。
#pragma once

#include "traincpp/traincpp.h"

#include <string>

namespace rvc_train {
namespace train {

// module: stft | mel | lsgan_d | lsgan_g | fmap | kl | mel_l1 | all
// 返回：0 = 全部通过；1 = 有失败；2 = 参数/环境错误
int run_loss_selftest(const std::string& golden_dir, const std::string& module,
                      traincpp::Device* dev);

} // namespace train
} // namespace rvc_train
