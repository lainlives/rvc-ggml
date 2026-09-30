// RVC-Train-GGML - M3-3 双图 G/D + 一步 AdamW 对拍入口（src/train/train_step_selftest.h）
//
// golden 机制：tools/gen_golden_train_step.py → work\golden\m3（manifest.train_step.json +
// train_step.*.raw，shape = torch 顺序）。语义基准 = 官方 81eed5e train.py:425-527
// （D 每步两次前向；G 用更新后的 D；AdamW(0.8,0.99,eps=1e-9,lr=1e-4,wd=1e-2)）。
// 判定：forward rel ≤1e-4；loss rel ≤1e-4；ΔW rel ≤1e-4；无梯度泄漏（冻结侧无 grad_acc、
// G step 后 D 逐位不变、D step 后 G 逐位不变）。
#pragma once

#include "traincpp/traincpp.h"

#include <string>

namespace rvc_train {
namespace train {

// weights_g：G 训练态 GGUF（f0G40k.train.gguf，含 enc_q）
// weights_d：D 训练态 GGUF（f0D40k.train.gguf）
// golden_dir：M3 golden 目录（含 manifest.train_step.json 与 train_step.*.raw）
// 返回：0 = 全部通过；1 = 有失败；2 = 参数/环境错误
int run_train_step_selftest(const std::string& weights_g, const std::string& weights_d,
                            const std::string& golden_dir, traincpp::Device* dev);

// M5-2 训练循环 10 步轨迹对拍（单 ctx TrainStep）：
//   golden = tools/gen_golden_train_loop.py 产物（manifest.train_loop.json +
//   train_loop.s{i}_*.raw），逐步注入 ids/q_noise/sine/wave 并对比 6 项 loss。
//   门 G-5.2b（Q24 修订）：首步 rel ≤1e-4（实测 5.3e-7）；后续逐步容差 = 混沌探针系综
//   ×3 包络（`<golden>/probe` 回退或 manifest `params.tols`，由 tools/make_train_loop_envelope.py
//   固化）+ loss 有限（F32 下逐值 1e-2 不可达：探针自身 10 步内即发散）。
// 返回：0 = 全部通过；1 = 有失败；2 = 参数/环境错误
int run_train_loop_selftest(const std::string& weights_g, const std::string& weights_d,
                            const std::string& golden_dir, traincpp::Device* dev);

} // namespace train
} // namespace rvc_train
