// RVC-Train-GGML - M7-2 从零随机初始化：张量规格生成 + 初始化（src/model/scratch.h）
//
// 目的：在不依赖任何 .pth / 训练态 GGUF 的前提下，按 ModelConfig **纯配置驱动**生成
// RVC 生成器 G（enc_p/enc_q/flow/dec/emb_g）与判别器 D（MPD：S + P×len(d_periods)）的
// 全部参数 (name, ne) 规格，并按 PyTorch 默认初始化填充 F32 数据。
//
// 规格正确性以 M1 夹具哈希清单为准：v1-40k G = 560 张量、v1 D = 129 张量
// （work/models/f0G40k.hashes.txt / f0D40k.hashes.txt；--selftest-scratch 逐项比对）。
//
// 初始化口径（PyTorch 默认；R-M7-1 已记录"与 RVC init_weights 不完全一致"的风险）：
//   - Conv/Linear/weight_v：kaiming_uniform(a=√5) → U(-1/√fan_in, +1/√fan_in)
//   - bias：U(-1/√fan_in, +1/√fan_in)
//   - Embedding（emb_g/emb_pitch）：N(0,1)
//   - 相对位置 emb_rel_k/v：N(0, 1/√dk)（dk = hidden/n_heads）
//   - LayerNorm gamma=1、beta=0
//   - weight_g = ‖weight_v‖（对除最后一维外的所有维求和）→ 折叠后 w = weight_v
#pragma once

#include "model/config.h"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace rvc_train {
namespace model {

enum class InitKind {
    Kaiming,    // U(-1/√fan_in, +1/√fan_in)
    Bias,       // U(-1/√fan_in, +1/√fan_in)
    Embedding,  // N(0,1)
    RelEmb,     // N(0, 1/√ne0)
    Ones,       // 常量 1（LayerNorm gamma）
    Zeros,      // 常量 0（LayerNorm beta）
    WnG,        // weight_g：由配对 weight_v 计算（g = ‖v‖，最后一维为通道）
};

// 单个参数张量的规格；ne 为 ggml 序（torch 形状反序），4 维补 1。
struct TensorSpec {
    std::string name;
    std::array<int64_t, 4> ne{1, 1, 1, 1};
    InitKind init = InitKind::Kaiming;
    int64_t fan_in = 1;  // Kaiming/Bias 使用；其余忽略
    int ch_dim = 2;      // weight_norm 的归一化/输出维（Conv1d/ConvT=2，Conv2d=3）
};

// 生成 G（enc_p/enc_q/flow/dec/emb_g）全部规格。失败写 err。
bool build_g_specs(const ModelConfig& mc, std::vector<TensorSpec>* out, std::string* err);
// 生成 D（discriminators.*）全部规格（S + P×len(d_periods)）。失败写 err。
bool build_d_specs(const ModelConfig& mc, std::vector<TensorSpec>* out, std::string* err);

// 按规格生成确定性 F32 数据（与 specs 等长；weight_g 由配对 weight_v 计算）。
bool fill_specs(const std::vector<TensorSpec>& specs, uint64_t seed,
                std::vector<std::vector<float>>* out, std::string* err);

} // namespace model
} // namespace rvc_train
