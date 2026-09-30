// RVC-Train-GGML - ResBlock1（src/model/resblock.h）
//
// 对应运行时 `infer_pack/modules.py:252-366` 的 `ResBlock1`：
//   每对 (convs1[i], convs2[i])：LeakyReLU(0.1) →（乘 mask）→ conv1（膨胀）→
//   LeakyReLU(0.1) →（乘 mask）→ conv2（膨胀 1）→ 残差相加；最后乘 mask。
// 权重为 weight_norm（训练态 GGUF 里的 weight_g/weight_v，不折叠）。
#pragma once

#include "model/model.h"

#include "traincpp/traincpp.h"

#include <cstdint>
#include <string>
#include <vector>

namespace rvc_train {
namespace model {

struct ResBlock1Config {
    std::string prefix;                // 如 "dec.resblocks.0"
    int64_t channels = 0;
    int64_t kernel_size = 3;
    std::vector<int64_t> dilations;    // 长度 3，如 {1,3,5}
};

struct ResBlock1Module {
    ResBlock1Config cfg;
    std::vector<traincpp::WeightNorm> c1_wn;
    std::vector<traincpp::WeightNorm> c2_wn;
    std::vector<traincpp::Conv1d> c1;
    std::vector<traincpp::Conv1d> c2;
};

// 按前缀取参数并装配（只做结构，不装载数据；张量名保持 GGUF 原名不变）。
bool resblock1_load(const ModelParams& params, const ResBlock1Config& cfg, ResBlock1Module* out,
                    std::string* err);

// 前向捕获（对拍用；不需要时传 nullptr）
struct ResBlock1Captures {
    traincpp::Tensor* t0 = nullptr;    // 第 0 对 convs1 的输出
};

// x=[T,C,B]，x_mask=[T,1,B]（可为 nullptr）；输出 [T,C,B]
traincpp::Tensor* resblock1_forward(traincpp::Context* ctx, const ResBlock1Module& rb,
                                    traincpp::Tensor* x, traincpp::Tensor* x_mask,
                                    ResBlock1Captures* caps = nullptr);

} // namespace model
} // namespace rvc_train
