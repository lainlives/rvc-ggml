// RVC-Train-GGML - WN 门控编码器（src/model/wn.h）
//
// 对应运行时 `infer_pack/modules.py:136-251` 的 `WN`：
//   g = cond_layer(g)（1×1，输出 2H·L 通道，按层切片）
//   每层：x_in = in_layer(x)（k=D 的膨胀卷积）；acts = tanh(x_in+g_l 前半)·sigmoid(后半)；
//          res_skip = res_skip_layer(acts)；
//          非末层：x = (x + res_skip 前半)·mask，output += res_skip 后半；
//          末层：output += res_skip；最后 output·mask。
// 权重为 weight_norm（训练态 GGUF 里的 weight_g/weight_v，不折叠）。
#pragma once

#include "model/model.h"

#include "traincpp/traincpp.h"

#include <cstdint>
#include <string>
#include <vector>

namespace rvc_train {
namespace model {

struct WnConfig {
    std::string prefix;             // 参数名前缀（torch 键名），如 "enc_q.enc"
    int64_t hidden_channels = 0;
    int64_t kernel_size = 0;
    int64_t n_layers = 0;
    int64_t gin_channels = 0;       // 0 表示无条件
    int64_t dilation_rate = 1;
};

struct WnModule {
    WnConfig cfg;
    traincpp::WeightNorm cond_wn;
    traincpp::Conv1d cond;
    std::vector<traincpp::WeightNorm> in_wn;
    std::vector<traincpp::WeightNorm> rs_wn;
    std::vector<traincpp::Conv1d> in_layers;
    std::vector<traincpp::Conv1d> res_skip_layers;
};

// 按前缀取参数并装配（只做结构，不装载数据；张量名保持 GGUF 原名不变）。
bool wn_load(const ModelParams& params, const WnConfig& cfg, WnModule* out, std::string* err);

// 前向捕获（对拍用；不需要时传 nullptr）
struct WnForwardCaptures {
    traincpp::Tensor* cond = nullptr;    // cond_layer 输出 [1, 2H·L, B]
    traincpp::Tensor* x_in0 = nullptr;   // 第 0 层 in_layer 输出 [T, 2H, B]
    traincpp::Tensor* acts0 = nullptr;   // 第 0 层激活 [T, H, B]
};

// x=[T,H,B]，x_mask=[T,1,B]，g=[1,gin,B]（无 g 传 nullptr）；输出 [T,H,B]
traincpp::Tensor* wn_forward(traincpp::Context* ctx, const WnModule& wn, traincpp::Tensor* x,
                             traincpp::Tensor* x_mask, traincpp::Tensor* g,
                             WnForwardCaptures* caps = nullptr);

} // namespace model
} // namespace rvc_train
