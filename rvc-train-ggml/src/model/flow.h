// RVC-Train-GGML - ResidualCouplingBlock flow（src/model/flow.h）
//
// 对应运行时 `infer_pack/models.py:82-147` 的 `ResidualCouplingBlock`（训练**正向**）：
//   flows 顺序 [RCL, Flip, RCL, Flip, RCL, Flip, RCL, Flip]；
//   RCL（`modules.py:477-537`，mean_only=True）：
//     x0,x1 = 对半切；h = pre(x0)·mask → WN(h,mask,g) → stats = post(h)·mask；
//     正向：x1 = stats + x1·mask；x = cat([x0,x1])；
//   Flip（`modules.py:440-457`）：沿通道维翻转（本实现用 C×C 置换矩阵 mul_mat，
//   与 RVC-GGML 推理同法；矩阵在 buffer 分配后由 `flow_fill_flip_perm` 填充）。
// 权重为训练态 GGUF（WN 的 weight_g/weight_v 不折叠；张量名保持原名）。
#pragma once

#include "model/model.h"
#include "model/wn.h"

#include "traincpp/traincpp.h"

#include <cstdint>
#include <string>
#include <vector>

namespace rvc_train {
namespace model {

struct FlowConfig {
    std::string prefix = "flow";    // 参数名前缀（torch 键名）
    int64_t channels = 0;           // 192（必须为偶数）
    int64_t hidden_channels = 0;    // 192
    int64_t kernel_size = 5;
    int64_t dilation_rate = 1;
    int64_t n_layers = 3;
    int64_t gin_channels = 0;       // 256
    int64_t n_flows = 4;
};

// 单个 ResidualCouplingLayer（weights 位于 flow.flows.{2i}.*）
struct FlowLayer {
    traincpp::Conv1d pre;           // 1×1: half → hidden
    WnModule enc;                   // WN
    traincpp::Conv1d post;          // 1×1: hidden → half（mean_only）
};

struct FlowModule {
    FlowConfig cfg;
    std::vector<FlowLayer> layers;  // n_flows 个
};

bool flow_load(const ModelParams& params, const FlowConfig& cfg, FlowModule* out, std::string* err);

// Flip 置换矩阵 [C,C]（ne=[C,C]）：y = mul_mat(P, x) 得到通道翻转。
// 需在 buffer 分配后调用（selftest 走 `SelftestBuildIO::prepare`；训练循环在分配后调用一次）。
bool flow_fill_flip_perm(traincpp::Tensor* perm, std::string* err = nullptr);

// 前向捕获（对拍用；不需要时传 nullptr）
struct FlowCaptures {
    std::vector<traincpp::Tensor*> pre;     // 每层 pre 卷积输出（乘 mask 前）
    std::vector<traincpp::Tensor*> enc;     // 每层 WN 输出
    std::vector<traincpp::Tensor*> stats;   // 每层 post 卷积输出（乘 mask 前）
};

// x=[T,C,B]，x_mask=[T,1,B]，g=[1,gin,B]，perm=[C,C]（已填充）；输出 y=[T,C,B]
traincpp::Tensor* flow_forward(traincpp::Context* ctx, const FlowModule& flow, traincpp::Tensor* x,
                               traincpp::Tensor* x_mask, traincpp::Tensor* g, traincpp::Tensor* perm,
                               FlowCaptures* caps = nullptr);

} // namespace model
} // namespace rvc_train
