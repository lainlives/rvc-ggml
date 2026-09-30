// RVC-Train-GGML - 判别器（MPD：DiscriminatorS/P + 汇总）头文件（src/model/discriminator.h）
//
// 对应运行时 `infer_pack/models.py:1071-1242`：
//   `MultiPeriodDiscriminator`（v1：6 P + 1 S）/ `MultiPeriodDiscriminatorV2`（8 P + 1 S）：
//   `discriminators[0] = DiscriminatorS`，其后每个 period 一个 `DiscriminatorP`；
//   `forward(y, y_hat)` 对每个子判别器分别跑 real/fake，返回 (out, fmap) 的四个列表。
//   判别器是训练专属（推理不需要），RVC-GGML 无对应实现。
//
// 子判别器结构（权重均为 weight_norm，weight_g/weight_v 不折叠）：
//   S：6×Conv1d（k=[15,41,41,41,41,5]、s=[1,4,4,4,4,1]、groups=[1,4,16,64,256,1]）
//      每层后 LeakyReLU(0.1) → conv_post(k3,s1) → flatten；fmap 共 7 项；
//   P：波形按 period 折叠为 [W=period, H=T/period]（不足时 reflect 右填充）→
//      5×Conv2d（k=(5,1)、s=(3,1)、padding=(2,0)）每层后 LeakyReLU(0.1) →
//      conv_post(k=(3,1)) → flatten；fmap 共 6 项。
//
// 布局：wave [T,1,B]（torch [B,1,T] 的 ne 反序）；P 折叠后 [period,T/period,1,B]；
// 子判别器输出 flatten 后 [N,B]（conv_post 输出通道为 1，无歧义）。
// 手工装配 POD（Q13），张量名保持 GGUF 原名。
#pragma once

#include "model/model.h"

#include "traincpp/traincpp.h"

#include <cstdint>
#include <string>
#include <vector>

namespace rvc_train {
namespace model {

// LeakyReLU 斜率（运行时 `modules.LRELU_SLOPE`）
constexpr float kMpdLreluSlope = 0.1f;

struct MpdConfig {
    std::string prefix = "discriminators";  // 参数名前缀（torch 键名）
    std::vector<int64_t> periods;            // P 的周期（顺序 = discriminators.1..n）
    int64_t kernel_size = 5;                 // P 的核（S 的核由权重形状决定）
    int64_t stride = 3;                      // P 的 stride
};

// DiscriminatorP：5 层 Conv2d + conv_post（全部 weight_norm）
struct DiscPModule {
    int64_t period = 0;
    traincpp::WeightNorm conv_wn[5];
    traincpp::Conv2d convs[5];
    traincpp::WeightNorm post_wn;
    traincpp::Conv2d conv_post;
};

// DiscriminatorS：6 层分组 Conv1d + conv_post（全部 weight_norm）
struct DiscSModule {
    traincpp::WeightNorm conv_wn[6];
    traincpp::Conv1d convs[6];
    traincpp::WeightNorm post_wn;
    traincpp::Conv1d conv_post;
};

struct MpdModule {
    MpdConfig cfg;
    DiscSModule s;                    // discriminators.0
    std::vector<DiscPModule> ps;      // discriminators.1..n（与 cfg.periods 对应）
};

// 按名装配（仅填 POD，不改张量名）；失败返回 false 并写 err。
bool mpd_load(const ModelParams& params, const MpdConfig& cfg, MpdModule* out, std::string* err);

struct DiscForward {
    traincpp::Tensor* out = nullptr;              // flatten 后 [N,B]
    std::vector<traincpp::Tensor*> fmap;          // 每层输出（含 conv_post；LeakyReLU 后）
};

struct MpdForward {
    std::vector<DiscForward> r;   // real（wave）
    std::vector<DiscForward> g;   // fake（wave_hat）
};

// y/y_hat: [T,1,B] F32。顺序 = [S, P(periods...)]，与运行时一致。
// 失败返回空（r 为空即视为失败）。
MpdForward mpd_forward(traincpp::Context* ctx, const MpdModule& m, traincpp::Tensor* wave,
                       traincpp::Tensor* wave_hat);

} // namespace model
} // namespace rvc_train
