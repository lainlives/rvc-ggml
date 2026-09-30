// RVC-Train-GGML - GeneratorNSF dec（src/model/dec.h）
//
// 对应运行时 `infer_pack/models.py:467-584` 的 `GeneratorNSF`（f0 版）：
//   sine 激励（主机侧 SineGen → m_source.l_linear + tanh）→ noise_convs 短路注入；
//   conv_pre(z) + cond(g) → 4 级 [LeakyReLU(0.1) → ConvT1d(weight_norm，中心裁剪 pad)
//   + noise_conv(har_source) → num_kernels 个 ResBlock1 取平均] →
//   LeakyReLU(0.01) → conv_post（无 bias）→ tanh。
// 权重为训练态 GGUF（weight_g/weight_v 不折叠；张量名保持原名）。
#pragma once

#include "model/model.h"
#include "model/resblock.h"

#include "traincpp/traincpp.h"

#include <cstdint>
#include <string>
#include <vector>

namespace rvc_train {
namespace model {

struct DecConfig {
    std::string prefix = "dec";         // 参数名前缀（torch 键名）
    int64_t in_channels = 0;            // z 通道（conv_pre 输入）
    int64_t upsample_initial_channel = 0;
    int64_t gin_channels = 0;           // 仅记录（cond 形状由权重决定）
    std::vector<int64_t> upsample_rates;
    std::vector<int64_t> upsample_kernel_sizes;
    std::vector<int64_t> resblock_kernel_sizes;
    std::vector<int64_t> resblock_dilation_sizes;   // 展平（每个 kernel 3 个 dilation）
};

struct DecModule {
    DecConfig cfg;
    traincpp::Linear l_linear;                      // m_source.l_linear（1→1，作用于激励源）
    traincpp::Conv1d conv_pre;
    traincpp::Conv1d cond;
    traincpp::Conv1d conv_post;
    std::vector<traincpp::WeightNorm> ups_wn;
    std::vector<traincpp::ConvTranspose1d> ups;
    std::vector<traincpp::Conv1d> noise_convs;
    std::vector<ResBlock1Module> resblocks;         // n_up × num_kernels
    int64_t num_kernels = 0;
};

// 按前缀取参数并装配（只做结构，不装载数据；张量名保持 GGUF 原名不变）。
bool dec_load(const ModelParams& params, const DecConfig& cfg, DecModule* out, std::string* err);

// 前向捕获（对拍用；不需要时传 nullptr）
struct DecCaptures {
    traincpp::Tensor* sine_merged = nullptr;    // l_linear+tanh 后激励源 [L,1,B]
    traincpp::Tensor* cond = nullptr;           // cond_layer 输出 [1,512,B]
    traincpp::Tensor* conv_pre = nullptr;       // conv_pre 输出 [T,512,B]
    traincpp::Tensor* x_cond = nullptr;         // conv_pre + cond 后 [T,512,B]
    std::vector<traincpp::Tensor*> ups;         // 每级 ConvT1d 输出
    std::vector<traincpp::Tensor*> levels;      // 每级 ResBlock 平均后
};

// z=[T,C,B]；sine_source=[L,1,B]（L = T·∏upsample_rates，主机 SineGen 产出）；
// g=[1,gin,B]（无条件传 nullptr）。返回 o=[L,1,B]；失败返回 nullptr。
traincpp::Tensor* dec_forward(traincpp::Context* ctx, const DecModule& dec, traincpp::Tensor* z,
                              traincpp::Tensor* sine_source, traincpp::Tensor* g = nullptr,
                              DecCaptures* caps = nullptr);

} // namespace model
} // namespace rvc_train
