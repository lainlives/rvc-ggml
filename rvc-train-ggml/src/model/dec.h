// RVC-Train-GGML - GeneratorNSF dec (src/model/dec.h)
//
// Corresponds to runtime `infer_pack/models.py:467-584` `GeneratorNSF` (f0 version):
//   sine excitation (host-side SineGen -> m_source.l_linear + tanh) -> noise_convs short-circuit injection;
//   conv_pre(z) + cond(g) -> 4 stages [LeakyReLU(0.1) -> ConvT1d(weight_norm, center-crop pad)
//   + noise_conv(har_source) -> average of num_kernels ResBlock1] ->
//   LeakyReLU(0.01) -> conv_post (no bias) -> tanh.
// Weights are training-state GGUF (weight_g/weight_v not folded; tensor names kept as-is).
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
    std::string prefix = "dec";         // parameter name prefix (torch key name)
    int64_t in_channels = 0;            // z channels (conv_pre input)
    int64_t upsample_initial_channel = 0;
    int64_t gin_channels = 0;           // recorded only (cond shape is determined by weights)
    std::vector<int64_t> upsample_rates;
    std::vector<int64_t> upsample_kernel_sizes;
    std::vector<int64_t> resblock_kernel_sizes;
    std::vector<int64_t> resblock_dilation_sizes;   // flattened (3 dilations per kernel)
};

struct DecModule {
    DecConfig cfg;
    traincpp::Linear l_linear;                      // m_source.l_linear (1->1, applied to the excitation source)
    traincpp::Conv1d conv_pre;
    traincpp::Conv1d cond;
    traincpp::Conv1d conv_post;
    std::vector<traincpp::WeightNorm> ups_wn;
    std::vector<traincpp::ConvTranspose1d> ups;
    std::vector<traincpp::Conv1d> noise_convs;
    std::vector<ResBlock1Module> resblocks;         // n_up × num_kernels
    int64_t num_kernels = 0;
};

// Fetch parameters by prefix and assemble (structure only, no data loading; tensor names keep their original GGUF names).
bool dec_load(const ModelParams& params, const DecConfig& cfg, DecModule* out, std::string* err);

// Forward captures (for comparison; pass nullptr if not needed)
struct DecCaptures {
    traincpp::Tensor* sine_merged = nullptr;    // excitation source after l_linear+tanh [L,1,B]
    traincpp::Tensor* cond = nullptr;           // cond_layer output [1,512,B]
    traincpp::Tensor* conv_pre = nullptr;       // conv_pre output [T,512,B]
    traincpp::Tensor* x_cond = nullptr;         // after conv_pre + cond [T,512,B]
    std::vector<traincpp::Tensor*> ups;         // output of each ConvT1d stage
    std::vector<traincpp::Tensor*> levels;      // after each stage's ResBlock average
};

// z=[T,C,B]; sine_source=[L,1,B] (L = T·∏upsample_rates, produced by host SineGen);
// g=[1,gin,B] (pass nullptr for unconditional). Returns o=[L,1,B]; nullptr on failure.
traincpp::Tensor* dec_forward(traincpp::Context* ctx, const DecModule& dec, traincpp::Tensor* z,
                              traincpp::Tensor* sine_source, traincpp::Tensor* g = nullptr,
                              DecCaptures* caps = nullptr);

} // namespace model
} // namespace rvc_train
