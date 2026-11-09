// RVC-Train-GGML - WN gated encoder (src/model/wn.h)
//
// Corresponds to runtime `infer_pack/modules.py:136-251` `WN`:
//   g = cond_layer(g) (1×1, outputs 2H·L channels, sliced per layer)
//   each layer: x_in = in_layer(x) (dilated convolution with k=D); acts = tanh(x_in + first half of g_l)·sigmoid(second half);
//          res_skip = res_skip_layer(acts);
//          non-final layer: x = (x + first half of res_skip)·mask, output += second half of res_skip;
//          final layer: output += res_skip; finally output·mask.
// Weights use weight_norm (weight_g/weight_v in the training-state GGUF, not folded).
#pragma once

#include "model/model.h"

#include "traincpp/traincpp.h"

#include <cstdint>
#include <string>
#include <vector>

namespace rvc_train {
namespace model {

struct WnConfig {
    std::string prefix;             // parameter name prefix (torch key name), e.g. "enc_q.enc"
    int64_t hidden_channels = 0;
    int64_t kernel_size = 0;
    int64_t n_layers = 0;
    int64_t gin_channels = 0;       // 0 means unconditional
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

// Fetch parameters by prefix and assemble (structure only, no data loading; tensor names keep their original GGUF names).
bool wn_load(const ModelParams& params, const WnConfig& cfg, WnModule* out, std::string* err);

// Forward captures (for comparison; pass nullptr if not needed)
struct WnForwardCaptures {
    traincpp::Tensor* cond = nullptr;    // cond_layer output [1, 2H·L, B]
    traincpp::Tensor* x_in0 = nullptr;   // layer 0 in_layer output [T, 2H, B]
    traincpp::Tensor* acts0 = nullptr;   // layer 0 activation [T, H, B]
};

// x=[T,H,B], x_mask=[T,1,B], g=[1,gin,B] (pass nullptr when no g); output [T,H,B]
traincpp::Tensor* wn_forward(traincpp::Context* ctx, const WnModule& wn, traincpp::Tensor* x,
                             traincpp::Tensor* x_mask, traincpp::Tensor* g,
                             WnForwardCaptures* caps = nullptr);

} // namespace model
} // namespace rvc_train
