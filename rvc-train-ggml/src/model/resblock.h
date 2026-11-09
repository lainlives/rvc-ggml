// RVC-Train-GGML - ResBlock1 (src/model/resblock.h)
//
// Corresponds to runtime `infer_pack/modules.py:252-366` `ResBlock1`:
//   for each pair (convs1[i], convs2[i]): LeakyReLU(0.1) -> (mask multiply) -> conv1 (dilated) ->
//   LeakyReLU(0.1) -> (mask multiply) -> conv2 (dilation 1) -> residual add; finally multiply by mask.
// Weights use weight_norm (weight_g/weight_v in the training-state GGUF, not folded).
#pragma once

#include "model/model.h"

#include "traincpp/traincpp.h"

#include <cstdint>
#include <string>
#include <vector>

namespace rvc_train {
namespace model {

struct ResBlock1Config {
    std::string prefix;                // e.g. "dec.resblocks.0"
    int64_t channels = 0;
    int64_t kernel_size = 3;
    std::vector<int64_t> dilations;    // length 3, e.g. {1,3,5}
};

struct ResBlock1Module {
    ResBlock1Config cfg;
    std::vector<traincpp::WeightNorm> c1_wn;
    std::vector<traincpp::WeightNorm> c2_wn;
    std::vector<traincpp::Conv1d> c1;
    std::vector<traincpp::Conv1d> c2;
};

// Fetch parameters by prefix and assemble (structure only, no data loading; tensor names keep their original GGUF names).
bool resblock1_load(const ModelParams& params, const ResBlock1Config& cfg, ResBlock1Module* out,
                    std::string* err);

// Forward captures (for comparison; pass nullptr if not needed)
struct ResBlock1Captures {
    traincpp::Tensor* t0 = nullptr;    // output of convs1 for pair 0
};

// x=[T,C,B], x_mask=[T,1,B] (may be nullptr); output [T,C,B]
traincpp::Tensor* resblock1_forward(traincpp::Context* ctx, const ResBlock1Module& rb,
                                    traincpp::Tensor* x, traincpp::Tensor* x_mask,
                                    ResBlock1Captures* caps = nullptr);

} // namespace model
} // namespace rvc_train
