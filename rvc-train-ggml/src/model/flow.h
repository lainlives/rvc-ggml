// RVC-Train-GGML - ResidualCouplingBlock flow (src/model/flow.h)
//
// Corresponds to runtime `infer_pack/models.py:82-147` `ResidualCouplingBlock` (training **forward**):
//   flow order [RCL, Flip, RCL, Flip, RCL, Flip, RCL, Flip];
//   RCL (`modules.py:477-537`, mean_only=True):
//     x0,x1 = split in half; h = pre(x0)·mask -> WN(h,mask,g) -> stats = post(h)·mask;
//     forward: x1 = stats + x1·mask; x = cat([x0,x1]);
//   Flip (`modules.py:440-457`): flip along the channel dim (this implementation uses a C×C permutation matrix mul_mat,
//   same as RVC-GGML inference; the matrix is filled by `flow_fill_flip_perm` after buffer allocation).
// Weights are training-state GGUF (WN weight_g/weight_v not folded; tensor names kept as-is).
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
    std::string prefix = "flow";    // parameter name prefix (torch key name)
    int64_t channels = 0;           // 192 (must be even)
    int64_t hidden_channels = 0;    // 192
    int64_t kernel_size = 5;
    int64_t dilation_rate = 1;
    int64_t n_layers = 3;
    int64_t gin_channels = 0;       // 256
    int64_t n_flows = 4;
};

// Single ResidualCouplingLayer (weights at flow.flows.{2i}.*)
struct FlowLayer {
    traincpp::Conv1d pre;           // 1×1: half -> hidden
    WnModule enc;                   // WN
    traincpp::Conv1d post;          // 1×1: hidden -> half (mean_only)
};

struct FlowModule {
    FlowConfig cfg;
    std::vector<FlowLayer> layers;  // n_flows entries
};

bool flow_load(const ModelParams& params, const FlowConfig& cfg, FlowModule* out, std::string* err);

// Flip permutation matrix [C,C] (ne=[C,C]): y = mul_mat(P, x) gives the channel flip.
// Must be called after buffer allocation (selftest goes through `SelftestBuildIO::prepare`; the training loop calls it once after allocation).
bool flow_fill_flip_perm(traincpp::Tensor* perm, std::string* err = nullptr);

// Forward captures (for comparison; pass nullptr if not needed)
struct FlowCaptures {
    std::vector<traincpp::Tensor*> pre;     // pre convolution output of each layer (before mask multiply)
    std::vector<traincpp::Tensor*> enc;     // WN output of each layer
    std::vector<traincpp::Tensor*> stats;   // post convolution output of each layer (before mask multiply)
};

// x=[T,C,B], x_mask=[T,1,B], g=[1,gin,B], perm=[C,C] (already filled); output y=[T,C,B]
traincpp::Tensor* flow_forward(traincpp::Context* ctx, const FlowModule& flow, traincpp::Tensor* x,
                               traincpp::Tensor* x_mask, traincpp::Tensor* g, traincpp::Tensor* perm,
                               FlowCaptures* caps = nullptr);

} // namespace model
} // namespace rvc_train
