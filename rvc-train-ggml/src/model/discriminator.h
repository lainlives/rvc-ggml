// RVC-Train-GGML - discriminator (MPD: DiscriminatorS/P + aggregation) header (src/model/discriminator.h)
//
// Corresponds to runtime `infer_pack/models.py:1071-1242`:
//   `MultiPeriodDiscriminator` (v1: 6 P + 1 S) / `MultiPeriodDiscriminatorV2` (8 P + 1 S):
//   `discriminators[0] = DiscriminatorS`, then one `DiscriminatorP` per period;
//   `forward(y, y_hat)` runs real/fake through each sub-discriminator respectively, returning four lists of (out, fmap).
//   The discriminator is training-only (not needed for inference); RVC-GGML has no corresponding implementation.
//
// Sub-discriminator structure (all weights are weight_norm; weight_g/weight_v not folded):
//   S: 6×Conv1d (k=[15,41,41,41,41,5], s=[1,4,4,4,4,1], groups=[1,4,16,64,256,1])
//      LeakyReLU(0.1) after each layer -> conv_post(k3,s1) -> flatten; 7 fmap entries in total;
//   P: waveform folded by period into [W=period, H=T/period] (reflect right padding if short) ->
//      5×Conv2d (k=(5,1), s=(3,1), padding=(2,0)) with LeakyReLU(0.1) after each layer ->
//      conv_post(k=(3,1)) -> flatten; 6 fmap entries in total.
//
// Layout: wave [T,1,B] (reversed ne order of torch [B,1,T]); after P folding [period,T/period,1,B];
// sub-discriminator output after flatten [N,B] (conv_post has 1 output channel, unambiguous).
// Manually assembled POD (Q13); tensor names keep their original GGUF names.
#pragma once

#include "model/model.h"

#include "traincpp/traincpp.h"

#include <cstdint>
#include <string>
#include <vector>

namespace rvc_train {
namespace model {

// LeakyReLU slope (runtime `modules.LRELU_SLOPE`)
constexpr float kMpdLreluSlope = 0.1f;

struct MpdConfig {
    std::string prefix = "discriminators";  // parameter name prefix (torch key name)
    std::vector<int64_t> periods;            // P periods (order = discriminators.1..n)
    int64_t kernel_size = 5;                 // P kernel (S kernel determined by weight shapes)
    int64_t stride = 3;                      // P stride
};

// DiscriminatorP: 5 Conv2d layers + conv_post (all weight_norm)
struct DiscPModule {
    int64_t period = 0;
    traincpp::WeightNorm conv_wn[5];
    traincpp::Conv2d convs[5];
    traincpp::WeightNorm post_wn;
    traincpp::Conv2d conv_post;
};

// DiscriminatorS: 6 grouped Conv1d layers + conv_post (all weight_norm)
struct DiscSModule {
    traincpp::WeightNorm conv_wn[6];
    traincpp::Conv1d convs[6];
    traincpp::WeightNorm post_wn;
    traincpp::Conv1d conv_post;
};

struct MpdModule {
    MpdConfig cfg;
    DiscSModule s;                    // discriminators.0
    std::vector<DiscPModule> ps;      // discriminators.1..n (corresponds to cfg.periods)
};

// Assemble by name (fills POD only, tensor names unchanged); returns false and writes err on failure.
bool mpd_load(const ModelParams& params, const MpdConfig& cfg, MpdModule* out, std::string* err);

struct DiscForward {
    traincpp::Tensor* out = nullptr;              // after flatten [N,B]
    std::vector<traincpp::Tensor*> fmap;          // per-layer outputs (including conv_post; after LeakyReLU)
};

struct MpdForward {
    std::vector<DiscForward> r;   // real (wave)
    std::vector<DiscForward> g;   // fake (wave_hat)
};

// y/y_hat: [T,1,B] F32. Order = [S, P(periods...)], consistent with runtime.
// Returns empty on failure (empty r means failure).
MpdForward mpd_forward(traincpp::Context* ctx, const MpdModule& m, traincpp::Tensor* wave,
                       traincpp::Tensor* wave_hat);

} // namespace model
} // namespace rvc_train
