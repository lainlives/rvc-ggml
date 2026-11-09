// RVC-Train-GGML - enc_q (PosteriorEncoder) header (src/model/enc_q.h)
//
// Corresponds to runtime `infer_pack/models.py:148-189` `PosteriorEncoder` (enc_q is training-only, not used for inference):
//   x_mask = sequence_mask(x_lengths)            ([B,1,T])
//   x = pre(x)·mask                              (pre: 1×1 Conv1d, spec_channels->hidden)
//   x = enc(x, mask, g)                          (WN: 16 layers k=5 d=1 gin=256)
//   stats = proj(x)·mask                         (proj: 1×1 Conv1d, hidden->2·out)
//   m_q/logs_q = split(stats);
//   z = (m_q + noise·exp(logs_q))·mask           (noise injected by the caller to guarantee the same source as golden)
//
// Internal layout [T,C,B] (reversed ne order of torch [B,C,T]); g [1,gin,B] (torch [B,gin,1]).
// Weights: pre/proj are plain Conv1d; enc uses weight_norm (weight_g/weight_v not folded);
// Manually assembled POD, no init helper (Q13); tensor names keep their original GGUF names.
#pragma once

#include "model/model.h"
#include "model/wn.h"

#include "traincpp/traincpp.h"

#include <cstdint>
#include <string>

namespace rvc_train {
namespace model {

struct EncQConfig {
    std::string prefix = "enc_q";   // parameter name prefix (torch key name)
    int64_t in_channels = 0;        // 1025 (spec_channels)
    int64_t out_channels = 0;       // 192 (inter_channels)
    int64_t hidden_channels = 0;    // 192
    int64_t kernel_size = 5;        // WN convolution kernel
    int64_t dilation_rate = 1;
    int64_t n_layers = 16;
    int64_t gin_channels = 0;       // 256
};

struct EncQModule {
    EncQConfig cfg;
    traincpp::Conv1d pre;       // 1×1: in -> hidden (no weight_norm)
    WnModule enc;               // prefix = cfg.prefix + ".enc"
    traincpp::Conv1d proj;      // 1×1: hidden -> 2*out (no weight_norm)
};

// Assemble by name (fills POD only, tensor names unchanged); returns false and writes err on failure.
bool enc_q_load(const ModelParams& params, const EncQConfig& cfg, EncQModule* out,
                std::string* err);

// Forward captures (for comparison; pass nullptr if not needed). Layout (golden ne reversed):
//   pre/enc [T,H,B]; stats [T,2*out,B] (the first two are the corresponding intermediates before/after mask multiply);
//   cond/x_in0/acts0 come from the internal WN (cond [1,2H·L,B], x_in0 [T,2H,B], acts0 [T,H,B])
struct EncQCaptures {
    traincpp::Tensor* pre = nullptr;     // pre convolution output (before mask multiply)
    traincpp::Tensor* cond = nullptr;    // WN cond_layer output
    traincpp::Tensor* x_in0 = nullptr;   // WN layer 0 in_layer output
    traincpp::Tensor* acts0 = nullptr;   // WN layer 0 activation
    traincpp::Tensor* enc = nullptr;     // WN output (mask already applied)
    traincpp::Tensor* stats = nullptr;   // proj output (before mask multiply)
};

struct EncQOutputs {
    traincpp::Tensor* m_q = nullptr;     // first half of stats ([out,T,B] view)
    traincpp::Tensor* logs_q = nullptr;  // second half of stats ([out,T,B] view)
    traincpp::Tensor* z = nullptr;       // (m_q + noise·exp(logs_q))·mask
};

// Inputs:
//   spec   [in,T,B] F32 (linear spectrogram; must be contiguous)
//   x_mask [T,1,B] F32
//   g      [1,gin,B] F32 (may be nullptr = unconditional)
//   noise  [T,out,B] F32 (deterministic noise injected by the caller; same source as golden q_noise)
// Output: m_q/logs_q (channel-split views of stats) and z. Returns empty on failure.
EncQOutputs enc_q_forward(traincpp::Context* ctx, const EncQModule& m, traincpp::Tensor* spec,
                          traincpp::Tensor* x_mask, traincpp::Tensor* g, traincpp::Tensor* noise,
                          EncQCaptures* caps = nullptr);

} // namespace model
} // namespace rvc_train
