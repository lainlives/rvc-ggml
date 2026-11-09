// RVC-Train-GGML - enc_p (TextEncoder, relative-position attention) header (src/model/enc_p.h)
//
// Corresponds to runtime `infer_pack/models.py:19-81` `TextEncoder` and
// `attentions.py:14-78` (Encoder) / `:166-387` (relative-position MultiHeadAttention):
//   phone -> emb_phone(Linear); pitch -> emb_pitch(get_rows); sum ×√H -> LeakyReLU(0.1);
//   n_layers layers [relative-position multi-head self-attention (window=10) + Post-LayerNorm + FFN(k3,pad1,ReLU)];
//   finally ×x_mask -> proj(1×1) -> stats ×x_mask -> split m_p/logs_p.
//
// Internal layout: main flow [T,C,B] (same as wn/flow/dec; reversed ne order of torch [B,C,T]);
// only emb (torch [B,T,hidden]) is [C,T,B], transposed once before entering the Encoder.
// Relative-position tables are expanded by emb_rel_k/v **inside the graph** (zero pad via pad_ext) -> gradients can flow back to the relative embedding weights.
// Attention intermediates [2T-1,T,H,B] / [T,T,H,B] (ne0 = torch last dim, matching attentions.py's
// relative->absolute / absolute->relative transforms, see implementation comments in enc_p.cpp).
//
// Weights are training-state GGUF (tensor names keep their original torch state_dict names; manually assembled POD, no init helper, see Q13).
#pragma once

#include "model/model.h"

#include "traincpp/traincpp.h"

#include <cstdint>
#include <string>
#include <vector>

namespace rvc_train {
namespace model {

struct EncPConfig {
    std::string prefix = "enc_p";   // parameter name prefix (torch key name)
    int64_t in_channels = 0;        // 256 (v1) / 768 (v2)
    int64_t hidden_channels = 0;    // 192
    int64_t filter_channels = 0;    // 768
    int64_t n_heads = 0;            // 2
    int64_t n_layers = 0;           // 6
    int64_t kernel_size = 3;        // FFN kernel (same-length padding)
    int64_t window_size = 10;       // relative-position window (2W+1=21)
    int64_t out_channels = 0;       // half of proj output (stats channels = 2*out)
};

// One layer (i-th item of enc_p.encoder.attn_layers/ffn_layers/norm_layers_1/norm_layers_2)
struct EncPAttnLayer {
    traincpp::Conv1d conv_q, conv_k, conv_v, conv_o;   // 1×1
    traincpp::Tensor* emb_rel_k = nullptr;             // [Dk, 2W+1, 1]
    traincpp::Tensor* emb_rel_v = nullptr;             // [Dk, 2W+1, 1]
    traincpp::LayerNorm norm1, norm2;                  // gamma/beta
    traincpp::Conv1d ffn_conv1, ffn_conv2;             // k=3, pad 1
};

struct EncPModule {
    EncPConfig cfg;
    traincpp::Linear emb_phone;          // [in,hidden]
    traincpp::Tensor* emb_pitch = nullptr;  // [hidden,256] (get_rows table; required for f0 models)
    std::vector<EncPAttnLayer> layers;   // n_layers entries
    traincpp::Conv1d proj;               // 1×1: hidden -> 2*out
};

// Assemble by name (fills POD only, tensor names unchanged); returns false and writes err on failure.
bool enc_p_load(const ModelParams& params, const EncPConfig& cfg, EncPModule* out,
                std::string* err);

// Forward captures (for comparison; pass nullptr if not needed). Tensor layout (golden ne reversed):
//   emb [C,T,B] (torch [B,T,C]); x/q/k/v/attn_y/norm1/ffn/norm2/encoder_out [T,C,B] (torch [B,C,T]);
//   rel_logits [2T-1,T,H,B]; scores_local/p_attn [T,T,H,B]; proj [T,2*out,B]
struct EncPCaptures {
    traincpp::Tensor* emb = nullptr;
    traincpp::Tensor* x = nullptr;   // Encoder input (emb × x_mask) [T,C,B]
    std::vector<traincpp::Tensor*> q, k, v;
    std::vector<traincpp::Tensor*> scores;         // after add(scores_local), before mask [T,T,H,B]
    std::vector<traincpp::Tensor*> scores_masked;  // after replicating masked_fill [T,T,H,B]
    std::vector<traincpp::Tensor*> attn_out;       // before conv_o (p@v + rel_w@rel_v) [Dk,T,H,B]
    std::vector<traincpp::Tensor*> merged;         // after head merge [T,C,B]
    std::vector<traincpp::Tensor*> rel_logits;
    std::vector<traincpp::Tensor*> scores_local;
    std::vector<traincpp::Tensor*> p_attn;
    std::vector<traincpp::Tensor*> attn_y;
    std::vector<traincpp::Tensor*> norm1;
    std::vector<traincpp::Tensor*> ffn;
    std::vector<traincpp::Tensor*> norm2;
    traincpp::Tensor* encoder_out = nullptr;
    traincpp::Tensor* proj = nullptr;
};

struct EncPOutputs {
    traincpp::Tensor* m_p = nullptr;     // first half of stats ([out,T,B] view)
    traincpp::Tensor* logs_p = nullptr;  // second half of stats ([out,T,B] view)
};

// Inputs:
//   phone   [Cin,T,B] F32 (HuBERT features; must be contiguous)
//   pitch   [T,B] I32 (f0 models; may be nullptr = non-f0)
//   x_mask  [T,1,B] F32
//   attn_valid [T,T,1,B] F32 (0/1, filled by the caller after buffer allocation; invalid positions replicate the torch
//              `masked_fill(mask==0, -1e4)` **replace** semantics: scores·M + (M−1)·1e4)
// Output: m_p/logs_p (channel-split views of stats). Returns empty on failure.
EncPOutputs enc_p_forward(traincpp::Context* ctx, const EncPModule& m, traincpp::Tensor* phone,
                          traincpp::Tensor* pitch, traincpp::Tensor* x_mask,
                          traincpp::Tensor* attn_valid, EncPCaptures* caps = nullptr);

} // namespace model
} // namespace rvc_train
