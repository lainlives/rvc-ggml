// RVC-Train-GGML - M7-2 random init from scratch: tensor spec generation + initialization (src/model/scratch.h)
//
// Purpose: without depending on any .pth / training-state GGUF, generate from ModelConfig in a **purely config-driven** way
// all parameter (name, ne) specs of the RVC generator G (enc_p/enc_q/flow/dec/emb_g) and discriminator D (MPD: S + P×len(d_periods)),
// and fill F32 data using PyTorch default initialization.
//
// Spec correctness is verified against the M1 fixture hash manifests: v1-40k G = 560 tensors, v1 D = 129 tensors
// (work/models/f0G40k.hashes.txt / f0D40k.hashes.txt; --selftest-scratch compares item by item).
//
// Initialization semantics (PyTorch defaults; R-M7-1 has recorded the risk of "not fully consistent with RVC init_weights"):
//   - Conv/Linear/weight_v: kaiming_uniform(a=√5) -> U(-1/√fan_in, +1/√fan_in)
//   - bias: U(-1/√fan_in, +1/√fan_in)
//   - Embedding (emb_g/emb_pitch): N(0,1)
//   - relative-position emb_rel_k/v: N(0, 1/√dk) (dk = hidden/n_heads)
//   - LayerNorm gamma=1, beta=0
//   - weight_g = ‖weight_v‖ (sum over all dims except the last) -> folded w = weight_v
#pragma once

#include "model/config.h"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace rvc_train {
namespace model {

enum class InitKind {
    Kaiming,    // U(-1/√fan_in, +1/√fan_in)
    Bias,       // U(-1/√fan_in, +1/√fan_in)
    Embedding,  // N(0,1)
    RelEmb,     // N(0, 1/√ne0)
    Ones,       // constant 1 (LayerNorm gamma)
    Zeros,      // constant 0 (LayerNorm beta)
    WnG,        // weight_g: computed from the paired weight_v (g = ‖v‖, last dim is channels)
};

// Spec of a single parameter tensor; ne is in ggml order (torch shape reversed), padded to 4 dims with 1.
struct TensorSpec {
    std::string name;
    std::array<int64_t, 4> ne{1, 1, 1, 1};
    InitKind init = InitKind::Kaiming;
    int64_t fan_in = 1;  // used by Kaiming/Bias; ignored otherwise
    int ch_dim = 2;      // weight_norm normalization/output dim (Conv1d/ConvT=2, Conv2d=3)
};

// Generate all specs for G (enc_p/enc_q/flow/dec/emb_g). Writes err on failure.
bool build_g_specs(const ModelConfig& mc, std::vector<TensorSpec>* out, std::string* err);
// Generate all specs for D (discriminators.*) (S + P×len(d_periods)). Writes err on failure.
bool build_d_specs(const ModelConfig& mc, std::vector<TensorSpec>* out, std::string* err);

// Generate deterministic F32 data per spec (same length as specs; weight_g computed from the paired weight_v).
bool fill_specs(const std::vector<TensorSpec>& specs, uint64_t seed,
                std::vector<std::vector<float>>* out, std::string* err);

} // namespace model
} // namespace rvc_train
