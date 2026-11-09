// RVC-Train-GGML - speaker embedding emb_g (src/model/emb.h)
//
// g = emb_g(sid): lookup by sid (torch Embedding).
#pragma once

#include "model/model.h"

#include "traincpp/traincpp.h"

#include <string>

namespace rvc_train {
namespace model {

// Forward: sid is I32 [B] (1-3 dims), weight emb_g.weight ne=[dim, n_spk] (torch [n_spk, dim]);
// output ne=[dim, B, ...]. Returns nullptr and writes err when the parameter is missing.
traincpp::Tensor* emb_g_forward(traincpp::Context* ctx, const ModelParams& params,
                                traincpp::Tensor* sid, std::string* err);

} // namespace model
} // namespace rvc_train
