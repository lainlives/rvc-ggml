// RVC-Train-GGML - 说话人嵌入 emb_g（src/model/emb.h）
//
// g = emb_g(sid)：按 sid 查表（torch Embedding）。
#pragma once

#include "model/model.h"

#include "traincpp/traincpp.h"

#include <string>

namespace rvc_train {
namespace model {

// 前向：sid 为 I32 [B]（1~3 维），权重 emb_g.weight ne=[dim, n_spk]（torch [n_spk, dim]）；
// 输出 ne=[dim, B, ...]。参数缺失时返回 nullptr 并写入 err。
traincpp::Tensor* emb_g_forward(traincpp::Context* ctx, const ModelParams& params,
                                traincpp::Tensor* sid, std::string* err);

} // namespace model
} // namespace rvc_train
