// RVC-Train-GGML - 说话人嵌入 emb_g 实现（src/model/emb.cpp）
#include "model/emb.h"

#include <cstdio>

namespace rvc_train {
namespace model {

traincpp::Tensor* emb_g_forward(traincpp::Context* ctx, const ModelParams& params,
                                traincpp::Tensor* sid, std::string* err) {
    using namespace traincpp;
    Tensor* weight = params.find("emb_g.weight");
    if (weight == nullptr) {
        if (err != nullptr) {
            *err = "缺失参数: emb_g.weight";
        }
        return nullptr;
    }
    if (sid == nullptr) {
        if (err != nullptr) {
            *err = "emb_g 输入 sid 为空";
        }
        return nullptr;
    }
    // 手工装配（避免 init helper 改写张量名；层结构是 POD）
    Embedding layer;
    layer.weight         = weight;
    layer.embedding_dim  = weight->ne[0];
    layer.num_embeddings = weight->ne[1];
    std::snprintf(layer.name, MAX_NAME, "%s", "emb_g");
    return embedding_forward(ctx, &layer, sid);
}

} // namespace model
} // namespace rvc_train
