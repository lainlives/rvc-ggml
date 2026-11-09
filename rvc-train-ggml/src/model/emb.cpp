// RVC-Train-GGML - speaker embedding emb_g implementation (src/model/emb.cpp)
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
            *err = "missing parameter: emb_g.weight";
        }
        return nullptr;
    }
    if (sid == nullptr) {
        if (err != nullptr) {
            *err = "emb_g input sid is null";
        }
        return nullptr;
    }
    // Manual assembly (avoid init helpers rewriting tensor names; the layer struct is POD)
    Embedding layer;
    layer.weight         = weight;
    layer.embedding_dim  = weight->ne[0];
    layer.num_embeddings = weight->ne[1];
    std::snprintf(layer.name, MAX_NAME, "%s", "emb_g");
    return embedding_forward(ctx, &layer, sid);
}

} // namespace model
} // namespace rvc_train
