// feat.cpp -- M4-3 HuBERT feature implementation (self-built executor; fixes the RVC-GGML pad1d uninitialized zero-tensor issue, Q22)
//
// Structure mirrors RVC-GGML `src/hubert/hubert.cpp::HubertEncoder` (graph cached by length, one of three outputs),
// but graph build/allocation uses the rvc_core public API, plus a `zero_pads()` addition: before every compute,
// the **uninitialized zero-constant leaves** in the graph (pad_l/pad_r of RVC-GGML pad1d) are explicitly zeroed -- the
// original implementation relies on "fresh buffers are zero", and after repeated computes the gallocr-reused buffers pollute them -> silent drift (see feat.h and the Q22 report).

#include "prep/feat.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>

#include "engine.h"

namespace rvc_train::prep {

struct HubertFeatureExtractor::Impl {
    const GgufModel* model = nullptr;
    ggml_backend_t backend = nullptr;
    rvc::hubert::Config cfg;
    std::unique_ptr<rvc::GgmlGraphRunner> runner;
    int64_t built_samples = -1;
    ggml_tensor* in = nullptr;
    ggml_tensor* out_proj = nullptr;
    ggml_tensor* out_l9 = nullptr;
    ggml_tensor* out_l12 = nullptr;
    std::vector<ggml_tensor*> pad_tensors;  // "pad_*" zero-constant leaves in the graph (zeroed before every compute)
};

HubertFeatureExtractor::HubertFeatureExtractor(const GgufModel& model, ggml_backend_t backend)
    : impl_(std::make_unique<Impl>()) {
    impl_->model = &model;
    impl_->backend = backend;
    impl_->cfg = rvc::hubert::read_config(model);
}

HubertFeatureExtractor::~HubertFeatureExtractor() = default;

const rvc::hubert::Config& HubertFeatureExtractor::config() const { return impl_->cfg; }

void HubertFeatureExtractor::build(int64_t samples) {
    Impl& im = *impl_;
    im.runner = std::make_unique<rvc::GgmlGraphRunner>(im.backend, 1024ull * 1024 * 1024, 32768);
    ggml_context* ctx = im.runner->ctx();
    im.in = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, samples, 1, 1);
    ggml_set_name(im.in, "hubert.audio");
    ggml_set_input(im.in);

    rvc::hubert::build_graph(ctx, *im.model, im.cfg, im.in, &im.out_proj, &im.out_l9, &im.out_l12);
    for (ggml_tensor* t : { im.out_proj, im.out_l9, im.out_l12 }) {
        ggml_set_output(t);
        im.runner->expand(t);
    }

    // Collect the "uninitialized zero-constant leaves" (before allocate: this ctx is no_alloc, so all tensors
    // created in-graph have buffer==nullptr; model weights already live in their own buffer). RVC-GGML pad1d's pad_l/pad_r are
    // two of them that **get read**; walking src finds them and does not depend on tensor names (ggml rewrites leaf names
    // to "leaf_N"). im2col's shape-placeholder tensors are also collected, but zeroing them has no side effects (shape is read-only).
    im.pad_tensors.clear();
    ggml_cgraph* g = im.runner->graph();
    const int n_nodes = ggml_graph_n_nodes(g);
    for (int i = 0; i < n_nodes; ++i) {
        ggml_tensor* node = ggml_graph_node(g, i);
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            ggml_tensor* t = node->src[s];
            if (t == nullptr || t == im.in || t->op != GGML_OP_NONE || t->buffer != nullptr) {
                continue;
            }
            if (std::find(im.pad_tensors.begin(), im.pad_tensors.end(), t) ==
                im.pad_tensors.end()) {
                im.pad_tensors.push_back(t);
            }
        }
    }
    if (im.pad_tensors.empty()) {
        std::fprintf(stderr,
                     "[rvc-prep] warning: no uninitialized zero-constant leaves found in the HuBERT graph (the RVC-GGML implementation may have changed)\n");
    }
    if (!im.runner->allocate()) {
        throw std::runtime_error("failed to allocate HuBERT graph buffers");
    }
    im.built_samples = samples;
}

void HubertFeatureExtractor::zero_pads() {
    Impl& im = *impl_;
    std::size_t max_bytes = 0;
    for (ggml_tensor* t : im.pad_tensors) {
        max_bytes = std::max(max_bytes, ggml_nbytes(t));
    }
    if (max_bytes == 0) {
        return;
    }
    static thread_local std::vector<unsigned char> zeros;
    zeros.assign(max_bytes, 0);
    for (ggml_tensor* t : im.pad_tensors) {
        ggml_backend_tensor_set(t, zeros.data(), 0, ggml_nbytes(t));
    }
}

FeatResult HubertFeatureExtractor::extract(const std::vector<float>& audio_16k, bool v2) {
    FeatResult out;
    if (audio_16k.empty()) {
        return out;
    }
    Impl& im = *impl_;
    const int64_t samples = static_cast<int64_t>(audio_16k.size());
    if (im.built_samples != samples) {
        build(samples);
    }
    zero_pads();
    im.runner->set_input_vec(im.in, audio_16k);
    im.runner->compute();

    ggml_tensor* t = v2 ? im.out_l12 : im.out_proj;
    out.dim = t->ne[0];
    out.frames = t->ne[1];
    im.runner->get_output_vec(t, out.values);
    return out;
}

FeatResult compute_hubert_feat(const std::vector<float>& audio16k,
                               HubertFeatureExtractor& hubert, bool v2) {
    return hubert.extract(audio16k, v2);
}

} // namespace rvc_train::prep
