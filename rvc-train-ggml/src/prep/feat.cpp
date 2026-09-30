// feat.cpp —— M4-3 HuBERT 特征实现（自建执行器；修正 RVC-GGML pad1d 未初始化零张量问题，Q22）
//
// 结构照 RVC-GGML `src/hubert/hubert.cpp::HubertEncoder`（按长度缓存图、输出三选一），
// 但图构建/分配改用 rvc_core 公共 API，并增加 `zero_pads()`：每次 compute 前把图中
// **未初始化的零常量叶子**（RVC-GGML pad1d 的 pad_l/pad_r）显式清零——其原实现依赖
// "新缓冲为零"，重复 compute 后被 gallocr 复用缓冲污染 → 静默漂移（详见 feat.h 与 Q22 报告）。

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
    std::vector<ggml_tensor*> pad_tensors;  // 图内 "pad_*" 零常量叶子（每次 compute 前清零）
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

    // 收集"未初始化零常量叶子"（在 allocate 前：此 ctx 为 no_alloc，所有图内新建张量
    // buffer==nullptr；模型权重已在自己 buffer 中）。RVC-GGML pad1d 的 pad_l/pad_r 是
    // 其中**会被读取**的两个；按 src 遍历即可发现，且不依赖张量名（ggml 会把叶子名改写成
    // "leaf_N"）。im2col 的形状占位张量也会被收集，但清零它无副作用（只读形状）。
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
                     "[rvc-prep] 警告：HuBERT 图未发现未初始化零常量叶子（RVC-GGML 实现可能已变化）\n");
    }
    if (!im.runner->allocate()) {
        throw std::runtime_error("HuBERT 图缓冲分配失败");
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
