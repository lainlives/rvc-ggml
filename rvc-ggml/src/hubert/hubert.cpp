// hubert.cpp
// HuBERT 特征提取图实现�?
#include "hubert/hubert.h"

#include "vits/modules.h"
#include "engine.h"
#include "utils.h"

#include <cmath>
#include <cstring>
#include <stdexcept>

namespace rvc {
namespace hubert {

using rvc::vits::layer_norm;
using rvc::vits::linear;
using rvc::vits::to_f32;
using rvc::vits::mul_checked;
using rvc::vits::add_checked;

namespace {

// 调试：中间张量收集（RVC_HUBERT_DUMP=1 时由 build_graph 填充）
static std::vector<DebugTensor> g_debug;

constexpr float kLayerNormEps = 1e-5f;

ggml_tensor* require(const GgufModel& model, const std::string& name) {
    ggml_tensor* t = model.weight(name);
    if (!t) {
        throw std::runtime_error("HuBERT 缺少张量: " + name);
    }
    return t;
}

// 带元素数校验的 reshape（定位断言）
ggml_tensor* resh(ggml_context* ctx, ggml_tensor* x, int64_t n0, int64_t n1, int64_t n2 = 1,
    const char* where = "") {
    const int64_t want = n0 * n1 * n2;
    const int64_t have = ggml_nelements(x);
    if (want != have) {
        char buf[256];
        snprintf(buf, sizeof(buf), "reshape(%s): %lld != %lld (tensor ne=[%lld,%lld,%lld,%lld])",
            where, static_cast<long long>(want), static_cast<long long>(have),
            static_cast<long long>(x->ne[0]), static_cast<long long>(x->ne[1]),
            static_cast<long long>(x->ne[2]), static_cast<long long>(x->ne[3]));
        throw std::runtime_error(buf);
    }
    if (n2 <= 1) {
        // 保持维度待定：按元素数推断 ne2
        int64_t nn1 = n1 > 0 ? n1 : 1;
        int64_t nn2 = want / (n0 * nn1);
        return nn2 > 1 ? ggml_reshape_3d(ctx, x, n0, nn1, nn2)
            : ggml_reshape_2d(ctx, x, n0, nn1);
    }
    return ggml_reshape_3d(ctx, x, n0, n1, n2);
}

// [T, C, B] -> �?dim0 两侧补零
ggml_tensor* pad1d(ggml_context* ctx, ggml_tensor* x, int left, int right) {
    const int64_t C = x->ne[1];
    const int64_t B = x->ne[2];
    if (left > 0) {
        ggml_tensor* z = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, left, C, B);
        ggml_set_name(z, "pad_l");
        x = ggml_concat(ctx, z, x, 0);
    }
    if (right > 0) {
        ggml_tensor* z = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, right, C, B);
        ggml_set_name(z, "pad_r");
        x = ggml_concat(ctx, x, z, 0);
    }
    return x;
}

// �?[T, C, B] 裁剪 dim0 �?[start, start+len)
ggml_tensor* slice_dim0(ggml_context* ctx, ggml_tensor* x, int64_t start, int64_t len) {
    return ggml_view_3d(ctx, x, len, x->ne[1], x->ne[2], x->nb[1], x->nb[2],
        static_cast<size_t>(start) * x->nb[0]);
}

// 卷积特征提取器：input [T, 1, 1] -> 输出 [T_tok, 512, 1]
ggml_tensor* feature_extractor(ggml_context* ctx, ggml_tensor* input,
    const GgufModel& model, const Config& cfg) {
    ggml_tensor* x = input;
    const int n = static_cast<int>(cfg.conv_dim.size());
    for (int i = 0; i < n; ++i) {
        const int s = cfg.conv_stride[i];
        const std::string prefix = "feature_extractor.conv_layers." + std::to_string(i);

        // conv_1d 要求 input [T, C_in, B]；flat 布局（rvc.conv_flat）时权重为展平 [I*K, O]
        ggml_tensor* w = require(model, prefix + ".0.weight");
        x = rvc::vits::conv1d(ctx, x, w, nullptr, s, 0, 1, cfg.conv_flat);   // [T_out, C, 1]
        const int64_t T_out = x->ne[0];
        const int64_t C = x->ne[1];

        if (i == 0) {
            // fairseq 结构：Conv(.0) -> Dropout(.1) -> Fp32GroupNorm(512,512, affine)(.2) -> GELU(.3)
            // GroupNorm(512,512) 即逐通道在时间轴归一化：reshape [T,1,C,1] 后 groups=C
            x = ggml_reshape_4d(ctx, x, T_out, 1, C, 1);
            x = ggml_group_norm(ctx, x, static_cast<int>(C), 1e-5f);
            x = mul_checked(ctx, x, ggml_reshape_4d(ctx,
                to_f32(ctx, require(model, prefix + ".2.weight")), 1, 1, C, 1));
            x = add_checked(ctx, x, ggml_reshape_4d(ctx,
                to_f32(ctx, require(model, prefix + ".2.bias")), 1, 1, C, 1));
            x = ggml_reshape_3d(ctx, x, T_out, C, 1);
        }
        // 1-6 层：Conv(.0) -> Dropout(.1) -> GELU(.2)，无 GroupNorm

        // fairseq GELU(approximate='none') = erf 精确版
        x = ggml_gelu_erf(ctx, x);
    }
    return x;   // [T_tok, 512, 1]
}

// Transformer Encoder Layer（fairseq 默认结构：Pre-Norm Attention + Pre-Norm FFN）
// x: [hidden, T]（内存序 (o,t)，o = h*D+d）
//
// 注意力用"逐头 matmul"（H=12 次小乘，代价可忽略），布局推导：
//   q/k/v [hidden,T] -> 头切片 q_h/k_h/v_h = view2d(D,T, nb1, h*D*4) + cont  -> [D,T]
//   sc_h = (k_h)^T @ q_h = mul_mat(k_h, q_h)          -> [Tk, Tq]（ne0=tk，softmax 轴正确）
//   out_h = (v_h)^T @ softmax(scale*sc_h) = mul_mat(v_h, sc_h) -> [D, Tq]
//   concat 各头沿 dim0 -> [D*H, Tq]，块序即 hidden 索引 o = h*D+d ✓
ggml_tensor* encoder_layer(ggml_context* ctx, ggml_tensor* x,
    const GgufModel& model, const Config& cfg, int layer_index) {
    const std::string prefix = "encoder.layers." + std::to_string(layer_index);
    const int64_t H = cfg.heads;
    const int64_t D = cfg.hidden / cfg.heads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));
    const int64_t T = x->ne[1];
    const int64_t hidden = x->ne[0];

    // fairseq layer_norm_first=False（Post-Norm）：
    //   residual = x; x = attn(x); x = residual + x; x = self_attn_layer_norm(x);
    //   residual = x; x = fc2(gelu(fc1(x))); x = residual + x; x = final_layer_norm(x)
    ggml_tensor* residual = to_f32(ctx, x);   // [hidden, T]

    // ---- 自注意力（无 Pre-Norm）----
    auto project = [&](const char* w, const char* b) {
        return linear(ctx, residual, require(model, prefix + "." + w), require(model, prefix + "." + b));
    };
    ggml_tensor* q = project("self_attn.q_proj.weight", "self_attn.q_proj.bias");   // [hidden, T]
    ggml_tensor* k = project("self_attn.k_proj.weight", "self_attn.k_proj.bias");
    ggml_tensor* v = project("self_attn.v_proj.weight", "self_attn.v_proj.bias");

    auto layer_dump = [&](const char* tag, ggml_tensor* t) {
        if (layer_index == 0 && getenv("RVC_HUBERT_DUMP")) {
            g_debug.push_back({tag, t});
        }
    };
    layer_dump("L0_q", q);
    layer_dump("L0_k", k);
    layer_dump("L0_v", v);

    ggml_tensor* out_cat = nullptr;
    for (int64_t h = 0; h < H; ++h) {
        const size_t head_off = static_cast<size_t>(h * D) * sizeof(float);
        ggml_tensor* q_h = ggml_cont(ctx, ggml_view_2d(ctx, q, D, T, q->nb[1], head_off));
        ggml_tensor* k_h = ggml_cont(ctx, ggml_view_2d(ctx, k, D, T, k->nb[1], head_off));
        ggml_tensor* v_h = ggml_cont(ctx, ggml_view_2d(ctx, v, D, T, v->nb[1], head_off));

        // sc_h[tk, tq] = sum_d k_h[d,tk] * q_h[d,tq]
        ggml_tensor* sc_h = ggml_mul_mat(ctx, k_h, q_h);   // [Tk, Tq]
        sc_h = ggml_scale(ctx, sc_h, scale);
        sc_h = ggml_soft_max(ctx, sc_h);                   // 沿 ne0=tk 归一化

        // out_h[d, tq] = sum_tk v_h[d,tk] * sc_h[tk,tq]
        //   mul_mat(a,b)=a^T@b：取 a = v_h^T [Tk,D]，b = sc_h [Tk,Tq] -> [D,Tq]
        ggml_tensor* v_h_t = ggml_cont(ctx, ggml_transpose(ctx, v_h));   // [Tk, D]
        ggml_tensor* out_h = ggml_mul_mat(ctx, v_h_t, sc_h);             // [D, Tq]
        out_cat = out_cat ? ggml_concat(ctx, out_cat, out_h, 0) : out_h;
    }

    layer_dump("L0_attn_cat", out_cat);
    ggml_tensor* o = linear(ctx, out_cat,
        require(model, prefix + ".self_attn.out_proj.weight"),
        require(model, prefix + ".self_attn.out_proj.bias"));        // [hidden, T]
    layer_dump("L0_attn_out", o);
    ggml_tensor* x1 = add_checked(ctx, residual, o);
    layer_dump("L0_x1", x1);
    x1 = layer_norm(ctx, ggml_reshape_2d(ctx, x1, hidden, T),
        require(model, prefix + ".self_attn_layer_norm.weight"),
        require(model, prefix + ".self_attn_layer_norm.bias"), kLayerNormEps);

    // ---- FFN（无 Pre-Norm，Post-Norm 收尾）----
    residual = x1;
    ggml_tensor* f = linear(ctx, x1, require(model, prefix + ".fc1.weight"),
        require(model, prefix + ".fc1.bias"));
    f = ggml_gelu_erf(ctx, f);   // fairseq activation_fn='gelu'（erf 精确版）
    f = linear(ctx, f, require(model, prefix + ".fc2.weight"),
        require(model, prefix + ".fc2.bias"));
    ggml_tensor* x2 = add_checked(ctx, residual, f);
    return layer_norm(ctx, ggml_reshape_2d(ctx, x2, hidden, T),
        require(model, prefix + ".final_layer_norm.weight"),
        require(model, prefix + ".final_layer_norm.bias"), kLayerNormEps);
}

} // namespace

const std::vector<DebugTensor>& debug_tensors() {
    return g_debug;
}

Config read_config(const GgufModel& model) {
    Config cfg;
    const std::string kp = "hubert.";
    cfg.n_layers = model.meta_u32(kp + "n_layers", cfg.n_layers);
    cfg.hidden = model.meta_u32(kp + "hidden_size", cfg.hidden);
    cfg.ffn = model.meta_u32(kp + "intermediate_size", cfg.ffn);
    cfg.heads = model.meta_u32(kp + "n_heads", cfg.heads);
    cfg.conv_pos = model.meta_u32(kp + "conv_pos", cfg.conv_pos);
    cfg.conv_pos_groups = model.meta_u32(kp + "conv_pos_groups", cfg.conv_pos_groups);
    cfg.final_dim = model.meta_u32(kp + "final_dim", cfg.final_dim);
    cfg.conv_flat = model.meta_bool("rvc.conv_flat", false);
    return cfg;
}

void build_graph(ggml_context* ctx,
    const GgufModel& model,
    const Config& cfg,
    ggml_tensor* input,
    ggml_tensor** out_proj,
    ggml_tensor** out_layer9,
    ggml_tensor** out_layer12) {

    const bool dump_debug = getenv("RVC_HUBERT_DUMP") != nullptr;
    g_debug.clear();
    auto dbg = [&](const char* tag, ggml_tensor* t) {
        if (!dump_debug) return;
        g_debug.push_back({tag, t});
        fprintf(stderr, "[hubert] %s: ne=[%lld,%lld,%lld,%lld]\n", tag,
            (long long)t->ne[0], (long long)t->ne[1], (long long)t->ne[2], (long long)t->ne[3]);
    };

    // input: [T, 1, 1]
    ggml_tensor* x = feature_extractor(ctx, input, model, cfg);      // [T_tok, 512, 1]
    dbg("conv_out", x);
    const int64_t T_tok = x->ne[0];

    // layer_norm（conv 输出层）：[T, C] -> [C, T]
    x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));            // [C=512, T]
    dbg("pre_ln", x);
    x = layer_norm(ctx, x, require(model, "layer_norm.weight"),
        require(model, "layer_norm.bias"), kLayerNormEps);
    dbg("post_ln", x);
    rvc::vits::check_dim(x, cfg.conv_dim.back(), T_tok, "hubert.layer_norm");
    // post_extract_proj：linear 期望 x 的 ne[0]=K（512），当前 x=[C,T] 正合适
    x = linear(ctx, x, require(model, "post_extract_proj.weight"),
        require(model, "post_extract_proj.bias"));                   // [hidden, T]
    dbg("post_proj", x);
    rvc::vits::check_dim(x, cfg.hidden, T_tok, "hubert.post_extract_proj");
    x = ggml_reshape_3d(ctx, x, cfg.hidden, T_tok, 1);               // [hidden, T, 1]

    // ---- pos_conv ----
    // 卷积输入 [T, hidden, 1]（对 hidden 维分 conv_pos_groups 组）
    ggml_tensor* xc = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));   // [T, hidden, 1]
    xc = pad1d(ctx, xc, cfg.conv_pos / 2, (cfg.conv_pos - 1) / 2);        // [T+127, hidden, 1]
    ggml_tensor* pos = rvc::vits::conv1d_grouped(ctx, xc,
        require(model, "encoder.pos_conv.0.weight"),
        require(model, "encoder.pos_conv.0.bias"),
        cfg.conv_pos_groups, 1, 0, 1);                                    // [T_out, hidden, 1]
    // 裁剪到 T（conv2d 退化路径输出长度可能多 1）
    if (pos->ne[0] > T_tok) {
        pos = ggml_view_3d(ctx, pos, T_tok, pos->ne[1], pos->ne[2],
            pos->nb[1], pos->nb[2], 0);
    }
    // fairseq make_conv_pos: Sequential(Conv1d, SamePad, GELU)，即 conv 后还有 GELU
    pos = ggml_gelu_erf(ctx, pos);
    // [T, hidden, 1] -> [hidden, T, 1]，与 x 对齐后相加
    pos = ggml_cont(ctx, ggml_permute(ctx, pos, 1, 0, 2, 3));
    dbg("pos_conv", pos);

    ggml_tensor* xe = add_checked(ctx, x, pos);                       // [hidden, T, 1]

    // ---- encoder layer_norm ----
    xe = layer_norm(ctx, ggml_reshape_2d(ctx, xe, cfg.hidden, T_tok),
        require(model, "encoder.layer_norm.weight"),
        require(model, "encoder.layer_norm.bias"), kLayerNormEps);
    dbg("encoder_input", xe);

    // ---- 12 层 ----
    ggml_tensor* layer9 = nullptr;
    ggml_tensor* layer12 = nullptr;
    for (int i = 0; i < cfg.n_layers; ++i) {
        xe = encoder_layer(ctx, xe, model, cfg, i);
        if (i == 0 || i == 5 || i == 11) {
            std::string tag = "encoder_layer_" + std::to_string(i);
            dbg(tag.c_str(), xe);
        }
        // fairseq output_layer 为 1-based：RVC v1 传 9 -> 第 9 层（索引 8）；
        // v2 传 12 -> 最后一层（索引 11）
        if (i == 8) layer9 = xe;
        if (i == cfg.n_layers - 1) layer12 = xe;
    }

    // 输出布局：[C, T]（C 在 ne0）。ggml 连续内存 i0 + ne0*i1 与 PyTorch 行主序
    // [T, C] 的 t*C + c 完全一致，因此 raw dump 后按 (T, C) 读取即为参考布局。
    if (layer9) *out_layer9 = layer9;        // [768, T]
    if (layer12) *out_layer12 = layer12;     // [768, T]

    // final_proj：linear 期望 x 为 [K=hidden, T]；layer9 本身即该布局
    *out_proj = linear(ctx, layer9,
        require(model, "final_proj.weight"), require(model, "final_proj.bias"));  // [256, T]
}

// ------------------------------------------------------------
// 可复用编码器
// ------------------------------------------------------------

struct HubertEncoder::Impl {
    const GgufModel& model;
    ggml_backend_t backend = nullptr;
    std::unique_ptr<GgmlGraphRunner> runner;
    int64_t built_samples = -1;
    ggml_tensor* in = nullptr;
    ggml_tensor* out_proj = nullptr;
    ggml_tensor* out_l9 = nullptr;
    ggml_tensor* out_l12 = nullptr;
};

HubertEncoder::HubertEncoder(const GgufModel& model, ggml_backend_t backend)
    : impl_(std::make_unique<Impl>(Impl{ model, backend })) {
    cfg_ = read_config(model);
}

HubertEncoder::~HubertEncoder() = default;

void HubertEncoder::build(int64_t samples) {
    impl_->runner = std::make_unique<GgmlGraphRunner>(impl_->backend,
        1024ull * 1024 * 1024, 32768);
    ggml_context* ctx = impl_->runner->ctx();
    impl_->in = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, samples, 1, 1);
    ggml_set_name(impl_->in, "hubert.audio");
    ggml_set_input(impl_->in);

    build_graph(ctx, impl_->model, cfg_, impl_->in,
        &impl_->out_proj, &impl_->out_l9, &impl_->out_l12);
    for (ggml_tensor* t : { impl_->out_proj, impl_->out_l9, impl_->out_l12 }) {
        ggml_set_output(t);
        impl_->runner->expand(t);
    }
    if (!impl_->runner->allocate()) {
        throw std::runtime_error("HuBERT 图缓冲分配失败");
    }
    impl_->built_samples = samples;
}

Features HubertEncoder::extract(const std::vector<float>& audio_16k, bool v2) {
    if (audio_16k.empty()) {
        throw std::runtime_error("HuBERT 输入为空");
    }
    const int64_t samples = static_cast<int64_t>(audio_16k.size());
    if (impl_->built_samples != samples) {
        build(samples);
    }
    impl_->runner->set_input_vec(impl_->in, audio_16k);
    impl_->runner->compute();

    ggml_tensor* t = v2 ? impl_->out_l12 : impl_->out_proj;
    Features f;
    f.dim = t->ne[0];
    f.frames = t->ne[1];
    impl_->runner->get_output_vec(t, f.values);
    return f;
}

} // namespace hubert
} // namespace rvc
