// RVC-Train-GGML - 判别器（MPD）实现（src/model/discriminator.cpp）
#include "model/discriminator.h"

#include <cstdio>

namespace rvc_train {
namespace model {

namespace {

using namespace traincpp;

Tensor* require_param(const ModelParams& params, const std::string& name, std::string* err) {
    Tensor* t = params.find(name.c_str());
    if (t == nullptr && err != nullptr) {
        *err = "缺失参数: " + name;
    }
    return t;
}

// 手工装配 Conv1d（含 groups；避免 init helper 改写张量名）
void wire_conv1d(Conv1d* layer, Tensor* v, Tensor* bias, int64_t stride, int64_t padding,
                 int64_t groups, const char* name) {
    layer->weight       = v;
    layer->bias         = bias;
    layer->has_bias     = bias != nullptr;
    layer->kernel_size  = v->ne[0];
    layer->in_channels  = v->ne[1] * groups;
    layer->out_channels = v->ne[2];
    layer->stride       = stride;
    layer->padding      = padding;
    layer->dilation     = 1;
    layer->groups       = groups;
    std::snprintf(layer->name, MAX_NAME, "%s", name);
}

// 手工装配 Conv2d（kernel_w/stride_w/padding_w 固定 1/1/0，对齐 DiscriminatorP）
void wire_conv2d(Conv2d* layer, Tensor* v, Tensor* bias, int64_t stride_h, int64_t padding_h,
                 const char* name) {
    layer->weight       = v;
    layer->bias         = bias;
    layer->has_bias     = bias != nullptr;
    layer->kernel_w     = v->ne[0];
    layer->kernel_h     = v->ne[1];
    layer->in_channels  = v->ne[2];
    layer->out_channels = v->ne[3];
    layer->stride_w     = 1;
    layer->stride_h     = stride_h;
    layer->padding_w    = 0;
    layer->padding_h    = padding_h;
    layer->dilation_w   = 1;
    layer->dilation_h   = 1;
    std::snprintf(layer->name, MAX_NAME, "%s", name);
}

// DiscriminatorS 固定结构（models.py:1135-1145）：stride/groups 与权重形状正交，按层序固定
constexpr int64_t kSStrides[6] = {1, 4, 4, 4, 4, 1};
constexpr int64_t kSGroups[6]  = {1, 4, 16, 64, 256, 1};

// 折叠 weight_norm：w = g·v/sqrt(Σv²+eps)。
// conv_post 的输出通道为 1（torch weight_g=[1,1,1]），GGUF 里 v/g 的尾部维被压缩成全 1，
// train.cpp `weightnorm_forward` 的 n_dims(g)==n_dims(v) 校验不适用 → 这里按**同一公式**
// （eps=1e-12 在根号内，与库内 kWeightNormEps 一致）把 v 展平成 [rest, oc] 折叠。
Tensor* fold_wn_weight(Context* ctx, Tensor* v, Tensor* g, int64_t oc) {
    Tensor* vf  = reshape_2d(ctx, cont(ctx, v), tensor_nelements(v) / oc, oc);
    Tensor* n2  = sum_rows(ctx, mul(ctx, vf, vf));   // [1, oc, 1, 1]
    Tensor* n2b = reshape_2d(ctx, n2, 1, oc);
    Tensor* e   = new_tensor_1d(ctx, TYPE_F32, 1);
    nn_init_constant(ctx, e, 1e-12f);
    Tensor* den = sqrt_op(ctx, add1(ctx, n2b, e));   // [1, oc]
    Tensor* gf  = reshape_2d(ctx, g, 1, oc);
    return mul(ctx, div(ctx, vf, den), gf);          // [rest, oc]（可直接 reshape 回层权形状）
}

// 取 weight_g/weight_v（+可选 bias）并装入 WeightNorm POD
bool require_wn(const ModelParams& params, const std::string& base, WeightNorm* wn,
                Tensor** bias, std::string* err) {
    Tensor* g = require_param(params, base + ".weight_g", err);
    if (g == nullptr) {
        return false;
    }
    Tensor* v = require_param(params, base + ".weight_v", err);
    if (v == nullptr) {
        return false;
    }
    wn->g = g;
    wn->v = v;
    *bias = params.find((base + ".bias").c_str());
    return true;
}

DiscForward disc_forward_s(Context* ctx, const DiscSModule& m, Tensor* x) {
    DiscForward f;
    f.fmap.reserve(7);
    Tensor* y = x;
    for (int i = 0; i < 6; ++i) {
        Tensor* w2 = fold_wn_weight(ctx, m.conv_wn[i].v, m.conv_wn[i].g, m.convs[i].out_channels);
        Tensor* w  = reshape_3d(ctx, w2, m.convs[i].kernel_size,
                                m.convs[i].in_channels / m.convs[i].groups, m.convs[i].out_channels);
        y = conv1d_forward_weight(ctx, &m.convs[i], w, y);
        y = leaky_relu(ctx, y, kMpdLreluSlope);
        f.fmap.push_back(y);
    }
    Tensor* w2 = fold_wn_weight(ctx, m.post_wn.v, m.post_wn.g, m.conv_post.out_channels);
    Tensor* w  = reshape_3d(ctx, w2, m.conv_post.kernel_size, m.conv_post.in_channels,
                            m.conv_post.out_channels);
    y = conv1d_forward_weight(ctx, &m.conv_post, w, y);
    f.fmap.push_back(y);
    // flatten(1,-1)：conv_post 输出通道 = 1 → [N,B] 无行序歧义
    const int64_t B = y->ne[2];
    f.out = reshape_2d(ctx, cont(ctx, y), tensor_nelements(y) / B, B);
    return f;
}

DiscForward disc_forward_p(Context* ctx, const DiscPModule& m, Tensor* x) {
    DiscForward f;
    f.fmap.reserve(6);
    const int64_t T = x->ne[0];
    const int64_t B = x->ne[2];

    // period 折叠：不足整周期时 reflect 右填充（torch `F.pad(x,(0,n_pad),"reflect")`）
    int64_t t2 = T;
    Tensor* xt = x;
    const int64_t rem = T % m.period;
    if (rem != 0) {
        const int64_t n_pad = m.period - rem;
        xt = pad_ext(ctx, x, 0, (int) n_pad, 0, 0, 0, 0, 0, 0, PAD_REFLECT);
        t2 = T + n_pad;
    }
    Tensor* y = reshape_4d(ctx, cont(ctx, xt), m.period, t2 / m.period, 1, B);

    for (int i = 0; i < 5; ++i) {
        Tensor* w2 = fold_wn_weight(ctx, m.conv_wn[i].v, m.conv_wn[i].g, m.convs[i].out_channels);
        Tensor* w  = reshape_4d(ctx, w2, m.convs[i].kernel_w, m.convs[i].kernel_h,
                                m.convs[i].in_channels, m.convs[i].out_channels);
        y = conv2d_forward_weight(ctx, &m.convs[i], w, y);
        y = leaky_relu(ctx, y, kMpdLreluSlope);
        f.fmap.push_back(y);
    }
    Tensor* w2 = fold_wn_weight(ctx, m.post_wn.v, m.post_wn.g, m.conv_post.out_channels);
    Tensor* w  = reshape_4d(ctx, w2, m.conv_post.kernel_w, m.conv_post.kernel_h,
                            m.conv_post.in_channels, m.conv_post.out_channels);
    y = conv2d_forward_weight(ctx, &m.conv_post, w, y);
    f.fmap.push_back(y);
    const int64_t B2 = y->ne[3];
    f.out = reshape_2d(ctx, cont(ctx, y), tensor_nelements(y) / B2, B2);
    return f;
}

} // namespace

bool mpd_load(const ModelParams& params, const MpdConfig& cfg, MpdModule* out, std::string* err) {
    if (out == nullptr) {
        if (err != nullptr) {
            *err = "MPD 输出为空";
        }
        return false;
    }
    if (cfg.periods.empty()) {
        if (err != nullptr) {
            *err = "MPD 配置非法（periods 为空）";
        }
        return false;
    }

    MpdModule m;
    m.cfg = cfg;

    // DiscriminatorS（discriminators.0）
    for (int i = 0; i < 6; ++i) {
        const std::string base = cfg.prefix + ".0.convs." + std::to_string(i);
        Tensor* bias = nullptr;
        if (!require_wn(params, base, &m.s.conv_wn[i], &bias, err)) {
            return false;
        }
        Tensor* v = m.s.conv_wn[i].v;
        if (v->ne[2] % kSGroups[i] != 0) {
            if (err != nullptr) {
                *err = "DiscriminatorS 第 " + std::to_string(i) + " 层输出通道不能被 groups 整除";
            }
            return false;
        }
        wire_conv1d(&m.s.convs[i], v, bias, kSStrides[i], v->ne[0] / 2, kSGroups[i],
                    "mpd.s.convs");
    }
    {
        const std::string base = cfg.prefix + ".0.conv_post";
        Tensor* bias = nullptr;
        if (!require_wn(params, base, &m.s.post_wn, &bias, err)) {
            return false;
        }
        Tensor* v = m.s.post_wn.v;
        wire_conv1d(&m.s.conv_post, v, bias, 1, v->ne[0] / 2, 1, "mpd.s.conv_post");
    }

    // DiscriminatorP（discriminators.1..n，与 cfg.periods 顺序一致）
    m.ps.resize(cfg.periods.size());
    for (size_t d = 0; d < cfg.periods.size(); ++d) {
        DiscPModule& p = m.ps[d];
        p.period = cfg.periods[d];
        const std::string dbase = cfg.prefix + "." + std::to_string(d + 1);
        for (int i = 0; i < 5; ++i) {
            const std::string base = dbase + ".convs." + std::to_string(i);
            Tensor* bias = nullptr;
            if (!require_wn(params, base, &p.conv_wn[i], &bias, err)) {
                return false;
            }
            Tensor* v = p.conv_wn[i].v;
            if (v->ne[0] != 1 || v->ne[1] != cfg.kernel_size) {
                if (err != nullptr) {
                    *err = "DiscriminatorP 卷积核形状不符（期望 [1," + std::to_string(cfg.kernel_size) +
                           "]）: " + base + ".weight_v";
                }
                return false;
            }
            // 前 4 层 stride=3、第 5 层 stride=1（models.py:1205-1213）
            const int64_t stride_h = (i < 4) ? cfg.stride : 1;
            wire_conv2d(&p.convs[i], v, bias, stride_h, v->ne[1] / 2, "mpd.p.convs");
        }
        {
            const std::string base = dbase + ".conv_post";
            Tensor* bias = nullptr;
            if (!require_wn(params, base, &p.post_wn, &bias, err)) {
                return false;
            }
            Tensor* v = p.post_wn.v;
            wire_conv2d(&p.conv_post, v, bias, 1, v->ne[1] / 2, "mpd.p.conv_post");
        }
    }

    *out = std::move(m);
    return true;
}

MpdForward mpd_forward(Context* ctx, const MpdModule& m, Tensor* wave, Tensor* wave_hat) {
    MpdForward out;
    if (ctx == nullptr || wave == nullptr || wave_hat == nullptr) {
        std::printf("[mpd] 输入为空\n");
        return out;
    }
    if (wave->ne[0] != wave_hat->ne[0] || wave->ne[2] != wave_hat->ne[2] ||
        wave->ne[1] != 1 || wave_hat->ne[1] != 1) {
        std::printf("[mpd] 输入形状不符: wave ne=[%lld,%lld,%lld] wave_hat ne=[%lld,%lld,%lld]\n",
                    (long long) wave->ne[0], (long long) wave->ne[1], (long long) wave->ne[2],
                    (long long) wave_hat->ne[0], (long long) wave_hat->ne[1],
                    (long long) wave_hat->ne[2]);
        return out;
    }
    if (wave->ne[1] != m.s.convs[0].in_channels) {
        std::printf("[mpd] 波形通道 %lld != DiscriminatorS in_channels %lld\n",
                    (long long) wave->ne[1], (long long) m.s.convs[0].in_channels);
        return out;
    }

    out.r.reserve(1 + m.ps.size());
    out.g.reserve(1 + m.ps.size());
    out.r.push_back(disc_forward_s(ctx, m.s, wave));
    out.g.push_back(disc_forward_s(ctx, m.s, wave_hat));
    for (const DiscPModule& p : m.ps) {
        out.r.push_back(disc_forward_p(ctx, p, wave));
        out.g.push_back(disc_forward_p(ctx, p, wave_hat));
    }
    return out;
}

} // namespace model
} // namespace rvc_train
