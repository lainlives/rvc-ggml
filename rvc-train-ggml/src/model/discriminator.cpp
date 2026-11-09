// RVC-Train-GGML - discriminator (MPD) implementation (src/model/discriminator.cpp)
#include "model/discriminator.h"

#include <cstdio>

namespace rvc_train {
namespace model {

namespace {

using namespace traincpp;

Tensor* require_param(const ModelParams& params, const std::string& name, std::string* err) {
    Tensor* t = params.find(name.c_str());
    if (t == nullptr && err != nullptr) {
        *err = "missing parameter: " + name;
    }
    return t;
}

// Manual Conv1d assembly (with groups; avoid init helpers rewriting tensor names)
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

// Manual Conv2d assembly (kernel_w/stride_w/padding_w fixed at 1/1/0, matching DiscriminatorP)
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

// DiscriminatorS fixed structure (models.py:1135-1145): stride/groups are orthogonal to weight shapes, fixed by layer index
constexpr int64_t kSStrides[6] = {1, 4, 4, 4, 4, 1};
constexpr int64_t kSGroups[6]  = {1, 4, 16, 64, 256, 1};

// Fold weight_norm: w = g·v/sqrt(Σv²+eps).
// conv_post has 1 output channel (torch weight_g=[1,1,1]); in GGUF the trailing dims of v/g are collapsed to all 1s,
// so the n_dims(g)==n_dims(v) check in train.cpp `weightnorm_forward` does not apply -> here we use **the same formula**
// (eps=1e-12 inside the sqrt, same as kWeightNormEps in the library) to flatten v to [rest, oc] and fold.
Tensor* fold_wn_weight(Context* ctx, Tensor* v, Tensor* g, int64_t oc) {
    Tensor* vf  = reshape_2d(ctx, cont(ctx, v), tensor_nelements(v) / oc, oc);
    Tensor* n2  = sum_rows(ctx, mul(ctx, vf, vf));   // [1, oc, 1, 1]
    Tensor* n2b = reshape_2d(ctx, n2, 1, oc);
    Tensor* e   = new_tensor_1d(ctx, TYPE_F32, 1);
    nn_init_constant(ctx, e, 1e-12f);
    Tensor* den = sqrt_op(ctx, add1(ctx, n2b, e));   // [1, oc]
    Tensor* gf  = reshape_2d(ctx, g, 1, oc);
    return mul(ctx, div(ctx, vf, den), gf);          // [rest, oc] (can be reshaped directly back to the layer weight shape)
}

// Fetch weight_g/weight_v (+ optional bias) and put them into a WeightNorm POD
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
    // flatten(1,-1): conv_post output channels = 1 -> no row-order ambiguity in [N,B]
    const int64_t B = y->ne[2];
    f.out = reshape_2d(ctx, cont(ctx, y), tensor_nelements(y) / B, B);
    return f;
}

DiscForward disc_forward_p(Context* ctx, const DiscPModule& m, Tensor* x) {
    DiscForward f;
    f.fmap.reserve(6);
    const int64_t T = x->ne[0];
    const int64_t B = x->ne[2];

    // period folding: reflect right padding when shorter than a full period (torch `F.pad(x,(0,n_pad),"reflect")`)
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
            *err = "MPD output is null";
        }
        return false;
    }
    if (cfg.periods.empty()) {
        if (err != nullptr) {
            *err = "invalid MPD config (periods is empty)";
        }
        return false;
    }

    MpdModule m;
    m.cfg = cfg;

    // DiscriminatorS (discriminators.0)
    for (int i = 0; i < 6; ++i) {
        const std::string base = cfg.prefix + ".0.convs." + std::to_string(i);
        Tensor* bias = nullptr;
        if (!require_wn(params, base, &m.s.conv_wn[i], &bias, err)) {
            return false;
        }
        Tensor* v = m.s.conv_wn[i].v;
        if (v->ne[2] % kSGroups[i] != 0) {
            if (err != nullptr) {
                *err = "DiscriminatorS layer " + std::to_string(i) + " output channels are not divisible by groups";
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

    // DiscriminatorP (discriminators.1..n, in the same order as cfg.periods)
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
                    *err = "DiscriminatorP convolution kernel shape mismatch (expected [1," + std::to_string(cfg.kernel_size) +
                           "]): " + base + ".weight_v";
                }
                return false;
            }
            // first 4 layers stride=3, 5th layer stride=1 (models.py:1205-1213)
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
        std::printf("[mpd] null input\n");
        return out;
    }
    if (wave->ne[0] != wave_hat->ne[0] || wave->ne[2] != wave_hat->ne[2] ||
        wave->ne[1] != 1 || wave_hat->ne[1] != 1) {
        std::printf("[mpd] input shape mismatch: wave ne=[%lld,%lld,%lld] wave_hat ne=[%lld,%lld,%lld]\n",
                    (long long) wave->ne[0], (long long) wave->ne[1], (long long) wave->ne[2],
                    (long long) wave_hat->ne[0], (long long) wave_hat->ne[1],
                    (long long) wave_hat->ne[2]);
        return out;
    }
    if (wave->ne[1] != m.s.convs[0].in_channels) {
        std::printf("[mpd] waveform channels %lld != DiscriminatorS in_channels %lld\n",
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
