// RVC-Train-GGML - enc_q (PosteriorEncoder) implementation (src/model/enc_q.cpp)
#include "model/enc_q.h"

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

// Manual Conv1d assembly (avoid init helpers rewriting tensor names; forward uses the layer's weight)
void wire_conv(Conv1d* layer, Tensor* weight, Tensor* bias, int64_t stride, int64_t padding,
               int64_t dilation, const char* name) {
    layer->weight       = weight;
    layer->bias         = bias;
    layer->has_bias     = bias != nullptr;
    layer->kernel_size  = weight->ne[0];
    layer->in_channels  = weight->ne[1];
    layer->out_channels = weight->ne[2];
    layer->stride       = stride;
    layer->padding      = padding;
    layer->dilation     = dilation;
    layer->groups       = 1;
    std::snprintf(layer->name, MAX_NAME, "%s", name);
}

bool check_ne(const Tensor* t, int64_t ne0, int64_t ne1, int64_t ne2, const std::string& name,
              std::string* err) {
    if (t->ne[0] != ne0 || t->ne[1] != ne1 || t->ne[2] != ne2) {
        if (err != nullptr) {
            char buf[256];
            std::snprintf(buf, sizeof(buf),
                          "shape mismatch: %s expected [%lld,%lld,%lld] got [%lld,%lld,%lld]",
                          name.c_str(), (long long) ne0, (long long) ne1, (long long) ne2,
                          (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2]);
            *err = buf;
        }
        return false;
    }
    return true;
}

} // namespace

bool enc_q_load(const ModelParams& params, const EncQConfig& cfg, EncQModule* out,
                std::string* err) {
    if (out == nullptr) {
        if (err != nullptr) {
            *err = "enc_q output is null";
        }
        return false;
    }
    if (cfg.in_channels <= 0 || cfg.out_channels <= 0 || cfg.hidden_channels <= 0 ||
        cfg.kernel_size <= 0 || cfg.n_layers <= 0) {
        if (err != nullptr) {
            *err = "invalid enc_q config (in/out/hidden/kernel/n_layers must be > 0)";
        }
        return false;
    }

    EncQModule m;
    m.cfg = cfg;

    Tensor* pw = require_param(params, cfg.prefix + ".pre.weight", err);
    if (pw == nullptr) {
        return false;
    }
    Tensor* pb = params.find((cfg.prefix + ".pre.bias").c_str());
    if (!check_ne(pw, 1, cfg.in_channels, cfg.hidden_channels, cfg.prefix + ".pre.weight", err)) {
        return false;
    }
    wire_conv(&m.pre, pw, pb, 1, 0, 1, "enc_q.pre");

    WnConfig wcfg;
    wcfg.prefix          = cfg.prefix + ".enc";
    wcfg.hidden_channels = cfg.hidden_channels;
    wcfg.kernel_size     = cfg.kernel_size;
    wcfg.n_layers        = cfg.n_layers;
    wcfg.gin_channels    = cfg.gin_channels;
    wcfg.dilation_rate   = cfg.dilation_rate;
    if (!wn_load(params, wcfg, &m.enc, err)) {
        return false;
    }

    Tensor* sw = require_param(params, cfg.prefix + ".proj.weight", err);
    if (sw == nullptr) {
        return false;
    }
    Tensor* sb = params.find((cfg.prefix + ".proj.bias").c_str());
    if (!check_ne(sw, 1, cfg.hidden_channels, 2 * cfg.out_channels, cfg.prefix + ".proj.weight",
                  err)) {
        return false;
    }
    wire_conv(&m.proj, sw, sb, 1, 0, 1, "enc_q.proj");

    *out = std::move(m);
    return true;
}

EncQOutputs enc_q_forward(Context* ctx, const EncQModule& m, Tensor* spec, Tensor* x_mask,
                          Tensor* g, Tensor* noise, EncQCaptures* caps) {
    if (ctx == nullptr || spec == nullptr || x_mask == nullptr || noise == nullptr) {
        std::printf("[enc_q] null input\n");
        return {};
    }
    const int64_t OC = m.cfg.out_channels;
    const int64_t T  = spec->ne[0];
    const int64_t B  = spec->ne[2];
    if (spec->ne[1] != m.cfg.in_channels || x_mask->ne[0] != T || x_mask->ne[1] != 1 ||
        x_mask->ne[2] != B || noise->ne[0] != T || noise->ne[1] != OC || noise->ne[2] != B) {
        std::printf("[enc_q] input shape mismatch: spec ne=[%lld,%lld,%lld] noise ne=[%lld,%lld,%lld]\n",
                    (long long) spec->ne[0], (long long) spec->ne[1], (long long) spec->ne[2],
                    (long long) noise->ne[0], (long long) noise->ne[1], (long long) noise->ne[2]);
        return {};
    }

    // pre(x)·mask
    Tensor* h = conv1d_forward(ctx, &m.pre, spec);
    if (h == nullptr) {
        std::printf("[enc_q] pre convolution failed\n");
        return {};
    }
    if (caps != nullptr) {
        caps->pre = h;
    }
    h = mul(ctx, h, x_mask);

    // WN
    WnForwardCaptures wcaps;
    Tensor* e = wn_forward(ctx, m.enc, h, x_mask, g, &wcaps);
    if (e == nullptr) {
        std::printf("[enc_q] WN forward failed\n");
        return {};
    }
    if (caps != nullptr) {
        caps->cond  = wcaps.cond;
        caps->x_in0 = wcaps.x_in0;
        caps->acts0 = wcaps.acts0;
        caps->enc   = e;
    }

    // proj(x)·mask -> split -> z sampling
    Tensor* stats_raw = conv1d_forward(ctx, &m.proj, e);
    if (stats_raw == nullptr) {
        std::printf("[enc_q] proj convolution failed\n");
        return {};
    }
    if (caps != nullptr) {
        caps->stats = stats_raw;
    }
    Tensor* stats = mul(ctx, stats_raw, x_mask);

    EncQOutputs out;
    out.m_q    = view_3d(ctx, stats, T, OC, B, 0);
    out.logs_q = view_3d(ctx, stats, T, OC, B, (size_t) OC * stats->nb[1]);
    Tensor* z0 = add(ctx, out.m_q, mul(ctx, noise, exp_op(ctx, out.logs_q)));
    out.z      = mul(ctx, z0, x_mask);
    return out;
}

} // namespace model
} // namespace rvc_train
