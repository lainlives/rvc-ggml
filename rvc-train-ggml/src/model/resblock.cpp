// RVC-Train-GGML - ResBlock1 implementation (src/model/resblock.cpp)
#include "model/resblock.h"

#include <cstdio>

namespace rvc_train {
namespace model {

namespace {

using namespace traincpp;

constexpr float kLeakySlope = 0.1f;

Tensor* require_param(const ModelParams& params, const std::string& name, std::string* err) {
    Tensor* t = params.find(name.c_str());
    if (t == nullptr && err != nullptr) {
        *err = "missing parameter: " + name;
    }
    return t;
}

void wire_conv(Conv1d* layer, Tensor* v, Tensor* bias, int64_t padding, int64_t dilation) {
    layer->weight       = v;
    layer->bias         = bias;
    layer->has_bias     = bias != nullptr;
    layer->kernel_size  = v->ne[0];
    layer->in_channels  = v->ne[1];
    layer->out_channels = v->ne[2];
    layer->stride       = 1;
    layer->padding      = padding;
    layer->dilation     = dilation;
    layer->groups       = 1;
    std::snprintf(layer->name, MAX_NAME, "%s", "resblock1");
}

} // namespace

bool resblock1_load(const ModelParams& params, const ResBlock1Config& cfg, ResBlock1Module* out,
                    std::string* err) {
    if (out == nullptr) {
        if (err != nullptr) {
            *err = "ResBlock1 output is null";
        }
        return false;
    }
    if (cfg.channels <= 0 || cfg.kernel_size <= 0 || cfg.dilations.size() != 3) {
        if (err != nullptr) {
            *err = "invalid ResBlock1 config (channels/kernel > 0, dilations length must be 3)";
        }
        return false;
    }

    ResBlock1Module m;
    m.cfg = cfg;
    m.c1_wn.resize(cfg.dilations.size());
    m.c2_wn.resize(cfg.dilations.size());
    m.c1.resize(cfg.dilations.size());
    m.c2.resize(cfg.dilations.size());

    for (size_t i = 0; i < cfg.dilations.size(); ++i) {
        const std::string c1base = cfg.prefix + ".convs1." + std::to_string(i);
        const std::string c2base = cfg.prefix + ".convs2." + std::to_string(i);

        Tensor* g1 = require_param(params, c1base + ".weight_g", err);
        if (g1 == nullptr) {
            return false;
        }
        Tensor* v1 = require_param(params, c1base + ".weight_v", err);
        if (v1 == nullptr) {
            return false;
        }
        Tensor* b1 = params.find((c1base + ".bias").c_str());
        m.c1_wn[i].g = g1;
        m.c1_wn[i].v = v1;
        const int64_t d1 = cfg.dilations[i];
        wire_conv(&m.c1[i], v1, b1, (cfg.kernel_size * d1 - d1) / 2, d1);

        Tensor* g2 = require_param(params, c2base + ".weight_g", err);
        if (g2 == nullptr) {
            return false;
        }
        Tensor* v2 = require_param(params, c2base + ".weight_v", err);
        if (v2 == nullptr) {
            return false;
        }
        Tensor* b2 = params.find((c2base + ".bias").c_str());
        m.c2_wn[i].g = g2;
        m.c2_wn[i].v = v2;
        wire_conv(&m.c2[i], v2, b2, (cfg.kernel_size - 1) / 2, 1);
    }

    *out = std::move(m);
    return true;
}

Tensor* resblock1_forward(Context* ctx, const ResBlock1Module& rb, Tensor* x, Tensor* x_mask,
                          ResBlock1Captures* caps) {
    if (ctx == nullptr || x == nullptr) {
        std::printf("[resblock1] null input\n");
        return nullptr;
    }
    if (x->ne[1] != rb.cfg.channels) {
        std::printf("[resblock1] input channels %lld != channels %lld\n", (long long) x->ne[1],
                    (long long) rb.cfg.channels);
        return nullptr;
    }

    for (size_t i = 0; i < rb.cfg.dilations.size(); ++i) {
        Tensor* xt = leaky_relu(ctx, x, kLeakySlope);
        if (x_mask != nullptr) {
            xt = mul(ctx, xt, x_mask);
        }
        Tensor* w1 = weightnorm_forward(ctx, &rb.c1_wn[i]);
        xt = conv1d_forward_weight(ctx, &rb.c1[i], w1, xt);
        if (caps != nullptr && i == 0) {
            caps->t0 = xt;
        }
        xt = leaky_relu(ctx, xt, kLeakySlope);
        if (x_mask != nullptr) {
            xt = mul(ctx, xt, x_mask);
        }
        Tensor* w2 = weightnorm_forward(ctx, &rb.c2_wn[i]);
        xt = conv1d_forward_weight(ctx, &rb.c2[i], w2, xt);
        x = add(ctx, xt, x);
    }
    if (x_mask != nullptr) {
        x = mul(ctx, x, x_mask);
    }
    return x;
}

} // namespace model
} // namespace rvc_train
