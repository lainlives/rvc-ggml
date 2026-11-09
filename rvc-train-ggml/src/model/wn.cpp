// RVC-Train-GGML - WN gated encoder implementation (src/model/wn.cpp)
#include "model/wn.h"

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

// Manual Conv1d assembly (avoid init helpers rewriting tensor names; forward uses the explicit weight)
void wire_conv(Conv1d* layer, Tensor* v, Tensor* bias, int64_t stride, int64_t padding,
               int64_t dilation) {
    layer->weight       = v;
    layer->bias         = bias;
    layer->has_bias     = bias != nullptr;
    layer->kernel_size  = v->ne[0];
    layer->in_channels  = v->ne[1];
    layer->out_channels = v->ne[2];
    layer->stride       = stride;
    layer->padding      = padding;
    layer->dilation     = dilation;
    layer->groups       = 1;
    std::snprintf(layer->name, MAX_NAME, "%s", "wn");
}

int64_t ipow(int64_t base, int64_t exp) {
    int64_t r = 1;
    for (int64_t i = 0; i < exp; ++i) {
        r *= base;
    }
    return r;
}

} // namespace

bool wn_load(const ModelParams& params, const WnConfig& cfg, WnModule* out, std::string* err) {
    if (out == nullptr) {
        if (err != nullptr) {
            *err = "WN output is null";
        }
        return false;
    }
    if (cfg.hidden_channels <= 0 || cfg.kernel_size <= 0 || cfg.n_layers <= 0) {
        if (err != nullptr) {
            *err = "invalid WN config (hidden/kernel/n_layers must be > 0)";
        }
        return false;
    }

    WnModule m;
    m.cfg = cfg;
    m.in_wn.resize((size_t) cfg.n_layers);
    m.rs_wn.resize((size_t) cfg.n_layers);
    m.in_layers.resize((size_t) cfg.n_layers);
    m.res_skip_layers.resize((size_t) cfg.n_layers);

    if (cfg.gin_channels > 0) {
        Tensor* g = require_param(params, cfg.prefix + ".cond_layer.weight_g", err);
        if (g == nullptr) {
            return false;
        }
        Tensor* v = require_param(params, cfg.prefix + ".cond_layer.weight_v", err);
        if (v == nullptr) {
            return false;
        }
        Tensor* b = params.find((cfg.prefix + ".cond_layer.bias").c_str());
        m.cond_wn.g = g;
        m.cond_wn.v = v;
        wire_conv(&m.cond, v, b, 1, 0, 1);
    }

    for (int64_t i = 0; i < cfg.n_layers; ++i) {
        const std::string base = cfg.prefix + ".in_layers." + std::to_string(i);
        const std::string rbase = cfg.prefix + ".res_skip_layers." + std::to_string(i);

        Tensor* ig = require_param(params, base + ".weight_g", err);
        if (ig == nullptr) {
            return false;
        }
        Tensor* iv = require_param(params, base + ".weight_v", err);
        if (iv == nullptr) {
            return false;
        }
        Tensor* ib = params.find((base + ".bias").c_str());
        const int64_t dilation = ipow(cfg.dilation_rate, i);
        const int64_t padding  = (cfg.kernel_size * dilation - dilation) / 2;
        m.in_wn[(size_t) i].g = ig;
        m.in_wn[(size_t) i].v = iv;
        wire_conv(&m.in_layers[(size_t) i], iv, ib, 1, padding, dilation);

        Tensor* rg = require_param(params, rbase + ".weight_g", err);
        if (rg == nullptr) {
            return false;
        }
        Tensor* rv = require_param(params, rbase + ".weight_v", err);
        if (rv == nullptr) {
            return false;
        }
        Tensor* rb = params.find((rbase + ".bias").c_str());
        m.rs_wn[(size_t) i].g = rg;
        m.rs_wn[(size_t) i].v = rv;
        wire_conv(&m.res_skip_layers[(size_t) i], rv, rb, 1, 0, 1);
    }

    *out = std::move(m);
    return true;
}

Tensor* wn_forward(Context* ctx, const WnModule& wn, Tensor* x, Tensor* x_mask, Tensor* g,
                   WnForwardCaptures* caps) {
    if (ctx == nullptr || x == nullptr || x_mask == nullptr) {
        std::printf("[wn] null input\n");
        return nullptr;
    }
    const int64_t H = wn.cfg.hidden_channels;
    const int64_t T = x->ne[0];
    const int64_t B = x->ne[2];
    if (x->ne[1] != H) {
        std::printf("[wn] input channels %lld != hidden_channels %lld\n", (long long) x->ne[1],
                    (long long) H);
        return nullptr;
    }

    Tensor* g_cond = nullptr;
    if (g != nullptr && wn.cfg.gin_channels > 0) {
        Tensor* w = weightnorm_forward(ctx, &wn.cond_wn);
        g_cond = conv1d_forward_weight(ctx, &wn.cond, w, g);
        if (caps != nullptr) {
            caps->cond = g_cond;
        }
    }

    Tensor* output = nullptr;
    for (int64_t i = 0; i < wn.cfg.n_layers; ++i) {
        Tensor* w_in = weightnorm_forward(ctx, &wn.in_wn[(size_t) i]);
        Tensor* x_in = conv1d_forward_weight(ctx, &wn.in_layers[(size_t) i], w_in, x);

        Tensor* in_act = x_in;
        if (g_cond != nullptr) {
            // Slice cond output channels per layer: [i*2H, (i+1)*2H)
            Tensor* g_l = cont(ctx, view_3d(ctx, g_cond, g_cond->ne[0], 2 * H, g_cond->ne[2],
                                            (size_t) i * 2 * H * g_cond->nb[1]));
            in_act = add(ctx, x_in, g_l);
        }
        // Split channels in half: first half tanh, second half sigmoid
        Tensor* a = cont(ctx, view_3d(ctx, in_act, T, H, B, 0));
        Tensor* b = cont(ctx, view_3d(ctx, in_act, T, H, B, (size_t) H * in_act->nb[1]));
        Tensor* acts = mul(ctx, tanh_op(ctx, a), sigmoid(ctx, b));
        if (caps != nullptr && i == 0) {
            caps->x_in0 = x_in;
            caps->acts0 = acts;
        }

        Tensor* w_rs = weightnorm_forward(ctx, &wn.rs_wn[(size_t) i]);
        Tensor* rs = conv1d_forward_weight(ctx, &wn.res_skip_layers[(size_t) i], w_rs, acts);
        if (i < wn.cfg.n_layers - 1) {
            Tensor* res  = cont(ctx, view_3d(ctx, rs, T, H, B, 0));
            Tensor* skip = cont(ctx, view_3d(ctx, rs, T, H, B, (size_t) H * rs->nb[1]));
            x = mul(ctx, add(ctx, x, res), x_mask);
            output = (output == nullptr) ? skip : add(ctx, output, skip);
        } else {
            output = (output == nullptr) ? rs : add(ctx, output, rs);
        }
    }
    return mul(ctx, output, x_mask);
}

} // namespace model
} // namespace rvc_train
