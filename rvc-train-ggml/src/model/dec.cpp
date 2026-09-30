// RVC-Train-GGML - GeneratorNSF 实现（src/model/dec.cpp）
#include "model/dec.h"

#include <cstdio>
#include <utility>

namespace rvc_train {
namespace model {

namespace {

using namespace traincpp;

constexpr float kLeakySlope = 0.1f;        // 级内 LeakyReLU 斜率（GeneratorNSF.lrelu_slope）
constexpr float kLeakySlopeFinal = 0.01f;  // 末级 F.leaky_relu 默认斜率

Tensor* require_param(const ModelParams& params, const std::string& name, std::string* err) {
    Tensor* t = params.find(name.c_str());
    if (t == nullptr && err != nullptr) {
        *err = "缺失参数: " + name;
    }
    return t;
}

void wire_conv1d(Conv1d* layer, Tensor* weight, Tensor* bias, int64_t stride, int64_t padding,
                 int64_t dilation, const char* name) {
    layer->weight       = weight;
    layer->bias         = bias;
    layer->has_bias     = (bias != nullptr);
    layer->kernel_size  = weight->ne[0];
    layer->in_channels  = weight->ne[1];
    layer->out_channels = weight->ne[2];
    layer->stride       = stride;
    layer->padding      = padding;
    layer->dilation     = dilation;
    layer->groups       = 1;
    std::snprintf(layer->name, MAX_NAME, "%s", name);
}

// ∏ rates[from..]（from >= size 时为 1）
int64_t prod_rates(const std::vector<int64_t>& rates, size_t from) {
    int64_t p = 1;
    for (size_t i = from; i < rates.size(); ++i) {
        p *= rates[i];
    }
    return p;
}

} // namespace

bool dec_load(const ModelParams& params, const DecConfig& cfg, DecModule* out, std::string* err) {
    if (out == nullptr) {
        if (err != nullptr) {
            *err = "dec 输出为空";
        }
        return false;
    }
    if (cfg.in_channels <= 0 || cfg.upsample_initial_channel <= 0 || cfg.upsample_rates.empty() ||
        cfg.upsample_rates.size() != cfg.upsample_kernel_sizes.size() ||
        cfg.resblock_kernel_sizes.empty() ||
        cfg.resblock_dilation_sizes.size() != cfg.resblock_kernel_sizes.size() * 3) {
        if (err != nullptr) {
            *err = "dec 配置非法（in_channels/up_init > 0；rates/kernels 等长；resblock 每 kernel 3 个 dilation）";
        }
        return false;
    }

    DecModule m;
    m.cfg = cfg;

    // z 激励源线性层（1→1）
    Tensor* lw = require_param(params, cfg.prefix + ".m_source.l_linear.weight", err);
    if (lw == nullptr) {
        return false;
    }
    m.l_linear.weight       = lw;
    m.l_linear.bias         = params.find((cfg.prefix + ".m_source.l_linear.bias").c_str());
    m.l_linear.in_features  = 1;
    m.l_linear.out_features = 1;
    m.l_linear.has_bias     = (m.l_linear.bias != nullptr);
    std::snprintf(m.l_linear.name, MAX_NAME, "%s", "l_linear");

    // conv_pre / cond / conv_post
    Tensor* w = require_param(params, cfg.prefix + ".conv_pre.weight", err);
    if (w == nullptr) {
        return false;
    }
    wire_conv1d(&m.conv_pre, w, params.find((cfg.prefix + ".conv_pre.bias").c_str()), 1,
                (w->ne[0] - 1) / 2, 1, "conv_pre");

    w = require_param(params, cfg.prefix + ".cond.weight", err);
    if (w == nullptr) {
        return false;
    }
    wire_conv1d(&m.cond, w, params.find((cfg.prefix + ".cond.bias").c_str()), 1, 0, 1, "cond");

    w = require_param(params, cfg.prefix + ".conv_post.weight", err);
    if (w == nullptr) {
        return false;
    }
    wire_conv1d(&m.conv_post, w, nullptr, 1, (w->ne[0] - 1) / 2, 1, "conv_post");

    // 上采样级与噪声卷积
    const size_t n_up = cfg.upsample_rates.size();
    m.ups.resize(n_up);
    m.ups_wn.resize(n_up);
    m.noise_convs.resize(n_up);
    for (size_t i = 0; i < n_up; ++i) {
        const std::string base = cfg.prefix + ".ups." + std::to_string(i);
        Tensor* g = require_param(params, base + ".weight_g", err);
        if (g == nullptr) {
            return false;
        }
        Tensor* v = require_param(params, base + ".weight_v", err);
        if (v == nullptr) {
            return false;
        }
        m.ups_wn[i].g = g;
        m.ups_wn[i].v = v;

        ConvTranspose1d& up = m.ups[i];
        up.weight       = v;
        up.bias         = params.find((base + ".bias").c_str());
        up.has_bias     = (up.bias != nullptr);
        up.kernel_size  = v->ne[0];
        up.out_channels = v->ne[1];
        up.in_channels  = v->ne[2];
        up.stride       = cfg.upsample_rates[i];
        up.padding      = (up.kernel_size - up.stride) / 2;
        up.dilation     = 1;
        std::snprintf(up.name, MAX_NAME, "%s", "dec.ups");
        if (up.kernel_size != cfg.upsample_kernel_sizes[i]) {
            if (err != nullptr) {
                *err = "dec.ups." + std::to_string(i) + " 核大小与配置不符（权重 " +
                       std::to_string(up.kernel_size) + " vs 配置 " +
                       std::to_string(cfg.upsample_kernel_sizes[i]) + "）";
            }
            return false;
        }

        const std::string nbase = cfg.prefix + ".noise_convs." + std::to_string(i);
        Tensor* nw = require_param(params, nbase + ".weight", err);
        if (nw == nullptr) {
            return false;
        }
        const int64_t n_stride  = (i + 1 < n_up) ? prod_rates(cfg.upsample_rates, i + 1) : 1;
        const int64_t n_padding = (i + 1 < n_up) ? n_stride / 2 : 0;
        wire_conv1d(&m.noise_convs[i], nw, params.find((nbase + ".bias").c_str()), n_stride,
                    n_padding, 1, "noise_conv");
    }

    // ResBlock1：每级 num_kernels 个（解复用 M2-2 已验收实现）
    const int64_t num_kernels = static_cast<int64_t>(cfg.resblock_kernel_sizes.size());
    m.num_kernels = num_kernels;
    m.resblocks.reserve(n_up * static_cast<size_t>(num_kernels));
    for (size_t i = 0; i < n_up; ++i) {
        for (int64_t j = 0; j < num_kernels; ++j) {
            ResBlock1Config rc;
            rc.prefix      = cfg.prefix + ".resblocks." + std::to_string(
                                 static_cast<int64_t>(i) * num_kernels + j);
            rc.channels    = m.ups[i].out_channels;
            rc.kernel_size = cfg.resblock_kernel_sizes[static_cast<size_t>(j)];
            rc.dilations   = {cfg.resblock_dilation_sizes[static_cast<size_t>(j * 3 + 0)],
                              cfg.resblock_dilation_sizes[static_cast<size_t>(j * 3 + 1)],
                              cfg.resblock_dilation_sizes[static_cast<size_t>(j * 3 + 2)]};
            ResBlock1Module rb;
            if (!resblock1_load(params, rc, &rb, err)) {
                return false;
            }
            m.resblocks.push_back(std::move(rb));
        }
    }

    *out = std::move(m);
    return true;
}

Tensor* dec_forward(Context* ctx, const DecModule& dec, Tensor* z, Tensor* sine_source, Tensor* g,
                    DecCaptures* caps) {
    if (ctx == nullptr || z == nullptr || sine_source == nullptr) {
        std::printf("[dec] 输入为空\n");
        return nullptr;
    }
    if (z->ne[1] != dec.cfg.in_channels) {
        std::printf("[dec] z 通道 %lld != in_channels %lld\n", (long long) z->ne[1],
                    (long long) dec.cfg.in_channels);
        return nullptr;
    }
    const int64_t frames = z->ne[0];
    const int64_t batch  = z->ne[2];
    const int64_t total  = frames * prod_rates(dec.cfg.upsample_rates, 0);
    // sine_source 布局约定：[1, L, B]（等价 torch [B,L,1]，L = T·∏rates）
    if (sine_source->ne[0] != 1 || sine_source->ne[1] != total || sine_source->ne[2] != batch) {
        std::printf("[dec] sine_source 形状 [%lld,%lld,%lld] != [1,%lld,%lld]\n",
                    (long long) sine_source->ne[0], (long long) sine_source->ne[1],
                    (long long) sine_source->ne[2], (long long) total, (long long) batch);
        return nullptr;
    }

    // 激励源：l_linear(1→1) + tanh（作用在最后一维，与 torch 一致；[L,1,B]→[1,L·B]→[L,1,B]）
    Tensor* src = cont(ctx, sine_source);
    Tensor* merged = reshape_2d(ctx, src, 1, total * batch);
    if (merged == nullptr) {
        return nullptr;
    }
    merged = linear_forward(ctx, &dec.l_linear, merged);
    if (merged == nullptr) {
        return nullptr;
    }
    merged = tanh_op(ctx, merged);
    merged = reshape_3d(ctx, merged, total, 1, batch);
    if (merged == nullptr) {
        return nullptr;
    }
    if (caps != nullptr) {
        caps->sine_merged = merged;
        caps->ups.clear();
        caps->levels.clear();
    }

    Tensor* x = conv1d_forward_weight(ctx, &dec.conv_pre, dec.conv_pre.weight, z);
    if (x == nullptr) {
        return nullptr;
    }
    if (caps != nullptr) {
        caps->conv_pre = x;
    }
    if (g != nullptr) {
        Tensor* gc = conv1d_forward(ctx, &dec.cond, g);
        if (gc == nullptr) {
            return nullptr;
        }
        if (caps != nullptr) {
            caps->cond = gc;
        }
        x = add(ctx, x, gc);
    }
    if (caps != nullptr) {
        caps->x_cond = x;
    }

    for (size_t i = 0; i < dec.ups.size(); ++i) {
        x = leaky_relu(ctx, x, kLeakySlope);
        Tensor* w = weightnorm_forward(ctx, &dec.ups_wn[i]);
        Tensor* xu = convtranspose1d_forward_weight(ctx, &dec.ups[i], w, x);
        Tensor* xs = conv1d_forward_weight(ctx, &dec.noise_convs[i], dec.noise_convs[i].weight,
                                           merged);
        if (xu == nullptr || xs == nullptr) {
            return nullptr;
        }
        x = add(ctx, xu, xs);

        Tensor* sum = nullptr;
        for (int64_t j = 0; j < dec.num_kernels; ++j) {
            Tensor* r = resblock1_forward(ctx, dec.resblocks[i * static_cast<size_t>(dec.num_kernels) +
                                                             static_cast<size_t>(j)],
                                          x, nullptr, nullptr);
            if (r == nullptr) {
                return nullptr;
            }
            sum = (sum == nullptr) ? r : add(ctx, sum, r);
        }
        x = scale(ctx, sum, 1.0f / static_cast<float>(dec.num_kernels));
        if (caps != nullptr) {
            caps->ups.push_back(xu);
            caps->levels.push_back(x);
        }
    }

    x = leaky_relu(ctx, x, kLeakySlopeFinal);
    x = conv1d_forward_weight(ctx, &dec.conv_post, dec.conv_post.weight, x);
    if (x == nullptr) {
        return nullptr;
    }
    return tanh_op(ctx, x);
}

} // namespace model
} // namespace rvc_train
