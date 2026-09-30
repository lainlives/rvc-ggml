// RVC-Train-GGML - flow（ResidualCouplingBlock 正向）实现（src/model/flow.cpp）
#include "model/flow.h"

#include <cstdio>
#include <utility>

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

void wire_conv1d(Conv1d* layer, Tensor* weight, Tensor* bias, const char* name) {
    layer->weight       = weight;
    layer->bias         = bias;
    layer->has_bias     = (bias != nullptr);
    layer->kernel_size  = weight->ne[0];
    layer->in_channels  = weight->ne[1];
    layer->out_channels = weight->ne[2];
    layer->stride       = 1;
    layer->padding      = 0;      // 1×1
    layer->dilation     = 1;
    layer->groups       = 1;
    std::snprintf(layer->name, MAX_NAME, "%s", name);
}

} // namespace

bool flow_load(const ModelParams& params, const FlowConfig& cfg, FlowModule* out, std::string* err) {
    if (out == nullptr) {
        if (err != nullptr) {
            *err = "flow 输出为空";
        }
        return false;
    }
    if (cfg.channels <= 0 || cfg.channels % 2 != 0 || cfg.n_flows <= 0 || cfg.n_layers <= 0) {
        if (err != nullptr) {
            *err = "flow 配置非法（channels 必须为正偶数；n_flows/n_layers > 0）";
        }
        return false;
    }

    FlowModule m;
    m.cfg = cfg;
    m.layers.resize(static_cast<size_t>(cfg.n_flows));

    WnConfig wcfg;
    wcfg.hidden_channels = cfg.hidden_channels;
    wcfg.kernel_size     = cfg.kernel_size;
    wcfg.n_layers        = cfg.n_layers;
    wcfg.gin_channels    = cfg.gin_channels;
    wcfg.dilation_rate   = cfg.dilation_rate;

    for (int64_t i = 0; i < cfg.n_flows; ++i) {
        const std::string base = cfg.prefix + ".flows." + std::to_string(i * 2);
        FlowLayer& L = m.layers[static_cast<size_t>(i)];

        Tensor* pw = require_param(params, base + ".pre.weight", err);
        if (pw == nullptr) {
            return false;
        }
        Tensor* pb = require_param(params, base + ".pre.bias", err);
        if (pb == nullptr) {
            return false;
        }
        wire_conv1d(&L.pre, pw, pb, "flow.pre");
        if (pw->ne[2] != cfg.hidden_channels || pw->ne[1] != cfg.channels / 2) {
            if (err != nullptr) {
                *err = base + ".pre.weight 形状与配置不符（期望 ne=[1," +
                       std::to_string(cfg.channels / 2) + "," + std::to_string(cfg.hidden_channels) + "]）";
            }
            return false;
        }

        Tensor* sw = require_param(params, base + ".post.weight", err);
        if (sw == nullptr) {
            return false;
        }
        Tensor* sb = require_param(params, base + ".post.bias", err);
        if (sb == nullptr) {
            return false;
        }
        wire_conv1d(&L.post, sw, sb, "flow.post");

        wcfg.prefix = base + ".enc";
        if (!wn_load(params, wcfg, &L.enc, err)) {
            return false;
        }
    }

    *out = std::move(m);
    return true;
}

bool flow_fill_flip_perm(Tensor* perm, std::string* err) {
    if (perm == nullptr || perm->type != TYPE_F32 || perm->ne[0] != perm->ne[1]) {
        if (err != nullptr) {
            *err = "flip 置换矩阵张量非法（需要方阵 F32）";
        }
        return false;
    }
    float* p = tensor_data_f32(perm);
    if (p == nullptr) {
        if (err != nullptr) {
            *err = "flip 置换矩阵未分配（需在 buffer 分配后调用）";
        }
        return false;
    }
    const int64_t c = perm->ne[0];
    for (int64_t i = 0; i < c; ++i) {
        for (int64_t j = 0; j < c; ++j) {
            // mul_mat(P, x) = P^T·x；P^T[i][j] = P[j + c*i]，置 1 当且仅当 j + i = c - 1
            p[j + c * i] = (j + i == c - 1) ? 1.0f : 0.0f;
        }
    }
    return true;
}

Tensor* flow_forward(Context* ctx, const FlowModule& flow, Tensor* x, Tensor* x_mask, Tensor* g,
                     Tensor* perm, FlowCaptures* caps) {
    if (ctx == nullptr || x == nullptr || x_mask == nullptr || perm == nullptr) {
        std::printf("[flow] 输入为空\n");
        return nullptr;
    }
    const int64_t frames   = x->ne[0];
    const int64_t channels = x->ne[1];
    const int64_t batch    = x->ne[2];
    const int64_t half     = channels / 2;
    if (channels != flow.cfg.channels || channels % 2 != 0) {
        std::printf("[flow] x 通道 %lld != channels %lld\n", (long long) channels,
                    (long long) flow.cfg.channels);
        return nullptr;
    }
    if (perm->ne[0] != channels || perm->ne[1] != channels) {
        std::printf("[flow] perm 形状 [%lld,%lld] != [%lld,%lld]\n", (long long) perm->ne[0],
                    (long long) perm->ne[1], (long long) channels, (long long) channels);
        return nullptr;
    }
    if (caps != nullptr) {
        caps->pre.clear();
        caps->enc.clear();
        caps->stats.clear();
    }

    Tensor* cur = x;
    for (size_t li = 0; li < flow.layers.size(); ++li) {
        const FlowLayer& L = flow.layers[li];

        // 通道对半切（视图 + cont，避免非连续输入喂卷积）
        Tensor* x0 = cont(ctx, view_3d(ctx, cur, frames, half, batch, 0));
        Tensor* x1 = cont(ctx, view_3d(ctx, cur, frames, half, batch,
                                       static_cast<size_t>(half) * cur->nb[1]));
        if (x0 == nullptr || x1 == nullptr) {
            return nullptr;
        }

        Tensor* h = conv1d_forward_weight(ctx, &L.pre, L.pre.weight, x0);
        if (h == nullptr) {
            return nullptr;
        }
        if (caps != nullptr) {
            caps->pre.push_back(h);
        }
        h = mul(ctx, h, x_mask);
        Tensor* e = wn_forward(ctx, L.enc, h, x_mask, g);
        if (e == nullptr) {
            return nullptr;
        }
        if (caps != nullptr) {
            caps->enc.push_back(e);
        }
        Tensor* stats = conv1d_forward_weight(ctx, &L.post, L.post.weight, e);
        if (stats == nullptr) {
            return nullptr;
        }
        if (caps != nullptr) {
            caps->stats.push_back(stats);
        }

        // 正向：x1 = stats·mask + x1·mask
        Tensor* m = mul(ctx, stats, x_mask);
        Tensor* x1n = add(ctx, mul(ctx, x1, x_mask), m);
        Tensor* cat = concat(ctx, x0, x1n, 1);
        if (cat == nullptr) {
            return nullptr;
        }

        // Flip：沿通道翻转。mul_mat(P,·) 的行索引对应 ne0，因此先把通道维换到 ne0：
        //   [T,C,B] → transpose → [C,T,B] → reshape → [C, T·B] → mul_mat(P) → 换回
        Tensor* ct   = cont(ctx, transpose(ctx, cat));
        if (ct == nullptr) {
            return nullptr;
        }
        Tensor* flat = reshape_2d(ctx, ct, channels, frames * batch);
        if (flat == nullptr) {
            return nullptr;
        }
        Tensor* flipped = mul_mat(ctx, perm, flat);
        if (flipped == nullptr) {
            return nullptr;
        }
        Tensor* f3 = reshape_3d(ctx, flipped, channels, frames, batch);
        if (f3 == nullptr) {
            return nullptr;
        }
        cur = cont(ctx, transpose(ctx, f3));
        if (cur == nullptr) {
            return nullptr;
        }
    }
    return cur;
}

} // namespace model
} // namespace rvc_train
