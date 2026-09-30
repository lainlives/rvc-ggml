// flow.cpp
// RVC flow（ResidualCouplingBlock）逆变换（reverse=True）：
//   顺序 = [Flip, flow3, Flip, flow2, Flip, flow1, Flip, flow0]
//   耦合层 mean_only：x1' = x1 - m，m = post(WN(pre(x0), g))

#include "vits/synthesizer.h"
#include "vits/modules.h"

#include <stdexcept>

namespace rvc {
namespace vits {

namespace {

    ggml_tensor* require(const GgufModel& model, const std::string& name) {
        ggml_tensor* t = model.weight(name);
        if (!t) {
            throw std::runtime_error("VITS 缺失权重: " + name);
        }
        return t;
    }

    // WN（WaveNet 残差块，dilation_rate=1）：x [H,T] -> [H,T]
    ggml_tensor* build_wn(ggml_context* ctx, const GgufModel& model, const Config& cfg,
        ggml_tensor* x, ggml_tensor* g, const std::string& prefix) {
        const int64_t H = cfg.hidden_channels;
        const int64_t n_layers = cfg.flow_n_layers;

        ggml_tensor* cond = conv1d_ct(ctx, g, require(model, prefix + ".cond_layer.weight"),
            require(model, prefix + ".cond_layer.bias"), 1, 0, 1, cfg.conv_flat);   // [2H*N, 1]
        ggml_tensor* output = ggml_scale(ctx, x, 0.0f);

        for (int64_t i = 0; i < n_layers; ++i) {
            const std::string lp = prefix + ".in_layers." + std::to_string(i);
            const std::string rp = prefix + ".res_skip_layers." + std::to_string(i);
            ggml_tensor* w = require(model, lp + ".weight");
            // x 为 [C,T] 布局，通道数在 ne[0]
            const int64_t kernel = cfg.conv_flat ? w->ne[0] / x->ne[0] : w->ne[0];
            const int64_t padding = (kernel - 1) / 2;   // dilation = 1

            ggml_tensor* x_in = conv1d_ct(ctx, x, w, require(model, lp + ".bias"),
                1, static_cast<int>(padding), 1, cfg.conv_flat);
            ggml_tensor* g_l = ggml_cont(ctx, slice_channels(ctx, cond, i * 2 * H, 2 * H));
            g_l = ggml_repeat(ctx, g_l, x_in);
            ggml_tensor* acts = add_checked(ctx, x_in, g_l);

            // 融合门：left=tanh, right=sigmoid
            ggml_tensor* left = ggml_tanh(ctx, slice_channels(ctx, acts, 0, H));
            ggml_tensor* right = ggml_sigmoid(ctx, slice_channels(ctx, acts, H, H));
            acts = mul_checked(ctx, left, right);

            const int64_t rs_channels = (i < n_layers - 1) ? 2 * H : H;
            ggml_tensor* rs = conv1d_ct(ctx, acts, require(model, rp + ".weight"),
                require(model, rp + ".bias"), 1, 0, 1, cfg.conv_flat);
            if (rs->ne[0] != rs_channels) {
                throw std::runtime_error("VITS flow res_skip 通道数异常");
            }
            if (i < n_layers - 1) {
                x = add_checked(ctx, x, ggml_cont(ctx, slice_channels(ctx, rs, 0, H)));
                output = add_checked(ctx, output,
                    ggml_cont(ctx, slice_channels(ctx, rs, H, H)));
            }
            else {
                output = add_checked(ctx, output, rs);
            }
        }
        return output;
    }

    // 残差耦合逆变换（mean_only=True）
    ggml_tensor* coupling_reverse(ggml_context* ctx, const GgufModel& model,
        const Config& cfg, ggml_tensor* x, ggml_tensor* g, int64_t flow_index) {
        const int64_t half = cfg.inter_channels / 2;
        const std::string prefix = "flow.flows." + std::to_string(flow_index);
        ggml_tensor* x0 = ggml_cont(ctx, slice_channels(ctx, x, 0, half));
        ggml_tensor* x1 = ggml_cont(ctx, slice_channels(ctx, x, half, half));

        ggml_tensor* h = conv1d_ct(ctx, x0, require(model, prefix + ".pre.weight"),
            require(model, prefix + ".pre.bias"), 1, 0, 1, cfg.conv_flat);
        h = build_wn(ctx, model, cfg, h, g, prefix + ".enc");
        ggml_tensor* m = conv1d_ct(ctx, h, require(model, prefix + ".post.weight"),
            require(model, prefix + ".post.bias"), 1, 0, 1, cfg.conv_flat);
        x1 = ggml_sub(ctx, x1, m);
        return ggml_concat(ctx, x0, x1, 0);
    }

} // namespace

ggml_tensor* build_flow_reverse(ggml_context* ctx, const GgufModel& model,
    const Config& cfg, ggml_tensor* z_p, ggml_tensor* g, int64_t frames,
    ggml_tensor* flip_matrix) {
    (void)frames;
    ggml_tensor* z = z_p;
    const int64_t order[4] = { 6, 4, 2, 0 };
    for (int64_t k : order) {
        // 通道翻转（用置换矩阵实现）
        z = ggml_mul_mat(ctx, flip_matrix, z);
        z = coupling_reverse(ctx, model, cfg, z, g, k);
    }
    return z;
}

} // namespace vits
} // namespace rvc
