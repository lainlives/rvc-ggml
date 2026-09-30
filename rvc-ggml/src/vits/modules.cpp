// modules.cpp
// ggml 算子封装实现。

#include "vits/modules.h"

#include "utils.h"

#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace rvc {
namespace vits {

namespace {

[[noreturn]] void shape_fail(const char* op, const ggml_tensor* a, const ggml_tensor* b) {
    char buf[512];
    snprintf(buf, sizeof(buf),
        "%s 形状不兼容: a=[%lld,%lld,%lld,%lld] b=[%lld,%lld,%lld,%lld]",
        op,
        static_cast<long long>(a->ne[0]), static_cast<long long>(a->ne[1]),
        static_cast<long long>(a->ne[2]), static_cast<long long>(a->ne[3]),
        static_cast<long long>(b->ne[0]), static_cast<long long>(b->ne[1]),
        static_cast<long long>(b->ne[2]), static_cast<long long>(b->ne[3]));
    throw std::runtime_error(buf);
}

void check_broadcast(const char* op, ggml_tensor* a, ggml_tensor* b) {
    if (getenv("RVC_DEBUG_OP")) {
        fprintf(stderr, "[op %s] a=[%lld,%lld,%lld,%lld] b=[%lld,%lld,%lld,%lld]\n", op,
            (long long)a->ne[0], (long long)a->ne[1], (long long)a->ne[2], (long long)a->ne[3],
            (long long)b->ne[0], (long long)b->ne[1], (long long)b->ne[2], (long long)b->ne[3]);
    }
    for (int i = 0; i < 4; ++i) {
        if (a->ne[i] != b->ne[i] && a->ne[i] != 1 && b->ne[i] != 1) {
            shape_fail(op, a, b);
        }
    }
}

} // namespace

ggml_tensor* mul_checked(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b) {
    check_broadcast("ggml_mul", a, b);
    return ggml_mul(ctx, a, b);
}

ggml_tensor* add_checked(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b) {
    check_broadcast("ggml_add", a, b);
    return ggml_add(ctx, a, b);
}

void check_dim(const ggml_tensor* t, int64_t n0, int64_t n1, const char* where) {
    if (t->ne[0] != n0 || t->ne[1] != n1) {
        char buf[256];
        snprintf(buf, sizeof(buf), "%s: 期望 [%lld,%lld]，实际 [%lld,%lld]",
            where, static_cast<long long>(n0), static_cast<long long>(n1),
            static_cast<long long>(t->ne[0]), static_cast<long long>(t->ne[1]));
        throw std::runtime_error(buf);
    }
}

void check_dim3(const ggml_tensor* t, int64_t n0, int64_t n1, int64_t n2, const char* where) {
    if (t->ne[0] != n0 || t->ne[1] != n1 || t->ne[2] != n2) {
        char buf[300];
        snprintf(buf, sizeof(buf), "%s: 期望 [%lld,%lld,%lld]，实际 [%lld,%lld,%lld]",
            where,
            static_cast<long long>(n0), static_cast<long long>(n1), static_cast<long long>(n2),
            static_cast<long long>(t->ne[0]), static_cast<long long>(t->ne[1]),
            static_cast<long long>(t->ne[2]));
        throw std::runtime_error(buf);
    }
}

ggml_tensor* permute_cont(ggml_context* ctx, ggml_tensor* x, int a0, int a1, int a2, int a3) {
    return ggml_cont(ctx, ggml_permute(ctx, x, a0, a1, a2, a3));
}

GraphContext::GraphContext(size_t mem_size, size_t max_nodes) {
    ggml_init_params params = { mem_size, nullptr, true };
    ctx_ = ggml_init(params);
    if (!ctx_) {
        throw std::runtime_error("ggml 上下文创建失败");
    }
    graph_ = ggml_new_graph_custom(ctx_, max_nodes, false);
}

GraphContext::~GraphContext() {
    // ggml_free 释放上下文内存；计算图数据（元数据）在上下文中，无需单独释放
    if (ctx_) {
        ggml_free(ctx_);
        ctx_ = nullptr;
    }
}

ggml_tensor* GraphContext::make_input(ggml_type type, int n0, int n1, int n2, int n3,
    const char* name) {
    ggml_tensor* t = nullptr;
    if (n3 > 1) {
        t = ggml_new_tensor_4d(ctx_, type, n0, n1, n2, n3);
    }
    else if (n2 > 1) {
        t = ggml_new_tensor_3d(ctx_, type, n0, n1, n2);
    }
    else if (n1 > 1) {
        t = ggml_new_tensor_2d(ctx_, type, n0, n1);
    }
    else {
        t = ggml_new_tensor_1d(ctx_, type, n0);
    }
    if (name) ggml_set_name(t, name);
    ggml_set_input(t);
    return t;
}

ggml_tensor* to_f32(ggml_context* ctx, ggml_tensor* x) {
    if (!x) return nullptr;
    if (x->type == GGML_TYPE_F32) return x;
    return ggml_cast(ctx, x, GGML_TYPE_F32);
}

ggml_tensor* sliding_window(ggml_context* ctx, ggml_tensor* x, int64_t n) {
    if (n == 1) return x;
    const int64_t len = x->ne[0] - n + 1;
    if (len <= 0) {
        throw std::runtime_error("sliding_window: 输入过短");
    }
    // 兼容 ggml 0.23：无 ggml_split，手工拼多段视图
    ggml_tensor* result = nullptr;
    for (int64_t i = 0; i < n; ++i) {
        ggml_tensor* part = ggml_view_1d(ctx, x, len,
            static_cast<size_t>(i) * x->nb[0]);
        result = result ? ggml_concat(ctx, result, part, 0) : part;
    }
    return ggml_reshape_3d(ctx, result, len, n, x->ne[1] * x->ne[2]);
}

ggml_tensor* layer_norm(ggml_context* ctx, ggml_tensor* x,
    ggml_tensor* gamma, ggml_tensor* beta, float eps) {
    ggml_tensor* norm = ggml_norm(ctx, to_f32(ctx, x), eps);
    // x: [C, T, ...]（C 在 dim0），gamma/beta 为 [C]
    ggml_tensor* g = ggml_reshape_3d(ctx, to_f32(ctx, gamma), gamma->ne[0], 1, 1);
    ggml_tensor* b = ggml_reshape_3d(ctx, to_f32(ctx, beta), beta->ne[0], 1, 1);
    norm = mul_checked(ctx, norm, g);
    return add_checked(ctx, norm, b);
}

ggml_tensor* linear(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b) {
    ggml_tensor* xf = to_f32(ctx, x);
    if (getenv("RVC_DEBUG_LINEAR")) {
        fprintf(stderr, "[linear] w=%s w.ne=[%lld,%lld,%lld,%lld] x.ne=[%lld,%lld,%lld,%lld]\n",
            w->name,
            (long long)w->ne[0], (long long)w->ne[1], (long long)w->ne[2], (long long)w->ne[3],
            (long long)xf->ne[0], (long long)xf->ne[1], (long long)xf->ne[2], (long long)xf->ne[3]);
    }
    // 约定：GGUF 里线性权重为 PyTorch [N, K]，ggml 读取后 ne=[K, N]。
    // ggml_mul_mat(a, b) 计算 a^T @ b：a=[K, N] + b=[K, T] -> [N, T]，与 PyTorch
    // Linear(x) = x @ W^T（x=[T,K], W=[N,K]）完全一致，无需转置。
    // 注意：此处 K = x->ne[0]，即 x 必须为 [K, T] 布局。
    const int64_t K = xf->ne[0];
    if (w->ne[0] != K) {
        char buf[384];
        snprintf(buf, sizeof(buf),
            "linear: 维度不匹配 (w ne=[%lld,%lld,%lld,%lld] x ne=[%lld,%lld,%lld,%lld])",
            static_cast<long long>(w->ne[0]), static_cast<long long>(w->ne[1]),
            static_cast<long long>(w->ne[2]), static_cast<long long>(w->ne[3]),
            static_cast<long long>(xf->ne[0]), static_cast<long long>(xf->ne[1]),
            static_cast<long long>(xf->ne[2]), static_cast<long long>(xf->ne[3]));
        throw std::runtime_error(buf);
    }
    ggml_tensor* out = ggml_mul_mat(ctx, w, xf);   // [N, T, ...]
    if (b) {
        // bias [N] -> [N, 1] 沿时间维广播
        out = add_checked(ctx, out, ggml_reshape_2d(ctx, to_f32(ctx, b), b->ne[0], 1));
    }
    return to_f32(ctx, out);
}

// 展平布局 Conv1d（GGUF rvc.conv_flat）：weight [I*K, O]（memory: o -> i -> k）。
// 实现 = im2col(F32/F16) + mul_mat(量化权重作 src0) + 输出转回卷积布局。
// 该路径让 conv 权重可以真正参与量化（ggml 量化张量的行宽必须是 32 的倍数）。
ggml_tensor* conv1d_flat_impl(ggml_context* ctx, ggml_tensor* input, ggml_tensor* weight,
    ggml_tensor* bias, int stride, int padding, int dilation) {
    const int64_t I = input->ne[1];
    const int64_t O = weight->ne[1];
    if (weight->ne[2] != 1 || weight->ne[3] != 1 || I <= 0 || weight->ne[0] % I != 0) {
        char buf[256];
        snprintf(buf, sizeof(buf),
            "conv1d(flat): 权重/输入不匹配（w=[%lld,%lld,%lld,%lld] in=[%lld,%lld,%lld,%lld]）",
            static_cast<long long>(weight->ne[0]), static_cast<long long>(weight->ne[1]),
            static_cast<long long>(weight->ne[2]), static_cast<long long>(weight->ne[3]),
            static_cast<long long>(input->ne[0]), static_cast<long long>(input->ne[1]),
            static_cast<long long>(input->ne[2]), static_cast<long long>(input->ne[3]));
        throw std::runtime_error(buf);
    }
    const int64_t K = weight->ne[0] / I;

    // im2col 只读取 kernel 的形状（不读数据），用微型占位张量避免额外显存
    ggml_tensor* kernel = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, I, 1);
    ggml_set_name(kernel, "conv1d_flat_kernel");
    ggml_set_input(kernel);

    // cols: [K*I, OW, B]
    ggml_tensor* cols = ggml_im2col(ctx, kernel, to_f32(ctx, input),
        stride, 0, padding, 0, dilation, 0, false,
        weight->type == GGML_TYPE_F16 ? GGML_TYPE_F16 : GGML_TYPE_F32);
    const int64_t OW = cols->ne[1];
    const int64_t B = cols->ne[2];
    ggml_tensor* cols2 = ggml_reshape_2d(ctx, cols, cols->ne[0], OW * B);

    ggml_tensor* out = ggml_mul_mat(ctx, weight, cols2);   // [O, OW*B]
    out = ggml_reshape_3d(ctx, out, O, OW, B);             // [O, OW, B]
    out = permute_cont(ctx, out, 1, 0, 2, 3);              // [OW, O, B]
    if (bias) {
        out = add_checked(ctx, out,
            ggml_reshape_3d(ctx, to_f32(ctx, bias), 1, bias->ne[0], 1));
    }
    return out;
}

ggml_tensor* conv1d(ggml_context* ctx, ggml_tensor* input, ggml_tensor* weight,
    ggml_tensor* bias, int stride, int padding, int dilation, bool flat) {
    // ggml_conv_1d 要求 input [T, C, B]，weight ne = [k, C_in, C_out]
    if (getenv("RVC_DEBUG_CONV")) {
        fprintf(stderr, "[conv1d] w=%s w.ne=[%lld,%lld,%lld,%lld] in.ne=[%lld,%lld,%lld,%lld] s=%d p=%d flat=%d\n",
            weight->name,
            (long long)weight->ne[0], (long long)weight->ne[1],
            (long long)weight->ne[2], (long long)weight->ne[3],
            (long long)input->ne[0], (long long)input->ne[1],
            (long long)input->ne[2], (long long)input->ne[3], stride, padding, flat ? 1 : 0);
    }
    if (flat) {
        return conv1d_flat_impl(ctx, input, weight, bias, stride, padding, dilation);
    }
    ggml_tensor* out = ggml_conv_1d(ctx, weight, to_f32(ctx, input), stride, padding, dilation);
    out = to_f32(ctx, out);
    if (bias) {
        // out: [T_out, C_out, B]；bias [C_out] 沿 dim1 广播
        out = add_checked(ctx, out, ggml_reshape_3d(ctx, to_f32(ctx, bias), 1, bias->ne[0], 1));
    }
    return out;
}

ggml_tensor* conv1d_grouped(ggml_context* ctx, ggml_tensor* input, ggml_tensor* weight,
    ggml_tensor* bias, int groups, int stride, int padding, int dilation) {
    // ggml_conv_2d 同样不支持 groups；按组切片后逐组卷积再拼接。
    // 输入 [T, 1, C, B]，权重 [K, 1, C/g, OC/g]，输出 [T_out, 1, OC, B]
    const int64_t T = input->ne[0];
    const int64_t C = input->ne[1];
    const int64_t B = input->ne[2];
    const int64_t K = weight->ne[0];
    const int64_t icg = weight->ne[1];
    const int64_t oc = weight->ne[2];
    if (icg * groups != C || oc % groups != 0) {
        char buf[256];
        snprintf(buf, sizeof(buf),
            "conv1d_grouped: 通道不匹配 (C=%lld, ic/g=%lld, groups=%d, oc=%lld)",
            static_cast<long long>(C), static_cast<long long>(icg), groups,
            static_cast<long long>(oc));
        throw std::runtime_error(buf);
    }
    const int64_t ocg = oc / groups;

    ggml_tensor* in4 = ggml_reshape_4d(ctx, to_f32(ctx, input), T, 1, C, B);
    ggml_tensor* w4 = ggml_reshape_4d(ctx, weight, K, 1, icg, oc);

    ggml_tensor* cat = nullptr;
    for (int g = 0; g < groups; ++g) {
        ggml_tensor* in_g = ggml_view_4d(ctx, in4,
            T, 1, icg, B,
            in4->nb[1], in4->nb[2], in4->nb[3],
            static_cast<size_t>(g * icg) * in4->nb[2]);
        in_g = ggml_cont(ctx, in_g);   // im2col 需连续内存
        ggml_tensor* w_g = ggml_view_4d(ctx, w4,
            K, 1, icg, ocg,
            w4->nb[1], w4->nb[2], w4->nb[3],
            static_cast<size_t>(g * ocg) * w4->nb[3]);
        w_g = ggml_cont(ctx, w_g);
        ggml_tensor* out_g = ggml_conv_2d(ctx, w_g, in_g, stride, 1, 0, 0, dilation, 1);
        cat = cat ? ggml_concat(ctx, cat, out_g, 2) : out_g;
    }
    ggml_tensor* out = to_f32(ctx, cat);   // [T_out, 1, OC, B]
    if (bias) {
        out = add_checked(ctx, out, ggml_reshape_4d(ctx, to_f32(ctx, bias), 1, 1, bias->ne[0], 1));
    }
    return ggml_reshape_3d(ctx, out, out->ne[0], out->ne[2], out->ne[3]);
}

ggml_tensor* group_norm(ggml_context* ctx, ggml_tensor* x,
    ggml_tensor* w, ggml_tensor* b, int n_groups, float eps) {
    ggml_tensor* norm = ggml_group_norm(ctx, to_f32(ctx, x), n_groups, eps);
    // x: [W, H, C, N]；w/b [C] -> [1, 1, C, 1]
    const int64_t C = x->ne[2];
    norm = ggml_mul(ctx, norm, ggml_reshape_4d(ctx, w, 1, 1, C, 1));
    norm = ggml_add(ctx, norm, ggml_reshape_4d(ctx, b, 1, 1, C, 1));
    return norm;
}

ggml_tensor* get_rows(ggml_context* ctx, ggml_tensor* table, ggml_tensor* idx) {
    return to_f32(ctx, ggml_get_rows(ctx, table, idx));
}

// 相对位置 -> 绝对位置（对齐 RVC attentions.py，单头）
//   x: [2T-1, T]（相对 logits，ne0=相对位置），返回 [T, T]（绝对 scores）
//   Python 展平序：[tq, rel] 行主序（rel 在内），与 [2T-1, T] 内存序一致
ggml_tensor* relative_to_absolute(ggml_context* ctx, ggml_tensor* x, int64_t frames, int64_t heads, int64_t head_dim) {
    const int64_t T = frames;
    if (x->ne[0] != 2 * T - 1 || x->ne[1] != T) {
        throw std::runtime_error("relative_to_absolute: 输入形状不匹配");
    }
    // pad(rel, +1) -> [2T, T]
    ggml_tensor* y = ggml_pad(ctx, to_f32(ctx, x), 1, 0, 0, 0);
    // 展平（内存序与 Python view 一致）并尾部 pad T-1
    y = ggml_reshape_1d(ctx, y, (2 * T) * T);
    y = ggml_pad(ctx, y, static_cast<int>(T - 1), 0, 0, 0);
    // [2T-1, T+1]，取前 T 列与前 T 行 -> [T, T]
    y = ggml_reshape_2d(ctx, y, 2 * T - 1, T + 1);
    ggml_tensor* view = ggml_view_2d(ctx, y, T, T, y->nb[1], static_cast<size_t>(T - 1) * y->nb[0]);
    (void)heads;
    (void)head_dim;
    return ggml_cont(ctx, view);
}

// 绝对位置 -> 相对位置（对齐 RVC attentions.py，单头）
//   x: [T, T]（注意力权重 p_attn），返回 [2T-1, T]
ggml_tensor* absolute_to_relative(ggml_context* ctx, ggml_tensor* x, int64_t frames, int64_t heads, int64_t head_dim) {
    const int64_t T = frames;
    if (x->ne[0] != T || x->ne[1] != T) {
        throw std::runtime_error("absolute_to_relative: 输入形状不匹配");
    }
    // 尾部 pad T-1 -> [2T-1, T]
    ggml_tensor* y = ggml_pad(ctx, to_f32(ctx, x), static_cast<int>(T - 1), 0, 0, 0);
    // 展平后前插 T 个 0 -> [2T*T]，再 [2T, T]，取 [1, 2T) 行 -> [2T-1, T]
    y = ggml_reshape_1d(ctx, y, (2 * T - 1) * T);
    ggml_tensor* zeros = ggml_scale(ctx, ggml_view_1d(ctx, y, T, 0), 0.0f);
    y = ggml_concat(ctx, zeros, y, 0);
    y = ggml_reshape_2d(ctx, y, 2 * T, T);
    ggml_tensor* view = ggml_view_2d(ctx, y, 2 * T - 1, T, y->nb[1], y->nb[0]);
    (void)heads;
    (void)head_dim;
    return ggml_cont(ctx, view);
}

// ------------------------------------------------------------
// VITS 专用辅助
// ------------------------------------------------------------

ggml_tensor* conv1d_ct(ggml_context* ctx, ggml_tensor* x_ct, ggml_tensor* weight,
    ggml_tensor* bias, int stride, int padding, int dilation, bool flat) {
    ggml_tensor* xt = permute_cont(ctx, x_ct, 1, 0, 2, 3);            // [T, C]
    xt = ggml_reshape_3d(ctx, xt, xt->ne[0], xt->ne[1], 1);           // [T, C, 1]
    xt = conv1d(ctx, xt, weight, bias, stride, padding, dilation, flat);  // [T_out, C_out, 1]
    return permute_cont(ctx, xt, 1, 0, 2, 3);                         // [C_out, T_out]
}

ggml_tensor* conv1x1_ct(ggml_context* ctx, ggml_tensor* x_ct, ggml_tensor* weight,
    ggml_tensor* bias) {
    // Conv1d 权重 [1, K, N]（ggml 序）-> Linear 权重 [K, N]
    ggml_tensor* w = ggml_reshape_2d(ctx, weight, weight->ne[1], weight->ne[2]);
    return linear(ctx, x_ct, w, bias);
}

ggml_tensor* slice_channels(ggml_context* ctx, ggml_tensor* x_ct, int64_t begin, int64_t len) {
    return ggml_view_2d(ctx, x_ct, len, x_ct->ne[1], x_ct->nb[1],
        static_cast<size_t>(begin) * x_ct->nb[0]);
}

ggml_tensor* make_const_f32(ggml_context* ctx, int64_t n0, int64_t n1, const char* name) {
    ggml_tensor* t = n1 > 1
        ? ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n0, n1)
        : ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n0);
    if (name) {
        ggml_set_name(t, name);
    }
    ggml_set_input(t);
    return t;
}

} // namespace vits
} // namespace rvc
