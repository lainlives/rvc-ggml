// RVC-Train-GGML - training losses implementation (src/train/losses.cpp)
#include "train/losses.h"

namespace rvc_train {
namespace train {

using namespace traincpp;

namespace {

// All-element mean -> [1] (cont + flatten first, avoiding the mean semantics that only reduces over ne0)
Tensor* mean_all(Context* ctx, Tensor* t) {
    Tensor* c = cont(ctx, t);
    if (c == nullptr) {
        return nullptr;
    }
    Tensor* flat = reshape_1d(ctx, c, tensor_nelements(c));
    if (flat == nullptr) {
        return nullptr;
    }
    return mean(ctx, flat);
}

// Add the two Σ terms (scalar [1])
Tensor* add_scalar(Context* ctx, Tensor* a, Tensor* b) {
    if (a == nullptr) {
        return b;
    }
    if (b == nullptr) {
        return a;
    }
    return add(ctx, a, b);
}

} // namespace

Tensor* loss_l1_mean(Context* ctx, Tensor* input, Tensor* target) {
    return mean_all(ctx, abs_op(ctx, sub(ctx, input, target)));
}

Tensor* loss_lsgan_discriminator(Context* ctx, const std::vector<Tensor*>& dr,
                                 const std::vector<Tensor*>& dg) {
    Tensor* loss = nullptr;
    const size_t n = dr.size() < dg.size() ? dr.size() : dg.size();
    for (size_t i = 0; i < n; ++i) {
        // mean((1 - dr)²)
        Tensor* r = mean_all(ctx, sqr(ctx, scale_bias(ctx, dr[i], -1.0f, 1.0f)));
        // mean(dg²)
        Tensor* g = mean_all(ctx, sqr(ctx, dg[i]));
        loss = add_scalar(ctx, add_scalar(ctx, loss, r), g);
    }
    return loss;
}

Tensor* loss_lsgan_generator(Context* ctx, const std::vector<Tensor*>& dg) {
    Tensor* loss = nullptr;
    for (Tensor* t : dg) {
        Tensor* g = mean_all(ctx, sqr(ctx, scale_bias(ctx, t, -1.0f, 1.0f)));
        loss = add_scalar(ctx, loss, g);
    }
    return loss;
}

Tensor* loss_feature_matching(Context* ctx,
                              const std::vector<std::vector<Tensor*>>& fmap_r,
                              const std::vector<std::vector<Tensor*>>& fmap_g) {
    Tensor* loss = nullptr;
    const size_t n = fmap_r.size() < fmap_g.size() ? fmap_r.size() : fmap_g.size();
    for (size_t i = 0; i < n; ++i) {
        const size_t m = fmap_r[i].size() < fmap_g[i].size() ? fmap_r[i].size() : fmap_g[i].size();
        for (size_t j = 0; j < m; ++j) {
            Tensor* d = abs_op(ctx, sub(ctx, fmap_r[i][j], fmap_g[i][j]));
            loss = add_scalar(ctx, loss, mean_all(ctx, d));
        }
    }
    if (loss == nullptr) {
        return nullptr;
    }
    // Official feature_loss appends ×2 at the end
    return scale(ctx, loss, 2.0f);
}

Tensor* loss_kl(Context* ctx, Tensor* z_p, Tensor* logs_q, Tensor* m_p, Tensor* logs_p,
                Tensor* z_mask) {
    // kl = logs_p - logs_q - 0.5 + 0.5·(z_p - m_p)²·exp(-2·logs_p)
    Tensor* base = scale_bias(ctx, sub(ctx, logs_p, logs_q), 1.0f, -0.5f);
    Tensor* diff = sqr(ctx, sub(ctx, z_p, m_p));
    Tensor* inv_var = exp_op(ctx, scale(ctx, logs_p, -2.0f));
    Tensor* term = scale(ctx, mul(ctx, diff, inv_var), 0.5f);
    Tensor* kl = add(ctx, base, term);
    Tensor* masked = mul(ctx, kl, z_mask);
    Tensor* num = sum(ctx, masked);
    Tensor* den = sum(ctx, z_mask);
    return div(ctx, num, den);
}

} // namespace train
} // namespace rvc_train
