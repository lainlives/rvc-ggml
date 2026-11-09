// RVC-Train-GGML - training losses (src/train/losses.h)
//
// Semantic baseline: official 81eed5e `train/losses.py` and `train/train.py:517-521` (training side, not inference).
// All return scalar (F32 [1]) loss tensors; differentiable within the graph.
#pragma once

#include "traincpp/traincpp.h"

#include <vector>

namespace rvc_train {
namespace train {

// L1 (mean|input-target|); caller multiplies by c_mel as needed (official F.l1_loss)
traincpp::Tensor* loss_l1_mean(traincpp::Context* ctx, traincpp::Tensor* input,
                               traincpp::Tensor* target);

// LSGAN(D): Σ_i [mean((1-dr_i)²) + mean(dg_i²)] (official discriminator_loss)
traincpp::Tensor* loss_lsgan_discriminator(traincpp::Context* ctx,
                                           const std::vector<traincpp::Tensor*>& dr,
                                           const std::vector<traincpp::Tensor*>& dg);

// LSGAN(G): Σ_i mean((1-dg_i)²) (official generator_loss)
traincpp::Tensor* loss_lsgan_generator(traincpp::Context* ctx,
                                       const std::vector<traincpp::Tensor*>& dg);

// Feature matching: 2·Σ_{i,j} mean(|fmap_r[i][j] - fmap_g[i][j]|)
// (official feature_loss; no gradients are built for the fmap_r side at the call site -- detach semantics in training)
traincpp::Tensor* loss_feature_matching(traincpp::Context* ctx,
                                        const std::vector<std::vector<traincpp::Tensor*>>& fmap_r,
                                        const std::vector<std::vector<traincpp::Tensor*>>& fmap_g);

// KL: Σ(kl·z_mask)/Σ(z_mask),
//   kl = logs_p - logs_q - 0.5 + 0.5·(z_p - m_p)²·exp(-2·logs_p) (official kl_loss)
traincpp::Tensor* loss_kl(traincpp::Context* ctx, traincpp::Tensor* z_p, traincpp::Tensor* logs_q,
                          traincpp::Tensor* m_p, traincpp::Tensor* logs_p, traincpp::Tensor* z_mask);

} // namespace train
} // namespace rvc_train
