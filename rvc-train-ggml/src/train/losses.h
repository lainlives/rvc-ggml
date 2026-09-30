// RVC-Train-GGML - 训练损失（src/train/losses.h）
//
// 语义基准：官方 81eed5e `train/losses.py` 与 `train/train.py:517-521`（训练侧，非推理）。
// 全部返回标量（F32 [1]）损失张量，图内可反传。
#pragma once

#include "traincpp/traincpp.h"

#include <vector>

namespace rvc_train {
namespace train {

// L1（mean|input-target|）；调用方按需乘 c_mel（官方 F.l1_loss）
traincpp::Tensor* loss_l1_mean(traincpp::Context* ctx, traincpp::Tensor* input,
                               traincpp::Tensor* target);

// LSGAN(D)：Σ_i [mean((1-dr_i)²) + mean(dg_i²)]（官方 discriminator_loss）
traincpp::Tensor* loss_lsgan_discriminator(traincpp::Context* ctx,
                                           const std::vector<traincpp::Tensor*>& dr,
                                           const std::vector<traincpp::Tensor*>& dg);

// LSGAN(G)：Σ_i mean((1-dg_i)²)（官方 generator_loss）
traincpp::Tensor* loss_lsgan_generator(traincpp::Context* ctx,
                                       const std::vector<traincpp::Tensor*>& dg);

// Feature matching：2·Σ_{i,j} mean(|fmap_r[i][j] - fmap_g[i][j]|)
// （官方 feature_loss；fmap_r 侧在调用侧不建梯度，训练中为 detach 语义）
traincpp::Tensor* loss_feature_matching(traincpp::Context* ctx,
                                        const std::vector<std::vector<traincpp::Tensor*>>& fmap_r,
                                        const std::vector<std::vector<traincpp::Tensor*>>& fmap_g);

// KL：Σ(kl·z_mask)/Σ(z_mask)，
//   kl = logs_p - logs_q - 0.5 + 0.5·(z_p - m_p)²·exp(-2·logs_p)（官方 kl_loss）
traincpp::Tensor* loss_kl(traincpp::Context* ctx, traincpp::Tensor* z_p, traincpp::Tensor* logs_q,
                          traincpp::Tensor* m_p, traincpp::Tensor* logs_p, traincpp::Tensor* z_mask);

} // namespace train
} // namespace rvc_train
