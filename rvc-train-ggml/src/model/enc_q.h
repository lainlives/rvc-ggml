// RVC-Train-GGML - enc_q（PosteriorEncoder）头文件（src/model/enc_q.h）
//
// 对应运行时 `infer_pack/models.py:148-189` 的 `PosteriorEncoder`（enc_q 为训练专属，推理不使用）：
//   x_mask = sequence_mask(x_lengths)            （[B,1,T]）
//   x = pre(x)·mask                              （pre：1×1 Conv1d，spec_channels→hidden）
//   x = enc(x, mask, g)                          （WN：16 层 k=5 d=1 gin=256）
//   stats = proj(x)·mask                         （proj：1×1 Conv1d，hidden→2·out）
//   m_q/logs_q = split(stats)；
//   z = (m_q + noise·exp(logs_q))·mask           （noise 由调用方注入，保证与 golden 同源）
//
// 内部布局 [T,C,B]（torch [B,C,T] 的 ne 反序）；g [1,gin,B]（torch [B,gin,1]）。
// 权重：pre/proj 为普通 Conv1d；enc 为 weight_norm（weight_g/weight_v 不折叠）；
// 手工装配 POD、不调 init helper（Q13），张量名保持 GGUF 原名。
#pragma once

#include "model/model.h"
#include "model/wn.h"

#include "traincpp/traincpp.h"

#include <cstdint>
#include <string>

namespace rvc_train {
namespace model {

struct EncQConfig {
    std::string prefix = "enc_q";   // 参数名前缀（torch 键名）
    int64_t in_channels = 0;        // 1025（spec_channels）
    int64_t out_channels = 0;       // 192（inter_channels）
    int64_t hidden_channels = 0;    // 192
    int64_t kernel_size = 5;        // WN 卷积核
    int64_t dilation_rate = 1;
    int64_t n_layers = 16;
    int64_t gin_channels = 0;       // 256
};

struct EncQModule {
    EncQConfig cfg;
    traincpp::Conv1d pre;       // 1×1: in → hidden（无 weight_norm）
    WnModule enc;               // prefix = cfg.prefix + ".enc"
    traincpp::Conv1d proj;      // 1×1: hidden → 2*out（无 weight_norm）
};

// 按名装配（仅填 POD，不改张量名）；失败返回 false 并写 err。
bool enc_q_load(const ModelParams& params, const EncQConfig& cfg, EncQModule* out,
                std::string* err);

// 前向捕获（对拍用；不需要时传 nullptr）。布局（golden ne 反序）：
//   pre/enc [T,H,B]；stats [T,2*out,B]（前两者为乘 mask 前/后的对应中间量）；
//   cond/x_in0/acts0 来自内部 WN（cond [1,2H·L,B]、x_in0 [T,2H,B]、acts0 [T,H,B]）
struct EncQCaptures {
    traincpp::Tensor* pre = nullptr;     // pre 卷积输出（乘 mask 前）
    traincpp::Tensor* cond = nullptr;    // WN cond_layer 输出
    traincpp::Tensor* x_in0 = nullptr;   // WN 第 0 层 in_layer 输出
    traincpp::Tensor* acts0 = nullptr;   // WN 第 0 层激活
    traincpp::Tensor* enc = nullptr;     // WN 输出（已乘 mask）
    traincpp::Tensor* stats = nullptr;   // proj 输出（乘 mask 前）
};

struct EncQOutputs {
    traincpp::Tensor* m_q = nullptr;     // stats 前半（[out,T,B] 视图）
    traincpp::Tensor* logs_q = nullptr;  // stats 后半（[out,T,B] 视图）
    traincpp::Tensor* z = nullptr;       // (m_q + noise·exp(logs_q))·mask
};

// 输入：
//   spec   [in,T,B] F32（线性谱；需连续）
//   x_mask [T,1,B] F32
//   g      [1,gin,B] F32（可为 nullptr = 无条件）
//   noise  [T,out,B] F32（确定性噪声，由调用方注入；与 golden q_noise 同源）
// 输出：m_q/logs_q（stats 的通道切分视图）与 z。失败返回空。
EncQOutputs enc_q_forward(traincpp::Context* ctx, const EncQModule& m, traincpp::Tensor* spec,
                          traincpp::Tensor* x_mask, traincpp::Tensor* g, traincpp::Tensor* noise,
                          EncQCaptures* caps = nullptr);

} // namespace model
} // namespace rvc_train
