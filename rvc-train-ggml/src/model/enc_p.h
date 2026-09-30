// RVC-Train-GGML - enc_p（TextEncoder，相对位置注意力）头文件（src/model/enc_p.h）
//
// 对应运行时 `infer_pack/models.py:19-81` 的 `TextEncoder` 与
// `attentions.py:14-78`（Encoder）/`:166-387`（相对位置 MultiHeadAttention）：
//   phone → emb_phone(Linear)；pitch → emb_pitch(get_rows)；相加 ×√H → LeakyReLU(0.1)；
//   n_layers 层 [相对位置多头自注意力（window=10）+ Post-LayerNorm + FFN(k3,pad1,ReLU)]；
//   末尾 ×x_mask → proj(1×1) → stats ×x_mask → split m_p/logs_p。
//
// 内部布局：主流程 [T,C,B]（与 wn/flow/dec 一致；torch [B,C,T] 的 ne 反序）；
// 仅 emb（torch [B,T,hidden]）为 [C,T,B]，进入 Encoder 前转置一次。
// 相对位置表由 emb_rel_k/v **图内**扩展（pad_ext 补零）→ 反传梯度可回相对嵌入权重。
// 注意力中间量 [2T-1,T,H,B] / [T,T,H,B]（ne0 = torch 最后一维，对齐 attentions.py 的
// relative→absolute / absolute→relative 变换，见 enc_p.cpp 实现注释）。
//
// 权重为训练态 GGUF（张量名保持 torch state_dict 原名；手工装配 POD，不调 init helper，见 Q13）。
#pragma once

#include "model/model.h"

#include "traincpp/traincpp.h"

#include <cstdint>
#include <string>
#include <vector>

namespace rvc_train {
namespace model {

struct EncPConfig {
    std::string prefix = "enc_p";   // 参数名前缀（torch 键名）
    int64_t in_channels = 0;        // 256（v1）/ 768（v2）
    int64_t hidden_channels = 0;    // 192
    int64_t filter_channels = 0;    // 768
    int64_t n_heads = 0;            // 2
    int64_t n_layers = 0;           // 6
    int64_t kernel_size = 3;        // FFN 核（同长 padding）
    int64_t window_size = 10;       // 相对位置窗口（2W+1=21）
    int64_t out_channels = 0;       // proj 输出的一半（stats 通道 = 2*out）
};

// 单层（enc_p.encoder.attn_layers/ffn_layers/norm_layers_1/norm_layers_2 的第 i 项）
struct EncPAttnLayer {
    traincpp::Conv1d conv_q, conv_k, conv_v, conv_o;   // 1×1
    traincpp::Tensor* emb_rel_k = nullptr;             // [Dk, 2W+1, 1]
    traincpp::Tensor* emb_rel_v = nullptr;             // [Dk, 2W+1, 1]
    traincpp::LayerNorm norm1, norm2;                  // gamma/beta
    traincpp::Conv1d ffn_conv1, ffn_conv2;             // k=3，pad 1
};

struct EncPModule {
    EncPConfig cfg;
    traincpp::Linear emb_phone;          // [in,hidden]
    traincpp::Tensor* emb_pitch = nullptr;  // [hidden,256]（get_rows 表；f0 模型必需）
    std::vector<EncPAttnLayer> layers;   // n_layers 个
    traincpp::Conv1d proj;               // 1×1: hidden → 2*out
};

// 按名装配（仅填 POD，不改张量名）；失败返回 false 并写 err。
bool enc_p_load(const ModelParams& params, const EncPConfig& cfg, EncPModule* out,
                std::string* err);

// 前向捕获（对拍用；不需要时传 nullptr）。张量布局（golden ne 反序）：
//   emb [C,T,B]（torch [B,T,C]）；x/q/k/v/attn_y/norm1/ffn/norm2/encoder_out [T,C,B]（torch [B,C,T]）；
//   rel_logits [2T-1,T,H,B]；scores_local/p_attn [T,T,H,B]；proj [T,2*out,B]
struct EncPCaptures {
    traincpp::Tensor* emb = nullptr;
    traincpp::Tensor* x = nullptr;   // Encoder 输入（emb × x_mask）[T,C,B]
    std::vector<traincpp::Tensor*> q, k, v;
    std::vector<traincpp::Tensor*> scores;         // add(scores_local) 后、mask 前 [T,T,H,B]
    std::vector<traincpp::Tensor*> scores_masked;  // 复刻 masked_fill 后 [T,T,H,B]
    std::vector<traincpp::Tensor*> attn_out;       // conv_o 前（p@v + rel_w@rel_v）[Dk,T,H,B]
    std::vector<traincpp::Tensor*> merged;         // 头合并后 [T,C,B]
    std::vector<traincpp::Tensor*> rel_logits;
    std::vector<traincpp::Tensor*> scores_local;
    std::vector<traincpp::Tensor*> p_attn;
    std::vector<traincpp::Tensor*> attn_y;
    std::vector<traincpp::Tensor*> norm1;
    std::vector<traincpp::Tensor*> ffn;
    std::vector<traincpp::Tensor*> norm2;
    traincpp::Tensor* encoder_out = nullptr;
    traincpp::Tensor* proj = nullptr;
};

struct EncPOutputs {
    traincpp::Tensor* m_p = nullptr;     // stats 前半（[out,T,B] 视图）
    traincpp::Tensor* logs_p = nullptr;  // stats 后半（[out,T,B] 视图）
};

// 输入：
//   phone   [Cin,T,B] F32（HuBERT 特征；需连续）
//   pitch   [T,B] I32（f0 模型；可为 nullptr = non-f0）
//   x_mask  [T,1,B] F32
//   attn_valid [T,T,1,B] F32（0/1，由调用方在 buffer 分配后填充；无效位复刻 torch
//              `masked_fill(mask==0, -1e4)` 的**替换**语义：scores·M + (M−1)·1e4）
// 输出：m_p/logs_p（stats 的通道切分视图）。失败返回空。
EncPOutputs enc_p_forward(traincpp::Context* ctx, const EncPModule& m, traincpp::Tensor* phone,
                          traincpp::Tensor* pitch, traincpp::Tensor* x_mask,
                          traincpp::Tensor* attn_valid, EncPCaptures* caps = nullptr);

} // namespace model
} // namespace rvc_train
