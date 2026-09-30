// RVC-Train-GGML - enc_p（TextEncoder，相对位置注意力）实现（src/model/enc_p.cpp）
//
// 相对位置注意力的 ne 布局推导（与RVC-GGML `src/vits/text_encoder.cpp` 的已验证结构同源）：
//   torch 的 p/scores 等张量最后一维对应本库的 ne0；相对↔绝对变换（attentions.py:327-374）
//   只作用于 torch 最后两维，因此在 ne 布局下等价于对 (ne0, ne1) 做 pad+reshape+view：
//     relative_to_absolute: [2T-1,T,..] → pad ne0 右 1 → [2T,T,..] → reshape 展平 ne0/ne1
//       → pad ne0 右 T-1（torch F.pad [0,l-1]）→ reshape [2T-1,T+1,..]
//       → view ne1[:T]、ne0 [T-1, T-1+T) → [T,T,..]
//     absolute_to_relative: [T,T,..] → pad ne0 右 T-1 → [2T-1,T,..] → reshape 展平
//       → pad ne0 左 T（前插 T 个 0）→ reshape [2T,T,..] → view ne0 [1, 2T) → [2T-1,T,..]
//   其中 reshape 展平 ne0/ne1 时必须保留外层的 (H,B) 平面（每个平面独立做变换）。
#include "model/enc_p.h"

#include <cmath>
#include <cstdio>
#include <utility>

namespace rvc_train {
namespace model {

namespace {

using namespace traincpp;

constexpr float kLayerNormEps = 1.0e-5f;
constexpr float kLeakySlope = 0.1f;

Tensor* require_param(const ModelParams& params, const std::string& name, std::string* err) {
    Tensor* t = params.find(name.c_str());
    if (t == nullptr && err != nullptr) {
        *err = "缺失参数: " + name;
    }
    return t;
}

void wire_conv1d(Conv1d* layer, Tensor* weight, Tensor* bias, int64_t padding, const char* name) {
    layer->weight       = weight;
    layer->bias         = bias;
    layer->has_bias     = (bias != nullptr);
    layer->kernel_size  = weight->ne[0];
    layer->in_channels  = weight->ne[1];
    layer->out_channels = weight->ne[2];
    layer->stride       = 1;
    layer->padding      = padding;
    layer->dilation     = 1;
    layer->groups       = 1;
    std::snprintf(layer->name, MAX_NAME, "%s", name);
}

void wire_linear(Linear* layer, Tensor* weight, Tensor* bias, const char* name) {
    layer->weight       = weight;
    layer->bias         = bias;
    layer->has_bias     = (bias != nullptr);
    layer->in_features  = weight->ne[0];
    layer->out_features = weight->ne[1];
    std::snprintf(layer->name, MAX_NAME, "%s", name);
}

void wire_layernorm(LayerNorm* layer, Tensor* weight, Tensor* bias, int64_t normalized_size,
                    const char* name) {
    layer->weight          = weight;
    layer->bias            = bias;
    layer->has_bias        = (bias != nullptr);
    layer->normalized_size = normalized_size;
    layer->eps             = kLayerNormEps;
    std::snprintf(layer->name, MAX_NAME, "%s", name);
}

// 头切分：[T,C,B] → reshape [T,Dk,H,B] → permute → [Dk,T,H,B]
// （通道序与 torch `view(b,H,Dk,t)` 一致：c = h·Dk + d = d + Dk·h）
Tensor* head_split(Context* ctx, Tensor* x, int64_t frames, int64_t dk, int64_t heads,
                   int64_t batch) {
    Tensor* v = reshape_4d(ctx, x, frames, dk, heads, batch);
    if (v == nullptr) {
        return nullptr;
    }
    return cont(ctx, permute(ctx, v, 1, 0, 2, 3));
}

// 头合并：[Dk,T,H,B] → transpose [T,Dk,H,B] → reshape [T,C,B]（c = d + Dk·h）
Tensor* head_merge(Context* ctx, Tensor* out, int64_t frames, int64_t channels, int64_t batch) {
    Tensor* p = cont(ctx, transpose(ctx, out));
    if (p == nullptr) {
        return nullptr;
    }
    return reshape_3d(ctx, p, frames, channels, batch);
}

// 相对 logits [2T-1,T,H,B] → 绝对 scores_local [T,T,H,B]（attentions.py:327-352）：
//   pad ne0 右 1 → [2T,T,H,B]；展平 ne0/ne1；pad ne0 右 T-1；reshape [2T-1,T+1,H,B]；
//   view ne1[:T]、ne0[T-1,T-1+T) → [T,T,H,B]
Tensor* relative_to_absolute(Context* ctx, Tensor* x, int64_t frames) {
    const int64_t T = frames;
    const int64_t H = x->ne[2];
    const int64_t B = x->ne[3];
    Tensor* y = pad_ext(ctx, x, 0, 1, 0, 0, 0, 0, 0, 0, PAD_ZERO);   // [2T,T,H,B]
    if (y == nullptr) {
        return nullptr;
    }
    y = reshape_3d(ctx, y, 2 * T * T, H, B);                         // 展平 ne0/ne1（保留平面）
    if (y == nullptr) {
        return nullptr;
    }
    y = pad_ext(ctx, y, 0, (int) (T - 1), 0, 0, 0, 0, 0, 0, PAD_ZERO);   // 右填充（torch F.pad [0,l-1]）
    if (y == nullptr) {
        return nullptr;
    }
    y = reshape_4d(ctx, y, 2 * T - 1, T + 1, H, B);
    if (y == nullptr) {
        return nullptr;
    }
    Tensor* v = view_4d(ctx, y, T, T, H, B, (size_t) (T - 1) * y->nb[0]);
    return cont(ctx, v);
}

// 绝对 p_attn [T,T,H,B] → 相对权重 [2T-1,T,H,B]（attentions.py:354-374）
Tensor* absolute_to_relative(Context* ctx, Tensor* x, int64_t frames, int64_t heads, int64_t batch) {
    const int64_t T = frames;
    Tensor* y = pad_ext(ctx, x, 0, (int) (T - 1), 0, 0, 0, 0, 0, 0, PAD_ZERO);   // [2T-1,T,H,B]
    if (y == nullptr) {
        return nullptr;
    }
    y = reshape_3d(ctx, y, (2 * T - 1) * T, heads, batch);
    if (y == nullptr) {
        return nullptr;
    }
    y = pad_ext(ctx, y, (int) T, 0, 0, 0, 0, 0, 0, 0, PAD_ZERO);                 // [2T*T,H,B]
    if (y == nullptr) {
        return nullptr;
    }
    y = reshape_4d(ctx, y, 2 * T, T, heads, batch);
    if (y == nullptr) {
        return nullptr;
    }
    Tensor* v = view_4d(ctx, y, 2 * T - 1, T, heads, batch, y->nb[0]);
    return cont(ctx, v);
}

} // namespace

bool enc_p_load(const ModelParams& params, const EncPConfig& cfg, EncPModule* out,
                std::string* err) {
    if (out == nullptr) {
        if (err != nullptr) {
            *err = "enc_p 输出为空";
        }
        return false;
    }
    if (cfg.in_channels <= 0 || cfg.hidden_channels <= 0 || cfg.filter_channels <= 0 ||
        cfg.n_heads <= 0 || cfg.hidden_channels % cfg.n_heads != 0 || cfg.n_layers <= 0 ||
        cfg.kernel_size <= 0 || cfg.kernel_size % 2 == 0 || cfg.window_size <= 0 ||
        cfg.out_channels <= 0) {
        if (err != nullptr) {
            *err = "enc_p 配置非法（通道/头数 > 0 且 hidden 可被 heads 整除；kernel 为正奇数；"
                   "window > 0）";
        }
        return false;
    }

    EncPModule m;
    m.cfg = cfg;
    const int64_t hidden = cfg.hidden_channels;
    const int64_t dk     = hidden / cfg.n_heads;

    Tensor* w = require_param(params, cfg.prefix + ".emb_phone.weight", err);
    if (w == nullptr) {
        return false;
    }
    if (w->ne[0] != cfg.in_channels || w->ne[1] != hidden) {
        if (err != nullptr) {
            *err = cfg.prefix + ".emb_phone.weight 形状与配置不符（期望 ne=[" +
                   std::to_string(cfg.in_channels) + "," + std::to_string(hidden) + "]）";
        }
        return false;
    }
    wire_linear(&m.emb_phone, w, params.find((cfg.prefix + ".emb_phone.bias").c_str()),
                "enc_p.emb_phone");

    m.emb_pitch = params.find((cfg.prefix + ".emb_pitch.weight").c_str());
    if (m.emb_pitch != nullptr &&
        (m.emb_pitch->ne[0] != hidden || m.emb_pitch->ne[1] != 256)) {
        if (err != nullptr) {
            *err = cfg.prefix + ".emb_pitch.weight 形状与配置不符（期望 ne=[hidden,256]）";
        }
        return false;
    }

    m.layers.resize((size_t) cfg.n_layers);
    for (int64_t i = 0; i < cfg.n_layers; ++i) {
        const std::string ibase = std::to_string(i);
        EncPAttnLayer& L = m.layers[(size_t) i];

        const std::string ab = cfg.prefix + ".encoder.attn_layers." + ibase;
        Tensor* qw = require_param(params, ab + ".conv_q.weight", err);
        if (qw == nullptr) {
            return false;
        }
        Tensor* kw = require_param(params, ab + ".conv_k.weight", err);
        if (kw == nullptr) {
            return false;
        }
        Tensor* vw = require_param(params, ab + ".conv_v.weight", err);
        if (vw == nullptr) {
            return false;
        }
        Tensor* ow = require_param(params, ab + ".conv_o.weight", err);
        if (ow == nullptr) {
            return false;
        }
        wire_conv1d(&L.conv_q, qw, params.find((ab + ".conv_q.bias").c_str()), 0, "enc_p.conv_q");
        wire_conv1d(&L.conv_k, kw, params.find((ab + ".conv_k.bias").c_str()), 0, "enc_p.conv_k");
        wire_conv1d(&L.conv_v, vw, params.find((ab + ".conv_v.bias").c_str()), 0, "enc_p.conv_v");
        wire_conv1d(&L.conv_o, ow, params.find((ab + ".conv_o.bias").c_str()), 0, "enc_p.conv_o");
        if (qw->ne[0] != 1 || qw->ne[1] != hidden || qw->ne[2] != hidden ||
            kw->ne[0] != 1 || kw->ne[1] != hidden || kw->ne[2] != hidden ||
            vw->ne[0] != 1 || vw->ne[1] != hidden || vw->ne[2] != hidden ||
            ow->ne[0] != 1 || ow->ne[1] != hidden || ow->ne[2] != hidden) {
            if (err != nullptr) {
                *err = ab + " 卷积权重形状与配置不符（期望 [1,hidden,hidden]）";
            }
            return false;
        }

        L.emb_rel_k = require_param(params, ab + ".emb_rel_k", err);
        if (L.emb_rel_k == nullptr) {
            return false;
        }
        L.emb_rel_v = require_param(params, ab + ".emb_rel_v", err);
        if (L.emb_rel_v == nullptr) {
            return false;
        }
        const int64_t window = 2 * cfg.window_size + 1;
        if (L.emb_rel_k->ne[0] != dk || L.emb_rel_k->ne[1] != window ||
            L.emb_rel_v->ne[0] != dk || L.emb_rel_v->ne[1] != window) {
            if (err != nullptr) {
                *err = ab + ".emb_rel_k/v 形状与配置不符（期望 ne=[Dk,2W+1,1]）";
            }
            return false;
        }

        const std::string n1 = cfg.prefix + ".encoder.norm_layers_1." + ibase;
        const std::string n2 = cfg.prefix + ".encoder.norm_layers_2." + ibase;
        Tensor* g1 = require_param(params, n1 + ".gamma", err);
        if (g1 == nullptr) {
            return false;
        }
        Tensor* g2 = require_param(params, n2 + ".gamma", err);
        if (g2 == nullptr) {
            return false;
        }
        wire_layernorm(&L.norm1, g1, params.find((n1 + ".beta").c_str()), hidden, "enc_p.norm1");
        wire_layernorm(&L.norm2, g2, params.find((n2 + ".beta").c_str()), hidden, "enc_p.norm2");
        if (g1->ne[0] != hidden || g2->ne[0] != hidden) {
            if (err != nullptr) {
                *err = n1 + "/norm_layers_2 gamma 形状与配置不符（期望 [hidden]）";
            }
            return false;
        }

        const std::string fb = cfg.prefix + ".encoder.ffn_layers." + ibase;
        Tensor* f1 = require_param(params, fb + ".conv_1.weight", err);
        if (f1 == nullptr) {
            return false;
        }
        Tensor* f2 = require_param(params, fb + ".conv_2.weight", err);
        if (f2 == nullptr) {
            return false;
        }
        wire_conv1d(&L.ffn_conv1, f1, params.find((fb + ".conv_1.bias").c_str()),
                    cfg.kernel_size / 2, "enc_p.ffn1");
        wire_conv1d(&L.ffn_conv2, f2, params.find((fb + ".conv_2.bias").c_str()),
                    cfg.kernel_size / 2, "enc_p.ffn2");
        if (f1->ne[0] != cfg.kernel_size || f1->ne[1] != hidden ||
            f1->ne[2] != cfg.filter_channels || f2->ne[0] != cfg.kernel_size ||
            f2->ne[1] != cfg.filter_channels || f2->ne[2] != hidden) {
            if (err != nullptr) {
                *err = fb + " 卷积权重形状与配置不符（conv_1 [k,hidden,filter] / "
                       "conv_2 [k,filter,hidden]）";
            }
            return false;
        }
    }

    Tensor* pw = require_param(params, cfg.prefix + ".proj.weight", err);
    if (pw == nullptr) {
        return false;
    }
    wire_conv1d(&m.proj, pw, params.find((cfg.prefix + ".proj.bias").c_str()), 0, "enc_p.proj");
    if (pw->ne[0] != 1 || pw->ne[1] != hidden || pw->ne[2] != 2 * cfg.out_channels) {
        if (err != nullptr) {
            *err = cfg.prefix + ".proj.weight 形状与配置不符（期望 [1,hidden,2*out]）";
        }
        return false;
    }

    *out = std::move(m);
    return true;
}

EncPOutputs enc_p_forward(Context* ctx, const EncPModule& m, Tensor* phone, Tensor* pitch,
                          Tensor* x_mask, Tensor* attn_valid, EncPCaptures* caps) {
    EncPOutputs out;
    if (ctx == nullptr || phone == nullptr || x_mask == nullptr || attn_valid == nullptr) {
        std::printf("[enc_p] 输入为空（需要 phone/x_mask/attn_valid）\n");
        return out;
    }
    const EncPConfig& cfg = m.cfg;
    const int64_t cin   = phone->ne[0];
    const int64_t T     = phone->ne[1];
    const int64_t B     = phone->ne[2];
    const int64_t C     = cfg.hidden_channels;
    const int64_t H     = cfg.n_heads;
    const int64_t dk    = C / H;
    if (cin != cfg.in_channels) {
        std::printf("[enc_p] phone 通道 %lld != in_channels %lld\n", (long long) cin,
                    (long long) cfg.in_channels);
        return out;
    }
    if (x_mask->ne[0] != T || x_mask->ne[1] != 1 || x_mask->ne[2] != B) {
        std::printf("[enc_p] x_mask 形状不匹配（期望 [T,1,B]=[%lld,1,%lld]）\n", (long long) T,
                    (long long) B);
        return out;
    }
    if (attn_valid->ne[0] != T || attn_valid->ne[1] != T || attn_valid->ne[2] != 1 ||
        attn_valid->ne[3] != B) {
        std::printf("[enc_p] attn_valid 形状不匹配（期望 [T,T,1,B]=[%lld,%lld,1,%lld]）\n",
                    (long long) T, (long long) T, (long long) B);
        return out;
    }
    if (T < cfg.window_size + 1) {
        std::printf("[enc_p] T=%lld < window+1=%lld（相对表的窗口切片分支暂未实现）\n",
                    (long long) T, (long long) (cfg.window_size + 1));
        return out;
    }

    if (caps != nullptr) {
        *caps = EncPCaptures();
    }

    // ---------- 输入嵌入 ----------
    // phone 内存 = [Cin, T, B]（ne0 快速）即 [Cin, T*B]（列序 t + T·b，与 torch [B,T,Cin] 展平一致）
    Tensor* ph2 = reshape_2d(ctx, phone, cin, T * B);                   // [Cin,T*B]
    Tensor* emb = linear_forward(ctx, &m.emb_phone, ph2);               // [C,T*B]
    emb = reshape_3d(ctx, emb, C, T, B);                                // [C,T,B]
    if (emb == nullptr) {
        std::printf("[enc_p] emb_phone 失败\n");
        return out;
    }
    if (pitch != nullptr) {
        if (pitch->type != TYPE_I32 || pitch->ne[0] != T || pitch->ne[1] != B) {
            std::printf("[enc_p] pitch 形状/类型不匹配（期望 I32 [T,B]）\n");
            return out;
        }
        if (m.emb_pitch == nullptr) {
            std::printf("[enc_p] 提供了 pitch 但模型无 emb_pitch 权重\n");
            return out;
        }
        // get_rows（与 ggml 一致）要求索引 ne[1]==表 ne[2]（即索引须为单批平面）；
        // 批维先展平 [T,B]→[T*B,1]，查表后再 reshape 回 [C,T,B]（B>1 训练必需）
        Tensor* pitch_flat = reshape_2d(ctx, cont(ctx, pitch), T * B, 1);
        Tensor* pe = get_rows(ctx, m.emb_pitch, pitch_flat);            // [C,T*B,1,1]
        pe = reshape_3d(ctx, pe, C, T, B);
        emb = add(ctx, emb, pe);
    }
    emb = scale(ctx, emb, std::sqrt((float) C));
    emb = leaky_relu(ctx, emb, kLeakySlope);
    if (emb == nullptr) {
        std::printf("[enc_p] 嵌入激活失败\n");
        return out;
    }
    if (caps != nullptr) {
        caps->emb = emb;
    }

    // Encoder 入口：转置为 [T,C,B]（torch [B,C,T]）后乘 x_mask（attentions.py:64/67）
    Tensor* x = cont(ctx, permute(ctx, emb, 1, 0, 2, 3));               // [T,C,B]
    x = mul(ctx, x, x_mask);                                            // [T,C,B] × [T,1,B]
    if (x == nullptr) {
        std::printf("[enc_p] Encoder 输入失败\n");
        return out;
    }
    if (caps != nullptr) {
        caps->x = x;
    }

    const float attn_scale = 1.0f / std::sqrt((float) dk);
    const int64_t rel_pad  = T - (cfg.window_size + 1);

    // ---------- n_layers 层 EncoderLayer ----------
    for (int64_t i = 0; i < cfg.n_layers; ++i) {
        const EncPAttnLayer& L = m.layers[(size_t) i];

        Tensor* q = conv1d_forward_weight(ctx, &L.conv_q, L.conv_q.weight, x);   // [T,C,B]
        Tensor* k = conv1d_forward_weight(ctx, &L.conv_k, L.conv_k.weight, x);
        Tensor* v = conv1d_forward_weight(ctx, &L.conv_v, L.conv_v.weight, x);
        if (q == nullptr || k == nullptr || v == nullptr) {
            std::printf("[enc_p] 第 %lld 层 conv_q/k/v 失败\n", (long long) i);
            return out;
        }
        if (caps != nullptr) {
            caps->q.push_back(q);
            caps->k.push_back(k);
            caps->v.push_back(v);
        }
        Tensor* q4 = head_split(ctx, q, T, dk, H, B);                   // [Dk,T,H,B]
        Tensor* k4 = head_split(ctx, k, T, dk, H, B);
        Tensor* v4 = head_split(ctx, v, T, dk, H, B);
        if (q4 == nullptr || k4 == nullptr || v4 == nullptr) {
            std::printf("[enc_p] 第 %lld 层头切分失败\n", (long long) i);
            return out;
        }
        // torch 在 query 上乘 1/√Dk（attentions.py:246/252）→ 这里同样先缩放，softmax 不再缩放
        // （口径与 golden 的 rel_logits/scores_local 一致）
        q4 = scale(ctx, q4, attn_scale);

        // 相对位置表：图内从 emb_rel_k/v 扩展 [Dk,2W+1,1] → [Dk,2T-1,1] / [2T-1,Dk,1]
        Tensor* rel_k = pad_ext(ctx, L.emb_rel_k, 0, 0, (int) rel_pad, (int) rel_pad, 0, 0, 0, 0,
                                PAD_ZERO);
        Tensor* rel_v_t = transpose(ctx, L.emb_rel_v);                  // [2W+1,Dk,1]
        Tensor* rel_v = pad_ext(ctx, rel_v_t, (int) rel_pad, (int) rel_pad, 0, 0, 0, 0, 0, 0,
                                PAD_ZERO);                              // [2T-1,Dk,1]
        if (rel_k == nullptr || rel_v == nullptr) {
            std::printf("[enc_p] 第 %lld 层相对表扩展失败\n", (long long) i);
            return out;
        }

        Tensor* scores = mul_mat(ctx, k4, q4);                          // [T,T,H,B]
        Tensor* rel_logits = mul_mat(ctx, rel_k, q4);                   // [2T-1,T,H,B]
        if (scores == nullptr || rel_logits == nullptr) {
            std::printf("[enc_p] 第 %lld 层注意力 matmul 失败\n", (long long) i);
            return out;
        }
        if (caps != nullptr) {
            caps->rel_logits.push_back(rel_logits);
        }
        Tensor* scores_local = relative_to_absolute(ctx, rel_logits, T);
        if (scores_local == nullptr) {
            std::printf("[enc_p] 第 %lld 层 relative_to_absolute 失败\n", (long long) i);
            return out;
        }
        if (caps != nullptr) {
            caps->scores_local.push_back(scores_local);
        }
        scores = add(ctx, scores, scores_local);
        if (caps != nullptr) {
            caps->scores.push_back(scores);
        }

        // 复刻 torch `scores.masked_fill(attn_mask == 0, -1e4)` 的**替换**语义：
        //   scores_masked = scores·M + (M−1)·1e4（M=attn_valid；无效位置恰为 -1e4，整行无效时均匀）
        Tensor* invalid_bias = scale_bias(ctx, attn_valid, 1.0e4f, -1.0e4f);   // M=1→0，M=0→-1e4
        Tensor* scores_masked = add(ctx, mul(ctx, scores, attn_valid), invalid_bias);
        if (caps != nullptr) {
            caps->scores_masked.push_back(scores_masked);
        }

        Tensor* p = soft_max(ctx, scores_masked);   // [T,T,H,B]（无效位梯度为 0，与 masked_fill 一致）
        if (p == nullptr) {
            std::printf("[enc_p] 第 %lld 层 soft_max 失败\n", (long long) i);
            return out;
        }
        if (caps != nullptr) {
            caps->p_attn.push_back(p);
        }

        Tensor* v_t = cont(ctx, permute(ctx, v4, 1, 0, 2, 3));          // [T,Dk,H,B]
        Tensor* out_h = mul_mat(ctx, v_t, p);                           // [Dk,T,H,B]
        Tensor* rel_w = absolute_to_relative(ctx, p, T, H, B);          // [2T-1,T,H,B]
        if (out_h == nullptr || rel_w == nullptr) {
            std::printf("[enc_p] 第 %lld 层注意力输出失败\n", (long long) i);
            return out;
        }
        out_h = add(ctx, out_h, mul_mat(ctx, rel_v, rel_w));            // 相对 value 项
        if (caps != nullptr) {
            caps->attn_out.push_back(out_h);
        }
        Tensor* merged = head_merge(ctx, out_h, T, C, B);               // [T,C,B]
        if (caps != nullptr) {
            caps->merged.push_back(merged);
        }
        Tensor* attn_y = conv1d_forward_weight(ctx, &L.conv_o, L.conv_o.weight, merged);
        if (attn_y == nullptr) {
            std::printf("[enc_p] 第 %lld 层 conv_o 失败\n", (long long) i);
            return out;
        }
        if (caps != nullptr) {
            caps->attn_y.push_back(attn_y);
        }

        // Post-LN1：x = LayerNorm(x + attn_y)（LayerNorm 沿通道 → 先转 [C,T,B]）
        x = add(ctx, x, attn_y);
        Tensor* xn1 = cont(ctx, permute(ctx, x, 1, 0, 2, 3));           // [C,T,B]
        Tensor* n1 = layernorm_forward(ctx, &L.norm1, xn1);             // [C,T,B]
        if (n1 == nullptr) {
            std::printf("[enc_p] 第 %lld 层 LayerNorm1 失败\n", (long long) i);
            return out;
        }
        Tensor* n1t = cont(ctx, permute(ctx, n1, 1, 0, 2, 3));          // [T,C,B]
        if (caps != nullptr) {
            caps->norm1.push_back(n1t);
        }

        // FFN（attentions.py:424-433）：mask → conv1 → relu → mask → conv2 → mask
        Tensor* f = mul(ctx, n1t, x_mask);
        f = conv1d_forward_weight(ctx, &L.ffn_conv1, L.ffn_conv1.weight, f);
        f = relu(ctx, f);
        f = mul(ctx, f, x_mask);
        f = conv1d_forward_weight(ctx, &L.ffn_conv2, L.ffn_conv2.weight, f);
        f = mul(ctx, f, x_mask);                                        // [T,C,B]
        if (f == nullptr) {
            std::printf("[enc_p] 第 %lld 层 FFN 失败\n", (long long) i);
            return out;
        }
        if (caps != nullptr) {
            caps->ffn.push_back(f);
        }

        // Post-LN2：x = LayerNorm(n1 + ffn)
        Tensor* xn2 = cont(ctx, permute(ctx, add(ctx, n1t, f), 1, 0, 2, 3));   // [C,T,B]
        Tensor* n2 = layernorm_forward(ctx, &L.norm2, xn2);             // [C,T,B]
        if (n2 == nullptr) {
            std::printf("[enc_p] 第 %lld 层 LayerNorm2 失败\n", (long long) i);
            return out;
        }
        x = cont(ctx, permute(ctx, n2, 1, 0, 2, 3));                    // [T,C,B]
        if (caps != nullptr) {
            caps->norm2.push_back(x);
        }
    }

    // Encoder 出口：x = x * x_mask（attentions.py:76）
    Tensor* encoder_out = mul(ctx, x, x_mask);
    if (encoder_out == nullptr) {
        std::printf("[enc_p] encoder 出口 mask 失败\n");
        return out;
    }
    if (caps != nullptr) {
        caps->encoder_out = encoder_out;
    }

    // proj（1×1）→ stats × mask → split m_p/logs_p
    Tensor* proj = conv1d_forward_weight(ctx, &m.proj, m.proj.weight, encoder_out);  // [T,2out,B]
    if (proj == nullptr) {
        std::printf("[enc_p] proj 失败\n");
        return out;
    }
    if (caps != nullptr) {
        caps->proj = proj;
    }
    Tensor* stats = mul(ctx, proj, x_mask);
    if (stats == nullptr) {
        std::printf("[enc_p] stats mask 失败\n");
        return out;
    }
    out.m_p    = view_3d(ctx, stats, T, cfg.out_channels, B, 0);
    out.logs_p = view_3d(ctx, stats, T, cfg.out_channels, B,
                         (size_t) cfg.out_channels * stats->nb[1]);
    return out;
}

} // namespace model
} // namespace rvc_train
