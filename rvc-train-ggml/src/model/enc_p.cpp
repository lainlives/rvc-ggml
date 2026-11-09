// RVC-Train-GGML - enc_p (TextEncoder, relative-position attention) implementation (src/model/enc_p.cpp)
//
// ne layout derivation for relative-position attention (same source as the verified structure in RVC-GGML `src/vits/text_encoder.cpp`):
//   the last dim of torch tensors such as p/scores corresponds to ne0 in this library; the relative<->absolute transforms (attentions.py:327-374)
//   only act on the last two dims in torch, so in ne layout they are equivalent to pad+reshape+view on (ne0, ne1):
//     relative_to_absolute: [2T-1,T,..] -> pad ne0 right by 1 -> [2T,T,..] -> reshape flattens ne0/ne1
//       -> pad ne0 right by T-1 (torch F.pad [0,l-1]) -> reshape [2T-1,T+1,..]
//       -> view ne1[:T], ne0 [T-1, T-1+T) -> [T,T,..]
//     absolute_to_relative: [T,T,..] -> pad ne0 right by T-1 -> [2T-1,T,..] -> reshape flattens
//       -> pad ne0 left by T (prepend T zeros) -> reshape [2T,T,..] -> view ne0 [1, 2T) -> [2T-1,T,..]
//   when reshape flattens ne0/ne1, the outer (H,B) planes must be preserved (each plane is transformed independently).
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
        *err = "missing parameter: " + name;
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

// Head split: [T,C,B] -> reshape [T,Dk,H,B] -> permute -> [Dk,T,H,B]
// (channel order matches torch `view(b,H,Dk,t)`: c = h·Dk + d = d + Dk·h)
Tensor* head_split(Context* ctx, Tensor* x, int64_t frames, int64_t dk, int64_t heads,
                   int64_t batch) {
    Tensor* v = reshape_4d(ctx, x, frames, dk, heads, batch);
    if (v == nullptr) {
        return nullptr;
    }
    return cont(ctx, permute(ctx, v, 1, 0, 2, 3));
}

// Head merge: [Dk,T,H,B] -> transpose [T,Dk,H,B] -> reshape [T,C,B] (c = d + Dk·h)
Tensor* head_merge(Context* ctx, Tensor* out, int64_t frames, int64_t channels, int64_t batch) {
    Tensor* p = cont(ctx, transpose(ctx, out));
    if (p == nullptr) {
        return nullptr;
    }
    return reshape_3d(ctx, p, frames, channels, batch);
}

// Relative logits [2T-1,T,H,B] -> absolute scores_local [T,T,H,B] (attentions.py:327-352):
//   pad ne0 right by 1 -> [2T,T,H,B]; flatten ne0/ne1; pad ne0 right by T-1; reshape [2T-1,T+1,H,B];
//   view ne1[:T], ne0[T-1,T-1+T) -> [T,T,H,B]
Tensor* relative_to_absolute(Context* ctx, Tensor* x, int64_t frames) {
    const int64_t T = frames;
    const int64_t H = x->ne[2];
    const int64_t B = x->ne[3];
    Tensor* y = pad_ext(ctx, x, 0, 1, 0, 0, 0, 0, 0, 0, PAD_ZERO);   // [2T,T,H,B]
    if (y == nullptr) {
        return nullptr;
    }
    y = reshape_3d(ctx, y, 2 * T * T, H, B);                         // flatten ne0/ne1 (preserve planes)
    if (y == nullptr) {
        return nullptr;
    }
    y = pad_ext(ctx, y, 0, (int) (T - 1), 0, 0, 0, 0, 0, 0, PAD_ZERO);   // right padding (torch F.pad [0,l-1])
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

// Absolute p_attn [T,T,H,B] -> relative weights [2T-1,T,H,B] (attentions.py:354-374)
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
            *err = "enc_p output is null";
        }
        return false;
    }
    if (cfg.in_channels <= 0 || cfg.hidden_channels <= 0 || cfg.filter_channels <= 0 ||
        cfg.n_heads <= 0 || cfg.hidden_channels % cfg.n_heads != 0 || cfg.n_layers <= 0 ||
        cfg.kernel_size <= 0 || cfg.kernel_size % 2 == 0 || cfg.window_size <= 0 ||
        cfg.out_channels <= 0) {
        if (err != nullptr) {
            *err = "invalid enc_p config (channels/heads > 0 and hidden divisible by heads; kernel is a positive odd number;"
                   "window > 0)";
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
            *err = cfg.prefix + ".emb_phone.weight shape does not match config (expected ne=[" +
                   std::to_string(cfg.in_channels) + "," + std::to_string(hidden) + "]";
        }
        return false;
    }
    wire_linear(&m.emb_phone, w, params.find((cfg.prefix + ".emb_phone.bias").c_str()),
                "enc_p.emb_phone");

    m.emb_pitch = params.find((cfg.prefix + ".emb_pitch.weight").c_str());
    if (m.emb_pitch != nullptr &&
        (m.emb_pitch->ne[0] != hidden || m.emb_pitch->ne[1] != 256)) {
        if (err != nullptr) {
            *err = cfg.prefix + ".emb_pitch.weight shape does not match config (expected ne=[hidden,256])";
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
                *err = ab + " convolution weight shape does not match config (expected [1,hidden,hidden])";
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
                *err = ab + ".emb_rel_k/v shape does not match config (expected ne=[Dk,2W+1,1])";
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
                *err = n1 + "/norm_layers_2 gamma shape does not match config (expected [hidden])";
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
                *err = fb + " convolution weight shape does not match config (conv_1 [k,hidden,filter] / "
                       "conv_2 [k,filter,hidden])";
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
            *err = cfg.prefix + ".proj.weight shape does not match config (expected [1,hidden,2*out])";
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
        std::printf("[enc_p] null input (phone/x_mask/attn_valid required)\n");
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
        std::printf("[enc_p] phone channels %lld != in_channels %lld\n", (long long) cin,
                    (long long) cfg.in_channels);
        return out;
    }
    if (x_mask->ne[0] != T || x_mask->ne[1] != 1 || x_mask->ne[2] != B) {
        std::printf("[enc_p] x_mask shape mismatch (expected [T,1,B]=[%lld,1,%lld])\n", (long long) T,
                    (long long) B);
        return out;
    }
    if (attn_valid->ne[0] != T || attn_valid->ne[1] != T || attn_valid->ne[2] != 1 ||
        attn_valid->ne[3] != B) {
        std::printf("[enc_p] attn_valid shape mismatch (expected [T,T,1,B]=[%lld,%lld,1,%lld])\n",
                    (long long) T, (long long) T, (long long) B);
        return out;
    }
    if (T < cfg.window_size + 1) {
        std::printf("[enc_p] T=%lld < window+1=%lld (window-slice branch of the relative table not implemented yet)\n",
                    (long long) T, (long long) (cfg.window_size + 1));
        return out;
    }

    if (caps != nullptr) {
        *caps = EncPCaptures();
    }

    // ---------- input embedding ----------
    // phone memory = [Cin, T, B] (ne0 fastest), i.e. [Cin, T*B] (column order t + T·b, consistent with torch [B,T,Cin] flattening)
    Tensor* ph2 = reshape_2d(ctx, phone, cin, T * B);                   // [Cin,T*B]
    Tensor* emb = linear_forward(ctx, &m.emb_phone, ph2);               // [C,T*B]
    emb = reshape_3d(ctx, emb, C, T, B);                                // [C,T,B]
    if (emb == nullptr) {
        std::printf("[enc_p] emb_phone failed\n");
        return out;
    }
    if (pitch != nullptr) {
        if (pitch->type != TYPE_I32 || pitch->ne[0] != T || pitch->ne[1] != B) {
            std::printf("[enc_p] pitch shape/type mismatch (expected I32 [T,B])\n");
            return out;
        }
        if (m.emb_pitch == nullptr) {
            std::printf("[enc_p] pitch provided but the model has no emb_pitch weights\n");
            return out;
        }
        // get_rows (consistent with ggml) requires index ne[1]==table ne[2] (i.e. the index must be a single-batch plane);
        // the batch dim is flattened first [T,B]->[T*B,1], then reshaped back to [C,T,B] after the lookup (required for training with B>1)
        Tensor* pitch_flat = reshape_2d(ctx, cont(ctx, pitch), T * B, 1);
        Tensor* pe = get_rows(ctx, m.emb_pitch, pitch_flat);            // [C,T*B,1,1]
        pe = reshape_3d(ctx, pe, C, T, B);
        emb = add(ctx, emb, pe);
    }
    emb = scale(ctx, emb, std::sqrt((float) C));
    emb = leaky_relu(ctx, emb, kLeakySlope);
    if (emb == nullptr) {
        std::printf("[enc_p] embedding activation failed\n");
        return out;
    }
    if (caps != nullptr) {
        caps->emb = emb;
    }

    // Encoder entry: transpose to [T,C,B] (torch [B,C,T]) then multiply by x_mask (attentions.py:64/67)
    Tensor* x = cont(ctx, permute(ctx, emb, 1, 0, 2, 3));               // [T,C,B]
    x = mul(ctx, x, x_mask);                                            // [T,C,B] × [T,1,B]
    if (x == nullptr) {
        std::printf("[enc_p] Encoder input failed\n");
        return out;
    }
    if (caps != nullptr) {
        caps->x = x;
    }

    const float attn_scale = 1.0f / std::sqrt((float) dk);
    const int64_t rel_pad  = T - (cfg.window_size + 1);

    // ---------- n_layers EncoderLayer layers ----------
    for (int64_t i = 0; i < cfg.n_layers; ++i) {
        const EncPAttnLayer& L = m.layers[(size_t) i];

        Tensor* q = conv1d_forward_weight(ctx, &L.conv_q, L.conv_q.weight, x);   // [T,C,B]
        Tensor* k = conv1d_forward_weight(ctx, &L.conv_k, L.conv_k.weight, x);
        Tensor* v = conv1d_forward_weight(ctx, &L.conv_v, L.conv_v.weight, x);
        if (q == nullptr || k == nullptr || v == nullptr) {
            std::printf("[enc_p] layer %lld conv_q/k/v failed\n", (long long) i);
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
            std::printf("[enc_p] layer %lld head split failed\n", (long long) i);
            return out;
        }
        // torch multiplies the query by 1/√Dk (attentions.py:246/252) -> here we scale first as well, and softmax does not scale again
        // (the convention matches golden rel_logits/scores_local)
        q4 = scale(ctx, q4, attn_scale);

        // Relative-position tables: expanded in-graph from emb_rel_k/v [Dk,2W+1,1] -> [Dk,2T-1,1] / [2T-1,Dk,1]
        Tensor* rel_k = pad_ext(ctx, L.emb_rel_k, 0, 0, (int) rel_pad, (int) rel_pad, 0, 0, 0, 0,
                                PAD_ZERO);
        Tensor* rel_v_t = transpose(ctx, L.emb_rel_v);                  // [2W+1,Dk,1]
        Tensor* rel_v = pad_ext(ctx, rel_v_t, (int) rel_pad, (int) rel_pad, 0, 0, 0, 0, 0, 0,
                                PAD_ZERO);                              // [2T-1,Dk,1]
        if (rel_k == nullptr || rel_v == nullptr) {
            std::printf("[enc_p] layer %lld relative table expansion failed\n", (long long) i);
            return out;
        }

        Tensor* scores = mul_mat(ctx, k4, q4);                          // [T,T,H,B]
        Tensor* rel_logits = mul_mat(ctx, rel_k, q4);                   // [2T-1,T,H,B]
        if (scores == nullptr || rel_logits == nullptr) {
            std::printf("[enc_p] layer %lld attention matmul failed\n", (long long) i);
            return out;
        }
        if (caps != nullptr) {
            caps->rel_logits.push_back(rel_logits);
        }
        Tensor* scores_local = relative_to_absolute(ctx, rel_logits, T);
        if (scores_local == nullptr) {
            std::printf("[enc_p] layer %lld relative_to_absolute failed\n", (long long) i);
            return out;
        }
        if (caps != nullptr) {
            caps->scores_local.push_back(scores_local);
        }
        scores = add(ctx, scores, scores_local);
        if (caps != nullptr) {
            caps->scores.push_back(scores);
        }

        // Replicate the **replace** semantics of torch `scores.masked_fill(attn_mask == 0, -1e4)`:
        //   scores_masked = scores·M + (M−1)·1e4 (M=attn_valid; invalid positions become exactly -1e4, and a fully invalid row stays uniform)
        Tensor* invalid_bias = scale_bias(ctx, attn_valid, 1.0e4f, -1.0e4f);   // M=1->0, M=0->-1e4
        Tensor* scores_masked = add(ctx, mul(ctx, scores, attn_valid), invalid_bias);
        if (caps != nullptr) {
            caps->scores_masked.push_back(scores_masked);
        }

        Tensor* p = soft_max(ctx, scores_masked);   // [T,T,H,B] (gradient is 0 at invalid positions, consistent with masked_fill)
        if (p == nullptr) {
            std::printf("[enc_p] layer %lld soft_max failed\n", (long long) i);
            return out;
        }
        if (caps != nullptr) {
            caps->p_attn.push_back(p);
        }

        Tensor* v_t = cont(ctx, permute(ctx, v4, 1, 0, 2, 3));          // [T,Dk,H,B]
        Tensor* out_h = mul_mat(ctx, v_t, p);                           // [Dk,T,H,B]
        Tensor* rel_w = absolute_to_relative(ctx, p, T, H, B);          // [2T-1,T,H,B]
        if (out_h == nullptr || rel_w == nullptr) {
            std::printf("[enc_p] layer %lld attention output failed\n", (long long) i);
            return out;
        }
        out_h = add(ctx, out_h, mul_mat(ctx, rel_v, rel_w));            // relative value term
        if (caps != nullptr) {
            caps->attn_out.push_back(out_h);
        }
        Tensor* merged = head_merge(ctx, out_h, T, C, B);               // [T,C,B]
        if (caps != nullptr) {
            caps->merged.push_back(merged);
        }
        Tensor* attn_y = conv1d_forward_weight(ctx, &L.conv_o, L.conv_o.weight, merged);
        if (attn_y == nullptr) {
            std::printf("[enc_p] layer %lld conv_o failed\n", (long long) i);
            return out;
        }
        if (caps != nullptr) {
            caps->attn_y.push_back(attn_y);
        }

        // Post-LN1: x = LayerNorm(x + attn_y) (LayerNorm is along channels -> transpose to [C,T,B] first)
        x = add(ctx, x, attn_y);
        Tensor* xn1 = cont(ctx, permute(ctx, x, 1, 0, 2, 3));           // [C,T,B]
        Tensor* n1 = layernorm_forward(ctx, &L.norm1, xn1);             // [C,T,B]
        if (n1 == nullptr) {
            std::printf("[enc_p] layer %lld LayerNorm1 failed\n", (long long) i);
            return out;
        }
        Tensor* n1t = cont(ctx, permute(ctx, n1, 1, 0, 2, 3));          // [T,C,B]
        if (caps != nullptr) {
            caps->norm1.push_back(n1t);
        }

        // FFN (attentions.py:424-433): mask -> conv1 -> relu -> mask -> conv2 -> mask
        Tensor* f = mul(ctx, n1t, x_mask);
        f = conv1d_forward_weight(ctx, &L.ffn_conv1, L.ffn_conv1.weight, f);
        f = relu(ctx, f);
        f = mul(ctx, f, x_mask);
        f = conv1d_forward_weight(ctx, &L.ffn_conv2, L.ffn_conv2.weight, f);
        f = mul(ctx, f, x_mask);                                        // [T,C,B]
        if (f == nullptr) {
            std::printf("[enc_p] layer %lld FFN failed\n", (long long) i);
            return out;
        }
        if (caps != nullptr) {
            caps->ffn.push_back(f);
        }

        // Post-LN2: x = LayerNorm(n1 + ffn)
        Tensor* xn2 = cont(ctx, permute(ctx, add(ctx, n1t, f), 1, 0, 2, 3));   // [C,T,B]
        Tensor* n2 = layernorm_forward(ctx, &L.norm2, xn2);             // [C,T,B]
        if (n2 == nullptr) {
            std::printf("[enc_p] layer %lld LayerNorm2 failed\n", (long long) i);
            return out;
        }
        x = cont(ctx, permute(ctx, n2, 1, 0, 2, 3));                    // [T,C,B]
        if (caps != nullptr) {
            caps->norm2.push_back(x);
        }
    }

    // Encoder exit: x = x * x_mask (attentions.py:76)
    Tensor* encoder_out = mul(ctx, x, x_mask);
    if (encoder_out == nullptr) {
        std::printf("[enc_p] encoder exit mask failed\n");
        return out;
    }
    if (caps != nullptr) {
        caps->encoder_out = encoder_out;
    }

    // proj (1×1) -> stats × mask -> split m_p/logs_p
    Tensor* proj = conv1d_forward_weight(ctx, &m.proj, m.proj.weight, encoder_out);  // [T,2out,B]
    if (proj == nullptr) {
        std::printf("[enc_p] proj failed\n");
        return out;
    }
    if (caps != nullptr) {
        caps->proj = proj;
    }
    Tensor* stats = mul(ctx, proj, x_mask);
    if (stats == nullptr) {
        std::printf("[enc_p] stats mask failed\n");
        return out;
    }
    out.m_p    = view_3d(ctx, stats, T, cfg.out_channels, B, 0);
    out.logs_p = view_3d(ctx, stats, T, cfg.out_channels, B,
                         (size_t) cfg.out_channels * stats->nb[1]);
    return out;
}

} // namespace model
} // namespace rvc_train
