// text_encoder.cpp
// RVC enc_p（TextEncoder）：
//   emb_phone + emb_pitch -> ×sqrt(H) -> LeakyReLU(0.1)
//   -> N × [相对位置多头自注意力 + LayerNorm + FFN(k=3,ReLU) + LayerNorm]
//   -> proj(1x1) -> stats [2*inter, T]

#include "vits/synthesizer.h"
#include "vits/modules.h"

#include <cmath>
#include <stdexcept>

namespace rvc {
namespace vits {

namespace {

    constexpr float kLayerNormEps = 1.0e-5f;
    constexpr float kLReluSlope = 0.1f;

    ggml_tensor* require(const GgufModel& model, const std::string& name) {
        ggml_tensor* t = model.weight(name);
        if (!t) {
            throw std::runtime_error("VITS 缺失权重: " + name);
        }
        return t;
    }

    // 展开相对位置嵌入（对齐 attentions.py _get_relative_embeddings）
    //   emb: [2W+1, D]（原始窗口）；输出对 T 帧的有效窗口：
    //   transposed=false -> [2T-1, D]（rel_k 用）；true -> [D, 2T-1]（rel_v 用）
    std::vector<float> expand_relative_embedding(const std::vector<float>& emb,
        int64_t T, int64_t D, int64_t window, bool transposed) {
        const int64_t src_positions = 2 * window + 1;
        const int64_t relative_positions = 2 * T - 1;
        if (static_cast<int64_t>(emb.size()) != src_positions * D) {
            throw std::runtime_error("相对位置嵌入尺寸不匹配");
        }
        const int64_t pad_length = std::max<int64_t>(T - (window + 1), 0);
        const int64_t slice_start = std::max<int64_t>((window + 1) - T, 0);
        std::vector<float> out(static_cast<size_t>(relative_positions * D), 0.0f);
        for (int64_t pos = 0; pos < relative_positions; ++pos) {
            const int64_t source_pos = slice_start + pos - pad_length;
            if (source_pos < 0 || source_pos >= src_positions) {
                continue;
            }
            for (int64_t d = 0; d < D; ++d) {
                const float value = emb[static_cast<size_t>(source_pos * D + d)];
                if (transposed) {
                    out[static_cast<size_t>(d * relative_positions + pos)] = value;
                }
                else {
                    out[static_cast<size_t>(pos * D + d)] = value;
                }
            }
        }
        return out;
    }

    // 相对位置注意力（单层，逐头小 matmul）
    ggml_tensor* attention_layer(ggml_context* ctx, const GgufModel& model,
        const Config& cfg, ggml_tensor* x, int64_t layer, int64_t T,
        std::vector<GraphInput>& constants) {
        const std::string prefix = "enc_p.encoder.attn_layers." + std::to_string(layer);
        const int64_t H = cfg.n_heads;
        const int64_t D = cfg.hidden_channels / H;
        const float scale = 1.0f / std::sqrt(static_cast<float>(D));

        ggml_tensor* q = conv1d_ct(ctx, x, require(model, prefix + ".conv_q.weight"),
            require(model, prefix + ".conv_q.bias"), 1, 0, 1, cfg.conv_flat);
        ggml_tensor* k = conv1d_ct(ctx, x, require(model, prefix + ".conv_k.weight"),
            require(model, prefix + ".conv_k.bias"), 1, 0, 1, cfg.conv_flat);
        ggml_tensor* v = conv1d_ct(ctx, x, require(model, prefix + ".conv_v.weight"),
            require(model, prefix + ".conv_v.bias"), 1, 0, 1, cfg.conv_flat);

        // 相对位置嵌入常量（每层独立权重）
        const auto emb_k = model.read_f32(prefix + ".emb_rel_k");
        const auto emb_v = model.read_f32(prefix + ".emb_rel_v");
        ggml_tensor* rel_k = make_const_f32(ctx, D, 2 * T - 1, "rel_k");
        ggml_tensor* rel_v = make_const_f32(ctx, 2 * T - 1, D, "rel_v");
        constants.push_back({ rel_k, expand_relative_embedding(
            emb_k, T, D, cfg.window_size, false) });
        constants.push_back({ rel_v, expand_relative_embedding(
            emb_v, T, D, cfg.window_size, true) });

        ggml_tensor* out_cat = nullptr;
        for (int64_t h = 0; h < H; ++h) {
            const size_t head_off = static_cast<size_t>(h * D) * sizeof(float);
            ggml_tensor* q_h = ggml_cont(ctx, ggml_view_2d(ctx, q, D, T, q->nb[1], head_off));
            ggml_tensor* k_h = ggml_cont(ctx, ggml_view_2d(ctx, k, D, T, k->nb[1], head_off));
            ggml_tensor* v_h = ggml_cont(ctx, ggml_view_2d(ctx, v, D, T, v->nb[1], head_off));
            ggml_tensor* q_s = ggml_scale(ctx, q_h, scale);

            // scores[tk,tq] = sum_d k[d,tk] * q_s[d,tq]
            ggml_tensor* scores = ggml_mul_mat(ctx, k_h, q_s);          // [Tk, Tq]
            // 相对 logits[rel,tq] = sum_d rel_k[d,rel] * q_s[d,tq]
            ggml_tensor* rel_logits = ggml_mul_mat(ctx, rel_k, q_s);    // [2T-1, Tq]
            scores = add_checked(ctx, scores, relative_to_absolute(ctx, rel_logits, T, 1, D));
            ggml_tensor* p_attn = ggml_soft_max(ctx, scores);           // 沿 ne0=Tk 归一化

            // out[d,tq] = sum_tk v[d,tk] * p[tk,tq]
            ggml_tensor* v_t = ggml_cont(ctx, ggml_transpose(ctx, v_h));    // [Tk, D]
            ggml_tensor* out_h = ggml_mul_mat(ctx, v_t, p_attn);            // [D, Tq]
            ggml_tensor* rel_w = absolute_to_relative(ctx, p_attn, T, 1, D); // [2T-1, Tq]
            out_h = add_checked(ctx, out_h, ggml_mul_mat(ctx, rel_v, rel_w));
            out_cat = out_cat ? ggml_concat(ctx, out_cat, out_h, 0) : out_h;
        }

        return conv1d_ct(ctx, out_cat, require(model, prefix + ".conv_o.weight"),
            require(model, prefix + ".conv_o.bias"), 1, 0, 1, cfg.conv_flat);
    }

    // FFN（k=3，same padding，ReLU）
    ggml_tensor* ffn_layer(ggml_context* ctx, const GgufModel& model,
        const Config& cfg, ggml_tensor* x, int64_t layer) {
        const std::string prefix = "enc_p.encoder.ffn_layers." + std::to_string(layer);
        const int k = cfg.kernel_size;
        const int padding = k % 2 == 0 ? (k / 2 - 1) : (k / 2);
        ggml_tensor* y = conv1d_ct(ctx, x, require(model, prefix + ".conv_1.weight"),
            require(model, prefix + ".conv_1.bias"), 1, padding, 1, cfg.conv_flat);
        y = ggml_relu(ctx, y);
        return conv1d_ct(ctx, y, require(model, prefix + ".conv_2.weight"),
            require(model, prefix + ".conv_2.bias"), 1, padding, 1, cfg.conv_flat);
    }

} // namespace

ggml_tensor* build_text_encoder_stats(ggml_context* ctx, const GgufModel& model,
    const Config& cfg, ggml_tensor* features, ggml_tensor* pitch_i32,
    int64_t frames, std::vector<GraphInput>& constants) {
    const int64_t H = cfg.hidden_channels;

    // emb_phone: [C_in, T] -> [hidden, T]
    ggml_tensor* x = linear(ctx, features, require(model, "enc_p.emb_phone.weight"),
        require(model, "enc_p.emb_phone.bias"));

    if (cfg.has_f0) {
        if (pitch_i32 == nullptr) {
            throw std::runtime_error("VITS: has_f0 但未提供 pitch 输入");
        }
        ggml_tensor* pitch = get_rows(ctx, require(model, "enc_p.emb_pitch.weight"), pitch_i32);
        x = add_checked(ctx, x, pitch);
    }

    x = ggml_scale(ctx, x, std::sqrt(static_cast<float>(H)));
    x = ggml_leaky_relu(ctx, x, kLReluSlope, false);

    for (int64_t layer = 0; layer < cfg.n_layers; ++layer) {
        const std::string norm1 = "enc_p.encoder.norm_layers_1." + std::to_string(layer);
        const std::string norm2 = "enc_p.encoder.norm_layers_2." + std::to_string(layer);
        ggml_tensor* y = attention_layer(ctx, model, cfg, x, layer, frames, constants);
        x = add_checked(ctx, x, y);
        x = layer_norm(ctx, x, require(model, norm1 + ".gamma"),
            require(model, norm1 + ".beta"), kLayerNormEps);

        y = ffn_layer(ctx, model, cfg, x, layer);
        x = add_checked(ctx, x, y);
        x = layer_norm(ctx, x, require(model, norm2 + ".gamma"),
            require(model, norm2 + ".beta"), kLayerNormEps);
    }

    // proj: Conv1d(hidden, 2*inter, 1)
    return conv1d_ct(ctx, x, require(model, "enc_p.proj.weight"),
        require(model, "enc_p.proj.bias"), 1, 0, 1, cfg.conv_flat);
}

} // namespace vits
} // namespace rvc
