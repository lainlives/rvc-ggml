// RVC-Train-GGML - SynthesizerTrn 训练前向实现（src/model/synthesizer.cpp）
#include "model/synthesizer.h"

#include "model/emb.h"

#include <cstdio>

namespace rvc_train {
namespace model {

using namespace traincpp;

bool synth_load(const ModelParams& params, const SynthConfig& cfg, SynthModules* out,
                std::string* err) {
    if (out == nullptr) {
        if (err != nullptr) {
            *err = "synth 输出为空";
        }
        return false;
    }
    SynthModules m;
    m.cfg = cfg;
    if (!enc_p_load(params, cfg.enc_p, &m.enc_p, err)) {
        return false;
    }
    if (!enc_q_load(params, cfg.enc_q, &m.enc_q, err)) {
        return false;
    }
    if (!flow_load(params, cfg.flow, &m.flow, err)) {
        return false;
    }
    if (!dec_load(params, cfg.dec, &m.dec, err)) {
        return false;
    }
    *out = std::move(m);
    return true;
}

SynthOutputs synth_forward(Context* ctx, const ModelParams& params, const SynthModules& m,
                           const SynthInputs& in) {
    SynthOutputs out;
    if (ctx == nullptr || in.sid == nullptr || in.phone == nullptr || in.spec == nullptr ||
        in.y_mask == nullptr || in.q_noise == nullptr || in.flip_perm == nullptr ||
        in.slice_mat == nullptr || in.sine_source_slice == nullptr ||
        (in.include_full && in.sine_source_full == nullptr)) {
        std::printf("[synth] 输入为空\n");
        return out;
    }

    // g = emb_g(sid) → [1,gin,B]（torch [B,gin,1]）
    std::string err;
    Tensor* g2d = emb_g_forward(ctx, params, in.sid, &err);
    if (g2d == nullptr) {
        std::printf("[synth] emb_g 失败: %s\n", err.c_str());
        return out;
    }
    const int64_t gin = m.cfg.enc_q.gin_channels;
    const int64_t B   = in.sid->ne[0];
    out.g = reshape_3d(ctx, cont(ctx, g2d), 1, gin, B);

    // enc_p → m_p/logs_p
    EncPOutputs p = enc_p_forward(ctx, m.enc_p, in.phone, in.pitch, in.x_mask, in.attn_valid,
                                  &out.enc_p_caps);
    if (p.m_p == nullptr || p.logs_p == nullptr) {
        std::printf("[synth] enc_p 前向失败\n");
        return {};
    }
    out.m_p    = p.m_p;
    out.logs_p = p.logs_p;

    // enc_q → z/m_q/logs_q
    EncQOutputs q = enc_q_forward(ctx, m.enc_q, in.spec, in.y_mask, out.g, in.q_noise,
                                  &out.enc_q_caps);
    if (q.z == nullptr) {
        std::printf("[synth] enc_q 前向失败\n");
        return {};
    }
    out.z      = q.z;
    out.m_q    = q.m_q;
    out.logs_q = q.logs_q;

    // flow 正向 → z_p（训练正向，mask 用 y_mask）
    out.z_p = flow_forward(ctx, m.flow, out.z, in.y_mask, out.g, in.flip_perm, &out.flow_caps);
    if (out.z_p == nullptr) {
        std::printf("[synth] flow 前向失败\n");
        return {};
    }

    // 不切片路径：dec(整段 z)（仅 include_full；训练步只用切片路径）
    if (in.include_full) {
        out.o_full = dec_forward(ctx, m.dec, out.z, in.sine_source_full, out.g,
                                 &out.dec_full_caps);
        if (out.o_full == nullptr) {
            std::printf("[synth] dec（不切片）前向失败\n");
            return {};
        }
    }

    // 固定 ids 切片路径：z_slice = slice_mat × z（逐样本选择；等价 torch slice_segments）
    out.z_slice = mul_mat(ctx, in.slice_mat, out.z);
    out.o_slice = dec_forward(ctx, m.dec, out.z_slice, in.sine_source_slice, out.g,
                              &out.dec_slice_caps);
    if (out.o_slice == nullptr) {
        std::printf("[synth] dec（切片）前向失败\n");
        return {};
    }
    return out;
}

} // namespace model
} // namespace rvc_train
