// RVC-Train-GGML - M7-2 从零随机初始化实现（src/model/scratch.cpp）
#include "model/scratch.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <unordered_map>

namespace rvc_train {
namespace model {

namespace {

int64_t ne_elems(const std::array<int64_t, 4>& ne) {
    return ne[0] * ne[1] * ne[2] * ne[3];
}

// fan_in = 元素数 / 输出维（ch_dim 维）
int64_t fan_in_of(const std::array<int64_t, 4>& ne, int ch_dim) {
    const int64_t denom = ne[ch_dim];
    return denom > 0 ? ne_elems(ne) / denom : 1;
}

// weight_g 形状：v 的 ch_dim 维保留，其余置 1
std::array<int64_t, 4> wn_g_ne(const std::array<int64_t, 4>& vne, int ch_dim) {
    std::array<int64_t, 4> g{1, 1, 1, 1};
    g[ch_dim] = vne[ch_dim];
    return g;
}

struct Builder {
    std::vector<TensorSpec>* v;
    bool ok = true;

    void add(const std::string& name, std::array<int64_t, 4> ne, InitKind init, int64_t fan_in) {
        TensorSpec s;
        s.name = name;
        s.ne = ne;
        s.init = init;
        s.fan_in = fan_in;
        v->push_back(std::move(s));
    }

    // 普通 Conv/Linear 权重（kaiming，fan_in 由形状推导）；ch_dim 为输出维
    void weight(const std::string& name, std::array<int64_t, 4> ne, int ch_dim) {
        TensorSpec s;
        s.name = name;
        s.ne = ne;
        s.init = InitKind::Kaiming;
        s.fan_in = fan_in_of(ne, ch_dim);
        s.ch_dim = ch_dim;
        v->push_back(std::move(s));
    }
    // weight_norm：配对 weight_g + weight_v（Kaiming），bias
    void wn(const std::string& base, std::array<int64_t, 4> vne, int ch_dim, bool with_bias,
            int64_t out_ch) {
        TensorSpec g;
        g.name = base + ".weight_g";
        g.ne = wn_g_ne(vne, ch_dim);
        g.init = InitKind::WnG;
        g.fan_in = 0;
        g.ch_dim = ch_dim;
        v->push_back(std::move(g));
        weight(base + ".weight_v", vne, ch_dim);
        if (with_bias) {
            std::array<int64_t, 4> bne{out_ch, 1, 1, 1};
            add(base + ".bias", bne, InitKind::Bias, fan_in_of(vne, ch_dim));
        }
    }
    void conv(const std::string& base, std::array<int64_t, 4> wne, int ch_dim, bool with_bias,
              int64_t out_ch) {
        weight(base + ".weight", wne, ch_dim);
        if (with_bias) {
            std::array<int64_t, 4> bne{out_ch, 1, 1, 1};
            add(base + ".bias", bne, InitKind::Bias, fan_in_of(wne, ch_dim));
        }
    }
    void linear(const std::string& base, int64_t in, int64_t out, bool with_bias) {
        std::array<int64_t, 4> wne{in, out, 1, 1};
        weight(base + ".weight", wne, 1);
        if (with_bias) {
            std::array<int64_t, 4> bne{out, 1, 1, 1};
            add(base + ".bias", bne, InitKind::Bias, in);
        }
    }
    void embedding(const std::string& base, int64_t n, int64_t dim) {
        std::array<int64_t, 4> wne{dim, n, 1, 1};
        add(base + ".weight", wne, InitKind::Embedding, 0);
    }
    void layernorm(const std::string& base, int64_t ch) {
        std::array<int64_t, 4> ne{ch, 1, 1, 1};
        add(base + ".gamma", ne, InitKind::Ones, 0);
        add(base + ".beta", ne, InitKind::Zeros, 0);
    }
    void rel_emb(const std::string& base, int64_t dk, int64_t window) {
        std::array<int64_t, 4> ne{dk, window, 1, 1};
        add(base + ".emb_rel_k", ne, InitKind::RelEmb, 0);
        add(base + ".emb_rel_v", ne, InitKind::RelEmb, 0);
    }
};

} // namespace

bool build_g_specs(const ModelConfig& mc, std::vector<TensorSpec>* out, std::string* err) {
    if (out == nullptr) {
        if (err != nullptr) {
            *err = "build_g_specs: out 为空";
        }
        return false;
    }
    if (!mc.validate(err)) {
        return false;
    }
    out->clear();
    Builder b{out};

    const int64_t hidden  = mc.hidden_channels;
    const int64_t filter  = mc.filter_channels;
    const int64_t nheads  = mc.n_heads;
    const int64_t nlayers = mc.n_layers;
    const int64_t ksize   = mc.kernel_size;
    const int64_t window  = 10;
    const int64_t dk      = hidden / nheads;
    const int64_t inter   = mc.inter_channels;
    const int64_t gin     = mc.gin_channels;
    const int64_t in_feat = mc.is_v2() ? 768 : 256;

    // ---------------- enc_p（TextEncoder）----------------
    b.linear("enc_p.emb_phone", in_feat, hidden, true);
    b.embedding("enc_p.emb_pitch", 256, hidden);
    for (int64_t i = 0; i < nlayers; ++i) {
        const std::string ab = "enc_p.encoder.attn_layers." + std::to_string((long long) i);
        std::array<int64_t, 4> cne{1, hidden, hidden, 1};
        b.conv(ab + ".conv_q", cne, 2, true, hidden);
        b.conv(ab + ".conv_k", cne, 2, true, hidden);
        b.conv(ab + ".conv_v", cne, 2, true, hidden);
        b.conv(ab + ".conv_o", cne, 2, true, hidden);
        b.rel_emb(ab, dk, 2 * window + 1);
    }
    for (int64_t i = 0; i < nlayers; ++i) {
        const std::string p1 = "enc_p.encoder.norm_layers_1." + std::to_string((long long) i);
        b.layernorm(p1, hidden);
    }
    for (int64_t i = 0; i < nlayers; ++i) {
        const std::string fb = "enc_p.encoder.ffn_layers." + std::to_string((long long) i);
        std::array<int64_t, 4> w1{ksize, hidden, filter, 1};
        std::array<int64_t, 4> w2{ksize, filter, hidden, 1};
        b.conv(fb + ".conv_1", w1, 2, true, filter);
        b.conv(fb + ".conv_2", w2, 2, true, hidden);
    }
    for (int64_t i = 0; i < nlayers; ++i) {
        const std::string p2 = "enc_p.encoder.norm_layers_2." + std::to_string((long long) i);
        b.layernorm(p2, hidden);
    }
    {
        std::array<int64_t, 4> pne{1, hidden, 2 * inter, 1};
        b.conv("enc_p.proj", pne, 2, true, 2 * inter);
    }

    // ---------------- enc_q（PosteriorEncoder；n_layers 固定 16）----------------
    const int64_t encq_layers = 16;
    {
        std::array<int64_t, 4> pre{1, mc.spec_channels, hidden, 1};
        b.conv("enc_q.pre", pre, 2, true, hidden);
    }
    for (int64_t i = 0; i < encq_layers; ++i) {
        const std::string ib = "enc_q.enc.in_layers." + std::to_string((long long) i);
        std::array<int64_t, 4> vne{5, hidden, 2 * hidden, 1};
        b.wn(ib, vne, 2, true, 2 * hidden);
    }
    for (int64_t i = 0; i < encq_layers; ++i) {
        const std::string rb = "enc_q.enc.res_skip_layers." + std::to_string((long long) i);
        const int64_t oc = (i < encq_layers - 1) ? 2 * hidden : hidden;
        std::array<int64_t, 4> vne{1, hidden, oc, 1};
        b.wn(rb, vne, 2, true, oc);
    }
    {
        std::array<int64_t, 4> cne{1, gin, 2 * hidden * encq_layers, 1};
        b.wn("enc_q.enc.cond_layer", cne, 2, true, 2 * hidden * encq_layers);
    }
    {
        std::array<int64_t, 4> pne{1, hidden, 2 * inter, 1};
        b.conv("enc_q.proj", pne, 2, true, 2 * inter);
    }

    // ---------------- flow（ResidualCouplingBlock；4 层索引 0/2/4/6；n_layers 固定 3）----------------
    const int64_t flow_layers = 3;
    for (int64_t f = 0; f < 4; ++f) {
        const std::string fb = "flow.flows." + std::to_string((long long) (f * 2));
        b.conv(fb + ".pre", {1, inter / 2, hidden, 1}, 2, true, hidden);
        for (int64_t i = 0; i < flow_layers; ++i) {
            const std::string ib = fb + ".enc.in_layers." + std::to_string((long long) i);
            b.wn(ib, {5, hidden, 2 * hidden, 1}, 2, true, 2 * hidden);
        }
        for (int64_t i = 0; i < flow_layers; ++i) {
            const std::string rb = fb + ".enc.res_skip_layers." + std::to_string((long long) i);
            const int64_t oc = (i < flow_layers - 1) ? 2 * hidden : hidden;
            b.wn(rb, {1, hidden, oc, 1}, 2, true, oc);
        }
        b.wn(fb + ".enc.cond_layer", {1, gin, 2 * hidden * flow_layers, 1}, 2, true,
             2 * hidden * flow_layers);
        b.conv(fb + ".post", {1, hidden, inter / 2, 1}, 2, true, inter / 2);
    }

    // ---------------- dec（GeneratorNSF）----------------
    const int64_t initial = mc.upsample_initial_channel;
    const int64_t n_up = (int64_t) mc.upsample_rates.size();
    b.linear("dec.m_source.l_linear", 1, 1, true);  // harmonic_num = 0 → (1,1)
    b.conv("dec.conv_pre", {7, inter, initial, 1}, 2, true, initial);
    b.conv("dec.cond", {1, gin, initial, 1}, 2, true, initial);
    for (int64_t i = 0; i < n_up; ++i) {
        const int64_t chin = initial >> i;
        const int64_t chout = initial >> (i + 1);
        const int64_t k = mc.upsample_kernel_sizes[(size_t) i];
        b.wn("dec.ups." + std::to_string((long long) i), {k, chout, chin, 1}, 2, true, chout);
    }
    for (int64_t i = 0; i < n_up; ++i) {
        const int64_t c_cur = initial >> (i + 1);
        int64_t stride_f0 = 1;
        for (int64_t j = i + 1; j < n_up; ++j) {
            stride_f0 *= mc.upsample_rates[(size_t) j];
        }
        const int64_t k = (i + 1 < n_up) ? stride_f0 * 2 : 1;
        b.conv("dec.noise_convs." + std::to_string((long long) i), {k, 1, c_cur, 1}, 2, true,
               c_cur);
    }
    const int64_t n_kernels = (int64_t) mc.resblock_kernel_sizes.size();
    for (int64_t i = 0; i < n_up; ++i) {
        const int64_t ch = initial >> (i + 1);
        for (int64_t j = 0; j < n_kernels; ++j) {
            const int64_t idx = i * n_kernels + j;
            const int64_t k = mc.resblock_kernel_sizes[(size_t) j];
            const std::string rb = "dec.resblocks." + std::to_string((long long) idx);
            for (int64_t c = 0; c < 3; ++c) {
                const std::string b1 = rb + ".convs1." + std::to_string((long long) c);
                const std::string b2 = rb + ".convs2." + std::to_string((long long) c);
                b.wn(b1, {k, ch, ch, 1}, 2, true, ch);
                b.wn(b2, {k, ch, ch, 1}, 2, true, ch);
            }
        }
    }
    const int64_t ch_last = initial >> n_up;
    b.conv("dec.conv_post", {7, ch_last, 1, 1}, 2, false, 1);

    // ---------------- emb_g（SpeakerEmbedding）----------------
    b.embedding("emb_g", mc.spk_embed_dim, gin);

    return true;
}

bool build_d_specs(const ModelConfig& mc, std::vector<TensorSpec>* out, std::string* err) {
    if (out == nullptr) {
        if (err != nullptr) {
            *err = "build_d_specs: out 为空";
        }
        return false;
    }
    if (!mc.validate(err)) {
        return false;
    }
    if (mc.d_periods.empty()) {
        if (err != nullptr) {
            *err = "D 配置缺少 d_periods";
        }
        return false;
    }
    out->clear();
    Builder b{out};

    // discriminators.0 = DiscriminatorS（Conv1d，groups 4/16/64/256）
    {
        const std::string s = "discriminators.0";
        // (out,in,k,groups)
        const int64_t cfgS[6][4] = {
            {16, 1, 15, 1}, {64, 16, 41, 4}, {256, 64, 41, 16},
            {1024, 256, 41, 64}, {1024, 1024, 41, 256}, {1024, 1024, 5, 1},
        };
        for (int i = 0; i < 6; ++i) {
            const int64_t O = cfgS[i][0], I = cfgS[i][1], K = cfgS[i][2], G = cfgS[i][3];
            std::array<int64_t, 4> vne{K, I / G, O, 1};
            b.wn(s + ".convs." + std::to_string(i), vne, 2, true, O);
        }
        b.wn(s + ".conv_post", {3, 1024, 1, 1}, 2, true, 1);
    }

    // discriminators.{1..N} = DiscriminatorP（Conv2d，kernel (5,1)）
    const int64_t P = (int64_t) mc.d_periods.size();
    for (int64_t p = 1; p <= P; ++p) {
        const std::string s = "discriminators." + std::to_string((long long) p);
        const int64_t cfgP[5][2] = {
            {32, 1}, {128, 32}, {512, 128}, {1024, 512}, {1024, 1024},
        };
        for (int i = 0; i < 5; ++i) {
            const int64_t O = cfgP[i][0], I = cfgP[i][1];
            std::array<int64_t, 4> vne{1, 5, I, O};  // [kW=1,kH=5,I,O]
            b.wn(s + ".convs." + std::to_string(i), vne, 3, true, O);
        }
        b.wn(s + ".conv_post", {1, 3, 1024, 1}, 3, true, 1);
    }
    return true;
}

bool fill_specs(const std::vector<TensorSpec>& specs, uint64_t seed,
                std::vector<std::vector<float>>* out, std::string* err) {
    if (out == nullptr) {
        if (err != nullptr) {
            *err = "fill_specs: out 为空";
        }
        return false;
    }
    out->assign(specs.size(), {});
    std::mt19937_64 rng(seed);

    for (size_t i = 0; i < specs.size(); ++i) {
        const TensorSpec& s = specs[i];
        if (s.init == InitKind::WnG) {
            continue;  // 二遍处理
        }
        std::vector<float>& d = (*out)[i];
        d.resize((size_t) ne_elems(s.ne));
        const float fin = s.fan_in > 0 ? (float) s.fan_in : 1.0f;
        switch (s.init) {
            case InitKind::Kaiming: {
                const float b = 1.0f / std::sqrt(fin);
                std::uniform_real_distribution<float> dist(-b, b);
                for (float& x : d) {
                    x = dist(rng);
                }
                break;
            }
            case InitKind::Bias: {
                const float b = 1.0f / std::sqrt(fin);
                std::uniform_real_distribution<float> dist(-b, b);
                for (float& x : d) {
                    x = dist(rng);
                }
                break;
            }
            case InitKind::Embedding: {
                std::normal_distribution<float> dist(0.0f, 1.0f);
                for (float& x : d) {
                    x = dist(rng);
                }
                break;
            }
            case InitKind::RelEmb: {
                const float std_dev = 1.0f / std::sqrt((float) (s.ne[0] > 0 ? s.ne[0] : 1));
                std::normal_distribution<float> dist(0.0f, std_dev);
                for (float& x : d) {
                    x = dist(rng);
                }
                break;
            }
            case InitKind::Ones:
                for (float& x : d) {
                    x = 1.0f;
                }
                break;
            case InitKind::Zeros:
                for (float& x : d) {
                    x = 0.0f;
                }
                break;
            case InitKind::WnG:
            default:
                break;
        }
    }

    // 二遍：weight_g = ‖weight_v‖（对除最后一维外的所有维求和）
    std::unordered_map<std::string, size_t> by_name;
    for (size_t i = 0; i < specs.size(); ++i) {
        by_name[specs[i].name] = i;
    }
    for (size_t i = 0; i < specs.size(); ++i) {
        const TensorSpec& s = specs[i];
        if (s.init != InitKind::WnG) {
            continue;
        }
        const std::string suffix = ".weight_g";
        if (s.name.size() <= suffix.size() ||
            s.name.compare(s.name.size() - suffix.size(), suffix.size(), suffix) != 0) {
            if (err != nullptr) {
                *err = "WnG 名称非法: " + s.name;
            }
            return false;
        }
        const std::string vname = s.name.substr(0, s.name.size() - suffix.size()) + ".weight_v";
        const auto it = by_name.find(vname);
        if (it == by_name.end()) {
            if (err != nullptr) {
                *err = "WnG 缺少配对 weight_v: " + s.name;
            }
            return false;
        }
        const TensorSpec& vs = specs[it->second];
        const std::vector<float>& vd = (*out)[it->second];
        const int ch = vs.ch_dim;
        const int64_t out_ch = vs.ne[ch];
        std::vector<double> acc((size_t) out_ch, 0.0);
        // ne 行主序解码：遍历全部元素，累加到对应通道
        int64_t idx[4] = {0, 0, 0, 0};
        const int64_t total = (int64_t) vd.size();
        for (int64_t lin = 0; lin < total; ++lin) {
            // 解码 lin → idx（ggml ne 行主序：dim0 最内层）
            int64_t rem = lin;
            for (int d = 0; d < 4; ++d) {
                idx[d] = vs.ne[d] > 0 ? rem % vs.ne[d] : 0;
                rem = vs.ne[d] > 0 ? rem / vs.ne[d] : 0;
            }
            const int64_t c = idx[ch];
            const float x = vd[(size_t) lin];
            acc[(size_t) c] += (double) x * (double) x;
        }
        std::vector<float>& gd = (*out)[i];
        gd.assign((size_t) out_ch, 0.0f);
        for (int64_t o = 0; o < out_ch; ++o) {
            gd[(size_t) o] = (float) std::sqrt(acc[(size_t) o]);
        }
    }
    return true;
}

} // namespace model
} // namespace rvc_train
