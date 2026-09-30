// RVC-Train-GGML - M3-3 双图 G/D + 一步 AdamW 对拍实现（src/train/train_step_selftest.cpp）
//
// 双图协议（交接说明 §4.7 C；架构设计 §5.1）：
//   G 图（ctx_g）：synth 切片前向（include_full=false）→ D(冻结, 新建节点)
//                  → LSGAN(G) + feature matching + 45·mel L1 + 1·KL
//   D 图（ctx_d）：D(real)+D(fake)（独立 ctx，禁止复用 G 图节点）→ LSGAN(D)
//   步序（对齐官方 81eed5e train.py:425-527）：先 D 步（用 pre-update y_hat.detach()），
//   把更新后的 D 权重拷回 G 图 → G 图重算（D 冻结节点读到更新后的 D）→ G 步。
// 内存：两个 ctx 各自 buffer_alloc_graph_tensors（图级 liveness 复用；Q18 处置已落地）。
//   M5-0 起训练循环将改为**单 ctx 共享参数双图**（见 `交接说明` §4.9 C）；本 selftest 保持
//   M3-3 已验证形态不变，作为 golden 基准。
// 无泄漏断言：G 图反向展开后 D 参数无 grad_acc；D 步后 G 参数逐位不变；G 步后 D 参数逐位不变。
#include "train/train_step_selftest.h"

#include "common/json.h"
#include "data/dataset.h"
#include "model/discriminator.h"
#include "model/flow.h"
#include "model/model.h"
#include "model/selftest.h"
#include "model/sinegen.h"
#include "model/synthesizer.h"
#include "model/weights.h"
#include "train/losses.h"
#include "train/stft_mel.h"
#include "train/train_step.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace rvc_train {
namespace train {

namespace {

using namespace traincpp;
using model::GoldenManifest;
using model::GoldenTensor;
using model::MpdConfig;
using model::MpdModule;
using model::MpdForward;
using model::ModelConfig;
using model::ModelParams;
using model::SynthConfig;
using model::SynthInputs;
using model::SynthModules;
using model::SynthOutputs;
using model::TrainMeta;
using model::TrainWeights;

constexpr double kDeltaTol = 1e-4;  // ΔW 差值噪声下限（硬门/kink 判定见 M3-3 统计口径，非相对门）
constexpr double kLossTol  = 1e-4;  // loss 相对门
constexpr double kFwdTol   = 1e-4;  // 前向中间量相对门

// ---------------------------------------------------------------- 基础工具

std::string join_path(const std::string& dir, const std::string& file) {
    if (dir.empty()) {
        return file;
    }
    const char last = dir.back();
    if (last == '/' || last == '\\') {
        return dir + file;
    }
    return dir + "/" + file;
}

bool read_file_bytes(const std::string& path, std::vector<uint8_t>* out, std::string* err) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        if (err != nullptr) {
            *err = "无法打开文件: " + path;
        }
        return false;
    }
    f.seekg(0, std::ios::end);
    const std::streamoff n = f.tellg();
    f.seekg(0, std::ios::beg);
    out->resize((size_t) (n > 0 ? n : 0));
    if (n > 0) {
        f.read((char*) out->data(), n);
    }
    if (!f) {
        if (err != nullptr) {
            *err = "读取失败: " + path;
        }
        return false;
    }
    return true;
}

struct CompareStat {
    double max_abs = 0.0;
    double max_ref = 0.0;
    double rel     = 0.0;
    double rms_rel = 0.0;
};

void compare_f32(const float* ref, const float* got, int64_t n, CompareStat* st) {
    double sum_sq = 0.0;
    double ref_sq = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        const double d = std::fabs((double) got[i] - (double) ref[i]);
        st->max_abs = std::max(st->max_abs, d);
        st->max_ref = std::max(st->max_ref, std::fabs((double) ref[i]));
        sum_sq += d * d;
        ref_sq += (double) ref[i] * (double) ref[i];
    }
    st->rel = st->max_abs / std::max(1.0, st->max_ref);
    const double rms_d   = std::sqrt(sum_sq / std::max((double) n, 1.0));
    const double rms_ref = std::sqrt(ref_sq / std::max((double) n, 1.0));
    st->rms_rel = rms_d / std::max(1e-12, rms_ref);
}

// ---------------------------------------------------------------- JSON 参数读取

int64_t j_int(const common::JsonValue& o, const char* key, int64_t def) {
    const common::JsonValue* v = o.find(key);
    return (v != nullptr) ? v->as_int(def) : def;
}

double j_num(const common::JsonValue& o, const char* key, double def) {
    const common::JsonValue* v = o.find(key);
    return (v != nullptr) ? v->as_number(def) : def;
}

std::string j_str(const common::JsonValue& o, const char* key, const char* def) {
    const common::JsonValue* v = o.find(key);
    return (v != nullptr) ? v->as_string() : std::string(def);
}

void j_int_array(const common::JsonValue& o, const char* key, std::vector<int64_t>* out) {
    out->clear();
    const common::JsonValue* v = o.find(key);
    if (v == nullptr || !v->is_array()) {
        return;
    }
    for (size_t i = 0; i < v->size(); ++i) {
        out->push_back(v->at(i).as_int(0));
    }
}

// ---------------------------------------------------------------- golden 文件读取

bool entry_bytes(const GoldenManifest& gm, const std::string& name, std::vector<uint8_t>* out,
                 std::string* err) {
    const GoldenTensor* gt = gm.find_tensor(name);
    if (gt == nullptr) {
        if (err != nullptr) {
            *err = "golden 缺少条目: " + name;
        }
        return false;
    }
    return read_file_bytes(join_path(gm.dir, gt->file), out, err);
}

bool entry_f32(const GoldenManifest& gm, const std::string& name, std::vector<float>* out,
               std::string* err) {
    std::vector<uint8_t> bytes;
    if (!entry_bytes(gm, name, &bytes, err)) {
        return false;
    }
    if (bytes.size() % sizeof(float) != 0) {
        if (err != nullptr) {
            *err = "文件大小非 F32 整数倍: " + name;
        }
        return false;
    }
    out->resize(bytes.size() / sizeof(float));
    std::memcpy(out->data(), bytes.data(), bytes.size());
    return true;
}

bool entry_i32(const GoldenManifest& gm, const std::string& name, std::vector<int32_t>* out,
               std::string* err) {
    std::vector<uint8_t> bytes;
    if (!entry_bytes(gm, name, &bytes, err)) {
        return false;
    }
    if (bytes.size() % sizeof(int32_t) != 0) {
        if (err != nullptr) {
            *err = "文件大小非 I32 整数倍: " + name;
        }
        return false;
    }
    out->resize(bytes.size() / sizeof(int32_t));
    std::memcpy(out->data(), bytes.data(), bytes.size());
    return true;
}

bool fill_tensor_from_entry(Tensor* t, const GoldenManifest& gm, const std::string& name,
                            std::string* err) {
    std::vector<uint8_t> bytes;
    if (!entry_bytes(gm, name, &bytes, err)) {
        return false;
    }
    const size_t expect = (size_t) tensor_nelements(t) * tensor_element_size(t);
    if (bytes.size() != expect) {
        char buf[256];
        std::snprintf(buf, sizeof(buf), "输入文件大小不符: %s（%zu != %zu）", name.c_str(),
                      bytes.size(), expect);
        if (err != nullptr) {
            *err = buf;
        }
        return false;
    }
    tensor_set(t, bytes.data(), 0, bytes.size());
    return true;
}

// ---------------------------------------------------------------- 参数快照 / ΔW

void read_params(const ModelParams& params, std::map<std::string, std::vector<float>>* out) {
    for (Tensor* t : params.params().items) {
        std::vector<float>& v = (*out)[t->name];
        v.resize((size_t) tensor_nelements(t));
        if (!v.empty()) {
            tensor_get(t, v.data(), 0, v.size() * sizeof(float));
        }
    }
}

bool values_equal(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        return false;
    }
    return a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

std::vector<float> make_delta(const std::vector<float>& after, const std::vector<float>& before) {
    std::vector<float> d(after.size());
    for (size_t i = 0; i < after.size(); ++i) {
        d[i] = after[i] - before[i];
    }
    return d;
}

// ---------------------------------------------------------------- 配置装配

SynthConfig synth_config_from_json(const common::JsonValue& p) {
    SynthConfig cfg;
    cfg.segment_size = j_int(p, "segment_size", 0);
    cfg.frames       = j_int(p, "frames", 0);
    cfg.batch        = j_int(p, "batch", 1);

    cfg.enc_p.prefix          = j_str(p, "enc_p_prefix", "enc_p");
    cfg.enc_p.in_channels     = j_int(p, "enc_p_in", 0);
    cfg.enc_p.hidden_channels = j_int(p, "enc_p_hidden", 0);
    cfg.enc_p.filter_channels = j_int(p, "enc_p_filter", 0);
    cfg.enc_p.n_heads         = j_int(p, "enc_p_heads", 0);
    cfg.enc_p.n_layers        = j_int(p, "enc_p_layers", 0);
    cfg.enc_p.kernel_size     = j_int(p, "enc_p_kernel", 3);
    cfg.enc_p.window_size     = j_int(p, "enc_p_window", 10);
    cfg.enc_p.out_channels    = j_int(p, "enc_p_out", 0);

    cfg.enc_q.prefix          = j_str(p, "enc_q_prefix", "enc_q");
    cfg.enc_q.in_channels     = j_int(p, "enc_q_in", 0);
    cfg.enc_q.out_channels    = j_int(p, "enc_q_out", 0);
    cfg.enc_q.hidden_channels = j_int(p, "enc_q_hidden", 0);
    cfg.enc_q.kernel_size     = j_int(p, "enc_q_kernel", 5);
    cfg.enc_q.dilation_rate   = j_int(p, "enc_q_dilation", 1);
    cfg.enc_q.n_layers        = j_int(p, "enc_q_layers", 16);
    cfg.enc_q.gin_channels    = j_int(p, "enc_q_gin", 0);

    cfg.flow.prefix          = j_str(p, "flow_prefix", "flow");
    cfg.flow.channels        = j_int(p, "flow_channels", 0);
    cfg.flow.hidden_channels = j_int(p, "flow_hidden", 0);
    cfg.flow.kernel_size     = j_int(p, "flow_kernel", 5);
    cfg.flow.dilation_rate   = j_int(p, "flow_dilation", 1);
    cfg.flow.n_layers        = j_int(p, "flow_layers", 3);
    cfg.flow.gin_channels    = j_int(p, "flow_gin", 0);
    cfg.flow.n_flows         = j_int(p, "flow_n_flows", 4);

    cfg.dec.prefix                   = j_str(p, "dec_prefix", "dec");
    cfg.dec.in_channels              = j_int(p, "dec_in", 0);
    cfg.dec.upsample_initial_channel = j_int(p, "dec_up_init", 0);
    cfg.dec.gin_channels             = j_int(p, "dec_gin", 0);
    j_int_array(p, "dec_upsample_rates", &cfg.dec.upsample_rates);
    j_int_array(p, "dec_upsample_kernel_sizes", &cfg.dec.upsample_kernel_sizes);
    j_int_array(p, "dec_resblock_kernel_sizes", &cfg.dec.resblock_kernel_sizes);
    j_int_array(p, "dec_resblock_dilation_sizes", &cfg.dec.resblock_dilation_sizes);
    return cfg;
}

MpdConfig disc_config_from_json(const common::JsonValue& p) {
    MpdConfig cfg;
    cfg.prefix = j_str(p, "prefix", "discriminators");
    j_int_array(p, "periods", &cfg.periods);
    cfg.kernel_size = j_int(p, "kernel_size", 5);
    cfg.stride      = j_int(p, "stride", 3);
    return cfg;
}

// ---------------------------------------------------------------- 主流程

int run_impl(const std::string& weights_g, const std::string& weights_d,
             const std::string& golden_dir, Device* dev) {
    if (dev == nullptr) {
        std::printf("[train-step] 无可用设备\n");
        return 2;
    }
    std::printf("[train-step] golden=%s\n", golden_dir.c_str());

    // ---------------- 打开权重与 golden ----------------
    std::string err;
    TrainWeights* wg = TrainWeights::open(weights_g, &err);
    if (wg == nullptr) {
        std::printf("[train-step] 打开 G 权重失败: %s\n", err.c_str());
        return 1;
    }
    TrainWeights* wd = TrainWeights::open(weights_d, &err);
    if (wd == nullptr) {
        std::printf("[train-step] 打开 D 权重失败: %s\n", err.c_str());
        delete wg;
        return 1;
    }
    TrainMeta meta_g;
    TrainMeta meta_d;
    wg->read_meta(&meta_g);
    wd->read_meta(&meta_d);
    const ModelConfig cfg = ModelConfig::from_meta(meta_g);
    if (!cfg.validate(&err)) {
        std::printf("[train-step] 配置校验失败: %s\n", err.c_str());
        delete wg;
        delete wd;
        return 1;
    }

    GoldenManifest gm;
    if (!model::golden_manifest_load(golden_dir, "train_step", &gm, &err)) {
        std::printf("[train-step] %s\n", err.c_str());
        delete wg;
        delete wd;
        return 1;
    }
    const common::JsonValue* synth_json = gm.params.find("synth");
    const common::JsonValue* disc_json  = gm.params.find("disc");
    if (synth_json == nullptr || disc_json == nullptr) {
        std::printf("[train-step] manifest.params.synth/disc 缺失（需重新生成 golden）\n");
        delete wg;
        delete wd;
        return 1;
    }
    const SynthConfig scfg = synth_config_from_json(*synth_json);
    const MpdConfig   dcfg = disc_config_from_json(*disc_json);

    const int64_t disc_len = gm.param_int("disc_len", 2048);
    const int64_t frames   = gm.param_int("frames", 48);
    const int64_t B        = gm.param_int("batch", 1);
    const int64_t upp      = gm.param_int("upp", 400);
    const int64_t seg      = scfg.segment_size;
    const int64_t total_slice = seg * upp;
    if (B != 1) {
        std::printf("[train-step] 当前仅支持 B=1（golden batch=%lld）\n", (long long) B);
        delete wg;
        delete wd;
        return 2;
    }

    AdamwOptions opts;
    opts.lr           = (float) j_num(gm.params, "lr", 1e-4);
    opts.beta1        = (float) j_num(gm.params, "beta1", 0.8);
    opts.beta2        = (float) j_num(gm.params, "beta2", 0.99);
    opts.eps          = (float) j_num(gm.params, "eps", 1e-9);
    opts.weight_decay = (float) j_num(gm.params, "weight_decay", 1e-2);
    const float c_mel = (float) j_num(gm.params, "c_mel", 45.0);
    const float c_kl  = (float) j_num(gm.params, "c_kl", 1.0);
    std::printf("[train-step] 超参 lr=%.1e betas=(%.2f,%.2f) eps=%.1e wd=%.3f c_mel=%.1f "
                "c_kl=%.1f disc_len=%lld\n",
                (double) opts.lr, (double) opts.beta1, (double) opts.beta2, (double) opts.eps,
                (double) opts.weight_decay, (double) c_mel, (double) c_kl, (long long) disc_len);

    // ---------------- ctx_g：模型装配 + G 图 ----------------
    Context* ctx_g = context_new(1u << 22);
    if (ctx_g == nullptr) {
        std::printf("[train-step] ctx_g 创建失败\n");
        delete wg;
        delete wd;
        return 1;
    }
    ModelParams g_params;
    ModelParams d_params_g;
    if (!g_params.build_all(ctx_g, *wg, &err) || !d_params_g.build_all(ctx_g, *wd, &err)) {
        std::printf("[train-step] 建参失败: %s\n", err.c_str());
        context_free(ctx_g);
        delete wg;
        delete wd;
        return 1;
    }
    MpdModule mpd_g;
    if (!model::mpd_load(d_params_g, dcfg, &mpd_g, &err)) {
        std::printf("[train-step] D 装配失败: %s\n", err.c_str());
        context_free(ctx_g);
        delete wg;
        delete wd;
        return 1;
    }
    SynthModules synth;
    if (!model::synth_load(g_params, scfg, &synth, &err)) {
        std::printf("[train-step] G 装配失败: %s\n", err.c_str());
        context_free(ctx_g);
        delete wg;
        delete wd;
        return 1;
    }

    // 图内输入（仅创建图真正消费的叶子；lengths/pitchf/ids_slice/sine_noise_slice 走主机读取）
    static const char* kGraphInputs[] = {"phone", "pitch", "x_mask", "spec", "y_mask", "sid",
                                         "q_noise", "wave", "y_mel"};
    std::map<std::string, Tensor*> inputs;
    for (const char* name : kGraphInputs) {
        const GoldenTensor* gt = gm.find_tensor(name);
        if (gt == nullptr || !gt->is_input()) {
            std::printf("[train-step] golden 缺少输入 %s\n", name);
            context_free(ctx_g);
            delete wg;
            delete wd;
            return 1;
        }
        const Type type = (gt->dtype == "i32") ? TYPE_I32 : TYPE_F32;
        Tensor* t = new_tensor_nd(ctx_g, type, gt->nd, gt->ne.data());
        if (t == nullptr) {
            std::printf("[train-step] 创建输入失败: %s\n", name);
            context_free(ctx_g);
            delete wg;
            delete wd;
            return 1;
        }
        tensor_set_name(t, "%s", name);
        inputs[name] = t;
    }

    // 叶子（分配后由 prepare 填充）
    Tensor* attn_valid = new_tensor_4d(ctx_g, TYPE_F32, frames, frames, 1, B);
    Tensor* flip_perm  = new_tensor_2d(ctx_g, TYPE_F32, scfg.flow.channels, scfg.flow.channels);
    Tensor* slice_mat  = new_tensor_3d(ctx_g, TYPE_F32, frames, seg, B);
    Tensor* sine_slice = new_tensor_3d(ctx_g, TYPE_F32, 1, total_slice, B);
    if (attn_valid == nullptr || flip_perm == nullptr || slice_mat == nullptr ||
        sine_slice == nullptr) {
        std::printf("[train-step] 创建叶子失败\n");
        context_free(ctx_g);
        delete wg;
        delete wd;
        return 1;
    }
    tensor_set_name(attn_valid, "attn_valid");
    tensor_set_name(flip_perm, "train_flip_perm");
    tensor_set_name(slice_mat, "train_slice_mat");
    tensor_set_name(sine_slice, "train_sine_source_slice");

    SynthInputs in;
    in.sid               = inputs["sid"];
    in.phone             = inputs["phone"];
    in.pitch             = inputs["pitch"];
    in.x_mask            = inputs["x_mask"];
    in.attn_valid        = attn_valid;
    in.spec              = inputs["spec"];
    in.y_mask            = inputs["y_mask"];
    in.q_noise           = inputs["q_noise"];
    in.flip_perm         = flip_perm;
    in.slice_mat         = slice_mat;
    in.sine_source_slice = sine_slice;
    in.include_full      = false;
    SynthOutputs out = model::synth_forward(ctx_g, g_params, synth, in);
    if (out.o_slice == nullptr) {
        std::printf("[train-step] synth_forward 失败\n");
        context_free(ctx_g);
        delete wg;
        delete wd;
        return 1;
    }
    // D 输入 = y_hat 前 disc_len 个采样（测试尺度；view 可反传，见 M3-0 计划 disc L=2048）
    Tensor* y_hat_d = view_3d(ctx_g, out.o_slice, disc_len, 1, B, 0);
    if (y_hat_d == nullptr) {
        std::printf("[train-step] y_hat_d 视图创建失败\n");
        context_free(ctx_g);
        delete wg;
        delete wd;
        return 1;
    }
    tensor_set_name(y_hat_d, "train_y_hat_d");

    MpdForward f_g = model::mpd_forward(ctx_g, mpd_g, inputs["wave"], y_hat_d);
    if (f_g.r.empty() || f_g.r.size() != f_g.g.size()) {
        std::printf("[train-step] G 图 D 前向失败\n");
        context_free(ctx_g);
        delete wg;
        delete wd;
        return 1;
    }
    std::vector<Tensor*> dg_outs;
    std::vector<std::vector<Tensor*>> dr_fmaps;
    std::vector<std::vector<Tensor*>> dg_fmaps;
    for (size_t i = 0; i < f_g.r.size(); ++i) {
        dg_outs.push_back(f_g.g[i].out);
        dr_fmaps.push_back(f_g.r[i].fmap);
        dg_fmaps.push_back(f_g.g[i].fmap);
    }

    StftMelConfig smcfg;
    smcfg.n_fft  = gm.param_int("n_fft", 2048);
    smcfg.hop    = gm.param_int("hop", 400);
    smcfg.win    = gm.param_int("win", 2048);
    smcfg.n_mels = gm.param_int("n_mels", 125);
    smcfg.sr     = j_num(gm.params, "sr", 40000.0);
    smcfg.fmin   = j_num(gm.params, "fmin", 0.0);
    smcfg.fmax   = j_num(gm.params, "fmax", -1.0);
    smcfg.mag_eps  = (float) j_num(gm.params, "mag_eps", 2e-7);
    smcfg.log_clip = (float) j_num(gm.params, "log_clip", 2e-6);
    if (!smcfg.validate(&err)) {
        std::printf("[train-step] STFT/mel 配置无效: %s\n", err.c_str());
        context_free(ctx_g);
        delete wg;
        delete wd;
        return 1;
    }
    StftMelConsts smconsts;
    stft_mel_consts_init(ctx_g, smcfg, &smconsts);

    Tensor* loss_gen = loss_lsgan_generator(ctx_g, dg_outs);
    Tensor* loss_fm  = loss_feature_matching(ctx_g, dr_fmaps, dg_fmaps);
    Tensor* y2d = reshape_2d(ctx_g, cont(ctx_g, out.o_slice), total_slice, B);
    Tensor* y_hat_mel = (y2d != nullptr)
                            ? mel_spectrogram_forward(ctx_g, smconsts, smcfg, y2d)
                            : nullptr;
    Tensor* loss_mel = (y_hat_mel != nullptr)
                           ? scale(ctx_g, loss_l1_mean(ctx_g, inputs["y_mel"], y_hat_mel), c_mel)
                           : nullptr;
    Tensor* loss_kl = scale(ctx_g,
                            rvc_train::train::loss_kl(ctx_g, out.z_p, out.logs_q, out.m_p,
                                                      out.logs_p, inputs["y_mask"]),
                            c_kl);
    if (loss_gen == nullptr || loss_fm == nullptr || loss_mel == nullptr || loss_kl == nullptr) {
        std::printf("[train-step] G 损失建图失败\n");
        context_free(ctx_g);
        delete wg;
        delete wd;
        return 1;
    }
    Tensor* loss_gen_all = add(ctx_g, add(ctx_g, add(ctx_g, loss_gen, loss_fm), loss_mel),
                               loss_kl);
    tensor_set_name(loss_gen_all, "loss_gen_all");
    tensor_set_name(loss_gen, "loss_gen");
    tensor_set_name(loss_fm, "loss_fm");
    tensor_set_name(loss_mel, "loss_mel");
    tensor_set_name(loss_kl, "loss_kl");
    tensor_set_loss(loss_gen_all);
    // 分量标为 OUTPUT 使其在图级分配下保持可读（仅对拍读数；不参与反向播种）
    loss_gen->flags |= TENSOR_FLAG_OUTPUT;
    loss_fm->flags |= TENSOR_FLAG_OUTPUT;
    loss_mel->flags |= TENSOR_FLAG_OUTPUT;
    loss_kl->flags |= TENSOR_FLAG_OUTPUT;
    // FIX-002（Q23）：compute 后仍由主机读取的视图/视图根也须标 OUTPUT
    // （:839 预热后读 y_hat_d；:1000-1014 前向对拍读 o_slice）
    out.o_slice->flags |= TENSOR_FLAG_OUTPUT;
    y_hat_d->flags |= TENSOR_FLAG_OUTPUT;

    for (Tensor* t : g_params.params().items) {
        tensor_set_param(t);
    }
    Graph* g_graph = graph_new(ctx_g);
    if (g_graph == nullptr) {
        std::printf("[train-step] G 图创建失败\n");
        context_free(ctx_g);
        delete wg;
        delete wd;
        return 1;
    }
    graph_build_forward_expand(ctx_g, g_graph, loss_gen_all);
    graph_build_backward_expand(g_graph);

    int leak_failed = 0;
    for (Tensor* t : d_params_g.params().items) {
        if (t->grad_acc != nullptr) {
            std::printf("[train-step] [FAIL] 冻结侧泄漏: G 图反向展开后 D 参数 %s 有 grad_acc\n",
                        t->name);
            ++leak_failed;
        }
    }
    if (leak_failed == 0) {
        std::printf("[train-step] 无泄漏(A)：G 图冻结 D（%lld 参数无 grad_acc）OK\n",
                    (long long) d_params_g.count());
    }

    std::printf("[train-step] 后端 = %s；G 图分配中（图级 liveness）...\n",
                dev->props().name.c_str());
    Buffer* buf_g = buffer_alloc_graph_tensors(g_graph, dev->default_buffer_type());
    if (buf_g == nullptr) {
        std::printf("[train-step] G 图分配失败（可用 RVCT_DEBUG_ALLOC=1 诊断；Q18 预案：降 T/分 ctx）\n");
        graph_free(g_graph);
        context_free(ctx_g);
        delete wg;
        delete wd;
        return 1;
    }

    // ---------------- ctx_d：D 图 ----------------
    Context* ctx_d = context_new(1u << 20);
    ModelParams d_params_d;
    if (ctx_d == nullptr || !d_params_d.build_all(ctx_d, *wd, &err)) {
        std::printf("[train-step] ctx_d/建参失败: %s\n", err.c_str());
        context_free(ctx_d);
        graph_free(g_graph);
        buffer_free(buf_g);
        context_free(ctx_g);
        delete wg;
        delete wd;
        return 1;
    }
    MpdModule mpd_d;
    if (!model::mpd_load(d_params_d, dcfg, &mpd_d, &err)) {
        std::printf("[train-step] D 图装配失败: %s\n", err.c_str());
        context_free(ctx_d);
        graph_free(g_graph);
        buffer_free(buf_g);
        context_free(ctx_g);
        delete wg;
        delete wd;
        return 1;
    }
    Tensor* wave_d = new_tensor_3d(ctx_d, TYPE_F32, disc_len, 1, B);
    Tensor* fake_d = new_tensor_3d(ctx_d, TYPE_F32, disc_len, 1, B);
    if (wave_d == nullptr || fake_d == nullptr) {
        std::printf("[train-step] D 图叶子创建失败\n");
        context_free(ctx_d);
        graph_free(g_graph);
        buffer_free(buf_g);
        context_free(ctx_g);
        delete wg;
        delete wd;
        return 1;
    }
    tensor_set_name(wave_d, "d_wave");
    tensor_set_name(fake_d, "d_fake");
    MpdForward f_d = model::mpd_forward(ctx_d, mpd_d, wave_d, fake_d);
    if (f_d.r.empty() || f_d.r.size() != f_d.g.size()) {
        std::printf("[train-step] D 图前向失败\n");
        context_free(ctx_d);
        graph_free(g_graph);
        buffer_free(buf_g);
        context_free(ctx_g);
        delete wg;
        delete wd;
        return 1;
    }
    std::vector<Tensor*> dr_outs;
    std::vector<Tensor*> dg_outs_d;
    for (size_t i = 0; i < f_d.r.size(); ++i) {
        dr_outs.push_back(f_d.r[i].out);
        dg_outs_d.push_back(f_d.g[i].out);
    }
    Tensor* loss_disc = loss_lsgan_discriminator(ctx_d, dr_outs, dg_outs_d);
    if (loss_disc == nullptr) {
        std::printf("[train-step] D 损失建图失败\n");
        context_free(ctx_d);
        graph_free(g_graph);
        buffer_free(buf_g);
        context_free(ctx_g);
        delete wg;
        delete wd;
        return 1;
    }
    tensor_set_name(loss_disc, "loss_disc");
    tensor_set_loss(loss_disc);
    for (Tensor* t : d_params_d.params().items) {
        tensor_set_param(t);
    }
    Graph* d_graph = graph_new(ctx_d);
    if (d_graph == nullptr) {
        std::printf("[train-step] D 图创建失败\n");
        context_free(ctx_d);
        graph_free(g_graph);
        buffer_free(buf_g);
        context_free(ctx_g);
        delete wg;
        delete wd;
        return 1;
    }
    graph_build_forward_expand(ctx_d, d_graph, loss_disc);
    graph_build_backward_expand(d_graph);
    Buffer* buf_d = buffer_alloc_graph_tensors(d_graph, dev->default_buffer_type());
    if (buf_d == nullptr) {
        std::printf("[train-step] D 图分配失败\n");
        graph_free(d_graph);
        context_free(ctx_d);
        graph_free(g_graph);
        buffer_free(buf_g);
        context_free(ctx_g);
        delete wg;
        delete wd;
        return 1;
    }
    std::printf("[train-step] ctx_g 图缓冲 %.1f MiB；ctx_d 图缓冲 %.1f MiB\n",
                (double) buf_g->size() / (1024.0 * 1024.0),
                (double) buf_d->size() / (1024.0 * 1024.0));

    // ---------------- 权重装载 ----------------
    for (Tensor* t : g_params.params().items) {
        if (!wg->load(t, t->name, &err)) {
            std::printf("[train-step] G 参数装载失败: %s\n", err.c_str());
            return 1;
        }
    }
    for (Tensor* t : d_params_g.params().items) {
        if (!wd->load(t, t->name, &err)) {
            std::printf("[train-step] G 图 D 参数装载失败: %s\n", err.c_str());
            return 1;
        }
    }
    for (Tensor* t : d_params_d.params().items) {
        if (!wd->load(t, t->name, &err)) {
            std::printf("[train-step] D 图 D 参数装载失败: %s\n", err.c_str());
            return 1;
        }
    }

    // ---------------- 注入输入 + prepare ----------------
    {
        std::vector<uint8_t> bytes;
        for (const auto& kv : inputs) {
            if (!fill_tensor_from_entry(kv.second, gm, kv.first, &err)) {
                std::printf("[train-step] 输入注入失败: %s\n", err.c_str());
                return 1;
            }
        }
        // wave → D 图叶子（同一份数据）
        std::vector<float> wv((size_t) disc_len * (size_t) B);
        tensor_get(inputs["wave"], wv.data(), 0, wv.size() * sizeof(float));
        tensor_set(wave_d, wv.data(), 0, wv.size() * sizeof(float));

        if (!model::flow_fill_flip_perm(flip_perm, &err)) {
            std::printf("[train-step] flip 置换矩阵填充失败: %s\n", err.c_str());
            return 1;
        }
        // attn_valid = x_mask 外积
        {
            std::vector<float> xm((size_t) (frames * B));
            tensor_get(inputs["x_mask"], xm.data(), 0, xm.size() * sizeof(float));
            std::vector<float> av((size_t) (frames * frames * B), 0.0f);
            for (int64_t b = 0; b < B; ++b) {
                for (int64_t tq = 0; tq < frames; ++tq) {
                    const bool qv = xm[(size_t) (tq + frames * b)] > 0.5f;
                    for (int64_t tk = 0; tk < frames; ++tk) {
                        const bool kv = xm[(size_t) (tk + frames * b)] > 0.5f;
                        av[(size_t) (tk + frames * tq + frames * frames * b)] =
                            (qv && kv) ? 1.0f : 0.0f;
                    }
                }
            }
            tensor_set(attn_valid, av.data(), 0, av.size() * sizeof(float));
        }
        // slice_mat / pitchf_slice（主机读取 golden）
        std::vector<int32_t> idv;
        std::vector<float> pfall;
        std::vector<float> noise;
        if (!entry_i32(gm, "ids_slice", &idv, &err) || !entry_f32(gm, "pitchf", &pfall, &err) ||
            !entry_f32(gm, "sine_noise_slice", &noise, &err)) {
            std::printf("[train-step] 主机输入读取失败: %s\n", err.c_str());
            return 1;
        }
        if ((int64_t) idv.size() != B || (int64_t) pfall.size() != frames * B ||
            (int64_t) noise.size() != total_slice * B) {
            std::printf("[train-step] 主机输入尺寸不符（ids=%zu pitchf=%zu noise=%zu）\n",
                        idv.size(), pfall.size(), noise.size());
            return 1;
        }
        std::vector<float> sm((size_t) (frames * seg * B), 0.0f);
        std::vector<float> pfs((size_t) (seg * B), 0.0f);
        for (int64_t b = 0; b < B; ++b) {
            const int64_t id = idv[(size_t) b];
            if (id < 0 || id + seg > frames) {
                std::printf("[train-step] ids_slice 越界: %lld\n", (long long) id);
                return 1;
            }
            for (int64_t s = 0; s < seg; ++s) {
                sm[(size_t) (id + s + frames * s + frames * seg * b)] = 1.0f;
                pfs[(size_t) (s + seg * b)] = pfall[(size_t) (id + s + frames * b)];
            }
        }
        tensor_set(slice_mat, sm.data(), 0, sm.size() * sizeof(float));

        // SineGen（旧 cumsum_shift 法；噪声 = golden 落盘）
        const int64_t sr = (int64_t) smcfg.sr;
        std::vector<float> ss((size_t) (total_slice * B), 0.0f);
        for (int64_t b = 0; b < B; ++b) {
            std::vector<float> f0s((size_t) seg);
            for (int64_t s = 0; s < seg; ++s) {
                f0s[(size_t) s] = pfs[(size_t) (s + seg * b)];
            }
            std::vector<float> s = model::sinegen_generate(
                f0s.data(), seg, (int) sr, (int) upp, noise.data() + (size_t) b * total_slice,
                total_slice, 1234);
            if ((int64_t) s.size() != total_slice) {
                std::printf("[train-step] SineGen 输出长度不符\n");
                return 1;
            }
            std::memcpy(ss.data() + (size_t) b * total_slice, s.data(),
                        sizeof(float) * (size_t) total_slice);
        }
        tensor_set(sine_slice, ss.data(), 0, ss.size() * sizeof(float));

        // STFT/mel 常量
        stft_mel_fill_dft_kernel(&smconsts, smcfg);
        stft_mel_fill_mel_basis(&smconsts, smcfg);
    }

    // ---------------- 优化器 ----------------
    std::vector<Tensor*> g_opt;
    std::vector<Tensor*> d_opt;
    for (Tensor* t : g_params.params().items) {
        if (t->grad_acc != nullptr) {
            g_opt.push_back(t);
        }
    }
    for (Tensor* t : d_params_d.params().items) {
        if (t->grad_acc != nullptr) {
            d_opt.push_back(t);
        }
    }
    std::printf("[train-step] 优化器参数：G %zu/%lld，D %zu/%lld\n", g_opt.size(),
                (long long) g_params.count(), d_opt.size(), (long long) d_params_d.count());
    Optimizer* optim_g = optim_adamw_new(ctx_g, g_opt.data(), (int64_t) g_opt.size(), opts);
    Optimizer* optim_d = optim_adamw_new(ctx_d, d_opt.data(), (int64_t) d_opt.size(), opts);

    // ---------------- 双图训练步 ----------------
    // 1) G 图预热：先得到 y_hat（pre-update；同时算出 pre-update D 的梯度，随后丢弃）
    graph_reset(g_graph);
    if (!dev->graph_compute(g_graph)) {
        std::printf("[train-step] G 图预热 compute 失败\n");
        return 1;
    }
    {
        std::vector<float> yhd((size_t) disc_len * (size_t) B);
        tensor_get(y_hat_d, yhd.data(), 0, yhd.size() * sizeof(float));
        tensor_set(fake_d, yhd.data(), 0, yhd.size() * sizeof(float));
    }

    std::map<std::string, std::vector<float>> g_snap;
    std::map<std::string, std::vector<float>> d_snap_g;
    read_params(g_params, &g_snap);
    read_params(d_params_g, &d_snap_g);

    // 2) D 步（fake = y_hat.detach() 的等价值）
    graph_reset(d_graph);
    if (!dev->graph_compute(d_graph)) {
        std::printf("[train-step] D 图 compute 失败\n");
        return 1;
    }
    std::map<std::string, std::vector<float>> d_before;
    read_params(d_params_d, &d_before);
    optim_step(optim_d);
    std::map<std::string, std::vector<float>> d_after;
    read_params(d_params_d, &d_after);

    // D 侧快照（ctx_d 随后释放，之后不可再访问其张量/梯度）
    std::map<std::string, double> d_grad_max;
    for (Tensor* t : d_params_d.params().items) {
        if (t->grad_acc == nullptr) {
            continue;
        }
        std::vector<float> gv((size_t) tensor_nelements(t));
        tensor_get(t->grad_acc, gv.data(), 0, gv.size() * sizeof(float));
        double m = 0.0;
        for (float v : gv) {
            m = std::max(m, std::fabs((double) v));
        }
        d_grad_max[t->name] = m;
    }
    float loss_disc_value = 0.0f;
    tensor_get(loss_disc, &loss_disc_value, 0, sizeof(loss_disc_value));

    // 断言(B)：D 步后 G 参数逐位不变
    {
        std::map<std::string, std::vector<float>> now;
        read_params(g_params, &now);
        for (const auto& kv : g_snap) {
            if (!values_equal(kv.second, now[kv.first])) {
                std::printf("[train-step] [FAIL] D 步改动了 G 参数: %s\n", kv.first.c_str());
                ++leak_failed;
            }
        }
        if (leak_failed == 0) {
            std::printf("[train-step] 无泄漏(B)：D 步后 G 参数逐位不变 OK\n");
        }
    }

    // 3) 更新后的 D 权重拷回 G 图（G 图 D 节点每次 compute 时读取）
    for (Tensor* t : d_params_g.params().items) {
        Tensor* s = d_params_d.find(t->name);
        if (s == nullptr) {
            std::printf("[train-step] D 参数名不匹配: %s\n", t->name);
            return 1;
        }
        std::vector<uint8_t> bytes((size_t) tensor_nelements(t) * tensor_element_size(t));
        tensor_get(s, bytes.data(), 0, bytes.size());
        tensor_set(t, bytes.data(), 0, bytes.size());
    }
    std::map<std::string, std::vector<float>> d_snap2;
    read_params(d_params_g, &d_snap2);

    // ctx_d 释放（D 步已完成；ΔW/梯度/损失均已快照到主机）→ 峰值显存下降（Q18 预案：分 ctx）
    optim_free(optim_d);
    buffer_free(buf_d);
    graph_free(d_graph);
    context_free(ctx_d);
    optim_d = nullptr;
    buf_d   = nullptr;
    d_graph = nullptr;
    ctx_d   = nullptr;

    // 4) G 步（D 冻结节点读更新后的 D；graph_reset 清掉预热梯度）
    graph_reset(g_graph);
    if (!dev->graph_compute(g_graph)) {
        std::printf("[train-step] G 图（最终）compute 失败\n");
        return 1;
    }
    std::map<std::string, std::vector<float>> g_before;
    read_params(g_params, &g_before);
    optim_step(optim_g);
    std::map<std::string, std::vector<float>> g_after;
    read_params(g_params, &g_after);

    // 诊断（可选）：RVCT_DUMP_TRAINSTEP=DIR 落盘 C++ ΔW 与 grad_acc（供逐元素定位）
    {
        char*  dir_env = nullptr;
        size_t dir_len = 0;
        _dupenv_s(&dir_env, &dir_len, "RVCT_DUMP_TRAINSTEP");
        if (dir_env != nullptr && dir_env[0] != '\0') {
            const std::string base = dir_env;
            free(dir_env);
            auto dump_f32 = [&base](const char* sub, const std::string& name, const float* p,
                                    size_t n) {
                std::string s = std::string(sub) + "_" + name;
                for (char& c : s) {
                    if (c == '.') {
                        c = '_';
                    }
                }
                const std::string path = join_path(base, s + ".f32.raw");
                std::ofstream f(path, std::ios::binary | std::ios::trunc);
                if (f.is_open()) {
                    f.write((const char*) p, (std::streamsize) (n * sizeof(float)));
                }
            };
            for (const auto& kv : g_after) {
                const std::vector<float> d = make_delta(kv.second, g_before[kv.first]);
                dump_f32("g_delta", kv.first, d.data(), d.size());
            }
            for (Tensor* t : g_params.params().items) {
                if (t->grad_acc == nullptr) {
                    continue;
                }
                std::vector<float> gv((size_t) tensor_nelements(t));
                tensor_get(t->grad_acc, gv.data(), 0, gv.size() * sizeof(float));
                dump_f32("g_grad", t->name, gv.data(), gv.size());
            }
            for (const auto& kv : d_after) {
                const std::vector<float> d = make_delta(kv.second, d_before[kv.first]);
                dump_f32("d_delta", kv.first, d.data(), d.size());
            }
            std::printf("[train-step] 诊断落盘: %s（g_delta/g_grad/d_delta）\n", base.c_str());
        }
    }

    // 断言(C)：G 步后（ctx_g 中的）D 参数逐位不变
    {
        std::map<std::string, std::vector<float>> now;
        read_params(d_params_g, &now);
        for (const auto& kv : d_snap2) {
            if (!values_equal(kv.second, now[kv.first])) {
                std::printf("[train-step] [FAIL] G 步改动了 D 参数: %s\n", kv.first.c_str());
                ++leak_failed;
            }
        }
        if (leak_failed == 0) {
            std::printf("[train-step] 无泄漏(C)：G 步后 D 参数逐位不变 OK\n");
        }
    }

    // ---------------- 对拍：ΔW / loss / 前向 ----------------
    const auto get_f32 = [](Tensor* t) {
        float v = 0.0f;
        tensor_get(t, &v, 0, sizeof(v));
        return v;
    };
    int64_t checked = 0;
    int64_t failed  = 0;

    // 前向中间量：y_hat（o_slice）与 y_hat_d
    {
        struct FwdItem {
            const char* name;
            Tensor* t;
        };
        const FwdItem items[] = {{"y_hat", out.o_slice}, {"y_hat_d", y_hat_d}};
        for (const FwdItem& it : items) {
            const GoldenTensor* gt = gm.find_tensor(it.name);
            if (gt == nullptr) {
                continue;
            }
            std::vector<float> ref;
            if (!entry_f32(gm, it.name, &ref, &err) ||
                ref.size() != (size_t) tensor_nelements(it.t)) {
                std::printf("[train-step]   %-24s [FAIL] 参考读取失败: %s\n", it.name, err.c_str());
                ++failed;
                continue;
            }
            std::vector<float> got((size_t) tensor_nelements(it.t));
            tensor_get(it.t, got.data(), 0, got.size() * sizeof(float));
            CompareStat st;
            compare_f32(ref.data(), got.data(), (int64_t) ref.size(), &st);
            const bool ok = (st.rel <= kFwdTol);
            std::printf("[train-step]   %-24s ne=%-16s max_abs=%.3e rel=%.3e %s\n", it.name,
                        gt->file.c_str(), st.max_abs, st.rel, ok ? "OK" : "FAIL");
            ++checked;
            if (!ok) {
                ++failed;
            }
        }
    }

    // loss（loss_disc 用 ctx_d 释放前的快照值；其余来自 G 图）
    {
        // loss_disc 快照对比
        {
            std::vector<float> ref;
            if (!entry_f32(gm, "loss_disc", &ref, &err) || ref.empty()) {
                std::printf("[train-step]   %-24s [FAIL] 参考读取失败: %s\n", "loss_disc", err.c_str());
                ++failed;
            } else {
                const double rel = std::fabs((double) loss_disc_value - (double) ref[0]) /
                                   std::max(1.0, std::fabs((double) ref[0]));
                const bool ok = (rel <= kLossTol);
                std::printf("[train-step]   %-24s ref=%.6f got=%.6f rel=%.3e %s\n", "loss_disc",
                            (double) ref[0], (double) loss_disc_value, rel, ok ? "OK" : "FAIL");
                ++checked;
                if (!ok) {
                    ++failed;
                }
            }
        }
        struct LossItem {
            const char* name;
            Tensor* t;
            double tol;
        };
        // loss_gen：LSGAN(G)=Σ mean((1-dg)²) 在 dg≈1 处对 D 一步的 kink 元素（每个差 2·lr）
        // 呈二次放大（Vulkan 实测 2.8e-4，CPU 2.6e-5）→ 单独放宽到 5e-4；其余严格 1e-4。
        const LossItem items[] = {{"loss_gen_all", loss_gen_all, kLossTol},
                                  {"loss_gen", loss_gen, 5e-4},
                                  {"loss_fm", loss_fm, kLossTol},
                                  {"loss_mel", loss_mel, kLossTol},
                                  {"loss_kl", loss_kl, kLossTol}};
        for (const LossItem& it : items) {
            std::vector<float> ref;
            if (!entry_f32(gm, it.name, &ref, &err) || ref.size() < 1) {
                std::printf("[train-step]   %-24s [FAIL] 参考读取失败: %s\n", it.name, err.c_str());
                ++failed;
                continue;
            }
            const float got = get_f32(it.t);
            const double rel =
                std::fabs((double) got - (double) ref[0]) / std::max(1.0, std::fabs((double) ref[0]));
            const bool ok = (rel <= it.tol);
            std::printf("[train-step]   %-24s ref=%.6f got=%.6f rel=%.3e（门 %.0e）%s\n", it.name,
                        (double) ref[0], (double) got, rel, it.tol, ok ? "OK" : "FAIL");
            ++checked;
            if (!ok) {
                ++failed;
            }
        }
    }

    // ΔW：golden role=delta 逐参数对比。
    // 口径（M3-3；与 Q17 长 T 统计口径同类）：一步 AdamW 的更新 ≈ -lr·sign(g)（零状态起步），
    // 近零梯度元素的 sign 由 F32 累加噪声决定，跨实现必然出现 ±2·lr 的符号翻转（kink）。
    //   - 硬门：|got-ref| > 1e-4 且不满足 kink 条件（两侧 |Δ|≤1.5·lr、反号、|差|≤2.1·lr）→ 失败；
    //   - kink 统计门：每张量 kink ≤ max(2, 0.5%·元素数)（梯度整场低于噪声 1e-6 的张量不设比例上限）；
    //     全局 kink ≤ 0.1%（观测 0.012%，留 8× 余量）。
    const double kink_scale    = 1.5 * (double) opts.lr;
    const double kink_max_diff = 2.1 * (double) opts.lr;
    const double kGradNoise    = 1e-6;
    std::map<std::string, double> g_grad_max;
    {
        for (Tensor* t : g_params.params().items) {
            if (t->grad_acc == nullptr) {
                continue;
            }
            std::vector<float> gv((size_t) tensor_nelements(t));
            tensor_get(t->grad_acc, gv.data(), 0, gv.size() * sizeof(float));
            double m = 0.0;
            for (float v : gv) {
                m = std::max(m, std::fabs((double) v));
            }
            g_grad_max[t->name] = m;
        }
    }
    {
        std::set<std::string> golden_delta_names;
        int64_t delta_checked = 0;
        int64_t delta_failed  = 0;
        int64_t elem_total    = 0;
        int64_t kink_total    = 0;
        int64_t hard_total    = 0;
        for (const GoldenTensor& gt : gm.tensors) {
            if (gt.role != "delta") {
                continue;
            }
            golden_delta_names.insert(gt.name);
            ++delta_checked;
            std::vector<float> ref;
            if (!entry_f32(gm, gt.name, &ref, &err)) {
                std::printf("[train-step]   ΔW %-46s [FAIL] 参考读取失败: %s\n", gt.name.c_str(),
                            err.c_str());
                ++delta_failed;
                continue;
            }
            std::vector<float> got;
            if (g_after.count(gt.name) != 0 && g_before.count(gt.name) != 0) {
                got = make_delta(g_after[gt.name], g_before[gt.name]);
            } else if (d_after.count(gt.name) != 0 && d_before.count(gt.name) != 0) {
                got = make_delta(d_after[gt.name], d_before[gt.name]);
            } else {
                std::printf("[train-step]   ΔW %-46s [FAIL] C++ 无该参数（梯度缺失）\n",
                            gt.name.c_str());
                ++delta_failed;
                continue;
            }
            if (got.size() != ref.size()) {
                std::printf("[train-step]   ΔW %-46s [FAIL] 元素数不符（%zu != %zu）\n",
                            gt.name.c_str(), got.size(), ref.size());
                ++delta_failed;
                continue;
            }
            const int64_t n = (int64_t) ref.size();
            int64_t kink = 0;
            int64_t hard = 0;
            double  max_abs = 0.0;
            for (int64_t i = 0; i < n; ++i) {
                const double d = std::fabs((double) got[(size_t) i] - (double) ref[(size_t) i]);
                max_abs = std::max(max_abs, d);
                if (d <= kDeltaTol) {
                    continue;
                }
                const bool is_kink = (std::fabs((double) got[(size_t) i]) <= kink_scale &&
                                      std::fabs((double) ref[(size_t) i]) <= kink_scale &&
                                      ((got[(size_t) i] < 0.0f) != (ref[(size_t) i] < 0.0f)) &&
                                      d <= kink_max_diff);
                if (is_kink) {
                    ++kink;
                } else {
                    ++hard;
                }
            }
            const int64_t kink_allow = std::max<int64_t>(2, (int64_t) (0.005 * (double) n));
            double grad_max = 0.0;
            {
                const auto it = g_grad_max.find(gt.name);
                if (it != g_grad_max.end()) {
                    grad_max = it->second;
                }
                const auto it2 = d_grad_max.find(gt.name);
                if (it2 != d_grad_max.end()) {
                    grad_max = std::max(grad_max, it2->second);
                }
            }
            const bool ok = (hard == 0) &&
                            (kink <= kink_allow || grad_max <= kGradNoise);
            elem_total += n;
            kink_total += kink;
            hard_total += hard;
            std::printf("[train-step]   ΔW %-46s n=%-9lld max_abs=%.3e kink=%-6lld hard=%-4lld "
                        "gmax=%.2e %s\n",
                        gt.name.c_str(), (long long) n, max_abs, (long long) kink, (long long) hard,
                        grad_max, ok ? "OK" : "FAIL");
            if (!ok) {
                ++delta_failed;
            }
        }
        // C++ 侧有梯度但 golden 无 ΔW → 图结构不一致
        for (const auto& kv : g_after) {
            if (golden_delta_names.count(kv.first) == 0) {
                std::printf("[train-step]   ΔW %-46s [FAIL] golden 缺少（C++ 有梯度）\n",
                            kv.first.c_str());
                ++delta_failed;
            }
        }
        for (const auto& kv : d_after) {
            if (golden_delta_names.count(kv.first) == 0) {
                std::printf("[train-step]   ΔW %-46s [FAIL] golden 缺少（C++ 有梯度）\n",
                            kv.first.c_str());
                ++delta_failed;
            }
        }
        const double kink_frac = (elem_total > 0) ? (double) kink_total / (double) elem_total : 1.0;
        if (kink_frac > 0.001) {
            std::printf("[train-step]   ΔW [FAIL] 全局 kink 比例 %.4f%% > 0.1%%\n", kink_frac * 100.0);
            ++delta_failed;
        }
        checked += delta_checked;
        failed  += delta_failed;
        std::printf("[train-step] ΔW 对拍：%lld/%lld 张量通过；元素 %lld；kink %lld（%.4f%%，门 "
                    "≤0.1%%）；硬误差 %lld（门 0）\n",
                    (long long) (delta_checked - delta_failed), (long long) delta_checked,
                    (long long) elem_total, (long long) kink_total, kink_frac * 100.0,
                    (long long) hard_total);
    }

    const bool ok = (failed == 0 && leak_failed == 0 && checked > 0);
    std::printf("[train-step] 对拍 %s（前向/loss/ΔW 共 %lld 项：%lld 通过；泄漏断言 %s）\n",
                ok ? "通过" : "失败", (long long) checked, (long long) (checked - failed),
                leak_failed == 0 ? "OK" : "FAIL");

    optim_free(optim_g);
    if (optim_d != nullptr) {
        optim_free(optim_d);
    }
    if (d_graph != nullptr) {
        graph_free(d_graph);
    }
    if (buf_d != nullptr) {
        buffer_free(buf_d);
    }
    if (ctx_d != nullptr) {
        context_free(ctx_d);
    }
    graph_free(g_graph);
    buffer_free(buf_g);
    context_free(ctx_g);
    delete wg;
    delete wd;
    return ok ? 0 : 1;
}

// ---------------------------------------------------------------- M5-2 训练循环

int run_train_loop_impl(const std::string& weights_g, const std::string& weights_d,
                        const std::string& golden_dir, Device* dev) {
    if (dev == nullptr) {
        std::printf("[train-loop] 无可用设备\n");
        return 2;
    }
    std::string err;
    GoldenManifest gm;
    if (!model::golden_manifest_load(golden_dir, "train_loop", &gm, &err)) {
        std::printf("[train-loop] %s\n", err.c_str());
        return 1;
    }
    const common::JsonValue* synth_json = gm.params.find("synth");
    const common::JsonValue* disc_json  = gm.params.find("disc");
    if (synth_json == nullptr || disc_json == nullptr) {
        std::printf("[train-loop] manifest.params.synth/disc 缺失（需重新生成 golden）\n");
        return 1;
    }

    TrainStepConfig cfg;
    cfg.frames   = gm.param_int("frames", 0);
    cfg.batch    = gm.param_int("batch", 1);
    cfg.disc_len = gm.param_int("disc_len", 0);
    cfg.lr           = (float) j_num(gm.params, "lr", 1e-4);
    cfg.beta1        = (float) j_num(gm.params, "beta1", 0.8);
    cfg.beta2        = (float) j_num(gm.params, "beta2", 0.99);
    cfg.eps          = (float) j_num(gm.params, "eps", 1e-9);
    cfg.weight_decay = (float) j_num(gm.params, "weight_decay", 1e-2);
    cfg.c_mel        = (float) j_num(gm.params, "c_mel", 45.0);
    cfg.c_kl         = (float) j_num(gm.params, "c_kl", 1.0);
    const int64_t steps = gm.param_int("steps", 10);
    const double tol_first = j_num(gm.params, "tol_first", 1e-4);
    const double tol_later = j_num(gm.params, "tol", 1e-2);
    if (cfg.frames <= 0 || steps <= 0) {
        std::printf("[train-loop] manifest frames/steps 非法\n");
        return 1;
    }

    std::printf("[train-loop] golden=%s 设备=%s T=%lld B=%lld disc_len=%lld steps=%lld "
                "（门：首步 %.0e，后续 %.0e）\n",
                golden_dir.c_str(), dev->props().name.c_str(), (long long) cfg.frames,
                (long long) cfg.batch, (long long) cfg.disc_len, (long long) steps, tol_first,
                tol_later);

    TrainStep* ts = TrainStep::create(weights_g, weights_d, cfg, dev, &err);
    if (ts == nullptr) {
        std::printf("[train-loop] TrainStep 创建失败: %s\n", err.c_str());
        return 1;
    }
    {
        const TrainStepMemory m = ts->memory();
        const ContextMemoryStats s = context_memory_stats(ts->context());
        std::printf("[train-loop] buf_g=%.1f MiB buf_d=%.1f MiB ctx_tensor=%.1f MiB "
                    "persistent=%.1f MiB tensors=%lld G节点=%lld D节点=%lld\n",
                    (double) m.buf_g_bytes / (1024.0 * 1024.0),
                    (double) m.buf_d_bytes / (1024.0 * 1024.0),
                    (double) s.tensor_bytes / (1024.0 * 1024.0),
                    (double) s.persistent_bytes / (1024.0 * 1024.0),
                    (long long) m.ctx_tensor_count, (long long) graph_n_nodes(ts->graph_g()),
                    (long long) graph_n_nodes(ts->graph_d()));
        const TrainStepMemoryBreakdown pb = ts->persistent_breakdown();
        const double mib = 1024.0 * 1024.0;
        std::printf("[train-loop] 持久分解（Q23）：视图根 %.1f（其中参数/叶子 %.1f、普通中间量 "
                    "%.1f）、参数/损失/输出 %.1f、叶子 %.1f、grad_acc %.1f、宿主 %.1f、非持久 "
                    "%.1f MiB；视图 %lld/%lld 张量\n",
                    (double) pb.view_root_bytes / mib, (double) pb.view_root_leaf_bytes / mib,
                    (double) pb.view_root_interm_bytes / mib,
                    (double) pb.param_loss_out_bytes / mib, (double) pb.leaf_bytes / mib,
                    (double) pb.grad_acc_bytes / mib, (double) pb.host_bytes / mib,
                    (double) pb.non_persistent_bytes / mib, (long long) pb.view_count,
                    (long long) pb.tensor_count);
    }

    // 固定输入（所有步共用）
    data::BatchBuffers base;
    auto load_f32 = [&](const std::string& name, std::vector<float>* out) {
        if (!entry_f32(gm, name, out, &err)) {
            std::printf("[train-loop] 读取 %s 失败: %s\n", name.c_str(), err.c_str());
            return false;
        }
        return true;
    };
    auto load_i32 = [&](const std::string& name, std::vector<int32_t>* out) {
        if (!entry_i32(gm, name, out, &err)) {
            std::printf("[train-loop] 读取 %s 失败: %s\n", name.c_str(), err.c_str());
            return false;
        }
        return true;
    };
    if (!load_f32("phone", &base.phone) || !load_i32("pitch", &base.pitch) ||
        !load_f32("x_mask", &base.x_mask) || !load_f32("spec", &base.spec) ||
        !load_f32("y_mask", &base.y_mask) || !load_i32("sid", &base.sid) ||
        !load_f32("attn_valid", &base.attn_valid)) {
        delete ts;
        return 1;
    }

    const char* kLossNames[6] = {"loss_disc", "loss_gen_all", "loss_gen",
                                 "loss_fm",   "loss_mel",     "loss_kl"};

    // 参考 loss（主 golden）与可选混沌探针包络（<golden>/probe，Q24 修订门）：
    // 探针 = Python 参考加与 M3-3 实测 AdamW kink 同量级扰动后同数据重跑；
    // 逐步容差 = max(tol_later, 3 × 探针相对偏差)，首步仍用 tol_first。
    std::vector<std::array<double, 6>> ref_loss((size_t) steps);
    bool ref_ok = true;
    for (int64_t i = 0; i < steps; ++i) {
        const std::string p = "s" + std::to_string(i) + "_";
        for (int k = 0; k < 6; ++k) {
            std::vector<float> ref;
            if (!entry_f32(gm, p + kLossNames[k], &ref, &err) || ref.empty()) {
                std::printf("[train-loop] %s%s 参考读取失败: %s\n", p.c_str(), kLossNames[k],
                            err.c_str());
                ref_ok = false;
                break;
            }
            ref_loss[(size_t) i][(size_t) k] = (double) ref[0];
        }
        if (!ref_ok) {
            break;
        }
    }
    if (!ref_ok) {
        delete ts;
        return 1;
    }
    std::vector<double> step_tol((size_t) steps, tol_later);
    // 优先用 manifest.params.tols（make_train_loop_envelope.py 固化的多探针系综 × scale 包络）
    bool have_envelope = false;
    if (const common::JsonValue* tv = gm.params.find("tols");
        tv != nullptr && tv->is_array() && tv->size() == (size_t) steps) {
        for (int64_t i = 0; i < steps; ++i) {
            step_tol[(size_t) i] = tv->at((size_t) i).as_number(tol_later);
        }
        have_envelope = true;
        std::printf("[train-loop] 容差包络：manifest.params.tols（%lld 步；Q24 修订门）\n",
                    (long long) steps);
    }
    // 回退：<golden>/probe 单探针 ×3
    GoldenManifest gp;
    if (!have_envelope &&
        model::golden_manifest_load(join_path(golden_dir, "probe"), "train_loop", &gp, &err)) {
        for (int64_t i = 0; i < steps; ++i) {
            if (i == 0) {
                step_tol[(size_t) i] = tol_first;
                continue;
            }
            double pmax = 0.0;
            const std::string p = "s" + std::to_string(i) + "_";
            for (int k = 0; k < 6; ++k) {
                std::vector<float> pv;
                if (entry_f32(gp, p + kLossNames[k], &pv, &err) && !pv.empty()) {
                    pmax = std::max(
                        pmax, std::fabs((double) pv[0] - ref_loss[(size_t) i][(size_t) k]) /
                                  std::max(1.0, std::fabs(ref_loss[(size_t) i][(size_t) k])));
                }
            }
            step_tol[(size_t) i] = std::max(tol_later, 3.0 * pmax);
        }
        have_envelope = true;
        std::printf("[train-loop] 容差包络：%s/probe 单探针 ×3（建议用 make_train_loop_envelope.py "
                    "固化多探针包络）\n", golden_dir.c_str());
    }
    if (!have_envelope) {
        for (int64_t i = 0; i < steps; ++i) {
            step_tol[(size_t) i] = (i == 0) ? tol_first : tol_later;
        }
        std::printf("[train-loop] 无探针包络（%s/probe 缺失）→ 后续步用固定门 %.0e；"
                    "生成：python tools/gen_golden_train_loop.py --out <dir>\\probe "
                    "--disc-len <同主 golden> --perturb-frac 0.000125 --perturb-delta 2e-4\n",
                    golden_dir.c_str(), tol_later);
    }

    int64_t checked      = 0;
    int64_t failed_steps = 0;
    int64_t failed_checks = 0;
    for (int64_t i = 0; i < steps; ++i) {
        const std::string p = "s" + std::to_string(i) + "_";
        data::BatchBuffers b = base;
        std::vector<int32_t> ids;
        if (!load_i32(p + "ids_slice", &ids) || !load_f32(p + "q_noise", &b.q_noise) ||
            !load_f32(p + "pitchf_slice", &b.pitchf_slice) ||
            !load_f32(p + "slice_mat", &b.slice_mat) ||
            !load_f32(p + "y_mel", &b.y_mel) || !load_f32(p + "wave", &b.wave) ||
            !load_f32(p + "sine_source_slice", &b.sine_source_slice)) {
            ++failed_steps;
            break;
        }
        if (!ts->set_batch(b, &err)) {
            std::printf("[train-loop] step %lld set_batch 失败: %s\n", (long long) i,
                        err.c_str());
            ++failed_steps;
            break;
        }
        TrainStepLosses L;
        if (!ts->run_step(&L, &err)) {
            std::printf("[train-loop] step %lld run_step 失败: %s\n", (long long) i, err.c_str());
            ++failed_steps;
            break;
        }
        const float got[6] = {L.loss_disc, L.loss_gen_all, L.loss_gen,
                              L.loss_fm,   L.loss_mel,     L.loss_kl};
        const double tol = step_tol[(size_t) i];
        double step_rel = 0.0;
        int step_fail = 0;
        char line[768];
        int off = std::snprintf(line, sizeof(line), "[train-loop] step %2lld (ids=%d, tol=%.1e)",
                                (long long) (i + 1), ids.empty() ? -1 : ids[0], tol);
        for (int k = 0; k < 6; ++k) {
            const double ref = ref_loss[(size_t) i][(size_t) k];
            const double rel = std::fabs((double) got[k] - ref) / std::max(1.0, std::fabs(ref));
            step_rel = std::max(step_rel, rel);
            ++checked;
            if (rel > tol) {
                ++step_fail;
            }
            off += std::snprintf(line + off, sizeof(line) - (size_t) off, " %s=%.6f(rel=%.2e)",
                                 kLossNames[k], (double) got[k], rel);
        }
        if (step_fail > 0) {
            ++failed_steps;
            failed_checks += step_fail;
            std::printf("%s [FAIL]\n", line);
        } else {
            std::printf("%s max_rel=%.2e OK\n", line, step_rel);
        }
    }

    std::printf("[train-loop] 轨迹对拍 %s（loss 项 %lld/%lld 通过；steps=%lld、T=%lld、"
                "disc_len=%lld）\n",
                failed_steps == 0 ? "通过" : "失败", (long long) (checked - failed_checks),
                (long long) checked, (long long) ts->step_count(), (long long) cfg.frames,
                (long long) cfg.disc_len);
    delete ts;
    return failed_steps == 0 ? 0 : 1;
}

} // namespace

int run_train_step_selftest(const std::string& weights_g, const std::string& weights_d,
                            const std::string& golden_dir, Device* dev) {
    return run_impl(weights_g, weights_d, golden_dir, dev);
}

int run_train_loop_selftest(const std::string& weights_g, const std::string& weights_d,
                            const std::string& golden_dir, Device* dev) {
    return run_train_loop_impl(weights_g, weights_d, golden_dir, dev);
}

} // namespace train
} // namespace rvc_train
