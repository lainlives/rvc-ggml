// RVC-Train-GGML - M3-3 dual-graph G/D + one AdamW step comparison-test implementation (src/train/train_step_selftest.cpp)
//
// Dual-graph protocol (docs/handover.md §4.7 C; docs/architecture.md §5.1):
//   G graph (ctx_g): synth slice forward (include_full=false) -> D (frozen, newly built nodes)
//                  → LSGAN(G) + feature matching + 45·mel L1 + 1·KL
//   D graph (ctx_d): D(real)+D(fake) (separate ctx; reusing G graph nodes is forbidden) -> LSGAN(D)
//   Step order (aligned with official 81eed5e train.py:425-527): D step first (using pre-update y_hat.detach()),
//   copy the updated D weights back into the G graph -> G graph recomputes (frozen D nodes read the updated D) -> G step.
// Memory: each of the two ctxs calls buffer_alloc_graph_tensors (graph-level liveness reuse; the Q18 mitigation is in place).
//   From M5-0 the training loop will switch to **single-ctx shared-parameter dual graphs** (see `docs/handover.md` §4.9 C); this selftest
//   keeps the verified M3-3 form unchanged as the golden baseline.
// Leak assertions: no grad_acc on D parameters after the G graph backward expansion; G parameters bit-unchanged after the D step; D parameters bit-unchanged after the G step.
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
#include "common/platform.h"
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

constexpr double kDeltaTol = 1e-4;  // ΔW difference noise floor (hard-gate/kink criteria follow the M3-3 statistical convention, not a relative gate)
constexpr double kLossTol  = 1e-4;  // loss relative gate
constexpr double kFwdTol   = 1e-4;  // forward intermediate relative gate

// ---------------------------------------------------------------- basic helpers

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
            *err = "cannot open file: " + path;
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
            *err = "read failed: " + path;
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

// ---------------------------------------------------------------- JSON parameter reading

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

// ---------------------------------------------------------------- golden file reading

bool entry_bytes(const GoldenManifest& gm, const std::string& name, std::vector<uint8_t>* out,
                 std::string* err) {
    const GoldenTensor* gt = gm.find_tensor(name);
    if (gt == nullptr) {
        if (err != nullptr) {
            *err = "golden missing entry: " + name;
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
            *err = "file size is not a multiple of F32: " + name;
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
            *err = "file size is not a multiple of I32: " + name;
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
        std::snprintf(buf, sizeof(buf), "input file size mismatch: %s (%zu != %zu)", name.c_str(),
                      bytes.size(), expect);
        if (err != nullptr) {
            *err = buf;
        }
        return false;
    }
    tensor_set(t, bytes.data(), 0, bytes.size());
    return true;
}

// ---------------------------------------------------------------- parameter snapshots / ΔW

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

// ---------------------------------------------------------------- config assembly

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

// ---------------------------------------------------------------- main flow

int run_impl(const std::string& weights_g, const std::string& weights_d,
             const std::string& golden_dir, Device* dev) {
    if (dev == nullptr) {
        std::printf("[train-step] no usable device\n");
        return 2;
    }
    std::printf("[train-step] golden=%s\n", golden_dir.c_str());

    // ---------------- open weights and golden ----------------
    std::string err;
    TrainWeights* wg = TrainWeights::open(weights_g, &err);
    if (wg == nullptr) {
        std::printf("[train-step] failed to open G weights: %s\n", err.c_str());
        return 1;
    }
    TrainWeights* wd = TrainWeights::open(weights_d, &err);
    if (wd == nullptr) {
        std::printf("[train-step] failed to open D weights: %s\n", err.c_str());
        delete wg;
        return 1;
    }
    TrainMeta meta_g;
    TrainMeta meta_d;
    wg->read_meta(&meta_g);
    wd->read_meta(&meta_d);
    const ModelConfig cfg = ModelConfig::from_meta(meta_g);
    if (!cfg.validate(&err)) {
        std::printf("[train-step] config validation failed: %s\n", err.c_str());
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
        std::printf("[train-step] manifest.params.synth/disc missing (regenerate the golden)\n");
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
        std::printf("[train-step] currently only B=1 is supported (golden batch=%lld)\n", (long long) B);
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
    std::printf("[train-step] hyperparameters lr=%.1e betas=(%.2f,%.2f) eps=%.1e wd=%.3f c_mel=%.1f "
                "c_kl=%.1f disc_len=%lld\n",
                (double) opts.lr, (double) opts.beta1, (double) opts.beta2, (double) opts.eps,
                (double) opts.weight_decay, (double) c_mel, (double) c_kl, (long long) disc_len);

    // ---------------- ctx_g: model assembly + G graph ----------------
    Context* ctx_g = context_new(1u << 22);
    if (ctx_g == nullptr) {
        std::printf("[train-step] ctx_g create failed\n");
        delete wg;
        delete wd;
        return 1;
    }
    ModelParams g_params;
    ModelParams d_params_g;
    if (!g_params.build_all(ctx_g, *wg, &err) || !d_params_g.build_all(ctx_g, *wd, &err)) {
        std::printf("[train-step] parameter building failed: %s\n", err.c_str());
        context_free(ctx_g);
        delete wg;
        delete wd;
        return 1;
    }
    MpdModule mpd_g;
    if (!model::mpd_load(d_params_g, dcfg, &mpd_g, &err)) {
        std::printf("[train-step] D assembly failed: %s\n", err.c_str());
        context_free(ctx_g);
        delete wg;
        delete wd;
        return 1;
    }
    SynthModules synth;
    if (!model::synth_load(g_params, scfg, &synth, &err)) {
        std::printf("[train-step] G assembly failed: %s\n", err.c_str());
        context_free(ctx_g);
        delete wg;
        delete wd;
        return 1;
    }

    // In-graph inputs (create only the leaves the graph actually consumes; lengths/pitchf/ids_slice/sine_noise_slice are read on the host)
    static const char* kGraphInputs[] = {"phone", "pitch", "x_mask", "spec", "y_mask", "sid",
                                         "q_noise", "wave", "y_mel"};
    std::map<std::string, Tensor*> inputs;
    for (const char* name : kGraphInputs) {
        const GoldenTensor* gt = gm.find_tensor(name);
        if (gt == nullptr || !gt->is_input()) {
            std::printf("[train-step] golden missing input %s\n", name);
            context_free(ctx_g);
            delete wg;
            delete wd;
            return 1;
        }
        const Type type = (gt->dtype == "i32") ? TYPE_I32 : TYPE_F32;
        Tensor* t = new_tensor_nd(ctx_g, type, gt->nd, gt->ne.data());
        if (t == nullptr) {
            std::printf("[train-step] failed to create input: %s\n", name);
            context_free(ctx_g);
            delete wg;
            delete wd;
            return 1;
        }
        tensor_set_name(t, "%s", name);
        inputs[name] = t;
    }

    // Leaves (filled by prepare after allocation)
    Tensor* attn_valid = new_tensor_4d(ctx_g, TYPE_F32, frames, frames, 1, B);
    Tensor* flip_perm  = new_tensor_2d(ctx_g, TYPE_F32, scfg.flow.channels, scfg.flow.channels);
    Tensor* slice_mat  = new_tensor_3d(ctx_g, TYPE_F32, frames, seg, B);
    Tensor* sine_slice = new_tensor_3d(ctx_g, TYPE_F32, 1, total_slice, B);
    if (attn_valid == nullptr || flip_perm == nullptr || slice_mat == nullptr ||
        sine_slice == nullptr) {
        std::printf("[train-step] failed to create leaves\n");
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
        std::printf("[train-step] synth_forward failed\n");
        context_free(ctx_g);
        delete wg;
        delete wd;
        return 1;
    }
    // D input = first disc_len samples of y_hat (test scale; the view is differentiable, see the M3-0 plan disc L=2048)
    Tensor* y_hat_d = view_3d(ctx_g, out.o_slice, disc_len, 1, B, 0);
    if (y_hat_d == nullptr) {
        std::printf("[train-step] failed to create the y_hat_d view\n");
        context_free(ctx_g);
        delete wg;
        delete wd;
        return 1;
    }
    tensor_set_name(y_hat_d, "train_y_hat_d");

    MpdForward f_g = model::mpd_forward(ctx_g, mpd_g, inputs["wave"], y_hat_d);
    if (f_g.r.empty() || f_g.r.size() != f_g.g.size()) {
        std::printf("[train-step] D forward in the G graph failed\n");
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
        std::printf("[train-step] invalid STFT/mel config: %s\n", err.c_str());
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
        std::printf("[train-step] G loss graph build failed\n");
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
    // Mark components as OUTPUT so they stay readable under graph-level allocation (comparison reads only; not part of backward seeding)
    loss_gen->flags |= TENSOR_FLAG_OUTPUT;
    loss_fm->flags |= TENSOR_FLAG_OUTPUT;
    loss_mel->flags |= TENSOR_FLAG_OUTPUT;
    loss_kl->flags |= TENSOR_FLAG_OUTPUT;
    // FIX-002 (Q23): views/view roots still read by the host after compute must also be flagged OUTPUT
    // (:839 reads y_hat_d after warm-up; :1000-1014 forward comparison reads o_slice)
    out.o_slice->flags |= TENSOR_FLAG_OUTPUT;
    y_hat_d->flags |= TENSOR_FLAG_OUTPUT;

    for (Tensor* t : g_params.params().items) {
        tensor_set_param(t);
    }
    Graph* g_graph = graph_new(ctx_g);
    if (g_graph == nullptr) {
        std::printf("[train-step] G graph create failed\n");
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
            std::printf("[train-step] [FAIL] frozen-side leak: D parameter %s has grad_acc after G graph backward expansion\n",
                        t->name);
            ++leak_failed;
        }
    }
    if (leak_failed == 0) {
        std::printf("[train-step] no leak (A): G graph freezes D (%lld parameters with no grad_acc) OK\n",
                    (long long) d_params_g.count());
    }

    std::printf("[train-step] backend = %s; allocating G graph (graph-level liveness)...\n",
                dev->props().name.c_str());
    Buffer* buf_g = buffer_alloc_graph_tensors(g_graph, dev->default_buffer_type());
    if (buf_g == nullptr) {
        std::printf("[train-step] G graph allocation failed (diagnose with RVCT_DEBUG_ALLOC=1; Q18 mitigation: reduce T / split ctx)\n");
        graph_free(g_graph);
        context_free(ctx_g);
        delete wg;
        delete wd;
        return 1;
    }

    // ---------------- ctx_d: D graph ----------------
    Context* ctx_d = context_new(1u << 20);
    ModelParams d_params_d;
    if (ctx_d == nullptr || !d_params_d.build_all(ctx_d, *wd, &err)) {
        std::printf("[train-step] ctx_d/parameter building failed: %s\n", err.c_str());
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
        std::printf("[train-step] D graph assembly failed: %s\n", err.c_str());
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
        std::printf("[train-step] failed to create D graph leaves\n");
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
        std::printf("[train-step] D graph forward failed\n");
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
        std::printf("[train-step] D loss graph build failed\n");
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
        std::printf("[train-step] D graph create failed\n");
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
        std::printf("[train-step] D graph allocation failed\n");
        graph_free(d_graph);
        context_free(ctx_d);
        graph_free(g_graph);
        buffer_free(buf_g);
        context_free(ctx_g);
        delete wg;
        delete wd;
        return 1;
    }
    std::printf("[train-step] ctx_g graph buffer %.1f MiB; ctx_d graph buffer %.1f MiB\n",
                (double) buf_g->size() / (1024.0 * 1024.0),
                (double) buf_d->size() / (1024.0 * 1024.0));

    // ---------------- weight loading ----------------
    for (Tensor* t : g_params.params().items) {
        if (!wg->load(t, t->name, &err)) {
            std::printf("[train-step] G parameter loading failed: %s\n", err.c_str());
            return 1;
        }
    }
    for (Tensor* t : d_params_g.params().items) {
        if (!wd->load(t, t->name, &err)) {
            std::printf("[train-step] G graph D parameter loading failed: %s\n", err.c_str());
            return 1;
        }
    }
    for (Tensor* t : d_params_d.params().items) {
        if (!wd->load(t, t->name, &err)) {
            std::printf("[train-step] D graph D parameter loading failed: %s\n", err.c_str());
            return 1;
        }
    }

    // ---------------- inject inputs + prepare ----------------
    {
        std::vector<uint8_t> bytes;
        for (const auto& kv : inputs) {
            if (!fill_tensor_from_entry(kv.second, gm, kv.first, &err)) {
                std::printf("[train-step] input injection failed: %s\n", err.c_str());
                return 1;
            }
        }
        // wave -> D graph leaf (same data)
        std::vector<float> wv((size_t) disc_len * (size_t) B);
        tensor_get(inputs["wave"], wv.data(), 0, wv.size() * sizeof(float));
        tensor_set(wave_d, wv.data(), 0, wv.size() * sizeof(float));

        if (!model::flow_fill_flip_perm(flip_perm, &err)) {
            std::printf("[train-step] flip permutation matrix fill failed: %s\n", err.c_str());
            return 1;
        }
        // attn_valid = x_mask outer product
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
        // slice_mat / pitchf_slice (host reads golden)
        std::vector<int32_t> idv;
        std::vector<float> pfall;
        std::vector<float> noise;
        if (!entry_i32(gm, "ids_slice", &idv, &err) || !entry_f32(gm, "pitchf", &pfall, &err) ||
            !entry_f32(gm, "sine_noise_slice", &noise, &err)) {
            std::printf("[train-step] host input read failed: %s\n", err.c_str());
            return 1;
        }
        if ((int64_t) idv.size() != B || (int64_t) pfall.size() != frames * B ||
            (int64_t) noise.size() != total_slice * B) {
            std::printf("[train-step] host input size mismatch (ids=%zu pitchf=%zu noise=%zu)\n",
                        idv.size(), pfall.size(), noise.size());
            return 1;
        }
        std::vector<float> sm((size_t) (frames * seg * B), 0.0f);
        std::vector<float> pfs((size_t) (seg * B), 0.0f);
        for (int64_t b = 0; b < B; ++b) {
            const int64_t id = idv[(size_t) b];
            if (id < 0 || id + seg > frames) {
                std::printf("[train-step] ids_slice out of range: %lld\n", (long long) id);
                return 1;
            }
            for (int64_t s = 0; s < seg; ++s) {
                sm[(size_t) (id + s + frames * s + frames * seg * b)] = 1.0f;
                pfs[(size_t) (s + seg * b)] = pfall[(size_t) (id + s + frames * b)];
            }
        }
        tensor_set(slice_mat, sm.data(), 0, sm.size() * sizeof(float));

        // SineGen (legacy cumsum_shift method; noise = golden dumped to disk)
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
                std::printf("[train-step] SineGen output length mismatch\n");
                return 1;
            }
            std::memcpy(ss.data() + (size_t) b * total_slice, s.data(),
                        sizeof(float) * (size_t) total_slice);
        }
        tensor_set(sine_slice, ss.data(), 0, ss.size() * sizeof(float));

        // STFT/mel constants
        stft_mel_fill_dft_kernel(&smconsts, smcfg);
        stft_mel_fill_mel_basis(&smconsts, smcfg);
    }

    // ---------------- optimizer ----------------
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
    std::printf("[train-step] optimizer parameters: G %zu/%lld, D %zu/%lld\n", g_opt.size(),
                (long long) g_params.count(), d_opt.size(), (long long) d_params_d.count());
    Optimizer* optim_g = optim_adamw_new(ctx_g, g_opt.data(), (int64_t) g_opt.size(), opts);
    Optimizer* optim_d = optim_adamw_new(ctx_d, d_opt.data(), (int64_t) d_opt.size(), opts);

    // ---------------- dual-graph training step ----------------
    // 1) G graph warm-up: first obtain y_hat (pre-update; the pre-update D gradients computed along the way are discarded)
    graph_reset(g_graph);
    if (!dev->graph_compute(g_graph)) {
        std::printf("[train-step] G graph warm-up compute failed\n");
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

    // 2) D step (fake = the detached equivalent of y_hat)
    graph_reset(d_graph);
    if (!dev->graph_compute(d_graph)) {
        std::printf("[train-step] D graph compute failed\n");
        return 1;
    }
    std::map<std::string, std::vector<float>> d_before;
    read_params(d_params_d, &d_before);
    optim_step(optim_d);
    std::map<std::string, std::vector<float>> d_after;
    read_params(d_params_d, &d_after);

    // D-side snapshots (ctx_d is released afterwards; its tensors/gradients must not be accessed again)
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

    // Assertion (B): G parameters bit-unchanged after the D step
    {
        std::map<std::string, std::vector<float>> now;
        read_params(g_params, &now);
        for (const auto& kv : g_snap) {
            if (!values_equal(kv.second, now[kv.first])) {
                std::printf("[train-step] [FAIL] D step modified G parameter: %s\n", kv.first.c_str());
                ++leak_failed;
            }
        }
        if (leak_failed == 0) {
            std::printf("[train-step] no leak (B): G parameters bit-unchanged after the D step OK\n");
        }
    }

    // 3) Copy the updated D weights back into the G graph (the G graph's D nodes read them on each compute)
    for (Tensor* t : d_params_g.params().items) {
        Tensor* s = d_params_d.find(t->name);
        if (s == nullptr) {
            std::printf("[train-step] D parameter name mismatch: %s\n", t->name);
            return 1;
        }
        std::vector<uint8_t> bytes((size_t) tensor_nelements(t) * tensor_element_size(t));
        tensor_get(s, bytes.data(), 0, bytes.size());
        tensor_set(t, bytes.data(), 0, bytes.size());
    }
    std::map<std::string, std::vector<float>> d_snap2;
    read_params(d_params_g, &d_snap2);

    // Release ctx_d (the D step is done; ΔW/gradients/losses are all snapshotted to the host) -> peak VRAM drops (Q18 mitigation: split ctx)
    optim_free(optim_d);
    buffer_free(buf_d);
    graph_free(d_graph);
    context_free(ctx_d);
    optim_d = nullptr;
    buf_d   = nullptr;
    d_graph = nullptr;
    ctx_d   = nullptr;

    // 4) G step (frozen D nodes read the updated D; graph_reset clears the warm-up gradients)
    graph_reset(g_graph);
    if (!dev->graph_compute(g_graph)) {
        std::printf("[train-step] G graph (final) compute failed\n");
        return 1;
    }
    std::map<std::string, std::vector<float>> g_before;
    read_params(g_params, &g_before);
    optim_step(optim_g);
    std::map<std::string, std::vector<float>> g_after;
    read_params(g_params, &g_after);

    // Optional diagnostic: RVCT_DUMP_TRAINSTEP=DIR dumps the C++ ΔW and grad_acc to disk (for element-wise localization)
    {
        const std::string base = platform::env_get("RVCT_DUMP_TRAINSTEP");
        if (!base.empty()) {
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
            std::printf("[train-step] diagnostic dump: %s (g_delta/g_grad/d_delta)\n", base.c_str());
        }
    }

    // Assertion (C): D parameters (in ctx_g) bit-unchanged after the G step
    {
        std::map<std::string, std::vector<float>> now;
        read_params(d_params_g, &now);
        for (const auto& kv : d_snap2) {
            if (!values_equal(kv.second, now[kv.first])) {
                std::printf("[train-step] [FAIL] G step modified D parameter: %s\n", kv.first.c_str());
                ++leak_failed;
            }
        }
        if (leak_failed == 0) {
            std::printf("[train-step] no leak (C): D parameters bit-unchanged after the G step OK\n");
        }
    }

    // ---------------- comparison: ΔW / loss / forward ----------------
    const auto get_f32 = [](Tensor* t) {
        float v = 0.0f;
        tensor_get(t, &v, 0, sizeof(v));
        return v;
    };
    int64_t checked = 0;
    int64_t failed  = 0;

    // Forward intermediates: y_hat (o_slice) and y_hat_d
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
                std::printf("[train-step]   %-24s [FAIL] reference read failed: %s\n", it.name, err.c_str());
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

    // loss (loss_disc uses the snapshot taken before ctx_d was released; the rest come from the G graph)
    {
        // loss_disc snapshot comparison
        {
            std::vector<float> ref;
            if (!entry_f32(gm, "loss_disc", &ref, &err) || ref.empty()) {
                std::printf("[train-step]   %-24s [FAIL] reference read failed: %s\n", "loss_disc", err.c_str());
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
        // loss_gen: LSGAN(G)=Σ mean((1-dg)²) near dg≈1; kink elements of the D step (each differing by 2·lr)
        // are amplified quadratically (measured 2.8e-4 on Vulkan, 2.6e-5 on CPU) -> relaxed to 5e-4; the rest stay strict at 1e-4.
        const LossItem items[] = {{"loss_gen_all", loss_gen_all, kLossTol},
                                  {"loss_gen", loss_gen, 5e-4},
                                  {"loss_fm", loss_fm, kLossTol},
                                  {"loss_mel", loss_mel, kLossTol},
                                  {"loss_kl", loss_kl, kLossTol}};
        for (const LossItem& it : items) {
            std::vector<float> ref;
            if (!entry_f32(gm, it.name, &ref, &err) || ref.size() < 1) {
                std::printf("[train-step]   %-24s [FAIL] reference read failed: %s\n", it.name, err.c_str());
                ++failed;
                continue;
            }
            const float got = get_f32(it.t);
            const double rel =
                std::fabs((double) got - (double) ref[0]) / std::max(1.0, std::fabs((double) ref[0]));
            const bool ok = (rel <= it.tol);
            std::printf("[train-step]   %-24s ref=%.6f got=%.6f rel=%.3e (gate %.0e) %s\n", it.name,
                        (double) ref[0], (double) got, rel, it.tol, ok ? "OK" : "FAIL");
            ++checked;
            if (!ok) {
                ++failed;
            }
        }
    }

    // ΔW: per-parameter comparison against golden role=delta.
    // Convention (M3-3; same class as the Q17 long-T statistical convention): one AdamW update ≈ -lr·sign(g) (zero state at start),
    // the sign of near-zero-gradient elements is determined by F32 accumulation noise, so a ±2·lr sign flip across implementations is inevitable (kink).
    //   - Hard gate: |got-ref| > 1e-4 and the kink condition does not hold (both sides |Δ|≤1.5·lr, opposite signs, |diff|≤2.1·lr) -> fail;
    //   - Kink statistical gate: per tensor kink ≤ max(2, 0.5%·element count) (no ratio cap for tensors whose entire gradient field is below the 1e-6 noise floor);
    //     global kink ≤ 0.1% (observed 0.012%, an 8× margin).
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
                std::printf("[train-step]   ΔW %-46s [FAIL] reference read failed: %s\n", gt.name.c_str(),
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
                std::printf("[train-step]   ΔW %-46s [FAIL] C++ lacks this parameter (gradient missing)\n",
                            gt.name.c_str());
                ++delta_failed;
                continue;
            }
            if (got.size() != ref.size()) {
                std::printf("[train-step]   ΔW %-46s [FAIL] element count mismatch (%zu != %zu)\n",
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
        // C++ has gradients but golden has no ΔW -> graph structure mismatch
        for (const auto& kv : g_after) {
            if (golden_delta_names.count(kv.first) == 0) {
                std::printf("[train-step]   ΔW %-46s [FAIL] golden missing (C++ has gradients)\n",
                            kv.first.c_str());
                ++delta_failed;
            }
        }
        for (const auto& kv : d_after) {
            if (golden_delta_names.count(kv.first) == 0) {
                std::printf("[train-step]   ΔW %-46s [FAIL] golden missing (C++ has gradients)\n",
                            kv.first.c_str());
                ++delta_failed;
            }
        }
        const double kink_frac = (elem_total > 0) ? (double) kink_total / (double) elem_total : 1.0;
        if (kink_frac > 0.001) {
            std::printf("[train-step]   ΔW [FAIL] global kink fraction %.4f%% > 0.1%%\n", kink_frac * 100.0);
            ++delta_failed;
        }
        checked += delta_checked;
        failed  += delta_failed;
        std::printf("[train-step] ΔW comparison: %lld/%lld tensors passed; elements %lld; kink %lld (%.4f%%, gate "
                    "≤0.1%%); hard errors %lld (gate 0)\n",
                    (long long) (delta_checked - delta_failed), (long long) delta_checked,
                    (long long) elem_total, (long long) kink_total, kink_frac * 100.0,
                    (long long) hard_total);
    }

    const bool ok = (failed == 0 && leak_failed == 0 && checked > 0);
    std::printf("[train-step] comparison %s (forward/loss/ΔW %lld items: %lld passed; leak assertions %s)\n",
                ok ? "PASSED" : "FAILED", (long long) checked, (long long) (checked - failed),
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

// ---------------------------------------------------------------- M5-2 training loop

int run_train_loop_impl(const std::string& weights_g, const std::string& weights_d,
                        const std::string& golden_dir, Device* dev) {
    if (dev == nullptr) {
        std::printf("[train-loop] no usable device\n");
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
        std::printf("[train-loop] manifest.params.synth/disc missing (regenerate the golden)\n");
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
        std::printf("[train-loop] invalid manifest frames/steps\n");
        return 1;
    }

    std::printf("[train-loop] golden=%s device=%s T=%lld B=%lld disc_len=%lld steps=%lld "
                "(gate: first step %.0e, later steps %.0e)\n",
                golden_dir.c_str(), dev->props().name.c_str(), (long long) cfg.frames,
                (long long) cfg.batch, (long long) cfg.disc_len, (long long) steps, tol_first,
                tol_later);

    TrainStep* ts = TrainStep::create(weights_g, weights_d, cfg, dev, &err);
    if (ts == nullptr) {
        std::printf("[train-loop] TrainStep create failed: %s\n", err.c_str());
        return 1;
    }
    {
        const TrainStepMemory m = ts->memory();
        const ContextMemoryStats s = context_memory_stats(ts->context());
        std::printf("[train-loop] buf_g=%.1f MiB buf_d=%.1f MiB ctx_tensor=%.1f MiB "
                    "persistent=%.1f MiB tensors=%lld G nodes=%lld D nodes=%lld\n",
                    (double) m.buf_g_bytes / (1024.0 * 1024.0),
                    (double) m.buf_d_bytes / (1024.0 * 1024.0),
                    (double) s.tensor_bytes / (1024.0 * 1024.0),
                    (double) s.persistent_bytes / (1024.0 * 1024.0),
                    (long long) m.ctx_tensor_count, (long long) graph_n_nodes(ts->graph_g()),
                    (long long) graph_n_nodes(ts->graph_d()));
        const TrainStepMemoryBreakdown pb = ts->persistent_breakdown();
        const double mib = 1024.0 * 1024.0;
        std::printf("[train-loop] persistent breakdown (Q23): view roots %.1f (of which params/leaves %.1f, ordinary intermediates "
                    "%.1f), param/loss/output %.1f, leaves %.1f, grad_acc %.1f, hosts %.1f, non-persistent "
                    "%.1f MiB; views %lld/%lld tensors\n",
                    (double) pb.view_root_bytes / mib, (double) pb.view_root_leaf_bytes / mib,
                    (double) pb.view_root_interm_bytes / mib,
                    (double) pb.param_loss_out_bytes / mib, (double) pb.leaf_bytes / mib,
                    (double) pb.grad_acc_bytes / mib, (double) pb.host_bytes / mib,
                    (double) pb.non_persistent_bytes / mib, (long long) pb.view_count,
                    (long long) pb.tensor_count);
    }

    // Fixed inputs (shared by all steps)
    data::BatchBuffers base;
    auto load_f32 = [&](const std::string& name, std::vector<float>* out) {
        if (!entry_f32(gm, name, out, &err)) {
            std::printf("[train-loop] failed to read %s: %s\n", name.c_str(), err.c_str());
            return false;
        }
        return true;
    };
    auto load_i32 = [&](const std::string& name, std::vector<int32_t>* out) {
        if (!entry_i32(gm, name, out, &err)) {
            std::printf("[train-loop] failed to read %s: %s\n", name.c_str(), err.c_str());
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

    // Reference losses (main golden) and optional chaotic-probe envelope (<golden>/probe, Q24 revised gate):
    // the probe = the Python reference rerun on the same data after perturbations at the same magnitude as the M3-3 measured AdamW kink;
    // per-step tolerance = max(tol_later, 3 × probe relative deviation); the first step still uses tol_first.
    std::vector<std::array<double, 6>> ref_loss((size_t) steps);
    bool ref_ok = true;
    for (int64_t i = 0; i < steps; ++i) {
        const std::string p = "s" + std::to_string(i) + "_";
        for (int k = 0; k < 6; ++k) {
            std::vector<float> ref;
            if (!entry_f32(gm, p + kLossNames[k], &ref, &err) || ref.empty()) {
                std::printf("[train-loop] %s%s reference read failed: %s\n", p.c_str(), kLossNames[k],
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
    // Prefer manifest.params.tols (multi-probe ensemble × scale envelope pinned by make_train_loop_envelope.py)
    bool have_envelope = false;
    if (const common::JsonValue* tv = gm.params.find("tols");
        tv != nullptr && tv->is_array() && tv->size() == (size_t) steps) {
        for (int64_t i = 0; i < steps; ++i) {
            step_tol[(size_t) i] = tv->at((size_t) i).as_number(tol_later);
        }
        have_envelope = true;
        std::printf("[train-loop] tolerance envelope: manifest.params.tols (%lld steps; Q24 revised gate)\n",
                    (long long) steps);
    }
    // Fallback: single <golden>/probe probe ×3
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
        std::printf("[train-loop] tolerance envelope: single %s/probe probe ×3 (recommend pinning a multi-probe "
                    "envelope with make_train_loop_envelope.py)\n", golden_dir.c_str());
    }
    if (!have_envelope) {
        for (int64_t i = 0; i < steps; ++i) {
            step_tol[(size_t) i] = (i == 0) ? tol_first : tol_later;
        }
        std::printf("[train-loop] no probe envelope (%s/probe missing) -> later steps use the fixed gate %.0e;"
                    "generate: python tools/gen_golden_train_loop.py --out <dir>/probe "
                    "--disc-len <same as main golden> --perturb-frac 0.000125 --perturb-delta 2e-4\n",
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
            std::printf("[train-loop] step %lld set_batch failed: %s\n", (long long) i,
                        err.c_str());
            ++failed_steps;
            break;
        }
        TrainStepLosses L;
        if (!ts->run_step(&L, &err)) {
            std::printf("[train-loop] step %lld run_step failed: %s\n", (long long) i, err.c_str());
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

    std::printf("[train-loop] trajectory comparison %s (loss items %lld/%lld passed; steps=%lld, T=%lld,"
                "disc_len=%lld)\n",
                failed_steps == 0 ? "PASSED" : "FAILED", (long long) (checked - failed_checks),
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
