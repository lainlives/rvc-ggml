// RVC-Train-GGML - M3-2 loss/frontend golden comparison-test implementation (src/train/loss_selftest.cpp)
//
// Structure mirrors the M3-1 backward comparison test (src/model/selftest.cpp::run_single_backward):
//   read manifest.loss.<module>.json -> build input leaves (F32 flagged trainable) -> build graph
//   (stft/mel: output + fixed cotangent VJP; loss modules: scalar loss with set_loss directly)
//   -> graph_build_backward_expand (before allocation) -> allocate -> inject -> fill constants -> compute
//   -> compare forward/loss/gradient item by item.
#include "train/loss_selftest.h"

#include "common/json.h"
#include "common/platform.h"
#include "model/selftest.h"
#include "train/losses.h"
#include "train/stft_mel.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace rvc_train {
namespace train {

using rvc_train::model::GoldenManifest;
using rvc_train::model::GoldenTensor;

namespace {

constexpr double kFwdTol  = 1e-4;
constexpr double kLossTol = 1e-4;
constexpr double kBackTol = 1e-3;
constexpr double kBackRmsTol = 2e-3;

// ---------------------------------------------------------------- helpers

std::string join_path(const std::string& dir, const std::string& file) {
    std::string d = dir;
    while (!d.empty() && (d.back() == '/' || d.back() == '\\')) {
        d.pop_back();
    }
    return d + "/" + file;
}

bool read_file_bytes(const std::string& path, std::vector<uint8_t>* out, std::string* err) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        if (err != nullptr) {
            *err = "cannot open file: " + path;
        }
        return false;
    }
    out->assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    f.close();
    return true;
}

std::string ne_string(const GoldenTensor& t) {
    std::string s = "[";
    for (int i = 0; i < t.nd; ++i) {
        if (i != 0) {
            s += ",";
        }
        s += std::to_string(t.ne[(size_t) i]);
    }
    s += "]";
    return s;
}

struct CompareStat {
    double max_abs = 0.0;
    double max_ref = 0.0;
    double rel = 0.0;
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

bool tensor_matches_ne(const traincpp::Tensor* t, const GoldenTensor& gt) {
    for (int i = 0; i < 4; ++i) {
        if (t->ne[i] != gt.ne[(size_t) i]) {
            return false;
        }
    }
    return true;
}

bool fill_tensor_from_file(traincpp::Tensor* t, const std::string& dir, const GoldenTensor& gt,
                           std::string* err) {
    std::vector<uint8_t> bytes;
    if (!read_file_bytes(join_path(dir, gt.file), &bytes, err)) {
        return false;
    }
    const size_t expect = (size_t) traincpp::tensor_nelements(t) * traincpp::tensor_element_size(t);
    if (bytes.size() != expect) {
        char buf[256];
        std::snprintf(buf, sizeof(buf), "file size mismatch: %s (%zu != %zu)", gt.file.c_str(),
                      bytes.size(), expect);
        if (err != nullptr) {
            *err = buf;
        }
        return false;
    }
    traincpp::tensor_set(t, bytes.data(), 0, bytes.size());
    return true;
}

bool set_inputs(const GoldenManifest& gm, const std::map<std::string, traincpp::Tensor*>& inputs,
                std::string* err) {
    for (const GoldenTensor& gt : gm.tensors) {
        if (gt.role != "input") {
            continue;
        }
        auto it = inputs.find(gt.name);
        if (it == inputs.end() || it->second == nullptr) {
            if (err != nullptr) {
                *err = "input tensor not created: " + gt.name;
            }
            return false;
        }
        if (!fill_tensor_from_file(it->second, gm.dir, gt, err)) {
            return false;
        }
    }
    return true;
}

float tensor_get_f32_at(const traincpp::Tensor* t, int64_t idx) {
    float v = 0.0f;
    traincpp::tensor_get(t, &v, (size_t) idx * sizeof(float), sizeof(float));
    return v;
}

bool is_no_grad(const GoldenManifest& gm, const std::string& name) {
    const common::JsonValue* v = gm.params.find("no_grad");
    if (v == nullptr || !v->is_array()) {
        return false;
    }
    for (size_t i = 0; i < v->size(); ++i) {
        if (v->at(i).as_string() == name) {
            return true;
        }
    }
    return false;
}

// ---------------------------------------------------------------- graph-building interface

struct LossBuildIO {
    traincpp::Context* ctx = nullptr;
    const GoldenManifest* golden = nullptr;
    std::map<std::string, traincpp::Tensor*> inputs;
    std::vector<std::pair<std::string, traincpp::Tensor*>> compares;  // forward outputs (for VJP)
    traincpp::Tensor* scalar_loss = nullptr;  // provided directly by scalar loss modules
    StftInternals stft_dbg;                   // stft/mel intermediates (debug)
    bool stft_dbg_valid = false;
    // Called after allocation and before compute (fills constants such as the DFT kernel/mel filterbank)
    std::function<bool(std::string*)> fill_consts;

    traincpp::Tensor* find_input(const std::string& name) const {
        auto it = inputs.find(name);
        return (it == inputs.end()) ? nullptr : it->second;
    }
    void add_compare(const std::string& name, traincpp::Tensor* t) {
        compares.emplace_back(name, t);
    }
};

using LossBuildFn = bool (*)(LossBuildIO& io, std::string* err);

// ---------------------------------------------------------------- stft/mel builders

bool config_from_golden(const GoldenManifest& gm, StftMelConfig* cfg, std::string* err) {
    const common::JsonValue* p = &gm.params;
    if (!p->is_object()) {
        if (err != nullptr) {
            *err = "manifest.params missing (STFT/mel needs n_fft/hop/win/n_mels/sr etc.)";
        }
        return false;
    }
    const common::JsonValue* v = p->find("n_fft");
    cfg->n_fft = (v != nullptr) ? v->as_int(2048) : 2048;
    v = p->find("hop");
    cfg->hop = (v != nullptr) ? v->as_int(400) : 400;
    v = p->find("win");
    cfg->win = (v != nullptr) ? v->as_int(2048) : 2048;
    v = p->find("n_mels");
    cfg->n_mels = (v != nullptr) ? v->as_int(125) : 125;
    v = p->find("sr");
    cfg->sr = (v != nullptr) ? v->as_number(40000.0) : 40000.0;
    v = p->find("fmin");
    cfg->fmin = (v != nullptr) ? v->as_number(0.0) : 0.0;
    v = p->find("fmax");
    cfg->fmax = (v != nullptr) ? v->as_number(0.0) : 0.0;
    v = p->find("mag_eps");
    cfg->mag_eps = (v != nullptr) ? (float) v->as_number(2e-7) : 2e-7f;
    v = p->find("log_clip");
    cfg->log_clip = (v != nullptr) ? (float) v->as_number(2e-6) : 2e-6f;
    return cfg->validate(err);
}

bool build_loss_stft(LossBuildIO& io, std::string* err) {
    StftMelConfig cfg;
    if (!config_from_golden(*io.golden, &cfg, err)) {
        return false;
    }
    traincpp::Tensor* y = io.find_input("y");
    if (y == nullptr) {
        if (err != nullptr) {
            *err = "missing golden input: y";
        }
        return false;
    }
    StftMelConsts c;
    stft_mel_consts_init(io.ctx, cfg, &c);
    traincpp::Tensor* spec = stft_forward(io.ctx, c, cfg, y, &io.stft_dbg);
    io.stft_dbg_valid = true;
    if (spec == nullptr) {
        if (err != nullptr) {
            *err = "stft_forward graph build failed";
        }
        return false;
    }
    io.add_compare("spec", spec);
    io.fill_consts = [cfg, c](std::string*) mutable {
        stft_mel_fill_dft_kernel(&c, cfg);
        return true;
    };
    return true;
}

bool build_loss_mel(LossBuildIO& io, std::string* err) {
    StftMelConfig cfg;
    if (!config_from_golden(*io.golden, &cfg, err)) {
        return false;
    }
    traincpp::Tensor* y = io.find_input("y");
    if (y == nullptr) {
        if (err != nullptr) {
            *err = "missing golden input: y";
        }
        return false;
    }
    StftMelConsts c;
    stft_mel_consts_init(io.ctx, cfg, &c);
    traincpp::Tensor* mel = mel_spectrogram_forward(io.ctx, c, cfg, y);
    if (mel == nullptr) {
        if (err != nullptr) {
            *err = "mel_spectrogram_forward graph build failed";
        }
        return false;
    }
    io.add_compare("mel", mel);

    const GoldenTensor* basis_gt = io.golden->find_tensor("mel_basis");
    const std::string dir = io.golden->dir;
    io.fill_consts = [cfg, c, basis_gt, dir](std::string* e) mutable {
        stft_mel_fill_dft_kernel(&c, cfg);
        stft_mel_fill_mel_basis(&c, cfg);
        if (basis_gt == nullptr) {
            return true;
        }
        // Element-wise check of our filterbank vs golden (librosa)
        std::vector<uint8_t> ref;
        if (!read_file_bytes(join_path(dir, basis_gt->file), &ref, e)) {
            return false;
        }
        const size_t n = (size_t) cfg.n_mels * (size_t) cfg.bins();
        if (ref.size() != n * sizeof(float)) {
            if (e != nullptr) {
                *e = "mel_basis golden size mismatch";
            }
            return false;
        }
        std::vector<float> gen(n);
        make_mel_filterbank_slaney(cfg, gen.data());
        const float* rp = (const float*) ref.data();
        CompareStat st;
        compare_f32(rp, gen.data(), (int64_t) n, &st);
        const bool ok = (st.max_abs <= 1e-6);
        std::printf("[selftest-loss]   mel_basis(ours vs librosa) max_abs=%.3e rel=%.3e %s\n",
                    st.max_abs, st.rel, ok ? "OK" : "FAIL");
        if (!ok && e != nullptr) {
            *e = "our mel_basis filterbank does not match golden";
        }
        return ok;
    };
    return true;
}

// ---------------------------------------------------------------- loss-module builders (M3-2c)

double golden_number(const GoldenManifest& gm, const char* key, double def) {
    const common::JsonValue* v = gm.params.find(key);
    return (v != nullptr && v->is_number()) ? v->as_number(def) : def;
}

bool build_loss_lsgan_d(LossBuildIO& io, std::string* err) {
    const int64_t n = io.golden->param_int("n_disc", 7);
    std::vector<traincpp::Tensor*> dr;
    std::vector<traincpp::Tensor*> dg;
    for (int64_t i = 0; i < n; ++i) {
        traincpp::Tensor* r = io.find_input("dr" + std::to_string(i));
        traincpp::Tensor* g = io.find_input("dg" + std::to_string(i));
        if (r == nullptr || g == nullptr) {
            if (err != nullptr) {
                *err = "missing golden input dr/dg" + std::to_string(i);
            }
            return false;
        }
        dr.push_back(r);
        dg.push_back(g);
    }
    io.scalar_loss = loss_lsgan_discriminator(io.ctx, dr, dg);
    return io.scalar_loss != nullptr;
}

bool build_loss_lsgan_g(LossBuildIO& io, std::string* err) {
    const int64_t n = io.golden->param_int("n_disc", 7);
    std::vector<traincpp::Tensor*> dg;
    for (int64_t i = 0; i < n; ++i) {
        traincpp::Tensor* g = io.find_input("dg" + std::to_string(i));
        if (g == nullptr) {
            if (err != nullptr) {
                *err = "missing golden input dg" + std::to_string(i);
            }
            return false;
        }
        dg.push_back(g);
    }
    io.scalar_loss = loss_lsgan_generator(io.ctx, dg);
    return io.scalar_loss != nullptr;
}

bool build_loss_fmap(LossBuildIO& io, std::string* err) {
    const int64_t n = io.golden->param_int("n_disc", 7);
    std::vector<int64_t> counts;
    io.golden->param_int_array("fmap_counts", &counts);
    if ((int64_t) counts.size() != n) {
        if (err != nullptr) {
            *err = "manifest fmap_counts does not match n_disc";
        }
        return false;
    }
    std::vector<std::vector<traincpp::Tensor*>> fr((size_t) n);
    std::vector<std::vector<traincpp::Tensor*>> fg((size_t) n);
    for (int64_t i = 0; i < n; ++i) {
        for (int64_t j = 0; j < counts[(size_t) i]; ++j) {
            const std::string rname = "fr" + std::to_string(i) + "_" + std::to_string(j);
            const std::string gname = "fg" + std::to_string(i) + "_" + std::to_string(j);
            traincpp::Tensor* r = io.find_input(rname);
            traincpp::Tensor* g = io.find_input(gname);
            if (r == nullptr || g == nullptr) {
                if (err != nullptr) {
                    *err = "missing golden input " + rname + "/" + gname;
                }
                return false;
            }
            fr[(size_t) i].push_back(r);
            fg[(size_t) i].push_back(g);
        }
    }
    io.scalar_loss = loss_feature_matching(io.ctx, fr, fg);
    return io.scalar_loss != nullptr;
}

bool build_loss_kl(LossBuildIO& io, std::string* err) {
    traincpp::Tensor* z_p = io.find_input("z_p");
    traincpp::Tensor* logs_q = io.find_input("logs_q");
    traincpp::Tensor* m_p = io.find_input("m_p");
    traincpp::Tensor* logs_p = io.find_input("logs_p");
    traincpp::Tensor* z_mask = io.find_input("z_mask");
    if (z_p == nullptr || logs_q == nullptr || m_p == nullptr || logs_p == nullptr ||
        z_mask == nullptr) {
        if (err != nullptr) {
            *err = "missing golden input z_p/logs_q/m_p/logs_p/z_mask";
        }
        return false;
    }
    traincpp::Tensor* kl = loss_kl(io.ctx, z_p, logs_q, m_p, logs_p, z_mask);
    if (kl == nullptr) {
        return false;
    }
    const float c_kl = (float) golden_number(*io.golden, "c_kl", 1.0);
    io.scalar_loss = traincpp::scale(io.ctx, kl, c_kl);
    return io.scalar_loss != nullptr;
}

bool build_loss_mel_l1(LossBuildIO& io, std::string* err) {
    traincpp::Tensor* y_mel = io.find_input("y_mel");
    traincpp::Tensor* y_hat_mel = io.find_input("y_hat_mel");
    if (y_mel == nullptr || y_hat_mel == nullptr) {
        if (err != nullptr) {
            *err = "missing golden input y_mel/y_hat_mel";
        }
        return false;
    }
    traincpp::Tensor* l1 = loss_l1_mean(io.ctx, y_mel, y_hat_mel);
    if (l1 == nullptr) {
        return false;
    }
    const float c_mel = (float) golden_number(*io.golden, "c_mel", 45.0);
    io.scalar_loss = traincpp::scale(io.ctx, l1, c_mel);
    return io.scalar_loss != nullptr;
}

// ---------------------------------------------------------------- registry

struct Entry {
    const char* name;
    LossBuildFn build;
};

// Order defines the cotangent-seed convention index (matches MODULES in tools/gen_golden_loss.py)
const Entry k_entries[] = {
    {"stft", build_loss_stft},
    {"mel", build_loss_mel},
    {"lsgan_d", build_loss_lsgan_d},
    {"lsgan_g", build_loss_lsgan_g},
    {"fmap", build_loss_fmap},
    {"kl", build_loss_kl},
    {"mel_l1", build_loss_mel_l1},
};

const Entry* find_entry(const std::string& module) {
    for (const Entry& e : k_entries) {
        if (module == e.name) {
            return &e;
        }
    }
    return nullptr;
}

// ---------------------------------------------------------------- single-module flow

int run_single_loss(const std::string& golden_dir, const std::string& module,
                    traincpp::Device* dev) {
    using namespace traincpp;
    const Entry* entry = find_entry(module);
    if (entry == nullptr) {
        std::printf("[selftest-loss] unknown module: %s\n", module.c_str());
        return 2;
    }
    if (dev == nullptr) {
        std::printf("[selftest-loss] no usable device\n");
        return 2;
    }

    std::string err;
    GoldenManifest gm;
    if (!rvc_train::model::golden_manifest_load(golden_dir, "loss." + module, &gm, &err)) {
        std::printf("[selftest-loss] %s\n", err.c_str());
        return 1;
    }
    std::printf("[selftest-loss] module=%s golden=%s\n", module.c_str(), golden_dir.c_str());

    Context* ctx = context_new((size_t) (16 << 20));
    if (ctx == nullptr) {
        std::printf("[selftest-loss] context_new failed\n");
        return 1;
    }

    LossBuildIO io;
    io.ctx    = ctx;
    io.golden = &gm;
    for (const GoldenTensor& gt : gm.tensors) {
        if (gt.role != "input") {
            continue;
        }
        const Type type = (gt.dtype == "i32") ? TYPE_I32 : TYPE_F32;
        Tensor* t = new_tensor_nd(ctx, type, gt.nd, gt.ne.data());
        if (t == nullptr) {
            std::printf("[selftest-loss] failed to create input tensor: %s\n", gt.name.c_str());
            context_free(ctx);
            return 1;
        }
        std::snprintf(t->name, MAX_NAME, "%s", gt.name.c_str());
        io.inputs[gt.name] = t;
    }
    for (auto& kv : io.inputs) {
        if (kv.second != nullptr && kv.second->type == TYPE_F32 && !is_no_grad(gm, kv.first)) {
            tensor_set_param(kv.second);
        }
    }
    if (!entry->build(io, &err)) {
        std::printf("[selftest-loss] graph build failed: %s\n", err.c_str());
        context_free(ctx);
        return 1;
    }

    std::printf("[selftest-loss] inputs: ");
    for (const auto& kv : io.inputs) {
        std::printf("%s ne=[%lld,%lld,%lld]  ", kv.first.c_str(), (long long) kv.second->ne[0],
                    (long long) kv.second->ne[1], (long long) kv.second->ne[2]);
    }
    std::printf("\n");

    Graph* graph = graph_new(ctx);
    if (graph == nullptr) {
        std::printf("[selftest-loss] graph_new failed\n");
        context_free(ctx);
        return 1;
    }
    Buffer* buf = nullptr;
    auto cleanup = [&]() {
        if (buf != nullptr) {
            buffer_free(buf);
        }
        graph_free(graph);
        context_free(ctx);
    };

    // Loss: scalar modules are used directly; stft/mel assemble the VJP from the output + cotangent
    Tensor* loss = io.scalar_loss;
    std::vector<std::pair<const GoldenTensor*, Tensor*>> cots;
    if (loss == nullptr) {
        for (const GoldenTensor& gt : gm.tensors) {
            if (gt.role != "cotangent") {
                continue;
            }
            Tensor* out_t = nullptr;
            for (const auto& c : io.compares) {
                if (c.first == gt.name) {
                    out_t = c.second;
                    break;
                }
            }
            if (out_t == nullptr) {
                std::printf("[selftest-loss] C++ did not provide the cotangent target tensor: %s\n", gt.name.c_str());
                cleanup();
                return 1;
            }
            if (!tensor_matches_ne(out_t, gt)) {
                std::printf("[selftest-loss] cotangent target shape mismatch: %s\n", gt.name.c_str());
                cleanup();
                return 1;
            }
            Tensor* cot = new_tensor_nd(ctx, TYPE_F32, gt.nd, gt.ne.data());
            if (cot == nullptr) {
                std::printf("[selftest-loss] failed to create cotangent leaf: %s\n", gt.name.c_str());
                cleanup();
                return 1;
            }
            std::snprintf(cot->name, MAX_NAME, "cot:%s", gt.name.c_str());
            Tensor* term = sum(ctx, mul(ctx, out_t, cot));
            if (term == nullptr) {
                std::printf("[selftest-loss] VJP graph build failed: %s\n", gt.name.c_str());
                cleanup();
                return 1;
            }
            loss = (loss == nullptr) ? term : add(ctx, loss, term);
            if (loss == nullptr) {
                std::printf("[selftest-loss] VJP accumulation failed: %s\n", gt.name.c_str());
                cleanup();
                return 1;
            }
            cots.emplace_back(&gt, cot);
        }
    }
    if (loss == nullptr) {
        std::printf("[selftest-loss] manifest has no loss/cotangent entries\n");
        cleanup();
        return 1;
    }
    std::snprintf(loss->name, MAX_NAME, "selftest_loss");
    tensor_set_loss(loss);

    graph_build_forward_expand(ctx, graph, loss);
    graph_build_backward_expand(graph);

    const DeviceProps& dp = dev->props();
    std::printf("[selftest-loss] backend = %s (%s)\n", dp.name.c_str(),
                dp.type == DeviceType::CPU ? "CPU" : "GPU");

    buf = buffer_alloc_ctx_tensors(ctx, dev->default_buffer_type());
    if (buf == nullptr) {
        std::printf("[selftest-loss] compute buffer allocation failed (use RVCT_DEBUG_ALLOC=1 to see total pending allocation)\n");
        cleanup();
        return 1;
    }
    if (!set_inputs(gm, io.inputs, &err)) {
        std::printf("[selftest-loss] %s\n", err.c_str());
        cleanup();
        return 1;
    }
    for (const auto& c : cots) {
        if (!fill_tensor_from_file(c.second, gm.dir, *c.first, &err)) {
            std::printf("[selftest-loss] cotangent injection failed: %s\n", err.c_str());
            cleanup();
            return 1;
        }
    }
    if (io.fill_consts && !io.fill_consts(&err)) {
        std::printf("[selftest-loss] constant fill failed: %s\n", err.c_str());
        cleanup();
        return 1;
    }
    const Tensor* unsupported = graph_first_unsupported(graph, dev);
    if (unsupported != nullptr) {
        std::printf("[selftest-loss] node not supported by device: %s\n", unsupported->name);
        cleanup();
        return 1;
    }
    graph_reset(graph);
    if (!dev->graph_compute(graph)) {
        std::printf("[selftest-loss] graph_compute failed\n");
        cleanup();
        return 1;
    }

    // RVCT_DEBUG_LOSS=1: print stft/mel intermediates (for localization)
    {
        if (platform::env_present("RVCT_DEBUG_LOSS")) {
            auto dump = [](const char* tag, const Tensor* t, int64_t from, int64_t n) {
                std::printf("[dbg] %s ne=[%lld,%lld,%lld,%lld] :", tag,
                            (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2],
                            (long long) t->ne[3]);
                const int64_t total = tensor_nelements(t);
                for (int64_t i = from; i < from + n && i < total; ++i) {
                    std::printf(" %.6f", (double) tensor_get_f32_at(t, i));
                }
                std::printf("\n");
            };
            auto it = io.inputs.find("y");
            if (it != io.inputs.end()) {
                dump("y", it->second, 0, 5);
            }
            if (io.stft_dbg_valid) {
                const Tensor* k = io.stft_dbg.kernel;
                const int64_t N = k->ne[0];
                const int64_t bins = k->ne[2] / 2;
                dump("kernel[cos k=0,n=0..3]", k, 0, 4);
                dump("kernel[cos k=1,n=0..3]", k, N, 4);
                dump("kernel[sin k=1,n=0..3]", k, (bins + 1) * N, 4);
                dump("padded[0..5]", io.stft_dbg.padded, 0, 6);
                dump("padded@824..829", io.stft_dbg.padded, 824, 6);
                dump("conv[frame0,oc0..3]", io.stft_dbg.conv, 0, 4);
                dump("conv[frame0,oc1025..1028]", io.stft_dbg.conv, bins * io.stft_dbg.conv->ne[0], 4);
                dump("re[0..3]", io.stft_dbg.re, 0, 4);
                dump("im[0..3]", io.stft_dbg.im, 0, 4);
            }
        }
    }

    int n_checked = 0;
    int n_failed  = 0;

    // 1) forward outputs (stft/mel)
    for (const GoldenTensor& gt : gm.tensors) {
        if (gt.role != "output") {
            continue;
        }
        Tensor* t = nullptr;
        for (const auto& c : io.compares) {
            if (c.first == gt.name) {
                t = c.second;
                break;
            }
        }
        if (t == nullptr || !tensor_matches_ne(t, gt)) {
            std::printf("[selftest-loss]   %-10s [FAIL] output missing/shape mismatch\n", gt.name.c_str());
            ++n_checked;
            ++n_failed;
            continue;
        }
        const size_t nbytes = (size_t) tensor_nelements(t) * tensor_element_size(t);
        std::vector<uint8_t> got(nbytes);
        std::vector<uint8_t> ref;
        tensor_get(t, got.data(), 0, nbytes);
        if (!read_file_bytes(join_path(gm.dir, gt.file), &ref, &err) || ref.size() != nbytes) {
            std::printf("[selftest-loss]   %-10s [FAIL] golden read failed\n", gt.name.c_str());
            ++n_checked;
            ++n_failed;
            continue;
        }
        CompareStat st;
        const int64_t n = (int64_t) (nbytes / sizeof(float));
        compare_f32((const float*) ref.data(), (const float*) got.data(), n, &st);
        const bool ok = (st.rel <= kFwdTol);
        std::printf("[selftest-loss]   %-10s ne=%-18s max_abs=%.3e rel=%.3e %s\n", gt.name.c_str(),
                    ne_string(gt).c_str(), st.max_abs, st.rel, ok ? "OK" : "FAIL");
        ++n_checked;
        if (!ok) {
            ++n_failed;
        }
    }

    // 2) loss scalar
    for (const GoldenTensor& gt : gm.tensors) {
        if (gt.role != "loss") {
            continue;
        }
        std::vector<uint8_t> ref;
        if (!read_file_bytes(join_path(gm.dir, gt.file), &ref, &err) || ref.size() < sizeof(float)) {
            std::printf("[selftest-loss]   loss       [FAIL] golden read failed\n");
            ++n_checked;
            ++n_failed;
            continue;
        }
        const float ref_v = *(const float*) ref.data();
        const float got_v = tensor_get_f32_at(loss, 0);
        const double rel = std::fabs((double) got_v - (double) ref_v) /
                           std::max(1.0, std::fabs((double) ref_v));
        const bool ok = (rel <= kLossTol);
        std::printf("[selftest-loss]   loss       %.6f vs %.6f rel=%.3e %s\n", (double) ref_v,
                    (double) got_v, rel, ok ? "OK" : "FAIL");
        ++n_checked;
        if (!ok) {
            ++n_failed;
        }
    }

    // 3) input gradients
    for (const GoldenTensor& gt : gm.tensors) {
        if (gt.role != "grad") {
            continue;
        }
        ++n_checked;
        auto it = io.inputs.find(gt.name);
        Tensor* target = (it == io.inputs.end()) ? nullptr : it->second;
        if (target == nullptr || !tensor_matches_ne(target, gt)) {
            std::printf("[selftest-loss]   %-28s [FAIL] gradient target missing/shape mismatch\n", gt.name.c_str());
            ++n_failed;
            continue;
        }
        Tensor* grad = target->grad_acc;
        if (grad == nullptr) {
            std::printf("[selftest-loss]   %-28s [FAIL] grad_acc=null (did not participate in backward)\n", gt.name.c_str());
            ++n_failed;
            continue;
        }
        const size_t nbytes = (size_t) tensor_nelements(grad) * tensor_element_size(grad);
        std::vector<uint8_t> got(nbytes);
        std::vector<uint8_t> ref;
        tensor_get(grad, got.data(), 0, nbytes);
        if (!read_file_bytes(join_path(gm.dir, gt.file), &ref, &err) || ref.size() != nbytes) {
            std::printf("[selftest-loss]   %-28s [FAIL] golden read failed\n", gt.name.c_str());
            ++n_failed;
            continue;
        }
        CompareStat st;
        const int64_t n = (int64_t) (nbytes / sizeof(float));
        compare_f32((const float*) ref.data(), (const float*) got.data(), n, &st);
        const bool ok = (st.rel <= kBackTol) && (st.rms_rel <= kBackRmsTol || st.max_abs <= 1e-5);
        std::printf("[selftest-loss]   %-28s ne=%-18s max_abs=%.3e rel=%.3e rms=%.3e %s\n",
                    gt.name.c_str(), ne_string(gt).c_str(), st.max_abs, st.rel, st.rms_rel,
                    ok ? "OK" : "FAIL");
        if (!ok) {
            ++n_failed;
        }
    }

    cleanup();
    std::printf("[selftest-loss] %s: %d items compared, %d failed\n", module.c_str(), n_checked, n_failed);
    return (n_failed == 0) ? 0 : 1;
}

} // namespace

int run_loss_selftest(const std::string& golden_dir, const std::string& module,
                      traincpp::Device* dev) {
    if (module != "all") {
        return run_single_loss(golden_dir, module, dev);
    }
    int failed = 0;
    for (const Entry& e : k_entries) {
        const int rc = run_single_loss(golden_dir, e.name, dev);
        if (rc != 0) {
            failed = 1;
        }
    }
    std::printf("[selftest-loss] all: %s\n", failed ? "FAIL" : "OK");
    return failed ? 1 : 0;
}

} // namespace train
} // namespace rvc_train
