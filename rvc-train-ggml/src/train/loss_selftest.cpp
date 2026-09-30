// RVC-Train-GGML - M3-2 损失/前端 golden 对拍实现（src/train/loss_selftest.cpp）
//
// 结构仿 M3-1 反向对拍（src/model/selftest.cpp::run_single_backward）：
//   读 manifest.loss.<module>.json → 建输入叶子（F32 标记可训练）→ 建图
//   （stft/mel：输出 + 固定 cotangent VJP；损失模块：标量 loss 直接 set_loss）
//   → graph_build_backward_expand（分配前）→ 分配 → 注入 → 填常量 → compute
//   → 前向/损失/梯度逐项对比。
#include "train/loss_selftest.h"

#include "common/json.h"
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

// ---------------------------------------------------------------- 小工具

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
            *err = "无法打开文件: " + path;
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
        std::snprintf(buf, sizeof(buf), "文件大小不符: %s（%zu != %zu）", gt.file.c_str(),
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
                *err = "输入张量未创建: " + gt.name;
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

// ---------------------------------------------------------------- 建图接口

struct LossBuildIO {
    traincpp::Context* ctx = nullptr;
    const GoldenManifest* golden = nullptr;
    std::map<std::string, traincpp::Tensor*> inputs;
    std::vector<std::pair<std::string, traincpp::Tensor*>> compares;  // 前向输出（VJP 用）
    traincpp::Tensor* scalar_loss = nullptr;  // 标量损失模块直接提供
    StftInternals stft_dbg;                   // stft/mel 中间量（调试）
    bool stft_dbg_valid = false;
    // 分配完成后、compute 之前调用（填 DFT 核/mel 滤波器组等常量）
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

// ---------------------------------------------------------------- stft/mel 构建器

bool config_from_golden(const GoldenManifest& gm, StftMelConfig* cfg, std::string* err) {
    const common::JsonValue* p = &gm.params;
    if (!p->is_object()) {
        if (err != nullptr) {
            *err = "manifest.params 缺失（STFT/mel 需要 n_fft/hop/win/n_mels/sr 等）";
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
            *err = "缺少 golden 输入: y";
        }
        return false;
    }
    StftMelConsts c;
    stft_mel_consts_init(io.ctx, cfg, &c);
    traincpp::Tensor* spec = stft_forward(io.ctx, c, cfg, y, &io.stft_dbg);
    io.stft_dbg_valid = true;
    if (spec == nullptr) {
        if (err != nullptr) {
            *err = "stft_forward 建图失败";
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
            *err = "缺少 golden 输入: y";
        }
        return false;
    }
    StftMelConsts c;
    stft_mel_consts_init(io.ctx, cfg, &c);
    traincpp::Tensor* mel = mel_spectrogram_forward(io.ctx, c, cfg, y);
    if (mel == nullptr) {
        if (err != nullptr) {
            *err = "mel_spectrogram_forward 建图失败";
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
        // 自研滤波器组 vs golden（librosa）逐元素校验
        std::vector<uint8_t> ref;
        if (!read_file_bytes(join_path(dir, basis_gt->file), &ref, e)) {
            return false;
        }
        const size_t n = (size_t) cfg.n_mels * (size_t) cfg.bins();
        if (ref.size() != n * sizeof(float)) {
            if (e != nullptr) {
                *e = "mel_basis golden 大小不符";
            }
            return false;
        }
        std::vector<float> gen(n);
        make_mel_filterbank_slaney(cfg, gen.data());
        const float* rp = (const float*) ref.data();
        CompareStat st;
        compare_f32(rp, gen.data(), (int64_t) n, &st);
        const bool ok = (st.max_abs <= 1e-6);
        std::printf("[selftest-loss]   mel_basis(自研 vs librosa) max_abs=%.3e rel=%.3e %s\n",
                    st.max_abs, st.rel, ok ? "OK" : "FAIL");
        if (!ok && e != nullptr) {
            *e = "mel_basis 自研滤波器组与 golden 不一致";
        }
        return ok;
    };
    return true;
}

// ---------------------------------------------------------------- 损失模块构建器（M3-2c）

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
                *err = "缺少 golden 输入 dr/dg" + std::to_string(i);
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
                *err = "缺少 golden 输入 dg" + std::to_string(i);
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
            *err = "manifest fmap_counts 与 n_disc 不一致";
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
                    *err = "缺少 golden 输入 " + rname + "/" + gname;
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
            *err = "缺少 golden 输入 z_p/logs_q/m_p/logs_p/z_mask";
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
            *err = "缺少 golden 输入 y_mel/y_hat_mel";
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

// ---------------------------------------------------------------- 注册表

struct Entry {
    const char* name;
    LossBuildFn build;
};

// 顺序即 cotangent 种子约定序号（与 tools/gen_golden_loss.py 的 MODULES 一致）
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

// ---------------------------------------------------------------- 单模块流程

int run_single_loss(const std::string& golden_dir, const std::string& module,
                    traincpp::Device* dev) {
    using namespace traincpp;
    const Entry* entry = find_entry(module);
    if (entry == nullptr) {
        std::printf("[selftest-loss] 未知模块: %s\n", module.c_str());
        return 2;
    }
    if (dev == nullptr) {
        std::printf("[selftest-loss] 无可用设备\n");
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
        std::printf("[selftest-loss] context_new 失败\n");
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
            std::printf("[selftest-loss] 创建输入张量失败: %s\n", gt.name.c_str());
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
        std::printf("[selftest-loss] 建图失败: %s\n", err.c_str());
        context_free(ctx);
        return 1;
    }

    std::printf("[selftest-loss] 输入: ");
    for (const auto& kv : io.inputs) {
        std::printf("%s ne=[%lld,%lld,%lld]  ", kv.first.c_str(), (long long) kv.second->ne[0],
                    (long long) kv.second->ne[1], (long long) kv.second->ne[2]);
    }
    std::printf("\n");

    Graph* graph = graph_new(ctx);
    if (graph == nullptr) {
        std::printf("[selftest-loss] graph_new 失败\n");
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

    // 损失：标量模块直接用；stft/mel 由输出 + cotangent 组装 VJP
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
                std::printf("[selftest-loss] C++ 未提供 cotangent 目标张量: %s\n", gt.name.c_str());
                cleanup();
                return 1;
            }
            if (!tensor_matches_ne(out_t, gt)) {
                std::printf("[selftest-loss] cotangent 目标形状不符: %s\n", gt.name.c_str());
                cleanup();
                return 1;
            }
            Tensor* cot = new_tensor_nd(ctx, TYPE_F32, gt.nd, gt.ne.data());
            if (cot == nullptr) {
                std::printf("[selftest-loss] cotangent 叶子创建失败: %s\n", gt.name.c_str());
                cleanup();
                return 1;
            }
            std::snprintf(cot->name, MAX_NAME, "cot:%s", gt.name.c_str());
            Tensor* term = sum(ctx, mul(ctx, out_t, cot));
            if (term == nullptr) {
                std::printf("[selftest-loss] VJP 建图失败: %s\n", gt.name.c_str());
                cleanup();
                return 1;
            }
            loss = (loss == nullptr) ? term : add(ctx, loss, term);
            if (loss == nullptr) {
                std::printf("[selftest-loss] VJP 累加失败: %s\n", gt.name.c_str());
                cleanup();
                return 1;
            }
            cots.emplace_back(&gt, cot);
        }
    }
    if (loss == nullptr) {
        std::printf("[selftest-loss] manifest 无 loss/cotangent 条目\n");
        cleanup();
        return 1;
    }
    std::snprintf(loss->name, MAX_NAME, "selftest_loss");
    tensor_set_loss(loss);

    graph_build_forward_expand(ctx, graph, loss);
    graph_build_backward_expand(graph);

    const DeviceProps& dp = dev->props();
    std::printf("[selftest-loss] backend = %s（%s）\n", dp.name.c_str(),
                dp.type == DeviceType::CPU ? "CPU" : "GPU");

    buf = buffer_alloc_ctx_tensors(ctx, dev->default_buffer_type());
    if (buf == nullptr) {
        std::printf("[selftest-loss] 计算缓冲分配失败（可用 RVCT_DEBUG_ALLOC=1 查看待分配总量）\n");
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
            std::printf("[selftest-loss] cotangent 注入失败: %s\n", err.c_str());
            cleanup();
            return 1;
        }
    }
    if (io.fill_consts && !io.fill_consts(&err)) {
        std::printf("[selftest-loss] 常量填充失败: %s\n", err.c_str());
        cleanup();
        return 1;
    }
    const Tensor* unsupported = graph_first_unsupported(graph, dev);
    if (unsupported != nullptr) {
        std::printf("[selftest-loss] 设备不支持节点: %s\n", unsupported->name);
        cleanup();
        return 1;
    }
    graph_reset(graph);
    if (!dev->graph_compute(graph)) {
        std::printf("[selftest-loss] graph_compute 失败\n");
        cleanup();
        return 1;
    }

    // RVCT_DEBUG_LOSS=1：打印 stft/mel 中间量（定位用）
    {
        char*  dbg_env = nullptr;
        size_t dbg_len = 0;
        _dupenv_s(&dbg_env, &dbg_len, "RVCT_DEBUG_LOSS");
        if (dbg_env != nullptr) {
            free(dbg_env);
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

    // 1) 前向输出（stft/mel）
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
            std::printf("[selftest-loss]   %-10s [FAIL] 输出缺失/形状不符\n", gt.name.c_str());
            ++n_checked;
            ++n_failed;
            continue;
        }
        const size_t nbytes = (size_t) tensor_nelements(t) * tensor_element_size(t);
        std::vector<uint8_t> got(nbytes);
        std::vector<uint8_t> ref;
        tensor_get(t, got.data(), 0, nbytes);
        if (!read_file_bytes(join_path(gm.dir, gt.file), &ref, &err) || ref.size() != nbytes) {
            std::printf("[selftest-loss]   %-10s [FAIL] golden 读取失败\n", gt.name.c_str());
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

    // 2) 损失标量
    for (const GoldenTensor& gt : gm.tensors) {
        if (gt.role != "loss") {
            continue;
        }
        std::vector<uint8_t> ref;
        if (!read_file_bytes(join_path(gm.dir, gt.file), &ref, &err) || ref.size() < sizeof(float)) {
            std::printf("[selftest-loss]   loss       [FAIL] golden 读取失败\n");
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

    // 3) 输入梯度
    for (const GoldenTensor& gt : gm.tensors) {
        if (gt.role != "grad") {
            continue;
        }
        ++n_checked;
        auto it = io.inputs.find(gt.name);
        Tensor* target = (it == io.inputs.end()) ? nullptr : it->second;
        if (target == nullptr || !tensor_matches_ne(target, gt)) {
            std::printf("[selftest-loss]   %-28s [FAIL] 梯度目标缺失/形状不符\n", gt.name.c_str());
            ++n_failed;
            continue;
        }
        Tensor* grad = target->grad_acc;
        if (grad == nullptr) {
            std::printf("[selftest-loss]   %-28s [FAIL] grad_acc=null（未参与反向）\n", gt.name.c_str());
            ++n_failed;
            continue;
        }
        const size_t nbytes = (size_t) tensor_nelements(grad) * tensor_element_size(grad);
        std::vector<uint8_t> got(nbytes);
        std::vector<uint8_t> ref;
        tensor_get(grad, got.data(), 0, nbytes);
        if (!read_file_bytes(join_path(gm.dir, gt.file), &ref, &err) || ref.size() != nbytes) {
            std::printf("[selftest-loss]   %-28s [FAIL] golden 读取失败\n", gt.name.c_str());
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
    std::printf("[selftest-loss] %s: %d 项比较，%d 失败\n", module.c_str(), n_checked, n_failed);
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
