// RVC-Train-GGML - 前向对拍 harness 实现（src/model/selftest.cpp）
#include "model/selftest.h"

#include "model/dec.h"
#include "model/discriminator.h"
#include "model/emb.h"
#include "model/enc_p.h"
#include "model/enc_q.h"
#include "model/flow.h"
#include "model/resblock.h"
#include "model/sinegen.h"
#include "model/synthesizer.h"
#include "model/wn.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>

using namespace traincpp;

namespace rvc_train {
namespace model {

namespace {

constexpr double kRelTol = 1e-4;

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
    double rms_rel = 0.0;  // RMS(差值) / RMS(参考)，对单元素抵消噪声更稳健
    int64_t mismatch = 0;
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

void compare_i32(const int32_t* ref, const int32_t* got, int64_t n, CompareStat* st) {
    for (int64_t i = 0; i < n; ++i) {
        if (ref[i] != got[i]) {
            ++st->mismatch;
        }
    }
}

bool create_inputs(Context* ctx, const GoldenManifest& gm, std::map<std::string, Tensor*>* inputs,
                   std::string* err) {
    for (const GoldenTensor& gt : gm.tensors) {
        if (!gt.is_input()) {
            continue;
        }
        const Type type = (gt.dtype == "i32") ? TYPE_I32 : TYPE_F32;
        Tensor* it = new_tensor_nd(ctx, type, gt.nd, gt.ne.data());
        if (it == nullptr) {
            if (err != nullptr) {
                *err = "创建输入张量失败: " + gt.name;
            }
            return false;
        }
        std::snprintf(it->name, MAX_NAME, "%s", gt.name.c_str());
        (*inputs)[gt.name] = it;
    }
    return true;
}

bool set_inputs(const GoldenManifest& gm, const std::map<std::string, Tensor*>& inputs,
                std::string* err) {
    for (const GoldenTensor& gt : gm.tensors) {
        if (!gt.is_input()) {
            continue;
        }
        auto it = inputs.find(gt.name);
        if (it == inputs.end() || it->second == nullptr) {
            if (err != nullptr) {
                *err = "输入张量未创建: " + gt.name;
            }
            return false;
        }
        Tensor* t = it->second;
        std::vector<uint8_t> bytes;
        if (!read_file_bytes(join_path(gm.dir, gt.file), &bytes, err)) {
            return false;
        }
        const size_t expect = (size_t) tensor_nelements(t) * tensor_element_size(t);
        if (bytes.size() != expect) {
            char buf[256];
            std::snprintf(buf, sizeof(buf), "输入文件大小不符: %s（%zu != %zu）", gt.file.c_str(),
                          bytes.size(), expect);
            if (err != nullptr) {
                *err = buf;
            }
            return false;
        }
        tensor_set(t, bytes.data(), 0, bytes.size());
    }
    return true;
}

bool compare_outputs(const GoldenManifest& gm,
                     const std::vector<std::pair<std::string, Tensor*>>& compares, int* n_checked,
                     int* n_fail, std::string* err) {
    *n_checked = 0;
    *n_fail = 0;
    for (const GoldenTensor& gt : gm.tensors) {
        if (gt.is_input()) {
            continue;
        }
        Tensor* t = nullptr;
        for (const auto& c : compares) {
            if (c.first == gt.name) {
                t = c.second;
                break;
            }
        }
        if (t == nullptr) {
            if (gt.role == "output") {
                if (err != nullptr) {
                    *err = "C++ 未提供输出张量: " + gt.name;
                }
                return false;
            }
            std::printf("[selftest-forward]   [warn] golden 中间量 %s 未提供，跳过\n",
                        gt.name.c_str());
            continue;
        }

        bool shape_ok = true;
        for (int i = 0; i < 4; ++i) {
            if (t->ne[i] != gt.ne[(size_t) i]) {
                shape_ok = false;
            }
        }
        const size_t nbytes = (size_t) tensor_nelements(t) * tensor_element_size(t);
        std::vector<uint8_t> got(nbytes);
        tensor_get(t, got.data(), 0, nbytes);
        std::vector<uint8_t> ref;
        if (!read_file_bytes(join_path(gm.dir, gt.file), &ref, err)) {
            return false;
        }
        if (ref.size() != nbytes) {
            char buf[256];
            std::snprintf(buf, sizeof(buf), "参考文件大小不符: %s（%zu != %zu）", gt.file.c_str(),
                          ref.size(), nbytes);
            if (err != nullptr) {
                *err = buf;
            }
            return false;
        }

        CompareStat st;
        const int64_t n = (int64_t) (nbytes / tensor_element_size(t));
        bool ok = shape_ok;
        if (gt.dtype == "i32") {
            std::vector<int32_t> gv((size_t) n);
            std::vector<int32_t> rv((size_t) n);
            std::memcpy(gv.data(), got.data(), nbytes);
            std::memcpy(rv.data(), ref.data(), nbytes);
            compare_i32(rv.data(), gv.data(), n, &st);
            ok = ok && (st.mismatch == 0);
            std::printf("[selftest-forward]   %-8s ne=%-16s %s（mismatch=%lld）\n", gt.name.c_str(),
                        ne_string(gt).c_str(), ok ? "OK" : "FAIL", (long long) st.mismatch);
        } else {
            std::vector<float> gv((size_t) n);
            std::vector<float> rv((size_t) n);
            std::memcpy(gv.data(), got.data(), nbytes);
            std::memcpy(rv.data(), ref.data(), nbytes);
            compare_f32(rv.data(), gv.data(), n, &st);
            ok = ok && (st.rel <= kRelTol);
            std::printf("[selftest-forward]   %-8s ne=%-16s max_abs=%.3e rel=%.3e %s\n",
                        gt.name.c_str(), ne_string(gt).c_str(), st.max_abs, st.rel,
                        ok ? "OK" : "FAIL");
        }
        ++(*n_checked);
        if (!ok) {
            ++(*n_fail);
        }
    }
    return true;
}

struct Entry {
    const char* name;
    SelftestBuildFn build;
};

const Entry k_entries[] = {
    {"emb_g", build_selftest_emb_g},
    {"wn", build_selftest_wn},
    {"resblock1", build_selftest_resblock1},
    {"dec", build_selftest_dec},
    {"flow", build_selftest_flow},
    {"enc_p", build_selftest_enc_p},
    {"enc_q", build_selftest_enc_q},
    {"disc", build_selftest_disc},
    {"synth", build_selftest_synth},
};

const Entry* find_entry(const std::string& module) {
    for (const Entry& e : k_entries) {
        if (module == e.name) {
            return &e;
        }
    }
    return nullptr;
}

// 调试开关 RVCT_DEBUG_ALLOC=1：打印上下文张量分配总量与 Top-15（定位超限/巨型张量）
void debug_print_alloc(const Context* ctx) {
    char*  dbg_env = nullptr;
    size_t dbg_len = 0;
    _dupenv_s(&dbg_env, &dbg_len, "RVCT_DEBUG_ALLOC");
    const bool debug_alloc = (dbg_env != nullptr);
    free(dbg_env);
    if (!debug_alloc) {
        return;
    }
    const int nt = context_tensor_count(ctx);
    size_t total = 0;
    std::vector<std::pair<size_t, std::string>> big;
    for (int i = 0; i < nt; ++i) {
        Tensor* t = context_tensor(ctx, i);
        if (t == nullptr || t->view_src != nullptr || t->data != nullptr) {
            continue;
        }
        const size_t nb = tensor_nbytes(t);
        total += nb;
        big.emplace_back(nb, std::string(t->name));
    }
    std::sort(big.begin(), big.end(),
              [](const std::pair<size_t, std::string>& a,
                 const std::pair<size_t, std::string>& b) { return a.first > b.first; });
    std::printf("[debug-alloc] tensors=%d 待分配总量=%.1f MiB（%zu 字节）\n", nt,
                (double) total / (1024.0 * 1024.0), total);
    for (size_t i = 0; i < big.size() && i < 15; ++i) {
        const std::string& nm = big[i].second;
        if (!nm.empty()) {
            std::printf("[debug-alloc]   %-38s %.2f MiB\n", nm.c_str(),
                        (double) big[i].first / (1024.0 * 1024.0));
        } else {
            std::printf("[debug-alloc]   （未命名）%.2f MiB\n",
                        (double) big[i].first / (1024.0 * 1024.0));
        }
    }
    // 未命名张量中按形状统计最大的几个（帮助定位广播/中间量）
    std::vector<std::pair<size_t, std::string>> big_anon;
    for (int i = 0; i < nt; ++i) {
        Tensor* t = context_tensor(ctx, i);
        if (t == nullptr || t->view_src != nullptr || t->data != nullptr || t->name[0] != '\0') {
            continue;
        }
        if (tensor_nbytes(t) < (size_t) (8u << 20)) {
            continue;
        }
        char buf[96];
        std::snprintf(buf, sizeof(buf), "[%lld,%lld,%lld,%lld]", (long long) t->ne[0],
                      (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3]);
        big_anon.emplace_back(tensor_nbytes(t), std::string(buf));
    }
    std::sort(big_anon.begin(), big_anon.end(),
              [](const std::pair<size_t, std::string>& a,
                 const std::pair<size_t, std::string>& b) { return a.first > b.first; });
    for (size_t i = 0; i < big_anon.size() && i < 10; ++i) {
        std::printf("[debug-alloc]   匿名 ne=%-24s %.2f MiB\n", big_anon[i].second.c_str(),
                    (double) big_anon[i].first / (1024.0 * 1024.0));
    }
    std::printf("[debug-alloc]   匿名张量（≥8MiB）数量=%zu\n", big_anon.size());
}

int run_single(const std::string& weights_path, const std::string& golden_dir,
               const std::string& module, Device* dev) {
    const Entry* entry = find_entry(module);
    if (entry == nullptr) {
        std::printf("[selftest-forward] 未知模块: %s（可用 emb_g/wn/resblock1/dec/flow/enc_p/enc_q/disc/synth/all）\n",
                    module.c_str());
        return 2;
    }
    if (dev == nullptr) {
        std::printf("[selftest-forward] 无可用设备\n");
        return 2;
    }

    std::printf("[selftest-forward] module=%s golden=%s\n", module.c_str(), golden_dir.c_str());

    std::string err;
    TrainWeights* w = TrainWeights::open(weights_path, &err);
    if (w == nullptr) {
        std::printf("[selftest-forward] 打开失败: %s\n", err.c_str());
        return 1;
    }
    TrainMeta meta;
    w->read_meta(&meta);
    const ModelConfig cfg = ModelConfig::from_meta(meta);
    if (!cfg.validate(&err)) {
        std::printf("[selftest-forward] 配置校验失败: %s\n", err.c_str());
        delete w;
        return 1;
    }

    GoldenManifest gm;
    if (!golden_manifest_load(golden_dir, module, &gm, &err)) {
        std::printf("[selftest-forward] %s\n", err.c_str());
        delete w;
        return 1;
    }

    const int64_t n = w->tensor_count();
    Context* ctx = context_new((size_t) (n + 2048) * 1024 + (1u << 20));
    if (ctx == nullptr) {
        std::printf("[selftest-forward] context_new 失败\n");
        delete w;
        return 1;
    }

    ModelParams params;
    if (!params.build_all(ctx, *w, &err)) {
        std::printf("[selftest-forward] 建参失败: %s\n", err.c_str());
        context_free(ctx);
        delete w;
        return 1;
    }

    SelftestBuildIO io;
    io.ctx      = ctx;
    io.cfg      = &cfg;
    io.params   = &params;
    io.golden   = &gm;
    if (!create_inputs(ctx, gm, &io.inputs, &err)) {
        std::printf("[selftest-forward] %s\n", err.c_str());
        context_free(ctx);
        delete w;
        return 1;
    }
    if (!entry->build(io, &err)) {
        std::printf("[selftest-forward] 建图失败: %s\n", err.c_str());
        context_free(ctx);
        delete w;
        return 1;
    }

    std::printf("[selftest-forward] 输入: ");
    for (const auto& kv : io.inputs) {
        std::printf("%s ne=[%lld,%lld,%lld]  ", kv.first.c_str(), (long long) kv.second->ne[0],
                    (long long) kv.second->ne[1], (long long) kv.second->ne[2]);
    }
    std::printf("\n");

    Graph* graph = graph_new(ctx);
    if (graph == nullptr) {
        std::printf("[selftest-forward] graph_new 失败\n");
        context_free(ctx);
        delete w;
        return 1;
    }
    for (const auto& c : io.compares) {
        graph_build_forward_expand(ctx, graph, c.second);
    }

    const DeviceProps& dp = dev->props();
    std::printf("[selftest-forward] backend = %s（%s）\n", dp.name.c_str(),
                dp.type == DeviceType::CPU ? "CPU" : "GPU");

    debug_print_alloc(ctx);
    Buffer* buf = buffer_alloc_ctx_tensors(ctx, dev->default_buffer_type());
    if (buf == nullptr) {
        std::printf("[selftest-forward] 计算缓冲分配失败\n");
        graph_free(graph);
        context_free(ctx);
        delete w;
        return 1;
    }

    int64_t loaded = 0;
    for (Tensor* t : params.params().items) {
        if (!w->load(t, t->name, &err)) {
            std::printf("[selftest-forward] 装载失败: %s\n", err.c_str());
            buffer_free(buf);
            graph_free(graph);
            context_free(ctx);
            delete w;
            return 1;
        }
        ++loaded;
    }
    std::printf("[selftest-forward] 参数装载: %lld/%lld\n", (long long) loaded, (long long) n);

    if (!set_inputs(gm, io.inputs, &err)) {
        std::printf("[selftest-forward] %s\n", err.c_str());
        buffer_free(buf);
        graph_free(graph);
        context_free(ctx);
        delete w;
        return 1;
    }

    if (io.prepare && !io.prepare(&err)) {
        std::printf("[selftest-forward] 准备失败: %s\n", err.c_str());
        buffer_free(buf);
        graph_free(graph);
        context_free(ctx);
        delete w;
        return 1;
    }

    const Tensor* unsupported = graph_first_unsupported(graph, dev);
    if (unsupported != nullptr) {
        std::printf("[selftest-forward] 设备不支持节点: %s\n", unsupported->name);
        buffer_free(buf);
        graph_free(graph);
        context_free(ctx);
        delete w;
        return 1;
    }

    if (!dev->graph_compute(graph)) {
        std::printf("[selftest-forward] graph_compute 失败\n");
        buffer_free(buf);
        graph_free(graph);
        context_free(ctx);
        delete w;
        return 1;
    }

    int checked = 0;
    int failed = 0;
    if (!compare_outputs(gm, io.compares, &checked, &failed, &err)) {
        std::printf("[selftest-forward] %s\n", err.c_str());
        buffer_free(buf);
        graph_free(graph);
        context_free(ctx);
        delete w;
        return 1;
    }

    const bool ok = (failed == 0 && checked > 0);
    std::printf("[selftest-forward] 对拍 %s（%d/%d 张量，F32 rel ≤ %.0e）\n",
                ok ? "通过" : "失败", checked - failed, checked, kRelTol);

    buffer_free(buf);
    graph_free(graph);
    context_free(ctx);
    delete w;
    return ok ? 0 : 1;
}

// ---------------------------------------------------------------- M3-1 反向对拍

constexpr double kBackTol      = 1e-3;   // 梯度对拍 max 门（短序列模块；M3 验收门）
constexpr double kBackTolLong  = 2e-2;   // dec/synth（L=19200）跨实现 F32 累加噪声见 Q17
constexpr double kBackRmsTol   = 2e-3;   // 梯度对拍 RMS 门（所有模块；抗单元素抵消噪声）
constexpr double kLossTol      = 1e-4;   // VJP 标量 loss 门（同前向）
constexpr float  kGradcheckEps = 1e-2f;  // 中心差分步长（F32）
constexpr double kGradcheckTol = 2e-2;   // 数值 gradcheck 相对门

Tensor* find_compare_tensor(const std::vector<std::pair<std::string, Tensor*>>& compares,
                            const std::string& name) {
    for (const auto& c : compares) {
        if (c.first == name) {
            return c.second;
        }
    }
    return nullptr;
}

bool tensor_matches_ne(const Tensor* t, const GoldenTensor& gt) {
    for (int i = 0; i < 4; ++i) {
        if (t->ne[i] != gt.ne[(size_t) i]) {
            return false;
        }
    }
    return true;
}

bool fill_tensor_from_file(Tensor* t, const std::string& dir, const GoldenTensor& gt,
                           std::string* err) {
    std::vector<uint8_t> bytes;
    if (!read_file_bytes(join_path(dir, gt.file), &bytes, err)) {
        return false;
    }
    const size_t expect = (size_t) tensor_nelements(t) * tensor_element_size(t);
    if (bytes.size() != expect) {
        char buf[256];
        std::snprintf(buf, sizeof(buf), "文件大小不符: %s（%zu != %zu）", gt.file.c_str(),
                      bytes.size(), expect);
        if (err != nullptr) {
            *err = buf;
        }
        return false;
    }
    tensor_set(t, bytes.data(), 0, bytes.size());
    return true;
}

float tensor_get_f32_at(const Tensor* t, int64_t idx) {
    float v = 0.0f;
    tensor_get(t, &v, (size_t) idx * sizeof(float), sizeof(float));
    return v;
}

void tensor_set_f32_at(Tensor* t, int64_t idx, float v) {
    tensor_set(t, &v, (size_t) idx * sizeof(float), sizeof(float));
}

// 与 golden <module>.<name>.grad.f32.raw 逐项比较（rel = max_abs / max(1, max|ref|)；
// 另看 rms_rel 抗单元素抵消噪声；dec/synth 因 L=19200 的跨实现 F32 累加噪声用宽容门，见 Q17）
void compare_grads(const GoldenManifest& gm, const SelftestBuildIO& io,
                   const ModelParams& params, bool long_seq, int* checked, int* failed) {
    *checked = 0;
    *failed  = 0;
    const double max_tol = long_seq ? kBackTolLong : kBackTol;
    for (const GoldenTensor& gt : gm.tensors) {
        if (gt.role != "grad") {
            continue;
        }
        ++(*checked);
        Tensor* target = nullptr;
        auto it = io.inputs.find(gt.name);
        if (it != io.inputs.end()) {
            target = it->second;
        } else {
            target = params.find(gt.name.c_str());
        }
        if (target == nullptr) {
            std::printf("[selftest-backward]   %-46s [FAIL] 找不到梯度目标\n", gt.name.c_str());
            ++(*failed);
            continue;
        }
        if (!tensor_matches_ne(target, gt)) {
            std::printf("[selftest-backward]   %-46s [FAIL] 目标形状不符\n", gt.name.c_str());
            ++(*failed);
            continue;
        }
        Tensor* grad = target->grad_acc;
        if (grad == nullptr) {
            std::printf("[selftest-backward]   %-46s [FAIL] grad_acc=null（未参与反向）\n",
                        gt.name.c_str());
            ++(*failed);
            continue;
        }
        const size_t nbytes = (size_t) tensor_nelements(grad) * tensor_element_size(grad);
        std::vector<uint8_t> got(nbytes);
        tensor_get(grad, got.data(), 0, nbytes);
        std::vector<uint8_t> ref;
        std::string err;
        if (!read_file_bytes(join_path(gm.dir, gt.file), &ref, &err) || ref.size() != nbytes) {
            std::printf("[selftest-backward]   %-46s [FAIL] 参考文件读取失败: %s\n", gt.name.c_str(),
                        err.c_str());
            ++(*failed);
            continue;
        }
        CompareStat st;
        const int64_t ne = (int64_t) (nbytes / sizeof(float));
        compare_f32((const float*) ref.data(), (const float*) got.data(), ne, &st);
        // 参考张量近似为零时（max_abs 本身极小）rms_rel 会失真 → 绝对值兜底
        const bool ok = (st.rel <= max_tol) &&
                        (st.rms_rel <= kBackRmsTol || st.max_abs <= 1e-5);
        std::printf("[selftest-backward]   %-46s ne=%-18s max_abs=%.3e rel=%.3e rms=%.3e %s\n",
                    gt.name.c_str(), ne_string(gt).c_str(), st.max_abs, st.rel, st.rms_rel,
                    ok ? "OK" : "FAIL");
        if (!ok) {
            ++(*failed);
        }
    }
}

std::string mangle_name(const std::string& name) {
    std::string s = name;
    for (char& c : s) {
        if (c == '.') {
            c = '_';
        }
    }
    return s;
}

int run_single_backward(const std::string& weights_path, const std::string& golden_dir,
                        const std::string& module, const std::string& gradcheck_spec,
                        const std::string& dump_grads, Device* dev) {
    const Entry* entry = find_entry(module);
    if (entry == nullptr) {
        std::printf("[selftest-backward] 未知模块: %s（可用 emb_g/wn/resblock1/dec/flow/enc_p/enc_q/disc/synth/all）\n",
                    module.c_str());
        return 2;
    }
    if (dev == nullptr) {
        std::printf("[selftest-backward] 无可用设备\n");
        return 2;
    }
    std::printf("[selftest-backward] module=%s golden=%s\n", module.c_str(), golden_dir.c_str());

    std::string err;
    TrainWeights* w = TrainWeights::open(weights_path, &err);
    if (w == nullptr) {
        std::printf("[selftest-backward] 打开失败: %s\n", err.c_str());
        return 1;
    }
    TrainMeta meta;
    w->read_meta(&meta);
    const ModelConfig cfg = ModelConfig::from_meta(meta);
    if (!cfg.validate(&err)) {
        std::printf("[selftest-backward] 配置校验失败: %s\n", err.c_str());
        delete w;
        return 1;
    }
    GoldenManifest gm;
    if (!golden_manifest_load(golden_dir, "backward." + module, &gm, &err)) {
        std::printf("[selftest-backward] %s\n", err.c_str());
        delete w;
        return 1;
    }

    const int64_t n = w->tensor_count();
    Context* ctx = context_new((size_t) (n + 4096) * 1024 + (1u << 20));
    if (ctx == nullptr) {
        std::printf("[selftest-backward] context_new 失败\n");
        delete w;
        return 1;
    }
    ModelParams params;
    if (!params.build_all(ctx, *w, &err)) {
        std::printf("[selftest-backward] 建参失败: %s\n", err.c_str());
        context_free(ctx);
        delete w;
        return 1;
    }
    // 反向对拍需要显式标记可训练参数（冻结用 tensor_clear_param；只影响下一次反向展开）
    for (Tensor* t : params.params().items) {
        tensor_set_param(t);
    }
    SelftestBuildIO io;
    io.ctx    = ctx;
    io.cfg    = &cfg;
    io.params = &params;
    io.golden = &gm;
    if (!create_inputs(ctx, gm, &io.inputs, &err)) {
        std::printf("[selftest-backward] %s\n", err.c_str());
        context_free(ctx);
        delete w;
        return 1;
    }
    // 浮点输入叶子也要参与反向（标记 PARAM 后才会分配 grad_acc；与权重同口径对比）
    for (auto& kv : io.inputs) {
        if (kv.second != nullptr && kv.second->type == TYPE_F32) {
            tensor_set_param(kv.second);
        }
    }
    if (!entry->build(io, &err)) {
        std::printf("[selftest-backward] 建图失败: %s\n", err.c_str());
        context_free(ctx);
        delete w;
        return 1;
    }

    std::printf("[selftest-backward] 输入: ");
    for (const auto& kv : io.inputs) {
        std::printf("%s ne=[%lld,%lld,%lld]  ", kv.first.c_str(), (long long) kv.second->ne[0],
                    (long long) kv.second->ne[1], (long long) kv.second->ne[2]);
    }
    std::printf("\n");

    Graph* graph = graph_new(ctx);
    if (graph == nullptr) {
        std::printf("[selftest-backward] graph_new 失败\n");
        context_free(ctx);
        delete w;
        return 1;
    }
    Buffer* buf = nullptr;
    auto cleanup = [&]() {
        if (buf != nullptr) {
            buffer_free(buf);
        }
        graph_free(graph);
        context_free(ctx);
        delete w;
    };

    // cotangent 叶子 + 标量 VJP loss（loss = Σ_out Σ(out ⊙ cot)）
    std::vector<std::pair<const GoldenTensor*, Tensor*>> cots;
    Tensor* acc = nullptr;
    for (const GoldenTensor& gt : gm.tensors) {
        if (gt.role != "cotangent") {
            continue;
        }
        Tensor* out_t = find_compare_tensor(io.compares, gt.name);
        if (out_t == nullptr) {
            std::printf("[selftest-backward] C++ 未提供 cotangent 目标张量: %s\n", gt.name.c_str());
            cleanup();
            return 1;
        }
        if (!tensor_matches_ne(out_t, gt)) {
            std::printf("[selftest-backward] cotangent 目标形状不符: %s\n", gt.name.c_str());
            cleanup();
            return 1;
        }
        Tensor* cot = new_tensor_nd(ctx, TYPE_F32, gt.nd, gt.ne.data());
        if (cot == nullptr) {
            std::printf("[selftest-backward] cotangent 叶子创建失败: %s\n", gt.name.c_str());
            cleanup();
            return 1;
        }
        std::snprintf(cot->name, MAX_NAME, "cot:%s", gt.name.c_str());
        Tensor* term = mul(ctx, out_t, cot);
        if (term == nullptr) {
            std::printf("[selftest-backward] mul 建图失败: %s\n", gt.name.c_str());
            cleanup();
            return 1;
        }
        // 每个输出先归约为标量再累加（输出形状/元素数可不同）
        term = sum(ctx, term);
        if (term == nullptr) {
            std::printf("[selftest-backward] sum 建图失败: %s\n", gt.name.c_str());
            cleanup();
            return 1;
        }
        acc = (acc == nullptr) ? term : add(ctx, acc, term);
        if (acc == nullptr) {
            std::printf("[selftest-backward] add 建图失败: %s\n", gt.name.c_str());
            cleanup();
            return 1;
        }
        cots.emplace_back(&gt, cot);
    }
    if (cots.empty()) {
        std::printf("[selftest-backward] manifest 无 cotangent 条目\n");
        cleanup();
        return 1;
    }
    Tensor* loss = acc;
    std::snprintf(loss->name, MAX_NAME, "vjp_loss");
    tensor_set_loss(loss);

    // 前向 + 反向展开（必须在 buffer 分配之前）
    graph_build_forward_expand(ctx, graph, loss);
    graph_build_backward_expand(graph);

    const DeviceProps& dp = dev->props();
    std::printf("[selftest-backward] backend = %s（%s）\n", dp.name.c_str(),
                dp.type == DeviceType::CPU ? "CPU" : "GPU");
    debug_print_alloc(ctx);

    // 大图（如 disc L=12800）全量 ctx 分配会超 Vulkan 4GiB 单缓冲；RVCT_BWD_GRAPH_ALLOC=1
    // 可切换到图级 liveness 分配（默认关闭，保持既有反向对拍行为不变）。
    bool graph_alloc = false;
    {
        char*  env = nullptr;
        size_t env_len = 0;
        _dupenv_s(&env, &env_len, "RVCT_BWD_GRAPH_ALLOC");
        if (env != nullptr) {
            graph_alloc = (env[0] == '1');
            free(env);
        }
    }
    buf = graph_alloc ? buffer_alloc_graph_tensors(graph, dev->default_buffer_type())
                      : buffer_alloc_ctx_tensors(ctx, dev->default_buffer_type());
    if (buf == nullptr) {
        std::printf("[selftest-backward] 计算缓冲分配失败（可用 RVCT_DEBUG_ALLOC=1 查看待分配总量）\n");
        cleanup();
        return 1;
    }

    int64_t loaded = 0;
    for (Tensor* t : params.params().items) {
        if (!w->load(t, t->name, &err)) {
            std::printf("[selftest-backward] 装载失败: %s\n", err.c_str());
            cleanup();
            return 1;
        }
        ++loaded;
    }
    std::printf("[selftest-backward] 参数装载: %lld/%lld\n", (long long) loaded, (long long) n);

    if (!set_inputs(gm, io.inputs, &err)) {
        std::printf("[selftest-backward] %s\n", err.c_str());
        cleanup();
        return 1;
    }
    for (const auto& c : cots) {
        if (!fill_tensor_from_file(c.second, gm.dir, *c.first, &err)) {
            std::printf("[selftest-backward] cotangent 注入失败: %s\n", err.c_str());
            cleanup();
            return 1;
        }
    }
    if (io.prepare && !io.prepare(&err)) {
        std::printf("[selftest-backward] 准备失败: %s\n", err.c_str());
        cleanup();
        return 1;
    }

    const Tensor* unsupported = graph_first_unsupported(graph, dev);
    if (unsupported != nullptr) {
        std::printf("[selftest-backward] 设备不支持节点: %s\n", unsupported->name);
        cleanup();
        return 1;
    }
    graph_reset(graph);
    if (!dev->graph_compute(graph)) {
        std::printf("[selftest-backward] graph_compute 失败\n");
        cleanup();
        return 1;
    }

    // 幂等性自检：同一图重复 compute 必须给出同一前向 loss（训练循环依赖此性质）
    float loss_first = 0.0f;
    {
        bool has_loss = false;
        for (const GoldenTensor& gt : gm.tensors) {
            if (gt.role == "loss") {
                has_loss = true;
            }
        }
        if (has_loss) {
            loss_first = tensor_get_f32_at(loss, 0);
            graph_reset(graph);
            dev->graph_compute(graph);
            const float loss_second = tensor_get_f32_at(loss, 0);
            const double d = std::fabs((double) loss_second - (double) loss_first);
            if (d > 1e-5 * std::max(1.0, std::fabs((double) loss_first))) {
                std::printf("[selftest-backward] [warn] 前向不幂等: loss %.6f -> %.6f（差 %.3e）\n",
                            (double) loss_first, (double) loss_second, d);
            }
        }
    }

    int grad_checked = 0;
    int grad_failed  = 0;
    const bool long_seq = (module == "dec" || module == "synth");
    compare_grads(gm, io, params, long_seq, &grad_checked, &grad_failed);

    int loss_checked = 0;
    int loss_failed  = 0;
    for (const GoldenTensor& gt : gm.tensors) {
        if (gt.role != "loss") {
            continue;
        }
        ++loss_checked;
        std::vector<uint8_t> ref;
        std::string e2;
        if (!read_file_bytes(join_path(gm.dir, gt.file), &ref, &e2) || ref.size() < sizeof(float)) {
            std::printf("[selftest-backward]   loss [FAIL] 参考读取失败: %s\n", e2.c_str());
            ++loss_failed;
            continue;
        }
        const float ref_v = *(const float*) ref.data();
        const float got_v = tensor_get_f32_at(loss, 0);
        const double rel = std::fabs((double) got_v - (double) ref_v) /
                           std::max(1.0, std::fabs((double) ref_v));
        const bool ok = (rel <= kLossTol);
        std::printf("[selftest-backward]   %-46s ref=%.6f got=%.6f rel=%.3e %s\n", "vjp_loss",
                    (double) ref_v, (double) got_v, rel, ok ? "OK" : "FAIL");
        if (!ok) {
            ++loss_failed;
        }
    }

    // 诊断：C++ 侧梯度/loss 落盘（与 Python golden 同命名，供与 F64 参考对拍）
    if (!dump_grads.empty()) {
        int dumped = 0;
        for (const GoldenTensor& gt : gm.tensors) {
            if (gt.role != "grad") {
                continue;
            }
            Tensor* target = nullptr;
            auto it = io.inputs.find(gt.name);
            target = (it != io.inputs.end()) ? it->second : params.find(gt.name.c_str());
            if (target == nullptr || target->grad_acc == nullptr) {
                continue;
            }
            const size_t nbytes = (size_t) tensor_nelements(target->grad_acc) * 4;
            std::vector<uint8_t> bytes(nbytes);
            tensor_get(target->grad_acc, bytes.data(), 0, nbytes);
            const std::string path =
                join_path(dump_grads, module + "." + mangle_name(gt.name) + ".grad.f32.raw");
            std::ofstream f(path, std::ios::binary | std::ios::trunc);
            if (f.is_open()) {
                f.write((const char*) bytes.data(), (std::streamsize) nbytes);
                ++dumped;
            }
        }
        {
            std::ofstream f(join_path(dump_grads, module + ".loss.f32.raw"),
                            std::ios::binary | std::ios::trunc);
            const float lv = tensor_get_f32_at(loss, 0);
            if (f.is_open()) {
                f.write((const char*) &lv, sizeof(lv));
            }
        }
        std::printf("[selftest-backward] C++ 梯度落盘: %d 项 -> %s\n", dumped, dump_grads.c_str());
    }

    // 数值 gradcheck（中心差分；仅参数；CPU 建议）
    int gc_checked = 0;
    int gc_failed  = 0;
    if (!gradcheck_spec.empty()) {
        float gc_eps = kGradcheckEps;
        {
            char*  eps_env = nullptr;
            size_t eps_len = 0;
            _dupenv_s(&eps_env, &eps_len, "RVCT_GRADCHECK_EPS");
            if (eps_env != nullptr) {
                gc_eps = (float) std::atof(eps_env);
                free(eps_env);
            }
        }
        struct GcItem {
            std::string name;
            int64_t idx;
        };
        std::vector<GcItem> items;
        if (gradcheck_spec == "all") {
            for (const GoldenTensor& gt : gm.tensors) {
                if (gt.role != "grad") {
                    continue;
                }
                Tensor* p = params.find(gt.name.c_str());
                if (p == nullptr) {
                    continue;
                }
                items.push_back({gt.name, 0});
                if (tensor_nelements(p) > 1) {
                    items.push_back({gt.name, tensor_nelements(p) / 2});
                }
            }
        } else {
            size_t pos = 0;
            while (pos < gradcheck_spec.size()) {
                const size_t semi = gradcheck_spec.find(';', pos);
                std::string tok = gradcheck_spec.substr(
                    pos, (semi == std::string::npos) ? std::string::npos : semi - pos);
                pos = (semi == std::string::npos) ? gradcheck_spec.size() : semi + 1;
                if (tok.empty()) {
                    continue;
                }
                int64_t idx = 0;
                const size_t colon = tok.find(':');
                if (colon != std::string::npos) {
                    idx  = std::atoll(tok.substr(colon + 1).c_str());
                    tok  = tok.substr(0, colon);
                }
                items.push_back({tok, idx});
            }
        }
        std::printf("[selftest-backward] gradcheck: %zu 项，eps=%.1e，门 rel ≤ %.1e\n", items.size(),
                    (double) gc_eps, kGradcheckTol);
        for (const GcItem& it : items) {
            ++gc_checked;
            Tensor* p = params.find(it.name.c_str());
            if (p == nullptr || p->grad_acc == nullptr) {
                std::printf("[selftest-backward]   gc %-44s [FAIL] 参数或梯度不存在\n",
                            it.name.c_str());
                ++gc_failed;
                continue;
            }
            const int64_t nelem = tensor_nelements(p);
            if (it.idx < 0 || it.idx >= nelem) {
                std::printf("[selftest-backward]   gc %-44s [FAIL] 下标越界（%lld >= %lld）\n",
                            it.name.c_str(), (long long) it.idx, (long long) nelem);
                ++gc_failed;
                continue;
            }
            const float ana  = tensor_get_f32_at(p->grad_acc, it.idx);
            const float orig = tensor_get_f32_at(p, it.idx);
            tensor_set_f32_at(p, it.idx, orig + gc_eps);
            graph_reset(graph);
            dev->graph_compute(graph);
            const float lp = tensor_get_f32_at(loss, 0);
            tensor_set_f32_at(p, it.idx, orig - gc_eps);
            graph_reset(graph);
            dev->graph_compute(graph);
            const float lm = tensor_get_f32_at(loss, 0);
            tensor_set_f32_at(p, it.idx, orig);
            const double num = ((double) lp - (double) lm) / (2.0 * (double) gc_eps);
            const double rel = std::fabs(num - (double) ana) /
                               std::max(1.0, std::max(std::fabs(num), std::fabs((double) ana)));
            const bool ok = (rel <= kGradcheckTol);
            std::printf("[selftest-backward]   gc %-44s [%lld] num=%.6e ana=%.6e rel=%.3e %s\n",
                        it.name.c_str(), (long long) it.idx, num, (double) ana, rel,
                        ok ? "OK" : "FAIL");
            if (!ok) {
                ++gc_failed;
            }
        }
        graph_reset(graph);
        dev->graph_compute(graph);
    }

    const int total_checked = grad_checked + loss_checked + gc_checked;
    const int total_failed  = grad_failed + loss_failed + gc_failed;
    const bool ok = (total_failed == 0 && total_checked > 0);
    const double max_tol = long_seq ? kBackTolLong : kBackTol;
    std::printf("[selftest-backward] 对拍 %s（梯度 %d/%d，loss %d/%d，gradcheck %d/%d；"
                "门 max≤%.0e/rms≤%.0e / loss≤%.0e / gc≤%.0e）\n",
                ok ? "通过" : "失败", grad_checked - grad_failed, grad_checked,
                loss_checked - loss_failed, loss_checked, gc_checked - gc_failed, gc_checked,
                max_tol, kBackRmsTol, kLossTol, kGradcheckTol);

    cleanup();
    return ok ? 0 : 1;
}

} // namespace

const GoldenTensor* GoldenManifest::find_tensor(const std::string& name) const {
    for (const GoldenTensor& t : tensors) {
        if (t.name == name) {
            return &t;
        }
    }
    return nullptr;
}

int64_t GoldenManifest::param_int(const char* key, int64_t def) const {
    const common::JsonValue* v = params.find(key);
    return (v != nullptr && v->is_number()) ? v->as_int(def) : def;
}

std::string GoldenManifest::param_str(const char* key, const char* def) const {
    const common::JsonValue* v = params.find(key);
    return (v != nullptr && v->is_string()) ? v->as_string() : std::string(def);
}

void GoldenManifest::param_int_array(const char* key, std::vector<int64_t>* out) const {
    out->clear();
    const common::JsonValue* v = params.find(key);
    if (v == nullptr || !v->is_array()) {
        return;
    }
    for (size_t i = 0; i < v->size(); ++i) {
        out->push_back(v->at(i).as_int(0));
    }
}

bool golden_manifest_load(const std::string& dir, const std::string& module, GoldenManifest* out,
                          std::string* err) {
    if (out == nullptr) {
        if (err != nullptr) {
            *err = "输出为空";
        }
        return false;
    }
    const std::string p1 = join_path(dir, "manifest." + module + ".json");
    const std::string p2 = join_path(dir, "manifest.json");
    common::JsonValue root;
    std::string e1;
    std::string e2;
    if (!common::json_read_file(p1, &root, &e1)) {
        if (!common::json_read_file(p2, &root, &e2)) {
            if (err != nullptr) {
                *err = "无法读取 golden manifest: " + p1 + "（" + e1 + "）；" + p2 + "（" + e2 + "）";
            }
            return false;
        }
    }
    const common::JsonValue* m = root.find("module");
    const std::string got = (m != nullptr) ? m->as_string() : std::string();
    if (got != module) {
        if (err != nullptr) {
            *err = "manifest 模块不匹配: 期望 " + module + "，实际 " +
                   (got.empty() ? std::string("(缺失)") : got);
        }
        return false;
    }

    GoldenManifest g;
    g.module = module;
    g.dir    = dir;
    const common::JsonValue* ts = root.find("tensors");
    if (ts == nullptr || !ts->is_array()) {
        if (err != nullptr) {
            *err = "manifest.tensors 缺失或非数组";
        }
        return false;
    }
    for (size_t i = 0; i < ts->size(); ++i) {
        const common::JsonValue& t = ts->at(i);
        const common::JsonValue* name  = t.find("name");
        const common::JsonValue* file  = t.find("file");
        const common::JsonValue* dtype = t.find("dtype");
        const common::JsonValue* role  = t.find("role");
        const common::JsonValue* shape = t.find("shape");
        if (name == nullptr || file == nullptr || dtype == nullptr || role == nullptr ||
            shape == nullptr || !shape->is_array() || shape->size() < 1 || shape->size() > 4) {
            if (err != nullptr) {
                *err = "manifest tensor 条目字段缺失/非法（index " + std::to_string(i) + "）";
            }
            return false;
        }
        GoldenTensor gt;
        gt.name  = name->as_string();
        gt.file  = file->as_string();
        gt.dtype = dtype->as_string();
        gt.role  = role->as_string();
        gt.nd    = (int) shape->size();
        for (size_t j = 0; j < shape->size(); ++j) {
            gt.ne[(size_t) (gt.nd - 1 - (int) j)] = shape->at(j).as_int(1);
        }
        g.tensors.push_back(std::move(gt));
    }
    const common::JsonValue* params = root.find("params");
    if (params != nullptr && params->is_object()) {
        g.params = *params;
    }
    *out = std::move(g);
    return true;
}

Tensor* SelftestBuildIO::find_input(const std::string& name) const {
    auto it = inputs.find(name);
    return (it == inputs.end()) ? nullptr : it->second;
}

void SelftestBuildIO::add_compare(const std::string& name, Tensor* t) {
    compares.emplace_back(name, t);
}

bool build_selftest_emb_g(SelftestBuildIO& io, std::string* err) {
    Tensor* sid = io.find_input("sid");
    if (sid == nullptr) {
        if (err != nullptr) {
            *err = "缺少 golden 输入: sid";
        }
        return false;
    }
    Tensor* g = emb_g_forward(io.ctx, *io.params, sid, err);
    if (g == nullptr) {
        return false;
    }
    io.add_compare("g", cont(io.ctx, g));
    return true;
}

bool build_selftest_wn(SelftestBuildIO& io, std::string* err) {
    WnConfig cfg;
    cfg.prefix          = io.golden->param_str("prefix", "enc_q.enc");
    cfg.hidden_channels = io.golden->param_int("hidden_channels");
    cfg.kernel_size     = io.golden->param_int("kernel_size");
    cfg.n_layers        = io.golden->param_int("n_layers");
    cfg.gin_channels    = io.golden->param_int("gin_channels");
    cfg.dilation_rate   = io.golden->param_int("dilation_rate", 1);

    WnModule wn;
    if (!wn_load(*io.params, cfg, &wn, err)) {
        return false;
    }
    Tensor* x      = io.find_input("x");
    Tensor* x_mask = io.find_input("x_mask");
    Tensor* g      = io.find_input("g");
    if (x == nullptr || x_mask == nullptr) {
        if (err != nullptr) {
            *err = "缺少 golden 输入: x/x_mask";
        }
        return false;
    }
    WnForwardCaptures caps;
    Tensor* y = wn_forward(io.ctx, wn, x, x_mask, g, &caps);
    if (y == nullptr) {
        if (err != nullptr) {
            *err = "wn_forward 失败";
        }
        return false;
    }
    if (caps.cond != nullptr) {
        io.add_compare("cond", cont(io.ctx, caps.cond));
    }
    if (caps.x_in0 != nullptr) {
        io.add_compare("x_in0", cont(io.ctx, caps.x_in0));
    }
    if (caps.acts0 != nullptr) {
        io.add_compare("acts0", cont(io.ctx, caps.acts0));
    }
    io.add_compare("y", cont(io.ctx, y));
    return true;
}

bool build_selftest_resblock1(SelftestBuildIO& io, std::string* err) {
    ResBlock1Config cfg;
    cfg.prefix      = io.golden->param_str("prefix", "dec.resblocks.0");
    cfg.channels    = io.golden->param_int("channels");
    cfg.kernel_size = io.golden->param_int("kernel_size", 3);
    std::vector<int64_t> dils;
    io.golden->param_int_array("dilations", &dils);
    cfg.dilations = dils;

    ResBlock1Module rb;
    if (!resblock1_load(*io.params, cfg, &rb, err)) {
        return false;
    }
    Tensor* x      = io.find_input("x");
    Tensor* x_mask = io.find_input("x_mask");
    if (x == nullptr) {
        if (err != nullptr) {
            *err = "缺少 golden 输入: x";
        }
        return false;
    }
    ResBlock1Captures caps;
    Tensor* y = resblock1_forward(io.ctx, rb, x, x_mask, &caps);
    if (y == nullptr) {
        if (err != nullptr) {
            *err = "resblock1_forward 失败";
        }
        return false;
    }
    if (caps.t0 != nullptr) {
        io.add_compare("t0", cont(io.ctx, caps.t0));
    }
    io.add_compare("y", cont(io.ctx, y));
    return true;
}

bool build_selftest_dec(SelftestBuildIO& io, std::string* err) {
    DecConfig cfg;
    cfg.prefix                   = io.golden->param_str("prefix", "dec");
    cfg.in_channels              = io.golden->param_int("in_channels");
    cfg.upsample_initial_channel = io.golden->param_int("upsample_initial_channel");
    cfg.gin_channels             = io.golden->param_int("gin_channels");
    io.golden->param_int_array("upsample_rates", &cfg.upsample_rates);
    io.golden->param_int_array("upsample_kernel_sizes", &cfg.upsample_kernel_sizes);
    io.golden->param_int_array("resblock_kernel_sizes", &cfg.resblock_kernel_sizes);
    io.golden->param_int_array("resblock_dilation_sizes", &cfg.resblock_dilation_sizes);
    const int64_t sample_rate = io.golden->param_int("sample_rate", 40000);

    DecModule dec;
    if (!dec_load(*io.params, cfg, &dec, err)) {
        return false;
    }

    Tensor* z          = io.find_input("z");
    Tensor* pitchf     = io.find_input("pitchf");
    Tensor* g          = io.find_input("g");
    Tensor* sine_noise = io.find_input("sine_noise");
    if (z == nullptr || pitchf == nullptr || sine_noise == nullptr) {
        if (err != nullptr) {
            *err = "缺少 golden 输入: z/pitchf/sine_noise";
        }
        return false;
    }

    // 主机侧 SineGen 依赖输入数据（runner 注入之后才能读）→ 先建叶子张量，再挂 prepare 回调
    int64_t upp = 1;
    for (int64_t u : cfg.upsample_rates) {
        upp *= u;
    }
    const int64_t frames = z->ne[0];
    const int64_t batch  = z->ne[2];
    const int64_t total  = frames * upp;
    Tensor* sine_src = new_tensor_3d(io.ctx, TYPE_F32, 1, total, batch);
    if (sine_src == nullptr) {
        if (err != nullptr) {
            *err = "创建 sine_source 张量失败";
        }
        return false;
    }
    tensor_set_name(sine_src, "sine_source");

    io.prepare = [pitchf, sine_noise, sine_src, frames, batch, total, sample_rate, upp](
                     std::string* perr) -> bool {
        std::vector<float> pf(static_cast<size_t>(frames * batch));
        std::vector<float> noise(static_cast<size_t>(total * batch));
        std::vector<float> src(static_cast<size_t>(total * batch));
        tensor_get(pitchf, pf.data(), 0, pf.size() * sizeof(float));
        tensor_get(sine_noise, noise.data(), 0, noise.size() * sizeof(float));
        for (int64_t b = 0; b < batch; ++b) {
            std::vector<float> s =
                sinegen_generate(pf.data() + b * frames, frames, static_cast<int>(sample_rate),
                                 static_cast<int>(upp), noise.data() + b * total, total, 1234);
            if (static_cast<int64_t>(s.size()) != total) {
                if (perr != nullptr) {
                    *perr = "SineGen 输出长度不符";
                }
                return false;
            }
            std::memcpy(src.data() + b * total, s.data(),
                        static_cast<size_t>(total) * sizeof(float));
        }
        tensor_set(sine_src, src.data(), 0, src.size() * sizeof(float));
        return true;
    };

    DecCaptures caps;
    Tensor* o = dec_forward(io.ctx, dec, z, sine_src, g, &caps);
    if (o == nullptr) {
        if (err != nullptr) {
            *err = "dec_forward 失败";
        }
        return false;
    }
    io.add_compare("sine_source", cont(io.ctx, sine_src));
    if (caps.sine_merged != nullptr) {
        // golden 布局 [B,L,1] = ne [1,L,B]（C++ 内部用 [L,1,B]，同内存重排视图）
        io.add_compare("sine_merged",
                       reshape_3d(io.ctx, cont(io.ctx, caps.sine_merged), 1, total, batch));
    }
    if (caps.cond != nullptr) {
        io.add_compare("cond", cont(io.ctx, caps.cond));
    }
    if (caps.conv_pre != nullptr) {
        io.add_compare("conv_pre", cont(io.ctx, caps.conv_pre));
    }
    if (caps.x_cond != nullptr) {
        io.add_compare("x_cond", cont(io.ctx, caps.x_cond));
    }
    for (size_t i = 0; i < caps.ups.size(); ++i) {
        io.add_compare("up" + std::to_string(i), cont(io.ctx, caps.ups[i]));
    }
    io.add_compare("o", cont(io.ctx, o));
    return true;
}

bool build_selftest_flow(SelftestBuildIO& io, std::string* err) {
    FlowConfig cfg;
    cfg.prefix          = io.golden->param_str("prefix", "flow");
    cfg.channels        = io.golden->param_int("channels");
    cfg.hidden_channels = io.golden->param_int("hidden_channels");
    cfg.kernel_size     = io.golden->param_int("kernel_size", 5);
    cfg.dilation_rate   = io.golden->param_int("dilation_rate", 1);
    cfg.n_layers        = io.golden->param_int("n_layers", 3);
    cfg.gin_channels    = io.golden->param_int("gin_channels");
    cfg.n_flows         = io.golden->param_int("n_flows", 4);

    FlowModule flow;
    if (!flow_load(*io.params, cfg, &flow, err)) {
        return false;
    }
    Tensor* x      = io.find_input("x");
    Tensor* x_mask = io.find_input("x_mask");
    Tensor* g      = io.find_input("g");
    if (x == nullptr || x_mask == nullptr || g == nullptr) {
        if (err != nullptr) {
            *err = "缺少 golden 输入: x/x_mask/g";
        }
        return false;
    }

    // Flip 置换矩阵：建叶子，buffer 分配后由 prepare 回调填充
    Tensor* perm = new_tensor_2d(io.ctx, TYPE_F32, cfg.channels, cfg.channels);
    if (perm == nullptr) {
        if (err != nullptr) {
            *err = "创建 flip 置换矩阵失败";
        }
        return false;
    }
    tensor_set_name(perm, "flip_perm");
    io.prepare = [perm](std::string* perr) -> bool { return flow_fill_flip_perm(perm, perr); };

    FlowCaptures caps;
    Tensor* y = flow_forward(io.ctx, flow, x, x_mask, g, perm, &caps);
    if (y == nullptr) {
        if (err != nullptr) {
            *err = "flow_forward 失败";
        }
        return false;
    }
    for (size_t i = 0; i < caps.pre.size(); ++i) {
        io.add_compare("pre" + std::to_string(i), cont(io.ctx, caps.pre[i]));
    }
    for (size_t i = 0; i < caps.enc.size(); ++i) {
        io.add_compare("enc" + std::to_string(i), cont(io.ctx, caps.enc[i]));
    }
    for (size_t i = 0; i < caps.stats.size(); ++i) {
        io.add_compare("stats" + std::to_string(i), cont(io.ctx, caps.stats[i]));
    }
    io.add_compare("y", cont(io.ctx, y));
    return true;
}

bool build_selftest_enc_p(SelftestBuildIO& io, std::string* err) {
    EncPConfig cfg;
    cfg.prefix          = io.golden->param_str("prefix", "enc_p");
    cfg.in_channels     = io.golden->param_int("in_channels");
    cfg.hidden_channels = io.golden->param_int("hidden_channels");
    cfg.filter_channels = io.golden->param_int("filter_channels");
    cfg.n_heads         = io.golden->param_int("n_heads");
    cfg.n_layers        = io.golden->param_int("n_layers");
    cfg.kernel_size     = io.golden->param_int("kernel_size", 3);
    cfg.window_size     = io.golden->param_int("window_size", 10);
    cfg.out_channels    = io.golden->param_int("out_channels");

    EncPModule epm;
    if (!enc_p_load(*io.params, cfg, &epm, err)) {
        return false;
    }
    Tensor* phone   = io.find_input("phone");
    Tensor* pitch   = io.find_input("pitch");
    Tensor* x_mask  = io.find_input("x_mask");
    if (phone == nullptr || x_mask == nullptr) {
        if (err != nullptr) {
            *err = "缺少 golden 输入: phone/x_mask";
        }
        return false;
    }
    const int64_t T = x_mask->ne[0];
    const int64_t B = x_mask->ne[2];

    // 注意力 valid 掩码 [T,T,1,B]（0/1）：依赖 x_mask 数据，buffer 分配后由 prepare 填充
    Tensor* attn_valid = new_tensor_4d(io.ctx, TYPE_F32, T, T, 1, B);
    if (attn_valid == nullptr) {
        if (err != nullptr) {
            *err = "创建 attn_valid 张量失败";
        }
        return false;
    }
    tensor_set_name(attn_valid, "attn_valid");
    io.prepare = [x_mask, attn_valid, T, B](std::string* perr) -> bool {
        std::vector<float> xm((size_t) (T * B));
        tensor_get(x_mask, xm.data(), 0, xm.size() * sizeof(float));
        float* p = tensor_data_f32(attn_valid);
        if (p == nullptr) {
            if (perr != nullptr) {
                *perr = "attn_valid 未分配（需在 buffer 分配后填充）";
            }
            return false;
        }
        for (int64_t b = 0; b < B; ++b) {
            for (int64_t tq = 0; tq < T; ++tq) {
                const bool qv = xm[(size_t) (tq + T * b)] > 0.5f;
                for (int64_t tk = 0; tk < T; ++tk) {
                    const bool kv = xm[(size_t) (tk + T * b)] > 0.5f;
                    p[(size_t) (tk + T * tq + T * T * b)] = (qv && kv) ? 1.0f : 0.0f;
                }
            }
        }
        return true;
    };

    io.add_compare("attn_valid", attn_valid);

    EncPCaptures caps;
    EncPOutputs out = enc_p_forward(io.ctx, epm, phone, pitch, x_mask, attn_valid, &caps);
    if (out.m_p == nullptr || out.logs_p == nullptr) {
        if (err != nullptr) {
            *err = "enc_p_forward 失败";
        }
        return false;
    }

    if (caps.emb != nullptr) {
        io.add_compare("emb", cont(io.ctx, caps.emb));
    }
    if (caps.x != nullptr) {
        io.add_compare("x", cont(io.ctx, caps.x));
    }
    for (size_t i = 0; i < caps.q.size(); ++i) {
        const std::string idx = std::to_string(i);
        io.add_compare("q" + idx, cont(io.ctx, caps.q[i]));
        io.add_compare("k" + idx, cont(io.ctx, caps.k[i]));
        io.add_compare("v" + idx, cont(io.ctx, caps.v[i]));
        io.add_compare("rel_logits" + idx, cont(io.ctx, caps.rel_logits[i]));
        io.add_compare("scores_local" + idx, cont(io.ctx, caps.scores_local[i]));
        io.add_compare("scores" + idx, cont(io.ctx, caps.scores[i]));
        io.add_compare("scores_masked" + idx, cont(io.ctx, caps.scores_masked[i]));
        io.add_compare("p_attn" + idx, cont(io.ctx, caps.p_attn[i]));
        io.add_compare("attn_out" + idx, cont(io.ctx, caps.attn_out[i]));
        io.add_compare("merged" + idx, cont(io.ctx, caps.merged[i]));
        io.add_compare("attn_y" + idx, cont(io.ctx, caps.attn_y[i]));
        io.add_compare("norm1_" + idx, cont(io.ctx, caps.norm1[i]));
        io.add_compare("ffn" + idx, cont(io.ctx, caps.ffn[i]));
        io.add_compare("norm2_" + idx, cont(io.ctx, caps.norm2[i]));
    }
    if (caps.encoder_out != nullptr) {
        io.add_compare("encoder_out", cont(io.ctx, caps.encoder_out));
    }
    if (caps.proj != nullptr) {
        io.add_compare("proj", cont(io.ctx, caps.proj));
    }
    io.add_compare("m_p", cont(io.ctx, out.m_p));
    io.add_compare("logs_p", cont(io.ctx, out.logs_p));
    return true;
}

bool build_selftest_enc_q(SelftestBuildIO& io, std::string* err) {
    EncQConfig cfg;
    cfg.prefix          = io.golden->param_str("prefix", "enc_q");
    cfg.in_channels     = io.golden->param_int("in_channels");
    cfg.out_channels    = io.golden->param_int("out_channels");
    cfg.hidden_channels = io.golden->param_int("hidden_channels");
    cfg.kernel_size     = io.golden->param_int("kernel_size", 5);
    cfg.dilation_rate   = io.golden->param_int("dilation_rate", 1);
    cfg.n_layers        = io.golden->param_int("n_layers", 16);
    cfg.gin_channels    = io.golden->param_int("gin_channels");

    EncQModule eqm;
    if (!enc_q_load(*io.params, cfg, &eqm, err)) {
        return false;
    }
    Tensor* spec   = io.find_input("spec");
    Tensor* x_mask = io.find_input("x_mask");
    Tensor* g      = io.find_input("g");
    Tensor* noise  = io.find_input("q_noise");
    if (spec == nullptr || x_mask == nullptr || noise == nullptr) {
        if (err != nullptr) {
            *err = "缺少 golden 输入: spec/x_mask/q_noise";
        }
        return false;
    }

    EncQCaptures caps;
    EncQOutputs out = enc_q_forward(io.ctx, eqm, spec, x_mask, g, noise, &caps);
    if (out.m_q == nullptr || out.logs_q == nullptr || out.z == nullptr) {
        if (err != nullptr) {
            *err = "enc_q_forward 失败";
        }
        return false;
    }
    if (caps.pre != nullptr) {
        io.add_compare("pre", cont(io.ctx, caps.pre));
    }
    if (caps.cond != nullptr) {
        io.add_compare("cond", cont(io.ctx, caps.cond));
    }
    if (caps.x_in0 != nullptr) {
        io.add_compare("x_in0", cont(io.ctx, caps.x_in0));
    }
    if (caps.acts0 != nullptr) {
        io.add_compare("acts0", cont(io.ctx, caps.acts0));
    }
    if (caps.enc != nullptr) {
        io.add_compare("enc", cont(io.ctx, caps.enc));
    }
    if (caps.stats != nullptr) {
        io.add_compare("stats", cont(io.ctx, caps.stats));
    }
    io.add_compare("m_q", cont(io.ctx, out.m_q));
    io.add_compare("logs_q", cont(io.ctx, out.logs_q));
    io.add_compare("z", cont(io.ctx, out.z));
    return true;
}

bool build_selftest_disc(SelftestBuildIO& io, std::string* err) {
    MpdConfig cfg;
    cfg.prefix = io.golden->param_str("prefix", "discriminators");
    io.golden->param_int_array("periods", &cfg.periods);
    cfg.kernel_size = io.golden->param_int("kernel_size", 5);
    cfg.stride      = io.golden->param_int("stride", 3);
    if (cfg.periods.empty()) {
        if (err != nullptr) {
            *err = "golden params.periods 缺失（需用 --module disc 重新生成）";
        }
        return false;
    }

    MpdModule mpd;
    if (!mpd_load(*io.params, cfg, &mpd, err)) {
        return false;
    }
    Tensor* wave     = io.find_input("wave");
    Tensor* wave_hat = io.find_input("wave_hat");
    if (wave == nullptr || wave_hat == nullptr) {
        if (err != nullptr) {
            *err = "缺少 golden 输入: wave/wave_hat";
        }
        return false;
    }

    MpdForward f = mpd_forward(io.ctx, mpd, wave, wave_hat);
    if (f.r.empty() || f.r.size() != f.g.size()) {
        if (err != nullptr) {
            *err = "mpd_forward 失败";
        }
        return false;
    }
    for (size_t i = 0; i < f.r.size(); ++i) {
        const std::string idx = std::to_string(i);
        io.add_compare("d" + idx + "_out_r", cont(io.ctx, f.r[i].out));
        io.add_compare("d" + idx + "_out_g", cont(io.ctx, f.g[i].out));
        for (size_t j = 0; j < f.r[i].fmap.size(); ++j) {
            const std::string jdx = std::to_string(j);
            io.add_compare("d" + idx + "_fmap" + jdx + "_r", cont(io.ctx, f.r[i].fmap[j]));
            io.add_compare("d" + idx + "_fmap" + jdx + "_g", cont(io.ctx, f.g[i].fmap[j]));
        }
    }
    return true;
}

bool build_selftest_synth(SelftestBuildIO& io, std::string* err) {
    SynthConfig cfg;
    cfg.segment_size = io.golden->param_int("segment_size");
    cfg.frames       = io.golden->param_int("frames");
    cfg.batch        = io.golden->param_int("batch", 1);

    cfg.enc_p.prefix          = io.golden->param_str("enc_p_prefix", "enc_p");
    cfg.enc_p.in_channels     = io.golden->param_int("enc_p_in");
    cfg.enc_p.hidden_channels = io.golden->param_int("enc_p_hidden");
    cfg.enc_p.filter_channels = io.golden->param_int("enc_p_filter");
    cfg.enc_p.n_heads         = io.golden->param_int("enc_p_heads");
    cfg.enc_p.n_layers        = io.golden->param_int("enc_p_layers");
    cfg.enc_p.kernel_size     = io.golden->param_int("enc_p_kernel", 3);
    cfg.enc_p.window_size     = io.golden->param_int("enc_p_window", 10);
    cfg.enc_p.out_channels    = io.golden->param_int("enc_p_out");

    cfg.enc_q.prefix          = io.golden->param_str("enc_q_prefix", "enc_q");
    cfg.enc_q.in_channels     = io.golden->param_int("enc_q_in");
    cfg.enc_q.out_channels    = io.golden->param_int("enc_q_out");
    cfg.enc_q.hidden_channels = io.golden->param_int("enc_q_hidden");
    cfg.enc_q.kernel_size     = io.golden->param_int("enc_q_kernel", 5);
    cfg.enc_q.dilation_rate   = io.golden->param_int("enc_q_dilation", 1);
    cfg.enc_q.n_layers        = io.golden->param_int("enc_q_layers", 16);
    cfg.enc_q.gin_channels    = io.golden->param_int("enc_q_gin");

    cfg.flow.prefix          = io.golden->param_str("flow_prefix", "flow");
    cfg.flow.channels        = io.golden->param_int("flow_channels");
    cfg.flow.hidden_channels = io.golden->param_int("flow_hidden");
    cfg.flow.kernel_size     = io.golden->param_int("flow_kernel", 5);
    cfg.flow.dilation_rate   = io.golden->param_int("flow_dilation", 1);
    cfg.flow.n_layers        = io.golden->param_int("flow_layers", 3);
    cfg.flow.gin_channels    = io.golden->param_int("flow_gin");
    cfg.flow.n_flows         = io.golden->param_int("flow_n_flows", 4);

    cfg.dec.prefix                   = io.golden->param_str("dec_prefix", "dec");
    cfg.dec.in_channels              = io.golden->param_int("dec_in");
    cfg.dec.upsample_initial_channel = io.golden->param_int("dec_up_init");
    cfg.dec.gin_channels             = io.golden->param_int("dec_gin");
    io.golden->param_int_array("dec_upsample_rates", &cfg.dec.upsample_rates);
    io.golden->param_int_array("dec_upsample_kernel_sizes", &cfg.dec.upsample_kernel_sizes);
    io.golden->param_int_array("dec_resblock_kernel_sizes", &cfg.dec.resblock_kernel_sizes);
    io.golden->param_int_array("dec_resblock_dilation_sizes", &cfg.dec.resblock_dilation_sizes);
    const int64_t sample_rate = io.golden->param_int("dec_sample_rate", 40000);
    const int64_t upp         = io.golden->param_int("upp", 400);

    SynthModules mods;
    if (!synth_load(*io.params, cfg, &mods, err)) {
        return false;
    }

    Tensor* sid     = io.find_input("sid");
    Tensor* phone   = io.find_input("phone");
    Tensor* pitch   = io.find_input("pitch");
    Tensor* x_mask  = io.find_input("x_mask");
    Tensor* spec    = io.find_input("spec");
    Tensor* y_mask  = io.find_input("y_mask");
    Tensor* q_noise = io.find_input("q_noise");
    Tensor* pitchf  = io.find_input("pitchf");
    Tensor* ids     = io.find_input("ids_slice");
    Tensor* n_full  = io.find_input("sine_noise_full");
    Tensor* n_slice = io.find_input("sine_noise_slice");
    if (sid == nullptr || phone == nullptr || pitch == nullptr || x_mask == nullptr ||
        spec == nullptr || y_mask == nullptr || q_noise == nullptr || pitchf == nullptr ||
        ids == nullptr || n_full == nullptr || n_slice == nullptr) {
        if (err != nullptr) {
            *err = "缺少 golden 输入: sid/phone/pitch/x_mask/spec/y_mask/q_noise/pitchf/"
                   "ids_slice/sine_noise_full/sine_noise_slice";
        }
        return false;
    }

    const int64_t T   = x_mask->ne[0];
    const int64_t B   = x_mask->ne[2];
    const int64_t seg = cfg.segment_size;
    const int64_t total_full  = T * upp;
    const int64_t total_slice = seg * upp;

    // 叶子（依赖输入/分配后填充）：attn_valid / flip_perm / slice_mat / 切片 pitchf / 两条 SineGen 源
    Tensor* attn_valid = new_tensor_4d(io.ctx, TYPE_F32, T, T, 1, B);
    Tensor* perm       = new_tensor_2d(io.ctx, TYPE_F32, cfg.flow.channels, cfg.flow.channels);
    Tensor* slice_mat  = new_tensor_3d(io.ctx, TYPE_F32, T, seg, B);
    Tensor* pf_slice   = new_tensor_2d(io.ctx, TYPE_F32, seg, B);
    Tensor* sine_full  = new_tensor_3d(io.ctx, TYPE_F32, 1, total_full, B);
    Tensor* sine_slice = new_tensor_3d(io.ctx, TYPE_F32, 1, total_slice, B);
    if (attn_valid == nullptr || perm == nullptr || slice_mat == nullptr || pf_slice == nullptr ||
        sine_full == nullptr || sine_slice == nullptr) {
        if (err != nullptr) {
            *err = "创建 synth 叶子张量失败";
        }
        return false;
    }
    tensor_set_name(attn_valid, "attn_valid");
    tensor_set_name(perm, "synth_flip_perm");
    tensor_set_name(slice_mat, "synth_slice_mat");
    tensor_set_name(pf_slice, "synth_pitchf_slice");
    tensor_set_name(sine_full, "synth_sine_source_full");
    tensor_set_name(sine_slice, "synth_sine_source_slice");

    io.prepare = [&, attn_valid, perm, slice_mat, pf_slice, sine_full, sine_slice, x_mask, pitchf,
                  ids, n_full, n_slice, T, B, seg, total_full, total_slice, sample_rate, upp](
                     std::string* perr) -> bool {
        if (!flow_fill_flip_perm(perm, perr)) {
            return false;
        }
        // attn_valid = x_mask 外积（0/1；复刻 torch masked_fill 替换语义）
        std::vector<float> xm((size_t) (T * B));
        tensor_get(x_mask, xm.data(), 0, xm.size() * sizeof(float));
        float* pa = tensor_data_f32(attn_valid);
        if (pa == nullptr) {
            if (perr != nullptr) {
                *perr = "attn_valid 未分配（需在 buffer 分配后填充）";
            }
            return false;
        }
        for (int64_t b = 0; b < B; ++b) {
            for (int64_t tq = 0; tq < T; ++tq) {
                const bool qv = xm[(size_t) (tq + T * b)] > 0.5f;
                for (int64_t tk = 0; tk < T; ++tk) {
                    const bool kv = xm[(size_t) (tk + T * b)] > 0.5f;
                    pa[(size_t) (tk + T * tq + T * T * b)] = (qv && kv) ? 1.0f : 0.0f;
                }
            }
        }
        // 固定 ids → slice_mat（0/1）与切片 pitchf（等价 torch slice_segments2）
        std::vector<int32_t> idv((size_t) B);
        tensor_get(ids, idv.data(), 0, idv.size() * sizeof(int32_t));
        std::vector<float> pf((size_t) (T * B));
        tensor_get(pitchf, pf.data(), 0, pf.size() * sizeof(float));
        std::vector<float> pfs((size_t) (seg * B), 0.0f);
        float* pm = tensor_data_f32(slice_mat);
        if (pm == nullptr) {
            if (perr != nullptr) {
                *perr = "slice_mat 未分配";
            }
            return false;
        }
        std::memset(pm, 0, sizeof(float) * (size_t) (T * seg * B));
        for (int64_t b = 0; b < B; ++b) {
            const int64_t id = idv[(size_t) b];
            if (id < 0 || id + seg > T) {
                if (perr != nullptr) {
                    *perr = "ids_slice 越界";
                }
                return false;
            }
            for (int64_t s = 0; s < seg; ++s) {
                pm[(size_t) (id + s + T * s + T * seg * b)] = 1.0f;
                pfs[(size_t) (s + seg * b)] = pf[(size_t) (id + s + T * b)];
            }
        }
        tensor_set(pf_slice, pfs.data(), 0, pfs.size() * sizeof(float));

        // SineGen（整段 / 切片）：噪声来自 golden 落盘，两侧同源
        std::vector<float> nf((size_t) (total_full * B));
        std::vector<float> ns((size_t) (total_slice * B));
        tensor_get(n_full, nf.data(), 0, nf.size() * sizeof(float));
        tensor_get(n_slice, ns.data(), 0, ns.size() * sizeof(float));
        std::vector<float> sf((size_t) (total_full * B));
        std::vector<float> ss((size_t) (total_slice * B));
        for (int64_t b = 0; b < B; ++b) {
            std::vector<float> f0f((size_t) T);
            for (int64_t t = 0; t < T; ++t) {
                f0f[(size_t) t] = pf[(size_t) (t + T * b)];
            }
            std::vector<float> s = sinegen_generate(f0f.data(), T, (int) sample_rate, (int) upp,
                                                    nf.data() + (size_t) b * total_full, total_full,
                                                    1234);
            if ((int64_t) s.size() != total_full) {
                if (perr != nullptr) {
                    *perr = "SineGen（整段）输出长度不符";
                }
                return false;
            }
            std::memcpy(sf.data() + (size_t) b * total_full, s.data(),
                        sizeof(float) * (size_t) total_full);

            std::vector<float> f0s((size_t) seg);
            for (int64_t t = 0; t < seg; ++t) {
                f0s[(size_t) t] = pfs[(size_t) (t + seg * b)];
            }
            std::vector<float> s2 = sinegen_generate(f0s.data(), seg, (int) sample_rate, (int) upp,
                                                     ns.data() + (size_t) b * total_slice,
                                                     total_slice, 1234);
            if ((int64_t) s2.size() != total_slice) {
                if (perr != nullptr) {
                    *perr = "SineGen（切片）输出长度不符";
                }
                return false;
            }
            std::memcpy(ss.data() + (size_t) b * total_slice, s2.data(),
                        sizeof(float) * (size_t) total_slice);
        }
        tensor_set(sine_full, sf.data(), 0, sf.size() * sizeof(float));
        tensor_set(sine_slice, ss.data(), 0, ss.size() * sizeof(float));
        return true;
    };

    SynthInputs in;
    in.sid               = sid;
    in.phone             = phone;
    in.pitch             = pitch;
    in.x_mask            = x_mask;
    in.attn_valid        = attn_valid;
    in.spec              = spec;
    in.y_mask            = y_mask;
    in.q_noise           = q_noise;
    in.flip_perm         = perm;
    in.slice_mat         = slice_mat;
    in.sine_source_full  = sine_full;
    in.sine_source_slice = sine_slice;

    SynthOutputs out = synth_forward(io.ctx, *io.params, mods, in);
    if (out.o_full == nullptr || out.o_slice == nullptr) {
        if (err != nullptr) {
            *err = "synth_forward 失败";
        }
        return false;
    }

    io.add_compare("g", cont(io.ctx, out.g));
    io.add_compare("m_p", cont(io.ctx, out.m_p));
    io.add_compare("logs_p", cont(io.ctx, out.logs_p));
    io.add_compare("z", cont(io.ctx, out.z));
    io.add_compare("m_q", cont(io.ctx, out.m_q));
    io.add_compare("logs_q", cont(io.ctx, out.logs_q));
    io.add_compare("z_p", cont(io.ctx, out.z_p));
    io.add_compare("attn_valid", attn_valid);
    io.add_compare("full_sine_source", cont(io.ctx, sine_full));
    if (out.dec_full_caps.sine_merged != nullptr) {
        io.add_compare("full_sine_merged",
                       reshape_3d(io.ctx, cont(io.ctx, out.dec_full_caps.sine_merged), 1,
                                  total_full, B));
    }
    io.add_compare("full_o", cont(io.ctx, out.o_full));
    io.add_compare("slice_pitchf", pf_slice);
    io.add_compare("slice_z_slice", cont(io.ctx, out.z_slice));
    io.add_compare("slice_sine_source", cont(io.ctx, sine_slice));
    if (out.dec_slice_caps.sine_merged != nullptr) {
        io.add_compare("slice_sine_merged",
                       reshape_3d(io.ctx, cont(io.ctx, out.dec_slice_caps.sine_merged), 1,
                                  total_slice, B));
    }
    io.add_compare("slice_o", cont(io.ctx, out.o_slice));
    return true;
}

int run_forward_selftest(const std::string& weights_path, const std::string& golden_dir,
                         const std::string& module, Device* dev) {
    if (module == "all") {
        // disc 的权重在独立的 D 文件（f0D40k）→ 不并入 all，需单独 --module disc 指定
        static const char* k_modules[] = {"emb_g", "wn", "resblock1", "dec", "flow", "enc_p",
                                          "enc_q", "synth"};
        int rc = 0;
        for (const char* m : k_modules) {
            const int r = run_single(weights_path, golden_dir, m, dev);
            if (r != 0) {
                rc = 1;
            }
        }
        return rc;
    }
    return run_single(weights_path, golden_dir, module, dev);
}

int run_backward_selftest(const std::string& weights_path, const std::string& golden_dir,
                          const std::string& module, const std::string& gradcheck,
                          const std::string& dump_grads, Device* dev) {
    if (module == "all") {
        // 同前向：disc 的权重在独立 D 文件，需单独 --module disc 指定
        static const char* k_modules[] = {"emb_g", "wn", "resblock1", "dec", "flow", "enc_p",
                                          "enc_q", "synth"};
        int rc = 0;
        for (const char* m : k_modules) {
            const int r = run_single_backward(weights_path, golden_dir, m, gradcheck, dump_grads, dev);
            if (r != 0) {
                rc = 1;
            }
        }
        return rc;
    }
    return run_single_backward(weights_path, golden_dir, module, gradcheck, dump_grads, dev);
}

} // namespace model
} // namespace rvc_train
