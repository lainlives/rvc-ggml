// dataset_selftest.cpp —— M5-1 数据加载器对拍（vs golden manifest.dataset.json）
//
// golden 由 tools/gen_golden_dataset.py 生成：cases[] 中每条为
//   kind=sample（单样本装配：phone/pitch/pitchf/spec/wav）
//   kind=batch （bucket 批次：pad 到桶顶 + 掩码 + ids/q_noise/sine + 派生缓冲）
// 约定：raw 为 torch 行主序；C++ 侧 ne 序与其线性布局一致（反序 reshape 不改字节序），
// 因此逐元素直接比较（F32 max_abs；I32 位级）。
// 负向测试：缺缓存 / 未知样本 / ids_slice 越界 / 桶顶过小。

#include "train/dataset_selftest.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "common/json.h"
#include "data/dataset.h"

namespace rvc_train {
namespace train {
namespace {

using data::BatchBuffers;
using data::BatchRandomInputs;
using data::Dataset;
using data::TrainDataConfig;
using data::TrainSample;

// ---------------------------------------------------------------- 基础工具

std::string join_path(const std::string& dir, const std::string& rel) {
    std::string d = dir;
    while (!d.empty() && (d.back() == '/' || d.back() == '\\')) {
        d.pop_back();
    }
    return d + "/" + rel;
}

bool read_bytes(const std::string& path, std::vector<uint8_t>* out, std::string* err) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        if (err != nullptr) {
            *err = "无法打开: " + path;
        }
        return false;
    }
    const std::streamoff size = f.tellg();
    if (size < 0) {
        if (err != nullptr) {
            *err = "无法获取大小: " + path;
        }
        return false;
    }
    out->resize(static_cast<size_t>(size));
    f.seekg(0);
    if (size > 0) {
        f.read(reinterpret_cast<char*>(out->data()), size);
    }
    if (!f) {
        if (err != nullptr) {
            *err = "读取失败: " + path;
        }
        return false;
    }
    return true;
}

bool load_f32(const std::string& path, std::vector<float>* out, std::string* err) {
    std::vector<uint8_t> bytes;
    if (!read_bytes(path, &bytes, err)) {
        return false;
    }
    if (bytes.size() % 4 != 0) {
        if (err != nullptr) {
            *err = "文件字节数非 F32 对齐: " + path;
        }
        return false;
    }
    out->resize(bytes.size() / 4);
    if (!bytes.empty()) {
        std::memcpy(out->data(), bytes.data(), bytes.size());
    }
    return true;
}

bool load_i32(const std::string& path, std::vector<int32_t>* out, std::string* err) {
    std::vector<uint8_t> bytes;
    if (!read_bytes(path, &bytes, err)) {
        return false;
    }
    if (bytes.size() % 4 != 0) {
        if (err != nullptr) {
            *err = "文件字节数非 I32 对齐: " + path;
        }
        return false;
    }
    out->resize(bytes.size() / 4);
    if (!bytes.empty()) {
        std::memcpy(out->data(), bytes.data(), bytes.size());
    }
    return true;
}

int64_t j_int(const common::JsonValue* v, int64_t fallback) {
    return (v != nullptr && v->is_number()) ? v->as_int(fallback) : fallback;
}

double j_num(const common::JsonValue* v, double fallback) {
    return (v != nullptr && v->is_number()) ? v->as_number(fallback) : fallback;
}

std::string j_str(const common::JsonValue* v, const std::string& fallback) {
    return (v != nullptr && v->is_string()) ? v->as_string() : fallback;
}

// ---------------------------------------------------------------- 实际缓冲视图

enum class ViewKind { F32, I32, I64 };

struct TensorView {
    const void* data = nullptr;
    int64_t count = 0;
    ViewKind kind = ViewKind::F32;
};

#define RVCT_VIEW(vec, k)                    \
    out->data = (vec).data();                \
    out->count = static_cast<int64_t>((vec).size()); \
    out->kind = (k);                         \
    return true

bool view_of(const TrainSample& s, const std::string& name, TensorView* out) {
    out->data = nullptr;
    out->count = 0;
    if (name == "phone") {
        RVCT_VIEW(s.phone, ViewKind::F32);
    }
    if (name == "pitch") {
        RVCT_VIEW(s.pitch, ViewKind::I32);
    }
    if (name == "pitchf") {
        RVCT_VIEW(s.pitchf, ViewKind::F32);
    }
    if (name == "spec") {
        RVCT_VIEW(s.spec, ViewKind::F32);
    }
    if (name == "wav") {
        RVCT_VIEW(s.wav, ViewKind::F32);
    }
    return false;
}

bool view_of(const BatchBuffers& b, const std::string& name, TensorView* out) {
    out->data = nullptr;
    out->count = 0;
    if (name == "phone") {
        RVCT_VIEW(b.phone, ViewKind::F32);
    }
    if (name == "pitch") {
        RVCT_VIEW(b.pitch, ViewKind::I32);
    }
    if (name == "pitchf") {
        RVCT_VIEW(b.pitchf, ViewKind::F32);
    }
    if (name == "spec") {
        RVCT_VIEW(b.spec, ViewKind::F32);
    }
    if (name == "wave") {
        RVCT_VIEW(b.wave, ViewKind::F32);
    }
    if (name == "y_mel") {
        RVCT_VIEW(b.y_mel, ViewKind::F32);
    }
    if (name == "sid") {
        RVCT_VIEW(b.sid, ViewKind::I32);
    }
    if (name == "q_noise") {
        RVCT_VIEW(b.q_noise, ViewKind::F32);
    }
    if (name == "x_mask") {
        RVCT_VIEW(b.x_mask, ViewKind::F32);
    }
    if (name == "y_mask") {
        RVCT_VIEW(b.y_mask, ViewKind::F32);
    }
    if (name == "attn_valid") {
        RVCT_VIEW(b.attn_valid, ViewKind::F32);
    }
    if (name == "ids_slice") {
        RVCT_VIEW(b.ids_slice, ViewKind::I32);
    }
    if (name == "lengths") {
        RVCT_VIEW(b.lengths, ViewKind::I64);
    }
    if (name == "slice_mat") {
        RVCT_VIEW(b.slice_mat, ViewKind::F32);
    }
    if (name == "pitchf_slice") {
        RVCT_VIEW(b.pitchf_slice, ViewKind::F32);
    }
    if (name == "sine_source_slice") {
        RVCT_VIEW(b.sine_source_slice, ViewKind::F32);
    }
    return false;
}

#undef RVCT_VIEW

bool compare_f32(const std::string& case_name, const std::string& name,
                 const std::vector<float>& expected, const TensorView& view, double tol,
                 bool* all_ok) {
    if (view.kind != ViewKind::F32 || view.count != static_cast<int64_t>(expected.size())) {
        std::printf("[dataset]   [FAIL] %s/%s 类型或元素数不符（期望 %zu，实际 %lld）\n",
                    case_name.c_str(), name.c_str(), expected.size(),
                    (long long) view.count);
        *all_ok = false;
        return false;
    }
    const float* act = static_cast<const float*>(view.data);
    double max_abs = 0.0;
    int64_t bad = 0;
    int64_t first = -1;
    double first_e = 0.0;
    double first_a = 0.0;
    for (int64_t i = 0; i < view.count; ++i) {
        const double diff = std::fabs(static_cast<double>(expected[static_cast<size_t>(i)]) -
                                      static_cast<double>(act[i]));
        if (diff > max_abs) {
            max_abs = diff;
        }
        if (diff > tol) {
            if (first < 0) {
                first = i;
                first_e = expected[static_cast<size_t>(i)];
                first_a = act[i];
            }
            ++bad;
        }
    }
    if (bad > 0) {
        std::printf("[dataset]   [FAIL] %s/%s 超差元素 %lld/%lld（max_abs=%.3g > tol=%.3g；"
                    "首个 @%lld 期望 %.9g 实际 %.9g）\n",
                    case_name.c_str(), name.c_str(), (long long) bad, (long long) view.count,
                    max_abs, tol, (long long) first, first_e, first_a);
        *all_ok = false;
        return false;
    }
    std::printf("[dataset]   %-18s max_abs=%.3g\n", name.c_str(), max_abs);
    return true;
}

bool compare_i32(const std::string& case_name, const std::string& name,
                 const std::vector<int32_t>& expected, const TensorView& view, bool* all_ok) {
    if (view.kind == ViewKind::F32 || view.count != static_cast<int64_t>(expected.size())) {
        std::printf("[dataset]   [FAIL] %s/%s 类型或元素数不符（期望 %zu，实际 %lld）\n",
                    case_name.c_str(), name.c_str(), expected.size(), (long long) view.count);
        *all_ok = false;
        return false;
    }
    int64_t bad = 0;
    int64_t first = -1;
    int64_t first_e = 0;
    int64_t first_a = 0;
    for (int64_t i = 0; i < view.count; ++i) {
        int64_t act = 0;
        if (view.kind == ViewKind::I32) {
            act = static_cast<const int32_t*>(view.data)[i];
        } else {
            act = static_cast<const int64_t*>(view.data)[i];
        }
        const int64_t exp = expected[static_cast<size_t>(i)];
        if (act != exp) {
            if (first < 0) {
                first = i;
                first_e = exp;
                first_a = act;
            }
            ++bad;
        }
    }
    if (bad > 0) {
        std::printf("[dataset]   [FAIL] %s/%s 位级不一致 %lld/%lld（首个 @%lld 期望 %lld 实际 "
                    "%lld）\n",
                    case_name.c_str(), name.c_str(), (long long) bad, (long long) view.count,
                    (long long) first, (long long) first_e, (long long) first_a);
        *all_ok = false;
        return false;
    }
    std::printf("[dataset]   %-18s 位级一致\n", name.c_str());
    return true;
}

// 比较一条 tensor 条目（path 已按 golden_dir 拼好）
bool compare_entry(const std::string& case_name, const common::JsonValue& t,
                   const std::string& golden_dir, const TensorView& view, double tol_default,
                   double tol_derived, bool* all_ok) {
    const std::string name = j_str(t.find("name"), "");
    const std::string dtype = j_str(t.find("dtype"), "f32");
    const std::string file = j_str(t.find("file"), "");
    if (name.empty() || file.empty()) {
        std::printf("[dataset]   [FAIL] %s 条目缺少 name/file\n", case_name.c_str());
        *all_ok = false;
        return false;
    }
    const std::string path = join_path(golden_dir, file);
    double tol = (name == "y_mel" || name == "sine_source_slice") ? tol_derived : tol_default;
    if (const common::JsonValue* tv = t.find("tol"); tv != nullptr && tv->is_number()) {
        tol = tv->as_number(tol);
    }
    std::string err;
    if (dtype == "i32") {
        std::vector<int32_t> expected;
        if (!load_i32(path, &expected, &err)) {
            std::printf("[dataset]   [FAIL] %s/%s 读取失败: %s\n", case_name.c_str(), name.c_str(),
                        err.c_str());
            *all_ok = false;
            return false;
        }
        return compare_i32(case_name, name, expected, view, all_ok);
    }
    std::vector<float> expected;
    if (!load_f32(path, &expected, &err)) {
        std::printf("[dataset]   [FAIL] %s/%s 读取失败: %s\n", case_name.c_str(), name.c_str(),
                    err.c_str());
        *all_ok = false;
        return false;
    }
    return compare_f32(case_name, name, expected, view, tol, all_ok);
}

// ---------------------------------------------------------------- cases

struct FirstBatch {
    bool valid = false;
    std::vector<int64_t> indices;
    int64_t bucket_top = 0;
    BatchRandomInputs injected;
    std::vector<std::string> files;  // 注入随机量文件（ids_slice/q_noise/sine_noise）
};

// 读取注入随机量（batch case 的 tensors 里按名字找）
bool load_injected(const common::JsonValue& tensors, const std::string& golden_dir, int64_t batch,
                   BatchRandomInputs* out, std::string* err) {
    for (std::size_t i = 0; i < tensors.size(); ++i) {
        const common::JsonValue& t = tensors.at(i);
        const std::string name = j_str(t.find("name"), "");
        const std::string dtype = j_str(t.find("dtype"), "f32");
        const std::string file = j_str(t.find("file"), "");
        if (file.empty()) {
            continue;
        }
        const std::string path = join_path(golden_dir, file);
        if (name == "ids_slice") {
            if (!load_i32(path, &out->ids_slice, err)) {
                return false;
            }
            if (static_cast<int64_t>(out->ids_slice.size()) != batch) {
                if (err != nullptr) {
                    *err = "golden ids_slice 尺寸与 batch 不符";
                }
                return false;
            }
        } else if (name == "q_noise") {
            if (dtype != "f32" || !load_f32(path, &out->q_noise, err)) {
                return false;
            }
        } else if (name == "sine_noise") {
            if (dtype != "f32" || !load_f32(path, &out->sine_noise, err)) {
                return false;
            }
        }
    }
    return true;
}

} // namespace

int run_dataset_selftest(const std::string& golden_dir, const std::string& exp_dir) {
    if (golden_dir.empty() || exp_dir.empty()) {
        std::printf("[dataset] 需要 --golden <dir> 与 --exp <M4 缓存目录>\n");
        return 2;
    }
    std::printf("[dataset] golden=%s exp=%s\n", golden_dir.c_str(), exp_dir.c_str());

    const std::string manifest_path = join_path(golden_dir, "manifest.dataset.json");
    common::JsonValue root;
    std::string err;
    if (!common::json_read_file(manifest_path, &root, &err)) {
        std::printf("[dataset] %s\n", err.c_str());
        return 2;
    }
    const common::JsonValue* params = root.find("params");
    const common::JsonValue* cases = root.find("cases");
    if (params == nullptr || !params->is_object() || cases == nullptr || !cases->is_array()) {
        std::printf("[dataset] manifest 缺少 params/cases\n");
        return 2;
    }

    TrainDataConfig cfg;
    cfg.exp_dir = exp_dir;
    cfg.batch_size = j_int(params->find("batch_size"), 1);
    cfg.segment_frames = j_int(params->find("segment_frames"), 32);
    cfg.segment_samples = j_int(params->find("segment_samples"), 12800);
    cfg.q_channels = j_int(params->find("q_channels"), 192);
    cfg.speaker_id = j_int(params->find("speaker_id"), 0);
    cfg.n_fft = j_int(params->find("n_fft"), 2048);
    cfg.win = j_int(params->find("win"), 2048);
    cfg.n_mels = j_int(params->find("n_mels"), 125);
    cfg.sr = j_num(params->find("sr"), 40000.0);
    cfg.fmin = j_num(params->find("fmin"), 0.0);
    cfg.fmax = j_num(params->find("fmax"), -1.0);
    cfg.log_clip = static_cast<float>(j_num(params->find("log_clip"), 2e-6));
    if (const common::JsonValue* buckets = params->find("buckets");
        buckets != nullptr && buckets->is_array()) {
        for (std::size_t i = 0; i < buckets->size(); ++i) {
            cfg.buckets.push_back(buckets->at(i).as_int());
        }
    }
    const double tol_default = j_num(params->find("tol_default"), 1e-6);
    const double tol_derived = j_num(params->find("tol_derived"), 1e-4);

    Dataset ds;
    if (!Dataset::open(exp_dir, cfg, &ds, &err)) {
        std::printf("[dataset] 缓存加载失败: %s\n", err.c_str());
        return 1;
    }
    std::printf("[dataset] 缓存样本=%lld buckets=%zu\n", (long long) ds.size(), cfg.buckets.size());

    bool all_ok = true;
    int64_t case_count = 0;
    FirstBatch first_batch;

    for (std::size_t ci = 0; ci < cases->size(); ++ci) {
        const common::JsonValue& c = cases->at(ci);
        const std::string name = j_str(c.find("name"), "");
        const std::string kind = j_str(c.find("kind"), "");
        const common::JsonValue* tensors = c.find("tensors");
        if (name.empty() || tensors == nullptr || !tensors->is_array()) {
            std::printf("[dataset] case[%zu] 非法（缺 name/tensors）\n", ci);
            all_ok = false;
            continue;
        }
        ++case_count;
        bool case_ok = true;
        if (kind == "sample") {
            const std::string id = j_str(c.find("sample_id"), "");
            TrainSample s;
            if (!ds.assemble_id(id, &s, &err)) {
                std::printf("[dataset] [FAIL] %s 装配失败: %s\n", name.c_str(), err.c_str());
                all_ok = false;
                continue;
            }
            std::printf("[dataset] case %s（sample %s, T=%lld）\n", name.c_str(), id.c_str(),
                        (long long) s.frames);
            for (std::size_t ti = 0; ti < tensors->size(); ++ti) {
                if (j_str(tensors->at(ti).find("role"), "") == "random") {
                    continue;  // 仅注入用（sine_noise 等）
                }
                TensorView view;
                const std::string tname = j_str(tensors->at(ti).find("name"), "");
                if (!view_of(s, tname, &view)) {
                    std::printf("[dataset]   [FAIL] 无此字段: %s\n", tname.c_str());
                    case_ok = false;
                    continue;
                }
                compare_entry(name, tensors->at(ti), golden_dir, view, tol_default, tol_derived,
                              &case_ok);
            }
        } else if (kind == "batch") {
            std::vector<int64_t> indices;
            const common::JsonValue* ids = c.find("sample_ids");
            if (ids == nullptr || !ids->is_array()) {
                std::printf("[dataset] [FAIL] %s 缺少 sample_ids\n", name.c_str());
                all_ok = false;
                continue;
            }
            bool ids_ok = true;
            for (std::size_t i = 0; i < ids->size(); ++i) {
                const int64_t idx = ds.index_of(ids->at(i).as_string());
                if (idx < 0) {
                    std::printf("[dataset] [FAIL] %s 样本不存在: %s\n", name.c_str(),
                                ids->at(i).as_string().c_str());
                    ids_ok = false;
                    break;
                }
                indices.push_back(idx);
            }
            if (!ids_ok) {
                all_ok = false;
                continue;
            }
            const int64_t bucket_top = j_int(c.find("bucket_top"), 0);
            BatchRandomInputs injected;
            if (!load_injected(*tensors, golden_dir, static_cast<int64_t>(indices.size()),
                               &injected, &err)) {
                std::printf("[dataset] [FAIL] %s 注入随机量读取失败: %s\n", name.c_str(),
                            err.c_str());
                all_ok = false;
                continue;
            }
            BatchBuffers b;
            if (!ds.build_batch(indices, bucket_top, nullptr, &injected, &b, &err)) {
                std::printf("[dataset] [FAIL] %s 批次构建失败: %s\n", name.c_str(), err.c_str());
                all_ok = false;
                continue;
            }
            std::printf("[dataset] case %s（batch B=%lld T_b=%lld）\n", name.c_str(),
                        (long long) b.batch, (long long) b.frames);
            for (std::size_t ti = 0; ti < tensors->size(); ++ti) {
                if (j_str(tensors->at(ti).find("role"), "") == "random") {
                    continue;  // 仅注入用（sine_noise 等）
                }
                TensorView view;
                const std::string tname = j_str(tensors->at(ti).find("name"), "");
                if (!view_of(b, tname, &view)) {
                    std::printf("[dataset]   [FAIL] 无此字段: %s\n", tname.c_str());
                    case_ok = false;
                    continue;
                }
                compare_entry(name, tensors->at(ti), golden_dir, view, tol_default, tol_derived,
                              &case_ok);
            }
            if (!first_batch.valid) {
                first_batch.valid = true;
                first_batch.indices = indices;
                first_batch.bucket_top = bucket_top;
                first_batch.injected = injected;
            }
        } else {
            std::printf("[dataset] case[%zu] 未知 kind=%s\n", ci, kind.c_str());
            all_ok = false;
            continue;
        }
        std::printf("[dataset] case %s: %s\n", name.c_str(), case_ok ? "OK" : "FAIL");
        if (!case_ok) {
            all_ok = false;
        }
    }

    // ---------------- 负向测试 ----------------
    int neg_ok = 0;
    {
        Dataset bad;
        if (!Dataset::open(join_path(exp_dir, "__no_such_cache__"), cfg, &bad, &err)) {
            std::printf("[dataset] 负向(缺缓存): 通过\n");
            ++neg_ok;
        } else {
            std::printf("[dataset] 负向(缺缓存): 失败（不应打开成功）\n");
        }
    }
    {
        TrainSample ts;
        if (!ds.assemble_id("__no_such_sample__", &ts, &err)) {
            std::printf("[dataset] 负向(未知样本): 通过\n");
            ++neg_ok;
        } else {
            std::printf("[dataset] 负向(未知样本): 失败\n");
        }
    }
    if (first_batch.valid && first_batch.bucket_top > 0) {
        // ids_slice 越界：设为该样本对齐帧数（> frames-seg）
        BatchRandomInputs inj = first_batch.injected;
        const int64_t frames0 = ds.aligned_frames(first_batch.indices[0]);
        inj.ids_slice[0] = static_cast<int32_t>(frames0);
        BatchBuffers b;
        if (!ds.build_batch(first_batch.indices, first_batch.bucket_top, nullptr, &inj, &b, &err)) {
            std::printf("[dataset] 负向(ids_slice 越界): 通过\n");
            ++neg_ok;
        } else {
            std::printf("[dataset] 负向(ids_slice 越界): 失败\n");
        }
        // 桶顶过小
        BatchRandomInputs inj2 = first_batch.injected;
        if (!ds.build_batch(first_batch.indices, frames0 - 1, nullptr, &inj2, &b, &err)) {
            std::printf("[dataset] 负向(桶顶过小): 通过\n");
            ++neg_ok;
        } else {
            std::printf("[dataset] 负向(桶顶过小): 失败\n");
        }
    } else {
        std::printf("[dataset] 负向(ids/桶顶): 跳过（无 batch case）\n");
    }
    const int neg_total = 4;
    if (neg_ok != neg_total) {
        all_ok = false;
    }

    std::printf("[dataset] cases=%lld，负向 %d/%d，%s\n", (long long) case_count, neg_ok, neg_total,
                all_ok ? "M5-1 数据加载器自检通过" : "M5-1 数据加载器自检失败");
    return all_ok ? 0 : 1;
}

} // namespace train
} // namespace rvc_train
