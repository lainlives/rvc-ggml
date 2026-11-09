// gguf_model.cpp
// GGUF loading implementation (modeled on MSST-GGML, with metadata and type support enhanced for RVC scenarios).

#include "gguf_model.h"
#include "utils.h"

#include <ggml-alloc.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <stdexcept>

namespace {

// Generic metadata copy: read all key-values into host memory so they survive gguf_context release
void copy_metadata(gguf_context* ctx, int64_t n_kv,
    std::vector<std::pair<std::string, int64_t>>& u32,
    std::vector<std::pair<std::string, float>>& f32,
    std::vector<std::pair<std::string, bool>>& booleans,
    std::vector<std::pair<std::string, std::string>>& strings,
    std::vector<std::pair<std::string, std::vector<std::string>>>& arrays) {
    for (int64_t i = 0; i < n_kv; ++i) {
        const char* key_raw = gguf_get_key(ctx, i);
        if (!key_raw) continue;
        const std::string key = key_raw;
        // Skip array types
        if (gguf_get_kv_type(ctx, i) == GGUF_TYPE_ARRAY) {
            continue;
        }
        switch (gguf_get_kv_type(ctx, i)) {
        case GGUF_TYPE_UINT8:
        case GGUF_TYPE_INT8:
        case GGUF_TYPE_UINT16:
        case GGUF_TYPE_INT16:
        case GGUF_TYPE_UINT32:
        case GGUF_TYPE_INT32:
        case GGUF_TYPE_UINT64:
        case GGUF_TYPE_INT64:
            u32.emplace_back(key, static_cast<int64_t>(gguf_get_val_u32(ctx, i)));
            break;
        case GGUF_TYPE_FLOAT32:
            f32.emplace_back(key, gguf_get_val_f32(ctx, i));
            break;
        case GGUF_TYPE_BOOL:
            booleans.emplace_back(key, gguf_get_val_bool(ctx, i));
            break;
        case GGUF_TYPE_STRING:
            strings.emplace_back(key, gguf_get_val_str(ctx, i));
            break;
        default:
            break;
        }
    }
}

} // namespace

GgufModel::~GgufModel() {
    if (buffer_fold_) ggml_backend_buffer_free(buffer_fold_);
    if (ctx_fold_) ggml_free(ctx_fold_);
    if (buffer_weights_) ggml_backend_buffer_free(buffer_weights_);
    if (ctx_weights_) ggml_free(ctx_weights_);
}

void GgufModel::load(const std::string& path, ggml_backend_t backend) {
    Logger::info("Loading GGUF: " + path);

    struct gguf_init_params params = {
        /*.no_alloc = */ true,
        /*.ctx      = */ &ctx_weights_,
    };
    gguf_context* ctx_gguf = gguf_init_from_file(path.c_str(), params);
    if (!ctx_gguf) {
        throw std::runtime_error("Failed to open GGUF file: " + path);
    }

    try {
        load_metadata(ctx_gguf);
        load_weights(ctx_gguf, path, backend);
    }
    catch (...) {
        gguf_free(ctx_gguf);
        throw;
    }
    gguf_free(ctx_gguf);
}

void GgufModel::load_meta_only(const std::string& path) {
    struct gguf_init_params params = {
        /*.no_alloc = */ true,
        /*.ctx      = */ nullptr,
    };
    gguf_context* ctx_gguf = gguf_init_from_file(path.c_str(), params);
    if (!ctx_gguf) {
        throw std::runtime_error("Failed to open GGUF file: " + path);
    }
    try {
        load_metadata(ctx_gguf);
    }
    catch (...) {
        gguf_free(ctx_gguf);
        throw;
    }
    gguf_free(ctx_gguf);
}

void GgufModel::load_metadata(gguf_context* ctx_gguf) {
    const int64_t n_kv = gguf_get_n_kv(ctx_gguf);
    for (int64_t i = 0; i < n_kv; ++i) {
        const char* key_raw = gguf_get_key(ctx_gguf, i);
        if (!key_raw) continue;
        const std::string key = key_raw;

        if (gguf_get_kv_type(ctx_gguf, i) == GGUF_TYPE_ARRAY) {
            if (key == "general.architecture") continue;
            const enum gguf_type at = gguf_get_arr_type(ctx_gguf, i);
            if (at == GGUF_TYPE_STRING) {
                std::vector<std::string> values;
                const int64_t n = gguf_get_arr_n(ctx_gguf, i);
                for (int64_t k = 0; k < n; ++k) {
                    const char* s = gguf_get_arr_str(ctx_gguf, i, k);
                    values.push_back(s ? s : "");
                }
                meta_.arrays.emplace_back(key, std::move(values));
            }
            else if (at == GGUF_TYPE_UINT8 || at == GGUF_TYPE_INT8 ||
                at == GGUF_TYPE_UINT16 || at == GGUF_TYPE_INT16 ||
                at == GGUF_TYPE_UINT32 || at == GGUF_TYPE_INT32 ||
                at == GGUF_TYPE_UINT64 || at == GGUF_TYPE_INT64) {
                std::vector<int64_t> values;
                const int64_t n = gguf_get_arr_n(ctx_gguf, i);
                const void* data = gguf_get_arr_data(ctx_gguf, i);
                values.reserve(static_cast<size_t>(n));
                for (int64_t k = 0; k < n; ++k) {
                    int64_t v = 0;
                    switch (at) {
                    case GGUF_TYPE_UINT8:  v = reinterpret_cast<const uint8_t*>(data)[k]; break;
                    case GGUF_TYPE_INT8:   v = reinterpret_cast<const int8_t*>(data)[k]; break;
                    case GGUF_TYPE_UINT16: v = reinterpret_cast<const uint16_t*>(data)[k]; break;
                    case GGUF_TYPE_INT16:  v = reinterpret_cast<const int16_t*>(data)[k]; break;
                    case GGUF_TYPE_UINT32: v = reinterpret_cast<const uint32_t*>(data)[k]; break;
                    case GGUF_TYPE_INT32:  v = reinterpret_cast<const int32_t*>(data)[k]; break;
                    case GGUF_TYPE_UINT64: v = static_cast<int64_t>(reinterpret_cast<const uint64_t*>(data)[k]); break;
                    default:               v = reinterpret_cast<const int64_t*>(data)[k]; break;
                    }
                    values.push_back(v);
                }
                meta_.arrays_int.emplace_back(key, std::move(values));
            }
            continue;
        }

        switch (gguf_get_kv_type(ctx_gguf, i)) {
        case GGUF_TYPE_UINT8:
        case GGUF_TYPE_INT8:
        case GGUF_TYPE_UINT16:
        case GGUF_TYPE_INT16:
        case GGUF_TYPE_UINT32:
        case GGUF_TYPE_INT32:
        case GGUF_TYPE_UINT64:
        case GGUF_TYPE_INT64:
            meta_.u32.emplace_back(key, static_cast<int64_t>(gguf_get_val_u32(ctx_gguf, i)));
            break;
        case GGUF_TYPE_FLOAT32:
            meta_.f32.emplace_back(key, gguf_get_val_f32(ctx_gguf, i));
            break;
        case GGUF_TYPE_BOOL:
            meta_.booleans.emplace_back(key, gguf_get_val_bool(ctx_gguf, i));
            break;
        case GGUF_TYPE_STRING:
            meta_.strings.emplace_back(key, gguf_get_val_str(ctx_gguf, i));
            break;
        default:
            break;
        }
    }

    int64_t arch_idx = gguf_find_key(ctx_gguf, "general.architecture");
    if (arch_idx < 0) {
        throw std::runtime_error("GGUF is missing the general.architecture key");
    }
    architecture_ = gguf_get_val_str(ctx_gguf, arch_idx);
    Logger::info("Architecture: " + architecture_);
}

void GgufModel::load_weights(gguf_context* ctx_gguf, const std::string& path, ggml_backend_t backend) {
    buffer_weights_ = ggml_backend_alloc_ctx_tensors_from_buft(
        ctx_weights_, ggml_backend_get_default_buffer_type(backend));
    if (!buffer_weights_) {
        throw std::runtime_error("Failed to allocate weight buffer (out of VRAM?)");
    }

#ifdef _WIN32
    std::filesystem::path p(path);
    FILE* file = _wfopen(p.wstring().c_str(), L"rb");
#else
    FILE* file = std::fopen(path.c_str(), "rb");
#endif
    if (!file) {
        throw std::runtime_error("Failed to open file: " + path);
    }

    const size_t data_offset = gguf_get_data_offset(ctx_gguf);
    std::vector<uint8_t> read_buf;
    int n_loaded = 0;

    ggml_tensor* t = ggml_get_first_tensor(ctx_weights_);
    while (t) {
        int tid = gguf_find_tensor(ctx_gguf, t->name);
        if (tid >= 0) {
            const size_t offset = data_offset + gguf_get_tensor_offset(ctx_gguf, tid);
            const size_t size = ggml_nbytes(t);
            if (read_buf.size() < size) read_buf.resize(size);

#ifdef _WIN32
            if (_fseeki64(file, static_cast<long long>(offset), SEEK_SET) != 0) {
#else
            if (fseeko(file, static_cast<off_t>(offset), SEEK_SET) != 0) {
#endif
                fclose(file);
                throw std::runtime_error("GGUF seek failed: " + std::string(t->name));
            }
            const size_t rd = fread(read_buf.data(), 1, size, file);
            if (rd != size) {
                fclose(file);
                throw std::runtime_error("GGUF data is incomplete: " + std::string(t->name));
            }

            ggml_backend_tensor_set(t, read_buf.data(), 0, size);
            ++n_loaded;
        }
        t = ggml_get_next_tensor(ctx_weights_, t);
    }
    fclose(file);

    // Q12: read training-state GGUF directly — fold weight_norm (materialize after buffer allocation and before graph building.
    // Only applies when X.weight_g/weight_v are present and X.weight is missing; inference-format GGUF is unaffected)
    fold_weight_norm_tensors(backend);

    tensor_count_ = n_loaded;
    Logger::info("Weight loading complete: " + std::to_string(n_loaded) + " tensors");
}

namespace {

// Real tensor dimensionality (ignoring trailing size-1 dims); used with reversed torch shapes
int fold_infer_ndim(const int64_t ne[4]) {
    int nd = 1;
    for (int i = 0; i < 4; ++i) {
        if (ne[i] > 1) nd = i + 1;
    }
    return nd;
}

bool fold_ends_with(const std::string& s, const char* suf) {
    const size_t n = std::strlen(suf);
    return s.size() >= n && std::memcmp(s.data() + (s.size() - n), suf, n) == 0;
}

// Read a tensor as F32 host data (supports F32/F16; other types raise an error)
void fold_tensor_to_host(ggml_tensor* t, std::vector<float>& out) {
    const size_t n = static_cast<size_t>(ggml_nelements(t));
    out.resize(n);
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, out.data(), 0, ggml_nbytes(t));
    }
    else if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> raw(n);
        ggml_backend_tensor_get(t, raw.data(), 0, ggml_nbytes(t));
        for (size_t i = 0; i < n; ++i) out[i] = ggml_fp16_to_fp32(raw[i]);
    }
    else {
        throw std::runtime_error(std::string("weight_norm folding: unsupported tensor type ") +
            ggml_type_name(t->type) + " (" + t->name + ")");
    }
}

} // namespace

void GgufModel::fold_weight_norm_tensors(ggml_backend_t backend) {
    if (!ctx_weights_) return;

    // Collect the (module_base, weight_g, weight_v) entries that need folding
    struct Pending { std::string base; ggml_tensor* g; ggml_tensor* v; };
    std::vector<Pending> pending;
    for (ggml_tensor* t = ggml_get_first_tensor(ctx_weights_); t;
         t = ggml_get_next_tensor(ctx_weights_, t)) {
        const std::string name = t->name;
        if (!fold_ends_with(name, ".weight_v")) continue;
        const std::string base = name.substr(0, name.size() - std::strlen(".weight_v"));
        if (ggml_get_tensor(ctx_weights_, (base + ".weight").c_str()) != nullptr) continue;
        ggml_tensor* g = ggml_get_tensor(ctx_weights_, (base + ".weight_g").c_str());
        if (g == nullptr) continue;
        pending.push_back({ base, g, t });
    }
    if (pending.empty()) return;

    // Separate context + buffer (the ctx_weights_ metadata area is full; no more tensors can be appended)
    const size_t mem = (pending.size() + 4) * ggml_tensor_overhead() + 1024 * 1024;
    struct ggml_init_params ip = { /*.mem_size=*/ mem, /*.mem_buffer=*/ nullptr,
                                   /*.no_alloc=*/ true };
    ggml_context* cfold = ggml_init(ip);
    if (!cfold) {
        throw std::runtime_error("weight_norm folding: failed to create context");
    }

    std::vector<ggml_tensor*> wlist;
    wlist.reserve(pending.size());
    for (const Pending& p : pending) {
        // weight has the same shape as weight_v (folding only changes values, not shapes)
        ggml_tensor* w = ggml_new_tensor(cfold, GGML_TYPE_F32, GGML_MAX_DIMS, p.v->ne);
        ggml_set_name(w, (p.base + ".weight").c_str());
        wlist.push_back(w);
    }

    buffer_fold_ = ggml_backend_alloc_ctx_tensors_from_buft(
        cfold, ggml_backend_get_default_buffer_type(backend));
    if (!buffer_fold_) {
        ggml_free(cfold);
        throw std::runtime_error("weight_norm folding: buffer allocation failed (out of VRAM?)");
    }

    int n_folded = 0;
    for (size_t i = 0; i < pending.size(); ++i) {
        const Pending& p = pending[i];
        std::vector<float> v, g;
        fold_tensor_to_host(p.v, v);
        fold_tensor_to_host(p.g, g);

        const int v_ndim = fold_infer_ndim(p.v->ne);
        int64_t vshape[4] = { 1, 1, 1, 1 };
        for (int d = 0; d < v_ndim; ++d) vshape[d] = p.v->ne[v_ndim - 1 - d];
        int64_t total = 1;
        for (int d = 0; d < v_ndim; ++d) total *= vshape[d];
        const int64_t O = vshape[0];
        if (static_cast<int64_t>(v.size()) != total) {
            throw std::runtime_error("weight_norm folding: unexpected tensor element count (" + p.base + ")");
        }
        if (static_cast<int64_t>(g.size()) != O) {
            throw std::runtime_error(
                "weight_norm folding: fairseq dim=(0,1) convention not yet supported (" + p.base +
                ": g.numel=" + std::to_string(g.size()) + ", dim0=" + std::to_string(O) + ")");
        }

        const int64_t inner = (O > 0) ? total / O : 0;
        std::vector<float> out(static_cast<size_t>(total));
        for (int64_t o = 0; o < O; ++o) {
            const float* vp = v.data() + static_cast<size_t>(o * inner);
            double ss = 0.0;
            for (int64_t j = 0; j < inner; ++j) {
                const double x = static_cast<double>(vp[j]);
                ss += x * x;
            }
            double norm = std::sqrt(ss);
            if (norm < 1e-12) norm = 1e-12;
            const double gg = static_cast<double>(g[static_cast<size_t>(o)]);
            float* op = out.data() + static_cast<size_t>(o * inner);
            for (int64_t j = 0; j < inner; ++j) {
                op[j] = static_cast<float>(static_cast<double>(vp[j]) / norm * gg);
            }
        }
        ggml_backend_tensor_set(wlist[i], out.data(), 0, out.size() * sizeof(float));
        ++n_folded;
    }

    ctx_fold_ = cfold;
    Logger::info("weight_norm folded: " + std::to_string(n_folded) + " tensors (training-state GGUF read directly)");
}

ggml_tensor* GgufModel::weight(const std::string& name) const {
    ggml_tensor* t = ggml_get_tensor(ctx_weights_, name.c_str());
    if (t == nullptr && ctx_fold_ != nullptr) {
        t = ggml_get_tensor(ctx_fold_, name.c_str());
    }
    return t;
}

namespace {

// Q12: read training-state GGUF directly — fall back to rvc.train.* when rvc.* is missing (key name loses .train; types correspond).
// Only affects the read layer; inference-format GGUF keys match as-is with no behavior change.
std::string meta_fallback_key(const std::string& key) {
    if (key.rfind("rvc.", 0) == 0 && key.rfind("rvc.train.", 0) != 0) {
        return "rvc.train." + key.substr(4);
    }
    return std::string();
}

} // namespace

int GgufModel::meta_u32(const std::string& key, int fallback) const {
    for (const auto& kv : meta_.u32) {
        if (kv.first == key) return static_cast<int>(kv.second);
    }
    const std::string fk = meta_fallback_key(key);
    if (!fk.empty()) {
        for (const auto& kv : meta_.u32) {
            if (kv.first == fk) return static_cast<int>(kv.second);
        }
    }
    return fallback;
}

float GgufModel::meta_f32(const std::string& key, float fallback) const {
    for (const auto& kv : meta_.f32) {
        if (kv.first == key) return kv.second;
    }
    const std::string fk = meta_fallback_key(key);
    if (!fk.empty()) {
        for (const auto& kv : meta_.f32) {
            if (kv.first == fk) return kv.second;
        }
    }
    return fallback;
}

bool GgufModel::meta_bool(const std::string& key, bool fallback) const {
    for (const auto& kv : meta_.booleans) {
        if (kv.first == key) return kv.second;
    }
    const std::string fk = meta_fallback_key(key);
    if (!fk.empty()) {
        for (const auto& kv : meta_.booleans) {
            if (kv.first == fk) return kv.second;
        }
    }
    return fallback;
}

std::string GgufModel::meta_str(const std::string& key, const std::string& fallback) const {
    for (const auto& kv : meta_.strings) {
        if (kv.first == key) return kv.second;
    }
    const std::string fk = meta_fallback_key(key);
    if (!fk.empty()) {
        for (const auto& kv : meta_.strings) {
            if (kv.first == fk) return kv.second;
        }
    }
    return fallback;
}

std::vector<std::string> GgufModel::meta_array_str(const std::string& key) const {
    for (const auto& kv : meta_.arrays) {
        if (kv.first == key) return kv.second;
    }
    const std::string fk = meta_fallback_key(key);
    if (!fk.empty()) {
        for (const auto& kv : meta_.arrays) {
            if (kv.first == fk) return kv.second;
        }
    }
    return {};
}

std::vector<int64_t> GgufModel::meta_array_i32(const std::string& key) const {
    for (const auto& kv : meta_.arrays_int) {
        if (kv.first == key) return kv.second;
    }
    const std::string fk = meta_fallback_key(key);
    if (!fk.empty()) {
        for (const auto& kv : meta_.arrays_int) {
            if (kv.first == fk) return kv.second;
        }
    }
    return {};
}

std::vector<float> GgufModel::read_f32(const std::string& name) const {
    ggml_tensor* t = weight(name);
    if (!t) {
        throw std::runtime_error("Missing tensor: " + name);
    }
    const size_t n = static_cast<size_t>(ggml_nelements(t));
    std::vector<float> out(n);
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, out.data(), 0, ggml_nbytes(t));
        return out;
    }
    if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> raw(n);
        ggml_backend_tensor_get(t, raw.data(), 0, ggml_nbytes(t));
        ggml_fp16_to_fp32_row(raw.data(), out.data(), static_cast<int64_t>(n));
        return out;
    }
    if (ggml_is_quantized(t->type)) {
        const auto* traits = ggml_get_type_traits(t->type);
        if (!traits || !traits->to_float) {
            throw std::runtime_error("read_f32: quantized type does not support dequantization: " + name);
        }
        std::vector<uint8_t> raw(ggml_nbytes(t));
        ggml_backend_tensor_get(t, raw.data(), 0, raw.size());
        traits->to_float(raw.data(), out.data(), static_cast<int64_t>(n));
        return out;
    }
    throw std::runtime_error("read_f32 does not support this tensor type: " + name);
}
