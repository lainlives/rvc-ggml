// RVC-Train-GGML - training-state GGUF loading implementation (src/model/weights.cpp)
#include "model/weights.h"

#include <cstdio>
#include <cstring>

using namespace traincpp;

namespace rvc_train {
namespace model {

namespace {

bool read_str(const GgufFile* f, const char* key, std::string* out) {
    if (!gguf_has_key(f, key)) {
        return false;
    }
    const char* s = gguf_get_str(f, key);
    if (s == nullptr) {
        return false;
    }
    *out = s;
    return true;
}

void read_u32(const GgufFile* f, const char* key, int64_t* out) {
    if (gguf_has_key(f, key)) {
        *out = (int64_t) gguf_get_u32(f, key);
    }
}

void read_i32(const GgufFile* f, const char* key, int64_t* out) {
    if (gguf_has_key(f, key)) {
        *out = (int64_t) gguf_get_i32(f, key);
    }
}

void read_f32(const GgufFile* f, const char* key, float* out) {
    if (gguf_has_key(f, key)) {
        *out = gguf_get_f32(f, key);
    }
}

void read_arr_i32(const GgufFile* f, const char* key, std::vector<int32_t>* out) {
    out->clear();
    if (!gguf_has_key(f, key)) {
        return;
    }
    int64_t n = 0;
    const int32_t* p = gguf_get_arr_i32(f, key, &n);
    if (p != nullptr && n > 0) {
        out->assign(p, p + n);
    }
}

} // namespace

TrainWeights* TrainWeights::open(const std::string& path, std::string* err) {
    GgufFile* f = gguf_open(path.c_str(), /*load_data=*/true);
    if (f == nullptr) {
        if (err != nullptr) {
            *err = "failed to open/parse GGUF: " + path;
        }
        return nullptr;
    }
    TrainWeights* w = new TrainWeights();
    w->file_  = f;
    w->path_  = path;
    w->count_ = gguf_tensor_count(f);
    return w;
}

TrainWeights::~TrainWeights() {
    if (file_ != nullptr) {
        gguf_close(file_);
        file_ = nullptr;
    }
}

const char* TrainWeights::tensor_name(int64_t index) const {
    if (index < 0 || index >= count_) {
        return nullptr;
    }
    const GgufTensorInfo* info = gguf_tensor_info(file_, index);
    return info != nullptr ? info->name : nullptr;
}

bool TrainWeights::has(const char* name) const {
    return gguf_find_tensor(file_, name) >= 0;
}

bool TrainWeights::tensor_ne(const char* name, std::array<int64_t, 4>* ne) const {
    const int64_t idx = gguf_find_tensor(file_, name);
    if (idx < 0) {
        return false;
    }
    const GgufTensorInfo* info = gguf_tensor_info(file_, idx);
    if (info == nullptr) {
        return false;
    }
    for (int i = 0; i < 4; ++i) {
        (*ne)[i] = info->ne[i];
    }
    return true;
}

Type TrainWeights::tensor_type(const char* name) const {
    const int64_t idx = gguf_find_tensor(file_, name);
    if (idx < 0) {
        std::printf("[weights] tensor does not exist: %s\n", name);
        return TYPE_F32;
    }
    const GgufTensorInfo* info = gguf_tensor_info(file_, idx);
    return info != nullptr ? info->type : TYPE_F32;
}

bool TrainWeights::read_meta(TrainMeta* meta) const {
    if (meta == nullptr || file_ == nullptr) {
        return false;
    }
    TrainMeta m;
    read_str(file_, "rvc.train.kind", &m.kind);
    read_str(file_, "rvc.train.version", &m.version);
    m.has_f0 = gguf_has_key(file_, "rvc.train.has_f0") ? gguf_get_bool(file_, "rvc.train.has_f0") : false;
    read_str(file_, "rvc.train.form", &m.form);
    read_str(file_, "rvc.train.source_pth", &m.source_pth);
    read_str(file_, "rvc.train.out_dtype", &m.out_dtype);
    read_u32(file_, "rvc.train.tensor_count", &m.tensor_count);

    read_u32(file_, "rvc.train.sample_rate", &m.sample_rate);
    read_u32(file_, "rvc.train.spec_channels", &m.spec_channels);
    read_u32(file_, "rvc.train.filter_length", &m.filter_length);
    read_u32(file_, "rvc.train.hop_length", &m.hop_length);
    read_u32(file_, "rvc.train.win_length", &m.win_length);
    read_u32(file_, "rvc.train.n_mel_channels", &m.n_mel_channels);
    read_f32(file_, "rvc.train.mel_fmin", &m.mel_fmin);
    read_f32(file_, "rvc.train.mel_fmax", &m.mel_fmax);
    read_u32(file_, "rvc.train.segment_size", &m.segment_size);

    read_u32(file_, "rvc.train.inter_channels", &m.inter_channels);
    read_u32(file_, "rvc.train.hidden_channels", &m.hidden_channels);
    read_u32(file_, "rvc.train.filter_channels", &m.filter_channels);
    read_u32(file_, "rvc.train.n_heads", &m.n_heads);
    read_u32(file_, "rvc.train.n_layers", &m.n_layers);
    read_u32(file_, "rvc.train.kernel_size", &m.kernel_size);
    read_f32(file_, "rvc.train.p_dropout", &m.p_dropout);
    read_u32(file_, "rvc.train.gin_channels", &m.gin_channels);
    read_u32(file_, "rvc.train.spk_embed_dim", &m.spk_embed_dim);
    read_u32(file_, "rvc.train.upsample_initial_channel", &m.upsample_initial_channel);
    read_str(file_, "rvc.train.resblock", &m.resblock);
    read_arr_i32(file_, "rvc.train.resblock_kernel_sizes", &m.resblock_kernel_sizes);
    read_arr_i32(file_, "rvc.train.resblock_dilation_sizes", &m.resblock_dilation_sizes);
    read_arr_i32(file_, "rvc.train.upsample_rates", &m.upsample_rates);
    read_arr_i32(file_, "rvc.train.upsample_kernel_sizes", &m.upsample_kernel_sizes);
    read_arr_i32(file_, "rvc.train.d_periods", &m.d_periods);

    *meta = m;
    return true;
}

Tensor* TrainWeights::new_tensor(Context* ctx, const char* name) const {
    if (!has(name)) {
        std::printf("[weights] tensor does not exist: %s\n", name);
        return nullptr;
    }
    return gguf_new_tensor(ctx, file_, name, TYPE_F32);
}

bool TrainWeights::load(Tensor* dst, const char* name, std::string* err) const {
    if (dst == nullptr) {
        if (err != nullptr) {
            *err = "destination tensor is null";
        }
        return false;
    }
    const int64_t idx = gguf_find_tensor(file_, name);
    if (idx < 0) {
        if (err != nullptr) {
            *err = std::string("missing tensor: ") + name;
        }
        return false;
    }
    const GgufTensorInfo* info = gguf_tensor_info(file_, idx);
    if (info == nullptr) {
        if (err != nullptr) {
            *err = std::string("corrupt tensor info: ") + name;
        }
        return false;
    }

    for (int i = 0; i < 4; ++i) {
        if (dst->ne[i] != info->ne[i]) {
            if (err != nullptr) {
                char buf[512];
                std::snprintf(buf, sizeof(buf),
                              "shape mismatch: %s file ne=[%lld,%lld,%lld,%lld] dst ne=[%lld,%lld,%lld,%lld]",
                              name,
                              (long long) info->ne[0], (long long) info->ne[1],
                              (long long) info->ne[2], (long long) info->ne[3],
                              (long long) dst->ne[0], (long long) dst->ne[1],
                              (long long) dst->ne[2], (long long) dst->ne[3]);
                *err = buf;
            }
            return false;
        }
    }

    if (dst->type != TYPE_F32) {
        if (err != nullptr) {
            *err = std::string("destination tensor type must be F32: ") + name;
        }
        return false;
    }

    if (!gguf_load_tensor(file_, name, dst)) {
        if (err != nullptr) {
            *err = std::string("gguf_load_tensor failed: ") + name;
        }
        return false;
    }
    return true;
}

void TrainWeights::check_expected(const std::vector<Expect>& expected, std::vector<std::string>* missing,
                                  std::vector<std::string>* mismatched) const {
    if (missing != nullptr) {
        missing->clear();
    }
    if (mismatched != nullptr) {
        mismatched->clear();
    }
    for (const Expect& e : expected) {
        std::array<int64_t, 4> ne{1, 1, 1, 1};
        if (!tensor_ne(e.name.c_str(), &ne)) {
            if (missing != nullptr) {
                missing->push_back(e.name);
            }
            continue;
        }
        if (ne != e.ne) {
            if (mismatched != nullptr) {
                mismatched->push_back(e.name);
            }
        }
    }
}

} // namespace model
} // namespace rvc_train
