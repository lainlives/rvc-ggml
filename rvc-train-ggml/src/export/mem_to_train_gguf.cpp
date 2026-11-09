// RVC-Train-GGML - M7-2 in-memory -> training-state GGUF bridge implementation (src/export/mem_to_train_gguf.cpp)
#include "export/mem_to_train_gguf.h"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <system_error>

#include "traincpp/traincpp.h"

using namespace traincpp;

namespace rvc_train {
namespace exporter {

namespace {

void set_u32(GgufWriter* w, const char* key, int64_t v) {
    gguf_writer_set_u32(w, key, (uint32_t) v);
}

void set_arr_i32(GgufWriter* w, const char* key, const std::vector<int32_t>& v) {
    gguf_writer_set_arr_i32(w, key, v.empty() ? nullptr : v.data(), (int64_t) v.size());
}

} // namespace

bool write_train_gguf(const model::ModelConfig& mc,
                      const std::vector<model::TensorSpec>& specs,
                      const std::vector<std::vector<float>>& data, const std::string& path,
                      const std::string& general_name, const std::string& source_pth,
                      std::string* err) {
    if (specs.size() != data.size()) {
        if (err != nullptr) {
            *err = "specs/data length mismatch";
        }
        return false;
    }
    std::error_code ec;
    const std::filesystem::path p(path);
    if (p.has_parent_path()) {
        std::filesystem::create_directories(p.parent_path(), ec);
        if (ec) {
            if (err != nullptr) {
                *err = "failed to create output directory: " + p.parent_path().string();
            }
            return false;
        }
    }

    ContextParams cp;
    cp.mem_size = (size_t) 64 << 20;
    cp.no_alloc = false;  // host allocates tensor data (for GGUF writing)
    Context* ctx = context_new(cp);
    if (ctx == nullptr) {
        if (err != nullptr) {
            *err = "context_new failed";
        }
        return false;
    }
    GgufWriter* w = gguf_writer_new();
    if (w == nullptr) {
        context_free(ctx);
        if (err != nullptr) {
            *err = "gguf_writer_new failed";
        }
        return false;
    }

    gguf_writer_set_arch(w, "rvc-train");
    gguf_writer_set_str(w, "general.name", general_name.empty() ? "rvc-scratch" : general_name.c_str());
    gguf_writer_set_str(w, "rvc.train.kind", mc.kind.c_str());
    gguf_writer_set_str(w, "rvc.train.version", mc.version.c_str());
    gguf_writer_set_bool(w, "rvc.train.has_f0", mc.has_f0);
    gguf_writer_set_str(w, "rvc.train.form", mc.form.empty() ? "train" : mc.form.c_str());
    gguf_writer_set_str(w, "rvc.train.source_pth", source_pth.empty() ? "scratch" : source_pth.c_str());
    gguf_writer_set_str(w, "rvc.train.out_dtype", "F32");
    set_u32(w, "rvc.train.tensor_count", (int64_t) specs.size());

    set_u32(w, "rvc.train.sample_rate", mc.sample_rate);
    set_u32(w, "rvc.train.spec_channels", mc.spec_channels);
    set_u32(w, "rvc.train.filter_length", mc.filter_length);
    set_u32(w, "rvc.train.hop_length", mc.hop_length);
    set_u32(w, "rvc.train.win_length", mc.win_length);
    set_u32(w, "rvc.train.n_mel_channels", mc.n_mel_channels);
    gguf_writer_set_f32(w, "rvc.train.mel_fmin", mc.mel_fmin);
    gguf_writer_set_f32(w, "rvc.train.mel_fmax", mc.mel_fmax);
    set_u32(w, "rvc.train.segment_size", mc.segment_size);

    set_u32(w, "rvc.train.inter_channels", mc.inter_channels);
    set_u32(w, "rvc.train.hidden_channels", mc.hidden_channels);
    set_u32(w, "rvc.train.filter_channels", mc.filter_channels);
    set_u32(w, "rvc.train.n_heads", mc.n_heads);
    set_u32(w, "rvc.train.n_layers", mc.n_layers);
    set_u32(w, "rvc.train.kernel_size", mc.kernel_size);
    gguf_writer_set_f32(w, "rvc.train.p_dropout", mc.p_dropout);
    set_u32(w, "rvc.train.gin_channels", mc.gin_channels);
    set_u32(w, "rvc.train.spk_embed_dim", mc.spk_embed_dim);
    set_u32(w, "rvc.train.upsample_initial_channel", mc.upsample_initial_channel);
    gguf_writer_set_str(w, "rvc.train.resblock", mc.resblock.c_str());
    set_arr_i32(w, "rvc.train.resblock_kernel_sizes", mc.resblock_kernel_sizes);
    set_arr_i32(w, "rvc.train.resblock_dilation_sizes", mc.resblock_dilation_sizes);
    set_arr_i32(w, "rvc.train.upsample_rates", mc.upsample_rates);
    set_arr_i32(w, "rvc.train.upsample_kernel_sizes", mc.upsample_kernel_sizes);
    set_arr_i32(w, "rvc.train.d_periods", mc.d_periods);

    for (size_t i = 0; i < specs.size(); ++i) {
        const model::TensorSpec& s = specs[i];
        Tensor* t = new_tensor_nd(ctx, TYPE_F32, 4, s.ne.data());
        if (t == nullptr) {
            if (err != nullptr) {
                *err = "failed to create tensor: " + s.name;
            }
            gguf_writer_free(w);
            context_free(ctx);
            return false;
        }
        tensor_set_name(t, "%s", s.name.c_str());
        float* dst = tensor_data_f32(t);
        if (dst == nullptr) {
            if (err != nullptr) {
                *err = "tensor data not allocated: " + s.name;
            }
            gguf_writer_free(w);
            context_free(ctx);
            return false;
        }
        const std::vector<float>& src = data[i];
        if ((int64_t) src.size() != tensor_nelements(t)) {
            if (err != nullptr) {
                *err = "tensor element count mismatch: " + s.name;
            }
            gguf_writer_free(w);
            context_free(ctx);
            return false;
        }
        std::memcpy(dst, src.data(), src.size() * sizeof(float));
        gguf_writer_add_tensor(w, s.name.c_str(), t, TYPE_F32);
    }

    const bool ok = gguf_writer_write(w, path.c_str());
    if (!ok && err != nullptr) {
        *err = "failed to write GGUF: " + path;
    }
    gguf_writer_free(w);
    context_free(ctx);
    return ok;
}

bool write_scratch_train_gguf(const model::ModelConfig& mc,
                              const std::vector<model::TensorSpec>& specs,
                              const std::vector<std::vector<float>>& data, const std::string& path,
                              std::string* err) {
    return write_train_gguf(mc, specs, data, path, "rvc-scratch", "scratch", err);
}

bool export_train_gguf_live(const model::ModelConfig& g_cfg, const model::ModelParams& g_params,
                            const std::string& path, const std::string& general_name,
                            const std::string& source_pth, std::string* err) {
    const traincpp::ParamList& pl = g_params.params();
    std::vector<model::TensorSpec> specs;
    specs.reserve(pl.items.size());
    for (traincpp::Tensor* t : pl.items) {
        if (t == nullptr) {
            continue;
        }
        model::TensorSpec s;
        s.name = t->name;
        s.ne = {t->ne[0], t->ne[1], t->ne[2], t->ne[3]};
        specs.push_back(std::move(s));
    }
    std::vector<std::vector<float>> data;
    data.reserve(specs.size());
    for (traincpp::Tensor* t : pl.items) {
        if (t == nullptr) {
            continue;
        }
        std::vector<float> v((size_t) tensor_nelements(t));
        tensor_get(t, v.data(), 0, v.size() * sizeof(float));
        data.push_back(std::move(v));
    }
    return write_train_gguf(g_cfg, specs, data, path, general_name, source_pth, err);
}

bool generate_scratch_g(const model::ModelConfig& g_cfg, uint64_t seed, const std::string& path,
                        std::string* err) {
    std::vector<model::TensorSpec> specs;
    if (!model::build_g_specs(g_cfg, &specs, err)) {
        return false;
    }
    std::vector<std::vector<float>> data;
    if (!model::fill_specs(specs, seed, &data, err)) {
        return false;
    }
    return write_scratch_train_gguf(g_cfg, specs, data, path, err);
}

bool generate_scratch_d(const model::ModelConfig& d_cfg, uint64_t seed, const std::string& path,
                        std::string* err) {
    std::vector<model::TensorSpec> specs;
    if (!model::build_d_specs(d_cfg, &specs, err)) {
        return false;
    }
    std::vector<std::vector<float>> data;
    if (!model::fill_specs(specs, seed, &data, err)) {
        return false;
    }
    return write_scratch_train_gguf(d_cfg, specs, data, path, err);
}

} // namespace exporter
} // namespace rvc_train
