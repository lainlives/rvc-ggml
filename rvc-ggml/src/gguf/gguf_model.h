// gguf_model.h
// GGUF model loading: metadata parsing + weight upload to a given ggml backend.
// Design modeled on the same-named MSST-GGML module, slimmed down for RVC-GGML needs.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <ggml.h>
#include <ggml-backend.h>
#include <gguf.h>

class GgufModel {
public:
    GgufModel() = default;
    ~GgufModel();
    GgufModel(const GgufModel&) = delete;
    GgufModel& operator=(const GgufModel&) = delete;

    // Load a GGUF file and upload all weights to backend (backend is owned by the caller)
    void load(const std::string& path, ggml_backend_t backend);

    // Read metadata only (no weight loading; lets the interactive UI scan models quickly)
    void load_meta_only(const std::string& path);

    const std::string& architecture() const { return architecture_; }

    ggml_tensor* weight(const std::string& name) const;
    bool has_weight(const std::string& name) const { return weight(name) != nullptr; }

    // Metadata reads (return fallback when the key is absent)
    int meta_u32(const std::string& key, int fallback) const;
    float meta_f32(const std::string& key, float fallback) const;
    bool meta_bool(const std::string& key, bool fallback) const;
    std::string meta_str(const std::string& key, const std::string& fallback) const;
    std::vector<std::string> meta_array_str(const std::string& key) const;
    std::vector<int64_t> meta_array_i32(const std::string& key) const;

    // Copy the given tensor back to host memory (float32 view; supports F32/F16 weights; quantized tensors unsupported)
    std::vector<float> read_f32(const std::string& name) const;

    int tensor_count() const { return tensor_count_; }

private:
    void load_metadata(gguf_context* ctx_gguf);
    void load_weights(gguf_context* ctx_gguf, const std::string& path, ggml_backend_t backend);
    // Q12: read training-state GGUF directly — fold weight_norm's weight_g/weight_v into weight
    // (only adds tensors to a separate context inside this class; no inference graph or numeric path changes)
    void fold_weight_norm_tensors(ggml_backend_t backend);

    std::string architecture_;
    ggml_context* ctx_weights_ = nullptr;
    ggml_backend_buffer_t buffer_weights_ = nullptr;
    // weight tensors produced by folding (separate context + buffer; weight() searches them too)
    ggml_context* ctx_fold_ = nullptr;
    ggml_backend_buffer_t buffer_fold_ = nullptr;
    int tensor_count_ = 0;

    // Metadata copies (host)
    struct Meta {
        std::vector<std::pair<std::string, int64_t>> u32;
        std::vector<std::pair<std::string, float>> f32;
        std::vector<std::pair<std::string, bool>> booleans;
        std::vector<std::pair<std::string, std::string>> strings;
        std::vector<std::pair<std::string, std::vector<std::string>>> arrays;
        std::vector<std::pair<std::string, std::vector<int64_t>>> arrays_int;
    };
    Meta meta_;
};
