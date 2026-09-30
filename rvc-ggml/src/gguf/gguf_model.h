// gguf_model.h
// GGUF 模型加载：元数据解析 + 权重上传到指定 ggml 后端。
// 设计参照 MSST-GGML 同名模块，按 RVC-GGML 需要精简。

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

    // 加载 GGUF 文件并将全部权重上传到 backend（backend 由调用者持有）
    void load(const std::string& path, ggml_backend_t backend);

    // 仅读取元数据（不加载权重，供交互式界面快速扫描模型）
    void load_meta_only(const std::string& path);

    const std::string& architecture() const { return architecture_; }

    ggml_tensor* weight(const std::string& name) const;
    bool has_weight(const std::string& name) const { return weight(name) != nullptr; }

    // 元数据读取（无键时返回 fallback）
    int meta_u32(const std::string& key, int fallback) const;
    float meta_f32(const std::string& key, float fallback) const;
    bool meta_bool(const std::string& key, bool fallback) const;
    std::string meta_str(const std::string& key, const std::string& fallback) const;
    std::vector<std::string> meta_array_str(const std::string& key) const;
    std::vector<int64_t> meta_array_i32(const std::string& key) const;

    // 把指定张量拷回主机内存（float32 视图，支持 F32/F16 权重；量化张量不支持）
    std::vector<float> read_f32(const std::string& name) const;

    int tensor_count() const { return tensor_count_; }

private:
    void load_metadata(gguf_context* ctx_gguf);
    void load_weights(gguf_context* ctx_gguf, const std::string& path, ggml_backend_t backend);
    // Q12：训练态 GGUF 直读——把 weight_norm 的 weight_g/weight_v 折叠为 weight
    // （只在本类内新增张量到独立 context，不改任何推理图/数值路径）
    void fold_weight_norm_tensors(ggml_backend_t backend);

    std::string architecture_;
    ggml_context* ctx_weights_ = nullptr;
    ggml_backend_buffer_t buffer_weights_ = nullptr;
    // 折叠产生的 weight 张量（独立 context + 缓冲；weight() 会一并查找）
    ggml_context* ctx_fold_ = nullptr;
    ggml_backend_buffer_t buffer_fold_ = nullptr;
    int tensor_count_ = 0;

    // 元数据副本（主机）
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
