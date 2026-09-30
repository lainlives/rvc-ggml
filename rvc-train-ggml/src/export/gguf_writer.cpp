// gguf_writer.cpp —— M4-5a 实现（薄封装 ggml gguf_* API；见头注释）

#include "export/gguf_writer.h"

#include <cstdio>
#include <vector>

namespace rvc_train::exporter {
namespace {

bool check_name(const std::string& name, std::string* err) {
    if (name.empty() || name.size() >= GGML_MAX_NAME) {
        if (err) {
            *err = "GGUF 张量名非法（空或 ≥ GGML_MAX_NAME）: " + name;
        }
        return false;
    }
    return true;
}

} // namespace

struct GgufWriter::Impl {
    ggml_context* tensor_ctx = nullptr;
    gguf_context* gguf = nullptr;
    std::vector<ggml_tensor*> tensors;  // 仅元数据（no_alloc），数据指针由调用方持有
};

GgufWriter::GgufWriter() : impl_(new Impl()) {
    impl_->gguf = gguf_init_empty();
    ggml_init_params p{};
    p.mem_size = 16 * 1024;  // 4 个张量元数据足够
    p.mem_buffer = nullptr;
    p.no_alloc = true;
    impl_->tensor_ctx = ggml_init(p);
}

GgufWriter::~GgufWriter() {
    if (impl_->tensor_ctx != nullptr) {
        ggml_free(impl_->tensor_ctx);
    }
    if (impl_->gguf != nullptr) {
        gguf_free(impl_->gguf);
    }
    delete impl_;
}

void GgufWriter::set_str(const std::string& key, const std::string& value) {
    gguf_set_val_str(impl_->gguf, key.c_str(), value.c_str());
}

void GgufWriter::set_u32(const std::string& key, uint32_t value) {
    gguf_set_val_u32(impl_->gguf, key.c_str(), value);
}

bool GgufWriter::add_f32_2d(const std::string& name, const float* data, int64_t rows, int64_t cols,
                            std::string* err) {
    if (!check_name(name, err)) {
        return false;
    }
    ggml_tensor* t = ggml_new_tensor_2d(impl_->tensor_ctx, GGML_TYPE_F32, cols, rows);
    std::snprintf(t->name, sizeof(t->name), "%s", name.c_str());
    gguf_add_tensor(impl_->gguf, t);
    gguf_set_tensor_data(impl_->gguf, t->name, data);
    impl_->tensors.push_back(t);
    return true;
}

bool GgufWriter::add_f32_1d(const std::string& name, const float* data, int64_t n,
                            std::string* err) {
    if (!check_name(name, err)) {
        return false;
    }
    ggml_tensor* t = ggml_new_tensor_1d(impl_->tensor_ctx, GGML_TYPE_F32, n);
    std::snprintf(t->name, sizeof(t->name), "%s", name.c_str());
    gguf_add_tensor(impl_->gguf, t);
    gguf_set_tensor_data(impl_->gguf, t->name, data);
    impl_->tensors.push_back(t);
    return true;
}

bool GgufWriter::write(const std::string& path, std::string* err) {
    if (!gguf_write_to_file(impl_->gguf, path.c_str(), false)) {
        if (err) {
            *err = "gguf_write_to_file 失败: " + path;
        }
        return false;
    }
    return true;
}

} // namespace rvc_train::exporter
