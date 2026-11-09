// gguf_writer.h -- M4-5a: minimal GGUF writer built on ggml's built-in gguf_* API
//
// Supports only F32 tensors + string/u32 KV (what index.gguf needs); layout/alignment/endianness are guaranteed by the ggml writer
// (GGUF v3, alignment 32, tensor info in ggml ne order, 32-byte alignment per tensor).
// Data pointers are owned by the caller until write() returns (gguf_set_tensor_data only stores the pointer).

#pragma once

#include <cstdint>
#include <string>

#include <ggml.h>
#include <gguf.h>

namespace rvc_train::exporter {

class GgufWriter {
public:
    GgufWriter();
    ~GgufWriter();
    GgufWriter(const GgufWriter&) = delete;
    GgufWriter& operator=(const GgufWriter&) = delete;

    void set_str(const std::string& key, const std::string& value);
    void set_u32(const std::string& key, uint32_t value);

    // numpy shape [rows, cols] (F32 row-major) -> ggml ne=[cols, rows]; data is not transposed
    bool add_f32_2d(const std::string& name, const float* data, int64_t rows, int64_t cols,
                    std::string* err);
    // numpy shape [n] -> ggml ne=[n]
    bool add_f32_1d(const std::string& name, const float* data, int64_t n, std::string* err);

    bool write(const std::string& path, std::string* err);

private:
    struct Impl;
    Impl* impl_;
};

} // namespace rvc_train::exporter
