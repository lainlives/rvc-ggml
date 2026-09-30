// gguf_writer.h —— M4-5a：基于 ggml 自带 gguf_* API 的最小 GGUF 写出器
//
// 只支持 F32 张量 + string/u32 KV（index.gguf 的需求）；布局/对齐/字节序由 ggml writer 保证
// （GGUF v3、alignment 32、tensor info 按 ggml ne 顺序、每张量 32 对齐）。
// 数据指针由调用方持有，直到 write() 返回为止（gguf_set_tensor_data 只存指针）。

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

    // numpy 形状 [rows, cols]（F32 行主序）→ ggml ne=[cols, rows]；数据不转置
    bool add_f32_2d(const std::string& name, const float* data, int64_t rows, int64_t cols,
                    std::string* err);
    // numpy 形状 [n] → ggml ne=[n]
    bool add_f32_1d(const std::string& name, const float* data, int64_t n, std::string* err);

    bool write(const std::string& path, std::string* err);

private:
    struct Impl;
    Impl* impl_;
};

} // namespace rvc_train::exporter
