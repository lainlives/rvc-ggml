// engine.h
// ggml 后端管理与通用图执行引擎：设备枚举、图构建/缓存、输入输出读写。

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <ggml.h>
#include <ggml-backend.h>

namespace rvc {

    struct GpuDeviceInfo {
        int index = 0;
        std::string name;
        std::string description;
    };

    // 枚举 GPU 类设备（Vulkan/CUDA 等）
    std::vector<GpuDeviceInfo> list_gpu_devices();

    // 设备管理（RAII）
    class GgmlDevice {
    public:
        // device_id: >=0 = 第 N 个 GPU 类设备；99 = 强制 CPU
        explicit GgmlDevice(int device_id = 99);
        ~GgmlDevice();
        GgmlDevice(const GgmlDevice&) = delete;
        GgmlDevice& operator=(const GgmlDevice&) = delete;

        ggml_backend_t backend() const { return backend_; }
        const std::string& name() const { return name_; }

    private:
        ggml_backend_t backend_ = nullptr;
        std::string name_;
    };

    // 通用图执行引擎：构建图并运行
    class GgmlGraphRunner {
    public:
        GgmlGraphRunner(ggml_backend_t backend, size_t ctx_mem_size, int max_nodes);
        ~GgmlGraphRunner();
        GgmlGraphRunner(const GgmlGraphRunner&) = delete;
        GgmlGraphRunner& operator=(const GgmlGraphRunner&) = delete;

        ggml_context* ctx() const { return ctx_; }
        ggml_cgraph* graph() const { return graph_; }

        // 扩展图（构建阶段调用）：把节点展开到计算图
        void expand(ggml_tensor* node);

        // 分配计算缓冲（图构建完成后调用一次；返回 false 表示失败）
        bool allocate();

        size_t buffer_size() const { return buffer_size_; }
        int node_count() const { return node_count_; }

        // 执行图
        void compute();

        // 上传输入 / 读取输出（主机侧 vector）
        void set_input(ggml_tensor* t, const void* data, size_t size);
        void get_output(ggml_tensor* t, void* data, size_t size) const;

        template <typename T>
        void set_input_vec(ggml_tensor* t, const std::vector<T>& v) {
            set_input(t, v.data(), v.size() * sizeof(T));
        }
        template <typename T>
        void get_output_vec(ggml_tensor* t, std::vector<T>& v) const {
            v.resize(static_cast<size_t>(ggml_nelements(t)));
            get_output(t, v.data(), v.size() * sizeof(T));
        }

        ggml_backend_t backend() const { return backend_; }

    private:
        ggml_backend_t backend_ = nullptr;
        ggml_context* ctx_ = nullptr;
        ggml_cgraph* graph_ = nullptr;
        ggml_gallocr_t allocr_ = nullptr;
        size_t buffer_size_ = 0;
        int node_count_ = 0;
    };

} // namespace rvc
