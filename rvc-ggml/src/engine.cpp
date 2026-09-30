// engine.cpp
// 后端管理与通用图执行引擎实现。

#include "engine.h"

#include "utils.h"

#include <ggml-alloc.h>
#include <ggml-backend.h>

#include <stdexcept>

namespace rvc {

namespace {

std::vector<ggml_backend_dev_t> gpu_devices() {
    std::vector<ggml_backend_dev_t> gpus;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        const enum ggml_backend_dev_type t = ggml_backend_dev_type(d);
        if (t == GGML_BACKEND_DEVICE_TYPE_GPU || t == GGML_BACKEND_DEVICE_TYPE_IGPU) {
            gpus.push_back(d);
        }
    }
    return gpus;
}

} // namespace

std::vector<GpuDeviceInfo> list_gpu_devices() {
    ggml_backend_load_all();
    std::vector<GpuDeviceInfo> out;
    const auto gpus = gpu_devices();
    for (size_t i = 0; i < gpus.size(); ++i) {
        GpuDeviceInfo e;
        e.index = static_cast<int>(i);
        e.name = ggml_backend_dev_name(gpus[i]);
        const char* desc = ggml_backend_dev_description(gpus[i]);
        e.description = desc ? desc : "";
        out.push_back(e);
    }
    return out;
}

GgmlDevice::GgmlDevice(int device_id) {
    ggml_backend_load_all();
    if (device_id != 99) {
        const auto gpus = gpu_devices();
        if (device_id >= 0 && device_id < static_cast<int>(gpus.size())) {
            backend_ = ggml_backend_dev_init(gpus[device_id], nullptr);
            if (!backend_) {
                Logger::warn("GPU 设备初始化失败，回退 CPU");
            }
        }
        else {
            Logger::warn("GPU 设备 ID " + std::to_string(device_id) +
                " 越界（可用 " + std::to_string(gpus.size()) + " 个），回退 CPU");
        }
    }
    if (!backend_) {
        ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (!cpu) throw std::runtime_error("无可用 CPU 后端");
        backend_ = ggml_backend_dev_init(cpu, nullptr);
        if (!backend_) throw std::runtime_error("CPU 后端初始化失败");
    }
    name_ = ggml_backend_name(backend_);
}

GgmlDevice::~GgmlDevice() {
    if (backend_) {
        ggml_backend_free(backend_);
        backend_ = nullptr;
    }
}

GgmlGraphRunner::GgmlGraphRunner(ggml_backend_t backend, size_t ctx_mem_size, int max_nodes)
    : backend_(backend) {
    ggml_init_params params = { ctx_mem_size, nullptr, true };
    ctx_ = ggml_init(params);
    if (!ctx_) {
        throw std::runtime_error("ggml 上下文创建失败");
    }
    graph_ = ggml_new_graph_custom(ctx_, static_cast<size_t>(max_nodes), false);
    if (!graph_) {
        ggml_free(ctx_);
        throw std::runtime_error("ggml 计算图创建失败");
    }
}

GgmlGraphRunner::~GgmlGraphRunner() {
    if (allocr_) {
        ggml_gallocr_free(allocr_);
        allocr_ = nullptr;
    }
    if (ctx_) {
        ggml_free(ctx_);
        ctx_ = nullptr;
    }
}

void GgmlGraphRunner::expand(ggml_tensor* node) {
    ggml_build_forward_expand(graph_, node);
}

bool GgmlGraphRunner::allocate() {
    allocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
    if (!allocr_) return false;
    if (!ggml_gallocr_reserve(allocr_, graph_)) {
        Logger::warn("gallocr_reserve 失败，尝试直接分配");
    }
    if (!ggml_gallocr_alloc_graph(allocr_, graph_)) {
        return false;
    }
    buffer_size_ = ggml_gallocr_get_buffer_size(allocr_, 0);
    node_count_ = ggml_graph_n_nodes(graph_);
    return true;
}

void GgmlGraphRunner::compute() {
    const enum ggml_status st = ggml_backend_graph_compute(backend_, graph_);
    if (st != GGML_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("GGML 计算失败: ") + ggml_status_to_string(st));
    }
}

void GgmlGraphRunner::set_input(ggml_tensor* t, const void* data, size_t size) {
    if (size != ggml_nbytes(t)) {
        throw std::runtime_error(std::string("输入尺寸不匹配: ") + t->name);
    }
    ggml_backend_tensor_set(t, data, 0, size);
}

void GgmlGraphRunner::get_output(ggml_tensor* t, void* data, size_t size) const {
    if (size != ggml_nbytes(t)) {
        throw std::runtime_error(std::string("输出尺寸不匹配: ") + t->name);
    }
    ggml_backend_tensor_get(t, data, 0, size);
}

} // namespace rvc
