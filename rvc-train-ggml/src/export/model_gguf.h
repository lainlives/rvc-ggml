// model_gguf.h —— M6：把训练态 G GGUF 导出为 RVC-GGML 可读推理 GGUF
//
// 约定（逐条对照 RVC-GGML/scripts/export_gguf.py 的 model 导出）：
//   - 只导推理子集：enc_p / flow / dec / emb_g（滤掉 enc_q 与判别器）
//   - 折叠 weight_norm：w = g·v/‖v‖（标准 dim0 约定）
//   - Conv1d 展平为 conv_flat v2：[O,I,K] → (O, I*K)；dec.ups.*（ConvT1d）不展平
//   - 写全 rvc.* 元数据（含 upsample_rates/kernel_sizes 必写、conv_flat=true）
//
// 实现走 train.cpp 自带 GGUF 读写（traincpp::GgufFile / GgufWriter），
// 不依赖 ggml；编入 rvc-train（纯主机，无需设备）。
#pragma once

#include <string>
#include <unordered_map>
#include <vector>

namespace rvc_train {
namespace exporter {

struct ModelExportOptions {
    std::string in_path;       // 训练态 G GGUF（general.architecture = "rvc-train"）
    std::string out_path;      // 输出推理 GGUF（general.architecture = "rvc"）
    std::string general_name;  // 可选 general.name；空则用输出文件主名
    // 当输入训练态 GGUF 因尾部 1 维折叠丢失真实维数（如 O=1 的 Conv1d 权重）时，
    // 用来源文件的真实 ndims==3 名单强制按 Conv1d 展平（M7-4 实时快照桥接用；
    // 列表为空时行为与 M6 完全一致）。
    std::vector<std::string> force_flatten;
};

// 导出推理 GGUF。成功返回 true；失败返回 false 并写入 err。
bool export_inference_gguf(const ModelExportOptions& opt, std::string* err);

// 扫描 GGUF 头，返回每个张量的**存储 n_dims**（低层，只读；供桥接修复折叠维）。
bool gguf_read_tensor_ndims(const std::string& path, std::unordered_map<std::string, int>* out,
                            std::string* err);

} // namespace exporter
} // namespace rvc_train
