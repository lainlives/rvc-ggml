// RVC-Train-GGML - M5-3 训练循环自检入口（src/train/trainer_selftest.h）
//
// G-5.3 验证（Trainer + checkpoint 往返）：
//   A = Trainer（真实缓存数据）跑 steps 步 → 检查 6 loss 有限 → 保存 checkpoint →
//   checkpoint_verify（文件与 A 内存状态一致）→ A 再跑 1 步记录 loss/step/lr/RNG；
//   B = 全新 Trainer 载入同一 checkpoint → 状态核对 + checkpoint_verify →
//   B 再跑 1 步；要求 B 的下一步 6 loss 与 A 逐步位一致（step/lr/RNG 亦一致）。
// 纯 Vulkan/CPU 皆可（本机小规模用 Vulkan；大参数由 CLI 控制）。
#pragma once

#include <cstdint>
#include <string>

namespace traincpp {
class Device;
}

namespace rvc_train {
namespace train {

// weights_g/weights_d = 训练态 GGUF；exp_dir = M4 缓存；out_dir = 日志/checkpoint 输出。
// bucket/batch/disc_len 为小规模测试参数（disc_len=0 表示官方整段 segment_samples）。
// config_path（V2-6 起）：非空时以此为训练配置基底（如 configs/v2-48k-f0.json），再覆盖上述测试参数；
//   用于 v2 缓存（hop/sr/特征维与 v1 默认不同）。空 = 旧行为（内置 v1 默认）。
// 返回 0 = 全部通过；1 = 失败；2 = 参数/环境错误。
int run_trainer_selftest(const std::string& weights_g, const std::string& weights_d,
                         const std::string& exp_dir, const std::string& out_dir,
                         traincpp::Device* dev, int64_t steps, int64_t bucket, int64_t batch,
                         int64_t disc_len, int64_t grad_accum = 1,
                         const std::string& config_path = std::string());

} // namespace train
} // namespace rvc_train
