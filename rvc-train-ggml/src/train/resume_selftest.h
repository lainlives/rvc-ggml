// RVC-Train-GGML - M7-5 续训自检入口（src/train/resume_selftest.h）
//
// 门 G-7.5：续训与不中断训练一致（逐位）+ 换配置明确报错。
// 协议：
//   A  = 全新 Trainer 跑 total_steps 步（跨 epoch 边界），记录每步 6 loss 与状态轨迹；
//   B1 = 全新 Trainer 跑 pre_steps 步（状态须与 A 一致）→ 保存 checkpoint（含 meta）；
//   B2 = 全新 Trainer 载入该 checkpoint（状态须与 A 的 pre 处一致）→ 续跑到 total_steps，
//        续跑段 6 loss 与状态须与 A 对应步**逐位一致**；
//   负向：batch_size/bucket/seed/grad_accum/learning_rate 改动后续训 → 必须明确报错；
//         篡改 meta 的样本指纹后续训 → 必须明确报错。
// 建议用很小的桶（如 bucket=200：cache_v1 仅 2 个样本 → 2 步/epoch）让 epoch 边界在少数步内跨越。
#pragma once

#include <cstdint>
#include <string>

namespace traincpp {
class Device;
}

namespace rvc_train {
namespace train {

// weights_g/weights_d = 训练态 GGUF；exp_dir = M4 缓存；out_dir = 日志/checkpoint 输出。
// total_steps >= 2；pre_steps 在 (0, total_steps)。
// config_path（V2-6 起）：非空时以此为训练配置基底（如 configs/v2-48k-f0.json），再覆盖测试参数；
//   用于 v2 缓存（hop/sr/特征维与 v1 默认不同）。空 = 旧行为（内置 v1 默认）。
// 返回 0 = 全部通过；1 = 失败；2 = 参数/环境错误。
int run_resume_selftest(const std::string& weights_g, const std::string& weights_d,
                        const std::string& exp_dir, const std::string& out_dir,
                        traincpp::Device* dev, int64_t total_steps, int64_t pre_steps,
                        int64_t bucket, int64_t batch, int64_t disc_len, int64_t grad_accum,
                        const std::string& config_path = std::string());

} // namespace train
} // namespace rvc_train
