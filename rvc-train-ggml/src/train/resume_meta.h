// RVC-Train-GGML - M7-5 续训元数据（src/train/resume_meta.h）
//
// 背景：traincpp checkpoint v2 只存参数/优化器状态/step/调度器 epoch/RNG，不含
//   "epoch / epoch 内优化步位置 / 训练相关配置指纹 / 样本集指纹"；而 Trainer 原先由
//   `step / steps_per_epoch` 反推 epoch/batch_pos，一旦续训时 batch_size/grad_accum/bucket/
//   seed 等与存档不一致，位置会静默错位（见 `交接说明` §4.11 M7-5、`环境与踩坑`）。
//
// 方案（用户 2026-09-28 拍板）：在 checkpoint 旁挂一个 meta JSON
//   （`train.ckpt.gguf` -> `train.ckpt.meta.json`；`train.ep{N}.ckpt.gguf` -> `train.ep{N}.ckpt.meta.json`），
//   记录 epoch/batch_pos/step（交叉校验）/grad_accum + 训练相关配置签名 + 样本指纹。
//   续训时校验不一致即明确报错退出（不静默错位）。
//   - 不动 traincpp 库（公共 API 无任意元数据接口），不改 GGUF schema；
//   - 指纹口径（用户拍板）：样本指纹 = 样本数 + 总帧数 + id 列表哈希。
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace rvc_train {
namespace train {

struct TrainerConfig;  // 前向声明（trainer.h）

// 续训元数据（旁挂 JSON）。
struct ResumeMeta {
    int      version = 1;        // meta 版本
    int64_t  epoch = 0;          // 存档时的 epoch（已自增口径，同 Trainer::epoch_index）
    int64_t  batch_pos = 0;      // epoch 内优化步位置（同 Trainer::batch_pos）
    int64_t  step = 0;           // G 优化器步数（与 checkpoint 交叉校验）
    int64_t  grad_accum = 1;     // 梯度累积步数
    uint64_t config_hash = 0;    // 训练相关配置签名 FNV-1a64（十六进制字符串存盘）
    int64_t  sample_count = 0;   // 缓存样本总数
    int64_t  total_frames = 0;   // 全部样本对齐帧数之和
    uint64_t id_hash = 0;        // 样本 id 列表哈希（FNV-1a64）
    std::vector<std::string> config_signature;  // 训练相关字段 "key=value" 行
};

// checkpoint 路径 -> meta 路径：去 ".gguf" 后加 ".meta.json"（无 ".gguf" 则直接追加）。
std::string resume_meta_path_for(const std::string& ckpt_path);

// 训练相关配置的规范签名（不含 run 段/out/device/resume/export/log_interval 等运行期字段；
// 含模型结构 + 影响训练轨迹的数据/优化超参）。
std::vector<std::string> trainer_config_signature(const TrainerConfig& cfg);

// FNV-1a 64 位哈希（对字符串字节）。
uint64_t resume_fnv1a64(const std::string& s);

// 写/读 meta JSON；失败写 *err。
bool resume_meta_write(const ResumeMeta& m, const std::string& path, std::string* err);
bool resume_meta_read(const std::string& path, ResumeMeta* out, std::string* err);

} // namespace train
} // namespace rvc_train
