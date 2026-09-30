// RVC-Train-GGML - M7-2 内存 → 训练态 GGUF 桥接（src/export/mem_to_train_gguf.h）
//
// 从零训练时按 ModelConfig 生成随机初始化的训练态 GGUF（G/D），随后走既有
// TrainWeights/TrainStep 路径（无需改动训练核心）。元数据键与 tools/pth_to_train_gguf.py
// 完全一致（rvc.train.*），保证 TrainMeta::read_meta / ModelConfig::validate 通过。
//
// 另为 M7-4 铺路：本桥接即"内存参数 → 训练态 GGUF"的通用出口。
#pragma once

#include "model/config.h"
#include "model/model.h"
#include "model/scratch.h"

#include <cstdint>
#include <string>
#include <vector>

namespace rvc_train {
namespace exporter {

// 依据已生成的 specs/data 写出训练态 GGUF（size 必须一致；general.name/source_pth 取默认）。
bool write_scratch_train_gguf(const model::ModelConfig& mc,
                              const std::vector<model::TensorSpec>& specs,
                              const std::vector<std::vector<float>>& data, const std::string& path,
                              std::string* err);

// 通用入口（M7-4）：依据 specs/data 写出训练态 GGUF，可指定 general.name 与
// rvc.train.source_pth（空则用 M7-2 scratch 默认值）。键与 pth_to_train_gguf.py 一致。
bool write_train_gguf(const model::ModelConfig& mc,
                      const std::vector<model::TensorSpec>& specs,
                      const std::vector<std::vector<float>>& data, const std::string& path,
                      const std::string& general_name, const std::string& source_pth,
                      std::string* err);

// M7-4：从实时参数（TrainStep 的 G ModelParams）快照训练态 GGUF（保留 weight_g/weight_v，
// 不折叠）。params 顺序 = 文件顺序；逐张量 tensor_get 到主机后写盘。
bool export_train_gguf_live(const model::ModelConfig& g_cfg, const model::ModelParams& g_params,
                            const std::string& path, const std::string& general_name,
                            const std::string& source_pth, std::string* err);

// 便捷入口：按配置生成规格 → 初始化 → 写文件。seed 控制确定性初始化。
bool generate_scratch_g(const model::ModelConfig& g_cfg, uint64_t seed, const std::string& path,
                        std::string* err);
bool generate_scratch_d(const model::ModelConfig& d_cfg, uint64_t seed, const std::string& path,
                        std::string* err);

} // namespace exporter
} // namespace rvc_train
