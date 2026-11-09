// RVC-Train-GGML - M7-2 in-memory -> training-state GGUF bridge (src/export/mem_to_train_gguf.h)
//
// For training from scratch, generate randomly initialized training-state GGUF (G/D) from ModelConfig, then use the existing
// TrainWeights/TrainStep path (no changes to the training core). Metadata keys match tools/pth_to_train_gguf.py
// exactly (rvc.train.*), so TrainMeta::read_meta / ModelConfig::validate pass.
//
// Also lays the groundwork for M7-4: this bridge is the general exit "in-memory parameters -> training-state GGUF".
#pragma once

#include "model/config.h"
#include "model/model.h"
#include "model/scratch.h"

#include <cstdint>
#include <string>
#include <vector>

namespace rvc_train {
namespace exporter {

// Write training-state GGUF from already-generated specs/data (sizes must match; general.name/source_pth use defaults).
bool write_scratch_train_gguf(const model::ModelConfig& mc,
                              const std::vector<model::TensorSpec>& specs,
                              const std::vector<std::vector<float>>& data, const std::string& path,
                              std::string* err);

// General entry point (M7-4): write training-state GGUF from specs/data; general.name and
// rvc.train.source_pth can be specified (empty uses the M7-2 scratch defaults). Keys match pth_to_train_gguf.py.
bool write_train_gguf(const model::ModelConfig& mc,
                      const std::vector<model::TensorSpec>& specs,
                      const std::vector<std::vector<float>>& data, const std::string& path,
                      const std::string& general_name, const std::string& source_pth,
                      std::string* err);

// M7-4: snapshot training-state GGUF from live parameters (TrainStep's G ModelParams) (keeps weight_g/weight_v,
// without folding). params order = file order; tensor_get each tensor to the host, then write to disk.
bool export_train_gguf_live(const model::ModelConfig& g_cfg, const model::ModelParams& g_params,
                            const std::string& path, const std::string& general_name,
                            const std::string& source_pth, std::string* err);

// Convenience entry points: generate specs from config -> initialize -> write file. seed controls deterministic initialization.
bool generate_scratch_g(const model::ModelConfig& g_cfg, uint64_t seed, const std::string& path,
                        std::string* err);
bool generate_scratch_d(const model::ModelConfig& d_cfg, uint64_t seed, const std::string& path,
                        std::string* err);

} // namespace exporter
} // namespace rvc_train
