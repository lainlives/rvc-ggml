// RVC-Train-GGML - M7-5 resume metadata (src/train/resume_meta.h)
//
// Background: traincpp checkpoint v2 stores only parameters/optimizer state/step/scheduler epoch/RNG, not
//   "epoch / optimizer step position within the epoch / training-related config fingerprint / sample-set fingerprint"; Trainer used to infer
//   epoch/batch_pos from `step / steps_per_epoch`; if batch_size/grad_accum/bucket/
//   seed etc. at resume time differ from the archive, the position silently desyncs (see `docs/handover.md` §4.11 M7-5, `environment and pitfalls`).
//
// Plan (user decision 2026-09-28): attach a meta JSON next to the checkpoint
//   (`train.ckpt.gguf` -> `train.ckpt.meta.json`; `train.ep{N}.ckpt.gguf` -> `train.ep{N}.ckpt.meta.json`),
//   recording epoch/batch_pos/step (cross-validated) / grad_accum + training-related config signature + sample fingerprint.
//   On resume, any mismatch is reported as an explicit error and exit (no silent desync).
//   - Do not touch the traincpp library (public API has no arbitrary-metadata interface), do not change the GGUF schema;
//   - Fingerprint definition (user decision): sample fingerprint = sample count + total frames + id list hash.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace rvc_train {
namespace train {

struct TrainerConfig;  // forward declaration (trainer.h)

// Resume metadata (sidecar JSON).
struct ResumeMeta {
    int      version = 1;        // meta version
    int64_t  epoch = 0;          // epoch at archive time (post-increment convention, same as Trainer::epoch_index)
    int64_t  batch_pos = 0;      // optimizer step position within the epoch (same as Trainer::batch_pos)
    int64_t  step = 0;           // G optimizer step count (cross-checked against the checkpoint)
    int64_t  grad_accum = 1;     // gradient accumulation steps
    uint64_t config_hash = 0;    // training-related config signature FNV-1a64 (stored as a hex string)
    int64_t  sample_count = 0;   // total number of cached samples
    int64_t  total_frames = 0;   // sum of aligned frames over all samples
    uint64_t id_hash = 0;        // sample id list hash (FNV-1a64)
    std::vector<std::string> config_signature;  // training-related "key=value" lines
};

// Checkpoint path -> meta path: strip ".gguf" then append ".meta.json" (append directly if there is no ".gguf").
std::string resume_meta_path_for(const std::string& ckpt_path);

// Canonical signature of training-related config (excludes runtime fields such as run section/out/device/resume/export/log_interval;
// includes model structure + data/optimization hyperparameters that affect the training trajectory).
std::vector<std::string> trainer_config_signature(const TrainerConfig& cfg);

// FNV-1a 64-bit hash (over the string bytes).
uint64_t resume_fnv1a64(const std::string& s);

// Write/read meta JSON; on failure writes *err.
bool resume_meta_write(const ResumeMeta& m, const std::string& path, std::string* err);
bool resume_meta_read(const std::string& path, ResumeMeta* out, std::string* err);

} // namespace train
} // namespace rvc_train
