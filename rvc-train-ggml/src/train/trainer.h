// RVC-Train-GGML - M5-3 training loop (src/train/trainer.h)
//
// Role: provide the minimal loopable training stack on top of M5-2's "one-step training core" (TrainStep):
//   - bucket data loop (M5-3 v1 single-bucket graph: bucket can be given explicitly, or take the largest of config buckets;
//     samples exceeding graph T are skipped and logged -- multi-bucket graph caching needs TrainStep to support rebuilding with a varying T, deferred to M5-4/M7, see Q25)
//   - epoch shuffling (seed+epoch, v1 simplification of the official DistributedBucketSampler; a trailing partial batch is dropped)
//   - one ExponentialLR each for G/D (stepped once per epoch, official train.py:283-288)
//   - logging (train.log + loss.csv) and periodic checkpoints (traincpp checkpoint v2, including RNG/scheduler)
//
// Semantic baseline: official 81eed5e train/train.py + data_utils.py (see `docs/handover.md` §4.9 A/D).
// M5-3 gate G-5.3: N steps with finite and sensibly trending loss; checkpoint save->load round trip (parameters/step/lr/RNG) consistent.
#pragma once

#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "data/dataset.h"
#include "model/config.h"
#include "train/resume_meta.h"
#include "train/train_step.h"

#include "traincpp/traincpp.h"

namespace rvc_train {
namespace train {

// Training run config (M5 provisional fields, finalized in M7):
//   - run.* (weights/cache/output/resume/device/export) comes from the CLI or the config `run` section;
//   - train.*/data.*/model.* may come from official configs/*.json (trainer_config_from_json);
//   - since M7-1 this is our own schema: train/data/model top-levels are kept, plus an optional `run` section (see `docs/handover.md` §4.11).
struct TrainerConfig {
    // run (M7-1: can be read from the config `run` section; CLI overrides take precedence)
    std::string weights_g;        // training-state G GGUF (config: run.pretrain_g)
    std::string weights_d;        // training-state D GGUF (config: run.pretrain_d)
    std::string exp_dir;          // M4 cache directory (config: run.exp)
    std::string out_dir;          // log/checkpoint output directory (config: run.out)
    std::string resume;           // optional: load this checkpoint and continue (config: run.resume)
    std::string export_run_json;  // optional: run summary JSON (config: run.export_run_json)
    int64_t device = -1;          // training device index (-1=auto; config: run.device; CLI --device takes precedence)
    int64_t export_every = 0;     // export inference GGUF every N epochs (0=off; config: run.export_every; M7-4)
    int64_t snapshot_keep = 0;    // number of epoch snapshots to keep (0=keep all; config: run.snapshot_keep; M7-4)
    bool    scratch = false;      // train from scratch: generate randomly initialized G/D from model_cfg (config: run.scratch; M7-2)
    // train
    int64_t batch_size = 1;
    int64_t epochs = 1;
    int64_t max_steps = 0;        // if >0, stop once this many steps are reached (smoke test/testing)
    int64_t bucket = 0;           // graph T (0 = take the largest of buckets)
    int64_t disc_len = 0;         // D input length (0 = official full segment_samples; >0 = test-scale truncation)
    int64_t log_interval = 50;
    int64_t save_interval = 0;    // save once every N epochs (0 = only at the end)
    int64_t seed = 1234;
    int64_t grad_accum = 1;       // gradient accumulation steps (effective batch = batch×accum; implemented in M7-3)
    float lr = 1e-4f;
    float beta1 = 0.8f;
    float beta2 = 0.99f;
    float eps = 1e-9f;
    float weight_decay = 1e-2f;
    float c_mel = 45.0f;
    float c_kl = 1.0f;
    float lr_decay = 0.999875f;   // ExponentialLR gamma (per epoch)
    std::vector<int64_t> buckets;
    // data (official config data.*)
    int64_t segment_frames = 32;     // z frames = segment_size/hop
    int64_t segment_samples = 12800; // official segment_size
    int64_t q_channels = 192;        // inter_channels
    int64_t n_fft = 2048;
    int64_t win = 2048;
    int64_t n_mels = 125;
    double sr = 40000.0;
    double fmin = 0.0;
    double fmax = -1.0;              // <=0 means sr/2
    float log_clip = 2e-6f;
    // Model structure (derived from the config data/model sections; used by M7-2 scratch); defaults to v1-40k
    model::ModelConfig model_cfg = model::default_model_config_v1_40k();   // kind set to "G"; D is derived separately
};

// Read the training config (train/data/model + the optional run section added in M7-1).
// Compatible with official configs/*.json (struct defaults apply when there is no run section); buckets/save_interval/disc_len are M5 extensions.
bool trainer_config_from_json(const std::string& path, TrainerConfig* out, std::string* err);
// Same as above, but parses in-memory text directly (for wizard equivalence self-tests/tests)
bool trainer_config_from_json_text(const std::string& text, TrainerConfig* out, std::string* err);

// M7-1: serialize the config to our own schema text (train/data/model/run); round-trip equivalent with trainer_config_from_json
bool trainer_config_to_json(const TrainerConfig& cfg, std::string* out, std::string* err);
// M7-1: write the config to a file (parent directories are created automatically)
bool trainer_config_write_json(const TrainerConfig& cfg, const std::string& path, std::string* err);
// M7-1: compare two configs field by field (including the run section); on mismatch write the first difference to *diff (may be nullptr)
bool trainer_config_equivalent(const TrainerConfig& a, const TrainerConfig& b, std::string* diff);

// Training loop (single-bucket graph v1): create -> run (or step-by-step run_one_step) -> final checkpoint.
class Trainer {
public:
    static Trainer* create(const TrainerConfig& cfg, traincpp::Device* dev, std::string* err);
    ~Trainer();

    Trainer(const Trainer&) = delete;
    Trainer& operator=(const Trainer&) = delete;

    // Full loop: run according to epochs/max_steps; save the final checkpoint (if out_dir is set) and the run JSON
    bool run(std::string* err);
    // Advance one step (scheduler stepped at epoch boundaries; periodic checkpointing at epoch boundaries)
    bool run_one_step(std::string* err);

    // checkpoint (traincpp schema v2: G/D partitions + shared RNG + per-partition scheduler)
    bool save_checkpoint(const std::string& path, std::string* err);
    bool load_checkpoint(const std::string& path, std::string* err);
    // Read-only verification: file contents match the current in-memory state item by item (caller guarantees the in-memory state is the archived state)
    bool verify_checkpoint_memory(const std::string& path, std::string* err);
    // ---------------- accessors (logging/testing/export) ----------------
    int64_t step_count() const;
    int64_t epoch_index() const { return epoch_; }
    int64_t batch_pos() const { return batch_pos_; }
    int64_t steps_per_epoch() const { return steps_per_epoch_; }
    int64_t bucket() const { return bucket_; }
    int64_t trainable_samples() const { return (int64_t) sample_indices_.size(); }
    int64_t skipped_samples() const { return skipped_samples_; }
    float lr_g() const;
    float lr_d() const;
    uint64_t rng_state() const { return rng_.state; }
    bool finished() const { return finished_; }
    // M5-4 performance capture
    double first_step_ms() const { return first_step_ms_; }
    double mean_step_ms() const {
        return timed_steps_ > 0 ? total_step_ms_ / (double) timed_steps_ : 0.0;
    }
    int64_t timed_steps() const { return timed_steps_; }
    const std::vector<TrainStepLosses>& first_losses() const { return first_losses_; }
    const TrainStepLosses& last_losses() const { return last_; }
    const TrainerConfig& config() const { return cfg_; }
    const data::Dataset& dataset() const { return ds_; }
    const std::string& checkpoint_path() const { return ckpt_path_; }
    const std::vector<int64_t>& sample_indices() const { return sample_indices_; }
    TrainStep* train_step() { return ts_.get(); }

private:
    Trainer() = default;
    bool fail(std::string* err, const std::string& msg) const;
    bool rebuild_epoch_batches(int64_t epoch, std::string* err);
    bool build_batch_into(const std::vector<int64_t>& indices, data::BatchBuffers* out,
                          std::string* err);
    bool open_logs(std::string* err);
    void log_step(const TrainStepLosses& losses, bool to_console);
    void write_run_json() const;
    // M7-4: epoch boundary (epoch_ already incremented) -- epoch summary line + periodic snapshots + inference GGUF export every N epochs
    bool on_epoch_end(std::string* err);
    void write_epoch_summary();
    bool save_epoch_snapshot(int64_t epoch, std::string* err);
    bool prune_snapshots(std::string* err);
    bool export_epoch_inference(int64_t epoch, std::string* err);
    // M7-5: resume metadata (sidecar <ckpt>.meta.json) -- write/sample fingerprint/load validation
    ResumeMeta build_resume_meta() const;
    void compute_sample_fingerprint(int64_t* count, int64_t* total_frames, uint64_t* id_hash) const;
    bool validate_resume_meta(const ResumeMeta& m, std::string* err) const;
    bool write_resume_meta(const std::string& ckpt_path, std::string* err);

    TrainerConfig cfg_;
    traincpp::Device* dev_ = nullptr;
    data::Dataset ds_;
    std::unique_ptr<TrainStep> ts_;
    traincpp::LrScheduler* sched_g_ = nullptr;
    traincpp::LrScheduler* sched_d_ = nullptr;
    traincpp::Rng rng_;

    int64_t bucket_ = 0;
    int64_t epoch_ = 0;
    int64_t batch_pos_ = 0;
    int64_t steps_per_epoch_ = 0;
    int64_t skipped_samples_ = 0;
    int64_t last_log_step_ = -1;
    bool finished_ = false;

    std::vector<int64_t> sample_indices_;              // trainable samples (Dataset indices)
    std::vector<std::vector<int64_t>> epoch_batches_;  // batches of the current epoch (length = steps_per_epoch_)
    bool epoch_batches_valid_ = false;
    int64_t epoch_batches_epoch_ = -1;

    TrainStepLosses last_{};
    // M5-4 performance capture (step time + first N losses; does not affect the numerical path)
    static constexpr int64_t kFirstLossCount = 10;
    double first_step_ms_ = 0.0;
    double total_step_ms_ = 0.0;
    int64_t timed_steps_ = 0;
    std::vector<TrainStepLosses> first_losses_;
    std::string ckpt_path_;
    std::string log_path_;
    std::string csv_path_;
    std::string epochs_csv_path_;  // M7-4: per-epoch summary
    std::ofstream log_file_;
    std::ofstream csv_file_;
    std::ofstream epochs_file_;
};

} // namespace train
} // namespace rvc_train
