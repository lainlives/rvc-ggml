// RVC-Train-GGML - M5-3 训练循环（src/train/trainer.h）
//
// 职责：在 M5-2 的"一步训练核心"（TrainStep）之上提供可循环运行的最小训练栈：
//   - bucket 数据循环（M5-3 v1 单桶图：bucket 可显式指定，或取 config buckets 最大者；
//     超出图 T 的样本跳过并日志——多桶图缓存需 TrainStep 支持变 T 重建，留 M5-4/M7，见 Q25）
//   - epoch 打乱（seed+epoch，官方 DistributedBucketSampler 的 v1 简化；尾部不足一个 batch 丢弃）
//   - G/D 各一个 ExponentialLR（每 epoch 各步进一次，官方 train.py:283-288）
//   - 日志（train.log + loss.csv）与周期性 checkpoint（traincpp checkpoint v2，含 RNG/调度器）
//
// 语义基准：官方 81eed5e train/train.py + data_utils.py（见 `交接说明` §4.9 A/D）。
// M5-3 门 G-5.3：N 步 loss 有限且趋势合理；checkpoint save→load 往返（参数/step/lr/RNG）一致。
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

// 训练运行配置（M5 临时字段，M7 定稿）：
//   - run.*（权重/缓存/输出/续训/设备/导出）由 CLI 或 config `run` 段提供；
//   - train.*/data.*/model.* 可来自官方 configs/*.json（trainer_config_from_json）；
//   - M7-1 起为自有 schema：train/data/model 顶层保留，新增可选 `run` 段（见 `交接说明` §4.11）。
struct TrainerConfig {
    // run（M7-1：可由 config `run` 段读取；CLI 覆盖优先）
    std::string weights_g;        // 训练态 G GGUF（config: run.pretrain_g）
    std::string weights_d;        // 训练态 D GGUF（config: run.pretrain_d）
    std::string exp_dir;          // M4 缓存目录（config: run.exp）
    std::string out_dir;          // 日志/checkpoint 输出目录（config: run.out）
    std::string resume;           // 可选：载入 checkpoint 后继续（config: run.resume）
    std::string export_run_json;  // 可选：运行摘要 JSON（config: run.export_run_json）
    int64_t device = -1;          // 训练设备下标（-1=自动；config: run.device；CLI --device 优先）
    int64_t export_every = 0;     // 每 N epoch 导出推理 GGUF（0=关；config: run.export_every；M7-4）
    int64_t snapshot_keep = 0;    // epoch 快照保留数（0=全部保留；config: run.snapshot_keep；M7-4）
    bool    scratch = false;      // 从零训练：按 model_cfg 生成随机初始化 G/D（config: run.scratch；M7-2）
    // train
    int64_t batch_size = 1;
    int64_t epochs = 1;
    int64_t max_steps = 0;        // >0 时达到步数即停（冒烟/测试）
    int64_t bucket = 0;           // 图 T（0 = 取 buckets 最大者）
    int64_t disc_len = 0;         // D 输入长度（0 = 官方整段 segment_samples；>0 为测试尺度截断）
    int64_t log_interval = 50;
    int64_t save_interval = 0;    // 每 N 个 epoch 存一次（0 = 仅收尾）
    int64_t seed = 1234;
    int64_t grad_accum = 1;       // 梯度累积步数（有效 batch = batch×accum；M7-3 实现）
    float lr = 1e-4f;
    float beta1 = 0.8f;
    float beta2 = 0.99f;
    float eps = 1e-9f;
    float weight_decay = 1e-2f;
    float c_mel = 45.0f;
    float c_kl = 1.0f;
    float lr_decay = 0.999875f;   // ExponentialLR gamma（每 epoch）
    std::vector<int64_t> buckets;
    // data（官方 config data.*）
    int64_t segment_frames = 32;     // z 帧 = segment_size/hop
    int64_t segment_samples = 12800; // 官方 segment_size
    int64_t q_channels = 192;        // inter_channels
    int64_t n_fft = 2048;
    int64_t win = 2048;
    int64_t n_mels = 125;
    double sr = 40000.0;
    double fmin = 0.0;
    double fmax = -1.0;              // <=0 表示 sr/2
    float log_clip = 2e-6f;
    // 模型结构（config data/model 段推导；M7-2 scratch 用）；默认 v1-40k
    model::ModelConfig model_cfg = model::default_model_config_v1_40k();   // kind 置 "G"；D 侧另派生
};

// 读取训练 config（train/data/model + M7-1 新增可选 run 段）。
// 兼容官方 configs/*.json（不含 run 段时按结构体默认值）；buckets/save_interval/disc_len 为 M5 扩展字段。
bool trainer_config_from_json(const std::string& path, TrainerConfig* out, std::string* err);
// 同上，但直接解析内存文本（供向导等价自检/测试使用）
bool trainer_config_from_json_text(const std::string& text, TrainerConfig* out, std::string* err);

// M7-1：把配置序列化为自有 schema 文本（train/data/model/run），与 trainer_config_from_json 往返等价
bool trainer_config_to_json(const TrainerConfig& cfg, std::string* out, std::string* err);
// M7-1：把配置写入文件（自动创建父目录）
bool trainer_config_write_json(const TrainerConfig& cfg, const std::string& path, std::string* err);
// M7-1：逐字段比较两配置（含 run 段）；不等价时把首个差异写入 *diff（可为 nullptr）
bool trainer_config_equivalent(const TrainerConfig& a, const TrainerConfig& b, std::string* diff);

// 训练循环（单桶图 v1）：create → run（或逐步 run_one_step）→ 收尾 checkpoint。
class Trainer {
public:
    static Trainer* create(const TrainerConfig& cfg, traincpp::Device* dev, std::string* err);
    ~Trainer();

    Trainer(const Trainer&) = delete;
    Trainer& operator=(const Trainer&) = delete;

    // 完整循环：按 epochs/max_steps 运行，收尾保存 checkpoint（若有 out_dir）与 run JSON
    bool run(std::string* err);
    // 单步推进（epoch 边界执行 scheduler；周期性 checkpoint 在 epoch 边界）
    bool run_one_step(std::string* err);

    // checkpoint（traincpp schema v2：G/D 双分区 + 共享 RNG + 每分区调度器）
    bool save_checkpoint(const std::string& path, std::string* err);
    bool load_checkpoint(const std::string& path, std::string* err);
    // 只读校验：文件内容与当前内存状态逐项一致（调用方保证内存状态即存档状态）
    bool verify_checkpoint_memory(const std::string& path, std::string* err);
    // ---------------- 访问器（日志/测试/导出） ----------------
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
    // M5-4 性能采集
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
    // M7-4：epoch 边界（epoch_ 已自增）——epoch 汇总行 + 周期性快照 + 每 N epoch 导出推理 GGUF
    bool on_epoch_end(std::string* err);
    void write_epoch_summary();
    bool save_epoch_snapshot(int64_t epoch, std::string* err);
    bool prune_snapshots(std::string* err);
    bool export_epoch_inference(int64_t epoch, std::string* err);
    // M7-5：续训元数据（旁挂 <ckpt>.meta.json）——写/样本指纹/载入校验
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

    std::vector<int64_t> sample_indices_;              // 可训练样本（Dataset 下标）
    std::vector<std::vector<int64_t>> epoch_batches_;  // 当前 epoch 的批次（长度 = steps_per_epoch_）
    bool epoch_batches_valid_ = false;
    int64_t epoch_batches_epoch_ = -1;

    TrainStepLosses last_{};
    // M5-4 性能采集（步时 + 前 N 步 loss；不影响数值路径）
    static constexpr int64_t kFirstLossCount = 10;
    double first_step_ms_ = 0.0;
    double total_step_ms_ = 0.0;
    int64_t timed_steps_ = 0;
    std::vector<TrainStepLosses> first_losses_;
    std::string ckpt_path_;
    std::string log_path_;
    std::string csv_path_;
    std::string epochs_csv_path_;  // M7-4：per-epoch 汇总
    std::ofstream log_file_;
    std::ofstream csv_file_;
    std::ofstream epochs_file_;
};

} // namespace train
} // namespace rvc_train
