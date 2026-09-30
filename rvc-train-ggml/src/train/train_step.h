// RVC-Train-GGML - M5-2 最小训练步核心（src/train/train_step.h）
//
// 单 ctx 共享参数双图（对齐 train.cpp tests/test_acceptance.cpp 的验收形态；
// 交接说明 §4.9 C / 集成指南 §5.4 路线 2）：
//   - 同一 Context 内 G/D 参数各建一次；G 图含 synth 切片前向 + D 冻结前向 + 4 项 G 损失；
//     D 图为独立叶子（wave_d/fake_d，主机拷贝 = detach）上的**新建** D 前向 + LSGAN(D)。
//   - 步序（官方 81eed5e train.py:425-527）：G 预热前向（取 y_hat）→ 主机拷 y_hat 到
//     fake_d → D 步（graph_reset/compute/optim_step）→ G 步（重置重算，D 节点读更新后权重）。
//   - 分配顺序：G 图反向构建后 buffer_alloc_graph_tensors(gg)，再建 D 图并
//     buffer_alloc_graph_tensors(dg)（G 侧张量已有 data 会被跳过）。
//
// 本类只负责"按批输入 → 一步更新"；数据装配由 src/data/dataset.* 负责，
// 训练循环/日志/checkpoint 由 M5-3 Trainer 负责。
#pragma once

#include <cstdint>
#include <string>

#include "model/discriminator.h"
#include "model/synthesizer.h"
#include "train/stft_mel.h"

#include "traincpp/traincpp.h"

namespace rvc_train {
namespace data {
struct BatchBuffers;  // data/dataset.h（仅 cpp 包含）
}

namespace train {

// 建图配置（固定形状；bucket 变化由 M5-3 图缓存管理）。
// 结构字段（层数/通道/采样率等）从训练态 GGUF 元数据推导，这里只给数据相关形状与超参。
struct TrainStepConfig {
    int64_t frames = 0;      // 桶顶 T_b（图输入序列长）
    int64_t batch = 1;       // B
    int64_t disc_len = 0;    // D 输入长度（采样）；0 = segment_frames*hop（官方 12800）
    // 官方 train.py 超参
    float lr = 1e-4f;
    float beta1 = 0.8f;
    float beta2 = 0.99f;
    float eps = 1e-9f;
    float weight_decay = 1e-2f;
    float c_mel = 45.0f;
    float c_kl = 1.0f;
    // M7-3 梯度累积：一个优化步 = 累积 K 个 micro-batch（有效 batch = batch×K）。
    // 损失按 1/K 缩放（仅 K>1 插入 scale 节点），使累积梯度 == 全批量均值梯度（对齐 batch=K）。
    int64_t grad_accum = 1;
    size_t ctx_arena_bytes = 1u << 22;  // 张量元数据 arena 初值
};

// 一步的 6 个损失（与 M3-3 golden 的 role=loss 一一对应）
struct TrainStepLosses {
    float loss_disc = 0.0f;
    float loss_gen_all = 0.0f;
    float loss_gen = 0.0f;
    float loss_fm = 0.0f;
    float loss_mel = 0.0f;
    float loss_kl = 0.0f;
};

struct TrainStepMemory {
    size_t buf_g_bytes = 0;
    size_t buf_d_bytes = 0;
    size_t ctx_tensor_bytes = 0;
    size_t ctx_persistent_bytes = 0;
    int64_t ctx_tensor_count = 0;
};

// 持久集合分解（Q23 诊断：视图根持久化是 M2.2b liveness 复用受限的主因假设）
struct TrainStepMemoryBreakdown {
    size_t view_root_bytes = 0;            // 被视图引用的根张量（context_persistent 优先标记）
    size_t view_root_leaf_bytes = 0;       //   其中 PARAM/LOSS/OUTPUT 或 nsrc==0（本就该持久）
    size_t view_root_interm_bytes = 0;     //   其中普通中间量（因被视图引用而被迫持久）
    size_t param_loss_out_bytes = 0;       // PARAM/LOSS/OUTPUT 标志（非视图根）
    size_t leaf_bytes = 0;                 // nsrc == 0 的输入/常量（非视图根）
    size_t grad_acc_bytes = 0;             // 参数梯度累加器
    size_t host_bytes = 0;                 // set_rows/acc(inplace) 宿主
    size_t non_persistent_bytes = 0;       // 可参与 Phase B 复用的中间量
    int64_t tensor_count = 0;              // 非视图张量数
    int64_t view_count = 0;
};

class TrainStep {
public:
    // weights_g/weights_d：训练态 GGUF（f0G40k.train.gguf / f0D40k.train.gguf）
    // 失败返回 nullptr 并写 err（不修改 weight 文件）。
    static TrainStep* create(const std::string& weights_g, const std::string& weights_d,
                             const TrainStepConfig& cfg, traincpp::Device* dev, std::string* err);
    ~TrainStep();

    TrainStep(const TrainStep&) = delete;
    TrainStep& operator=(const TrainStep&) = delete;

    // 注入一个 bucket 批次（BatchBuffers 的 ne 序与图叶子一致；尺寸强校验）
    bool set_batch(const data::BatchBuffers& b, std::string* err);
    // 单独注入 D 真实波形（长度 disc_len*B；golden 对拍用；正常训练走 set_batch）
    bool set_wave(const float* wave, int64_t n, std::string* err);

    // 完整一步：G 预热 → D 步 → G 步（D 节点读更新后的权重）
    bool run_step(TrainStepLosses* out, std::string* err);

    // M7-3 梯度累积：一个优化步累积 count 个 micro-batch（有效 batch = batch×count）。
    //   count==1 时等价 set_batch(micro[0]) + run_step（走旧路径，零回归）；
    //   count>1 时 Phase A 累积 D 梯度、Phase B 累积 G 梯度，各一次 optim_step。
    //   micro[i] 的 batch 必须等于 config().batch（每 micro 单独注入，顺序即累积顺序）。
    //   out 返回**有效 batch 均值**损失（图中已含 1/count 缩放，分量按 count 平均）。
    bool run_step_accum(const data::BatchBuffers* micro, int64_t count, TrainStepLosses* out,
                        std::string* err);

    // ---------------- 访问器（对拍/日志/导出） ----------------
    traincpp::Context* context() { return ctx_; }
    traincpp::Device* device() { return dev_; }
    int64_t step_count() const { return step_count_; }
    int64_t frames() const { return cfg_.frames; }
    int64_t batch() const { return cfg_.batch; }
    int64_t disc_len() const { return disc_len_; }
    const TrainStepConfig& config() const { return cfg_; }
    const model::SynthConfig& synth_config() const { return synth_cfg_; }
    const model::MpdConfig& disc_config() const { return disc_cfg_; }
    const StftMelConfig& stft_config() const { return smcfg_; }
    model::ModelParams& g_params() { return g_params_; }
    model::ModelParams& d_params() { return d_params_; }
    traincpp::Optimizer* optim_g() { return optim_g_; }
    traincpp::Optimizer* optim_d() { return optim_d_; }
    traincpp::Graph* graph_g() { return gg_; }
    traincpp::Graph* graph_d() { return dg_; }
    traincpp::Buffer* buffer_g() { return buf_g_; }
    traincpp::Buffer* buffer_d() { return buf_d_; }
    // 损失张量名："loss_gen_all"/"loss_gen"/"loss_fm"/"loss_mel"/"loss_kl"/"loss_disc"
    traincpp::Tensor* loss_tensor(const char* name) const;
    // 前向输出："y_hat"（o_slice，[seg*hop,1,B]）/"y_hat_d"（D 输入视图）
    traincpp::Tensor* output_tensor(const char* name) const;
    // G 图叶子（按名；"phone"/"pitch"/"x_mask"/"spec"/"y_mask"/"sid"/"q_noise"/"y_mel"/
    // "wave"/"attn_valid"/"slice_mat"/"sine_source_slice"/"flip_perm"；D 叶子 "d_wave"/"d_fake"）
    traincpp::Tensor* leaf(const char* name) const;
    TrainStepMemory memory() const;
    // 诊断：持久集合按类别分解（与 train.cpp context_persistent_tensors 同逻辑）
    TrainStepMemoryBreakdown persistent_breakdown() const;
    // 诊断：占用最大的图张量（非视图且有数据），min_bytes 过滤，返回多行文本
    std::string top_tensors(int top_n, size_t min_bytes) const;

private:
    TrainStep() = default;
    bool fail(std::string* err, const std::string& msg);

    traincpp::Device* dev_ = nullptr;
    traincpp::Context* ctx_ = nullptr;
    TrainStepConfig cfg_;
    model::SynthConfig synth_cfg_;
    model::MpdConfig disc_cfg_;
    StftMelConfig smcfg_;

    model::ModelParams g_params_;
    model::ModelParams d_params_;
    model::SynthModules synth_;
    model::MpdModule mpd_;
    StftMelConsts smconsts_{};

    traincpp::Graph* gg_ = nullptr;
    traincpp::Graph* dg_ = nullptr;
    traincpp::Buffer* buf_g_ = nullptr;
    traincpp::Buffer* buf_d_ = nullptr;
    traincpp::Optimizer* optim_g_ = nullptr;
    traincpp::Optimizer* optim_d_ = nullptr;

    // G 图叶子
    traincpp::Tensor* sid_ = nullptr;
    traincpp::Tensor* phone_ = nullptr;
    traincpp::Tensor* pitch_ = nullptr;
    traincpp::Tensor* x_mask_ = nullptr;
    traincpp::Tensor* attn_valid_ = nullptr;
    traincpp::Tensor* spec_ = nullptr;
    traincpp::Tensor* y_mask_ = nullptr;
    traincpp::Tensor* q_noise_ = nullptr;
    traincpp::Tensor* y_mel_ = nullptr;
    traincpp::Tensor* wave_g_ = nullptr;
    traincpp::Tensor* flip_perm_ = nullptr;
    traincpp::Tensor* slice_mat_ = nullptr;
    traincpp::Tensor* sine_slice_ = nullptr;
    // D 图叶子
    traincpp::Tensor* wave_d_ = nullptr;
    traincpp::Tensor* fake_d_ = nullptr;

    // 输出/损失
    traincpp::Tensor* y_hat_ = nullptr;      // o_slice
    traincpp::Tensor* y_hat_d_ = nullptr;    // view_3d(o_slice)
    traincpp::Tensor* loss_disc_ = nullptr;
    traincpp::Tensor* loss_gen_ = nullptr;
    traincpp::Tensor* loss_fm_ = nullptr;
    traincpp::Tensor* loss_mel_ = nullptr;
    traincpp::Tensor* loss_kl_ = nullptr;
    traincpp::Tensor* loss_gen_all_ = nullptr;

    int64_t disc_len_ = 0;
    int64_t total_slice_ = 0;
    int64_t upp_ = 0;
    int64_t step_count_ = 0;
    // M7-3：损失缩放因子（= 1/grad_accum；K=1 时为 1 且不插入缩放节点）
    float loss_scale_ = 1.0f;
};

// ---------------- M5-2 spike：单 ctx 双图可行性/内存实测（诊断入口，已验收） ----------------
// 从 M4 缓存取第一个样本装配 bucket 批次（seed 固定 1234），建 TrainStep 跑 steps 步，
// 打印逐步 loss、图缓冲/ctx 内存统计与最大张量清单。返回 0 成功。
// 结论（2026-09-25）：单 ctx 两次分配可行；官方尺度 T=384/disc=12800 的 G 图缓冲
// 5129.7 MiB > Vulkan 单缓冲 4GiB，显存分档见 `docs\回传-train.cpp-图级内存复用受限-Q23.md`。
int run_single_ctx_spike(const std::string& weights_g, const std::string& weights_d,
                         const std::string& exp_dir, traincpp::Device* dev, int64_t steps,
                         int64_t bucket_top);

} // namespace train
} // namespace rvc_train
