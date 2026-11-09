// RVC-Train-GGML - M5-2 minimal training-step core (src/train/train_step.h)
//
// Single-ctx shared-parameter dual graph (aligned with the acceptance form of train.cpp tests/test_acceptance.cpp;
// docs/handover.md §4.9 C / integration_and_comparison/RVC_integration_guide.md §5.4 route 2):
//   - G/D parameters are each built once inside the same Context; the G graph contains the synth slice forward + frozen D forward + 4 G losses;
//     the D graph is a **new** D forward + LSGAN(D) on independent leaves (wave_d/fake_d, host copy = detach).
//   - Step order (official 81eed5e train.py:425-527): G warm-up forward (to get y_hat) -> host-copy y_hat to
//     fake_d -> D step (graph_reset/compute/optim_step) -> G step (reset and recompute; D nodes read the updated weights).
//   - Allocation order: after building the G graph backward, buffer_alloc_graph_tensors(gg), then build the D graph and
//     buffer_alloc_graph_tensors(dg) (G-side tensors that already have data are skipped).
//
// This class is only responsible for "batch input -> one update"; data assembly is handled by src/data/dataset.*,
// the training loop/logging/checkpoint are handled by M5-3 Trainer.
#pragma once

#include <cstdint>
#include <string>

#include "model/discriminator.h"
#include "model/synthesizer.h"
#include "train/stft_mel.h"

#include "traincpp/traincpp.h"

namespace rvc_train {
namespace data {
struct BatchBuffers;  // data/dataset.h (cpp inclusion only)
}

namespace train {

// Graph-building config (fixed shapes; bucket changes are managed by the M5-3 graph cache).
// Structural fields (layers/channels/sample rate etc.) are derived from the training-state GGUF metadata; only data-related shapes and hyperparameters are given here.
struct TrainStepConfig {
    int64_t frames = 0;      // bucket top T_b (graph input sequence length)
    int64_t batch = 1;       // B
    int64_t disc_len = 0;    // D input length (samples); 0 = segment_frames*hop (official 12800)
    // Official train.py hyperparameters
    float lr = 1e-4f;
    float beta1 = 0.8f;
    float beta2 = 0.99f;
    float eps = 1e-9f;
    float weight_decay = 1e-2f;
    float c_mel = 45.0f;
    float c_kl = 1.0f;
    // M7-3 gradient accumulation: one optimizer step = accumulate K micro-batches (effective batch = batch×K).
    // The loss is scaled by 1/K (a scale node is inserted only when K>1), so the accumulated gradient == full-batch mean gradient (aligned with batch=K).
    int64_t grad_accum = 1;
    size_t ctx_arena_bytes = 1u << 22;  // initial tensor-metadata arena size
};

// The 6 losses of one step (one-to-one with role=loss entries of the M3-3 golden)
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

// Persistent-set breakdown (Q23 diagnosis: view-root persistence is the main hypothesis for limited M2.2b liveness reuse)
struct TrainStepMemoryBreakdown {
    size_t view_root_bytes = 0;            // root tensors referenced by views (context_persistent preferentially marks them)
    size_t view_root_leaf_bytes = 0;       //   of which PARAM/LOSS/OUTPUT or nsrc==0 (persistent anyway)
    size_t view_root_interm_bytes = 0;     //   of which ordinary intermediates (forced persistent because views reference them)
    size_t param_loss_out_bytes = 0;       // PARAM/LOSS/OUTPUT flags (not view roots)
    size_t leaf_bytes = 0;                 // inputs/constants with nsrc == 0 (not view roots)
    size_t grad_acc_bytes = 0;             // parameter gradient accumulators
    size_t host_bytes = 0;                 // set_rows/acc(inplace) hosts
    size_t non_persistent_bytes = 0;       // intermediates eligible for Phase B reuse
    int64_t tensor_count = 0;              // number of non-view tensors
    int64_t view_count = 0;
};

class TrainStep {
public:
    // weights_g/weights_d: training-state GGUF (f0G40k.train.gguf / f0D40k.train.gguf)
    // On failure returns nullptr and writes err (does not modify the weight files).
    static TrainStep* create(const std::string& weights_g, const std::string& weights_d,
                             const TrainStepConfig& cfg, traincpp::Device* dev, std::string* err);
    ~TrainStep();

    TrainStep(const TrainStep&) = delete;
    TrainStep& operator=(const TrainStep&) = delete;

    // Inject one bucket batch (BatchBuffers ne order matches the graph leaves; strict size validation)
    bool set_batch(const data::BatchBuffers& b, std::string* err);
    // Inject D real waveform separately (length disc_len*B; for golden comparison; normal training uses set_batch)
    bool set_wave(const float* wave, int64_t n, std::string* err);

    // Full step: G warm-up -> D step -> G step (D nodes read the updated weights)
    bool run_step(TrainStepLosses* out, std::string* err);

    // M7-3 gradient accumulation: one optimizer step accumulates count micro-batches (effective batch = batch×count).
    //   count==1 is equivalent to set_batch(micro[0]) + run_step (takes the old path, zero regression);
    //   count>1 accumulates D gradients in Phase A and G gradients in Phase B, each followed by one optim_step.
    //   micro[i]'s batch must equal config().batch (each micro is injected separately; order is the accumulation order).
    //   out returns the **effective batch mean** loss (the graph already includes 1/count scaling; components are averaged over count).
    bool run_step_accum(const data::BatchBuffers* micro, int64_t count, TrainStepLosses* out,
                        std::string* err);

    // ---------------- accessors (comparison/logging/export) ----------------
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
    // Loss tensor names: "loss_gen_all"/"loss_gen"/"loss_fm"/"loss_mel"/"loss_kl"/"loss_disc"
    traincpp::Tensor* loss_tensor(const char* name) const;
    // Forward outputs: "y_hat" (o_slice, [seg*hop,1,B]) / "y_hat_d" (D input view)
    traincpp::Tensor* output_tensor(const char* name) const;
    // G graph leaves (by name; "phone"/"pitch"/"x_mask"/"spec"/"y_mask"/"sid"/"q_noise"/"y_mel"/
    // "wave"/"attn_valid"/"slice_mat"/"sine_source_slice"/"flip_perm"; D leaves "d_wave"/"d_fake")
    traincpp::Tensor* leaf(const char* name) const;
    TrainStepMemory memory() const;
    // Diagnostic: breakdown of the persistent set by category (same logic as train.cpp context_persistent_tensors)
    TrainStepMemoryBreakdown persistent_breakdown() const;
    // Diagnostic: largest graph tensors by footprint (non-view with data), filtered by min_bytes; returns multi-line text
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

    // G graph leaves
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
    // D graph leaves
    traincpp::Tensor* wave_d_ = nullptr;
    traincpp::Tensor* fake_d_ = nullptr;

    // outputs/losses
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
    // M7-3: loss scaling factor (= 1/grad_accum; 1 when K=1 and no scale node is inserted)
    float loss_scale_ = 1.0f;
};

// ---------------- M5-2 spike: single-ctx dual-graph feasibility/memory measurement (diagnostic entry, accepted) ----------------
// Take the first sample from the M4 cache, assemble a bucket batch (seed fixed at 1234), build a TrainStep and run steps steps,
// printing per-step loss, graph-buffer/ctx memory stats and the largest-tensor list. Returns 0 on success.
// Conclusion (2026-09-25): two allocations in a single ctx are feasible; at official scale T=384/disc=12800 the G graph buffer is
// 5129.7 MiB > the 4GiB Vulkan single-buffer limit; see `docs\feedback-train-cpp-graph-memory-reuse-limited-Q23.md` for the VRAM tiers.
int run_single_ctx_spike(const std::string& weights_g, const std::string& weights_d,
                         const std::string& exp_dir, traincpp::Device* dev, int64_t steps,
                         int64_t bucket_top);

} // namespace train
} // namespace rvc_train
