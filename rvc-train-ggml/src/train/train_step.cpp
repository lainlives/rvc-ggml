// RVC-Train-GGML - M5-2 minimal training-step core implementation (src/train/train_step.cpp)
//
// See the train_step.h header comment for single-ctx shared-parameter dual-graph details; key constraints (train.cpp M2.1e/R4-R15):
//   - During D graph backward construction the PARAM flags of G parameters must be cleared (the G graph backward is already built; prevents wrong attribution to the D graph);
//   - Allocation order of the two graphs: G graph first (at that point the ctx holds only G-side tensors, so the overwrite check passes), D graph second
//     (G-side tensors already have data, so the overwrite check is skipped);
//   - The D input uses independent leaves wave_d/fake_d (host copy = detach); G graph nodes must not be reused (R4).
#include "train/train_step.h"

#include "data/dataset.h"
#include "model/config.h"
#include "model/flow.h"
#include "model/weights.h"
#include "train/losses.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_set>
#include <vector>

namespace rvc_train {
namespace train {

using namespace traincpp;

namespace {

int64_t product_i64(const std::vector<int32_t>& v) {
    int64_t p = 1;
    for (int32_t x : v) {
        p *= (int64_t) x;
    }
    return p;
}

// Train-side structural config (data-independent fields come from the training-state GGUF metadata; frames/batch are given by the caller)
model::SynthConfig make_synth_config(const model::ModelConfig& mc, int64_t frames, int64_t batch) {
    model::SynthConfig c;
    c.segment_size = mc.z_frames();
    c.frames       = frames;
    c.batch        = batch;

    c.enc_p.prefix          = "enc_p";
    c.enc_p.in_channels     = mc.is_v2() ? 768 : 256;
    c.enc_p.hidden_channels = mc.hidden_channels;
    c.enc_p.filter_channels = mc.filter_channels;
    c.enc_p.n_heads         = mc.n_heads;
    c.enc_p.n_layers        = mc.n_layers;
    c.enc_p.kernel_size     = mc.kernel_size;
    c.enc_p.window_size     = 10;
    c.enc_p.out_channels    = mc.inter_channels;

    c.enc_q.prefix          = "enc_q";
    c.enc_q.in_channels     = mc.spec_channels;
    c.enc_q.out_channels    = mc.inter_channels;
    c.enc_q.hidden_channels = mc.hidden_channels;
    c.enc_q.kernel_size     = 5;
    c.enc_q.dilation_rate   = 1;
    c.enc_q.n_layers        = 16;
    c.enc_q.gin_channels    = mc.gin_channels;

    c.flow.prefix          = "flow";
    c.flow.channels        = mc.inter_channels;
    c.flow.hidden_channels = mc.hidden_channels;
    c.flow.kernel_size     = 5;
    c.flow.dilation_rate   = 1;
    c.flow.n_layers        = 3;
    c.flow.gin_channels    = mc.gin_channels;
    c.flow.n_flows         = 4;

    c.dec.prefix                   = "dec";
    c.dec.in_channels              = mc.inter_channels;
    c.dec.upsample_initial_channel = mc.upsample_initial_channel;
    c.dec.gin_channels             = mc.gin_channels;
    c.dec.upsample_rates.assign(mc.upsample_rates.begin(), mc.upsample_rates.end());
    c.dec.upsample_kernel_sizes.assign(mc.upsample_kernel_sizes.begin(),
                                       mc.upsample_kernel_sizes.end());
    c.dec.resblock_kernel_sizes.assign(mc.resblock_kernel_sizes.begin(),
                                       mc.resblock_kernel_sizes.end());
    c.dec.resblock_dilation_sizes.assign(mc.resblock_dilation_sizes.begin(),
                                         mc.resblock_dilation_sizes.end());
    return c;
}

model::MpdConfig make_mpd_config(const model::ModelConfig& mc) {
    model::MpdConfig c;
    c.prefix = "discriminators";
    c.periods.assign(mc.d_periods.begin(), mc.d_periods.end());
    c.kernel_size = 5;
    c.stride      = 3;
    return c;
}

StftMelConfig make_stft_mel_config(const model::ModelConfig& mc) {
    StftMelConfig c;
    c.n_fft  = mc.filter_length;
    c.hop    = mc.hop_length;
    c.win    = mc.win_length;
    c.n_mels = mc.n_mel_channels;
    c.sr     = (double) mc.sample_rate;
    c.fmin   = (double) mc.mel_fmin;
    c.fmax   = (double) mc.mel_fmax;
    return c;
}

bool load_all_params(const model::TrainWeights& w, model::ModelParams* p, std::string* err) {
    for (Tensor* t : p->params().items) {
        if (!w.load(t, t->name, err)) {
            return false;
        }
    }
    return true;
}

} // namespace

TrainStep::~TrainStep() {
    if (optim_g_ != nullptr) {
        optim_free(optim_g_);
    }
    if (optim_d_ != nullptr) {
        optim_free(optim_d_);
    }
    if (buf_g_ != nullptr) {
        buffer_free(buf_g_);
    }
    if (buf_d_ != nullptr) {
        buffer_free(buf_d_);
    }
    if (gg_ != nullptr) {
        graph_free(gg_);
    }
    if (dg_ != nullptr) {
        graph_free(dg_);
    }
    if (ctx_ != nullptr) {
        context_free(ctx_);
    }
}

bool TrainStep::fail(std::string* err, const std::string& msg) {
    if (err != nullptr) {
        *err = msg;
    }
    return false;
}

TrainStep* TrainStep::create(const std::string& weights_g, const std::string& weights_d,
                             const TrainStepConfig& cfg, Device* dev, std::string* err) {
    if (dev == nullptr) {
        if (err != nullptr) {
            *err = "no usable device";
        }
        return nullptr;
    }
    if (cfg.frames <= 0 || cfg.batch <= 0) {
        if (err != nullptr) {
            *err = "TrainStepConfig.frames/batch must be > 0";
        }
        return nullptr;
    }
    if (cfg.grad_accum <= 0) {
        if (err != nullptr) {
            *err = "TrainStepConfig.grad_accum must be >= 1";
        }
        return nullptr;
    }

    std::string e;
    model::TrainWeights* wg = model::TrainWeights::open(weights_g, &e);
    if (wg == nullptr) {
        if (err != nullptr) {
            *err = "failed to open G weights: " + e;
        }
        return nullptr;
    }
    model::TrainWeights* wd = model::TrainWeights::open(weights_d, &e);
    if (wd == nullptr) {
        if (err != nullptr) {
            *err = "failed to open D weights: " + e;
        }
        delete wg;
        return nullptr;
    }

    model::TrainMeta mg;
    model::TrainMeta md;
    wg->read_meta(&mg);
    wd->read_meta(&md);
    const model::ModelConfig mcg = model::ModelConfig::from_meta(mg);
    const model::ModelConfig mcd = model::ModelConfig::from_meta(md);
    if (!mcg.validate(&e) || !mcg.is_generator()) {
        if (err != nullptr) {
            *err = "invalid G weight config: " + e;
        }
        delete wg;
        delete wd;
        return nullptr;
    }
    if (!mcd.validate(&e) || !mcd.is_discriminator()) {
        if (err != nullptr) {
            *err = "invalid D weight config: " + e;
        }
        delete wg;
        delete wd;
        return nullptr;
    }

    TrainStep* ts = new TrainStep();
    ts->dev_       = dev;
    ts->cfg_       = cfg;
    ts->synth_cfg_ = make_synth_config(mcg, cfg.frames, cfg.batch);
    ts->disc_cfg_  = make_mpd_config(mcd);
    ts->smcfg_     = make_stft_mel_config(mcg);
    ts->upp_       = product_i64(mcg.upsample_rates);
    ts->total_slice_ = ts->synth_cfg_.segment_size * ts->upp_;
    ts->disc_len_ = cfg.disc_len > 0 ? cfg.disc_len : ts->total_slice_;
    // M7-3: loss scaling factor for the accumulation window (K=1 -> 1.0; no scale node inserted, graph bit-identical to the old version)
    ts->loss_scale_ = 1.0f / (float) cfg.grad_accum;

    auto cleanup_early = [&]() {
        delete ts;
        delete wg;
        delete wd;
    };

    if (!ts->smcfg_.validate(&e)) {
        if (err != nullptr) {
            *err = "invalid STFT/mel config: " + e;
        }
        cleanup_early();
        return nullptr;
    }
    if (ts->disc_len_ <= 0 || ts->disc_len_ > ts->total_slice_) {
        if (err != nullptr) {
            *err = "invalid disc_len (must be 1..segment sample count)";
        }
        cleanup_early();
        return nullptr;
    }

    const int64_t T      = cfg.frames;
    const int64_t B      = cfg.batch;
    const int64_t seg    = ts->synth_cfg_.segment_size;
    const int64_t feats  = ts->synth_cfg_.enc_p.in_channels;
    const int64_t bins   = ts->synth_cfg_.enc_q.in_channels;
    const int64_t q_ch   = ts->synth_cfg_.enc_q.out_channels;
    const int64_t flow_c = ts->synth_cfg_.flow.channels;

    // ---------------- parameter building + model assembly ----------------
    ts->ctx_ = context_new(cfg.ctx_arena_bytes);
    if (ts->ctx_ == nullptr) {
        if (err != nullptr) {
            *err = "context_new failed";
        }
        cleanup_early();
        return nullptr;
    }
    if (!ts->g_params_.build_all(ts->ctx_, *wg, &e) ||
        !ts->d_params_.build_all(ts->ctx_, *wd, &e)) {
        if (err != nullptr) {
            *err = "parameter building failed: " + e;
        }
        cleanup_early();
        return nullptr;
    }
    if (!model::synth_load(ts->g_params_, ts->synth_cfg_, &ts->synth_, &e)) {
        if (err != nullptr) {
            *err = "G assembly failed: " + e;
        }
        cleanup_early();
        return nullptr;
    }
    if (!model::mpd_load(ts->d_params_, ts->disc_cfg_, &ts->mpd_, &e)) {
        if (err != nullptr) {
            *err = "D assembly failed: " + e;
        }
        cleanup_early();
        return nullptr;
    }

    // ---------------- G graph leaves ----------------
    auto new_leaf = [&](Tensor* t, const char* name) {
        if (t != nullptr) {
            tensor_set_name(t, "%s", name);
        }
        return t;
    };
    ts->sid_       = new_leaf(new_tensor_1d(ts->ctx_, TYPE_I32, B), "train_sid");
    ts->phone_     = new_leaf(new_tensor_3d(ts->ctx_, TYPE_F32, feats, T, B), "train_phone");
    ts->pitch_     = new_leaf(new_tensor_2d(ts->ctx_, TYPE_I32, T, B), "train_pitch");
    ts->x_mask_    = new_leaf(new_tensor_3d(ts->ctx_, TYPE_F32, T, 1, B), "train_x_mask");
    ts->attn_valid_ = new_leaf(new_tensor_4d(ts->ctx_, TYPE_F32, T, T, 1, B), "train_attn_valid");
    ts->spec_      = new_leaf(new_tensor_3d(ts->ctx_, TYPE_F32, T, bins, B), "train_spec");
    ts->y_mask_    = new_leaf(new_tensor_3d(ts->ctx_, TYPE_F32, T, 1, B), "train_y_mask");
    ts->q_noise_   = new_leaf(new_tensor_3d(ts->ctx_, TYPE_F32, T, q_ch, B), "train_q_noise");
    ts->y_mel_     = new_leaf(new_tensor_3d(ts->ctx_, TYPE_F32, seg, ts->smcfg_.n_mels, B),
                              "train_y_mel");
    ts->wave_g_    = new_leaf(new_tensor_3d(ts->ctx_, TYPE_F32, ts->disc_len_, 1, B),
                              "train_wave");
    ts->flip_perm_ = new_leaf(new_tensor_2d(ts->ctx_, TYPE_F32, flow_c, flow_c),
                              "train_flip_perm");
    ts->slice_mat_ = new_leaf(new_tensor_3d(ts->ctx_, TYPE_F32, T, seg, B), "train_slice_mat");
    ts->sine_slice_ = new_leaf(new_tensor_3d(ts->ctx_, TYPE_F32, 1, ts->total_slice_, B),
                               "train_sine_source_slice");
    if (ts->sid_ == nullptr || ts->phone_ == nullptr || ts->pitch_ == nullptr ||
        ts->x_mask_ == nullptr || ts->attn_valid_ == nullptr || ts->spec_ == nullptr ||
        ts->y_mask_ == nullptr || ts->q_noise_ == nullptr || ts->y_mel_ == nullptr ||
        ts->wave_g_ == nullptr || ts->flip_perm_ == nullptr || ts->slice_mat_ == nullptr ||
        ts->sine_slice_ == nullptr) {
        if (err != nullptr) {
            *err = "failed to create G graph leaves";
        }
        cleanup_early();
        return nullptr;
    }

    model::SynthInputs in;
    in.sid               = ts->sid_;
    in.phone             = ts->phone_;
    in.pitch             = ts->pitch_;
    in.x_mask            = ts->x_mask_;
    in.attn_valid        = ts->attn_valid_;
    in.spec              = ts->spec_;
    in.y_mask            = ts->y_mask_;
    in.q_noise           = ts->q_noise_;
    in.flip_perm         = ts->flip_perm_;
    in.slice_mat         = ts->slice_mat_;
    in.sine_source_slice = ts->sine_slice_;
    in.include_full      = false;  // training uses only the slice path
    model::SynthOutputs out = model::synth_forward(ts->ctx_, ts->g_params_, ts->synth_, in);
    if (out.o_slice == nullptr) {
        if (err != nullptr) {
            *err = "synth_forward failed";
        }
        cleanup_early();
        return nullptr;
    }
    ts->y_hat_ = out.o_slice;
    tensor_set_name(ts->y_hat_, "train_y_hat");
    ts->y_hat_d_ = view_3d(ts->ctx_, out.o_slice, ts->disc_len_, 1, B, 0);
    if (ts->y_hat_d_ == nullptr) {
        if (err != nullptr) {
            *err = "failed to create the y_hat_d view";
        }
        cleanup_early();
        return nullptr;
    }
    tensor_set_name(ts->y_hat_d_, "train_y_hat_d");

    // D forward in the G graph (frozen: D parameters are not flagged PARAM here)
    model::MpdForward f_g = model::mpd_forward(ts->ctx_, ts->mpd_, ts->wave_g_, ts->y_hat_d_);
    if (f_g.r.empty() || f_g.r.size() != f_g.g.size()) {
        if (err != nullptr) {
            *err = "D forward in the G graph failed";
        }
        cleanup_early();
        return nullptr;
    }
    std::vector<Tensor*> dg_outs;
    std::vector<std::vector<Tensor*>> dr_fmaps;
    std::vector<std::vector<Tensor*>> dg_fmaps;
    for (size_t i = 0; i < f_g.r.size(); ++i) {
        dg_outs.push_back(f_g.g[i].out);
        dr_fmaps.push_back(f_g.r[i].fmap);
        dg_fmaps.push_back(f_g.g[i].fmap);
    }

    // STFT/mel constants (filled after allocation)
    stft_mel_consts_init(ts->ctx_, ts->smcfg_, &ts->smconsts_);
    if (ts->smconsts_.dft_kernel == nullptr || ts->smconsts_.mel_basis == nullptr ||
        ts->smconsts_.mag_eps == nullptr) {
        if (err != nullptr) {
            *err = "failed to create STFT/mel constants";
        }
        cleanup_early();
        return nullptr;
    }

    const float c_mel = cfg.c_mel;
    const float c_kl  = cfg.c_kl;
    ts->loss_gen_ = loss_lsgan_generator(ts->ctx_, dg_outs);
    ts->loss_fm_  = loss_feature_matching(ts->ctx_, dr_fmaps, dg_fmaps);
    Tensor* y2d   = reshape_2d(ts->ctx_, cont(ts->ctx_, out.o_slice), ts->total_slice_, B);
    Tensor* y_hat_mel =
        (y2d != nullptr) ? mel_spectrogram_forward(ts->ctx_, ts->smconsts_, ts->smcfg_, y2d)
                         : nullptr;
    ts->loss_mel_ = (y_hat_mel != nullptr)
                        ? scale(ts->ctx_, loss_l1_mean(ts->ctx_, ts->y_mel_, y_hat_mel), c_mel)
                        : nullptr;
    ts->loss_kl_  = scale(ts->ctx_,
                          loss_kl(ts->ctx_, out.z_p, out.logs_q, out.m_p, out.logs_p,
                                  ts->y_mask_),
                          c_kl);
    if (ts->loss_gen_ == nullptr || ts->loss_fm_ == nullptr || ts->loss_mel_ == nullptr ||
        ts->loss_kl_ == nullptr) {
        if (err != nullptr) {
            *err = "G loss graph build failed";
        }
        cleanup_early();
        return nullptr;
    }
    Tensor* loss_gen_raw = add(ts->ctx_,
                               add(ts->ctx_, add(ts->ctx_, ts->loss_gen_, ts->loss_fm_),
                                   ts->loss_mel_),
                               ts->loss_kl_);
    if (loss_gen_raw == nullptr) {
        if (err != nullptr) {
            *err = "loss_gen_all graph build failed";
        }
        cleanup_early();
        return nullptr;
    }
    // M7-3: the accumulation window scales the total loss by 1/K (when K=1 the raw sum is reused, zero graph regression)
    if (ts->loss_scale_ < 1.0f) {
        ts->loss_gen_all_ = scale(ts->ctx_, loss_gen_raw, ts->loss_scale_);
        if (ts->loss_gen_all_ == nullptr) {
            if (err != nullptr) {
                *err = "loss_gen_all scaling graph build failed";
            }
            cleanup_early();
            return nullptr;
        }
    } else {
        ts->loss_gen_all_ = loss_gen_raw;
    }
    tensor_set_name(ts->loss_gen_all_, "loss_gen_all");
    tensor_set_name(ts->loss_gen_, "loss_gen");
    tensor_set_name(ts->loss_fm_, "loss_fm");
    tensor_set_name(ts->loss_mel_, "loss_mel");
    tensor_set_name(ts->loss_kl_, "loss_kl");
    tensor_set_loss(ts->loss_gen_all_);
    // Mark components as OUTPUT so they stay readable under graph-level allocation (comparison/logging)
    ts->loss_gen_->flags |= TENSOR_FLAG_OUTPUT;
    ts->loss_fm_->flags |= TENSOR_FLAG_OUTPUT;
    ts->loss_mel_->flags |= TENSOR_FLAG_OUTPUT;
    ts->loss_kl_->flags |= TENSOR_FLAG_OUTPUT;
    // FIX-002 (Q23): views/view roots still read by the host after compute under graph-level allocation must also be flagged OUTPUT
    // (run_step:644 reads y_hat_d_ to feed the D graph; output_tensor("y_hat") promises readability)
    ts->y_hat_->flags |= TENSOR_FLAG_OUTPUT;
    ts->y_hat_d_->flags |= TENSOR_FLAG_OUTPUT;

    for (Tensor* t : ts->g_params_.params().items) {
        tensor_set_param(t);
    }
    ts->gg_ = graph_new(ts->ctx_);
    if (ts->gg_ == nullptr) {
        if (err != nullptr) {
            *err = "G graph create failed";
        }
        cleanup_early();
        return nullptr;
    }
    graph_build_forward_expand(ts->ctx_, ts->gg_, ts->loss_gen_all_);
    graph_build_backward_expand(ts->gg_);
    ts->buf_g_ = buffer_alloc_graph_tensors(ts->gg_, dev->default_buffer_type());
    if (ts->buf_g_ == nullptr) {
        if (err != nullptr) {
            *err = "G graph buffer allocation failed (single-ctx target ≤3.5GiB)";
        }
        cleanup_early();
        return nullptr;
    }

    // ---------------- D graph (independent leaves; new D forward nodes) ----------------
    ts->wave_d_ = new_tensor_3d(ts->ctx_, TYPE_F32, ts->disc_len_, 1, B);
    ts->fake_d_ = new_tensor_3d(ts->ctx_, TYPE_F32, ts->disc_len_, 1, B);
    if (ts->wave_d_ == nullptr || ts->fake_d_ == nullptr) {
        if (err != nullptr) {
            *err = "failed to create D graph leaves";
        }
        cleanup_early();
        return nullptr;
    }
    tensor_set_name(ts->wave_d_, "train_d_wave");
    tensor_set_name(ts->fake_d_, "train_d_fake");
    model::MpdForward f_d = model::mpd_forward(ts->ctx_, ts->mpd_, ts->wave_d_, ts->fake_d_);
    if (f_d.r.empty() || f_d.r.size() != f_d.g.size()) {
        if (err != nullptr) {
            *err = "D graph forward failed";
        }
        cleanup_early();
        return nullptr;
    }
    std::vector<Tensor*> dr_outs;
    std::vector<Tensor*> dg_outs_d;
    for (size_t i = 0; i < f_d.r.size(); ++i) {
        dr_outs.push_back(f_d.r[i].out);
        dg_outs_d.push_back(f_d.g[i].out);
    }
    Tensor* loss_disc_raw = loss_lsgan_discriminator(ts->ctx_, dr_outs, dg_outs_d);
    if (loss_disc_raw == nullptr) {
        if (err != nullptr) {
            *err = "D loss graph build failed";
        }
        cleanup_early();
        return nullptr;
    }
    // M7-3: the accumulation window scales the D loss by 1/K (when K=1 the raw loss is reused, zero graph regression)
    if (ts->loss_scale_ < 1.0f) {
        ts->loss_disc_ = scale(ts->ctx_, loss_disc_raw, ts->loss_scale_);
        if (ts->loss_disc_ == nullptr) {
            if (err != nullptr) {
                *err = "D loss scaling graph build failed";
            }
            cleanup_early();
            return nullptr;
        }
    } else {
        ts->loss_disc_ = loss_disc_raw;
    }
    tensor_set_name(ts->loss_disc_, "loss_disc");
    tensor_set_loss(ts->loss_disc_);
    for (Tensor* t : ts->d_params_.params().items) {
        tensor_set_param(t);
    }
    // Clear PARAM flags on G parameters before the D graph backward (consistent with the multi-graph freeze protocol within one ctx)
    for (Tensor* t : ts->g_params_.params().items) {
        tensor_clear_param(t);
    }
    ts->dg_ = graph_new(ts->ctx_);
    if (ts->dg_ == nullptr) {
        if (err != nullptr) {
            *err = "D graph create failed";
        }
        cleanup_early();
        return nullptr;
    }
    graph_build_forward_expand(ts->ctx_, ts->dg_, ts->loss_disc_);
    graph_build_backward_expand(ts->dg_);
    for (Tensor* t : ts->g_params_.params().items) {
        tensor_set_param(t);
    }
    ts->buf_d_ = buffer_alloc_graph_tensors(ts->dg_, dev->default_buffer_type());
    if (ts->buf_d_ == nullptr) {
        if (err != nullptr) {
            *err = "D graph buffer allocation failed";
        }
        cleanup_early();
        return nullptr;
    }

    // ---------------- optimizer (parameters collected by presence of a gradient accumulator) ----------------
    AdamwOptions opts;
    opts.lr           = cfg.lr;
    opts.beta1        = cfg.beta1;
    opts.beta2        = cfg.beta2;
    opts.eps          = cfg.eps;
    opts.weight_decay = cfg.weight_decay;
    std::vector<Tensor*> g_opt;
    std::vector<Tensor*> d_opt;
    for (Tensor* t : ts->g_params_.params().items) {
        if (t->grad_acc != nullptr) {
            g_opt.push_back(t);
        }
    }
    for (Tensor* t : ts->d_params_.params().items) {
        if (t->grad_acc != nullptr) {
            d_opt.push_back(t);
        }
    }
    if (g_opt.empty() || d_opt.empty()) {
        if (err != nullptr) {
            *err = "optimizer parameter collection is empty (abnormal G/D gradient graph)";
        }
        cleanup_early();
        return nullptr;
    }
    ts->optim_g_ = optim_adamw_new(ts->ctx_, g_opt.data(), (int64_t) g_opt.size(), opts);
    ts->optim_d_ = optim_adamw_new(ts->ctx_, d_opt.data(), (int64_t) d_opt.size(), opts);
    if (ts->optim_g_ == nullptr || ts->optim_d_ == nullptr) {
        if (err != nullptr) {
            *err = "optimizer creation failed";
        }
        cleanup_early();
        return nullptr;
    }

    // ---------------- weight loading + constant fill (after allocation) ----------------
    if (!load_all_params(*wg, &ts->g_params_, &e) || !load_all_params(*wd, &ts->d_params_, &e)) {
        if (err != nullptr) {
            *err = "parameter loading failed: " + e;
        }
        cleanup_early();
        return nullptr;
    }
    delete wg;
    delete wd;
    wg = nullptr;
    wd = nullptr;

    if (!model::flow_fill_flip_perm(ts->flip_perm_, &e)) {
        if (err != nullptr) {
            *err = "flip permutation matrix fill failed: " + e;
        }
        delete ts;
        return nullptr;
    }
    stft_mel_fill_dft_kernel(&ts->smconsts_, ts->smcfg_);
    stft_mel_fill_mel_basis(&ts->smconsts_, ts->smcfg_);

    return ts;
}

// ---------------------------------------------------------------- batch input

bool TrainStep::set_batch(const data::BatchBuffers& b, std::string* err) {
    const int64_t T    = cfg_.frames;
    const int64_t B    = cfg_.batch;
    const int64_t seg  = synth_cfg_.segment_size;
    const int64_t feat = synth_cfg_.enc_p.in_channels;
    const int64_t bins = synth_cfg_.enc_q.in_channels;
    const int64_t q_ch = synth_cfg_.enc_q.out_channels;
    const int64_t nmel = smcfg_.n_mels;

    auto expect = [&](const char* name, size_t got, int64_t want) {
        if ((int64_t) got != want) {
            char buf[192];
            std::snprintf(buf, sizeof(buf), "batch field %s element count mismatch (%zu != %lld)", name, got,
                          (long long) want);
            return fail(err, buf);
        }
        return true;
    };
    if (!expect("phone", b.phone.size(), feat * T * B) ||
        !expect("pitch", b.pitch.size(), T * B) || !expect("x_mask", b.x_mask.size(), T * B) ||
        !expect("y_mask", b.y_mask.size(), T * B) ||
        !expect("attn_valid", b.attn_valid.size(), T * T * B) ||
        !expect("spec", b.spec.size(), T * bins * B) ||
        !expect("q_noise", b.q_noise.size(), T * q_ch * B) ||
        !expect("sid", b.sid.size(), B) ||
        !expect("y_mel", b.y_mel.size(), seg * nmel * B) ||
        !expect("slice_mat", b.slice_mat.size(), T * seg * B) ||
        !expect("sine_source_slice", b.sine_source_slice.size(), total_slice_ * B) ||
        !expect("wave", b.wave.size(), disc_len_ * B)) {
        return false;
    }

    tensor_set(phone_, b.phone.data(), 0, b.phone.size() * sizeof(float));
    tensor_set(pitch_, b.pitch.data(), 0, b.pitch.size() * sizeof(int32_t));
    tensor_set(x_mask_, b.x_mask.data(), 0, b.x_mask.size() * sizeof(float));
    tensor_set(y_mask_, b.y_mask.data(), 0, b.y_mask.size() * sizeof(float));
    tensor_set(attn_valid_, b.attn_valid.data(), 0, b.attn_valid.size() * sizeof(float));
    tensor_set(spec_, b.spec.data(), 0, b.spec.size() * sizeof(float));
    tensor_set(q_noise_, b.q_noise.data(), 0, b.q_noise.size() * sizeof(float));
    tensor_set(sid_, b.sid.data(), 0, b.sid.size() * sizeof(int32_t));
    tensor_set(y_mel_, b.y_mel.data(), 0, b.y_mel.size() * sizeof(float));
    tensor_set(slice_mat_, b.slice_mat.data(), 0, b.slice_mat.size() * sizeof(float));
    tensor_set(sine_slice_, b.sine_source_slice.data(), 0,
               b.sine_source_slice.size() * sizeof(float));
    tensor_set(wave_g_, b.wave.data(), 0, b.wave.size() * sizeof(float));
    tensor_set(wave_d_, b.wave.data(), 0, b.wave.size() * sizeof(float));
    return true;
}

bool TrainStep::set_wave(const float* wave, int64_t n, std::string* err) {
    if (wave == nullptr || n != disc_len_ * cfg_.batch) {
        return fail(err, "set_wave: length mismatch (expected disc_len*batch)");
    }
    const size_t bytes = (size_t) n * sizeof(float);
    tensor_set(wave_g_, wave, 0, bytes);
    tensor_set(wave_d_, wave, 0, bytes);
    return true;
}

// ---------------------------------------------------------------- one-step update

// Environment variable switch (under MSVC getenv triggers C4996, locally suppressed; for debug output only)
static bool env_flag(const char* name) {
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4996)
#endif
    const char* v = std::getenv(name);
#if defined(_MSC_VER)
#pragma warning(pop)
#endif
    return v != nullptr && v[0] != '\0';
}

// Extract the first disc_len samples of each sample from the G output o_slice [total_slice,1,B] into dst [disc_len,B].
// Note: y_hat_d_ is a cross-sample stride view and tensor_get has **raw byte semantics** (ignores nb); for B>1,
// reading the view directly would read sample 0's subsequent frames (a Q23-class pitfall); here we copy from the contiguous o_slice by sample offset.
static void extract_y_hat(const Tensor* o_slice, int64_t total_slice, int64_t disc_len, int64_t B,
                          float* dst) {
    const size_t total = (size_t) total_slice * (size_t) B;
    std::vector<float> tmp(total);
    tensor_get(o_slice, tmp.data(), 0, total * sizeof(float));
    for (int64_t b = 0; b < B; ++b) {
        std::memcpy(dst + (size_t) b * disc_len,
                    tmp.data() + (size_t) b * total_slice,
                    (size_t) disc_len * sizeof(float));
    }
}

bool TrainStep::run_step(TrainStepLosses* out, std::string* err) {
    if (dev_ == nullptr || gg_ == nullptr || dg_ == nullptr) {
        return fail(err, "run_step: TrainStep not initialized");
    }
    // 1) G warm-up forward: produces y_hat (pre-update D; the G gradients computed along the way are then reset and discarded)
    graph_reset(gg_);
    if (!dev_->graph_compute(gg_)) {
        return fail(err, "G graph warm-up compute failed");
    }
    // 2) y_hat.detach() -> D graph leaf (host copy; equivalent to the official y_hat.detach())
    const int64_t n = disc_len_ * cfg_.batch;
    std::vector<float> y_hat((size_t) n);
    extract_y_hat(y_hat_, total_slice_, disc_len_, cfg_.batch, y_hat.data());
    tensor_set(fake_d_, y_hat.data(), 0, y_hat.size() * sizeof(float));

    // 3) D step (the updated D weights are read by the G graph's frozen nodes)
    graph_reset(dg_);
    if (!dev_->graph_compute(dg_)) {
        return fail(err, "D graph compute failed");
    }
    if (out != nullptr) {
        tensor_get(loss_disc_, &out->loss_disc, 0, sizeof(float));
    }
    optim_step(optim_d_);

    // 4) G step (D nodes read the updated weights)
    graph_reset(gg_);
    if (!dev_->graph_compute(gg_)) {
        return fail(err, "G graph compute failed");
    }
    if (out != nullptr) {
        tensor_get(loss_gen_all_, &out->loss_gen_all, 0, sizeof(float));
        tensor_get(loss_gen_, &out->loss_gen, 0, sizeof(float));
        tensor_get(loss_fm_, &out->loss_fm, 0, sizeof(float));
        tensor_get(loss_mel_, &out->loss_mel, 0, sizeof(float));
        tensor_get(loss_kl_, &out->loss_kl, 0, sizeof(float));
    }
    optim_step(optim_g_);
    ++step_count_;
    return true;
}

// M7-3 gradient accumulation: one optimizer step accumulates count micro-batches (count == config.grad_accum).
//   In-graph losses are already scaled by 1/count, so the accumulated gradient == full-batch mean gradient (aligned with batch=count).
//   Phase W: do a G warm-up forward per micro, saving y_hat to host (the detach source).
//   Phase A: inject the real waveform + detach y_hat per micro, accumulate D gradients, then one optim_step(D).
//   Phase B: re-inject the inputs per micro, accumulate G gradients, then one optim_step(G).
//   Note: graph_reset(gg_) zeroes the gradients of gg nodes (including frozen D parameters), so all warm-ups must be
//   completed **before D accumulation starts** (Phase W); graph_reset(gg_) must not be called during D accumulation (D accumulation uses graph_reset(dg_)).
bool TrainStep::run_step_accum(const data::BatchBuffers* micro, int64_t count,
                               TrainStepLosses* out, std::string* err) {
    if (dev_ == nullptr || gg_ == nullptr || dg_ == nullptr) {
        return fail(err, "run_step_accum: TrainStep not initialized");
    }
    if (micro == nullptr || count <= 0) {
        return fail(err, "run_step_accum: micro is null or count<=0");
    }
    if (count != cfg_.grad_accum) {
        return fail(err, "run_step_accum: count must equal config.grad_accum");
    }
    if (count == 1) {
        if (!set_batch(micro[0], err)) {
            return false;
        }
        return run_step(out, err);
    }

    const int64_t nb = disc_len_ * cfg_.batch;

    // Phase W: G warm-up forward, save each micro's y_hat.detach() to host
    std::vector<std::vector<float>> y_hat((size_t) count, std::vector<float>((size_t) nb));
    for (int64_t m = 0; m < count; ++m) {
        if (!set_batch(micro[m], err)) {
            return false;
        }
        graph_reset(gg_);
        if (!dev_->graph_compute(gg_)) {
            return fail(err, "G graph warm-up compute failed (accumulated)");
        }
        extract_y_hat(y_hat_, total_slice_, disc_len_, cfg_.batch, y_hat[(size_t) m].data());
    }

    // Phase A: D gradient accumulation
    float sum_disc = 0.0f;
    for (int64_t m = 0; m < count; ++m) {
        if (!set_batch(micro[m], err)) {
            return false;
        }
        tensor_set(fake_d_, y_hat[(size_t) m].data(), 0, (size_t) nb * sizeof(float));
        if (m == 0) {
            graph_reset(dg_);
        } else {
            graph_reset_accumulate(dg_);
        }
        if (!dev_->graph_compute(dg_)) {
            return fail(err, "D graph compute failed (accumulated)");
        }
        float ld = 0.0f;
        tensor_get(loss_disc_, &ld, 0, sizeof(float));
        sum_disc += ld;  // each micro already includes 1/count scaling
        if (env_flag("RVCT_DEBUG_ACC")) {
            std::printf("[acc] D micro %lld loss_disc=%.6f\n", (long long) m, (double) ld);
        }
    }
    optim_step(optim_d_);

    // Phase B: G gradient accumulation
    float sum_gen_all = 0.0f, sum_gen = 0.0f, sum_fm = 0.0f, sum_mel = 0.0f, sum_kl = 0.0f;
    for (int64_t m = 0; m < count; ++m) {
        if (!set_batch(micro[m], err)) {
            return false;
        }
        if (m == 0) {
            graph_reset(gg_);
        } else {
            graph_reset_accumulate(gg_);
        }
        if (!dev_->graph_compute(gg_)) {
            return fail(err, "G graph compute failed (accumulated)");
        }
        float a = 0.0f, b = 0.0f, c = 0.0f, d = 0.0f, e = 0.0f;
        tensor_get(loss_gen_all_, &a, 0, sizeof(float));  // already includes 1/count
        tensor_get(loss_gen_, &b, 0, sizeof(float));
        tensor_get(loss_fm_, &c, 0, sizeof(float));
        tensor_get(loss_mel_, &d, 0, sizeof(float));
        tensor_get(loss_kl_, &e, 0, sizeof(float));
        sum_gen_all += a;
        sum_gen += b;
        sum_fm += c;
        sum_mel += d;
        sum_kl += e;
        if (env_flag("RVCT_DEBUG_ACC")) {
            std::printf("[acc] G micro %lld gen_all=%.6f gen=%.6f fm=%.6f mel=%.6f kl=%.6f\n",
                        (long long) m, (double) a, (double) b, (double) c, (double) d,
                        (double) e);
        }
    }
    optim_step(optim_g_);
    ++step_count_;

    if (out != nullptr) {
        const float inv = 1.0f / (float) count;
        out->loss_disc = sum_disc;              // sum after scaling = mean
        out->loss_gen_all = sum_gen_all;        // same as above
        out->loss_gen = sum_gen * inv;
        out->loss_fm = sum_fm * inv;
        out->loss_mel = sum_mel * inv;
        out->loss_kl = sum_kl * inv;
    }
    return true;
}

// ---------------------------------------------------------------- accessors/diagnostics

Tensor* TrainStep::loss_tensor(const char* name) const {
    const std::string n = name != nullptr ? name : "";
    if (n == "loss_disc") {
        return loss_disc_;
    }
    if (n == "loss_gen_all") {
        return loss_gen_all_;
    }
    if (n == "loss_gen") {
        return loss_gen_;
    }
    if (n == "loss_fm") {
        return loss_fm_;
    }
    if (n == "loss_mel") {
        return loss_mel_;
    }
    if (n == "loss_kl") {
        return loss_kl_;
    }
    return nullptr;
}

Tensor* TrainStep::output_tensor(const char* name) const {
    const std::string n = name != nullptr ? name : "";
    if (n == "y_hat") {
        return y_hat_;
    }
    if (n == "y_hat_d") {
        return y_hat_d_;
    }
    return nullptr;
}

Tensor* TrainStep::leaf(const char* name) const {
    const std::string n = name != nullptr ? name : "";
    if (n == "phone") {
        return phone_;
    }
    if (n == "pitch") {
        return pitch_;
    }
    if (n == "x_mask") {
        return x_mask_;
    }
    if (n == "attn_valid") {
        return attn_valid_;
    }
    if (n == "spec") {
        return spec_;
    }
    if (n == "y_mask") {
        return y_mask_;
    }
    if (n == "q_noise") {
        return q_noise_;
    }
    if (n == "y_mel") {
        return y_mel_;
    }
    if (n == "sid") {
        return sid_;
    }
    if (n == "wave") {
        return wave_g_;
    }
    if (n == "flip_perm") {
        return flip_perm_;
    }
    if (n == "slice_mat") {
        return slice_mat_;
    }
    if (n == "sine_source_slice") {
        return sine_slice_;
    }
    if (n == "d_wave") {
        return wave_d_;
    }
    if (n == "d_fake") {
        return fake_d_;
    }
    return nullptr;
}

TrainStepMemory TrainStep::memory() const {
    TrainStepMemory m;
    m.buf_g_bytes = (buf_g_ != nullptr) ? buf_g_->size() : 0;
    m.buf_d_bytes = (buf_d_ != nullptr) ? buf_d_->size() : 0;
    if (ctx_ != nullptr) {
        const ContextMemoryStats s = context_memory_stats(ctx_);
        m.ctx_tensor_bytes     = s.tensor_bytes;
        m.ctx_persistent_bytes = s.persistent_bytes;
        m.ctx_tensor_count     = context_tensor_count(ctx_);
    }
    return m;
}

TrainStepMemoryBreakdown TrainStep::persistent_breakdown() const {
    TrainStepMemoryBreakdown b;
    if (ctx_ == nullptr) {
        return b;
    }
    const int n = context_tensor_count(ctx_);
    std::unordered_set<const Tensor*> view_roots;
    std::unordered_set<const Tensor*> grad_accs;
    std::unordered_set<const Tensor*> hosts;
    for (int i = 0; i < n; ++i) {
        const Tensor* t = context_tensor(ctx_, i);
        if (t->view_src == nullptr) {
            continue;
        }
        ++b.view_count;
        const Tensor* root = t;
        while (root->view_src != nullptr) {
            root = root->view_src;
        }
        view_roots.insert(root);
    }
    for (int i = 0; i < n; ++i) {
        const Tensor* t = context_tensor(ctx_, i);
        if (t->grad_acc != nullptr) {
            grad_accs.insert(t->grad_acc);
        }
        if (t->op == OP_SET_ROWS && t->src[2] != nullptr) {
            hosts.insert(t->src[2]);
        }
        if (t->op == OP_ACC && t->src[0] != nullptr && t->op_params[4] != 0) {
            hosts.insert(t->src[0]);
        }
    }
    for (int i = 0; i < n; ++i) {
        const Tensor* t = context_tensor(ctx_, i);
        if (t->view_src != nullptr || t->data == nullptr) {
            continue;
        }
        ++b.tensor_count;
        const size_t bytes = tensor_nbytes(t);
        if (view_roots.count(t) != 0) {
            b.view_root_bytes += bytes;
            if ((t->flags & (TENSOR_FLAG_PARAM | TENSOR_FLAG_LOSS | TENSOR_FLAG_OUTPUT)) != 0 ||
                t->nsrc == 0) {
                b.view_root_leaf_bytes += bytes;
            } else {
                b.view_root_interm_bytes += bytes;
            }
        } else if ((t->flags & (TENSOR_FLAG_PARAM | TENSOR_FLAG_LOSS | TENSOR_FLAG_OUTPUT)) != 0) {
            b.param_loss_out_bytes += bytes;
        } else if (t->nsrc == 0) {
            b.leaf_bytes += bytes;
        } else if (grad_accs.count(t) != 0) {
            b.grad_acc_bytes += bytes;
        } else if (hosts.count(t) != 0) {
            b.host_bytes += bytes;
        } else {
            b.non_persistent_bytes += bytes;
        }
    }
    return b;
}

std::string TrainStep::top_tensors(int top_n, size_t min_bytes) const {
    struct Item {
        size_t bytes;
        std::string name;
    };
    std::vector<Item> items;
    if (ctx_ != nullptr) {
        const int n = context_tensor_count(ctx_);
        items.reserve((size_t) n / 4 + 8);
        for (int i = 0; i < n; ++i) {
            const Tensor* t = context_tensor(ctx_, i);
            if (t == nullptr || t->view_src != nullptr || t->data == nullptr) {
                continue;
            }
            const size_t bytes = tensor_nbytes(t);
            if (bytes < min_bytes) {
                continue;
            }
            items.push_back({bytes, t->name[0] != '\0' ? std::string(t->name) : "<unnamed>"});
        }
    }
    std::sort(items.begin(), items.end(), [](const Item& a, const Item& b) {
        return a.bytes != b.bytes ? a.bytes > b.bytes : a.name < b.name;
    });
    std::string s;
    const int lim = std::min<int>((int) items.size(), top_n);
    for (int i = 0; i < lim; ++i) {
        char buf[256];
        std::snprintf(buf, sizeof(buf), "  #%d %8.1f MiB  %s\n", i + 1,
                      (double) items[(size_t) i].bytes / (1024.0 * 1024.0),
                      items[(size_t) i].name.c_str());
        s += buf;
    }
    if (items.empty()) {
        s = "  (no tensors above the threshold)\n";
    }
    return s;
}

// ---------------------------------------------------------------- spike

int run_single_ctx_spike(const std::string& weights_g, const std::string& weights_d,
                         const std::string& exp_dir, Device* dev, int64_t steps,
                         int64_t bucket_top) {
    if (dev == nullptr) {
        std::printf("[spike] no usable device\n");
        return 2;
    }
    if (steps <= 0) {
        steps = 1;
    }
    if (bucket_top <= 0) {
        bucket_top = 384;
    }

    std::string err;
    model::TrainWeights* wg = model::TrainWeights::open(weights_g, &err);
    if (wg == nullptr) {
        std::printf("[spike] failed to open G weights: %s\n", err.c_str());
        return 1;
    }
    model::TrainMeta mg;
    wg->read_meta(&mg);
    const model::ModelConfig mcg = model::ModelConfig::from_meta(mg);
    delete wg;
    if (!mcg.validate(&err) || !mcg.is_generator()) {
        std::printf("[spike] invalid G weight config: %s\n", err.c_str());
        return 1;
    }

    data::TrainDataConfig dcfg;
    dcfg.exp_dir         = exp_dir;
    dcfg.batch_size      = 1;
    dcfg.segment_frames  = mcg.z_frames() > 0 ? mcg.z_frames() : 32;
    dcfg.segment_samples = mcg.segment_size > 0 ? mcg.segment_size : 12800;
    dcfg.q_channels      = mcg.inter_channels;
    dcfg.speaker_id      = 0;
    dcfg.buckets         = {bucket_top};
    dcfg.n_fft           = mcg.filter_length;
    dcfg.win             = mcg.win_length;
    dcfg.n_mels          = mcg.n_mel_channels;
    dcfg.sr              = (double) mcg.sample_rate;
    dcfg.fmin            = (double) mcg.mel_fmin;
    dcfg.fmax            = (double) mcg.mel_fmax;
    data::Dataset ds;
    if (!data::Dataset::open(exp_dir, dcfg, &ds, &err)) {
        std::printf("[spike] cache load failed: %s\n", err.c_str());
        return 1;
    }
    if (ds.size() <= 0) {
        std::printf("[spike] cache has no samples\n");
        return 1;
    }

    TrainStepConfig cfg;
    cfg.frames = bucket_top;
    cfg.batch  = 1;
    std::printf("[spike] device=%s  frames=%lld batch=%lld disc_len=%lld buckets=%zu steps=%lld\n",
                dev->props().name.c_str(), (long long) cfg.frames, (long long) cfg.batch,
                (long long) (dcfg.segment_samples), dcfg.buckets.size(), (long long) steps);

    TrainStep* ts = TrainStep::create(weights_g, weights_d, cfg, dev, &err);
    if (ts == nullptr) {
        std::printf("[spike] TrainStep create failed: %s\n", err.c_str());
        return 1;
    }
    {
        const TrainStepMemory m = ts->memory();
        const ContextMemoryStats s = context_memory_stats(ts->context());
        std::printf("[spike] buf_g=%.1f MiB  buf_d=%.1f MiB  ctx_tensor=%.1f MiB "
                    "persistent=%.1f MiB  tensors=%lld views=%zu\n",
                    (double) m.buf_g_bytes / (1024.0 * 1024.0),
                    (double) m.buf_d_bytes / (1024.0 * 1024.0),
                    (double) s.tensor_bytes / (1024.0 * 1024.0),
                    (double) s.persistent_bytes / (1024.0 * 1024.0), (long long) m.ctx_tensor_count,
                    s.view_count);
        std::printf("[spike] G graph nodes=%lld  D graph nodes=%lld\n", (long long) graph_n_nodes(ts->graph_g()),
                    (long long) graph_n_nodes(ts->graph_d()));
        std::printf("[spike] largest graph tensors (>=8MiB):\n%s", ts->top_tensors(20, 8u << 20).c_str());
    }

    Rng rng;
    rng_seed(&rng, 1234);
    const std::vector<int64_t> idx = {0};
    for (int64_t i = 0; i < steps; ++i) {
        data::BatchBuffers b;
        if (!ds.build_batch(idx, bucket_top, &rng, nullptr, &b, &err)) {
            std::printf("[spike] batch assembly failed: %s\n", err.c_str());
            delete ts;
            return 1;
        }
        if (!ts->set_batch(b, &err)) {
            std::printf("[spike] set_batch failed: %s\n", err.c_str());
            delete ts;
            return 1;
        }
        TrainStepLosses L;
        if (!ts->run_step(&L, &err)) {
            std::printf("[spike] run_step failed: %s\n", err.c_str());
            delete ts;
            return 1;
        }
        std::printf("[spike] step %lld  loss_disc=%.6f loss_gen_all=%.6f loss_gen=%.6f "
                    "loss_fm=%.6f loss_mel=%.6f loss_kl=%.6f\n",
                    (long long) (i + 1), (double) L.loss_disc, (double) L.loss_gen_all,
                    (double) L.loss_gen, (double) L.loss_fm, (double) L.loss_mel,
                    (double) L.loss_kl);
    }
    {
        const TrainStepMemory m = ts->memory();
        std::printf("[spike] done: steps=%lld buf_g=%.1f MiB buf_d=%.1f MiB\n",
                    (long long) ts->step_count(), (double) m.buf_g_bytes / (1024.0 * 1024.0),
                    (double) m.buf_d_bytes / (1024.0 * 1024.0));
    }
    delete ts;
    return 0;
}

} // namespace train
} // namespace rvc_train
