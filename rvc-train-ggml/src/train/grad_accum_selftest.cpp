// RVC-Train-GGML - M7-3 gradient accumulation equivalence self-test implementation (src/train/grad_accum_selftest.cpp)
//
// Gate G-7.3 (gradient-level equivalence):
//   With the same weights and data, build arm A (batch=B, grad_accum=K) and arm B (batch=B*K, grad_accum=1):
//     A: TrainStep::run_step_accum(K micro-batches) -> accumulated gradient (in-graph losses already scaled by 1/K)
//     B: TrainStep::set_batch(combined batch) + run_step -> full-batch mean gradient
//   Compare grad_acc of all G/D parameters: max_abs / normalized relative error / rms; gate max_rel ≤ 1e-3.
//   Also print the 6 losses of both arms for inspection (losses should be the effective batch mean).
#include "train/grad_accum_selftest.h"

#include "data/dataset.h"
#include "model/config.h"
#include "model/weights.h"
#include "train/train_step.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace rvc_train {
namespace train {

namespace {

// Environment variable switch (MSVC getenv triggers C4996, locally suppressed)
bool env_flag(const char* name) {
#pragma warning(push)
#pragma warning(disable : 4996)
    const char* v = std::getenv(name);
#pragma warning(pop)
    return v != nullptr && v[0] != '\0';
}

bool finite6(const TrainStepLosses& L) {
    return std::isfinite(L.loss_disc) && std::isfinite(L.loss_gen_all) &&
           std::isfinite(L.loss_gen) && std::isfinite(L.loss_fm) &&
           std::isfinite(L.loss_mel) && std::isfinite(L.loss_kl);
}

// Same as Trainer::build_batch_into: truncate each sample waveform from L to d
void truncate_wave(data::BatchBuffers* b, int64_t d, int64_t L) {
    if (d <= 0 || d >= L || b->batch <= 0) {
        return;
    }
    const int64_t B = b->batch;
    std::vector<float> w((size_t) (d * B));
    for (int64_t i = 0; i < B; ++i) {
        std::memcpy(&w[(size_t) (i * d)], &b->wave[(size_t) (i * L)],
                    (size_t) d * sizeof(float));
    }
    b->wave.swap(w);
}

// Append src (B_s samples, B_s is the slowest dimension) to dst (B_d samples), yielding B_d+B_s
void append_batch(data::BatchBuffers* dst, const data::BatchBuffers& src) {
    auto ap = [](std::vector<float>& d, const std::vector<float>& s) {
        d.insert(d.end(), s.begin(), s.end());
    };
    ap(dst->phone, src.phone);
    dst->pitch.insert(dst->pitch.end(), src.pitch.begin(), src.pitch.end());
    dst->pitchf.insert(dst->pitchf.end(), src.pitchf.begin(), src.pitchf.end());
    ap(dst->spec, src.spec);
    ap(dst->wave, src.wave);
    ap(dst->y_mel, src.y_mel);
    dst->sid.insert(dst->sid.end(), src.sid.begin(), src.sid.end());
    ap(dst->q_noise, src.q_noise);
    ap(dst->x_mask, src.x_mask);
    ap(dst->y_mask, src.y_mask);
    ap(dst->attn_valid, src.attn_valid);
    ap(dst->slice_mat, src.slice_mat);
    ap(dst->pitchf_slice, src.pitchf_slice);
    ap(dst->sine_source_slice, src.sine_source_slice);
    dst->lengths.insert(dst->lengths.end(), src.lengths.begin(), src.lengths.end());
    dst->ids_slice.insert(dst->ids_slice.end(), src.ids_slice.begin(), src.ids_slice.end());
    dst->batch += src.batch;
}

// Collect parameter gradients (grad_acc preferred, fallback to grad) into a flat vector
void collect_list(const traincpp::ParamList& pl, std::vector<float>* out) {
    const int64_t n = traincpp::param_list_count(&pl);
    for (int64_t i = 0; i < n; ++i) {
        traincpp::Tensor* t = traincpp::param_list_get(&pl, i);
        traincpp::Tensor* g = (t->grad_acc != nullptr) ? t->grad_acc : t->grad;
        if (g == nullptr) {
            continue;
        }
        const size_t ne = (size_t) tensor_nelements(g);
        const size_t base = out->size();
        out->resize(base + ne);
        tensor_get(g, out->data() + base, 0, ne * sizeof(float));
    }
}

// Compare two gradient vectors, print and return the normalized relative error
double grad_stats(const char* tag, const std::vector<float>& a, const std::vector<float>& b,
                  int* failed) {
    if (a.size() != b.size()) {
        std::printf("[grad-accum] [FAIL] %s element count mismatch (%zu != %zu)\n", tag, a.size(), b.size());
        ++(*failed);
        return 0.0;
    }
    double max_abs = 0.0, max_ref = 0.0, sum_sq = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double diff = std::fabs((double) a[i] - (double) b[i]);
        max_abs = std::max(max_abs, diff);
        max_ref = std::max(max_ref, std::fabs((double) b[i]));
        sum_sq += diff * diff;
    }
    const double rms = a.empty() ? 0.0 : std::sqrt(sum_sq / (double) a.size());
    const double rel = max_abs / std::max(max_ref, 1e-12);
    std::printf("[grad-accum] %s: max_abs=%.3e max_ref=%.3e norm_rel=%.3e rms=%.3e\n", tag,
                max_abs, max_ref, rel, rms);
    return rel;
}

} // namespace

int run_grad_accum_selftest(const std::string& weights_g, const std::string& weights_d,
                            const std::string& exp_dir, traincpp::Device* dev, int64_t bucket,
                            int64_t batch, int64_t grad_accum, int64_t disc_len) {
    if (dev == nullptr) {
        std::printf("[grad-accum] no usable device\n");
        return 2;
    }
    const int64_t K = grad_accum > 0 ? grad_accum : 1;
    const int64_t B = batch > 0 ? batch : 1;
    if (K < 2) {
        std::printf("[grad-accum] --grad-accum must be >= 2 (current %lld)\n", (long long) K);
        return 2;
    }
    if (weights_g.empty() || weights_d.empty() || exp_dir.empty()) {
        std::printf("[grad-accum] need --weights <G> --d-weights <D> --exp <cache>\n");
        return 2;
    }

    // Derive cache shapes from the G metadata
    std::string err;
    std::unique_ptr<model::TrainWeights> wg(model::TrainWeights::open(weights_g, &err));
    if (wg == nullptr) {
        std::printf("[grad-accum] failed to open G weights: %s\n", err.c_str());
        return 2;
    }
    model::TrainMeta mg;
    wg->read_meta(&mg);
    const model::ModelConfig mc = model::ModelConfig::from_meta(mg);
    const int64_t seg_frames = mc.z_frames() > 0 ? mc.z_frames() : 32;
    const int64_t seg_samples = mc.segment_size > 0 ? mc.segment_size : 12800;
    const int64_t Tb = bucket > 0 ? bucket : 384;
    if (Tb < seg_frames) {
        std::printf("[grad-accum] bucket T=%lld < segment_frames=%lld\n", (long long) Tb,
                    (long long) seg_frames);
        return 2;
    }
    const int64_t D = disc_len > 0 ? disc_len : seg_samples;

    data::TrainDataConfig dc;
    dc.exp_dir         = exp_dir;
    dc.batch_size      = B;
    dc.segment_frames  = seg_frames;
    dc.segment_samples = seg_samples;
    dc.q_channels      = mc.inter_channels;
    dc.buckets         = {Tb};
    dc.n_fft           = mc.filter_length;
    dc.win             = mc.win_length;
    dc.n_mels          = mc.n_mel_channels;
    dc.sr              = (double) mc.sample_rate;
    dc.fmin            = (double) mc.mel_fmin;
    dc.fmax            = mc.mel_fmax > 0 ? (double) mc.mel_fmax : -1.0;

    data::Dataset ds;
    if (!data::Dataset::open(exp_dir, dc, &ds, &err)) {
        std::printf("[grad-accum] failed to open cache: %s\n", err.c_str());
        return 2;
    }
    std::vector<int64_t> eligible;
    for (int64_t i = 0; i < ds.size(); ++i) {
        const int64_t f = ds.aligned_frames(i);
        if (f >= seg_frames && f <= Tb) {
            eligible.push_back(i);
        }
    }
    const int64_t need = K * B;
    if ((int64_t) eligible.size() < need) {
        std::printf("[grad-accum] eligible samples %lld < needed %lld (K*B)\n", (long long) eligible.size(),
                    (long long) need);
        return 2;
    }
    // Pick K*B samples with identical aligned_frames: equal length -> all reductions (including the official KL's global sum/sum)
    // are per-sample additive, so grad_accum=K and batch=K should be exactly equivalent. Across lengths the official KL's global
    // normalization is not separable, a known semantic difference (see docs/dev_progress.md M7-3 log).
    std::vector<int64_t> idx;
    {
        std::map<int64_t, std::vector<int64_t>> by_frames;
        for (int64_t i : eligible) {
            by_frames[ds.aligned_frames(i)].push_back(i);
        }
        for (const auto& kv : by_frames) {
            if ((int64_t) kv.second.size() >= need) {
                idx.assign(kv.second.begin(), kv.second.begin() + (size_t) need);
                std::printf("[grad-accum] selected equal-length samples: aligned_frames=%lld × %lld\n",
                            (long long) kv.first, (long long) need);
                break;
            }
        }
        if (idx.empty()) {
            std::printf("[grad-accum] no %lld equal-length samples in the cache, falling back to repeated samples (mechanism check only)\n",
                        (long long) need);
            for (int64_t i = 0; i < need; ++i) {
                idx.push_back(eligible[(size_t) (i % eligible.size())]);
            }
        }
    }

    std::printf("[grad-accum] device=%s bucket=%lld batch=%lld K=%lld disc_len=%lld "
                "eligible samples=%lld\n",
                dev->props().name.c_str(), (long long) Tb, (long long) B, (long long) K,
                (long long) D, (long long) idx.size());

    // Assemble K micro-batches (B samples each) with a fixed seed, then append them into a combined B*K batch
    traincpp::Rng rng;
    traincpp::rng_seed(&rng, 20260928u);
    std::vector<data::BatchBuffers> micro((size_t) K);
    for (int64_t m = 0; m < K; ++m) {
        std::vector<int64_t> mi(idx.begin() + (size_t) (m * B),
                                idx.begin() + (size_t) (m * B + B));
        if (!ds.build_batch(mi, Tb, &rng, nullptr, &micro[(size_t) m], &err)) {
            std::printf("[grad-accum] failed to assemble micro %lld: %s\n", (long long) m, err.c_str());
            return 1;
        }
        truncate_wave(&micro[(size_t) m], D, seg_samples);
    }
    if (env_flag("RVCT_ACC_DUP")) {
        for (int64_t m = 1; m < K; ++m) {
            micro[(size_t) m] = micro[0];
        }
        std::printf("[grad-accum] RVCT_ACC_DUP: all micro-batches use the same data (diagnosing D batch correctness)\n");
    }
    data::BatchBuffers combined = micro[0];
    for (int64_t m = 1; m < K; ++m) {
        append_batch(&combined, micro[(size_t) m]);
    }
    combined.batch = K * B;

    TrainStepConfig ca;
    ca.frames = Tb;
    ca.batch = B;
    ca.grad_accum = K;
    ca.disc_len = D;

    TrainStepConfig cb;
    cb.frames = Tb;
    cb.batch = K * B;
    cb.grad_accum = 1;
    cb.disc_len = D;

    int failed = 0;

    // ---------------- arm A: accumulate K micro-batches ----------------
    std::vector<float> ga_g, ga_d;
    TrainStepLosses La{};
    {
        std::unique_ptr<TrainStep> a(TrainStep::create(weights_g, weights_d, ca, dev, &err));
        if (a == nullptr) {
            std::printf("[grad-accum] A create failed: %s\n", err.c_str());
            return 1;
        }
        if (!a->run_step_accum(micro.data(), K, &La, &err)) {
            std::printf("[grad-accum] A run_step_accum failed: %s\n", err.c_str());
            return 1;
        }
        collect_list(a->g_params().params(), &ga_g);
        collect_list(a->d_params().params(), &ga_d);
    }
    std::printf("[grad-accum] A(batch=%lld,K=%lld) loss_disc=%.6f loss_gen_all=%.6f "
                "loss_gen=%.6f loss_fm=%.6f loss_mel=%.6f loss_kl=%.6f (G %zu / D %zu elements)\n",
                (long long) B, (long long) K, (double) La.loss_disc, (double) La.loss_gen_all,
                (double) La.loss_gen, (double) La.loss_fm, (double) La.loss_mel,
                (double) La.loss_kl, ga_g.size(), ga_d.size());

    // ---------------- arm B: full batch B*K ----------------
    std::vector<float> gb_g, gb_d;
    TrainStepLosses Lb{};
    {
        std::unique_ptr<TrainStep> b(TrainStep::create(weights_g, weights_d, cb, dev, &err));
        if (b == nullptr) {
            std::printf("[grad-accum] B create failed: %s\n", err.c_str());
            return 1;
        }
        if (!b->set_batch(combined, &err)) {
            std::printf("[grad-accum] B set_batch failed: %s\n", err.c_str());
            return 1;
        }
        if (!b->run_step(&Lb, &err)) {
            std::printf("[grad-accum] B run_step failed: %s\n", err.c_str());
            return 1;
        }
        collect_list(b->g_params().params(), &gb_g);
        collect_list(b->d_params().params(), &gb_d);
    }
    std::printf("[grad-accum] B(batch=%lld,K=1) loss_disc=%.6f loss_gen_all=%.6f "
                "loss_gen=%.6f loss_fm=%.6f loss_mel=%.6f loss_kl=%.6f (G %zu / D %zu elements)\n",
                (long long) (K * B), (double) Lb.loss_disc, (double) Lb.loss_gen_all,
                (double) Lb.loss_gen, (double) Lb.loss_fm, (double) Lb.loss_mel,
                (double) Lb.loss_kl, gb_g.size(), gb_d.size());

    // ---------------- comparison ----------------
    const double rel_g = grad_stats("G gradients", ga_g, gb_g, &failed);
    const double rel_d = grad_stats("D gradients", ga_d, gb_d, &failed);

    const bool loss_finite = finite6(La) && finite6(Lb);
    if (!loss_finite) {
        std::printf("[grad-accum] [FAIL] loss not finite\n");
        ++failed;
    }
    const double kGate = 1e-3;
    const double rel = std::max(rel_g, rel_d);
    if (!(rel <= kGate)) {
        std::printf("[grad-accum] [FAIL] gradient normalized relative error max(%.3e, %.3e) > gate %.0e\n", rel_g,
                    rel_d, kGate);
        ++failed;
    } else {
        std::printf("[grad-accum] gradients equivalent OK (max norm_rel=%.3e <= gate %.0e)\n", rel, kGate);
    }

    const double lrel = std::fabs((double) La.loss_gen_all - (double) Lb.loss_gen_all) /
                        std::max(std::fabs((double) Lb.loss_gen_all), 1e-12);
    std::printf("[grad-accum] loss_gen_all relative difference=%.3e (informational)\n", lrel);

    std::printf("[grad-accum] %s\n", failed == 0 ? "passed (gate G-7.3)" : "failed");
    return failed == 0 ? 0 : 1;
}

} // namespace train
} // namespace rvc_train
