// RVC-Train-GGML - M7-3 梯度累积等价自检实现（src/train/grad_accum_selftest.cpp）
//
// 门 G-7.3（梯度级等价）：
//   用同权重、同数据构造 A 臂（batch=B, grad_accum=K）与 B 臂（batch=B*K, grad_accum=1）：
//     A：TrainStep::run_step_accum(K 个 micro-batch) → 累积梯度（图内损失已按 1/K 缩放）
//     B：TrainStep::set_batch(合并批) + run_step → 全批量均值梯度
//   比较 G/D 全部参数 grad_acc：max_abs / 归一化相对误差 / rms；门 max_rel ≤ 1e-3。
//   同时打印两臂 6 项 loss 供观察（loss 应为有效 batch 均值）。
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

// 环境变量开关（MSVC getenv 触发 C4996，局部屏蔽）
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

// 与 Trainer::build_batch_into 一致：把每样本波形从 L 截断到 d
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

// 追加 src（B_s 个样本，B_s 为最慢维）到 dst（B_d 个样本），得到 B_d+B_s
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

// 收集参数梯度（grad_acc 优先，退回 grad）到扁平向量
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

// 对比两梯度向量，打印并返回归一化相对误差
double grad_stats(const char* tag, const std::vector<float>& a, const std::vector<float>& b,
                  int* failed) {
    if (a.size() != b.size()) {
        std::printf("[grad-accum] [FAIL] %s 元素数不一致（%zu != %zu）\n", tag, a.size(), b.size());
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
    std::printf("[grad-accum] %s：max_abs=%.3e max_ref=%.3e norm_rel=%.3e rms=%.3e\n", tag,
                max_abs, max_ref, rel, rms);
    return rel;
}

} // namespace

int run_grad_accum_selftest(const std::string& weights_g, const std::string& weights_d,
                            const std::string& exp_dir, traincpp::Device* dev, int64_t bucket,
                            int64_t batch, int64_t grad_accum, int64_t disc_len) {
    if (dev == nullptr) {
        std::printf("[grad-accum] 无可用设备\n");
        return 2;
    }
    const int64_t K = grad_accum > 0 ? grad_accum : 1;
    const int64_t B = batch > 0 ? batch : 1;
    if (K < 2) {
        std::printf("[grad-accum] --grad-accum 必须 >= 2（当前 %lld）\n", (long long) K);
        return 2;
    }
    if (weights_g.empty() || weights_d.empty() || exp_dir.empty()) {
        std::printf("[grad-accum] 需要 --weights <G> --d-weights <D> --exp <缓存>\n");
        return 2;
    }

    // 从 G 元数据推导缓存形状
    std::string err;
    std::unique_ptr<model::TrainWeights> wg(model::TrainWeights::open(weights_g, &err));
    if (wg == nullptr) {
        std::printf("[grad-accum] 打开 G 权重失败: %s\n", err.c_str());
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
        std::printf("[grad-accum] 打开缓存失败: %s\n", err.c_str());
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
        std::printf("[grad-accum] 可用样本 %lld < 需要 %lld（K*B）\n", (long long) eligible.size(),
                    (long long) need);
        return 2;
    }
    // 选取 aligned_frames 相同的 K*B 个样本：等长 → 所有归约（含官方 KL 的全局 sum/sum）
    // 逐样本可加，grad_accum=K 与 batch=K 应精确等价。跨长度时官方 KL 的全局归一化不可分，
    // 属已知语义差异（见 docs/开发进度.md M7-3 日志）。
    std::vector<int64_t> idx;
    {
        std::map<int64_t, std::vector<int64_t>> by_frames;
        for (int64_t i : eligible) {
            by_frames[ds.aligned_frames(i)].push_back(i);
        }
        for (const auto& kv : by_frames) {
            if ((int64_t) kv.second.size() >= need) {
                idx.assign(kv.second.begin(), kv.second.begin() + (size_t) need);
                std::printf("[grad-accum] 选定等长样本：aligned_frames=%lld × %lld\n",
                            (long long) kv.first, (long long) need);
                break;
            }
        }
        if (idx.empty()) {
            std::printf("[grad-accum] 缓存中无 %lld 个等长样本，退回重复样本（仅验证机制）\n",
                        (long long) need);
            for (int64_t i = 0; i < need; ++i) {
                idx.push_back(eligible[(size_t) (i % eligible.size())]);
            }
        }
    }

    std::printf("[grad-accum] 设备=%s bucket=%lld batch=%lld K=%lld disc_len=%lld "
                "可用样本=%lld\n",
                dev->props().name.c_str(), (long long) Tb, (long long) B, (long long) K,
                (long long) D, (long long) idx.size());

    // 用固定 seed 装配 K 个 micro-batch（B 样本/个），再 append 成 B*K 合并批
    traincpp::Rng rng;
    traincpp::rng_seed(&rng, 20260928u);
    std::vector<data::BatchBuffers> micro((size_t) K);
    for (int64_t m = 0; m < K; ++m) {
        std::vector<int64_t> mi(idx.begin() + (size_t) (m * B),
                                idx.begin() + (size_t) (m * B + B));
        if (!ds.build_batch(mi, Tb, &rng, nullptr, &micro[(size_t) m], &err)) {
            std::printf("[grad-accum] 装配 micro %lld 失败: %s\n", (long long) m, err.c_str());
            return 1;
        }
        truncate_wave(&micro[(size_t) m], D, seg_samples);
    }
    if (env_flag("RVCT_ACC_DUP")) {
        for (int64_t m = 1; m < K; ++m) {
            micro[(size_t) m] = micro[0];
        }
        std::printf("[grad-accum] RVCT_ACC_DUP：全部 micro 使用同一数据（诊断 D 的 batch 正确性）\n");
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

    // ---------------- A 臂：累积 K 个 micro-batch ----------------
    std::vector<float> ga_g, ga_d;
    TrainStepLosses La{};
    {
        std::unique_ptr<TrainStep> a(TrainStep::create(weights_g, weights_d, ca, dev, &err));
        if (a == nullptr) {
            std::printf("[grad-accum] A 创建失败: %s\n", err.c_str());
            return 1;
        }
        if (!a->run_step_accum(micro.data(), K, &La, &err)) {
            std::printf("[grad-accum] A run_step_accum 失败: %s\n", err.c_str());
            return 1;
        }
        collect_list(a->g_params().params(), &ga_g);
        collect_list(a->d_params().params(), &ga_d);
    }
    std::printf("[grad-accum] A(batch=%lld,K=%lld) loss_disc=%.6f loss_gen_all=%.6f "
                "loss_gen=%.6f loss_fm=%.6f loss_mel=%.6f loss_kl=%.6f（G %zu / D %zu 元素）\n",
                (long long) B, (long long) K, (double) La.loss_disc, (double) La.loss_gen_all,
                (double) La.loss_gen, (double) La.loss_fm, (double) La.loss_mel,
                (double) La.loss_kl, ga_g.size(), ga_d.size());

    // ---------------- B 臂：全批量 B*K ----------------
    std::vector<float> gb_g, gb_d;
    TrainStepLosses Lb{};
    {
        std::unique_ptr<TrainStep> b(TrainStep::create(weights_g, weights_d, cb, dev, &err));
        if (b == nullptr) {
            std::printf("[grad-accum] B 创建失败: %s\n", err.c_str());
            return 1;
        }
        if (!b->set_batch(combined, &err)) {
            std::printf("[grad-accum] B set_batch 失败: %s\n", err.c_str());
            return 1;
        }
        if (!b->run_step(&Lb, &err)) {
            std::printf("[grad-accum] B run_step 失败: %s\n", err.c_str());
            return 1;
        }
        collect_list(b->g_params().params(), &gb_g);
        collect_list(b->d_params().params(), &gb_d);
    }
    std::printf("[grad-accum] B(batch=%lld,K=1) loss_disc=%.6f loss_gen_all=%.6f "
                "loss_gen=%.6f loss_fm=%.6f loss_mel=%.6f loss_kl=%.6f（G %zu / D %zu 元素）\n",
                (long long) (K * B), (double) Lb.loss_disc, (double) Lb.loss_gen_all,
                (double) Lb.loss_gen, (double) Lb.loss_fm, (double) Lb.loss_mel,
                (double) Lb.loss_kl, gb_g.size(), gb_d.size());

    // ---------------- 对比 ----------------
    const double rel_g = grad_stats("G 梯度", ga_g, gb_g, &failed);
    const double rel_d = grad_stats("D 梯度", ga_d, gb_d, &failed);

    const bool loss_finite = finite6(La) && finite6(Lb);
    if (!loss_finite) {
        std::printf("[grad-accum] [FAIL] loss 非有限\n");
        ++failed;
    }
    const double kGate = 1e-3;
    const double rel = std::max(rel_g, rel_d);
    if (!(rel <= kGate)) {
        std::printf("[grad-accum] [FAIL] 梯度归一化相对误差 max(%.3e, %.3e) > 门 %.0e\n", rel_g,
                    rel_d, kGate);
        ++failed;
    } else {
        std::printf("[grad-accum] 梯度等价 OK（max norm_rel=%.3e <= 门 %.0e）\n", rel, kGate);
    }

    const double lrel = std::fabs((double) La.loss_gen_all - (double) Lb.loss_gen_all) /
                        std::max(std::fabs((double) Lb.loss_gen_all), 1e-12);
    std::printf("[grad-accum] loss_gen_all 相对差=%.3e（信息性）\n", lrel);

    std::printf("[grad-accum] %s\n", failed == 0 ? "通过（门 G-7.3）" : "失败");
    return failed == 0 ? 0 : 1;
}

} // namespace train
} // namespace rvc_train
