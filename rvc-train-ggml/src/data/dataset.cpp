// dataset.cpp —— M5 训练数据加载器实现（装配语义见 dataset.h 头注释）

#include "data/dataset.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "model/sinegen.h"
#include "train/stft_mel.h"

namespace rvc_train {
namespace data {

namespace {

constexpr int64_t kMaxPhoneFrames = 900;  // 官方 get_labels n_num 上界
constexpr uint32_t kSineSeed = 1234;      // 噪声缺省时的 SineGen 种子（与 M3-3 一致）

bool validate_config(const TrainDataConfig& cfg, std::string* err) {
    auto fail = [&](const std::string& msg) {
        if (err != nullptr) {
            *err = msg;
        }
        return false;
    };
    if (cfg.exp_dir.empty()) {
        return fail("TrainDataConfig.exp_dir 为空");
    }
    if (cfg.batch_size <= 0 || cfg.segment_frames <= 0 || cfg.segment_samples <= 0) {
        return fail("batch_size/segment_frames/segment_samples 必须 > 0");
    }
    if (cfg.segment_samples % cfg.segment_frames != 0) {
        return fail("segment_samples 必须是 segment_frames 的整数倍");
    }
    if (cfg.q_channels <= 0) {
        return fail("q_channels 必须 > 0");
    }
    if (cfg.n_fft <= 0 || cfg.win <= 0 || cfg.n_mels <= 0 || cfg.sr <= 0.0) {
        return fail("mel 配置非法（n_fft/win/n_mels/sr）");
    }
    int64_t prev = 0;
    for (const int64_t b : cfg.buckets) {
        if (b <= 0 || b <= prev) {
            return fail("buckets 必须为严格升序正整数");
        }
        prev = b;
    }
    return true;
}

} // namespace

bool Dataset::open(const std::string& cache_dir, const TrainDataConfig& cfg, Dataset* out,
                   std::string* err) {
    if (out == nullptr) {
        if (err != nullptr) {
            *err = "Dataset::open: out 为空";
        }
        return false;
    }
    if (!validate_config(cfg, err)) {
        return false;
    }
    Dataset ds;
    ds.cfg_ = cfg;
    ds.cfg_.exp_dir = cache_dir;
    if (!prep::load_cache_index(cache_dir, &ds.cache_, err)) {
        return false;
    }
    const prep::CacheMeta& meta = ds.cache_.meta;
    const int64_t hop = cfg.segment_samples / cfg.segment_frames;
    if (meta.spec.hop != hop) {
        if (err != nullptr) {
            *err = "缓存 spec.hop=" + std::to_string(meta.spec.hop) + " 与配置 hop=" +
                   std::to_string(hop) + " 不一致";
        }
        return false;
    }
    if (meta.spec.n_fft != cfg.n_fft || meta.spec.win != cfg.win) {
        if (err != nullptr) {
            *err = "缓存 spec.n_fft/win 与配置不一致";
        }
        return false;
    }
    if (meta.target_sr != static_cast<int>(cfg.sr)) {
        if (err != nullptr) {
            *err = "缓存 target_sr=" + std::to_string(meta.target_sr) + " 与配置 sr=" +
                   std::to_string(static_cast<int>(cfg.sr)) + " 不一致";
        }
        return false;
    }
    // mel 滤波器组（librosa slaney；与 M3-2 golden 同源）
    train::StftMelConfig smcfg;
    smcfg.n_fft = cfg.n_fft;
    smcfg.hop = hop;
    smcfg.win = cfg.win;
    smcfg.n_mels = cfg.n_mels;
    smcfg.sr = cfg.sr;
    smcfg.fmin = cfg.fmin;
    smcfg.fmax = cfg.fmax;
    smcfg.log_clip = cfg.log_clip;
    std::string merr;
    if (!smcfg.validate(&merr)) {
        if (err != nullptr) {
            *err = "mel 配置无效: " + merr;
        }
        return false;
    }
    const int64_t bins = smcfg.bins();
    ds.mel_basis_.resize(static_cast<size_t>(cfg.n_mels * bins));
    train::make_mel_filterbank_slaney(smcfg, ds.mel_basis_.data());

    ds.index_map_.clear();
    for (std::size_t i = 0; i < ds.cache_.samples.size(); ++i) {
        ds.index_map_.emplace(ds.cache_.samples[i].id, static_cast<int64_t>(i));
    }
    *out = std::move(ds);
    if (err != nullptr) {
        err->clear();
    }
    return true;
}

const prep::CacheSampleInfo& Dataset::sample_info(int64_t index) const {
    static const prep::CacheSampleInfo kEmpty;
    if (index < 0 || index >= size()) {
        return kEmpty;
    }
    return cache_.samples[static_cast<size_t>(index)];
}

int64_t Dataset::index_of(const std::string& id) const {
    const auto it = index_map_.find(id);
    return it == index_map_.end() ? -1 : it->second;
}

int64_t Dataset::aligned_frames(int64_t index) const {
    const prep::CacheSampleInfo& s = sample_info(index);
    if (s.spec_frames <= 0) {
        return 0;
    }
    const int64_t phone_rows = std::min<int64_t>(2 * s.feat_frames, kMaxPhoneFrames);
    return std::min(phone_rows, s.spec_frames);
}

int64_t Dataset::bucket_of(int64_t frames) const {
    for (const int64_t b : cfg_.buckets) {
        if (b >= frames) {
            return b;
        }
    }
    return -1;
}

bool Dataset::assemble(int64_t index, TrainSample* out, std::string* err) const {
    if (out == nullptr || index < 0 || index >= size()) {
        if (err != nullptr) {
            *err = "assemble: 样本下标越界 " + std::to_string(index);
        }
        return false;
    }
    const prep::CacheSampleInfo& s = cache_.samples[static_cast<size_t>(index)];
    std::vector<float> feat;
    std::vector<int32_t> f0bin;
    std::vector<float> f0hz;
    std::vector<float> spec;
    std::vector<float> gt;
    if (!prep::read_cache_sample_f32(cache_.dir, s, "feat", &feat, err) ||
        !prep::read_cache_sample_i32(cache_.dir, s, "f0bin", &f0bin, err) ||
        !prep::read_cache_sample_f32(cache_.dir, s, "f0hz", &f0hz, err) ||
        !prep::read_cache_sample_f32(cache_.dir, s, "spec", &spec, err) ||
        !prep::read_cache_sample_f32(cache_.dir, s, "gt", &gt, err)) {
        return false;
    }
    auto fail = [&](const std::string& msg) {
        if (err != nullptr) {
            *err = "样本 " + s.id + ": " + msg;
        }
        return false;
    };

    const int64_t feat_dim = s.feat_dim;
    const int64_t spec_bins = s.spec_bins;
    const int64_t hop = cfg_.segment_samples / cfg_.segment_frames;
    if (feat_dim <= 0 || spec_bins <= 0) {
        return fail("feat_dim/spec_bins 非正");
    }
    // 官方 get_labels：repeat(feat,2) → min(·,900)
    const int64_t phone_rows = std::min<int64_t>(2 * s.feat_frames, kMaxPhoneFrames);
    if (s.f0_frames < phone_rows) {
        return fail("f0_frames=" + std::to_string(s.f0_frames) + " 少于 phone 行数 " +
                    std::to_string(phone_rows));
    }
    // 与 spec 对齐取 min（官方 get_audio_text_pair）
    const int64_t T = std::min(phone_rows, s.spec_frames);
    if (T <= 0) {
        return fail("对齐后帧数 T<=0");
    }
    if (s.gt_samples < T * hop) {
        return fail("gt_samples=" + std::to_string(s.gt_samples) + " 少于 T*hop=" +
                    std::to_string(T * hop));
    }

    out->id = s.id;
    out->frames = T;
    out->feat_dim = feat_dim;
    out->spec_bins = spec_bins;
    out->hop = hop;
    out->phone.resize(static_cast<size_t>(T * feat_dim));
    for (int64_t t = 0; t < T; ++t) {
        const int64_t src = t / 2;
        std::memcpy(&out->phone[static_cast<size_t>(t * feat_dim)],
                    &feat[static_cast<size_t>(src * feat_dim)],
                    static_cast<size_t>(feat_dim) * sizeof(float));
    }
    out->pitch.assign(f0bin.begin(), f0bin.begin() + static_cast<std::ptrdiff_t>(T));
    out->pitchf.assign(f0hz.begin(), f0hz.begin() + static_cast<std::ptrdiff_t>(T));
    out->spec.resize(static_cast<size_t>(spec_bins * T));
    for (int64_t f = 0; f < spec_bins; ++f) {
        std::memcpy(&out->spec[static_cast<size_t>(f * T)],
                    &spec[static_cast<size_t>(f * s.spec_frames)],
                    static_cast<size_t>(T) * sizeof(float));
    }
    out->wav.assign(gt.begin(), gt.begin() + static_cast<std::ptrdiff_t>(T * hop));
    if (err != nullptr) {
        err->clear();
    }
    return true;
}

bool Dataset::assemble_id(const std::string& id, TrainSample* out, std::string* err) const {
    const int64_t index = index_of(id);
    if (index < 0) {
        if (err != nullptr) {
            *err = "样本不存在: " + id;
        }
        return false;
    }
    return assemble(index, out, err);
}

bool Dataset::build_batch(const std::vector<int64_t>& indices, int64_t bucket_top,
                          traincpp::Rng* rng, const BatchRandomInputs* injected, BatchBuffers* out,
                          std::string* err) const {
    if (out == nullptr) {
        if (err != nullptr) {
            *err = "build_batch: out 为空";
        }
        return false;
    }
    auto fail = [&](const std::string& msg) {
        if (err != nullptr) {
            *err = "build_batch: " + msg;
        }
        return false;
    };
    const int64_t B = static_cast<int64_t>(indices.size());
    if (B <= 0) {
        return fail("批次为空");
    }
    const int64_t seg = cfg_.segment_frames;
    if (bucket_top < seg) {
        return fail("bucket_top=" + std::to_string(bucket_top) + " 小于 segment_frames=" +
                    std::to_string(seg));
    }
    std::vector<TrainSample> samples(static_cast<size_t>(B));
    for (int64_t b = 0; b < B; ++b) {
        if (!assemble(indices[static_cast<size_t>(b)], &samples[static_cast<size_t>(b)], err)) {
            return false;
        }
        const TrainSample& s = samples[static_cast<size_t>(b)];
        if (s.frames < seg) {
            return fail("样本 " + s.id + " 帧数 " + std::to_string(s.frames) +
                        " 小于 segment_frames（官方 min_text_len 过滤）");
        }
        if (s.frames > bucket_top) {
            return fail("样本 " + s.id + " 帧数 " + std::to_string(s.frames) +
                        " 超过桶顶 " + std::to_string(bucket_top));
        }
        if (b > 0 && (s.feat_dim != samples[0].feat_dim || s.spec_bins != samples[0].spec_bins)) {
            return fail("批次内 feat_dim/spec_bins 不一致");
        }
    }

    out->frames = bucket_top;
    out->batch = B;
    out->feat_dim = samples[0].feat_dim;
    out->spec_bins = samples[0].spec_bins;
    out->segment_frames = seg;
    out->segment_samples = cfg_.segment_samples;
    out->q_channels = cfg_.q_channels;
    out->hop = samples[0].hop;
    const int64_t C = out->feat_dim;
    const int64_t bins = out->spec_bins;
    const int64_t L = cfg_.segment_samples;
    const int64_t Q = cfg_.q_channels;
    const int64_t Tb = bucket_top;

    out->lengths.resize(static_cast<size_t>(B));
    out->ids_slice.assign(static_cast<size_t>(B), 0);
    out->phone.assign(static_cast<size_t>(C * Tb * B), 0.0f);
    out->pitch.assign(static_cast<size_t>(Tb * B), 0);
    out->pitchf.assign(static_cast<size_t>(Tb * B), 0.0f);
    out->spec.assign(static_cast<size_t>(Tb * bins * B), 0.0f);
    out->wave.assign(static_cast<size_t>(L * B), 0.0f);
    out->y_mel.assign(static_cast<size_t>(seg * cfg_.n_mels * B), 0.0f);
    out->sid.assign(static_cast<size_t>(B), static_cast<int32_t>(cfg_.speaker_id));
    out->q_noise.assign(static_cast<size_t>(Tb * Q * B), 0.0f);
    out->x_mask.assign(static_cast<size_t>(Tb * B), 0.0f);
    out->y_mask.assign(static_cast<size_t>(Tb * B), 0.0f);
    out->attn_valid.assign(static_cast<size_t>(Tb * Tb * B), 0.0f);
    out->slice_mat.assign(static_cast<size_t>(Tb * seg * B), 0.0f);
    out->pitchf_slice.assign(static_cast<size_t>(seg * B), 0.0f);
    out->sine_source_slice.assign(static_cast<size_t>(L * B), 0.0f);

    for (int64_t b = 0; b < B; ++b) {
        const TrainSample& s = samples[static_cast<size_t>(b)];
        const int64_t Ti = s.frames;
        out->lengths[static_cast<size_t>(b)] = Ti;
        for (int64_t t = 0; t < Ti; ++t) {
            std::memcpy(&out->phone[static_cast<size_t>(C * t + C * Tb * b)],
                        &s.phone[static_cast<size_t>(t * C)],
                        static_cast<size_t>(C) * sizeof(float));
            out->pitch[static_cast<size_t>(t + Tb * b)] = s.pitch[static_cast<size_t>(t)];
            out->pitchf[static_cast<size_t>(t + Tb * b)] = s.pitchf[static_cast<size_t>(t)];
            out->x_mask[static_cast<size_t>(t + Tb * b)] = 1.0f;
            out->y_mask[static_cast<size_t>(t + Tb * b)] = 1.0f;
        }
        for (int64_t f = 0; f < bins; ++f) {
            std::memcpy(&out->spec[static_cast<size_t>(Tb * f + Tb * bins * b)],
                        &s.spec[static_cast<size_t>(f * Ti)],
                        static_cast<size_t>(Ti) * sizeof(float));
        }
        // attn_valid = x_mask 外积（与 train_step_selftest 同式）
        for (int64_t tq = 0; tq < Ti; ++tq) {
            for (int64_t tk = 0; tk < Ti; ++tk) {
                out->attn_valid[static_cast<size_t>(tk + Tb * tq + Tb * Tb * b)] = 1.0f;
            }
        }
    }

    // ---------------- 随机量：ids_slice / q_noise / SineGen 噪声 ----------------
    if (injected != nullptr) {
        if (static_cast<int64_t>(injected->ids_slice.size()) != B) {
            return fail("注入 ids_slice 尺寸不符");
        }
        if (static_cast<int64_t>(injected->q_noise.size()) != Tb * Q * B) {
            return fail("注入 q_noise 尺寸不符（期望 " + std::to_string(Tb * Q * B) + "）");
        }
        if (static_cast<int64_t>(injected->sine_noise.size()) != L * B) {
            return fail("注入 sine_noise 尺寸不符");
        }
        out->ids_slice = injected->ids_slice;
        out->q_noise = injected->q_noise;
    } else if (rng != nullptr) {
        for (int64_t b = 0; b < B; ++b) {
            const int64_t span = samples[static_cast<size_t>(b)].frames - seg + 1;
            const float u = traincpp::rng_uniform01(rng);
            out->ids_slice[static_cast<size_t>(b)] =
                static_cast<int32_t>(std::floor(static_cast<double>(u) * static_cast<double>(span)));
        }
        for (int64_t b = 0; b < B; ++b) {
            for (int64_t c = 0; c < Q; ++c) {
                for (int64_t t = 0; t < Tb; ++t) {
                    out->q_noise[static_cast<size_t>(t + Tb * c + Tb * Q * b)] =
                        traincpp::rng_normal(rng, 0.0f, 1.0f);
                }
            }
        }
    } else {
        return fail("未注入随机量且 rng 为空");
    }
    for (int64_t b = 0; b < B; ++b) {
        const int64_t id = out->ids_slice[static_cast<size_t>(b)];
        const int64_t span = samples[static_cast<size_t>(b)].frames - seg + 1;
        if (id < 0 || id > span - 1) {
            return fail("ids_slice[" + std::to_string(b) + "]=" + std::to_string(id) +
                        " 越界（样本帧数 " + std::to_string(samples[static_cast<size_t>(b)].frames) +
                        "）");
        }
    }

    // ---------------- 派生缓冲：slice_mat / pitchf_slice / wave / y_mel / sine ----------------
    std::vector<float> sine_noise_tmp;
    const float* sine_noise = nullptr;
    if (injected != nullptr) {
        sine_noise = injected->sine_noise.data();
    } else {
        sine_noise_tmp.resize(static_cast<size_t>(L * B));
        for (float& v : sine_noise_tmp) {
            v = traincpp::rng_normal(rng, 0.0f, 1.0f);
        }
        sine_noise = sine_noise_tmp.data();
    }

    const int64_t upp = L / seg;
    for (int64_t b = 0; b < B; ++b) {
        const TrainSample& s = samples[static_cast<size_t>(b)];
        const int64_t id = out->ids_slice[static_cast<size_t>(b)];
        for (int64_t k = 0; k < seg; ++k) {
            out->slice_mat[static_cast<size_t>(id + k + Tb * k + Tb * seg * b)] = 1.0f;
            out->pitchf_slice[static_cast<size_t>(k + seg * b)] = s.pitchf[static_cast<size_t>(id + k)];
        }
        std::memcpy(&out->wave[static_cast<size_t>(L * b)],
                    &s.wav[static_cast<size_t>(id * s.hop)],
                    static_cast<size_t>(L) * sizeof(float));
        // y_mel = log(clamp(mel_basis · spec[:, id:id+seg], min=log_clip))；双精度累加
        for (int64_t m = 0; m < seg; ++m) {
            const int64_t col = id + m;
            for (int64_t n = 0; n < cfg_.n_mels; ++n) {
                double acc = 0.0;
                const float* basis = &mel_basis_[static_cast<size_t>(n * bins)];
                for (int64_t f = 0; f < bins; ++f) {
                    acc += static_cast<double>(basis[static_cast<size_t>(f)]) *
                           static_cast<double>(s.spec[static_cast<size_t>(f * s.frames + col)]);
                }
                const double clipped = std::max(acc, static_cast<double>(cfg_.log_clip));
                out->y_mel[static_cast<size_t>(m + seg * n + seg * cfg_.n_mels * b)] =
                    static_cast<float>(std::log(clipped));
            }
        }
        // SineGen（旧 cumsum_shift 法；复用已验证的主机实现）
        std::vector<float> f0s(static_cast<size_t>(seg));
        for (int64_t k = 0; k < seg; ++k) {
            f0s[static_cast<size_t>(k)] = out->pitchf_slice[static_cast<size_t>(k + seg * b)];
        }
        std::vector<float> sine = model::sinegen_generate(
            f0s.data(), seg, static_cast<int>(cfg_.sr), static_cast<int>(upp),
            sine_noise + static_cast<std::ptrdiff_t>(L * b), L, kSineSeed);
        if (static_cast<int64_t>(sine.size()) != L) {
            return fail("SineGen 输出长度不符");
        }
        std::memcpy(&out->sine_source_slice[static_cast<size_t>(L * b)], sine.data(),
                    static_cast<size_t>(L) * sizeof(float));
    }
    if (err != nullptr) {
        err->clear();
    }
    return true;
}

} // namespace data
} // namespace rvc_train
