// pipeline.cpp
// RVC full offline conversion pipeline implementation (aligned sentence by sentence with infer/modules/vc/pipeline.py).

#include "rvc/pipeline.h"

#include "audio/audio_io.h"
#include "audio/dsp.h"
#include "hubert/hubert.h"
#include "rmvpe/rmvpe.h"
#include "rvc/retrieval.h"
#include "rvc/sinegen.h"
#include "utils.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <stdexcept>

namespace rvc {

namespace {

    constexpr int64_t kWindow = 160;         // 10 ms per frame @16k
    constexpr float kRmvpeThreshold = 0.03f;

    bool load_raw_f32(const std::string& path, std::vector<float>& out) {
        std::ifstream in(path, std::ios::binary | std::ios::ate);
        if (!in) {
            return false;
        }
        const size_t bytes = static_cast<size_t>(in.tellg());
        out.resize(bytes / sizeof(float));
        in.seekg(0);
        in.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(bytes));
        return static_cast<bool>(in) || in.eof();
    }

    // ×2 nearest-neighbor upsampling (aligned with F.interpolate(scale_factor=2); mode defaults to nearest)
    std::vector<float> upsample2_nearest(const std::vector<float>& x, int64_t frames, int64_t dim) {
        std::vector<float> out(static_cast<size_t>(2 * frames * dim));
        for (int64_t t = 0; t < frames; ++t) {
            const float* src = x.data() + t * dim;
            std::copy(src, src + dim, out.data() + (2 * t) * dim);
            std::copy(src, src + dim, out.data() + (2 * t + 1) * dim);
        }
        return out;
    }

} // namespace

struct RvcPipeline::Impl {
    const GgufModel& hubert_model;
    const GgufModel& rmvpe_model;
    const GgufModel& voice_model;
    ggml_backend_t backend = nullptr;

    hubert::HubertEncoder hubert;
    rmvpe::RmvpeExtractor rmvpe;
    vits::Synthesizer synth;

    PipelineConfig cfg;
    float l_linear_w = 0.0f;
    float l_linear_b = 0.0f;
    int call_index = 0;

    Impl(const GgufModel& hubert_m, const GgufModel& rmvpe_m, const GgufModel& voice_m,
        ggml_backend_t backend_)
        : hubert_model(hubert_m), rmvpe_model(rmvpe_m), voice_model(voice_m),
        backend(backend_),
        hubert(hubert_m, backend_),
        rmvpe(rmvpe_m, backend_),
        synth(voice_m, backend_) {
        if (synth.config().has_f0) {
            l_linear_w = voice_model.read_f32("dec.m_source.l_linear.weight")[0];
            l_linear_b = voice_model.read_f32("dec.m_source.l_linear.bias")[0];
        }
    }

    // Single-chunk VC: aligned with Pipeline.vc()
    std::vector<float> vc(const std::vector<float>& audio0,
        const std::vector<int32_t>& pitch_all, const std::vector<float>& pitchf_all,
        int64_t p_begin, int64_t p_end, const ConvertOptions& opt) {
        const vits::Config& vcfg = synth.config();
        const bool v2 = (vcfg.version == "v2");
        const bool has_f0 = vcfg.has_f0;
        const int64_t inter = vcfg.inter_channels;

        // HuBERT features [T_tok, C]
        hubert::Features feats = hubert.extract(audio0, v2);
        const int64_t T_tok = feats.frames;
        const int64_t C = feats.dim;

        std::vector<float> feats0;
        if (has_f0 && opt.protect < 0.5f) {
            feats0 = feats.values;
        }

        // Retrieval blending (before ×2 interpolation)
        if (opt.index != nullptr && opt.index->loaded() && opt.index_rate != 0.0f) {
            feats.values = opt.index->blend(feats.values, T_tok, 8, opt.index_rate);
        }

        // ×2 nearest-neighbor interpolation
        feats.values = upsample2_nearest(feats.values, T_tok, C);
        feats.frames = T_tok * 2;
        if (!feats0.empty()) {
            feats0 = upsample2_nearest(feats0, T_tok, C);
        }

        // p_len: the smaller of audio0/160 and feats frame count
        int64_t p_len = static_cast<int64_t>(audio0.size()) / kWindow;
        if (feats.frames < p_len) {
            p_len = feats.frames;
        }

        // pitch slicing + truncation
        std::vector<int32_t> pitch;
        std::vector<float> pitchf;
        if (has_f0) {
            const int64_t avail = std::max<int64_t>(0, p_end - p_begin);
            if (p_len > avail) {
                p_len = avail;
            }
            pitch.assign(pitch_all.begin() + p_begin, pitch_all.begin() + p_begin + p_len);
            pitchf.assign(pitchf_all.begin() + p_begin, pitchf_all.begin() + p_begin + p_len);
        }
        if (p_len > feats.frames) {
            p_len = feats.frames;
        }
        if (p_len <= 0) {
            return {};
        }

        // Validation mode: overwrite with reference f0 (isolates flips caused by near-ties in RMVPE)
        if (has_f0 && opt.f0_inject && !opt.noise_dir.empty()) {
            char name[128];
            snprintf(name, sizeof(name), "e2e_call%03d_pitchf.raw", call_index);
            std::vector<float> injected;
            if (!load_raw_f32(opt.noise_dir + "/" + name, injected) || injected.empty()) {
                throw std::runtime_error("f0 injection file missing: " + std::string(name));
            }
            p_len = std::min<int64_t>(static_cast<int64_t>(injected.size()), feats.frames);
            pitchf.assign(injected.begin(), injected.begin() + p_len);
            pitch.resize(static_cast<size_t>(p_len));
            for (int64_t i = 0; i < p_len; ++i) {
                pitch[static_cast<size_t>(i)] = dsp::coarse_pitch_bin(pitchf[static_cast<size_t>(i)]);
            }
        }

        // protect: replace unvoiced frames with the original features (aligned with pitchff semantics)
        if (!feats0.empty()) {
            for (int64_t t = 0; t < p_len; ++t) {
                const float pf = pitchf[static_cast<size_t>(t)] >= 1.0f ? 1.0f : opt.protect;
                float* dst = feats.values.data() + t * C;
                const float* src0 = feats0.data() + t * C;
                for (int64_t c = 0; c < C; ++c) {
                    dst[c] = dst[c] * pf + src0[c] * (1.0f - pf);
                }
            }
        }

        // z_p noise ([p_len, inter] row-major)
        std::vector<float> noise;
        std::vector<float> sine_noise;
        const std::vector<float>* sine_noise_ptr = nullptr;
        if (!opt.noise_dir.empty()) {
            char name[128];
            snprintf(name, sizeof(name), "e2e_call%03d_rnd.raw", call_index);
            if (!load_raw_f32(opt.noise_dir + "/" + name, noise) ||
                noise.size() != static_cast<size_t>(p_len * inter)) {
                throw std::runtime_error("noise file missing / size mismatch: " + std::string(name));
            }
            if (has_f0) {
                snprintf(name, sizeof(name), "e2e_call%03d_sine_noise.raw", call_index);
                if (!load_raw_f32(opt.noise_dir + "/" + name, sine_noise) ||
                    sine_noise.size() != static_cast<size_t>(p_len * vcfg.hop_samples)) {
                    throw std::runtime_error("Sine noise file missing / size mismatch: " + std::string(name));
                }
                sine_noise_ptr = &sine_noise;
            }
        }
        else {
            // Production mode: fixed seed + per-chunk offset (reproducible, with different noise per chunk)
            std::mt19937 rng(1234u + static_cast<uint32_t>(call_index) * 7919u);
            std::normal_distribution<float> dist(0.0f, 1.0f);
            noise.resize(static_cast<size_t>(p_len * inter));
            for (auto& v : noise) {
                v = dist(rng);
            }
        }
        const uint32_t sine_seed = 4321u + static_cast<uint32_t>(call_index) * 7919u;
        ++call_index;

        // Excitation source: SineGen -> tanh(l_linear(·))
        std::vector<float> source;
        if (has_f0) {
            source = sinegen::generate(pitchf, vcfg.sample_rate,
                static_cast<int>(vcfg.hop_samples), sine_noise_ptr, sine_seed);
            for (auto& v : source) {
                v = std::tanh(l_linear_w * v + l_linear_b);
            }
        }

        // VITS
        vits::Input in;
        in.frames = p_len;
        in.feature_dim = static_cast<int>(C);
        in.speaker_id = opt.sid;
        in.features.assign(feats.values.begin(), feats.values.begin() + p_len * C);
        in.pitch = std::move(pitch);
        in.source = std::move(source);
        in.noise = std::move(noise);

        // Validation mode: dump intermediates for comparison (pitchf is in Hz, aligned with the reference dump)
        if (!opt.noise_dir.empty()) {
            char name[128];
            snprintf(name, sizeof(name), "e2e_cpp_call%03d_features.raw", call_index - 1);
            std::ofstream f(opt.noise_dir + "/" + name, std::ios::binary);
            f.write(reinterpret_cast<const char*>(in.features.data()),
                static_cast<std::streamsize>(in.features.size() * sizeof(float)));
            snprintf(name, sizeof(name), "e2e_cpp_call%03d_pitchf.raw", call_index - 1);
            std::ofstream fp(opt.noise_dir + "/" + name, std::ios::binary);
            fp.write(reinterpret_cast<const char*>(pitchf.data()),
                static_cast<std::streamsize>(pitchf.size() * sizeof(float)));
            snprintf(name, sizeof(name), "e2e_cpp_call%03d_pitch_bins.raw", call_index - 1);
            std::ofstream fb(opt.noise_dir + "/" + name, std::ios::binary);
            fb.write(reinterpret_cast<const char*>(in.pitch.data()),
                static_cast<std::streamsize>(in.pitch.size() * sizeof(int32_t)));
        }

        vits::Output out = synth.infer(in);

        // Trim leading/trailing pad (t_pad_tgt = tgt_sr * x_pad)
        const int64_t trim = static_cast<int64_t>(vcfg.sample_rate) * cfg.x_pad;
        if (static_cast<int64_t>(out.audio.size()) <= 2 * trim) {
            return {};
        }
        return std::vector<float>(out.audio.begin() + trim, out.audio.end() - trim);
    }

    ConvertResult run(const std::vector<float>& audio_in, const ConvertOptions& opt) {
        const vits::Config& vcfg = synth.config();
        const int tgt_sr = vcfg.sample_rate;
        const bool has_f0 = vcfg.has_f0;
        call_index = 0;

        // 1) 48Hz high-pass
        std::vector<float> audio = audio_in;
        dsp::high_pass_48hz_in_place(audio);
        const int64_t n = static_cast<int64_t>(audio.size());

        // 2) Chunking at silent points (aligned with pipeline.py; the condition includes the window offset)
        std::vector<int64_t> opt_ts;
        if (n + kWindow > static_cast<int64_t>(cfg.x_max) * 16000) {
            const auto sum = dsp::audio_sum_curve(audio);
            const int64_t t_query = static_cast<int64_t>(cfg.x_query) * 16000;
            const int64_t t_center = static_cast<int64_t>(cfg.x_center) * 16000;
            for (int64_t t = t_center; t < n; t += t_center) {
                const int64_t begin = std::max<int64_t>(0, t - t_query);
                const int64_t end = std::min<int64_t>(n, t + t_query);
                if (begin >= end) {
                    continue;
                }
                int64_t best = begin;
                float best_value = sum[static_cast<size_t>(begin)];
                for (int64_t i = begin + 1; i < end; ++i) {
                    const float v = sum[static_cast<size_t>(i)];
                    if (v < best_value) {
                        best_value = v;
                        best = i;
                    }
                }
                opt_ts.push_back(t - t_query + (best - begin));
            }
        }

        // 3) Reflection pad
        const int64_t t_pad = 16000 * cfg.x_pad;
        std::vector<float> audio_pad = dsp::reflect_pad(audio, t_pad, t_pad);
        const int64_t p_len_full = static_cast<int64_t>(audio_pad.size()) / kWindow;

        // 4) f0 (whole segment, based on padded audio)
        std::vector<int32_t> pitch_all;
        std::vector<float> pitchf_all;
        if (has_f0) {
            rmvpe::Result r = rmvpe.infer(audio_pad, kRmvpeThreshold);
            std::vector<float> f0 = std::move(r.f0);
            const float semitone = std::pow(2.0f, opt.f0_up_key / 12.0f);
            for (auto& v : f0) {
                v *= semitone;
            }
            if (static_cast<int64_t>(f0.size()) > p_len_full) {
                f0.resize(static_cast<size_t>(p_len_full));
            }
            pitchf_all = f0;
            pitch_all.resize(f0.size());
            for (size_t i = 0; i < f0.size(); ++i) {
                pitch_all[i] = dsp::coarse_pitch_bin(f0[i]);
            }
            if (opt.verbose) {
                const auto voiced = static_cast<int64_t>(std::count_if(
                    f0.begin(), f0.end(), [](float v) { return v > 0.0f; }));
                Logger::info("f0: " + std::to_string(f0.size()) +
                    " frames (voiced " + std::to_string(voiced) + ")");
            }
        }

        // 5) Chunked VC
        std::vector<std::vector<float>> chunks;
        int64_t s = 0;
        for (int64_t t : opt_ts) {
            t = t / kWindow * kWindow;
            const int64_t chunk_end = std::min<int64_t>(t + 2 * t_pad + kWindow,
                static_cast<int64_t>(audio_pad.size()));
            std::vector<float> audio0(audio_pad.begin() + s, audio_pad.begin() + chunk_end);
            int64_t pb = s / kWindow;
            int64_t pe = (t + 2 * t_pad) / kWindow;
            if (pe > static_cast<int64_t>(pitch_all.size())) {
                pe = static_cast<int64_t>(pitch_all.size());
            }
            if (pb > pe) {
                pb = pe;
            }
            if (opt.verbose) {
                Logger::info("Chunk: samples [" + std::to_string(s) + ", " +
                    std::to_string(chunk_end) + ")");
            }
            chunks.push_back(vc(audio0, pitch_all, pitchf_all, pb, pe, opt));
            s = t;
        }
        {
            std::vector<float> audio0(audio_pad.begin() + s, audio_pad.end());
            const int64_t pb = s / kWindow;
            const int64_t pe = static_cast<int64_t>(pitch_all.size());
            if (opt.verbose) {
                Logger::info("Last chunk: samples [" + std::to_string(s) + ", " +
                    std::to_string(audio_pad.size()) + ")");
            }
            chunks.push_back(vc(audio0, pitch_all, pitchf_all, pb, pe, opt));
        }

        size_t total = 0;
        for (const auto& c : chunks) {
            total += c.size();
        }
        std::vector<float> audio_opt;
        audio_opt.reserve(total);
        for (const auto& c : chunks) {
            audio_opt.insert(audio_opt.end(), c.begin(), c.end());
        }

        // Validation mode: dump the float output (before RMS blending)
        if (!opt.noise_dir.empty() && !audio_opt.empty()) {
            std::ofstream f(opt.noise_dir + "/e2e_cpp_float.raw", std::ios::binary);
            f.write(reinterpret_cast<const char*>(audio_opt.data()),
                static_cast<std::streamsize>(audio_opt.size() * sizeof(float)));
        }

        // 6) RMS blending
        if (opt.rms_mix_rate != 1.0f && !audio_opt.empty()) {
            dsp::apply_rms_mix(audio, audio_opt, tgt_sr, opt.rms_mix_rate);
        }

        // 7) Resampling (tgt_sr != resample_sr >= 16000)
        int out_sr = tgt_sr;
        if (opt.resample_sr >= 16000 && opt.resample_sr != tgt_sr && !audio_opt.empty()) {
            audio_opt = resample_mono(audio_opt, tgt_sr, opt.resample_sr);
            out_sr = opt.resample_sr;
        }

        // 8) int16 (aligned with Python: normalize by max/0.99, then truncate)
        ConvertResult result;
        result.sample_rate = out_sr;
        result.pcm.resize(audio_opt.size());
        if (!audio_opt.empty()) {
            float peak = 0.0f;
            for (float v : audio_opt) {
                peak = std::max(peak, std::abs(v));
            }
            float scale = 32768.0f;
            const float audio_max = peak / 0.99f;
            if (audio_max > 1.0f) {
                scale /= audio_max;
            }
            for (size_t i = 0; i < audio_opt.size(); ++i) {
                result.pcm[i] = static_cast<int16_t>(audio_opt[i] * scale);
            }
        }
        return result;
    }
};

RvcPipeline::RvcPipeline(const GgufModel& hubert_model, const GgufModel& rmvpe_model,
    const GgufModel& voice_model, ggml_backend_t backend)
    : impl_(std::make_unique<Impl>(hubert_model, rmvpe_model, voice_model, backend)) {
}

RvcPipeline::~RvcPipeline() = default;

const vits::Config& RvcPipeline::voice_config() const {
    return impl_->synth.config();
}

ConvertResult RvcPipeline::convert(const std::vector<float>& audio_16k, const ConvertOptions& opt) {
    if (audio_16k.empty()) {
        throw std::runtime_error("Input audio is empty");
    }
    return impl_->run(audio_16k, opt);
}

} // namespace rvc
