// RVC-Train-GGML - differentiable STFT / mel frontend implementation (src/train/stft_mel.cpp)
#include "train/stft_mel.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace rvc_train {
namespace train {

namespace {

constexpr double kPi = 3.14159265358979323846;

bool fail(std::string* err, const char* msg) {
    if (err != nullptr) {
        *err = msg;
    }
    return false;
}

// ------------------------------------------------ librosa slaney mel scale (norm='slaney')

double hz_to_mel_slaney(double f) {
    const double f_min = 0.0;
    const double f_sp  = 200.0 / 3.0;
    double mels = (f - f_min) / f_sp;
    const double min_log_hz  = 1000.0;
    const double min_log_mel = (min_log_hz - f_min) / f_sp;
    const double logstep     = std::log(6.4) / 27.0;
    if (f >= min_log_hz) {
        mels = min_log_mel + std::log(f / min_log_hz) / logstep;
    }
    return mels;
}

double mel_to_hz_slaney(double m) {
    const double f_min = 0.0;
    const double f_sp  = 200.0 / 3.0;
    double freqs = f_min + f_sp * m;
    const double min_log_hz  = 1000.0;
    const double min_log_mel = (min_log_hz - f_min) / f_sp;
    const double logstep     = std::log(6.4) / 27.0;
    if (m >= min_log_mel) {
        freqs = min_log_hz * std::exp(logstep * (m - min_log_mel));
    }
    return freqs;
}

} // namespace

bool StftMelConfig::validate(std::string* err) const {
    if (n_fft <= 0 || hop <= 0 || win <= 0) {
        return fail(err, "n_fft/hop/win must be positive");
    }
    if (n_fft % 2 != 0) {
        return fail(err, "n_fft must be even (onesided STFT)");
    }
    if (win != n_fft) {
        return fail(err, "currently only win == n_fft is supported (true for all official configs)");
    }
    if (n_mels <= 0) {
        return fail(err, "n_mels must be positive");
    }
    if (sr <= 0.0) {
        return fail(err, "sample rate must be positive");
    }
    if (!(fmin >= 0.0) || !(fmax_eff() > fmin)) {
        return fail(err, "invalid mel frequency range (need 0 <= fmin < fmax)");
    }
    return true;
}

int64_t StftMelConfig::frames(int64_t t) const {
    const int64_t len = t + 2 * pad();
    if (len < n_fft) {
        return 0;
    }
    return (len - n_fft) / hop + 1;
}

void stft_mel_consts_init(traincpp::Context* ctx, const StftMelConfig& cfg, StftMelConsts* out) {
    if (ctx == nullptr || out == nullptr) {
        return;
    }
    const int64_t bins = cfg.bins();
    {
        const int64_t ne[3] = { cfg.n_fft, 1, 2 * bins };
        out->dft_kernel = traincpp::new_tensor_nd(ctx, traincpp::TYPE_F32, 3, ne);
        if (out->dft_kernel != nullptr) {
            std::snprintf(out->dft_kernel->name, traincpp::MAX_NAME, "stft_dft_kernel");
        }
    }
    {
        const int64_t ne[2] = { bins, cfg.n_mels };
        out->mel_basis = traincpp::new_tensor_nd(ctx, traincpp::TYPE_F32, 2, ne);
        if (out->mel_basis != nullptr) {
            std::snprintf(out->mel_basis->name, traincpp::MAX_NAME, "mel_basis");
        }
    }
    out->mag_eps = traincpp::new_tensor_1d(ctx, traincpp::TYPE_F32, 1);
    if (out->mag_eps != nullptr) {
        std::snprintf(out->mag_eps->name, traincpp::MAX_NAME, "stft_mag_eps");
    }
}

void stft_mel_fill_dft_kernel(StftMelConsts* c, const StftMelConfig& cfg) {
    if (c == nullptr || c->dft_kernel == nullptr) {
        return;
    }
    const int64_t N    = cfg.n_fft;
    const int64_t bins = cfg.bins();
    std::vector<float> k((size_t) (N * 2 * bins));
    // periodic Hann (torch.hann_window(win) defaults to periodic=True): 0.5 - 0.5*cos(2πn/N)
    for (int64_t n = 0; n < N; ++n) {
        const double w = 0.5 - 0.5 * std::cos(2.0 * kPi * (double) n / (double) N);
        for (int64_t b = 0; b < bins; ++b) {
            const double theta = 2.0 * kPi * (double) b * (double) n / (double) N;
            k[(size_t) (n + b * N)]               = (float) (w * std::cos(theta));
            k[(size_t) (n + (bins + b) * N)]      = (float) (-w * std::sin(theta));
        }
    }
    traincpp::tensor_set(c->dft_kernel, k.data(), 0, k.size() * sizeof(float));
    if (c->mag_eps != nullptr) {
        const float eps = cfg.mag_eps;
        traincpp::tensor_set(c->mag_eps, &eps, 0, sizeof(eps));
    }
}

void make_mel_filterbank_slaney(const StftMelConfig& cfg, float* out) {
    if (out == nullptr) {
        return;
    }
    const int64_t n_mels = cfg.n_mels;
    const int64_t bins   = cfg.bins();
    const double  fmin   = cfg.fmin;
    const double  fmax   = cfg.fmax_eff();

    std::vector<double> f_pts((size_t) (n_mels + 2));
    const double min_mel = hz_to_mel_slaney(fmin);
    const double max_mel = hz_to_mel_slaney(fmax);
    for (int64_t i = 0; i < n_mels + 2; ++i) {
        const double m = min_mel + (max_mel - min_mel) * (double) i / (double) (n_mels + 1);
        f_pts[(size_t) i] = mel_to_hz_slaney(m);
    }
    for (int64_t m = 0; m < n_mels; ++m) {
        const double lower  = f_pts[(size_t) m];
        const double center = f_pts[(size_t) (m + 1)];
        const double upper  = f_pts[(size_t) (m + 2)];
        const double enorm  = 2.0 / (upper - lower);
        for (int64_t b = 0; b < bins; ++b) {
            const double freq = cfg.sr * (double) b / (double) cfg.n_fft;
            double w = 0.0;
            if (freq >= lower && freq <= center) {
                w = (freq - lower) / (center - lower);
            } else if (freq > center && freq <= upper) {
                w = (upper - freq) / (upper - center);
            }
            out[(size_t) (m * bins + b)] = (float) (w * enorm);
        }
    }
}

void stft_mel_fill_mel_basis(StftMelConsts* c, const StftMelConfig& cfg) {
    if (c == nullptr || c->mel_basis == nullptr) {
        return;
    }
    std::vector<float> basis((size_t) (cfg.n_mels * cfg.bins()));
    make_mel_filterbank_slaney(cfg, basis.data());
    traincpp::tensor_set(c->mel_basis, basis.data(), 0, basis.size() * sizeof(float));
}

traincpp::Tensor* stft_forward(traincpp::Context* ctx, const StftMelConsts& c,
                               const StftMelConfig& cfg, traincpp::Tensor* y,
                               StftInternals* internals) {
    using namespace traincpp;
    const int64_t pad = cfg.pad();
    if (internals != nullptr) {
        internals->kernel = c.dft_kernel;
    }
    Tensor* p = y;
    if (pad > 0) {
        p = pad_ext(ctx, y, (int) pad, (int) pad, 0, 0, 0, 0, 0, 0, PAD_REFLECT);
        if (p == nullptr) {
            return nullptr;
        }
    }
    if (internals != nullptr) {
        internals->padded = p;
    }
    const int64_t Tp = p->ne[0];
    const int64_t B  = p->ne[1] > 0 ? p->ne[1] : 1;
    Tensor* x3 = reshape_3d(ctx, p, Tp, 1, B);
    if (x3 == nullptr) {
        return nullptr;
    }
    Tensor* conv = conv_1d(ctx, c.dft_kernel, x3, (int) cfg.hop, 0, 1);
    if (conv == nullptr) {
        return nullptr;
    }
    if (internals != nullptr) {
        internals->conv = conv;
    }
    const int64_t frames = conv->ne[0];
    const int64_t bins   = cfg.bins();
    // Channels [0,bins)=cos branch, [bins,2bins)=-sin branch; cont after splitting with views
    Tensor* re = cont(ctx, view_3d(ctx, conv, frames, bins, B, 0));
    Tensor* im = cont(ctx, view_3d(ctx, conv, frames, bins, B, (size_t) bins * conv->nb[1]));
    if (re == nullptr || im == nullptr) {
        return nullptr;
    }
    if (internals != nullptr) {
        internals->re = re;
        internals->im = im;
    }
    Tensor* mag2 = add(ctx, sqr(ctx, re), sqr(ctx, im));
    if (mag2 == nullptr) {
        return nullptr;
    }
    Tensor* mag = sqrt_op(ctx, add1(ctx, mag2, c.mag_eps));
    if (internals != nullptr) {
        internals->spec = mag;
    }
    return mag;
}

traincpp::Tensor* mel_spectrogram_forward(traincpp::Context* ctx, const StftMelConsts& c,
                                          const StftMelConfig& cfg, traincpp::Tensor* y) {
    using namespace traincpp;
    Tensor* spec = stft_forward(ctx, c, cfg, y);
    if (spec == nullptr) {
        return nullptr;
    }
    // [frames,bins,B] -> [bins,frames,B] (mul_mat's k is in ne0)
    Tensor* spec_t = cont(ctx, transpose(ctx, spec));
    if (spec_t == nullptr) {
        return nullptr;
    }
    // basis [bins,n_mels] (a) × spec_t [bins,frames,B] (b) -> [n_mels,frames,B]
    Tensor* lin_t = mul_mat(ctx, c.mel_basis, spec_t);
    if (lin_t == nullptr) {
        return nullptr;
    }
    Tensor* lin = cont(ctx, transpose(ctx, lin_t));
    if (lin == nullptr) {
        return nullptr;
    }
    return log_op(ctx, clamp(ctx, lin, cfg.log_clip, 3.4e38f));
}

} // namespace train
} // namespace rvc_train
