// RVC-Train-GGML - differentiable STFT / mel frontend (src/train/stft_mel.h)
//
// Semantic baseline: official 81eed5e `train/mel_processing.py` (training-loss only; slightly
// different mel parameters from the runtime infer_pack -- official training code uses sqrt(re²+im²+2e-7),
// clamp min=2e-6, and spectrogram_torch explicitly reflect-pads (n_fft-hop)/2 before center=False).
//
// Implementation (docs/architecture.md §5.3): reflect pad -> Hann window folded into the DFT kernel -> conv_1d
// (kernel [n_fft,1,2*bins], constant graph tensor, not PARAM) -> sum of squares then sqrt -> mul_mat(mel_basis) -> log/clamp.
// Layout: y=[T,B] -> spec=[frames,bins,B] -> mel=[frames,n_mels,B] (ne order = reverse of torch shape).
#pragma once

#include "traincpp/traincpp.h"

#include <string>

namespace rvc_train {
namespace train {

// STFT/mel config (aligned with the data section of official configs/*.json)
struct StftMelConfig {
    int64_t n_fft  = 2048;
    int64_t hop    = 400;
    int64_t win    = 2048;
    int64_t n_mels = 125;
    double  sr     = 40000.0;
    double  fmin   = 0.0;
    double  fmax   = 0.0;       // <=0 means sr/2 (mel_fmax is null in the official config)
    float   mag_eps  = 2e-7f;   // |X| = sqrt(re²+im²+mag_eps)
    float   log_clip = 2e-6f;   // log(clamp(x, min=log_clip))
    bool    validate(std::string* err) const;
    int64_t bins() const { return n_fft / 2 + 1; }
    int64_t pad() const { return (n_fft - hop) / 2; }
    double  fmax_eff() const { return fmax > 0.0 ? fmax : sr * 0.5; }
    // Frame count after explicit reflect pad (= frames of torch.stft(center=False))
    int64_t frames(int64_t t) const;
};

// In-graph constant leaves (created before allocation; data written by stft_mel_fill_* after buffer allocation)
struct StftMelConsts {
    traincpp::Tensor* dft_kernel = nullptr;  // [n_fft,1,2*bins], channels [0,bins)=cos, [bins,2bins)=-sin
    traincpp::Tensor* mel_basis  = nullptr;  // [bins,n_mels] (reverse of torch [n_mels,bins])
    traincpp::Tensor* mag_eps    = nullptr;  // [1] scalar leaf (the 2e-7 inside sqrt)
};

void stft_mel_consts_init(traincpp::Context* ctx, const StftMelConfig& cfg, StftMelConsts* out);
void stft_mel_fill_dft_kernel(StftMelConsts* c, const StftMelConfig& cfg);
void stft_mel_fill_mel_basis(StftMelConsts* c, const StftMelConfig& cfg);
// Host-side generation of the triangular filterbank in librosa (norm='slaney', htk=False) convention;
// out layout [n_mels][bins] row-major (same layout as the golden mel_basis file)
void make_mel_filterbank_slaney(const StftMelConfig& cfg, float* out);

// Forward (all differentiable): y=[T,B] (raw waveform, no padding)
// When internals is non-null, fills in intermediate tensor pointers (debug use; lifetime same as Context)
struct StftInternals {
    traincpp::Tensor* kernel = nullptr;  // DFT kernel [n_fft,1,2*bins]
    traincpp::Tensor* padded = nullptr;  // after reflect pad [T+2p,1,B]
    traincpp::Tensor* conv   = nullptr;  // DFT conv output [frames,2*bins,B]
    traincpp::Tensor* re     = nullptr;  // real part [frames,bins,B]
    traincpp::Tensor* im     = nullptr;  // imaginary part [frames,bins,B]
    traincpp::Tensor* spec   = nullptr;  // |X| [frames,bins,B]
};
traincpp::Tensor* stft_forward(traincpp::Context* ctx, const StftMelConsts& c,
                               const StftMelConfig& cfg, traincpp::Tensor* y,
                               StftInternals* internals = nullptr);
traincpp::Tensor* mel_spectrogram_forward(traincpp::Context* ctx, const StftMelConsts& c,
                                          const StftMelConfig& cfg, traincpp::Tensor* y);

} // namespace train
} // namespace rvc_train
