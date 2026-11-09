// mel.h
// RMVPE mel front end: STFT(1024/160, center, Hann) + HTK mel filterbank + log.
// Aligned with librosa.filters.mel(htk=True) and torch.stft(center=True, reflect pad).

#pragma once

#include <vector>
#include <cstdint>

namespace rvc {
namespace audio {

    // Build the librosa HTK mel filterbank [n_mels, n_fft/2+1]
    std::vector<float> make_mel_filterbank(int sample_rate, int n_fft, int n_mels,
        float fmin, float fmax);

    // Compute the log-mel spectrogram:
    //   audio: mono 16k
    //   returns [n_mels, frames], frame count = audio_len / hop + 1 (center mode)
    std::vector<float> log_mel_spectrogram(const std::vector<float>& audio,
        const std::vector<float>& mel_basis,
        int sample_rate, int n_fft, int hop, int n_mels);

} // namespace audio
} // namespace rvc
