// mel.h
// RMVPE mel 前端：STFT(1024/160, center, Hann) + HTK mel 滤波器组 + log。
// 对齐 librosa.filters.mel(htk=True) 与 torch.stft(center=True, reflect pad)。

#pragma once

#include <vector>
#include <cstdint>

namespace rvc {
namespace audio {

    // 生成 librosa HTK mel 滤波器组 [n_mels, n_fft/2+1]
    std::vector<float> make_mel_filterbank(int sample_rate, int n_fft, int n_mels,
        float fmin, float fmax);

    // 计算 log-mel 频谱：
    //   audio: 单声道 16k
    //   返回 [n_mels, frames]，帧数 = audio_len / hop + 1（center 模式）
    std::vector<float> log_mel_spectrogram(const std::vector<float>& audio,
        const std::vector<float>& mel_basis,
        int sample_rate, int n_fft, int hop, int n_mels);

} // namespace audio
} // namespace rvc
