// audio_io.h
// WAV 音频文件读写封装，基于 dr_wav.h
// 依赖：dr_wav.h（include 目录下）

#pragma once

#include <vector>
#include <string>
#include <cstdint>

struct AudioData {
    std::vector<float> samples;    // 交错存储，长度 = num_channels * total_samples
    int sample_rate = 0;
    int num_channels = 0;
    uint64_t total_samples = 0;    // 每个通道的样本数

    size_t total_frames() const { return static_cast<size_t>(total_samples); }
    size_t size_bytes() const { return samples.size() * sizeof(float); }
};

// 从 WAV 文件加载音频（int16/int24/int32/float32 均支持，自动转 float），失败抛出 std::runtime_error
AudioData load_wav(const std::string& filepath);

// 将音频保存为 WAV 文件（16-bit PCM），返回成功与否
bool save_wav(const std::string& filepath,
    const float* samples,
    int num_channels,
    int sample_rate,
    uint64_t total_samples);

// 将 int16 PCM 原样保存为 WAV（用于管线输出，避免二次量化）
bool save_wav_i16(const std::string& filepath,
    const int16_t* samples,
    int num_channels,
    int sample_rate,
    uint64_t total_samples);

// 线性插值重采样（与输入声道数一致）
AudioData resample(const AudioData& audio, int target_sr);

// 单声道 16k 转换：下混 + 多相 FIR 重采样（质量与 torchaudio kaiser_best 同量级）
std::vector<float> to_mono_16k(const AudioData& audio, int target_sr);

// 单声道重采样（任意采样率，多相 FIR）
std::vector<float> resample_mono(const std::vector<float>& samples, int src_sr, int dst_sr);
