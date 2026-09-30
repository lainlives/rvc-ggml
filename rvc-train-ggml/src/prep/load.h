// load.h —— M4-1 音频载入与重采样（rvc-prep 侧）
//
// 语义基准（官方 81eed5e train/preprocess.py:90 → infer/audio.py:310-335，RVC_AUDIO_FORCE_CPU=1）：
//   ffmpeg -f f32le -ac 1 -ar <target_sr> 解码 + 下混 + 重采样。
// C++ 侧：dr_wav 读 WAV（仅 WAV；非 WAV 请用户先转）→ 下混（立体声 1/√2）→ 自研 Kaiser 窗 sinc
// 重采样（与 ffmpeg swresample 不可逐样本一致 → G4 统计门；见 docs/交接说明.md §4.8 D/G）。
//
// 精度链：本文件输出 double（与官方 lfilter 的 float64 运算衔接）。

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace rvc_train::prep {

// 单声道音频（double 样本）
struct MonoAudio {
    std::vector<double> samples;
    int sample_rate = 0;
};

// WAV → 单声道（各声道平均）→ 重采样到 target_sr；失败返回 false 并填 err
bool load_wav_mono(const std::string& path, int target_sr, MonoAudio* out, std::string* err);

// 原始 F32 小端单声道文件（注入模式：golden 的 post-load 波形）
bool load_raw_f32_mono(const std::string& path, int sample_rate, MonoAudio* out, std::string* err);

// F32 原始 dump（小端、连续）
bool write_raw_f32(const std::string& path, const float* data, std::size_t count, std::string* err);
bool write_raw_f32(const std::string& path, const std::vector<float>& data, std::string* err);

// I32 原始 dump（小端、连续；M4-2 f0bin）
bool write_raw_i32(const std::string& path, const int32_t* data, std::size_t count, std::string* err);
bool write_raw_i32(const std::string& path, const std::vector<int32_t>& data, std::string* err);

// double → float（截断为 F32，四舍五入到最近）
std::vector<float> to_f32(const std::vector<double>& v);

// 重采样（double 包装自研 sinc 重采样；仅 src != dst 时调用）
std::vector<double> resample_mono_double(const std::vector<double>& in, int src_sr, int dst_sr);

} // namespace rvc_train::prep
