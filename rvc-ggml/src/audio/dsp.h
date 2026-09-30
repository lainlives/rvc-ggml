// dsp.h
// RVC 音频信号处理：高通滤波、RMS 混合、静音切块、反射 pad、F0 粗糙量化。
// 参数与 Python 参考实现（scipy/librosa/numpy）对齐。

#pragma once

#include <vector>
#include <cstdint>

namespace rvc {
namespace dsp {

    // 5 阶 Butterworth 48Hz 高通（等效 scipy.signal.filtfilt，16k 采样率）
    // 输入输出为单声道 16k 浮点序列；长度过短时抛出
    void high_pass_48hz_in_place(std::vector<float>& samples);

    // 反射 pad（numpy.pad mode='reflect'）：左右各 left/right 个样本
    std::vector<float> reflect_pad(const std::vector<float>& samples, int64_t left, int64_t right);

    // 帧 RMS（librosa.feature.rms 对齐：frame_length = sr/2*2, hop = sr/2, 中心对齐, 边界反射）
    std::vector<float> frame_rms(const std::vector<float>& audio, int sample_rate);

    // 将 RMS 曲线线性插值到 output_frames 长度（对齐 torch F.interpolate mode='linear'）
    std::vector<float> interpolate_linear(const std::vector<float>& values, int64_t output_frames);

    // RVC RMS 混合：converted *= rms_src^(1-rate) * rms_out^(rate-1)
    // source_16k: 原始 16k 单声道；converted: 目标采样率输出（原地修改）
    void apply_rms_mix(const std::vector<float>& source_16k,
        std::vector<float>& converted,
        int target_sample_rate,
        float rms_mix_rate);

    // audio_sum 曲线（静音切块内部使用；导出供验证）
    std::vector<float> audio_sum_curve(const std::vector<float>& audio);

    // 长音频静音点切块（对齐 Python Pipeline.pipeline）
    // 返回切点样本位置（16k 域），无切点返回空
    std::vector<int64_t> quiet_split_points(const std::vector<float>& audio,
        int split_query_sec,
        int split_center_sec,
        int split_threshold_sec);

    // F0(Hz) -> RVC coarse pitch 量化（1~255）
    int coarse_pitch_bin(float f0_hz);

    // 中值滤波（半径 radius）
    void median_filter_f0(std::vector<float>& f0, int radius);

    // 未发声段线性插值（f0==0 处用相邻非零值插值）
    void interpolate_unvoiced(std::vector<float>& f0);

} // namespace dsp
} // namespace rvc
