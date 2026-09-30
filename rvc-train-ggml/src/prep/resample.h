// resample.h —— M4 预处理重采样（两套 profile）
//
// 1) resample_sinc：载入分支（任意 src→target_sr）的自研 Kaiser 窗 sinc 重采样。
//    背景（M4-1 实测）：rvc_core 的 windowed_sinc_resample 截止频率公式
//    fc = 0.95*min(src,dst)/max(U,D) 仅在推理固定比例下“可用”，对 44.1k→40k 会产生
//    严重失真与过冲（peak 0.84→3.87、corr 0.92）。本项目自研核用反射边界 + 逐输出归一化，
//    与官方 ffmpeg swresample 语义不同 → G4 只做统计门（corr；int16 逐样本为信息性）。
//
// 2) resample_kaiser_best：16k 分支（target_sr→16000）的 **resampy kaiser_best 位级仿真**。
//    官方语义 = librosa.resample(res_type="kaiser_best")（本机无 soxr）；输入 float64（scipy
//    lfilter 输出），输出 float64，调用方再 astype(float32)。实现严格复刻
//    resampy core.py + interpn.py（预计算表 + 相位线性插值、边界零填充、输出长度截断）；
//    表参数：num_zeros=50、num_table=8192、rolloff=0.91734737126087607、
//    beta=12.984585153568657（npz 未存 beta，按 npz half_window 反推；表最大偏差 ~5e-11）。
//    循环语义已由 Python 原型对 resampy 逐位验证（maxdiff=0）。

#pragma once

#include <vector>

namespace rvc_train::prep {

// 单声道重采样（输入/输出均为 F32 样本；src==dst 时原样返回）
std::vector<float> resample_sinc(const std::vector<float>& in, int src_sr, int dst_sr);

// resampy kaiser_best 位级仿真（输入/输出 double；src==dst 原样返回；
// 输出长度 = int(n * dst / src)，与 resampy 一致）
std::vector<double> resample_kaiser_best(const std::vector<double>& in, int src_sr, int dst_sr);

} // namespace rvc_train::prep
