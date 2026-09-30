// RVC-Train-GGML - 主机侧 SineGen（src/model/sinegen.h）
//
// 移植自 RVC-GGML `src/rvc/sinegen.cpp:24-95`（旧 `cumsum_shift` 法，Q11 已拍板），
// 与运行时 infer_pack `models.py:353-407`、RVC-GGML `scripts/make_ref.py:160-199` 同源。
// 纯主机实现、不进图；噪声由调用方注入（对拍走 golden 落盘的同一份噪声，训练期由训练 RNG 统一生成）。
#pragma once

#include <cstdint>
#include <vector>

namespace rvc_train {
namespace model {

// 生成 raw sine waves（SineGen 输出，未经 m_source.l_linear/tanh）：
//   f0:    [frames] Hz（<=0 视为清音）
//   noise: [frames*upp] 高斯噪声；为空（nullptr 或长度不足）时用 seed 内部生成
//   seed:  仅 noise 缺省时生效
// 返回 [frames*upp]；参数非法时返回空 vector。
std::vector<float> sinegen_generate(const float* f0, int64_t frames, int sample_rate, int upp,
                                    const float* noise, int64_t noise_len, uint32_t seed = 1234);

} // namespace model
} // namespace rvc_train
