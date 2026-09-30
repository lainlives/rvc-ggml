// norm.cpp —— 官方 norm_write 数学复刻

#include "prep/norm.h"

#include <algorithm>
#include <cmath>
#include <cstddef>

#include "prep/load.h"      // to_f32
#include "prep/resample.h"  // resample_kaiser_best（16k 分支 resampy 仿真）

namespace rvc_train::prep {

NormChunk normalize_chunk(const std::vector<double>& chunk, int target_sr, bool with_16k) {
    NormChunk out;
    if (chunk.empty()) {
        return out;  // 官方空块会在 np.max 抛异常 → 视为无效
    }

    double tmp_max = 0.0;
    for (const double v : chunk) {
        tmp_max = std::max(tmp_max, std::fabs(v));
    }
    if (!std::isfinite(tmp_max) || tmp_max <= 0.0 || tmp_max > 2.5) {
        return out;
    }

    const double kMax = 0.9;
    const double kAlpha = 0.75;
    std::vector<double> scaled(chunk.size());
    out.gt.resize(chunk.size());
    for (std::size_t i = 0; i < chunk.size(); ++i) {
        // 官方运算顺序：x / tmp_max * (max*alpha) + (1-alpha) * x
        scaled[i] = chunk[i] / tmp_max * (kMax * kAlpha) + (1.0 - kAlpha) * chunk[i];
        out.gt[i] = static_cast<float>(scaled[i]);
    }
    out.valid = true;

    if (with_16k) {
        // 官方：librosa.resample(scaled, orig_sr, 16000).astype(np.float32)
        // scaled 为 float64（scipy lfilter 输出），故 resampy kaiser_best 仿真用 double 输入/输出
        out.audio16k = to_f32(resample_kaiser_best(scaled, target_sr, 16000));
    }
    return out;
}

} // namespace rvc_train::prep
