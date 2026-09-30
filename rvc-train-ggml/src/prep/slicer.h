// slicer.h —— 官方 Slicer2 的位语义复刻（M4-1）
//
// 语义基准：官方 81eed5e train/dataset/slicer2.py（整文件逐行复刻）与 train/preprocess.py:40-47：
//   Slicer(sr=sr, threshold=-42, min_length=1500, min_interval=400, hop_size=15, max_sil_kept=500)
// 复刻要点（不得改动）：
//   - Python round() = banker's rounding（half-even）→ 用 std::nearbyint（FE_TONEAREST）；
//   - Slicer.min_length/min_interval/max_sil_kept 内部是“帧数”；
//   - `samples.shape[0] <= min_length` 的怪比较（样本数 vs 帧数）按原样复刻；
//   - get_rms：frame_length=win_size、hop=hop_size、两侧各 frame_length//2 补零、平方均值开方；
//   - argmin 取首次出现的最小值（np.argmin 语义）。

#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace rvc_train::prep {

// 与官方一致的毫秒参数（阈值 dB 单独给）
struct SlicerParams {
    double threshold_db = -42.0;
    double min_length_ms = 1500.0;
    double min_interval_ms = 400.0;
    double hop_size_ms = 15.0;
    double max_sil_kept_ms = 500.0;
};

// 构造函数派生的帧数参数（供对拍/文档校准）
struct SlicerDerived {
    double threshold = 0.0;  // 10^(threshold_db/20)
    int hop_size = 0;        // 帧移（样本数）
    int win_size = 0;        // RMS 窗长（样本数）
    int min_length = 0;      // 帧数
    int min_interval = 0;    // 帧数
    int max_sil_kept = 0;    // 帧数
};

// 切片范围（样本域 [begin_sample, end_sample)）
struct SliceRange {
    int64_t begin_sample = 0;
    int64_t end_sample = 0;
};

// Python round() 的 half-even 语义
double round_half_even(double v);

// librosa 风格帧 RMS（frame_length/hop_length 为样本数；两侧零填充 frame_length/2）
std::vector<double> get_rms(const std::vector<double>& samples, int frame_length, int hop_length);

class Slicer {
public:
    Slicer(int sample_rate, const SlicerParams& params);
    const SlicerDerived& derived() const { return derived_; }
    int sample_rate() const { return sample_rate_; }

    // 输入：单声道 double 波形；输出：切片（样本域），顺序与官方 slice() 返回的 chunks 一致
    std::vector<SliceRange> slice(const std::vector<double>& samples) const;

    // 调试：最近一次 slice() 得到的 sil_tags（帧域；对拍/排障用）
    const std::vector<std::pair<int64_t, int64_t>>& last_sil_tags() const { return last_sil_tags_; }
    const std::vector<double>& last_rms() const { return last_rms_; }
    const std::string& last_debug() const { return last_debug_; }

private:
    int sample_rate_ = 0;
    SlicerDerived derived_;
    mutable std::vector<std::pair<int64_t, int64_t>> last_sil_tags_;
    mutable std::vector<double> last_rms_;
    mutable std::string last_debug_;
};

} // namespace rvc_train::prep
