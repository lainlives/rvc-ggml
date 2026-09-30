// norm.h —— 官方 norm_write 数学的复刻（M4-1；不含 WAV 写盘与 16k 之前的落盘）
//
// 语义基准：官方 81eed5e train/preprocess.py:62-86：
//   tmp_max = np.abs(tmp_audio).max()
//   if not np.isfinite(tmp_max) or tmp_max <= 0 or tmp_max > 2.5: 丢弃
//   tmp_audio = tmp_audio / tmp_max * (0.9*0.75) + 0.25 * tmp_audio
//   wavfile.write(..., tmp_audio.astype(np.float32))     # 0_gt（F32）
//   audio_16k = librosa.resample(tmp_audio, orig_sr=sr, target_sr=16000).astype(np.float32)
//
// C++：缩放全程 double → F32；16k 用 resampy kaiser_best 位级仿真（见 resample.h；统计门 G4）。

#pragma once

#include <vector>

namespace rvc_train::prep {

struct NormChunk {
    bool valid = false;          // 官方有效性判定（峰值有限、>0、≤2.5）
    std::vector<float> gt;       // 0_gt 样本（F32）
    std::vector<float> audio16k; // 1_16k 样本（F32；with_16k=false 时为空）
};

// 对单个分块执行官方归一化数学；空块视为无效（官方会在 np.max 处抛异常）
NormChunk normalize_chunk(const std::vector<double>& chunk, int target_sr, bool with_16k);

} // namespace rvc_train::prep
