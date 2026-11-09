// norm.h -- replica of the official norm_write math (M4-1; excludes WAV writing and the pre-16k disk dump)
//
// Semantic baseline: official 81eed5e train/preprocess.py:62-86:
//   tmp_max = np.abs(tmp_audio).max()
//   if not np.isfinite(tmp_max) or tmp_max <= 0 or tmp_max > 2.5: discard
//   tmp_audio = tmp_audio / tmp_max * (0.9*0.75) + 0.25 * tmp_audio
//   wavfile.write(..., tmp_audio.astype(np.float32))     # 0_gt (F32)
//   audio_16k = librosa.resample(tmp_audio, orig_sr=sr, target_sr=16000).astype(np.float32)
//
// C++: scaling stays double all the way -> F32; 16k uses a bit-level resampy kaiser_best emulation (see resample.h; statistical gate G4).

#pragma once

#include <vector>

namespace rvc_train::prep {

struct NormChunk {
    bool valid = false;          // official validity check (finite peak, >0, <=2.5)
    std::vector<float> gt;       // 0_gt samples (F32)
    std::vector<float> audio16k; // 1_16k samples (F32; empty when with_16k=false)
};

// Apply the official normalization math to one chunk; empty chunks are invalid (official raises at np.max)
NormChunk normalize_chunk(const std::vector<double>& chunk, int target_sr, bool with_16k);

} // namespace rvc_train::prep
