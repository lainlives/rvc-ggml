// f0.h -- M4-2 RMVPE f0 and coarse bins (official 81eed5e extract_f0 semantics)
//
// Semantic baseline: 81eed5e `train/dataset/extract_f0.py::FeatureInput` (lines 91-123):
//   f0 = RMVPE(assets/rmvpe/rmvpe.pt, is_half=False, device="cpu").infer_from_audio(x, thred=0.03)
//   uv = f0==0; f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])   # UV interpolation only
//   all zeros -> np.interp raises -> official skips this sample (this implementation sets valid=false)
//   coarse_f0: 1127*ln(1+f0/700) scaled to [1,255] -> np.rint(half-even) -> int
//
// C++: RMVPE reuses rvc_core `rvc::rmvpe::RmvpeExtractor`; interpolation reuses
// `rvc::dsp::interpolate_unvoiced` (semantics = np.interp: interior linear + endpoint values at both ends);
// bins reuses `rvc::dsp::coarse_pitch_bin` (same function as RVC-GGML inference, ensuring training/inference consistency).

#pragma once

#include <cstdint>
#include <vector>

#include "rmvpe/rmvpe.h"

namespace rvc_train::prep {

struct F0Result {
    bool valid = false;           // voiced frames present (all zeros -> false, matches official skipping this chunk)
    std::vector<float> raw;       // raw RMVPE f0 (Hz, 0=unvoiced; for diagnostics/cross-checks)
    std::vector<float> f0_hz;     // f0 after UV interpolation (empty when valid=false)
    std::vector<int32_t> f0_bin;  // coarse bins 1..255 (empty when valid=false)
};

// Input: normalized 16k chunk (matches official 1_16k_wavs content)
F0Result compute_f0_rmvpe(const std::vector<float>& audio16k, rvc::rmvpe::RmvpeExtractor& rmvpe);

} // namespace rvc_train::prep
