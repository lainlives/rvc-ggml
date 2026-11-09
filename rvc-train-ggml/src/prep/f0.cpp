// f0.cpp -- M4-2 RMVPE f0 and coarse bins implementation

#include "prep/f0.h"

#include <algorithm>

#include "audio/dsp.h"  // rvc::dsp::interpolate_unvoiced / coarse_pitch_bin

namespace rvc_train::prep {

F0Result compute_f0_rmvpe(const std::vector<float>& audio16k, rvc::rmvpe::RmvpeExtractor& rmvpe) {
    F0Result out;
    if (audio16k.empty()) {
        return out;
    }

    // official thred=0.03 (FeatureInput.compute_f0)
    const rvc::rmvpe::Result rmv = rmvpe.infer(audio16k, 0.03f);
    if (rmv.f0.empty()) {
        return out;
    }
    out.raw = rmv.f0;

    // official: an empty xp (all zeros) makes np.interp raise -> go() skips this sample
    const bool any_voiced =
        std::any_of(out.raw.begin(), out.raw.end(), [](float v) { return v > 0.0f; });
    if (!any_voiced) {
        return out;
    }

    out.f0_hz = out.raw;
    rvc::dsp::interpolate_unvoiced(out.f0_hz);  // semantics = np.interp (interior linear + endpoint values at both ends)
    out.f0_bin.resize(out.f0_hz.size());
    for (std::size_t i = 0; i < out.f0_hz.size(); ++i) {
        out.f0_bin[i] = rvc::dsp::coarse_pitch_bin(out.f0_hz[i]);
    }
    out.valid = true;
    return out;
}

} // namespace rvc_train::prep
