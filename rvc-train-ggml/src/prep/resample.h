// resample.h -- M4 preprocessing resampling (two profiles)
//
// 1) resample_sinc: in-house Kaiser-window sinc resampler for the load branch (any src->target_sr).
//    Background (M4-1 measurements): rvc_core's windowed_sinc_resample cutoff formula
//    fc = 0.95*min(src,dst)/max(U,D) is only usable at inference's fixed ratios; for 44.1k->40k it produces
//    severe distortion and overshoot (peak 0.84->3.87, corr 0.92). This project's in-house kernel uses reflect boundaries + per-output normalization,
//    so its semantics differ from official ffmpeg swresample -> G4 is a statistical gate only (corr; int16 sample-exact is informational).
//
// 2) resample_kaiser_best: **bit-level resampy kaiser_best emulation** for the 16k branch (target_sr->16000).
//    Official semantics = librosa.resample(res_type="kaiser_best") (no soxr on this machine); input float64 (scipy
//    lfilter output), output float64, caller then does astype(float32). The implementation strictly replicates
//    resampy core.py + interpn.py (precomputed table + linear phase interpolation, zero-padded boundaries, output length truncation);
//    table params: num_zeros=50, num_table=8192, rolloff=0.91734737126087607,
//    beta=12.984585153568657 (beta is not stored in the npz; back-derived from the npz half_window; max table deviation ~5e-11).
//    The loop semantics were verified bit-for-bit against resampy by the Python prototype (maxdiff=0).

#pragma once

#include <vector>

namespace rvc_train::prep {

// Mono resampling (F32 samples in/out; returned unchanged when src==dst)
std::vector<float> resample_sinc(const std::vector<float>& in, int src_sr, int dst_sr);

// Bit-level resampy kaiser_best emulation (double in/out; returned unchanged when src==dst;
// output length = int(n * dst / src), matching resampy)
std::vector<double> resample_kaiser_best(const std::vector<double>& in, int src_sr, int dst_sr);

} // namespace rvc_train::prep
