// pipeline.h -- M4 preprocessing core pipeline (rvc-prep host side)
//
// Data flow (replica of official 81eed5e train/preprocess.py + extract_f0.py):
//   WAV -> mono+resample -> [double] causal lfilter(48Hz) -> Slicer2 slicing ->
//   3.7s/3.4s chunking -> peak normalization (0_gt) -> 16k resample ->
//   [M4-2] RMVPE f0 + UV interpolation + coarse bins; [M4-3] HuBERT features (linear spectrogram belongs to M4-4)
//
// Two quirks of the official chunking loop must be replicated as-is (see docs/handover.md §4.8 A):
//   1) idx1 is incremented in the else branch but that tail chunk is not written there; after the for loop, the last chunk's tail is written once more with the then-current idx1;
//   2) i restarts at 0 for each slice, but idx1 stays continuous across slices.
// Result: with multiple slices, intermediate slices' tail chunks are skipped and indices are non-contiguous (matching official behavior).

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "prep/cache.h"
#include "prep/load.h"
#include "prep/slicer.h"
#include "prep/spec.h"

namespace rvc::rmvpe {
class RmvpeExtractor;
} // namespace rvc::rmvpe

namespace rvc_train::prep {

// prep/feat.h: HuBERT feature extractor (self-built executor, safe for repeated computes; see Q22)
class HubertFeatureExtractor;

struct PrepOptions {
    int target_sr = 40000;
    std::string version = "v1";
    bool has_f0 = true;
    double per = 3.7;
    double overlap = 0.3;
    SlicerParams slicer;
    bool no_16k = false;          // skip 16k resampling (do not enable for G4 cross-checks)
    std::string debug_dump_dir;   // if non-empty, write load/hp/gt/16k/f0/slices/chunks/coeffs to disk

    // M4-2: f0 (computed for every valid chunk when non-null and with_f0; caller owns the extractor lifetime)
    bool with_f0 = true;
    rvc::rmvpe::RmvpeExtractor* rmvpe = nullptr;

    // M4-3: HuBERT features (computed for every valid chunk when non-null and with_feat; dims decided by version)
    bool with_feat = true;
    HubertFeatureExtractor* hubert = nullptr;

    // M4-4: linear spectrogram (computed from each valid chunk's 0_gt; config derived by the caller from sr/version)
    bool with_spec = true;
    SpecConfig spec;

    // M4-4: training cache (if non-empty, write samples/<id>.{gt,16k,f0hz,f0bin,feat,spec}.bin for every valid chunk;
    // dataset.json/spec.json are written by the caller after processing all files)
    std::string cache_dir;
};

struct ChunkInfo {
    int index = 0;             // official idx1
    int slice_index = 0;
    int64_t start_in_slice = 0;  // samples
    int64_t n_samples = 0;
    bool valid = false;
    bool f0_valid = false;     // M4-2: voiced frames present (official skips all-zero chunks)
    int64_t f0_frames = 0;     // M4-2: f0 frame count (= 1 + samples16k/160)
    bool feat_valid = false;   // M4-3: feature frames present
    int64_t feat_frames = 0;   // M4-3: HuBERT frame count (= (samples16k-400)/320 + 1)
    int64_t feat_dim = 0;      // M4-3: 256(v1)/768(v2)
    bool spec_valid = false;   // M4-4: spectrogram frames present
    int64_t spec_frames = 0;   // M4-4: linear spectrogram frame count (= gt_samples//hop, v1-40k)
    int64_t spec_bins = 0;     // M4-4: n_fft/2+1 = 1025 (v1/v2)
};

struct PrepFileResult {
    std::string case_name;
    int64_t load_samples = 0;
    std::vector<SliceRange> slices;
    std::vector<ChunkInfo> chunks;
    int64_t total_gt_samples = 0;
    int64_t total_16k_samples = 0;
    int64_t total_f0_frames = 0;    // M4-2
    int64_t total_feat_frames = 0;  // M4-3
    int64_t total_spec_frames = 0;  // M4-4
    std::vector<CacheSampleInfo> samples;  // M4-4: cache manifest (valid chunks only; filled when cache_dir is non-empty)
};

// Normal mode: WAV -> downmix/resample -> core (the load dump is also written in this mode)
bool run_prep_wav(const std::string& wav_path, const std::string& case_name,
                  const PrepOptions& opt, PrepFileResult* out, std::string* err);

// Injection mode: skip decode/resample (golden post-load waveform injection; isolates exact consistency of slicing/hp/norm)
bool run_prep_injected(const MonoAudio& load, const std::string& case_name,
                       const PrepOptions& opt, PrepFileResult* out, std::string* err);

} // namespace rvc_train::prep
