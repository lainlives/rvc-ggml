// cache.h -- M4-4 training cache (dataset.json/spec.json + samples/*.bin)
//
// Final format spec: docs/interfaces_and_formats.md or docs/architecture.md §5.2 / docs/handover.md §4.8 C:
//   <exp_dir>/
//     dataset.json     cache metadata + sample manifest (per-sample frame counts / file table)
//     spec.json        field dictionary (dtype/shape/row-major/frame-count formulas)
//     samples/<id>.gt.bin     F32 [gt_samples]        (0_gt, target_sr)
//     samples/<id>.16k.bin    F32 [frames16k]         (16000Hz)
//     samples/<id>.f0hz.bin   F32 [f0_frames]         (after UV interpolation; all-unvoiced samples have no such file)
//     samples/<id>.f0bin.bin  I32 [f0_frames]
//     samples/<id>.feat.bin   F32 [feat_frames,dim]   (row-major = torch [T,C])
//     samples/<id>.spec.bin   F32 [bins,spec_frames]  (row-major = torch [F,T])
//
// Conventions: F32/I32 little-endian; id = `<case>_<official idx1>`; pipeline writes samples/*.bin per valid chunk,
// main_prep finalizes by writing the index; `validate_cache` (`--validate-cache`) self-checks frame counts / file sizes / field consistency (G9).

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "prep/slicer.h"
#include "prep/spec.h"

namespace rvc_train::prep {

struct CacheSampleInfo {
    std::string id;      // <case>_<chunk>
    std::string source;  // WAV file name (empty in injection mode)
    int chunk = 0;       // official idx1
    int64_t gt_samples = 0;
    int64_t frames16k = 0;
    int64_t f0_frames = 0;
    int64_t feat_frames = 0;
    int64_t feat_dim = 0;
    int64_t spec_frames = 0;
    int64_t spec_bins = 0;
    bool has_16k = false;
    bool has_f0 = false;    // f0hz/f0bin written (official skips all-unvoiced samples -> false)
    bool has_feat = false;
    bool has_spec = false;
};

struct CacheMeta {
    int target_sr = 40000;
    std::string version = "v1";
    bool has_f0 = true;  // whether the model has f0 (--has-f0)
    double per = 3.7;
    double overlap = 0.3;
    SlicerParams slicer;
    SpecConfig spec;
    int f0_bin = 256;
    double f0_min = 50.0;
    double f0_max = 1100.0;
};

// "samples/<id>.<kind>.bin" (kind = gt/16k/f0hz/f0bin/feat/spec)
std::string cache_sample_relpath(const std::string& id, const char* kind);

// Write dataset.json + spec.json (samples/*.bin already written by pipeline); cache_dir must exist or be creatable
bool write_cache_index(const std::string& cache_dir, const CacheMeta& meta,
                       const std::vector<CacheSampleInfo>& samples, std::string* err);

// Self-check: format/version/unique ids/file existence/byte sizes/frame-count formula consistency
bool validate_cache(const std::string& cache_dir, std::string* summary, std::string* err);

// ---------------- M5 read-only parsing (for rvc-train; pure file I/O, no rvc_core/ggml dependency) ----------------

// Full dataset.json parse result; sample presence flags inferred from the files table (matches the writing side)
struct CacheDataset {
    std::string dir;
    CacheMeta meta;
    std::vector<CacheSampleInfo> samples;

    const CacheSampleInfo* find(const std::string& id) const;
};

// Parse dataset.json and fill meta/samples (does no file-existence checks; validate_cache handles integrity)
bool load_cache_index(const std::string& cache_dir, CacheDataset* out, std::string* err);

// Read a single sample bin:
//   f32 kinds = gt / 16k / f0hz / feat / spec; i32 kinds = f0bin
// Check presence flag + file existence + element count against CacheSampleInfo fields
bool read_cache_sample_f32(const std::string& cache_dir, const CacheSampleInfo& s, const char* kind,
                           std::vector<float>* out, std::string* err);
bool read_cache_sample_i32(const std::string& cache_dir, const CacheSampleInfo& s, const char* kind,
                           std::vector<int32_t>* out, std::string* err);

} // namespace rvc_train::prep
