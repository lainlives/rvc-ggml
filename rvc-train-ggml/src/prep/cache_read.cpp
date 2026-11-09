// cache_read.cpp -- M5-1 training cache read-only parsing (declarations in cache.h)
//
// For rvc-train to read the M4 cache (dataset.json + samples/*.bin); pure file I/O,
// no prep/spec.cpp/pocketfft/rvc_core, only common/json and the cache.h structs.
// Keeps the same format version as the writing side (cache.cpp); integrity checks (file sizes/frame-count
// formulas) still belong to rvc-prep's validate_cache, this file only guarantees correct read semantics (presence/element counts).

#include "prep/cache.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include "common/json.h"

namespace fs = std::filesystem;

namespace rvc_train::prep {
namespace {

constexpr int64_t kCacheFormatVersion = 1;

// Same format as cache.cpp's cache_sample_relpath (avoids rvc-train linking the writing-side cache.cpp)
std::string sample_relpath(const std::string& id, const char* kind) {
    return "samples/" + id + "." + kind + ".bin";
}

int64_t j_int(const common::JsonValue* v, int64_t fallback) {
    return (v != nullptr && v->is_number()) ? v->as_int(fallback) : fallback;
}

double j_num(const common::JsonValue* v, double fallback) {
    return (v != nullptr && v->is_number()) ? v->as_number(fallback) : fallback;
}

bool j_bool(const common::JsonValue* v, bool fallback) {
    return (v != nullptr && v->is_bool()) ? v->as_bool(fallback) : fallback;
}

// Read the whole file into a byte buffer (caller decides on size mismatches)
bool read_file_bytes(const std::string& path, std::vector<uint8_t>* out, std::string* err) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        if (err != nullptr) {
            *err = "cannot open: " + path;
        }
        return false;
    }
    const std::streamoff size = f.tellg();
    if (size < 0) {
        if (err != nullptr) {
            *err = "cannot get size: " + path;
        }
        return false;
    }
    out->resize(static_cast<size_t>(size));
    f.seekg(0);
    if (size > 0) {
        f.read(reinterpret_cast<char*>(out->data()), size);
    }
    if (!f) {
        if (err != nullptr) {
            *err = "read failed: " + path;
        }
        return false;
    }
    return true;
}

bool kind_present(const CacheSampleInfo& s, const std::string& kind, std::string* err) {
    bool present = false;
    if (kind == "gt") {
        present = s.gt_samples > 0;
    } else if (kind == "16k") {
        present = s.has_16k;
    } else if (kind == "f0hz" || kind == "f0bin") {
        present = s.has_f0;
    } else if (kind == "feat") {
        present = s.has_feat;
    } else if (kind == "spec") {
        present = s.has_spec;
    } else if (err != nullptr) {
        *err = "sample " + s.id + ": unknown file kind " + kind;
    }
    if (!present && err != nullptr) {
        *err = "sample " + s.id + ": missing file kind " + kind;
    }
    return present;
}

int64_t expected_elements(const CacheSampleInfo& s, const std::string& kind) {
    if (kind == "gt") {
        return s.gt_samples;
    }
    if (kind == "16k") {
        return s.frames16k;
    }
    if (kind == "f0hz" || kind == "f0bin") {
        return s.f0_frames;
    }
    if (kind == "feat") {
        return s.feat_frames * s.feat_dim;
    }
    if (kind == "spec") {
        return s.spec_frames * s.spec_bins;
    }
    return -1;
}

bool parse_sample(const common::JsonValue& v, std::size_t index, CacheSampleInfo* out,
                  std::string* err) {
    if (!v.is_object()) {
        if (err != nullptr) {
            *err = "samples[" + std::to_string(index) + "] is not an object";
        }
        return false;
    }
    const common::JsonValue* id = v.find("id");
    if (id == nullptr || !id->is_string() || id->as_string().empty()) {
        if (err != nullptr) {
            *err = "samples[" + std::to_string(index) + "] is missing id";
        }
        return false;
    }
    out->id = id->as_string();
    const common::JsonValue* source = v.find("source");
    out->source = (source != nullptr && source->is_string()) ? source->as_string() : std::string();
    out->chunk = static_cast<int>(j_int(v.find("chunk"), 0));
    out->gt_samples = j_int(v.find("gt_samples"), 0);
    out->frames16k = j_int(v.find("frames16k"), 0);
    out->f0_frames = j_int(v.find("f0_frames"), 0);
    out->feat_frames = j_int(v.find("feat_frames"), 0);
    out->feat_dim = j_int(v.find("feat_dim"), 0);
    out->spec_frames = j_int(v.find("spec_frames"), 0);
    out->spec_bins = j_int(v.find("spec_bins"), 0);

    const common::JsonValue* files = v.find("files");
    if (files == nullptr || !files->is_object()) {
        if (err != nullptr) {
            *err = "sample " + out->id + " is missing the files object";
        }
        return false;
    }
    out->has_16k = files->find("16k") != nullptr;
    out->has_f0 = (files->find("f0hz") != nullptr) || (files->find("f0bin") != nullptr);
    out->has_feat = files->find("feat") != nullptr;
    out->has_spec = files->find("spec") != nullptr;
    return true;
}

} // namespace

const CacheSampleInfo* CacheDataset::find(const std::string& id) const {
    for (const CacheSampleInfo& s : samples) {
        if (s.id == id) {
            return &s;
        }
    }
    return nullptr;
}

bool load_cache_index(const std::string& cache_dir, CacheDataset* out, std::string* err) {
    if (out == nullptr) {
        if (err != nullptr) {
            *err = "load_cache_index: out is null";
        }
        return false;
    }
    const fs::path dataset_path = fs::path(cache_dir) / "dataset.json";
    common::JsonValue root;
    if (!common::json_read_file(dataset_path.string(), &root, err)) {
        return false;
    }
    if (!root.is_object()) {
        if (err != nullptr) {
            *err = "dataset.json top level is not an object: " + dataset_path.string();
        }
        return false;
    }
    const common::JsonValue* format = root.find("format");
    const common::JsonValue* version = root.find("format_version");
    if (format == nullptr || !format->is_string() || format->as_string() != "rvc-train-cache" ||
        version == nullptr || !version->is_number() || version->as_int(-1) != kCacheFormatVersion) {
        if (err != nullptr) {
            *err = "dataset.json format/format_version mismatch (expected rvc-train-cache v" +
                   std::to_string(kCacheFormatVersion) + ")";
        }
        return false;
    }

    CacheDataset ds;
    ds.dir = cache_dir;
    ds.meta.target_sr = static_cast<int>(j_int(root.find("target_sr"), 40000));
    const common::JsonValue* mv = root.find("model_version");
    ds.meta.version = (mv != nullptr && mv->is_string()) ? mv->as_string() : std::string("v1");
    ds.meta.has_f0 = j_bool(root.find("has_f0"), true);
    ds.meta.per = j_num(root.find("per"), 3.7);
    ds.meta.overlap = j_num(root.find("overlap"), 0.3);

    if (const common::JsonValue* slicer = root.find("slicer"); slicer != nullptr && slicer->is_object()) {
        ds.meta.slicer.threshold_db = j_num(slicer->find("threshold_db"), -42.0);
        ds.meta.slicer.min_length_ms = j_num(slicer->find("min_length_ms"), 1500.0);
        ds.meta.slicer.min_interval_ms = j_num(slicer->find("min_interval_ms"), 400.0);
        ds.meta.slicer.hop_size_ms = j_num(slicer->find("hop_size_ms"), 15.0);
        ds.meta.slicer.max_sil_kept_ms = j_num(slicer->find("max_sil_kept_ms"), 500.0);
    }
    if (const common::JsonValue* f0 = root.find("f0"); f0 != nullptr && f0->is_object()) {
        ds.meta.f0_bin = static_cast<int>(j_int(f0->find("bin"), 256));
        ds.meta.f0_min = j_num(f0->find("min"), 50.0);
        ds.meta.f0_max = j_num(f0->find("max"), 1100.0);
    }
    if (const common::JsonValue* spec = root.find("spec"); spec != nullptr && spec->is_object()) {
        ds.meta.spec.n_fft = static_cast<int>(j_int(spec->find("n_fft"), 2048));
        ds.meta.spec.hop = static_cast<int>(j_int(spec->find("hop"), 400));
        ds.meta.spec.win = static_cast<int>(j_int(spec->find("win"), 2048));
    }
    ds.meta.spec.sample_rate = ds.meta.target_sr;

    const common::JsonValue* samples = root.find("samples");
    if (samples == nullptr || !samples->is_array()) {
        if (err != nullptr) {
            *err = "dataset.json is missing the samples array";
        }
        return false;
    }
    ds.samples.reserve(samples->size());
    for (std::size_t i = 0; i < samples->size(); ++i) {
        CacheSampleInfo info;
        if (!parse_sample(samples->at(i), i, &info, err)) {
            return false;
        }
        if (ds.find(info.id) != nullptr) {
            if (err != nullptr) {
                *err = "duplicate sample id: " + info.id;
            }
            return false;
        }
        ds.samples.push_back(std::move(info));
    }
    *out = std::move(ds);
    if (err != nullptr) {
        err->clear();
    }
    return true;
}

bool read_cache_sample_f32(const std::string& cache_dir, const CacheSampleInfo& s, const char* kind,
                           std::vector<float>* out, std::string* err) {
    if (out == nullptr || kind == nullptr) {
        if (err != nullptr) {
            *err = "read_cache_sample_f32: null argument";
        }
        return false;
    }
    const std::string k = kind;
    if (!kind_present(s, k, err)) {
        return false;
    }
    const int64_t count = expected_elements(s, k);
    if (count < 0) {
        if (err != nullptr) {
            *err = "sample " + s.id + ": file kind " + k + " is not F32";
        }
        return false;
    }
    const std::string path = (fs::path(cache_dir) / sample_relpath(s.id, kind)).string();
    std::vector<uint8_t> bytes;
    if (!read_file_bytes(path, &bytes, err)) {
        return false;
    }
    const int64_t want = count * 4;
    if (static_cast<int64_t>(bytes.size()) != want) {
        if (err != nullptr) {
            *err = "file " + path + " size " + std::to_string(bytes.size()) + " != expected " +
                   std::to_string(want) + " bytes";
        }
        return false;
    }
    out->resize(static_cast<size_t>(count));
    if (count > 0) {
        std::memcpy(out->data(), bytes.data(), static_cast<size_t>(want));
    }
    return true;
}

bool read_cache_sample_i32(const std::string& cache_dir, const CacheSampleInfo& s, const char* kind,
                           std::vector<int32_t>* out, std::string* err) {
    if (out == nullptr || kind == nullptr) {
        if (err != nullptr) {
            *err = "read_cache_sample_i32: null argument";
        }
        return false;
    }
    const std::string k = kind;
    if (k != "f0bin") {
        if (err != nullptr) {
            *err = "read_cache_sample_i32: only f0bin is supported (got " + k + ")";
        }
        return false;
    }
    if (!kind_present(s, k, err)) {
        return false;
    }
    const int64_t count = expected_elements(s, k);
    const std::string path = (fs::path(cache_dir) / sample_relpath(s.id, kind)).string();
    std::vector<uint8_t> bytes;
    if (!read_file_bytes(path, &bytes, err)) {
        return false;
    }
    const int64_t want = count * 4;
    if (static_cast<int64_t>(bytes.size()) != want) {
        if (err != nullptr) {
            *err = "file " + path + " size " + std::to_string(bytes.size()) + " != expected " +
                   std::to_string(want) + " bytes";
        }
        return false;
    }
    out->resize(static_cast<size_t>(count));
    if (count > 0) {
        std::memcpy(out->data(), bytes.data(), static_cast<size_t>(want));
    }
    return true;
}

} // namespace rvc_train::prep
