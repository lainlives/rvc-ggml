// cache.cpp —— M4-4 训练缓存索引写入与自校验（格式见 cache.h 头注释）

#include "prep/cache.h"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>
#include <system_error>

#include "common/json.h"
#include "prep/load.h"

namespace fs = std::filesystem;

namespace rvc_train::prep {
namespace {

constexpr int kCacheFormatVersion = 1;

std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 2);
    for (const unsigned char c : s) {
        switch (c) {
        case '"': out += "\\\""; break;
        case '\\': out += "\\\\"; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        default:
            if (c < 0x20) {
                char buf[8];
                std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned>(c));
                out += buf;
            } else {
                out += static_cast<char>(c);
            }
        }
    }
    return out;
}

bool ensure_dir(const std::string& dir, std::string* err) {
    std::error_code ec;
    fs::create_directories(dir, ec);
    if (ec) {
        if (err) {
            *err = "创建目录失败: " + dir + " (" + ec.message() + ")";
        }
        return false;
    }
    return true;
}

bool write_text(const std::string& path, const std::string& text, std::string* err) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) {
        if (err) {
            *err = "无法写入: " + path;
        }
        return false;
    }
    f << text;
    return static_cast<bool>(f);
}

int64_t expected_bytes(const std::string& kind, const CacheSampleInfo& s) {
    if (kind == "gt") {
        return s.gt_samples * 4;
    }
    if (kind == "16k") {
        return s.frames16k * 4;
    }
    if (kind == "f0hz" || kind == "f0bin") {
        return s.f0_frames * 4;
    }
    if (kind == "feat") {
        return s.feat_frames * s.feat_dim * 4;
    }
    if (kind == "spec") {
        return s.spec_frames * s.spec_bins * 4;
    }
    return -1;
}

const char* kFileKinds[] = {"gt", "16k", "f0hz", "f0bin", "feat", "spec"};

std::string sample_files_json(const CacheSampleInfo& s, const std::string& indent) {
    std::ostringstream o;
    bool first = true;
    o << "{";
    for (const char* kind : kFileKinds) {
        const std::string k = kind;
        bool present = false;
        if (k == "gt") {
            present = s.gt_samples > 0;
        } else if (k == "16k") {
            present = s.has_16k;
        } else if (k == "f0hz" || k == "f0bin") {
            present = s.has_f0;
        } else if (k == "feat") {
            present = s.has_feat;
        } else if (k == "spec") {
            present = s.has_spec;
        }
        if (!present) {
            continue;
        }
        if (!first) {
            o << ",";
        }
        first = false;
        o << "\n" << indent << "  \"" << k << "\": \""
          << json_escape(cache_sample_relpath(s.id, kind)) << "\"";
    }
    o << "\n" << indent << "}";
    return o.str();
}

} // namespace

std::string cache_sample_relpath(const std::string& id, const char* kind) {
    return "samples/" + id + "." + kind + ".bin";
}

bool write_cache_index(const std::string& cache_dir, const CacheMeta& meta,
                       const std::vector<CacheSampleInfo>& samples, std::string* err) {
    if (!ensure_dir((fs::path(cache_dir) / "samples").string(), err)) {
        return false;
    }

    // ---------- dataset.json ----------
    std::ostringstream o;
    o << "{\n";
    o << "  \"format\": \"rvc-train-cache\",\n";
    o << "  \"format_version\": " << kCacheFormatVersion << ",\n";
    o << "  \"created_by\": \"rvc-prep M4-4\",\n";
    o << "  \"target_sr\": " << meta.target_sr << ",\n";
    o << "  \"model_version\": \"" << json_escape(meta.version) << "\",\n";
    o << "  \"has_f0\": " << (meta.has_f0 ? "true" : "false") << ",\n";
    o << "  \"per\": " << meta.per << ",\n";
    o << "  \"overlap\": " << meta.overlap << ",\n";
    o << "  \"slicer\": {\"threshold_db\": " << meta.slicer.threshold_db
      << ", \"min_length_ms\": " << meta.slicer.min_length_ms
      << ", \"min_interval_ms\": " << meta.slicer.min_interval_ms
      << ", \"hop_size_ms\": " << meta.slicer.hop_size_ms
      << ", \"max_sil_kept_ms\": " << meta.slicer.max_sil_kept_ms << "},\n";
    o << "  \"f0\": {\"bin\": " << meta.f0_bin << ", \"min\": " << meta.f0_min
      << ", \"max\": " << meta.f0_max << "},\n";
    o << "  \"spec\": {\"n_fft\": " << meta.spec.n_fft << ", \"hop\": " << meta.spec.hop
      << ", \"win\": " << meta.spec.win << ", \"bins\": " << meta.spec.bins() << "},\n";
    o << "  \"samples\": [";
    for (std::size_t i = 0; i < samples.size(); ++i) {
        const CacheSampleInfo& s = samples[i];
        o << (i == 0 ? "\n" : ",\n");
        o << "    {\n";
        o << "      \"id\": \"" << json_escape(s.id) << "\",\n";
        o << "      \"source\": \"" << json_escape(s.source) << "\",\n";
        o << "      \"chunk\": " << s.chunk << ",\n";
        o << "      \"gt_samples\": " << s.gt_samples << ",\n";
        o << "      \"frames16k\": " << s.frames16k << ",\n";
        o << "      \"f0_frames\": " << s.f0_frames << ",\n";
        o << "      \"feat_frames\": " << s.feat_frames << ",\n";
        o << "      \"feat_dim\": " << s.feat_dim << ",\n";
        o << "      \"spec_frames\": " << s.spec_frames << ",\n";
        o << "      \"spec_bins\": " << s.spec_bins << ",\n";
        o << "      \"files\": " << sample_files_json(s, "      ") << "\n";
        o << "    }";
    }
    o << (samples.empty() ? "]\n" : "\n  ]\n");
    o << "}\n";
    if (!write_text((fs::path(cache_dir) / "dataset.json").string(), o.str(), err)) {
        return false;
    }

    // ---------- spec.json（字段字典；供 M5 训练与 verify 工具读取） ----------
    std::ostringstream sp;
    sp << "{\n";
    sp << "  \"format\": \"rvc-train-cache-spec\",\n";
    sp << "  \"format_version\": " << kCacheFormatVersion << ",\n";
    sp << "  \"samples_dir\": \"samples\",\n";
    sp << "  \"byte_order\": \"little-endian\",\n";
    sp << "  \"file_kinds\": {\n";
    sp << "    \"gt\": {\"dtype\": \"f32\", \"shape\": \"[gt_samples]\"},\n";
    sp << "    \"16k\": {\"dtype\": \"f32\", \"shape\": \"[frames16k]\"},\n";
    sp << "    \"f0hz\": {\"dtype\": \"f32\", \"shape\": \"[f0_frames]\", \"note\": \"UV 插值"
          "后连续 f0 Hz\"},\n";
    sp << "    \"f0bin\": {\"dtype\": \"i32\", \"shape\": \"[f0_frames]\", \"note\": \"1..255\"},\n";
    sp << "    \"feat\": {\"dtype\": \"f32\", \"shape\": \"[feat_frames,feat_dim]\", "
          "\"layout\": \"row-major（= torch [T,C]）\"},\n";
    sp << "    \"spec\": {\"dtype\": \"f32\", \"shape\": \"[spec_bins,spec_frames]\", "
          "\"layout\": \"row-major（= torch [F,T]）\"}\n";
    sp << "  },\n";
    sp << "  \"frame_formulas\": {\n";
    sp << "    \"frames16k\": \"audio16k.size()\",\n";
    sp << "    \"f0_frames\": \"frames16k//160+1（RMVPE torch.stft(center=True)）\",\n";
    sp << "    \"feat_frames\": \"(frames16k-400)//320+1（HuBERT conv 前端无 padding）\",\n";
    sp << "    \"spec_frames\": \"gt_samples//hop（reflect pad (n_fft-hop)/2 + center=False）\"\n";
    sp << "  },\n";
    sp << "  \"notes\": \"id=<case>_<官方 idx1>；不兼容 Python .npy/.spec.pt（对拍工具另写读取器）\"\n";
    sp << "}\n";
    if (!write_text((fs::path(cache_dir) / "spec.json").string(), sp.str(), err)) {
        return false;
    }
    return true;
}

bool validate_cache(const std::string& cache_dir, std::string* summary, std::string* err) {
    const fs::path dir(cache_dir);
    const fs::path dataset_path = dir / "dataset.json";
    const fs::path spec_path = dir / "spec.json";

    common::JsonValue root;
    if (!common::json_read_file(dataset_path.string(), &root, err)) {
        return false;
    }
    if (!root.is_object()) {
        if (err) {
            *err = "dataset.json 顶层不是对象: " + dataset_path.string();
        }
        return false;
    }
    const common::JsonValue* format = root.find("format");
    const common::JsonValue* version = root.find("format_version");
    if (format == nullptr || !format->is_string() || format->as_string() != "rvc-train-cache" ||
        version == nullptr || version->as_int() != kCacheFormatVersion) {
        if (err) {
            *err = "dataset.json format/format_version 不符（期望 rvc-train-cache v" +
                   std::to_string(kCacheFormatVersion) + "）";
        }
        return false;
    }
    if (!fs::is_regular_file(spec_path)) {
        if (err) {
            *err = "缺少 spec.json: " + spec_path.string();
        }
        return false;
    }
    common::JsonValue spec_root;
    if (!common::json_read_file(spec_path.string(), &spec_root, err)) {
        return false;
    }

    const common::JsonValue* spec = root.find("spec");
    int64_t n_fft = 2048;
    if (spec != nullptr && spec->is_object()) {
        if (const common::JsonValue* v = spec->find("n_fft")) {
            n_fft = v->as_int(2048);
        }
    }

    const common::JsonValue* samples = root.find("samples");
    if (samples == nullptr || !samples->is_array()) {
        if (err) {
            *err = "dataset.json 缺少 samples 数组";
        }
        return false;
    }

    std::set<std::string> ids;
    int64_t total_bytes = 0;
    int64_t total_samples = 0;
    int64_t total_f0 = 0;
    int64_t total_feat = 0;
    int64_t total_spec = 0;

    for (std::size_t i = 0; i < samples->size(); ++i) {
        const common::JsonValue& s = samples->at(i);
        if (!s.is_object()) {
            if (err) {
                *err = "samples[" + std::to_string(i) + "] 不是对象";
            }
            return false;
        }
        const std::string id = s.find("id") ? s.find("id")->as_string() : std::string();
        if (id.empty()) {
            if (err) {
                *err = "samples[" + std::to_string(i) + "] 缺少 id";
            }
            return false;
        }
        if (!ids.insert(id).second) {
            if (err) {
                *err = "样本 id 重复: " + id;
            }
            return false;
        }
        CacheSampleInfo info;
        info.id = id;
        info.gt_samples = s.find("gt_samples") ? s.find("gt_samples")->as_int() : 0;
        info.frames16k = s.find("frames16k") ? s.find("frames16k")->as_int() : 0;
        info.f0_frames = s.find("f0_frames") ? s.find("f0_frames")->as_int() : 0;
        info.feat_frames = s.find("feat_frames") ? s.find("feat_frames")->as_int() : 0;
        info.feat_dim = s.find("feat_dim") ? s.find("feat_dim")->as_int() : 0;
        info.spec_frames = s.find("spec_frames") ? s.find("spec_frames")->as_int() : 0;
        info.spec_bins = s.find("spec_bins") ? s.find("spec_bins")->as_int() : 0;

        const common::JsonValue* files = s.find("files");
        if (files == nullptr || !files->is_object()) {
            if (err) {
                *err = "样本 " + id + " 缺少 files 对象";
            }
            return false;
        }

        auto fail = [&](const std::string& msg) {
            if (err) {
                *err = "样本 " + id + ": " + msg;
            }
            return false;
        };

        // 帧数公式一致性
        if (info.frames16k <= 0 || info.gt_samples <= 0) {
            return fail("gt_samples/frames16k 非正");
        }
        const int64_t expect_f0 = info.frames16k / 160 + 1;
        const int64_t expect_feat = (info.frames16k - 400) / 320 + 1;
        SpecConfig sc;
        sc.n_fft = static_cast<int>(n_fft);
        const common::JsonValue* hop_v = spec ? spec->find("hop") : nullptr;
        if (hop_v != nullptr) {
            sc.hop = static_cast<int>(hop_v->as_int(400));
        }
        const int64_t spec_frames_expect = sc.frames_for(info.gt_samples);
        if (info.spec_bins > 0 && info.spec_bins != n_fft / 2 + 1) {
            return fail("spec_bins=" + std::to_string(info.spec_bins) + " != n_fft/2+1=" +
                        std::to_string(n_fft / 2 + 1));
        }
        if (info.spec_frames > 0 && info.spec_frames != spec_frames_expect) {
            return fail("spec_frames=" + std::to_string(info.spec_frames) +
                        " != 公式值 " + std::to_string(spec_frames_expect));
        }

        for (std::size_t k = 0; k < files->size(); ++k) {
            const std::string& kind = files->key_at(k);
            const std::string rel = files->value_at(k).as_string();
            const fs::path p = dir / rel;
            if (!fs::is_regular_file(p)) {
                return fail("缺少文件 " + rel);
            }
            if (kind == "f0hz" || kind == "f0bin") {
                if (info.f0_frames != expect_f0) {
                    return fail("f0_frames=" + std::to_string(info.f0_frames) +
                                " != 公式值 " + std::to_string(expect_f0));
                }
            } else if (kind == "feat") {
                if (info.feat_frames != expect_feat) {
                    return fail("feat_frames=" + std::to_string(info.feat_frames) +
                                " != 公式值 " + std::to_string(expect_feat));
                }
            }
            const int64_t want = expected_bytes(kind, info);
            if (want < 0) {
                return fail("未知文件类型 " + kind);
            }
            std::error_code ec;
            const auto got = static_cast<int64_t>(fs::file_size(p, ec));
            if (ec || got != want) {
                return fail("文件 " + rel + " 大小 " + std::to_string(got) + " != " +
                            std::to_string(want) + " 字节");
            }
            total_bytes += got;
        }
        ++total_samples;
        if (files->find("f0hz") != nullptr) {
            ++total_f0;
        }
        if (files->find("feat") != nullptr) {
            ++total_feat;
        }
        if (files->find("spec") != nullptr) {
            ++total_spec;
        }
    }

    if (summary != nullptr) {
        std::ostringstream ss;
        ss << "cache ok: samples=" << total_samples << " f0=" << total_f0
           << " feat=" << total_feat << " spec=" << total_spec
           << " bytes=" << total_bytes << " dir=" << cache_dir;
        *summary = ss.str();
    }
    if (err) {
        err->clear();
    }
    return true;
}

} // namespace rvc_train::prep
