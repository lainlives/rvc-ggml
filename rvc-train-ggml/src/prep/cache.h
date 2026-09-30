// cache.h —— M4-4 训练缓存（dataset.json/spec.json + samples/*.bin）
//
// 格式定稿见 docs/接口与格式.md 或 docs/架构设计.md §5.2 / docs/交接说明.md §4.8 C：
//   <exp_dir>/
//     dataset.json     缓存元信息 + 样本清单（含每样本帧数/文件表）
//     spec.json        字段字典（dtype/形状/行主序/帧数公式）
//     samples/<id>.gt.bin     F32 [gt_samples]        （0_gt，target_sr）
//     samples/<id>.16k.bin    F32 [frames16k]         （16000Hz）
//     samples/<id>.f0hz.bin   F32 [f0_frames]         （UV 插值后；全无声样本无此文件）
//     samples/<id>.f0bin.bin  I32 [f0_frames]
//     samples/<id>.feat.bin   F32 [feat_frames,dim]   （行主序 = torch [T,C]）
//     samples/<id>.spec.bin   F32 [bins,spec_frames]  （行主序 = torch [F,T]）
//
// 约定：F32/I32 小端；id = `<case>_<官方 idx1>`；pipeline 逐 valid chunk 写 samples/*.bin，
// main_prep 收尾写索引；`validate_cache`（`--validate-cache`）做帧数/文件大小/字段一致性自校验（G9）。

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "prep/slicer.h"
#include "prep/spec.h"

namespace rvc_train::prep {

struct CacheSampleInfo {
    std::string id;      // <case>_<chunk>
    std::string source;  // WAV 文件名（注入模式为空）
    int chunk = 0;       // 官方 idx1
    int64_t gt_samples = 0;
    int64_t frames16k = 0;
    int64_t f0_frames = 0;
    int64_t feat_frames = 0;
    int64_t feat_dim = 0;
    int64_t spec_frames = 0;
    int64_t spec_bins = 0;
    bool has_16k = false;
    bool has_f0 = false;    // f0hz/f0bin 已写出（全无声样本官方跳过 → false）
    bool has_feat = false;
    bool has_spec = false;
};

struct CacheMeta {
    int target_sr = 40000;
    std::string version = "v1";
    bool has_f0 = true;  // 模型是否 f0（--has-f0）
    double per = 3.7;
    double overlap = 0.3;
    SlicerParams slicer;
    SpecConfig spec;
    int f0_bin = 256;
    double f0_min = 50.0;
    double f0_max = 1100.0;
};

// "samples/<id>.<kind>.bin"（kind = gt/16k/f0hz/f0bin/feat/spec）
std::string cache_sample_relpath(const std::string& id, const char* kind);

// 写 dataset.json + spec.json（samples/*.bin 已由 pipeline 写出）；cache_dir 需已存在或可创建
bool write_cache_index(const std::string& cache_dir, const CacheMeta& meta,
                       const std::vector<CacheSampleInfo>& samples, std::string* err);

// 自校验：格式/版本/唯一 id/文件存在/字节大小/帧数公式一致
bool validate_cache(const std::string& cache_dir, std::string* summary, std::string* err);

// ---------------- M5 只读解析（rvc-train 用；纯文件 I/O，不依赖 rvc_core/ggml） ----------------

// dataset.json 全量解析结果；样本 presence 标志由 files 表推断（与写入侧一致）
struct CacheDataset {
    std::string dir;
    CacheMeta meta;
    std::vector<CacheSampleInfo> samples;

    const CacheSampleInfo* find(const std::string& id) const;
};

// 解析 dataset.json 并填充 meta/samples（不做文件存在性检查；validate_cache 负责完整性）
bool load_cache_index(const std::string& cache_dir, CacheDataset* out, std::string* err);

// 读取单样本 bin：
//   f32 种类 = gt / 16k / f0hz / feat / spec；i32 种类 = f0bin
// 校验 presence 标志 + 文件存在 + 元素数与 CacheSampleInfo 字段一致
bool read_cache_sample_f32(const std::string& cache_dir, const CacheSampleInfo& s, const char* kind,
                           std::vector<float>* out, std::string* err);
bool read_cache_sample_i32(const std::string& cache_dir, const CacheSampleInfo& s, const char* kind,
                           std::vector<int32_t>* out, std::string* err);

} // namespace rvc_train::prep
