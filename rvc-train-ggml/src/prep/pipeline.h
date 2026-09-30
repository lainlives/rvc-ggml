// pipeline.h —— M4 预处理核心流程（rvc-prep 主机侧）
//
// 数据流（官方 81eed5e train/preprocess.py + extract_f0.py 复刻）：
//   WAV → 单声道+重采样 → [double] 因果 lfilter(48Hz) → Slicer2 切片 →
//   3.7s/3.4s 分块 → 峰值归一化（0_gt）→ 16k 重采样 →
//   [M4-2] RMVPE f0 + UV 插值 + coarse bins；[M4-3] HuBERT 特征（线性谱属 M4-4）
//
// 官方分块循环的两个“怪点”必须原样复刻（见 docs/交接说明.md §4.8 A）：
//   1) idx1 在 else 分支自增但该尾部块不在此写出；for 结束后用当时的 idx1 再写一次末块的尾部；
//   2) 每个切片 i 从 0 重新计数，但 idx1 跨切片连续。
// 结果：多切片时中间切片的尾部块被跳过，且索引不连续（与官方一致）。

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

// prep/feat.h：HuBERT 特征提取器（自建执行器，重复 compute 安全；见 Q22）
class HubertFeatureExtractor;

struct PrepOptions {
    int target_sr = 40000;
    std::string version = "v1";
    bool has_f0 = true;
    double per = 3.7;
    double overlap = 0.3;
    SlicerParams slicer;
    bool no_16k = false;          // 跳过 16k 重采样（对拍 G4 时不要开）
    std::string debug_dump_dir;   // 非空则落盘 load/hp/gt/16k/f0/slices/chunks/coeffs

    // M4-2：f0（非空且 with_f0 时对每个 valid chunk 计算；调用方持有 extractor 生命周期）
    bool with_f0 = true;
    rvc::rmvpe::RmvpeExtractor* rmvpe = nullptr;

    // M4-3：HuBERT 特征（非空且 with_feat 时对每个 valid chunk 计算；维度由 version 决定）
    bool with_feat = true;
    HubertFeatureExtractor* hubert = nullptr;

    // M4-4：线性谱（对每个 valid chunk 的 0_gt 计算；配置由调用方按 sr/version 派生）
    bool with_spec = true;
    SpecConfig spec;

    // M4-4：训练缓存（非空则对每个 valid chunk 写 samples/<id>.{gt,16k,f0hz,f0bin,feat,spec}.bin；
    // dataset.json/spec.json 由调用方在全部文件处理后写）
    std::string cache_dir;
};

struct ChunkInfo {
    int index = 0;             // 官方 idx1
    int slice_index = 0;
    int64_t start_in_slice = 0;  // 样本
    int64_t n_samples = 0;
    bool valid = false;
    bool f0_valid = false;     // M4-2：有发声帧（全零 chunk 官方跳过）
    int64_t f0_frames = 0;     // M4-2：f0 帧数（= 1 + samples16k/160）
    bool feat_valid = false;   // M4-3：有特征帧
    int64_t feat_frames = 0;   // M4-3：HuBERT 帧数（= (samples16k-400)/320 + 1）
    int64_t feat_dim = 0;      // M4-3：256(v1)/768(v2)
    bool spec_valid = false;   // M4-4：有谱帧
    int64_t spec_frames = 0;   // M4-4：线性谱帧数（= gt_samples//hop，v1-40k）
    int64_t spec_bins = 0;     // M4-4：n_fft/2+1 = 1025（v1/v2）
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
    std::vector<CacheSampleInfo> samples;  // M4-4：缓存清单（仅 valid chunk；cache_dir 非空时填充）
};

// 普通模式：WAV → 下混/重采样 → core（load dump 也在此模式落盘）
bool run_prep_wav(const std::string& wav_path, const std::string& case_name,
                  const PrepOptions& opt, PrepFileResult* out, std::string* err);

// 注入模式：跳过解码/重采样（golden post-load 波形注入；隔离切片/hp/norm 的精确一致）
bool run_prep_injected(const MonoAudio& load, const std::string& case_name,
                       const PrepOptions& opt, PrepFileResult* out, std::string* err);

} // namespace rvc_train::prep
