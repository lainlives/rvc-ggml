// RVC-Train-GGML - M5 训练数据加载器（src/data/dataset.h）
//
// 职责：把 M4 训练缓存（src/prep/cache.*）装配为训练步所需的主机缓冲：
//   样本装配（对齐官方 81eed5e train/data_utils.py::TextAudioLoaderMultiNSFsid.get_labels/
//   get_audio_text_pair）：phone = repeat(feat,2) → min(·,900) → 与 spec 取 min 对齐，
//   spec/wav/phone/pitch/pitchf 同截断；pitchf = UV 插值后 f0 Hz（无 mask/缩放）、
//   pitch = coarse bin（1..255，cache f0bin）。
//   批次（bucket）装配：pad 到桶顶 T_b + x_mask/y_mask 置零；ids_slice/q_noise/SineGen 噪声
//   由 traincpp::Rng 产生（golden 重放可用 BatchRandomInputs 注入固定值）。
//
// 布局口径（与 src/model/synthesizer.h、train_step_selftest.cpp 的图输入一致；ne = torch 反序）：
//   phone  [feat_dim,T,B]   pitch [T,B] i32      pitchf [T,B]（主机输入）
//   spec   [T,spec_bins,B]  wave  [seg_samples,1,B]（ids*hop 处切片）
//   y_mel  [seg,n_mels,B]   q_noise [T,q_channels,B]
//   x_mask/y_mask [T,1,B]   attn_valid [T,T,1,B]  sid [B] i32
//   slice_mat [T,seg,B]     pitchf_slice [seg,B]  sine_source_slice [1,seg_samples,B]
// 训练步（M5-2）按上述视图建张量；本模块不做任何图/Vulkan 操作（纯主机）。
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "prep/cache.h"
#include "traincpp/traincpp.h"

namespace rvc_train {
namespace data {

// 数据加载配置（M5 临时字段；M7 由 JSON config 定稿）
struct TrainDataConfig {
    std::string exp_dir;             // M4 缓存目录（dataset.json + samples/）
    int64_t batch_size = 1;
    int64_t segment_frames = 32;     // z 帧 = segment_size/hop（v1-40k = 32）
    int64_t segment_samples = 12800; // 官方 segment_size（v1-40k = 32*400）
    int64_t q_channels = 192;        // q_noise 通道（= inter_channels；v1 = 192）
    int64_t speaker_id = 0;          // 单说话人固定 0（官方 sid）
    std::vector<int64_t> buckets;    // 桶上界（对齐后 T）；首个 >= T 的桶命中
    // y_mel 目标（官方 train/mel_processing.py::spec_to_mel_torch 语义）
    int64_t n_fft = 2048;
    int64_t win = 2048;
    int64_t n_mels = 125;
    double sr = 40000.0;
    double fmin = 0.0;
    double fmax = -1.0;              // <=0 表示 sr/2
    float log_clip = 2e-6f;          // dynamic_range_compression 的 clamp min
};

// 单样本装配结果（torch 布局、行主序；hop = segment_samples/segment_frames）
struct TrainSample {
    std::string id;
    int64_t frames = 0;       // 对齐后 T
    int64_t feat_dim = 0;
    int64_t spec_bins = 0;
    int64_t hop = 0;
    std::vector<float> phone;    // [T, feat_dim]（repeat 后）
    std::vector<int32_t> pitch;  // [T]（f0bin）
    std::vector<float> pitchf;   // [T]（f0hz）
    std::vector<float> spec;     // [spec_bins, T]（行主序）
    std::vector<float> wav;      // [T*hop]（0_gt 截断）
};

// golden/对拍注入的随机量（线性布局与 BatchBuffers 对应字段一致）；未注入的项由 rng 生成
struct BatchRandomInputs {
    std::vector<int32_t> ids_slice;  // [B]
    std::vector<float> q_noise;      // 线性 = [B, q_channels, T_b]（torch）
    std::vector<float> sine_noise;   // 线性 = [B, segment_samples]（逐样本连续）
};

// 一个 bucket 批次的主机缓冲（ne 序；可直接 tensor_set 到图输入/叶子）
struct BatchBuffers {
    int64_t frames = 0;           // T_b（桶顶）
    int64_t batch = 0;
    int64_t feat_dim = 0;
    int64_t spec_bins = 0;
    int64_t segment_frames = 0;
    int64_t segment_samples = 0;
    int64_t q_channels = 0;
    int64_t hop = 0;
    std::vector<int64_t> lengths;    // [B] 有效 T_i
    std::vector<int32_t> ids_slice;  // [B]
    std::vector<float> phone;        // ne [feat_dim, T_b, B]
    std::vector<int32_t> pitch;      // ne [T_b, B]
    std::vector<float> pitchf;       // ne [T_b, B]（主机输入）
    std::vector<float> spec;         // ne [T_b, spec_bins, B]
    std::vector<float> wave;         // ne [segment_samples, 1, B]
    std::vector<float> y_mel;        // ne [segment_frames, n_mels, B]
    std::vector<int32_t> sid;        // ne [B]
    std::vector<float> q_noise;      // ne [T_b, q_channels, B]
    std::vector<float> x_mask;       // ne [T_b, 1, B]
    std::vector<float> y_mask;       // ne [T_b, 1, B]
    std::vector<float> attn_valid;   // ne [T_b, T_b, 1, B]
    std::vector<float> slice_mat;    // ne [T_b, segment_frames, B]
    std::vector<float> pitchf_slice; // ne [segment_frames, B]
    std::vector<float> sine_source_slice;  // ne [1, segment_samples, B]
};

class Dataset {
public:
    static bool open(const std::string& cache_dir, const TrainDataConfig& cfg, Dataset* out,
                     std::string* err);

    const prep::CacheDataset& cache() const { return cache_; }
    const TrainDataConfig& config() const { return cfg_; }
    int64_t size() const { return static_cast<int64_t>(cache_.samples.size()); }
    const prep::CacheSampleInfo& sample_info(int64_t index) const;
    int64_t index_of(const std::string& id) const;  // 未找到 -1

    // 官方对齐后的帧数 T（不需读 bin；供 bucket 采样器）
    int64_t aligned_frames(int64_t index) const;
    // 命中的桶上界（首个 >= frames）；无命中返回 -1
    int64_t bucket_of(int64_t frames) const;

    // 装配单个样本（读 cache bin）；index 越界/缺文件/帧数不足 → false + err
    bool assemble(int64_t index, TrainSample* out, std::string* err) const;
    bool assemble_id(const std::string& id, TrainSample* out, std::string* err) const;

    // 构建 bucket 批次：
    //   indices 保持调用方顺序（官方 collate 由调用方按长度降序传入）；
    //   bucket_top 必须 >= 所有样本 T_i，且 >= segment_frames；
    //   injected 非空时按注入随机量（golden 重放，尺寸/范围强校验）；否则用 rng 生成。
    bool build_batch(const std::vector<int64_t>& indices, int64_t bucket_top, traincpp::Rng* rng,
                     const BatchRandomInputs* injected, BatchBuffers* out, std::string* err) const;

private:
    prep::CacheDataset cache_;
    TrainDataConfig cfg_;
    std::vector<float> mel_basis_;             // [n_mels][spec_bins] 行主序
    std::map<std::string, int64_t> index_map_; // id → samples 下标
};

} // namespace data
} // namespace rvc_train
