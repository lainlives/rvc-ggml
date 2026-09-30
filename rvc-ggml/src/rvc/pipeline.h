// pipeline.h
// RVC 全链路离线转换管线（对齐 infer/modules/vc/pipeline.py）：
//   16k 音频 -> 48Hz 高通 -> 静音点切块 -> HuBERT 特征 ->(检索混合)-> ×2 插值
//   -> protect 混合 -> VITS（RMVPE f0 / SineGen 激励）-> RMS 混合 ->(重采样)-> int16

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <ggml-backend.h>

#include "gguf/gguf_model.h"
#include "vits/synthesizer.h"

namespace rvc {

    class RetrievalIndex;

    struct PipelineConfig {
        // 对齐 configs/config.py device_config()（CPU/非 half：1/6/38/41）
        int x_pad = 1;
        int x_query = 6;
        int x_center = 38;
        int x_max = 41;
    };

    struct ConvertOptions {
        int f0_up_key = 0;              // 半音
        int sid = 0;
        float index_rate = 0.0f;
        float protect = 0.33f;
        float rms_mix_rate = 1.0f;
        int resample_sr = 0;            // 0 = 不重采样
        int filter_radius = 3;          // rmvpe 不使用（保留对齐接口）
        const RetrievalIndex* index = nullptr;
        // 确定性验证：非空时按 e2e_call{i}_{rnd,sine_noise}.raw 注入噪声
        std::string noise_dir;
        // 验证用：以参考 e2e_call{i}_pitchf.raw 覆盖本块 f0（隔离 RMVPE 近似并列翻转）
        bool f0_inject = false;
        bool verbose = true;
    };

    struct ConvertResult {
        std::vector<int16_t> pcm;
        int sample_rate = 0;
    };

    class RvcPipeline {
    public:
        RvcPipeline(const GgufModel& hubert_model, const GgufModel& rmvpe_model,
            const GgufModel& voice_model, ggml_backend_t backend);
        ~RvcPipeline();
        RvcPipeline(const RvcPipeline&) = delete;
        RvcPipeline& operator=(const RvcPipeline&) = delete;

        const vits::Config& voice_config() const;

        // audio_16k: 单声道 16k 浮点（调用方负责加载/重采样）
        ConvertResult convert(const std::vector<float>& audio_16k, const ConvertOptions& opt);

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

} // namespace rvc
