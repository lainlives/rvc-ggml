// pipeline.h
// RVC full offline conversion pipeline (aligned with infer/modules/vc/pipeline.py):
//   16k audio -> 48Hz high-pass -> chunking at silent points -> HuBERT features ->(retrieval blending)-> ×2 interpolation
//   -> protect blending -> VITS (RMVPE f0 / SineGen excitation) -> RMS blending ->(resample)-> int16

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
        // Aligned with configs/config.py device_config() (CPU/non-half: 1/6/38/41)
        int x_pad = 1;
        int x_query = 6;
        int x_center = 38;
        int x_max = 41;
    };

    struct ConvertOptions {
        int f0_up_key = 0;              // semitones
        int sid = 0;
        float index_rate = 0.0f;
        float protect = 0.33f;
        float rms_mix_rate = 1.0f;
        int resample_sr = 0;            // 0 = no resampling
        int filter_radius = 3;          // unused by rmvpe (kept for interface alignment)
        const RetrievalIndex* index = nullptr;
        // Deterministic validation: when non-empty, inject noise from e2e_call{i}_{rnd,sine_noise}.raw
        std::string noise_dir;
        // For validation: overwrite this chunk's f0 with reference e2e_call{i}_pitchf.raw (isolates near-tie flips in RMVPE)
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

        // audio_16k: mono 16k float (caller handles loading/resampling)
        ConvertResult convert(const std::vector<float>& audio_16k, const ConvertOptions& opt);

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

} // namespace rvc
