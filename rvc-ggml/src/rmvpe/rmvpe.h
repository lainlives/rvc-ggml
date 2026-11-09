// rmvpe.h
// RMVPE pitch extraction (DeepUnet + BiGRU + 360-class salience head).
// Structure follows the reference implementation (0xShug0/audio.cpp, MIT): 4 subgraphs
//   feature (DeepUnet windowed inference) / gru_forward / gru_reverse / head.
// Numeric target: corr ≥ 0.999 with RVC Python (infer/lib/rmvpe.py).

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <ggml.h>
#include <ggml-backend.h>

#include "gguf/gguf_model.h"

namespace rvc {
namespace rmvpe {

    struct Config {
        int sample_rate = 16000;
        int n_fft = 1024;
        int hop_length = 160;
        int win_length = 1024;
        int n_mels = 128;
        int n_classes = 360;
        int hidden_dim = 256;
        int feature_dim = 384;
        float mel_fmin = 30.0f;
        float mel_fmax = 8000.0f;
        float mel_clamp = 1e-5f;
        bool conv_flat = false;     // GGUF rvc.conv_flat: conv weights use a flattened layout (quantization-friendly)
    };

    Config read_config(const GgufModel& model);

    struct Result {
        int64_t frames = 0;                 // log-mel frame count (unpadded)
        int64_t padded_frames = 0;          // frame count after alignment to 32
        std::vector<float> mel;             // [n_mels, frames] (mel row-major)
        std::vector<float> hidden;          // [frames, n_classes] salience
        std::vector<float> hidden_padded;   // [padded_frames, n_classes]
        std::vector<float> f0;              // [frames] Hz; 0 means unvoiced
    };

    // Extractor: holds contexts and caches for the 4 subgraphs; reusable across calls;
    // rebuilds graphs for the corresponding size when the input length changes.
    class RmvpeExtractor {
    public:
        RmvpeExtractor(const GgufModel& model, ggml_backend_t backend);
        ~RmvpeExtractor();
        RmvpeExtractor(const RmvpeExtractor&) = delete;
        RmvpeExtractor& operator=(const RmvpeExtractor&) = delete;

        // Mono 16k float audio -> salience / f0 (threshold corresponds to Python thred=0.03)
        Result infer(const std::vector<float>& audio_16k, float threshold = 0.03f);

        // mel front end only (for the P3 self-test)
        std::vector<float> compute_mel(const std::vector<float>& audio_16k,
            int64_t* frames_out = nullptr) const;

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

} // namespace rmvpe
} // namespace rvc
