#pragma once
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>
// Forward declaration
class BSRoformer;

// Forward declaration
namespace ggml {
struct context;
struct cgraph;
}

class Inference {
public:
    using CancelCallback = std::function<bool()>;

    Inference(const std::string& model_path);
    ~Inference();

    // Process a full audio track (interleaved stereo float32)
    // Uses overlap-add chunking to handle long files
    // Process a full audio track (interleaved stereo float32)
    // Returns a vector of stems, where each stem is an interleaved stereo float vector
    std::vector<std::vector<float>> Process(const std::vector<float>& input_audio,
        int chunk_size = 352800,
        int num_overlap = 2,
        std::function<void(float)> progress_callback = nullptr,
        CancelCallback cancel_callback = nullptr);

    // Low-level chunk processing (public for testing)
    std::vector<std::vector<float>> ProcessChunk(const std::vector<float>& chunk_audio);

    // Get model's recommended inference defaults
    int GetDefaultChunkSize() const;
    int GetDefaultNumOverlap() const;
    int GetSampleRate() const;
    int GetNumStems() const;

    // Static helper for Overlap-Add logic (matches Python exactly)
    // model_func: input [samples], output [stems][samples] (interleaved stereo)
    using ModelCallback = std::function<std::vector<std::vector<float>>(const std::vector<float>&)>;
    static std::vector<std::vector<float>> ProcessOverlapAdd(const std::vector<float>& input_audio,
        int chunk_size,
        int num_overlap,
        ModelCallback model_func,
        std::function<void(float)> progress_callback = nullptr,
        CancelCallback cancel_callback = nullptr);

private:
    // Pipelined Overlap-Add
    std::vector<std::vector<float>> ProcessOverlapAddPipelined(const std::vector<float>& input_audio,
        int chunk_size,
        int num_overlap,
        std::function<void(float)> progress_callback,
        CancelCallback cancel_callback);

private:
    std::unique_ptr<BSRoformer> model_;

    // Persistent Graph State
    struct ggml_context* ctx_ = nullptr;
    struct ggml_cgraph* gf_ = nullptr;
    struct ggml_gallocr* allocr_ = nullptr;

    // Cached Input Tensors (owned by ctx_)
    struct ggml_tensor* input_tensor_ = nullptr;
    struct ggml_tensor* pos_time_ = nullptr;
    struct ggml_tensor* pos_freq_ = nullptr;
    struct ggml_tensor* mask_out_tensor_ = nullptr;

    // Cached Host Data (to avoid reallocation)
    std::vector<int32_t> pos_time_data_;
    std::vector<int32_t> pos_freq_data_;

    // Current config state
    int cached_n_frames_ = -1;

    // Pipelined State Data
    struct ChunkState {
        int id = -1;
        std::vector<float> input_audio; // Original chunk audio
        std::vector<float> stft_flattened; // [Prepared Input for GPU]
        std::vector<std::vector<float>> stft_outputs; // Kept for reconstruction
        int n_frames = 0;

        std::vector<float> mask_output; // Output from GPU
        std::vector<std::vector<float>> final_audio; // Result after ISTFT [stems][samples]
    };

    // Helper to ensure graph is built for specific n_frames
    bool EnsureGraph(int n_frames);

    void ComputeSTFT(const std::vector<float>& input_audio,
        std::vector<std::vector<float>>& stft_outputs,
        int& n_frames);

    void PrepareModelInput(const std::vector<std::vector<float>>& stft_outputs,
        int n_frames,
        std::vector<float>& model_input_rearranged);

    void PostProcessAndISTFT(const std::vector<float>& mask_output,
        const std::vector<std::vector<float>>& stft_outputs,
        int n_frames,
        std::vector<std::vector<float>>& output_audio);

    // Pipeline Steps
    std::shared_ptr<ChunkState> PreProcessChunk(const std::vector<float>& chunk_audio, int id);
    void RunInference(std::shared_ptr<ChunkState> state);
    void PostProcessChunk(std::shared_ptr<ChunkState> state);
};
