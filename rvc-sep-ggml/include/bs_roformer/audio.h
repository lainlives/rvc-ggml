#pragma once
#include <cstddef>
#include <string>
#include <vector>

/**
 * Audio buffer: interleaved float32 samples (L, R, L, R, ... for stereo).
 */
struct AudioBuffer {
    std::vector<float> data;  // interleaved samples
    unsigned int channels;
    unsigned int sampleRate;
    size_t samples;           // total samples (frames * channels)
};

/**
 * WAV file I/O for rvc-sep.
 *
 * Audio is automatically resampled to the target sample rate (default 44100 Hz,
 * the Roformer models' training rate) and converted to interleaved stereo float32.
 * Unlike the upstream BSRoformer.cpp (which used FFmpeg/libav), this build reads
 * and writes WAV only; use ffmpeg to convert other formats first.
 */
class AudioFile {
public:
    /**
     * Load audio from a WAV file (any PCM/IEEE-float bit depth supported by dr_wav).
     * Mono is expanded to stereo, >2 channels are downmixed, and the result is
     * resampled to target_sample_rate when necessary.
     * @throws std::runtime_error if the file cannot be opened or decoded.
     */
    static AudioBuffer Load(const std::string& path, int target_sample_rate = 44100);

    /**
     * Save interleaved stereo float32 audio to a WAV file (IEEE float32).
     * @throws std::runtime_error if the file cannot be written.
     */
    static void Save(const std::string& path, const AudioBuffer& buffer);
};
