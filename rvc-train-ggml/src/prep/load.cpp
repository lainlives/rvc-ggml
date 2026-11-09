// load.cpp -- M4-1 audio loading and resampling implementation

#include "prep/load.h"

#include <cmath>
#include <cstdio>
#include <fstream>

#include "audio/audio_io.h"  // rvc_core public header (load_wav)
#include "prep/resample.h"   // in-house sinc resampler (rvc resample_mono distorts 44.1k->40k)

namespace rvc_train::prep {

std::vector<float> to_f32(const std::vector<double>& v) {
    std::vector<float> out(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) {
        out[i] = static_cast<float>(v[i]);
    }
    return out;
}

std::vector<double> resample_mono_double(const std::vector<double>& in, int src_sr, int dst_sr) {
    if (src_sr == dst_sr || in.empty()) {
        return in;
    }
    std::vector<float> f(in.size());
    for (std::size_t i = 0; i < in.size(); ++i) {
        f[i] = static_cast<float>(in[i]);
    }
    const std::vector<float> r = resample_sinc(f, src_sr, dst_sr);
    return std::vector<double>(r.begin(), r.end());
}

bool load_wav_mono(const std::string& path, int target_sr, MonoAudio* out, std::string* err) {
    try {
        const AudioData audio = load_wav(path);
        if (audio.total_samples == 0 || audio.num_channels <= 0) {
            if (err) {
                *err = "WAV is empty or has an invalid channel count: " + path;
            }
            return false;
        }
        if (audio.num_channels > 2) {
            if (err) {
                *err = "WAV with >2 channels is not supported yet (convert to mono/stereo first): " + path;
            }
            return false;
        }

        // Downmix to double; coefficients match ffmpeg swresample's default matrix (M4-1 measurements):
        //   mono -> 1.0; stereo -> 1/sqrt(2) (energy normalization, not 1/2 averaging;
        //   using 1/2 scales everything by 0.707, producing ~30% amplitude deviation from the official 0_gt)
        const int ch = audio.num_channels;
        const double scale = (ch == 1) ? 1.0 : 1.0 / std::sqrt(2.0);
        std::vector<double> mono(static_cast<std::size_t>(audio.total_samples), 0.0);
        for (std::size_t t = 0; t < mono.size(); ++t) {
            double sum = 0.0;
            for (int c = 0; c < ch; ++c) {
                sum += static_cast<double>(audio.samples[t * static_cast<std::size_t>(ch) +
                                                        static_cast<std::size_t>(c)]);
            }
            mono[t] = sum * scale;
        }

        out->samples = resample_mono_double(mono, audio.sample_rate, target_sr);
        out->sample_rate = target_sr;
        if (err) {
            err->clear();
        }
        return true;
    } catch (const std::exception& e) {
        if (err) {
            *err = std::string("failed to read WAV: ") + e.what();
        }
        return false;
    }
}

bool load_raw_f32_mono(const std::string& path, int sample_rate, MonoAudio* out, std::string* err) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        if (err) {
            *err = "cannot open F32 raw: " + path;
        }
        return false;
    }
    const std::streamoff size = f.tellg();
    if (size <= 0 || size % 4 != 0) {
        if (err) {
            *err = "invalid F32 raw size (must be non-empty and 4-byte aligned): " + path;
        }
        return false;
    }
    f.seekg(0, std::ios::beg);
    std::vector<float> f32(static_cast<std::size_t>(size) / 4);
    f.read(reinterpret_cast<char*>(f32.data()), size);
    if (!f) {
        if (err) {
            *err = "F32 raw read failed: " + path;
        }
        return false;
    }
    out->samples.resize(f32.size());
    for (std::size_t i = 0; i < f32.size(); ++i) {
        out->samples[i] = static_cast<double>(f32[i]);
    }
    out->sample_rate = sample_rate;
    if (err) {
        err->clear();
    }
    return true;
}

bool write_raw_f32(const std::string& path, const float* data, std::size_t count, std::string* err) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) {
        if (err) {
            *err = "cannot write: " + path;
        }
        return false;
    }
    if (count > 0) {
        f.write(reinterpret_cast<const char*>(data),
                static_cast<std::streamsize>(count * sizeof(float)));
    }
    if (!f) {
        if (err) {
            *err = "write failed: " + path;
        }
        return false;
    }
    return true;
}

bool write_raw_f32(const std::string& path, const std::vector<float>& data, std::string* err) {
    return write_raw_f32(path, data.data(), data.size(), err);
}

bool write_raw_i32(const std::string& path, const int32_t* data, std::size_t count, std::string* err) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) {
        if (err) {
            *err = "cannot write: " + path;
        }
        return false;
    }
    if (count > 0) {
        f.write(reinterpret_cast<const char*>(data),
                static_cast<std::streamsize>(count * sizeof(int32_t)));
    }
    if (!f) {
        if (err) {
            *err = "write failed: " + path;
        }
        return false;
    }
    return true;
}

bool write_raw_i32(const std::string& path, const std::vector<int32_t>& data, std::string* err) {
    return write_raw_i32(path, data.data(), data.size(), err);
}

} // namespace rvc_train::prep
