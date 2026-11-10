// rvc-sep — Roformer source separation CLI.
//
// Separates an audio track into its stems (vocals / instrumental) with a
// Mel-Band Roformer or BS Roformer GGUF model. Input is resampled to 44.1 kHz
// automatically; output is written as float32 WAV.
//
// Usage:
//   rvc-sep <model.gguf> <input.wav> <output.wav> [--chunk-size N] [--overlap N] [--device gpu|cpu]

#include "bs_roformer/audio.h"
#include "bs_roformer/inference.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

void print_usage(const char* program_name) {
    std::cout
        << "Usage: " << program_name << " <model.gguf> <input.wav> <output.wav> [options]\n"
        << "\n"
        << "Separates <input.wav> into stems with a Roformer GGUF model and writes them as\n"
        << "float32 WAV at 44100 Hz. With multiple stems the files are named\n"
        << "<output>_stem_0.wav, <output>_stem_1.wav, ... (stem 0 is usually the vocals).\n"
        << "\n"
        << "Options:\n"
        << "  --chunk-size <N>   chunk size in samples (default: the model's value)\n"
        << "  --overlap <N>      number of overlaps for cross-fading (default: the model's value)\n"
        << "  --device <dev>     gpu (default) or cpu\n"
        << "  -h, --help         show this help message\n";
}

void set_force_cpu_env() {
#ifdef _WIN32
    _putenv_s("BSR_FORCE_CPU", "1");
#else
    setenv("BSR_FORCE_CPU", "1", 1);
#endif
}

} // namespace

int main(int argc, char* argv[]) {
    int chunk_size = -1;   // -1 = use the model default
    int num_overlap = -1;  // -1 = use the model default
    bool chunk_size_set = false;
    bool overlap_set = false;
    std::vector<std::string> positional;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--chunk-size" && i + 1 < argc) {
            try {
                chunk_size = std::stoi(argv[++i]);
            } catch (...) {
                std::cerr << "Error: invalid --chunk-size value\n";
                return 1;
            }
            if (chunk_size <= 0) {
                std::cerr << "Error: --chunk-size must be a positive integer\n";
                return 1;
            }
            chunk_size_set = true;
        } else if (arg == "--overlap" && i + 1 < argc) {
            try {
                num_overlap = std::stoi(argv[++i]);
            } catch (...) {
                std::cerr << "Error: invalid --overlap value\n";
                return 1;
            }
            if (num_overlap < 1) {
                std::cerr << "Error: --overlap must be at least 1\n";
                return 1;
            }
            overlap_set = true;
        } else if (arg == "--device" && i + 1 < argc) {
            const std::string dev = argv[++i];
            if (dev == "cpu") {
                set_force_cpu_env();
            } else if (dev != "gpu") {
                std::cerr << "Error: --device must be 'gpu' or 'cpu'\n";
                return 1;
            }
        } else if (!arg.empty() && arg[0] == '-') {
            std::cerr << "Error: unknown option: " << arg << "\n";
            print_usage(argv[0]);
            return 1;
        } else {
            positional.push_back(arg);
        }
    }

    if (positional.size() != 3) {
        print_usage(argv[0]);
        return 1;
    }
    const std::string model_path = positional[0];
    const std::string input_path = positional[1];
    const std::string output_path = positional[2];

    try {
        std::cout << "rvc-sep — Roformer source separation (ggml)\n";
        Inference engine(model_path);

        if (!chunk_size_set) {
            chunk_size = engine.GetDefaultChunkSize();
        }
        if (!overlap_set) {
            num_overlap = engine.GetDefaultNumOverlap();
        }

        const AudioBuffer input_audio = AudioFile::Load(input_path);
        std::cout << "Audio loaded: " << input_audio.samples << " samples, "
                  << input_audio.channels << " channels, "
                  << input_audio.sampleRate << " Hz\n";
        std::cout << "Processing with chunk_size=" << chunk_size
                  << ", overlap=" << num_overlap << std::endl;

        const auto start = std::chrono::high_resolution_clock::now();
        auto progress_callback = [](float progress) {
            const int bar_width = 50;
            std::cout << "[";
            const int pos = static_cast<int>(bar_width * progress);
            for (int i = 0; i < bar_width; ++i) {
                std::cout << (i < pos ? '=' : (i == pos ? '>' : ' '));
            }
            std::cout << "] " << static_cast<int>(progress * 100.0f) << " %\r";
            std::cout.flush();
        };

        std::vector<std::vector<float>> stems =
            engine.Process(input_audio.data, chunk_size, num_overlap, progress_callback);

        std::cout << std::string(70, ' ') << "\r";
        const double seconds = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - start).count();
        std::cout << "Processed in " << seconds << " s; model returned "
                  << stems.size() << " stem(s).\n";

        for (size_t i = 0; i < stems.size(); ++i) {
            std::string current_output_path = output_path;
            if (stems.size() > 1) {
                const size_t dot_pos = output_path.find_last_of('.');
                if (dot_pos != std::string::npos) {
                    current_output_path = output_path.substr(0, dot_pos) +
                        "_stem_" + std::to_string(i) + output_path.substr(dot_pos);
                } else {
                    current_output_path = output_path + "_stem_" + std::to_string(i);
                }
            }

            AudioBuffer out;
            out.data = std::move(stems[i]);
            out.channels = 2;  // Roformer output is always stereo
            out.sampleRate = 44100;
            out.samples = out.data.size();

            AudioFile::Save(current_output_path, out);
            std::cout << "Wrote stem " << i << ": " << current_output_path << "\n";
        }
        std::cout << "Done.\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
