// main.cpp — RVC-GGML
// ggml (Vulkan/CUDA/CPU) based RVC voice conversion inference
// P0 stage: CLI skeleton + audio pipeline self-test

#ifdef _WIN32
#include <windows.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "utils.h"
#include "engine.h"
#include "gguf/gguf_model.h"
#include "hubert/hubert.h"
#include "rmvpe/rmvpe.h"
#include "vits/synthesizer.h"
#include "rvc/sinegen.h"
#include "rvc/pipeline.h"
#include "rvc/retrieval.h"
#include "audio/audio_io.h"
#include "audio/dsp.h"

#include <memory>

using namespace std;

namespace {

    int run_audio_selftest(const string& input_wav, const string& out_dir,
        int split_query, int split_center, int split_max) {
        utils::ensure_directory(out_dir);

        Logger::info("Loading audio: " + input_wav);
        AudioData audio = load_wav(input_wav);
        cout << "  Sample rate: " << audio.sample_rate
            << "  Channels: " << audio.num_channels
            << "  Samples: " << audio.total_samples << endl;

        vector<float> mono = to_mono_16k(audio, 16000);
        cout << "  Mono 16k samples: " << mono.size() << endl;
        {
            ofstream f(out_dir + "/a_mono16k.raw", ios::binary);
            f.write(reinterpret_cast<const char*>(mono.data()),
                static_cast<streamsize>(mono.size() * sizeof(float)));
        }
        save_wav(out_dir + "/a_mono16k.wav", mono.data(), 1, 16000, mono.size());

        vector<float> filtered = mono;
        rvc::dsp::high_pass_48hz_in_place(filtered);
        {
            ofstream f(out_dir + "/b_hp48.raw", ios::binary);
            f.write(reinterpret_cast<const char*>(filtered.data()),
                static_cast<streamsize>(filtered.size() * sizeof(float)));
        }
        save_wav(out_dir + "/b_hp48.wav", filtered.data(), 1, 16000, filtered.size());

        // Silent split points (Python Pipeline computes these on the filtered 16k audio)
        const auto splits = rvc::dsp::quiet_split_points(filtered, split_query, split_center, split_max);
        cout << "  Silent split points: " << splits.size() << " total";
        if (!splits.empty()) {
            cout << " (first=" << splits.front() << ")";
        }
        cout << endl;
        {
            ofstream f(out_dir + "/c_splits.txt");
            for (auto s : splits) f << s << "\n";
        }
        {
            const auto asum = rvc::dsp::audio_sum_curve(filtered);
            ofstream f(out_dir + "/e_audiosum.raw", ios::binary);
            f.write(reinterpret_cast<const char*>(asum.data()),
                static_cast<streamsize>(asum.size() * sizeof(float)));
        }

        // RMS curve (16k)
        const auto rms = rvc::dsp::frame_rms(filtered, 16000);
        {
            ofstream f(out_dir + "/d_rms16k.raw", ios::binary);
            f.write(reinterpret_cast<const char*>(rms.data()),
                static_cast<streamsize>(rms.size() * sizeof(float)));
        }
        cout << "  RMS frames: " << rms.size() << endl;

        Logger::info("Self-test output written to: " + out_dir);
        return 0;
    }

    void print_usage() {
        cout << "Usage: RVC-GGML.exe [options]" << endl;
        cout << "  --input <wav>        input audio file (entering conversion mode when given)" << endl;
        cout << "  --output <wav>       output file (default output/converted.wav)" << endl;
        cout << "  --model <gguf>       character model (default models/guanguanV1.gguf)" << endl;
        cout << "  --pitch <n>          pitch shift in semitones (default 0)" << endl;
        cout << "  --proposed-pitch <hz>  auto pitch: shift median f0 to <hz> (155 male / 255 female), clamp +/-12" << endl;
        cout << "  --sid <n>            speaker ID (default 0)" << endl;
        cout << "  --index <gguf>       retrieval index (unused by default)" << endl;
        cout << "  --index-rate <r>     retrieval mix rate 0~1 (default 0)" << endl;
        cout << "  --protect <p>        unvoiced protection 0~0.5 (default 0.33)" << endl;
        cout << "  --rms-mix-rate <r>   RMS mix rate 0~1 (default 1=no mixing)" << endl;
        cout << "  --resample-sr <n>    output resample rate (default 0=model sample rate)" << endl;
        cout << "  --noise-dir <dir>    deterministic validation: e2e_call{i}_{rnd,sine_noise}.raw" << endl;
        cout << "  --f0-inject          for validation: override each chunk's f0 with e2e_call{i}_pitchf.raw" << endl;
        cout << "  --device <id>        inference device (default 99=CPU, 0+=GPU)" << endl;
        cout << "  --list-devices       list available GPU devices" << endl;
        cout << "  --audio-selftest     audio pipeline self-test (writes intermediate results to output/selftest)" << endl;
        cout << "  --hubert-selftest    HuBERT graph self-test (requires output/ref reference data)" << endl;
        cout << "  --rmvpe-selftest     RMVPE graph self-test (requires output/ref reference data)" << endl;
        cout << "  --vits-selftest      VITS synthesizer self-test (requires --model to specify a character GGUF)" << endl;
        cout << "  --ref-dir <dir>      reference data directory (default output/ref)" << endl;
        cout << "  --model-dir <dir>    GGUF model directory (default models)" << endl;
        cout << "  --hubert <gguf>      feature-extractor GGUF (default: <model-dir>/hubert.gguf)" << endl;
        cout << "  --rmvpe <gguf>       f0-extractor GGUF (default: <model-dir>/rmvpe.gguf)" << endl;
        cout << "  --quant <level>      quant level: fp16/q8_0/q5_1/q4_1/q4_0" << endl;
        cout << "                       (loads hubert.<level>.gguf / rmvpe.<level>.gguf / <character>.<level>.gguf)" << endl;
        cout << "  --x-pad/--x-query/--x-center/--x-max  chunk parameters in seconds (default 1/6/38/41)" << endl;
        cout << "  --interactive, -i    force the interactive wizard (entered by default when started without arguments)" << endl;
        cout << "  --help, -h           show help" << endl;
    }

    // Read a raw float32 file
    bool load_raw_f32(const string& path, vector<float>& out) {
        std::ifstream in(path, ios::binary | ios::ate);
        if (!in) {
            return false;
        }
        const size_t n = static_cast<size_t>(in.tellg()) / sizeof(float);
        out.resize(n);
        in.seekg(0);
        in.read(reinterpret_cast<char*>(out.data()), static_cast<streamsize>(n * sizeof(float)));
        return true;
    }

    void dump_raw_f32(const string& path, const vector<float>& data) {
        ofstream f(path, ios::binary);
        f.write(reinterpret_cast<const char*>(data.data()),
            static_cast<streamsize>(data.size() * sizeof(float)));
    }

    // Read a raw file as a byte stream
    bool load_raw_bytes(const string& path, vector<uint8_t>& out) {
        std::ifstream in(path, ios::binary | ios::ate);
        if (!in) {
            return false;
        }
        const size_t n = static_cast<size_t>(in.tellg());
        out.resize(n);
        in.seekg(0);
        in.read(reinterpret_cast<char*>(out.data()), static_cast<streamsize>(n));
        return true;
    }

    bool load_raw_i32(const string& path, vector<int32_t>& out) {
        vector<uint8_t> bytes;
        if (!load_raw_bytes(path, bytes) || bytes.empty()) {
            return false;
        }
        out.resize(bytes.size() / sizeof(int32_t));
        memcpy(out.data(), bytes.data(), out.size() * sizeof(int32_t));
        return true;
    }


    // ------------------------------------------------------------
    // HuBERT self-test: load reference input -> build graph -> run -> dump intermediate tensors
    // ------------------------------------------------------------
    int run_hubert_selftest(const string& model_dir, const string& ref_dir, int device_id) {
        const string model_path = model_dir + "/hubert.gguf";
        if (!utils::file_exists(model_path)) {
            Logger::error("Model not found: " + model_path);
            return 1;
        }
        const string ref_input = ref_dir + "/hubert_input.raw";
        if (!utils::file_exists(ref_input)) {
            Logger::error("Reference input not found: " + ref_input + " (self-test requires reference tensors, not shipped with the distribution)");
            return 1;
        }

        rvc::GgmlDevice device(device_id);
        Logger::info("Compute backend: " + device.name());

        GgufModel model;
        model.load(model_path, device.backend());
        const auto cfg = rvc::hubert::read_config(model);
        cout << "  HuBERT: layers=" << cfg.n_layers << " hidden=" << cfg.hidden
            << " heads=" << cfg.heads << " conv_pos=" << cfg.conv_pos << endl;


        // Read reference input
        std::ifstream in(ref_input, ios::binary | ios::ate);
        if (!in) {
            Logger::error("Failed to read reference input");
            return 1;
        }
        const size_t n_samples = static_cast<size_t>(in.tellg()) / sizeof(float);
        vector<float> audio(n_samples);
        in.seekg(0);
        in.read(reinterpret_cast<char*>(audio.data()), n_samples * sizeof(float));
        in.close();

        const int64_t T = static_cast<int64_t>(n_samples);
        const int64_t T_pad = ((T + 159) / 160) * 160;
        audio.resize(static_cast<size_t>(T_pad), 0.0f);
        cout << "  Input samples: " << T << " -> pad " << T_pad << endl;

        rvc::GgmlGraphRunner runner(device.backend(), 1024ull * 1024 * 1024, 32768);
        ggml_tensor* input = ggml_new_tensor_3d(runner.ctx(), GGML_TYPE_F32, T_pad, 1, 1);
        ggml_set_name(input, "hubert.audio");
        ggml_set_input(input);

        ggml_tensor* out_proj = nullptr;
        ggml_tensor* out_l9 = nullptr;
        ggml_tensor* out_l12 = nullptr;
        rvc::hubert::build_graph(runner.ctx(), model, cfg, input, &out_proj, &out_l9, &out_l12);

        if (out_proj != nullptr) {
            ggml_set_output(out_proj);
            runner.expand(out_proj);
        }
        ggml_set_output(out_l9);
        ggml_set_output(out_l12);
        runner.expand(out_l9);
        runner.expand(out_l12);

        // Debug: expand intermediate tensors (RVC_HUBERT_DUMP=1)
        const bool dump_debug = getenv("RVC_HUBERT_DUMP") != nullptr;
        if (dump_debug) {
            for (const auto& d : rvc::hubert::debug_tensors()) {
                ggml_set_output(d.tensor);
                runner.expand(d.tensor);
            }
        }
        if (!runner.allocate()) {
            Logger::error("Failed to allocate compute graph buffer");
            return 1;
        }
        cout << "  Graph: nodes=" << runner.node_count()
            << " buffer=" << runner.buffer_size() / (1024 * 1024) << " MiB" << endl;

        Timer timer;
        runner.set_input_vec(input, audio);
        runner.compute();
        cout << "  Inference time: " << timer.elapsed_ms() << " ms" << endl;

        const int64_t frames = out_l9->ne[1];
        cout << "  Output frames: " << frames << endl;

        // Tensor is [C, T] (C in ne0): memory matches PyTorch row-major [T, C],
        // so reading it as numpy reshape(T, C) gives the reference layout.
        auto dump = [&](ggml_tensor* t, const string& name) {
            const size_t total = static_cast<size_t>(ggml_nelements(t));
            vector<float> data(total);
            runner.get_output_vec(t, data);
            ofstream f(ref_dir + "/" + name, ios::binary);
            f.write(reinterpret_cast<const char*>(data.data()),
                static_cast<streamsize>(data.size() * sizeof(float)));
            cout << "  Wrote " << name << ": [" << t->ne[1] << ", " << t->ne[0] << "]" << endl;
        };
        dump(out_l9, "cpp_hubert_v1_layer9.raw");
        dump(out_l12, "cpp_hubert_v2_layer12.raw");
        if (out_proj != nullptr) {
            dump(out_proj, "cpp_hubert_v1_finalproj.raw");
        }

        if (dump_debug) {
            for (const auto& d : rvc::hubert::debug_tensors()) {
                dump(d.tensor, "cpp_dbg_" + d.name + ".raw");
            }
        }
        return 0;
    }

    // ------------------------------------------------------------
    // RMVPE self-test: reference input (input_16k.raw) -> mel/salience/f0 -> dump to disk
    // ------------------------------------------------------------
    int run_rmvpe_selftest(const string& model_dir, const string& ref_dir, int device_id,
        const string& input_wav) {
        const string model_path = model_dir + "/rmvpe.gguf";
        if (!utils::file_exists(model_path)) {
            Logger::error("Model not found: " + model_path);
            return 1;
        }
        vector<float> audio;
        if (!input_wav.empty()) {
            // Long-audio window-path smoke test: WAV -> mono 16k
            AudioData wav = load_wav(input_wav);
            audio = to_mono_16k(wav, 16000);
            cout << "  Input WAV: " << input_wav << " -> " << audio.size() << " samples @16k" << endl;
        }
        else {
            const string ref_input = ref_dir + "/input_16k.raw";
            if (!load_raw_f32(ref_input, audio) || audio.empty()) {
                Logger::error("Not found/empty: " + ref_input + " (self-test requires reference tensors, not shipped with the distribution)");
                return 1;
            }
            cout << "  Input samples: " << audio.size() << endl;
        }

        rvc::GgmlDevice device(device_id);
        Logger::info("Compute backend: " + device.name());

        GgufModel model;
        model.load(model_path, device.backend());
        const auto cfg = rvc::rmvpe::read_config(model);
        cout << "  RMVPE: sr=" << cfg.sample_rate << " n_fft=" << cfg.n_fft
            << " hop=" << cfg.hop_length << " mels=" << cfg.n_mels
            << " classes=" << cfg.n_classes << endl;

        rvc::rmvpe::RmvpeExtractor extractor(model, device.backend());

        Timer timer;
        rvc::rmvpe::Result result = extractor.infer(audio, 0.03f);
        cout << "  Inference time: " << timer.elapsed_ms() << " ms" << endl;
        cout << "  Frames: " << result.frames << " (pad " << result.padded_frames << ")" << endl;
        int voiced = 0;
        for (float v : result.f0) {
            if (v > 0.0f) ++voiced;
        }
        cout << "  f0 voiced frames: " << voiced << endl;

        if (!input_wav.empty()) {
            // Long-audio smoke test: dump mel/hidden/f0 without overwriting the 5s reference artifacts
            dump_raw_f32(ref_dir + "/cpp_rmvpe_long_mel.raw", result.mel);
            dump_raw_f32(ref_dir + "/cpp_rmvpe_long_hidden.raw", result.hidden);
            dump_raw_f32(ref_dir + "/cpp_rmvpe_long_f0.raw", result.f0);
            cout << "  Wrote cpp_rmvpe_long_{mel,hidden,f0}.raw" << endl;
            return 0;
        }
        dump_raw_f32(ref_dir + "/cpp_rmvpe_mel.raw", result.mel);
        dump_raw_f32(ref_dir + "/cpp_rmvpe_hidden.raw", result.hidden);
        dump_raw_f32(ref_dir + "/cpp_rmvpe_hidden_padded.raw", result.hidden_padded);
        dump_raw_f32(ref_dir + "/cpp_rmvpe_f0.raw", result.f0);
        cout << "  Wrote cpp_rmvpe_{mel,hidden,hidden_padded,f0}.raw" << endl;
        return 0;
    }

    // ------------------------------------------------------------
    // End-to-end conversion (P5): WAV -> 16k -> pipeline -> WAV
    // ------------------------------------------------------------
    int run_convert(const string& input_wav, const string& output_wav,
        const string& model_path, const string& model_dir,
        const string& index_path, int device_id,
        const rvc::ConvertOptions& opt_in, const rvc::PipelineConfig& pcfg,
        const string& hubert_path_in = {}, const string& rmvpe_path_in = {}) {
        const string hubert_path = hubert_path_in.empty() ? model_dir + "/hubert.gguf" : hubert_path_in;
        const string rmvpe_path = rmvpe_path_in.empty() ? model_dir + "/rmvpe.gguf" : rmvpe_path_in;
        AudioData wav = load_wav(input_wav);
        vector<float> audio = to_mono_16k(wav, 16000);
        cout << "  Input: " << input_wav << " (" << wav.sample_rate << "Hz/" << wav.num_channels
            << "ch) -> 16k " << audio.size() << " samples" << endl;

        rvc::GgmlDevice device(device_id);
        Logger::info("Compute backend: " + device.name());

        GgufModel hubert_model;
        hubert_model.load(hubert_path, device.backend());
        GgufModel rmvpe_model;
        rmvpe_model.load(rmvpe_path, device.backend());
        GgufModel voice_model;
        voice_model.load(model_path, device.backend());

        rvc::ConvertOptions opt = opt_in;
        rvc::RetrievalIndex index;
        if (!index_path.empty() && opt.index_rate != 0.0f) {
            index.load(index_path, device.backend());
            opt.index = &index;
            cout << "  Index: dim=" << index.dim() << " nlist=" << index.nlist()
                << " ntotal=" << index.ntotal() << " mix rate=" << opt.index_rate << endl;
        }

        rvc::RvcPipeline pipeline(hubert_model, rmvpe_model, voice_model, device.backend());
        const auto& vcfg = pipeline.voice_config();
        cout << "  Model: " << vcfg.version << " f0=" << vcfg.has_f0
            << " sr=" << vcfg.sample_rate << " hop=" << vcfg.hop_samples << endl;

        Timer timer;
        rvc::ConvertResult result = pipeline.convert(audio, opt);
        cout << "  Conversion time: " << timer.elapsed_ms() << " ms" << endl;
        cout << "  Output: " << result.pcm.size() << " samples @" << result.sample_rate << endl;

        const size_t slash = output_wav.find_last_of("/\\");
        if (slash != string::npos) {
            utils::ensure_directory(output_wav.substr(0, slash));
        }
        if (!save_wav_i16(output_wav, result.pcm.data(), 1, result.sample_rate,
            static_cast<uint64_t>(result.pcm.size()))) {
            Logger::error("Write failed: " + output_wav);
            return 1;
        }
        Logger::info("Wrote: " + output_wav);
        return 0;
    }

    // ------------------------------------------------------------
    // VITS self-test: reference input (features/pitch/rnd/sine noise) -> full graph -> dump to disk
    // ------------------------------------------------------------
    int run_vits_selftest(const string& model_path, const string& ref_dir, int device_id) {
        if (!utils::file_exists(model_path)) {
            Logger::error("Model not found: " + model_path);
            return 1;
        }
        const string ref_features = ref_dir + "/model_features_upsampled.raw";
        const string ref_pitch = ref_dir + "/model_pitch_coarse_i32.raw";
        const string ref_pitchf = ref_dir + "/model_pitchf.raw";
        const string ref_rnd = ref_dir + "/model_rnd_bct.raw";
        const string ref_sine_noise = ref_dir + "/model_sine_noise.raw";
        for (const auto& p : { ref_features, ref_rnd }) {
            if (!utils::file_exists(p)) {
                Logger::error("Reference data not found: " + p + " (self-test requires reference tensors, not shipped with the distribution)");
                return 1;
            }
        }

        rvc::GgmlDevice device(device_id);
        Logger::info("Compute backend: " + device.name());

        GgufModel model;
        model.load(model_path, device.backend());
        rvc::vits::Synthesizer synth(model, device.backend());
        const auto& cfg = synth.config();
        cout << "  VITS: " << cfg.version << " f0=" << cfg.has_f0
            << " sr=" << cfg.sample_rate << " hop=" << cfg.hop_samples
            << " inter=" << cfg.inter_channels << " heads=" << cfg.n_heads
            << " layers=" << cfg.n_layers << " flow_layers=" << cfg.flow_n_layers << endl;

        vector<float> features;
        if (!load_raw_f32(ref_features, features) || features.empty()) {
            Logger::error("Failed to read features: " + ref_features);
            return 1;
        }
        ggml_tensor* emb_phone = model.weight("enc_p.emb_phone.weight");
        if (!emb_phone) {
            Logger::error("Model is missing enc_p.emb_phone.weight");
            return 1;
        }
        const int feature_dim = static_cast<int>(emb_phone->ne[0]);
        if (features.size() % static_cast<size_t>(feature_dim) != 0) {
            Logger::error("features size does not match the model input dimension (--model and reference data must match)");
            return 1;
        }
        const int64_t frames = static_cast<int64_t>(features.size() / feature_dim);
        cout << "  Features: " << frames << " frames x " << feature_dim << endl;

        vector<int32_t> pitch;
        vector<float> pitchf;
        vector<float> sine;
        if (cfg.has_f0) {
            if (!load_raw_i32(ref_pitch, pitch)) {
                Logger::error("Failed to read pitch: " + ref_pitch);
                return 1;
            }
            pitch.resize(static_cast<size_t>(frames));

            if (!load_raw_f32(ref_pitchf, pitchf) || pitchf.size() < static_cast<size_t>(frames)) {
                Logger::error("Failed to read pitchf: " + ref_pitchf);
                return 1;
            }
            pitchf.resize(static_cast<size_t>(frames));

            // SineGen (injected noise is used to guarantee sample-by-sample alignment with the Python reference)
            vector<float> sine_noise;
            const bool has_noise = load_raw_f32(ref_sine_noise, sine_noise) &&
                sine_noise.size() == static_cast<size_t>(frames) * cfg.hop_samples;
            sine = rvc::sinegen::generate(pitchf, cfg.sample_rate,
                static_cast<int>(cfg.hop_samples), has_noise ? &sine_noise : nullptr, 1234);
            dump_raw_f32(ref_dir + "/cpp_model_sine_source.raw", sine);

            // Merge excitation: tanh(l_linear(sine)) (host side)
            const auto lw = model.read_f32("dec.m_source.l_linear.weight");
            const auto lb = model.read_f32("dec.m_source.l_linear.bias");
            vector<float> merged(sine.size());
            for (size_t i = 0; i < sine.size(); ++i) {
                merged[i] = tanhf(lw[0] * sine[i] + lb[0]);
            }
            dump_raw_f32(ref_dir + "/cpp_model_sine_merged.raw", merged);
            sine = std::move(merged);
        }

        vector<float> noise;
        if (!load_raw_f32(ref_rnd, noise) ||
            noise.size() != static_cast<size_t>(frames) * cfg.inter_channels) {
            Logger::error("Failed to read rnd / size mismatch: " + ref_rnd);
            return 1;
        }

        rvc::vits::Input input;
        input.frames = frames;
        input.feature_dim = feature_dim;
        input.speaker_id = 0;
        input.features = std::move(features);
        input.pitch = std::move(pitch);
        input.source = std::move(sine);
        input.noise = std::move(noise);

        Timer timer;
        rvc::vits::Output out = synth.infer(input);
        cout << "  Inference time: " << timer.elapsed_ms() << " ms" << endl;
        cout << "  Audio samples: " << out.samples << " @" << out.sample_rate << endl;

        dump_raw_f32(ref_dir + "/cpp_model_m_p.raw", out.m_p);
        dump_raw_f32(ref_dir + "/cpp_model_logs_p.raw", out.logs_p);
        dump_raw_f32(ref_dir + "/cpp_model_z_p.raw", out.z_p);
        dump_raw_f32(ref_dir + "/cpp_model_z.raw", out.z);
        dump_raw_f32(ref_dir + "/cpp_model_audio.raw", out.audio);
        cout << "  Wrote cpp_model_{m_p,logs_p,z_p,z,audio}.raw + cpp_model_sine_{source,merged}.raw" << endl;
        return 0;
    }

    // ------------------------------------------------------------
    // Interactive wizard (entered when there are no command-line arguments; exits after one run)
    // ------------------------------------------------------------
    std::string prompt_line(const std::string& label, const std::string& default_value) {
        std::cout << label;
        if (!default_value.empty()) {
            std::cout << " (default " << default_value << ")";
        }
        std::cout << ": ";
        std::cout.flush();
        std::string line;
        if (!std::getline(std::cin, line)) {
            return default_value;
        }
        line = utils::trim(line);
        return line.empty() ? default_value : line;
    }

    int prompt_int(const std::string& label, int default_value) {
        for (int attempt = 0; attempt < 3; ++attempt) {
            const std::string s = prompt_line(label, std::to_string(default_value));
            try {
                size_t pos = 0;
                const int v = std::stoi(s, &pos);
                if (pos == s.size()) {
                    return v;
                }
            }
            catch (...) {
            }
            std::cout << "  Invalid input, please enter an integer." << std::endl;
        }
        return default_value;
    }

    float prompt_float(const std::string& label, float default_value) {
        for (int attempt = 0; attempt < 3; ++attempt) {
            char buf[64];
            snprintf(buf, sizeof(buf), "%g", default_value);
            const std::string s = prompt_line(label, buf);
            try {
                size_t pos = 0;
                const float v = std::stof(s, &pos);
                if (pos == s.size()) {
                    return v;
                }
            }
            catch (...) {
            }
            std::cout << "  Invalid input, please enter a number." << std::endl;
        }
        return default_value;
    }

    struct VoiceModelEntry {
        std::string path;
        std::string file;
        std::string version;
        int sample_rate = 0;
        bool has_f0 = true;
        int n_spk = 0;
        std::string quant;
        std::vector<std::string> speaker_names;
    };

    std::string file_type_name(int id) {
        switch (id) {
        case 0: return "F32";
        case 1: return "F16";
        case 2: return "Q4_0";
        case 3: return "Q4_1";
        case 7: return "Q8_0";
        case 8: return "Q5_0";
        case 9: return "Q5_1";
        default: return "-";
        }
    }

    bool read_voice_model_entry(const std::string& dir, const std::string& file,
        VoiceModelEntry& e) {
        const std::string path = dir + "/" + file;
        try {
            GgufModel m;
            m.load_meta_only(path);
            const std::string version = m.meta_str("rvc.version", "");
            if (version.empty()) {
                return false;   // not a character model (hubert/rmvpe/index, etc.)
            }
            e.path = path;
            e.file = file;
            e.version = version;
            e.sample_rate = m.meta_u32("rvc.sample_rate", 40000);
            e.has_f0 = m.meta_bool("rvc.has_f0", true);
            e.n_spk = m.meta_u32("rvc.n_spk", 1);
            e.speaker_names = m.meta_array_str("rvc.speaker_names");
            e.quant = file_type_name(m.meta_u32("general.file_type", 1));
            return true;
        }
        catch (...) {
            return false;
        }
    }

    // Look for a retrieval index with the same name as the character model under model_dir (strip the quantization suffix)
    std::string find_index_for(const std::string& model_dir, const std::string& model_file) {
        const std::string stem = utils::stem(model_file);
        std::vector<std::string> candidates;
        const size_t dot = stem.find('.');
        if (dot != std::string::npos) {
            candidates.push_back(stem.substr(0, dot));
        }
        candidates.push_back(stem);
        for (const auto& base : candidates) {
            const std::string path = model_dir + "/" + base + ".index.gguf";
            if (utils::file_exists(path)) {
                return path;
            }
        }
        return {};
    }

    // Quant levels (fp16 = base file with no suffix)
    const std::vector<std::string>& quant_levels() {
        static const std::vector<std::string> k = { "fp16", "q8_0", "q5_1", "q4_1", "q4_0" };
        return k;
    }

    // Strip the quantization suffix from the filename (guanguanV1.q8_0.gguf -> guanguanV1.gguf)
    std::string strip_quant_suffix(const std::string& path) {
        std::string out = path;
        if (out.size() >= 5 && out.compare(out.size() - 5, 5, ".gguf") == 0) {
            out = out.substr(0, out.size() - 5);
        }
        for (const auto& q : quant_levels()) {
            if (q == "fp16") continue;
            const std::string suffix = "." + q;
            if (out.size() > suffix.size() &&
                out.compare(out.size() - suffix.size(), suffix.size(), suffix) == 0) {
                out = out.substr(0, out.size() - suffix.size());
                break;
            }
        }
        if (path.size() >= 5 && path.compare(path.size() - 5, 5, ".gguf") == 0) {
            out += ".gguf";
        }
        return out;
    }

    std::string with_quant_suffix(const std::string& path, const std::string& quant) {
        if (quant.empty() || quant == "fp16" || quant == "f16") {
            return path;
        }
        const std::string base = strip_quant_suffix(path);
        if (base.size() >= 5 && base.compare(base.size() - 5, 5, ".gguf") == 0) {
            return base.substr(0, base.size() - 5) + "." + quant + ".gguf";
        }
        return base + "." + quant;
    }

    // Detect the quant level from the filename (guanguanV1.q4_0.gguf -> q4_0; no suffix -> fp16)
    std::string detect_quant(const std::string& file) {
        const std::string stem = utils::stem(file);
        for (const auto& q : quant_levels()) {
            if (q == "fp16") continue;
            const std::string suffix = "." + q;
            if (stem.size() > suffix.size() &&
                stem.compare(stem.size() - suffix.size(), suffix.size(), suffix) == 0) {
                return q;
            }
        }
        return "fp16";
    }

    int run_interactive(const std::string& model_dir, const std::string& input_dir,
        int default_device) {
        // ---------- 1) Character model ----------
        auto files = utils::list_files(model_dir, ".gguf");
        std::vector<VoiceModelEntry> models;
        for (const auto& f : files) {
            if (f.find(".index.") != std::string::npos) {
                continue;
            }
            VoiceModelEntry e;
            if (read_voice_model_entry(model_dir, f, e)) {
                models.push_back(e);
            }
        }
        if (models.empty()) {
            Logger::error("No character model (.gguf) found under models/! Convert one with scripts/export_gguf.py first.");
            return 1;
        }
        std::cout << "Available character models (" << model_dir << "/):" << std::endl;
        for (size_t i = 0; i < models.size(); ++i) {
            const auto& m = models[i];
            std::cout << "  " << std::setw(3) << (i + 1) << ". " << std::left << std::setw(34)
                << m.file << std::right << m.version << " " << (m.sample_rate / 1000) << "kHz"
                << (m.has_f0 ? " f0" : " non-f0")
                << " spk=" << m.n_spk
                << (m.quant.empty() || m.quant == "-" ? "" : (" [" + m.quant + "]"))
                << std::endl;
        }
        std::cout << std::endl;

        std::string sel = prompt_line("Select a character model (index or filename)", "1");
        VoiceModelEntry model;
        {
            bool ok = false;
            if (!sel.empty() && sel.find_first_not_of("0123456789") == std::string::npos) {
                const int idx = std::stoi(sel);
                if (idx >= 1 && idx <= static_cast<int>(models.size())) {
                    model = models[static_cast<size_t>(idx - 1)];
                    ok = true;
                }
            }
            else {
                if (sel.size() < 5 || sel.substr(sel.size() - 5) != ".gguf") {
                    sel += ".gguf";
                }
                for (const auto& m : models) {
                    if (m.file == sel) {
                        model = m;
                        ok = true;
                        break;
                    }
                }
            }
            if (!ok) {
                Logger::error("Invalid model selection: " + sel);
                return 1;
            }
        }

        // ---------- 1.5) Quant level (shared by hubert/rmvpe/character model) ----------
        std::vector<std::string> available_levels;
        for (const auto& q : quant_levels()) {
            const std::string h = q == "fp16" ? model_dir + "/hubert.gguf"
                : model_dir + "/hubert." + q + ".gguf";
            const std::string r = q == "fp16" ? model_dir + "/rmvpe.gguf"
                : model_dir + "/rmvpe." + q + ".gguf";
            if (utils::file_exists(h) && utils::file_exists(r)) {
                available_levels.push_back(q);
            }
        }
        std::string quant = detect_quant(model.file);
        bool quant_ok = false;
        for (const auto& q : available_levels) {
            if (q == quant) quant_ok = true;
        }
        if (!quant_ok) {
            quant = available_levels.empty() ? "fp16" : available_levels.front();
        }
        if (available_levels.size() > 1) {
            std::cout << "Available quant levels: ";
            for (size_t i = 0; i < available_levels.size(); ++i) {
                std::cout << (i ? ", " : "") << available_levels[i];
            }
            std::cout << std::endl;
            const std::string sel_q = prompt_line("Select a quant level (shared by hubert/rmvpe/character model)", quant);
            bool valid = false;
            for (const auto& q : available_levels) {
                if (q == sel_q) {
                    quant = q;
                    valid = true;
                }
            }
            if (!valid) {
                Logger::warn("Quant level " + sel_q + " not found, using " + quant);
            }
        }
        std::string hubert_path = with_quant_suffix(model_dir + "/hubert.gguf", quant);
        std::string rmvpe_path = with_quant_suffix(model_dir + "/rmvpe.gguf", quant);
        std::string voice_path = model.path;
        if (quant != "fp16" && detect_quant(model.file) != quant) {
            const std::string candidate = with_quant_suffix(model.path, quant);
            if (utils::file_exists(candidate)) {
                voice_path = candidate;
            }
            else {
                Logger::warn("Character model quant level file not found: " + candidate + ", falling back to " + model.path);
            }
        }
        std::cout << std::endl;

        // ---------- 2) Input audio ----------
        auto wavs = utils::list_files(input_dir, ".wav");
        std::cout << "Available input audio (" << input_dir << "/):" << std::endl;
        for (size_t i = 0; i < wavs.size(); ++i) {
            std::cout << "  " << std::setw(3) << (i + 1) << ". " << wavs[i] << std::endl;
        }
        std::cout << std::endl;
        const std::string input_default = wavs.empty() ? "" : "1";
        std::string input_sel = prompt_line("Select input audio (index or path)", input_default);
        std::string input_wav;
        if (!input_sel.empty() && input_sel.find_first_not_of("0123456789") == std::string::npos) {
            const int idx = std::stoi(input_sel);
            if (idx >= 1 && idx <= static_cast<int>(wavs.size())) {
                input_wav = input_dir + "/" + wavs[static_cast<size_t>(idx - 1)];
            }
        }
        else {
            input_wav = input_sel;
        }
        if (input_wav.empty() || !utils::file_exists(input_wav)) {
            Logger::error("Input audio does not exist: " + input_wav);
            return 1;
        }

        // ---------- 3) Device ----------
        const auto gpus = rvc::list_gpu_devices();
        std::cout << std::endl << "Available inference devices:" << std::endl;
        for (const auto& d : gpus) {
            std::cout << "  " << std::setw(3) << d.index << ". " << d.name
                << "  (" << d.description << ")" << std::endl;
        }
        std::cout << "   99. CPU (software fallback)" << std::endl;
        const int device_default = gpus.empty() ? 99 : default_device;
        const int device_id = prompt_int("Select a device", device_default);

        // ---------- 4) Conversion parameters ----------
        std::cout << std::endl;
        rvc::ConvertOptions opt;
        opt.f0_up_key = prompt_int("Pitch shift in semitones", 0);
        if (model.n_spk > 1) {
            std::string names;
            for (size_t i = 0; i < model.speaker_names.size() && i < 8; ++i) {
                if (!model.speaker_names[i].empty()) {
                    names += (names.empty() ? "" : ", ") + std::to_string(i) + "=" + model.speaker_names[i];
                }
            }
            if (!names.empty()) {
                std::cout << "  Available speakers: " << names << std::endl;
            }
            opt.sid = prompt_int("Speaker ID", 0);
        }

        const std::string index_path = find_index_for(model_dir, model.file);
        std::string use_index;
        if (!index_path.empty()) {
            std::cout << "  Retrieval index found: " << index_path << std::endl;
            use_index = prompt_line("Index mix rate 0~1 (0 = disabled)", "0");
            opt.index_rate = std::stof(use_index);
            if (opt.index_rate < 0.0f) opt.index_rate = 0.0f;
            if (opt.index_rate > 1.0f) opt.index_rate = 1.0f;
        }

        opt.rms_mix_rate = prompt_float("RMS mix rate 0~1 (1 = no mixing)", 0.75f);
        opt.protect = prompt_float("Unvoiced protection 0~0.5", 0.33f);

        const std::string adv = prompt_line("Modify advanced parameters (resampling, etc.)? (y/N)", "n");
        int resample_sr = 0;
        if (!adv.empty() && (adv[0] == 'y' || adv[0] == 'Y')) {
            resample_sr = prompt_int("Output resample rate (0 = model sample rate)", 0);
        }
        opt.resample_sr = resample_sr;

        // Output path: output/<input name>_<model short name>[_<quant level>].wav
        std::string model_tag = utils::stem(model.file);
        if (quant != "fp16") {
            model_tag = utils::stem(strip_quant_suffix(model.file));
            model_tag += "_" + quant;
        }
        const std::string output_wav = "output/" + utils::stem(input_wav) + "_" + model_tag + ".wav";

        // ---------- 5) Confirm ----------
        std::cout << std::endl;
        std::cout << "----------------------------------------" << std::endl;
        std::cout << "  Model:         " << voice_path << "  (" << model.version
            << " " << (model.sample_rate / 1000) << "kHz"
            << (model.has_f0 ? " f0" : " non-f0") << ")" << std::endl;
        std::cout << "  Quant level:   " << quant << " (hubert: " << hubert_path
            << ", rmvpe: " << rmvpe_path << ")" << std::endl;
        std::cout << "  Input audio:   " << input_wav << std::endl;
        std::cout << "  Output file:   " << output_wav << std::endl;
        std::cout << "  Device:        " << (device_id == 99 ? "CPU" : ("GPU " + std::to_string(device_id))) << std::endl;
        std::cout << "  Pitch shift:   " << opt.f0_up_key << " semitones";
        if (opt.sid != 0) std::cout << ", speaker " << opt.sid;
        if (opt.index_rate > 0.0f) std::cout << ", index mix " << opt.index_rate;
        std::cout << ", RMS mix " << opt.rms_mix_rate << ", protect " << opt.protect;
        if (opt.resample_sr > 0) std::cout << ", resample " << opt.resample_sr << "Hz";
        std::cout << std::endl;
        std::cout << "----------------------------------------" << std::endl;
        const std::string go = prompt_line("Press Enter to start conversion (type n to cancel)", "");
        if (!go.empty() && (go[0] == 'n' || go[0] == 'N')) {
            std::cout << "Canceled." << std::endl;
            return 0;
        }

        rvc::PipelineConfig pcfg;
        std::cout << std::endl;
        const int rc = run_convert(input_wav, output_wav, voice_path, model_dir,
            index_path, device_id, opt, pcfg, hubert_path, rmvpe_path);
        if (rc == 0) {
            std::cout << std::endl << "Conversion complete: " << output_wav << std::endl;
        }
        return rc;
    }

} // namespace

int main(int argc, char* argv[]) {
#ifdef _WIN32
    SetConsoleOutputCP(65001);
    SetConsoleCP(65001);
#endif

    // On Intel drivers, the Vulkan integer dot-product path may produce NaN/zero output when a quantized model runs the graph repeatedly
    // (observed in testing; reproduced on both ggml 0.18 and 0.23); disabled by default, re-enable with RVC_VK_ENABLE_INTDOT=1.
    if (!getenv("RVC_VK_ENABLE_INTDOT")) {
        utils::set_env("GGML_VK_DISABLE_INTEGER_DOT_PRODUCT", "1");
    }

    // The Vulkan fp16 compute path (including fp16 accumulation in im2col/GEMM) significantly amplifies error:
    // In P4 testing, audio SNR dropped from ~60dB to ~33dB (corr still meets spec, but perceptual risk increases),
    // and on some Intel integrated GPUs there is almost no speed difference; disabled by default, re-enable with RVC_VK_ENABLE_F16=1.
    if (!getenv("RVC_VK_ENABLE_F16")) {
        utils::set_env("GGML_VK_DISABLE_F16", "1");
    }

    string input_wav;
    string output_wav = "output/converted.wav";
    string ref_dir = "output/ref";
    string model_dir = "models";
    string model_path = "models/guanguanV1.gguf";
    string hubert_arg;   // --hubert: explicit feature-extractor GGUF (overrides model-dir)
    string rmvpe_arg;    // --rmvpe: explicit f0-extractor GGUF (overrides model-dir)
    string index_path;
    string noise_dir;
    string quant;
    bool audio_selftest = false;
    bool hubert_selftest = false;
    bool rmvpe_selftest = false;
    bool vits_selftest = false;
    bool list_devices = false;
    bool interactive = false;       // enter the wizard automatically when no action argument is given
    bool interactive_flag = false;  // --interactive forces it
    bool action_given = false;
    int device_id = 99;
    int split_query = 5, split_center = 30, split_max = 32;
    rvc::ConvertOptions convert_opt;
    rvc::PipelineConfig pipeline_cfg;

    for (int i = 1; i < argc; ++i) {
        const string a = argv[i];
        if (a == "--input" && i + 1 < argc) {
            input_wav = argv[++i];
            action_given = true;
        }
        else if (a == "--ref-dir" && i + 1 < argc) {
            ref_dir = argv[++i];
        }
        else if (a == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        }
        else if (a == "--hubert" && i + 1 < argc) {
            hubert_arg = argv[++i];
        }
        else if (a == "--rmvpe" && i + 1 < argc) {
            rmvpe_arg = argv[++i];
        }
        else if (a == "--vits-selftest") {
            vits_selftest = true;
            action_given = true;
        }
        else if (a == "--output" && i + 1 < argc) {
            output_wav = argv[++i];
        }
        else if (a == "--index" && i + 1 < argc) {
            index_path = argv[++i];
        }
        else if (a == "--index-rate" && i + 1 < argc) {
            convert_opt.index_rate = stof(argv[++i]);
        }
        else if (a == "--pitch" && i + 1 < argc) {
            convert_opt.f0_up_key = stoi(argv[++i]);
        }
        else if (a == "--proposed-pitch" && i + 1 < argc) {
            convert_opt.proposed_pitch = true;
            convert_opt.proposed_pitch_threshold = stof(argv[++i]);
        }
        else if (a == "--sid" && i + 1 < argc) {
            convert_opt.sid = stoi(argv[++i]);
        }
        else if (a == "--protect" && i + 1 < argc) {
            convert_opt.protect = stof(argv[++i]);
        }
        else if (a == "--rms-mix-rate" && i + 1 < argc) {
            convert_opt.rms_mix_rate = stof(argv[++i]);
        }
        else if (a == "--resample-sr" && i + 1 < argc) {
            convert_opt.resample_sr = stoi(argv[++i]);
        }
        else if (a == "--noise-dir" && i + 1 < argc) {
            noise_dir = argv[++i];
        }
        else if (a == "--f0-inject") {
            convert_opt.f0_inject = true;
        }
        else if (a == "--list-devices") {
            list_devices = true;
            action_given = true;
        }
        else if (a == "--x-pad" && i + 1 < argc) {
            pipeline_cfg.x_pad = stoi(argv[++i]);
        }
        else if (a == "--x-query" && i + 1 < argc) {
            pipeline_cfg.x_query = stoi(argv[++i]);
        }
        else if (a == "--x-center" && i + 1 < argc) {
            pipeline_cfg.x_center = stoi(argv[++i]);
        }
        else if (a == "--x-max" && i + 1 < argc) {
            pipeline_cfg.x_max = stoi(argv[++i]);
        }
        else if (a == "--model-dir" && i + 1 < argc) {
            model_dir = argv[++i];
        }
        else if (a == "--quant" && i + 1 < argc) {
            quant = argv[++i];
        }
        else if (a == "--device" && i + 1 < argc) {
            device_id = stoi(argv[++i]);
        }
        else if (a == "--audio-selftest") {
            audio_selftest = true;
            action_given = true;
        }
        else if (a == "--hubert-selftest") {
            hubert_selftest = true;
            action_given = true;
        }
        else if (a == "--rmvpe-selftest") {
            rmvpe_selftest = true;
            action_given = true;
        }
        else if (a == "--split-query" && i + 1 < argc) {
            split_query = stoi(argv[++i]);
        }
        else if (a == "--split-center" && i + 1 < argc) {
            split_center = stoi(argv[++i]);
        }
        else if (a == "--split-max" && i + 1 < argc) {
            split_max = stoi(argv[++i]);
        }
        else if (a == "--interactive" || a == "-i") {
            interactive_flag = true;
        }
        else if (a == "--help" || a == "-h") {
            print_usage();
            return 0;
        }
    }
    interactive = interactive_flag || !action_given;

    cout << "================================================" << endl;
    cout << "  RVC-GGML - ggml RVC voice conversion" << endl;
    cout << "  Hardware-accelerated inference (Vulkan / CUDA / CPU)" << endl;
    cout << "================================================" << endl;
    cout << endl;

    try {
        if (audio_selftest) {
            if (input_wav.empty()) {
                Logger::error("--audio-selftest requires --input <wav>");
                return 1;
            }
            return run_audio_selftest(input_wav, "output/selftest", split_query, split_center, split_max);
        }
        if (hubert_selftest) {
            return run_hubert_selftest(model_dir, ref_dir, device_id);
        }
        if (rmvpe_selftest) {
            return run_rmvpe_selftest(model_dir, ref_dir, device_id, input_wav);
        }
        if (vits_selftest) {
            return run_vits_selftest(model_path, ref_dir, device_id);
        }
        if (list_devices) {
            const auto devices = rvc::list_gpu_devices();
            cout << "Available GPU devices (--device <id>):" << endl;
            for (const auto& d : devices) {
                cout << "  " << d.index << ": " << d.name << "  (" << d.description << ")" << endl;
            }
            cout << "  CPU fallback: --device 99" << endl;
            return 0;
        }
        if (!input_wav.empty()) {
            convert_opt.noise_dir = noise_dir;
            string hubert_path;
            string rmvpe_path;
            string voice_path = model_path;
            if (!quant.empty() && quant != "fp16" && quant != "f16") {
                hubert_path = with_quant_suffix(model_dir + "/hubert.gguf", quant);
                rmvpe_path = with_quant_suffix(model_dir + "/rmvpe.gguf", quant);
                voice_path = with_quant_suffix(model_path, quant);
                for (const auto& p : { hubert_path, rmvpe_path, voice_path }) {
                    if (!utils::file_exists(p)) {
                        Logger::error("Quant level file does not exist: " + p);
                        return 1;
                    }
                }
                Logger::info("Quant level: " + quant);
            }
            // Explicit extractor overrides win over the model-dir/quant defaults.
            if (!hubert_arg.empty()) {
                hubert_path = hubert_arg;
            }
            if (!rmvpe_arg.empty()) {
                rmvpe_path = rmvpe_arg;
            }
            return run_convert(input_wav, output_wav, voice_path, model_dir,
                index_path, device_id, convert_opt, pipeline_cfg, hubert_path, rmvpe_path);
        }

        if (interactive) {
            const int rc = run_interactive(model_dir, "input", device_id == 99 ? 0 : device_id);
            cout << endl << "Press Enter to exit...";
            string dummy;
            getline(cin, dummy);
            return rc;
        }

        print_usage();
        return 0;
    }
    catch (const exception& e) {
        Logger::error("Error: " + string(e.what()));
        return 1;
    }
}
