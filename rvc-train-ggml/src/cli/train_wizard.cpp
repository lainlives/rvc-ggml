// RVC-Train-GGML - interactive training wizard implementation (src/cli/train_wizard.cpp)
//
// See train_wizard.h and `handover.md` §4.11/§4.13. Supports both finetune and from-scratch (M7-2) branches.
// M8-1a: device selection in list form (traincpp device API) + Q31 large-config warning.
// M8-1b: dual-mode data entry - existing cache directory / raw audio directory (spawn an rvc-prep.exe subprocess to preprocess).
#include "cli/train_wizard.h"

#include "common/platform.h"
#include "prep/cache.h"
#include "traincpp/traincpp.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <system_error>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/wait.h>
#endif

namespace rvc_train {
namespace train {
namespace {

std::string trim(const std::string& s) {
    size_t b = 0;
    size_t e = s.size();
    while (b < e && (s[b] == ' ' || s[b] == '\t' || s[b] == '\r' || s[b] == '\n')) {
        ++b;
    }
    while (e > b && (s[e - 1] == ' ' || s[e - 1] == '\t' || s[e - 1] == '\r' || s[e - 1] == '\n')) {
        --e;
    }
    return s.substr(b, e - b);
}

// Read one answer line; returns false at end of stream.
bool read_answer(std::istream& in, std::string* out) {
    if (!std::getline(in, *out)) {
        return false;
    }
    *out = trim(*out);
    return true;
}

// Print a prompt and read one line: blank line/EOF uses the default.
std::string ask(const std::string& label, const std::string& def, std::istream& in,
                std::ostream& out, bool* eof) {
    out << label;
    if (!def.empty()) {
        out << " [" << def << "]";
    }
    out << ": ";
    out.flush();
    std::string line;
    if (!read_answer(in, &line)) {
        *eof = true;
        out << "\n";
        return def;
    }
    return line.empty() ? def : line;
}

std::string ask_nonempty(const std::string& label, const std::string& def, std::istream& in,
                         std::ostream& out, bool* eof) {
    std::string cur = def;
    for (;;) {
        const std::string s = ask(label, cur, in, out, eof);
        if (!s.empty()) {
            return s;
        }
        if (*eof) {
            return s;
        }
        out << "  Cannot be empty, please retry.\n";
    }
}

std::string ask_existing_file(const std::string& label, const std::string& def, std::istream& in,
                              std::ostream& out, bool* eof) {
    std::string cur = def;
    for (;;) {
        const std::string s = ask(label, cur, in, out, eof);
        std::error_code ec;
        if (!s.empty() && std::filesystem::is_regular_file(s, ec)) {
            return s;
        }
        if (*eof) {
            return s;
        }
        out << "  File does not exist: " << s << ", please retry.\n";
        cur = s;
    }
}

std::string ask_cache_dir(const std::string& label, const std::string& def, std::istream& in,
                          std::ostream& out, bool* eof) {
    std::string cur = def;
    for (;;) {
        const std::string s = ask(label, cur, in, out, eof);
        std::error_code ec;
        const std::filesystem::path p(s);
        if (!s.empty() && std::filesystem::is_directory(p, ec) &&
            std::filesystem::is_regular_file(p / "dataset.json", ec)) {
            return s;
        }
        if (*eof) {
            return s;
        }
        out << "  Invalid cache directory (must exist and contain dataset.json): " << s << ", please retry.\n";
        cur = s;
    }
}

// M8-1b: existing directory (input audio directory, etc.).
std::string ask_existing_dir(const std::string& label, const std::string& def, std::istream& in,
                             std::ostream& out, bool* eof) {
    std::string cur = def;
    for (;;) {
        const std::string s = ask(label, cur, in, out, eof);
        std::error_code ec;
        if (!s.empty() && std::filesystem::is_directory(s, ec)) {
            return s;
        }
        if (*eof) {
            return s;
        }
        out << "  Directory does not exist: " << s << ", please retry.\n";
        cur = s;
    }
}

// M8-1b: optional file - blank line means leave empty (e.g. an empty RMVPE skips f0); non-empty must be an existing file.
std::string ask_optional_file(const std::string& label, const std::string& def, std::istream& in,
                              std::ostream& out, bool* eof) {
    std::string cur = def;
    for (;;) {
        const std::string s = ask(label, cur, in, out, eof);
        if (s.empty() || *eof) {
            return s;
        }
        std::error_code ec;
        if (std::filesystem::is_regular_file(s, ec)) {
            return s;
        }
        out << "  File does not exist: " << s << ", please retry (press Enter to leave empty).\n";
        cur = s;
    }
}

int64_t ask_int(const std::string& label, int64_t def, int64_t min_v, int64_t max_v,
                std::istream& in, std::ostream& out, bool* eof) {
    int64_t cur = def;
    for (;;) {
        const std::string s = ask(label, std::to_string((long long) cur), in, out, eof);
        if (!s.empty()) {
            char* end = nullptr;
            const long long v = std::strtoll(s.c_str(), &end, 10);
            if (end != s.c_str() && *end == '\0' && v >= min_v && v <= max_v) {
                return (int64_t) v;
            }
        } else if (*eof) {
            return cur;
        } else {
            return cur;  // blank line uses the default (already displayed)
        }
        if (*eof) {
            return cur;
        }
        out << "  Please enter an integer in range " << min_v << ".." << max_v << ", please retry.\n";
        cur = def;
    }
}

float ask_float(const std::string& label, float def, std::istream& in, std::ostream& out,
                bool* eof) {
    char defbuf[64];
    std::snprintf(defbuf, sizeof(defbuf), "%.*g", 9, (double) def);
    float cur = def;
    for (;;) {
        const std::string s = ask(label, defbuf, in, out, eof);
        if (!s.empty()) {
            char* end = nullptr;
            const double v = std::strtod(s.c_str(), &end);
            if (end != s.c_str() && *end == '\0' && v > 0.0) {
                return (float) v;
            }
        } else {
            return cur;
        }
        if (*eof) {
            return cur;
        }
        out << "  Please enter a value > 0, please retry.\n";
    }
}

void print_summary(const TrainerConfig& c, std::ostream& out) {
    out << "\n---- Config summary ----\n";
    out << "  Data cache   : " << c.exp_dir << "\n";
    if (c.scratch) {
        out << "  Model source : train from scratch (randomly initialize G/D per config)\n";
    } else {
        out << "  Pretrained G : " << c.weights_g << "\n";
        out << "  Pretrained D : " << c.weights_d << "\n";
    }
    out << "  Output dir   : " << c.out_dir << "\n";
    out << "  epochs       : " << (long long) c.epochs << "\n";
    out << "  batch_size   : " << (long long) c.batch_size << "\n";
    out << "  bucket       : " << (long long) c.bucket << "\n";
    out << "  disc_len     : " << (long long) c.disc_len << " (0=whole segment_size)\n";
    out << "  learning_rate: " << c.lr << "\n";
    out << "  seed         : " << (long long) c.seed << "\n";
    out << "  device       : " << (long long) c.device << " (-1=auto)\n";
    if (!c.resume.empty()) {
        out << "  Resume from  : " << c.resume << "\n";
    }
    out << "------------------\n";
}

bool confirm_yes(const std::string& yn) {
    return yn == "y" || yn == "Y" || yn == "yes" || yn == "YES" || yn == "是";
}

// M8-1b: directory of this process's exe (used to locate the rvc-prep executable in the same directory).
std::string current_exe_dir() {
#ifdef _WIN32
    wchar_t buf[MAX_PATH];
    const DWORD n = GetModuleFileNameW(nullptr, buf, MAX_PATH);
    if (n == 0 || n >= MAX_PATH) {
        return {};
    }
    const std::filesystem::path p(buf);
#else
    std::error_code link_ec;
    const std::filesystem::path p = std::filesystem::read_symlink("/proc/self/exe", link_ec);
    if (link_ec || p.empty()) {
        return {};
    }
#endif
    std::error_code ec;
    const std::filesystem::path dir = p.parent_path();
    return std::filesystem::is_directory(dir, ec) ? dir.string() : std::string();
}

// M8-1b: concatenate directory + file name using the native separator.
std::string join_simple(const std::string& dir, const std::string& name) {
    if (dir.empty()) {
        return name;
    }
    const char last = dir.back();
    if (last == '/' || last == '\\') {
        return dir + name;
    }
    return dir + platform::path_sep() + name;
}

#ifdef _WIN32
constexpr const char* kPrepExeName = "rvc-prep.exe";
#else
constexpr const char* kPrepExeName = "rvc-prep";
#endif

// M8-1b: locate the rvc-prep executable - prefer this exe's directory, then the working directory; return empty if not found.
std::string find_rvc_prep_exe(std::ostream& out) {
    std::error_code ec;
    const std::string candidates[2] = {
        join_simple(current_exe_dir(), kPrepExeName),
        std::string(kPrepExeName),
    };
    for (const std::string& c : candidates) {
        if (std::filesystem::is_regular_file(c, ec)) {
            return c;
        }
    }
    out << "  " << kPrepExeName << " not found (tried this program's directory and the current directory).\n";
    return {};
}

// M8-1b: build a quoted command fragment.
std::string quoted_arg(const std::string& s) {
    return "\"" + s + "\"";
}

// M8-1c: find the newest resumable checkpoint in the output directory (train*.ckpt.gguf with a sibling .meta.json present).
// pick the newest by file modification time; return empty if none.
std::string find_latest_checkpoint(const std::string& out_dir) {
    std::error_code ec;
    const std::filesystem::path dir(out_dir);
    if (out_dir.empty() || !std::filesystem::is_directory(dir, ec)) {
        return {};
    }
    std::filesystem::path best;
    std::filesystem::file_time_type best_time{};
    for (const std::filesystem::directory_entry& e : std::filesystem::directory_iterator(dir, ec)) {
        if (!e.is_regular_file(ec)) {
            continue;
        }
        const std::filesystem::path p = e.path();
        const std::string name = p.filename().string();
        if (name.rfind("train", 0) != 0 ||
            name.rfind(".ckpt.gguf") != name.size() - std::string(".ckpt.gguf").size()) {
            continue;
        }
        // sibling meta naming matches resume_meta_path_for: <ckpt>.gguf -> <ckpt>.meta.json
        const std::string meta_name = name.substr(0, name.size() - std::string(".gguf").size()) +
                                      ".meta.json";
        if (!std::filesystem::is_regular_file(dir / meta_name, ec)) {
            continue;
        }
        const std::filesystem::file_time_type t = e.last_write_time(ec);
        if (best.empty() || t > best_time) {
            best = p;
            best_time = t;
        }
    }
    return best.empty() ? std::string() : best.string();
}

// M8-1b: spawn an rvc-prep subprocess to run preprocessing (the two Vulkan backends live in separate processes, per the isolation rule).
// Windows: CreateProcessW (avoids cmd quote-parsing issues and natively supports non-ASCII paths); POSIX: std::system.
// inherits the console for live output.
// success requires exit code 0 and a dataset.json appearing in the cache.
bool run_prep_subprocess(const std::string& exe, const std::string& input, int sr,
                         const std::string& version, const std::string& rmvpe,
                         const std::string& hubert, const std::string& exp_dir,
                         std::string* err) {
    std::string cmd = quoted_arg(exe) + " --input " + quoted_arg(input) + " --sr " +
                      std::to_string(sr) + " --version " + version + " --exp " +
                      quoted_arg(exp_dir);
    if (!rmvpe.empty()) {
        cmd += " --rmvpe " + quoted_arg(rmvpe);
    }
    if (!hubert.empty()) {
        cmd += " --hubert " + quoted_arg(hubert);
    }
    std::printf("[rvc-train] running preprocessing (may take a while):\n  %s\n", cmd.c_str());
    std::fflush(stdout);

#ifdef _WIN32
    // UTF-8 -> UTF-16 (required when the command line contains non-ASCII paths)
    const int wlen = MultiByteToWideChar(CP_UTF8, 0, cmd.c_str(), -1, nullptr, 0);
    if (wlen <= 0) {
        if (err != nullptr) {
            *err = "failed to convert the command line to wide chars";
        }
        return false;
    }
    std::wstring wcmd((size_t) wlen, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, cmd.c_str(), -1, wcmd.data(), wlen);

    STARTUPINFOW si{};
    si.cb = sizeof(si);
    PROCESS_INFORMATION pi{};
    if (!CreateProcessW(nullptr, wcmd.data(), nullptr, nullptr, FALSE, 0, nullptr, nullptr, &si,
                        &pi)) {
        if (err != nullptr) {
            *err = "failed to start " + std::string(kPrepExeName) + " (GetLastError=" + std::to_string(GetLastError()) + ")";
        }
        return false;
    }
    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD exit_code = 1;
    GetExitCodeProcess(pi.hProcess, &exit_code);
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    const int exit_code_plain = static_cast<int>(exit_code);
#else
    // POSIX: spawn through the shell and wait; WEXITSTATUS decodes the wait status.
    const int status = std::system(cmd.c_str());
    const int exit_code_plain = (status == -1) ? -1 : (WIFEXITED(status) ? WEXITSTATUS(status) : 1);
#endif
    if (exit_code_plain != 0) {
        if (err != nullptr) {
            *err = std::string(kPrepExeName) + " exit code " + std::to_string(exit_code_plain) + " (preprocessing failed, see the log above)";
        }
        return false;
    }
    std::error_code ec;
    if (!std::filesystem::is_regular_file(std::filesystem::path(exp_dir) / "dataset.json", ec)) {
        if (err != nullptr) {
            *err = "preprocessing command succeeded but did not produce " + exp_dir + platform::path_sep() + "dataset.json";
        }
        return false;
    }
    return true;
}

const char* wizard_device_type_name(traincpp::DeviceType t) {
    using traincpp::DeviceType;
    switch (t) {
        case DeviceType::CPU:   return "CPU";
        case DeviceType::GPU:   return "GPU";
        case DeviceType::IGPU:  return "IGPU";
        case DeviceType::ACCEL: return "ACCEL";
    }
    return "?";
}

// M8-1a: print the traincpp device list (with VRAM/unified memory) in the wizard as a reference for device selection.
void print_wizard_devices(std::ostream& out) {
    const size_t n = traincpp::device_count();
    out << "\nAvailable training devices (traincpp):\n";
    if (n == 0) {
        out << "  (no devices enumerated)\n";
        return;
    }
    for (size_t i = 0; i < n; ++i) {
        const traincpp::Device* d = traincpp::device_get(i);
        const traincpp::DeviceProps& p = d->props();
        out << "  [" << i << "] " << p.name << " | " << p.description << " | "
            << wizard_device_type_name(p.type) << " | free "
            << (double) p.memory_free / (1024.0 * 1024.0) << " MiB / total "
            << (double) p.memory_total / (1024.0 * 1024.0) << " MiB"
            << (p.unified_memory ? " | unified memory" : "") << "\n";
    }
    out << "  -1 = auto (GPU -> IGPU -> CPU)\n";
}

// M8-1a: Q31 framing - measured-cost warning for large bucket/large disc_len (a user's discrete GPU crashed after only 5 epochs in 7 hours).
void warn_large_config(const TrainerConfig& c, std::ostream& out) {
    if (c.bucket > 384 || c.disc_len > 6400) {
        out << "  Warning: large bucket/disc_len configs are extremely costly - on a discrete GPU the measured "
               "B=1/T=384/disc=12800 config crashed after only 5 epochs in 7 hours.\n";
        out << "        Recommended: bucket<=300, disc_len<=4800 (measured stable for 70+ epochs on the same machine).\n";
    }
}

} // namespace

bool run_train_wizard(std::istream& in, std::ostream& out, TrainerConfig* cfg, std::string* err) {
    if (cfg == nullptr) {
        if (err != nullptr) {
            *err = "run_train_wizard: cfg is null";
        }
        return false;
    }
    bool eof = false;
    out << "==== RVC training wizard (finetune / from scratch; M8) ====\n";
    out << "Press Enter to accept the default shown in [brackets].\n\n";

    // M8-1b: dual-mode data entry.
    const int64_t entry = ask_int("Data entry (1=existing cache directory 2=raw audio directory, preprocess first)", 1, 1, 2,
                                  in, out, &eof);
    if (entry == 2) {
        const std::string input_dir =
            ask_existing_dir("Raw audio directory (wav files)", "", in, out, &eof);
        const int64_t tgt = ask_int("Preprocessing target (1=v1-40k / 2=v2-48k)", 1, 1, 2, in, out, &eof);
        const char* version = tgt == 2 ? "v2" : "v1";
        const int sr = tgt == 2 ? 48000 : 40000;
        const std::string rmvpe =
            ask_optional_file("RMVPE weights path (press Enter to skip f0; recommended for training)", "", in, out, &eof);
        const std::string hubert =
            ask_existing_file("HuBERT weights path (feature extraction, required for training)", "", in, out, &eof);
        const std::string exp_def = tgt == 2 ? std::string("work/cache_auto_48k")
                                             : std::string("work/cache_auto_40k");
        const std::string exp_out =
            ask_nonempty("Preprocessing cache output directory", exp_def, in, out, &eof);
        if (eof) {
            if (err != nullptr) {
                *err = "wizard: end of input stream";
            }
            return false;
        }
        const std::string exe = find_rvc_prep_exe(out);
        if (exe.empty()) {
            if (err != nullptr) {
                *err = "rvc-prep.exe not found; run it manually (or put it in this program's directory and retry):\n"
                       "  rvc-prep --input \"" + input_dir + "\" --sr " + std::to_string(sr) +
                       " --version " + version + " --exp \"" + exp_out + "\"" +
                       (rmvpe.empty() ? "" : " --rmvpe \"" + rmvpe + "\"") +
                       " --hubert \"" + hubert + "\"";
            }
            return false;
        }
        std::string perr;
        if (!run_prep_subprocess(exe, input_dir, sr, version, rmvpe, hubert, exp_out, &perr)) {
            if (err != nullptr) {
                *err = perr;
            }
            return false;
        }
        out << "Preprocessing complete: " << exp_out << "\n\n";
        cfg->exp_dir = exp_out;
    } else {
        cfg->exp_dir = ask_cache_dir("Data cache directory (rvc-prep --exp output)", cfg->exp_dir, in, out,
                                     &eof);
    }

    const int64_t src = ask_int("Model source (1=pretrained finetune 2=train from scratch)", cfg->scratch ? 2 : 1, 1, 2, in,
                                out, &eof);
    if (src == 2) {
        cfg->scratch = true;
    } else {
        cfg->scratch = false;
        cfg->weights_g = ask_existing_file("Pretrained generator G (training-state GGUF)", cfg->weights_g, in, out, &eof);
        cfg->weights_d = ask_existing_file("Pretrained discriminator D (training-state GGUF)", cfg->weights_d, in, out, &eof);
    }

    const std::string out_def = cfg->out_dir.empty() ? std::string("work/train_out") : cfg->out_dir;
    cfg->out_dir = ask_nonempty("Output directory (logs/checkpoints/exports)", out_def, in, out, &eof);

    // M8-1c: resume entry - ask whether to resume when the output directory already has a checkpoint (with meta).
    if (cfg->resume.empty()) {
        const std::string ckpt = find_latest_checkpoint(cfg->out_dir);
        if (!ckpt.empty()) {
            out << "  Found an existing checkpoint: " << ckpt << "\n";
            const std::string yn = ask("Resume from this checkpoint? (y/N)", "n", in, out, &eof);
            if (confirm_yes(yn)) {
                cfg->resume = ckpt;
                out << "  Note: resume requires batch/bucket/seed/grad_accum/lr to match the original training "
                       "and the data cache to be unchanged (a mismatch is rejected).\n";
            }
        }
    }

    cfg->epochs = ask_int("Number of training epochs", cfg->epochs > 0 ? cfg->epochs : 1, 1, 1000000000, in,
                          out, &eof);
    cfg->batch_size = ask_int("batch_size", cfg->batch_size > 0 ? cfg->batch_size : 1, 1, 1024, in,
                              out, &eof);
    const int64_t bucket_def = cfg->bucket > 0 ? cfg->bucket : cfg->segment_frames;
    cfg->bucket = ask_int("bucket (graph T)", bucket_def, cfg->segment_frames, 1000000, in, out, &eof);
    cfg->disc_len = ask_int("disc_len (0=whole segment_size)", cfg->disc_len, 0, 100000000, in, out,
                            &eof);
    warn_large_config(*cfg, out);
    cfg->lr = ask_float("learning_rate", cfg->lr, in, out, &eof);
    cfg->seed = ask_int("Random seed", cfg->seed, 0, 2147483647, in, out, &eof);

    print_wizard_devices(out);
    const int64_t dev_max = traincpp::device_count() > 0
                                ? (int64_t) traincpp::device_count() - 1
                                : -1;
    cfg->device = ask_int("Training device index (-1=auto)", cfg->device, -1, dev_max > 1024 ? 1024 : dev_max,
                          in, out, &eof);

    // M8-1c: advanced parameters (collapsed by default; press Enter to skip).
    const std::string adv = ask("Configure advanced parameters? (y/N)", "n", in, out, &eof);
    if (confirm_yes(adv)) {
        cfg->grad_accum = ask_int("grad_accum (gradient accumulation steps, 1=off)", cfg->grad_accum > 0
                                                                      ? cfg->grad_accum
                                                                      : 1,
                                  1, 64, in, out, &eof);
        cfg->save_interval = ask_int("save_interval (save a checkpoint every N epochs, 0=finalize only)",
                                     cfg->save_interval, 0, 1000000, in, out, &eof);
        cfg->export_every = ask_int("export_every (export an inference GGUF every N epochs, 0=off)",
                                    cfg->export_every, 0, 1000000, in, out, &eof);
        cfg->snapshot_keep = ask_int("snapshot_keep (epoch snapshots to keep, 0=keep all)",
                                     cfg->snapshot_keep, 0, 10000, in, out, &eof);
        cfg->log_interval = ask_int("log_interval (log every N optimization steps)",
                                    cfg->log_interval > 0 ? cfg->log_interval : 50, 1, 1000000,
                                    in, out, &eof);
    }

    // M8-1c: enhanced pre-confirmation checks - cache version/sample-count/estimated-steps summary + consistency hints against the model config.
    if (!cfg->exp_dir.empty()) {
        prep::CacheDataset ds;
        std::string cerr2;
        if (load_cache_index(cfg->exp_dir, &ds, &cerr2)) {
            const int64_t n = (int64_t) ds.samples.size();
            out << "\n  Cache info: " << n << " samples | version " << ds.meta.version << " | sr "
                << ds.meta.target_sr << "\n";
            const int64_t batch = cfg->batch_size > 0 ? cfg->batch_size : 1;
            const int64_t accum = cfg->grad_accum > 0 ? cfg->grad_accum : 1;
            const int64_t micro = n / batch;      // drop the tail
            const int64_t steps = micro / accum;  // Trainer counts optimization steps
            out << "  Estimated: about " << steps << " optimization steps/epoch (batch=" << batch << "xaccum="
                << accum << "; samples exceeding the bucket are skipped, so the actual count may be lower)\n";
            if (ds.meta.version != cfg->model_cfg.version ||
                ds.meta.target_sr != cfg->model_cfg.sample_rate) {
                out << "  Warning: cache version/sample rate (" << ds.meta.version << "/"
                    << ds.meta.target_sr << ") does not match the model config (" << cfg->model_cfg.version << "/"
                    << cfg->model_cfg.sample_rate << ") - training will be rejected by the preflight check, "
                    "use matching weights and --config.\n";
            }
        } else {
            out << "  Note: failed to read the cache index (" << cerr2 << "); training startup will validate it again.\n";
        }
    }

    print_summary(*cfg, out);
    const std::string yn = ask("Confirm and start training? (y/N)", "n", in, out, &eof);
    if (!confirm_yes(yn)) {
        if (err != nullptr) {
            *err = "user cancelled";
        }
        return false;
    }

    if (cfg->exp_dir.empty() || cfg->out_dir.empty() ||
        (!cfg->scratch && (cfg->weights_g.empty() || cfg->weights_d.empty()))) {
        if (err != nullptr) {
            *err = "wizard: a required path is empty (exp/pretrain_g/pretrain_d/out)";
        }
        return false;
    }
    return true;
}

bool run_train_wizard_from_file(const std::string& answers_path, std::ostream& out,
                                TrainerConfig* cfg, std::string* err) {
    std::ifstream f(answers_path);
    if (!f.is_open()) {
        if (err != nullptr) {
            *err = "cannot open answers file: " + answers_path;
        }
        return false;
    }
    return run_train_wizard(f, out, cfg, err);
}

int run_train_wizard_selftest() {
    // Fixed answers (same order as the wizard questions): data entry(1=existing cache) / exp / model source(1=finetune) / G / D / out /
    // epochs / batch / bucket / disc_len / lr / seed / device / advanced params(n) / confirm
    static const char* kAnswers =
        "1\n"
        "work/cache_v1\n"
        "1\n"
        "work/models/f0G40k.train.gguf\n"
        "work/models/f0D40k.train.gguf\n"
        "work/wizard_selftest\n"
        "3\n"
        "1\n"
        "384\n"
        "1024\n"
        "0.0001\n"
        "1234\n"
        "1\n"
        "n\n"
        "y\n";

    // "Hand-written JSON" golden config: should produce a TrainerConfig fully equivalent to the answers above (remaining fields take struct defaults).
    static const char* kGoldenJson = R"json({
  "run": {
    "exp": "work/cache_v1",
    "out": "work/wizard_selftest",
    "pretrain_g": "work/models/f0G40k.train.gguf",
    "pretrain_d": "work/models/f0D40k.train.gguf",
    "device": 1,
    "resume": "",
    "export_every": 0,
    "export_run_json": ""
  },
  "train": {
    "batch_size": 1,
    "learning_rate": 0.0001,
    "betas": [0.8, 0.99],
    "eps": 1e-09,
    "weight_decay": 0.01,
    "lr_decay": 0.999875,
    "c_mel": 45,
    "c_kl": 1,
    "epochs": 3,
    "max_steps": 0,
    "segment_size": 12800,
    "bucket": 384,
    "disc_len": 1024,
    "log_interval": 50,
    "save_interval": 0,
    "seed": 1234,
    "grad_accum": 1,
    "buckets": []
  },
  "data": {
    "sampling_rate": 40000,
    "filter_length": 2048,
    "hop_length": 400,
    "win_length": 2048,
    "n_mel_channels": 125,
    "mel_fmin": 0,
    "mel_fmax": null
  },
  "model": { "inter_channels": 192 }
})json";

    std::printf("[wizard-selftest] ==== wizard equivalence self-test (G-7.1) ====\n");
    TrainerConfig wiz;
    std::istringstream in(kAnswers);
    std::string err;
    if (!run_train_wizard(in, std::cout, &wiz, &err)) {
        std::printf("[wizard-selftest] FAIL: wizard generation failed: %s\n", err.c_str());
        return 1;
    }
    TrainerConfig gold;
    if (!trainer_config_from_json_text(kGoldenJson, &gold, &err)) {
        std::printf("[wizard-selftest] FAIL: hand-written JSON parse failed: %s\n", err.c_str());
        return 1;
    }
    std::string diff;
    if (!trainer_config_equivalent(wiz, gold, &diff)) {
        std::printf("[wizard-selftest] FAIL: wizard config is not equivalent to hand-written JSON - %s\n", diff.c_str());
        return 1;
    }
    std::printf("[wizard-selftest] PASS: wizard-generated config is field-by-field equivalent to hand-written JSON\n");
    return 0;
}

} // namespace train
} // namespace rvc_train
