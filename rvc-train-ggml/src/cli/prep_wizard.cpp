// RVC-Train-GGML - M8-2 rvc-prep interactive preprocessing wizard implementation (src/cli/prep_wizard.cpp)
//
// See prep_wizard.h and `handover.md` §4.13. This file belongs to rvc-prep (may call the rvc_core public API
// rvc::list_gpu_devices for device listing); does not link traincpp.
#include "cli/prep_wizard.h"

#include "engine.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <system_error>

namespace rvc_train {
namespace prep {
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

bool read_answer(std::istream& in, std::string* out) {
    if (!std::getline(in, *out)) {
        return false;
    }
    *out = trim(*out);
    return true;
}

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

// Either a directory or a single wav file is accepted (same as CLI --input).
std::string ask_input_path(const std::string& label, const std::string& def, std::istream& in,
                           std::ostream& out, bool* eof) {
    std::string cur = def;
    for (;;) {
        const std::string s = ask(label, cur, in, out, eof);
        std::error_code ec;
        const std::filesystem::path p(s);
        if (!s.empty() &&
            (std::filesystem::is_directory(p, ec) || std::filesystem::is_regular_file(p, ec))) {
            return s;
        }
        if (*eof) {
            return s;
        }
        out << "  Path does not exist: " << s << ", please retry.\n";
        cur = s;
    }
}

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
        } else {
            return cur;
        }
        if (*eof) {
            return cur;
        }
        out << "  Please enter an integer in " << min_v << ".." << max_v << ", please retry.\n";
        cur = def;
    }
}

bool confirm_yes(const std::string& yn) {
    return yn == "y" || yn == "Y" || yn == "yes" || yn == "YES" || yn == "是";
}

void print_gpu_devices(std::ostream& out) {
    const std::vector<rvc::GpuDeviceInfo> gpus = rvc::list_gpu_devices();
    out << "\nAvailable compute devices (ggml):\n";
    if (gpus.empty()) {
        out << "  (no GPU enumerated, falling back to CPU=99)\n";
    }
    for (const rvc::GpuDeviceInfo& g : gpus) {
        out << "  [" << g.index << "] " << g.name << " | " << g.description << "\n";
    }
    out << "  99 = CPU (software fallback, debug only)\n";
    out << "  -1 = auto (use [0] if a GPU exists, otherwise CPU)\n";
}

} // namespace

bool run_prep_wizard(std::istream& in, std::ostream& out, PrepWizardResult* res, std::string* err) {
    if (res == nullptr) {
        if (err != nullptr) {
            *err = "run_prep_wizard: res is null";
        }
        return false;
    }
    bool eof = false;
    out << "==== RVC preprocessing wizard (raw audio -> training cache; M8) ====\n";
    out << "Press Enter to accept the default in [brackets]. When done, start training in rvc-train.\n\n";

    res->input = ask_input_path("Input audio directory (or a single wav file)", res->input, in, out, &eof);
    res->target = (int) ask_int("Preprocessing target (1=v1-40k / 2=v2-48k)", res->target, 1, 2, in, out,
                                &eof);
    res->rmvpe = ask_optional_file("RMVPE weights path (press Enter to skip f0; recommended for training)", res->rmvpe, in,
                                   out, &eof);
    res->hubert = ask_optional_file("HuBERT weights path (press Enter to skip feature extraction; required for training)", res->hubert,
                                    in, out, &eof);
    if (!res->hubert.empty() && !res->rmvpe.empty()) {
        out << "  (f0 and features will both be extracted; the cache can be used for training directly)\n";
    } else if (!res->hubert.empty()) {
        out << "  warning: RMVPE not provided, f0 will be skipped -- this cache cannot be used for f0 model training yet.\n";
    } else {
        out << "  warning: HuBERT not provided, feature extraction will be skipped -- this cache cannot be used for training.\n";
    }
    const std::string exp_def = res->target == 2 ? std::string("work/cache_auto_48k")
                                                 : std::string("work/cache_auto_40k");
    res->exp_dir = ask_nonempty("Training cache output directory", exp_def, in, out, &eof);

    print_gpu_devices(out);
    res->device = (int) ask_int("Compute device index (-1=auto, 99=CPU)", res->device, -1, 99, in, out,
                                &eof);

    out << "\n---- Preprocessing summary ----\n";
    out << "  Input      : " << res->input << "\n";
    out << "  Target     : " << (res->target == 2 ? "v2-48k (sr 48000 / 768-dim features)"
                                                  : "v1-40k (sr 40000 / 256-dim features)")
        << "\n";
    out << "  RMVPE      : " << (res->rmvpe.empty() ? "(skip f0)" : res->rmvpe) << "\n";
    out << "  HuBERT     : " << (res->hubert.empty() ? "(skip feature extraction)" : res->hubert) << "\n";
    out << "  Cache dir  : " << res->exp_dir << "\n";
    out << "  Device     : " << res->device << " (-1=auto)\n";
    out << "--------------------\n";

    const std::string yn = ask("Confirm and start preprocessing? (y/N)", "n", in, out, &eof);
    if (!confirm_yes(yn)) {
        if (err != nullptr) {
            *err = "user cancelled";
        }
        return false;
    }

    if (res->input.empty() || res->exp_dir.empty()) {
        if (err != nullptr) {
            *err = "wizard: input/cache output is empty";
        }
        return false;
    }
    return true;
}

bool run_prep_wizard_from_file(const std::string& answers_path, std::ostream& out,
                               PrepWizardResult* res, std::string* err) {
    std::ifstream f(answers_path);
    if (!f.is_open()) {
        if (err != nullptr) {
            *err = "cannot open answers file: " + answers_path;
        }
        return false;
    }
    return run_prep_wizard(f, out, res, err);
}

int run_prep_wizard_selftest() {
    // Fixed answers (in prompt order): input / target / rmvpe / hubert / exp / device / confirm
    static const char* kAnswers =
        ".\n"
        "2\n"
        "\n"
        "\n"
        "work/prep_wiz_selftest\n"
        "-1\n"
        "y\n";

    std::printf("[prep-wizard-selftest] ==== preprocessing wizard self-test (M8-2) ====\n");
    PrepWizardResult res;
    std::istringstream in(kAnswers);
    std::string err;
    if (!run_prep_wizard(in, std::cout, &res, &err)) {
        std::printf("[prep-wizard-selftest] FAIL: wizard not completed: %s\n", err.c_str());
        return 1;
    }
    int fails = 0;
    auto check = [&](bool ok, const char* what) {
        if (!ok) {
            std::printf("[prep-wizard-selftest] FAIL: %s\n", what);
            ++fails;
        }
    };
    check(res.input == ".", "input should be .");
    check(res.target == 2, "target should be 2 (v2-48k)");
    check(res.rmvpe.empty(), "rmvpe should be empty");
    check(res.hubert.empty(), "hubert should be empty");
    check(res.exp_dir == "work/prep_wiz_selftest", "exp_dir should be work/prep_wiz_selftest");
    check(res.device == -1, "device should be -1 (auto)");
    if (fails == 0) {
        std::printf("[prep-wizard-selftest] PASS: wizard answers match expectations field by field\n");
        return 0;
    }
    return 1;
}

} // namespace prep
} // namespace rvc_train
