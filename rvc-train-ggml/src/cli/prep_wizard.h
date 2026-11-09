// RVC-Train-GGML - M8-2 rvc-prep interactive preprocessing wizard (src/cli/prep_wizard.h)
//
// Responsibility: collect preprocessing parameters via "guided input after launch" (input audio/
// target version/RMVPE/HuBERT/cache output/device); main_prep.cpp runs the --input-equivalent
// pipeline after confirmation, then auto-runs --validate-cache and reports the sample count plus the "next step" hint.
//
// Design points (same as train_wizard):
//   - injectable input/output streams, enabling self-tests and batch runs with fixed answers (--answers);
//   - prompts + defaults (press Enter to accept the default); retry on invalid input; keep the default on EOF;
//   - entered only via explicit --interactive / --answers; existing command-line usage is unaffected.
#pragma once

#include <iosfwd>
#include <string>

namespace rvc_train {
namespace prep {

struct PrepWizardResult {
    std::string input;    // audio directory or single wav
    int target = 1;       // 1 = v1-40k (sr 40000); 2 = v2-48k (sr 48000)
    std::string rmvpe;    // empty = skip f0
    std::string hubert;   // empty = skip feature extraction (required for training; the wizard warns)
    std::string exp_dir;  // training cache output directory
    int device = -1;      // -1 = auto (GPU → CPU 99); 0..n-1 = ggml devices; 99 = CPU
};

// Interactive wizard: read answers line by line from in, print prompts to out, fill *res.
// Returns false and writes err when the user declines confirmation (answers n to y/N).
bool run_prep_wizard(std::istream& in, std::ostream& out, PrepWizardResult* res, std::string* err);

// Read from an answers file (non-interactive; answers_path holds one answer per line, in prompt order).
bool run_prep_wizard_from_file(const std::string& answers_path, std::ostream& out,
                               PrepWizardResult* res, std::string* err);

// Self-test: drive the wizard with built-in answers and compare field by field against the expected PrepWizardResult. 0 = PASS, 1 = FAIL.
int run_prep_wizard_selftest();

} // namespace prep
} // namespace rvc_train
