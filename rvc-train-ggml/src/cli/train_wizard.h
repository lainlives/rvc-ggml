// RVC-Train-GGML - M7 interactive training wizard (src/cli/train_wizard.h)
//
// Role: collect training parameters through "guided input after launching the program", filling TrainerConfig;
// main_train.cpp writes <out>/train_config.json after confirmation and starts training immediately.
//
// Semantic baseline: `handover.md` §4.11 C. **Finetune (load pretrained training-state GGUF) and from scratch (--scratch, since M7-2)
// the two branches** are both supported.
//
// Design notes:
//   - Injectable input/output streams (run_train_wizard) ease equivalence self-tests with fixed answers and batch processing;
//   - Prompts + defaults (press Enter to accept the default); retry on invalid input; on end of input stream (EOF) keep the default and continue.
#pragma once

#include <iosfwd>
#include <string>

#include "train/trainer.h"

namespace rvc_train {
namespace train {

// Interactive wizard: read answers from in line by line, print prompts to out, fill *cfg.
// cfg may be pre-initialized from a base config (supplying train/data/model defaults); the wizard only overwrites the fields it asks about.
// Returns false and sets err when the user declines confirmation (answers n to y/N) or a parameter cannot be parsed.
bool run_train_wizard(std::istream& in, std::ostream& out, TrainerConfig* cfg, std::string* err);

// Read from an answers file (non-interactive; one answer per line in answers_path, same order as the wizard questions). Prompts are still printed to out.
bool run_train_wizard_from_file(const std::string& answers_path, std::ostream& out,
                                TrainerConfig* cfg, std::string* err);

// G-7.1 equivalence self-test: drive the wizard with a built-in set of valid answers and compare the generated config against the built-in "hand-written JSON" golden config
// field by field (trainer_config_equivalent). Returns 0 = PASS, 1 = FAIL.
int run_train_wizard_selftest();

} // namespace train
} // namespace rvc_train
