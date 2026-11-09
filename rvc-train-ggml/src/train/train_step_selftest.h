// RVC-Train-GGML - M3-3 dual-graph G/D + one AdamW step comparison-test entry (src/train/train_step_selftest.h)
//
// Golden mechanism: tools/gen_golden_train_step.py -> work\golden\m3 (manifest.train_step.json +
// train_step.*.raw, shape = torch order). Semantic baseline = official 81eed5e train.py:425-527
// (D runs two forwards per step; G uses the updated D; AdamW(0.8,0.99,eps=1e-9,lr=1e-4,wd=1e-2)).
// Criteria: forward rel ≤1e-4; loss rel ≤1e-4; ΔW rel ≤1e-4; no gradient leakage (no grad_acc on the frozen side,
// D bit-unchanged after the G step, G bit-unchanged after the D step).
#pragma once

#include "traincpp/traincpp.h"

#include <string>

namespace rvc_train {
namespace train {

// weights_g: G training-state GGUF (f0G40k.train.gguf, including enc_q)
// weights_d: D training-state GGUF (f0D40k.train.gguf)
// golden_dir: M3 golden directory (contains manifest.train_step.json and train_step.*.raw)
// Returns: 0 = all passed; 1 = some failed; 2 = argument/environment error
int run_train_step_selftest(const std::string& weights_g, const std::string& weights_d,
                            const std::string& golden_dir, traincpp::Device* dev);

// M5-2 10-step trajectory comparison of the training loop (single-ctx TrainStep):
//   golden = output of tools/gen_golden_train_loop.py (manifest.train_loop.json +
//   train_loop.s{i}_*.raw); inject ids/q_noise/sine/wave per step and compare the 6 losses.
//   Gate G-5.2b (Q24 revision): first step rel ≤1e-4 (measured 5.3e-7); per-step tolerances for later steps = chaotic-probe ensemble
//   ×3 envelope (`<golden>/probe` fallback or manifest `params.tols`, pinned by tools/make_train_loop_envelope.py)
//   + finite loss (a per-value 1e-2 is unreachable in F32: the probe itself diverges within 10 steps).
// Returns: 0 = all passed; 1 = some failed; 2 = argument/environment error
int run_train_loop_selftest(const std::string& weights_g, const std::string& weights_d,
                            const std::string& golden_dir, traincpp::Device* dev);

} // namespace train
} // namespace rvc_train
