// highpass.h -- bit-level replica of the official training preprocessing 48Hz high-pass (causal lfilter) (M4-1)
//
// Semantic baseline: official 81eed5e train/preprocess.py:49/:93
//   self.bh, self.ah = signal.butter(N=5, Wn=48, btype="high", fs=self.sr)
//   audio = signal.lfilter(self.bh, self.ah, audio)   # causal; filtfilt disabled
//
// Note: do not use rvc_core::dsp::high_pass_48hz_in_place (that is the inference-side 16k filtfilt).
// To pass G2 (injection-mode max_abs <=1e-6), this implementation matches scipy/numpy rounding semantics bit-for-bit:
//   - coefficients: replicate the scipy.signal.butter -> iirfilter -> buttap/lp2hp_zpk/bilinear_zpk/zpk2tf
//     operation order; np.poly's polynomial multiplication follows the exact OpenBLAS zdotu (n=2) association (see .cpp);
//   - filtering: replicate scipy.signal.lfilter's transposed direct form II (zero initial state, a[0] normalized).
// This bit-level consistency is continuously verified by tools/gen_golden_prep.py + tools/verify_prep.py (G2).

#pragma once

#include <vector>

namespace rvc_train::prep {

// b/a of scipy.signal.butter(N=order, Wn=cutoff_hz, btype='high', fs=fs)
void butter_highpass_ba(int order, double cutoff_hz, double fs,
                        std::vector<double>* b, std::vector<double>* a);

// scipy.signal.lfilter(b, a, x) (causal IIR, zero initial state; output same length as input)
std::vector<double> lfilter(const std::vector<double>& b, const std::vector<double>& a,
                            const std::vector<double>& x);

} // namespace rvc_train::prep
