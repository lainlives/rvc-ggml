// highpass.h —— 官方训练预处理 48Hz 高通（因果 lfilter）的位级复刻（M4-1）
//
// 语义基准：官方 81eed5e train/preprocess.py:49/:93
//   self.bh, self.ah = signal.butter(N=5, Wn=48, btype="high", fs=self.sr)
//   audio = signal.lfilter(self.bh, self.ah, audio)   # 因果；filtfilt 已停用
//
// 注意：不用 rvc_core::dsp::high_pass_48hz_in_place（那是推理侧 16k filtfilt）。
// 为通过 G2（注入模式 max_abs ≤1e-6），本实现对 scipy/numpy 的舍入语义做到位级一致：
//   - 系数：复刻 scipy.signal.butter → iirfilter → buttap/lp2hp_zpk/bilinear_zpk/zpk2tf
//     的运算顺序；其中 np.poly 的多项式乘法走 OpenBLAS zdotu（n=2）的精确关联（见 .cpp）；
//   - 滤波：复刻 scipy.signal.lfilter 的 transposed direct form II（零初值，a[0] 归一化）。
// 该位级一致性由 tools/gen_golden_prep.py + tools/verify_prep.py（G2）持续校验。

#pragma once

#include <vector>

namespace rvc_train::prep {

// scipy.signal.butter(N=order, Wn=cutoff_hz, btype='high', fs=fs) 的 b/a
void butter_highpass_ba(int order, double cutoff_hz, double fs,
                        std::vector<double>* b, std::vector<double>* a);

// scipy.signal.lfilter(b, a, x)（因果 IIR，零初值；输出与输入等长）
std::vector<double> lfilter(const std::vector<double>& b, const std::vector<double>& a,
                            const std::vector<double>& x);

} // namespace rvc_train::prep
