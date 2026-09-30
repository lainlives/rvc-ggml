// highpass.cpp —— scipy butter / lfilter 的位级复刻实现（M4-1）
//
// 关键舍入事实（经 tools 原型与 scipy 1.9.3 + numpy 1.23.5 逐位比对确认，2026-09-24）：
//   1) scipy.signal.butter 的系数生成链：
//        buttap(5)  → lp2hp_zpk(wo=warped) → bilinear_zpk(fs=2) → zpk2tf
//      其中 zpk2tf 用 np.poly 展开多项式，而 np.poly 内部是 np.convolve(a, [1,-r])；
//      np.convolve 对 complex128 走 multiarray.correlate → CDOUBLE_dot → OpenBLAS zdotu。
//      OpenBLAS zdotu 对 n=2 的精确关联为：
//        real = (x0r*y0r + x1r*y1r) - (x0i*y0i + x1i*y1i)
//        imag = (x0r*y0i + x1r*y1i) + (x0i*y0r + x1i*y1r)
//      （不是“先复数乘再顺序累加”；后者差 1 ulp，但该滤波器极点极靠近单位圆，
//        1 ulp 系数差会被放大到 ~1e-6 输出量级，因此必须复刻。）
//   2) 复数除法用 numpy 的 cdiv（Smith 算法 + 1/(br+bi*rat) 缩放）。
//   3) lfilter 用 transposed direct form II。
//      实测 fs∈{16000,32000,40000,44100,48000}、fc∈{30,48,60} 系数与 20 万样本输出
//      均与 scipy 位级一致（maxdiff = 0）。

#include "prep/highpass.h"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace rvc_train::prep {
namespace {

const double kPi = 3.14159265358979323846264338327950288;

// 复数（double 分量；所有运算显式展开，避免 std::complex 的实现差异）
struct Cx {
    double r = 0.0;
    double i = 0.0;
};

// numpy cmul：ar*br - ai*bi, ar*bi + ai*br（逐分量独立舍入）
Cx cx_mul(const Cx& x, const Cx& y) {
    return Cx{x.r * y.r - x.i * y.i, x.r * y.i + x.i * y.r};
}

// numpy npy_cdiv（Smith 算法 + 缩放；见 numpy/core/src/npymath/npy_math_complex.c.src）
#if defined(_MSC_VER)
#pragma warning(push)
// C4723：显式复刻 numpy 的除零分支（本项目 butter 链路不会触发；保留语义完整性）
#pragma warning(disable : 4723)
#endif
Cx cx_div(const Cx& x, const Cx& y) {
    const double abs_br = std::fabs(y.r);
    const double abs_bi = std::fabs(y.i);
    if (abs_br >= abs_bi) {
        if (abs_br == 0.0 && abs_bi == 0.0) {
            return Cx{x.r / abs_br, x.i / abs_bi};
        }
        const double rat = y.i / y.r;
        const double scl = 1.0 / (y.r + y.i * rat);
        return Cx{(x.r + x.i * rat) * scl, (x.i - x.r * rat) * scl};
    }
    const double rat = y.r / y.i;
    const double scl = 1.0 / (y.i + y.r * rat);
    return Cx{(x.r * rat + x.i) * scl, (x.i * rat - x.r) * scl};
}
#if defined(_MSC_VER)
#pragma warning(pop)
#endif

// OpenBLAS zdotu（n=2）精确关联（见文件头说明）
Cx zdotu2(const Cx& x0, const Cx& x1, const Cx& y0, const Cx& y1) {
    const double sr = (x0.r * y0.r + x1.r * y1.r) - (x0.i * y0.i + x1.i * y1.i);
    const double si = (x0.r * y0.i + x1.r * y1.i) + (x0.i * y0.r + x1.i * y1.r);
    return Cx{sr, si};
}

// numpy.poly 的位级复刻：convolve(多项式, [1, -r], 'full')
//   numpy 内部为 correlate(c, [-r, 1])；首尾 n=1（单乘积），中间段 n=2（zdotu2）。
std::vector<Cx> poly_from_roots(const std::vector<Cx>& roots) {
    std::vector<Cx> c{Cx{1.0, 0.0}};
    for (const Cx& r : roots) {
        const Cx v0{-r.r, -r.i};  // v_rev[0] = -r
        const Cx v1{1.0, 0.0};    // v_rev[1] = 1
        std::vector<Cx> out(c.size() + 1);
        out[0] = cx_mul(c[0], v1);
        for (std::size_t k = 1; k < c.size(); ++k) {
            out[k] = zdotu2(c[k - 1], c[k], v0, v1);
        }
        out[c.size()] = cx_mul(c.back(), v0);
        c.swap(out);
    }
    return c;
}

} // namespace

void butter_highpass_ba(int order, double cutoff_hz, double fs,
                        std::vector<double>* b, std::vector<double>* a) {
    // scipy.signal.iirfilter：Wn = 2*Wn/fs（归一化），再 prewarp（内部 fs 置 2）
    const double wn = 2.0 * cutoff_hz / fs;
    const double warp_fs = 2.0;
    const double warped = 2.0 * warp_fs * std::tan(kPi * wn / warp_fs);

    // buttap(order)：极点 p = -exp(1j*pi*m/(2N))；numpy 的 1j 乘法与复数除法舍入需复刻
    const int n = order;
    std::vector<Cx> p_lp(static_cast<std::size_t>(n));
    for (int k = 0; k < n; ++k) {
        const int m = -n + 1 + 2 * k;
        const double pi_m = kPi * static_cast<double>(m);  // complex(0,pi) * m → (0, pi*m)
        const Cx theta = cx_div(Cx{0.0, pi_m}, Cx{2.0 * n, 0.0});
        p_lp[static_cast<std::size_t>(k)] = Cx{-std::cos(theta.i), -std::sin(theta.i)};
    }

    // lp2hp_zpk：p_hp = wo/p；k_hp = k * real(prod(-z)/prod(-p))（z 为空，prod(-z)=1）
    Cx prod_neg_p{1.0, 0.0};
    for (const Cx& p : p_lp) {
        prod_neg_p = cx_mul(prod_neg_p, Cx{-p.r, -p.i});
    }
    const double k_hp = cx_div(Cx{1.0, 0.0}, prod_neg_p).r;

    std::vector<Cx> p_hp(static_cast<std::size_t>(n));
    for (int k = 0; k < n; ++k) {
        p_hp[static_cast<std::size_t>(k)] =
            cx_div(Cx{warped, 0.0}, p_lp[static_cast<std::size_t>(k)]);
    }

    // bilinear_zpk：z_z = (4+0)/(4-0) = 1；p_z = (4+p_hp)/(4-p_hp)
    //   k_z = k * real(prod(4-z_hp)/prod(4-p_hp))（z_hp 为全 0，prod(4-z)=4^N）
    const double fs2 = 2.0 * warp_fs;
    std::vector<Cx> z_z(static_cast<std::size_t>(n), Cx{1.0, 0.0});
    std::vector<Cx> p_z(static_cast<std::size_t>(n));
    Cx prod_4mz{1.0, 0.0};
    Cx prod_4mp{1.0, 0.0};
    for (int k = 0; k < n; ++k) {
        const Cx& p = p_hp[static_cast<std::size_t>(k)];
        prod_4mz = cx_mul(prod_4mz, Cx{fs2 - 0.0, 0.0 - 0.0});  // 4 - z_hp（z_hp=0）
        prod_4mp = cx_mul(prod_4mp, Cx{fs2 - p.r, 0.0 - p.i});
        p_z[static_cast<std::size_t>(k)] = cx_div(Cx{fs2 + p.r, 0.0 + p.i}, Cx{fs2 - p.r, 0.0 - p.i});
    }
    const double k_z = k_hp * cx_div(prod_4mz, prod_4mp).r;

    // zpk2tf：b = k*poly(z)（共轭对称 → 取实部）；a = poly(p)（取实部）
    const std::vector<Cx> bz = poly_from_roots(z_z);
    const std::vector<Cx> az = poly_from_roots(p_z);
    b->resize(static_cast<std::size_t>(n) + 1);
    a->resize(static_cast<std::size_t>(n) + 1);
    for (std::size_t k = 0; k < b->size(); ++k) {
        (*b)[k] = k_z * bz[k].r;
        (*a)[k] = az[k].r;
    }
}

std::vector<double> lfilter(const std::vector<double>& b, const std::vector<double>& a,
                            const std::vector<double>& x) {
    // scipy.signal.lfilter：transposed direct form II，零初值，a[0] 归一化
    std::vector<double> bn = b;
    std::vector<double> an = a;
    const double a0 = an.empty() ? 1.0 : an[0];
    for (double& v : bn) {
        v /= a0;
    }
    for (double& v : an) {
        v /= a0;
    }

    const std::size_t n = std::max(a.size(), b.size()) - 1;  // 阶数
    std::vector<double> y(x.size(), 0.0);
    std::vector<double> z(n, 0.0);

    for (std::size_t m = 0; m < x.size(); ++m) {
        y[m] = bn[0] * x[m] + (n > 0 ? z[0] : 0.0);
        for (std::size_t i = 0; i + 1 < n; ++i) {
            z[i] = z[i + 1] + bn[i + 1] * x[m] - an[i + 1] * y[m];
        }
        if (n > 0) {
            z[n - 1] = bn[n] * x[m] - an[n] * y[m];
        }
    }
    return y;
}

} // namespace rvc_train::prep
