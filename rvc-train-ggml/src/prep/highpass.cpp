// highpass.cpp -- bit-level replica of scipy butter / lfilter (M4-1)
//
// Key rounding facts (confirmed bit-for-bit by the tools prototype and scipy 1.9.3 + numpy 1.23.5, 2026-09-24):
//   1) scipy.signal.butter coefficient generation chain:
//        buttap(5)  -> lp2hp_zpk(wo=warped) -> bilinear_zpk(fs=2) -> zpk2tf
//      where zpk2tf expands the polynomial with np.poly, and np.poly internals are np.convolve(a, [1,-r]);
//      np.convolve dispatches complex128 to multiarray.correlate -> CDOUBLE_dot -> OpenBLAS zdotu.
//      OpenBLAS zdotu's exact association for n=2 is:
//        real = (x0r*y0r + x1r*y1r) - (x0i*y0i + x1i*y1i)
//        imag = (x0r*y0i + x1r*y1i) + (x0i*y0r + x1i*y1r)
//      (not complex-multiply-then-sequential-accumulate; the latter differs by 1 ulp, but this filter's poles
//        sit very close to the unit circle, so a 1-ulp coefficient difference is amplified to ~1e-6 at the output, hence the replica is required.)
//   2) complex division uses numpy's cdiv (Smith's algorithm + 1/(br+bi*rat) scaling).
//   3) lfilter uses transposed direct form II.
//      Measured: for fs in {16000,32000,40000,44100,48000} and fc in {30,48,60}, coefficients and 200k-sample outputs
//      are bit-identical to scipy (maxdiff = 0).

#include "prep/highpass.h"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace rvc_train::prep {
namespace {

const double kPi = 3.14159265358979323846264338327950288;

// Complex numbers (double components; all operations expanded explicitly to avoid std::complex implementation differences)
struct Cx {
    double r = 0.0;
    double i = 0.0;
};

// numpy cmul: ar*br - ai*bi, ar*bi + ai*br (independent rounding per component)
Cx cx_mul(const Cx& x, const Cx& y) {
    return Cx{x.r * y.r - x.i * y.i, x.r * y.i + x.i * y.r};
}

// numpy npy_cdiv (Smith's algorithm + scaling; see numpy/core/src/npymath/npy_math_complex.c.src)
#if defined(_MSC_VER)
#pragma warning(push)
// C4723: explicitly replicate numpy's divide-by-zero branch (not triggered by this project's butter chain; kept for semantic completeness)
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

// OpenBLAS zdotu (n=2) exact association (see the file header notes)
Cx zdotu2(const Cx& x0, const Cx& x1, const Cx& y0, const Cx& y1) {
    const double sr = (x0.r * y0.r + x1.r * y1.r) - (x0.i * y0.i + x1.i * y1.i);
    const double si = (x0.r * y0.i + x1.r * y1.i) + (x0.i * y0.r + x1.i * y1.r);
    return Cx{sr, si};
}

// Bit-level replica of numpy.poly: convolve(polynomial, [1, -r], 'full')
//   numpy internals are correlate(c, [-r, 1]); head/tail n=1 (single product), middle sections n=2 (zdotu2).
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
    // scipy.signal.iirfilter: Wn = 2*Wn/fs (normalized), then prewarp (internal fs set to 2)
    const double wn = 2.0 * cutoff_hz / fs;
    const double warp_fs = 2.0;
    const double warped = 2.0 * warp_fs * std::tan(kPi * wn / warp_fs);

    // buttap(order): poles p = -exp(1j*pi*m/(2N)); numpy's 1j multiplication and complex division rounding must be replicated
    const int n = order;
    std::vector<Cx> p_lp(static_cast<std::size_t>(n));
    for (int k = 0; k < n; ++k) {
        const int m = -n + 1 + 2 * k;
        const double pi_m = kPi * static_cast<double>(m);  // complex(0,pi) * m -> (0, pi*m)
        const Cx theta = cx_div(Cx{0.0, pi_m}, Cx{2.0 * n, 0.0});
        p_lp[static_cast<std::size_t>(k)] = Cx{-std::cos(theta.i), -std::sin(theta.i)};
    }

    // lp2hp_zpk: p_hp = wo/p; k_hp = k * real(prod(-z)/prod(-p)) (z is empty, prod(-z)=1)
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

    // bilinear_zpk: z_z = (4+0)/(4-0) = 1; p_z = (4+p_hp)/(4-p_hp)
    //   k_z = k * real(prod(4-z_hp)/prod(4-p_hp)) (z_hp is all 0, prod(4-z)=4^N)
    const double fs2 = 2.0 * warp_fs;
    std::vector<Cx> z_z(static_cast<std::size_t>(n), Cx{1.0, 0.0});
    std::vector<Cx> p_z(static_cast<std::size_t>(n));
    Cx prod_4mz{1.0, 0.0};
    Cx prod_4mp{1.0, 0.0};
    for (int k = 0; k < n; ++k) {
        const Cx& p = p_hp[static_cast<std::size_t>(k)];
        prod_4mz = cx_mul(prod_4mz, Cx{fs2 - 0.0, 0.0 - 0.0});  // 4 - z_hp (z_hp=0)
        prod_4mp = cx_mul(prod_4mp, Cx{fs2 - p.r, 0.0 - p.i});
        p_z[static_cast<std::size_t>(k)] = cx_div(Cx{fs2 + p.r, 0.0 + p.i}, Cx{fs2 - p.r, 0.0 - p.i});
    }
    const double k_z = k_hp * cx_div(prod_4mz, prod_4mp).r;

    // zpk2tf: b = k*poly(z) (conjugate-symmetric -> take real part); a = poly(p) (take real part)
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
    // scipy.signal.lfilter: transposed direct form II, zero initial state, a[0] normalized
    std::vector<double> bn = b;
    std::vector<double> an = a;
    const double a0 = an.empty() ? 1.0 : an[0];
    for (double& v : bn) {
        v /= a0;
    }
    for (double& v : an) {
        v /= a0;
    }

    const std::size_t n = std::max(a.size(), b.size()) - 1;  // order
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
