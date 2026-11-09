// resample.cpp -- M4 preprocessing resampling implementation (load-branch in-house kernel + 16k resampy kaiser_best emulation)

#include "prep/resample.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <utility>

namespace rvc_train::prep {
namespace {

const double kPi = 3.14159265358979323846264338327950288;

int64_t gcd_i64(int64_t a, int64_t b) {
    while (b != 0) {
        const int64_t t = a % b;
        a = b;
        b = t;
    }
    return a;
}

double sinc(double x) {
    if (std::fabs(x) < 1e-12) {
        return 1.0;
    }
    const double px = kPi * x;
    return std::sin(px) / px;
}

// Zeroth-order modified Bessel function I0 (series expansion)
double bessel_i0(double x) {
    double sum = 1.0;
    double term = 1.0;
    const double y = x * x / 4.0;
    for (int k = 1; k < 128; ++k) {
        term *= y / (static_cast<double>(k) * static_cast<double>(k));
        sum += term;
        if (term < 1e-18 * sum) {
            break;
        }
    }
    return sum;
}

// numpy 'reflect' semantics (no edge repetition)
int64_t reflect_index(int64_t i, int64_t n) {
    if (n <= 1) {
        return 0;
    }
    while (i < 0 || i >= n) {
        if (i < 0) {
            i = -i;
        } else {
            i = 2 * n - 2 - i;
        }
    }
    return i;
}

} // namespace

std::vector<float> resample_sinc(const std::vector<float>& in, int src_sr, int dst_sr) {
    if (src_sr == dst_sr || in.empty() || src_sr <= 0 || dst_sr <= 0) {
        return in;
    }
    const int64_t g = gcd_i64(src_sr, dst_sr);
    const int64_t up = dst_sr / g;    // U
    const int64_t down = src_sr / g;  // D

    const double scale = std::min(1.0, static_cast<double>(dst_sr) / static_cast<double>(src_sr));
    const double half = 32.0 / scale;  // kernel half-width (input sample domain; widened on downsampling to preserve the stopband)
    const int64_t i_half = static_cast<int64_t>(std::ceil(half));
    const double beta = 8.0;
    const double i0_beta = bessel_i0(beta);

    const int64_t n = static_cast<int64_t>(in.size());
    const int64_t out_len = std::max<int64_t>(
        1, static_cast<int64_t>(std::llround(static_cast<double>(n) * dst_sr / src_sr)));
    std::vector<float> out(static_cast<std::size_t>(out_len));

    for (int64_t i = 0; i < out_len; ++i) {
        const double pos = static_cast<double>(i) * static_cast<double>(down) /
                           static_cast<double>(up);
        const int64_t j0 = static_cast<int64_t>(std::floor(pos));

        double acc = 0.0;
        double wsum = 0.0;
        for (int64_t j = j0 - i_half + 1; j <= j0 + i_half; ++j) {
            const double t = pos - static_cast<double>(j);
            const double x = t / half;
            double w = 0.0;
            if (x * x < 1.0) {
                w = bessel_i0(beta * std::sqrt(1.0 - x * x)) / i0_beta;
            }
            const double h = scale * sinc(scale * t) * w;
            if (h == 0.0) {
                continue;
            }
            const int64_t idx = reflect_index(j, n);
            acc += static_cast<double>(in[static_cast<std::size_t>(idx)]) * h;
            wsum += h;
        }
        out[static_cast<std::size_t>(i)] =
            static_cast<float>(wsum != 0.0 ? acc / wsum : 0.0);
    }
    return out;
}

// ---------------------------------------------------------------------------
// resampy kaiser_best bit-level emulation (16k branch)
// ---------------------------------------------------------------------------

namespace {

// npz table params (kaiser_best.npz: half_window length 409601, precision=8192, rolloff full precision);
// beta is not stored in the npz and was back-derived from the npz half_window (scipy.signal.windows.kaiser semantics).
constexpr int kKaiserNumZeros = 50;
constexpr int kKaiserNumTable = 8192;
constexpr double kKaiserRolloff = 0.91734737126087607;
constexpr double kKaiserBeta = 12.984585153568657;

// right half of np.kaiser/scipy.signal.windows.kaiser + sinc: win[j] = taper[j]*rolloff*sinc(rolloff*p)
// p = linspace(0, 50, n+1)[j] (n = 8192*50 = 409600, p = j/8192 exactly)
const std::vector<double>& kaiser_best_table() {
    static const std::vector<double> table = [] {
        const int64_t n = static_cast<int64_t>(kKaiserNumTable) * kKaiserNumZeros;
        std::vector<double> w(static_cast<std::size_t>(n) + 1);
        const double i0_beta = bessel_i0(kKaiserBeta);
        for (int64_t j = 0; j <= n; ++j) {
            const double p = static_cast<double>(j) * kKaiserNumZeros / static_cast<double>(n);
            const double sinc_win = kKaiserRolloff * sinc(kKaiserRolloff * p);
            const double r = static_cast<double>(j) / static_cast<double>(n);
            const double taper = bessel_i0(kKaiserBeta * std::sqrt(1.0 - r * r)) / i0_beta;
            w[static_cast<std::size_t>(j)] = taper * sinc_win;
        }
        return w;
    }();
    return table;
}

// resampy core.py: when ratio<1 the whole table is multiplied by ratio; interp_delta = np.diff(win, append=win[-1])
struct KaiserBestKernel {
    double ratio = 0.0;
    std::vector<double> win;
    std::vector<double> delta;
};

const KaiserBestKernel& kaiser_best_kernel(double ratio) {
    static std::vector<std::unique_ptr<KaiserBestKernel>> cache;
    for (const auto& kv : cache) {
        if (kv->ratio == ratio) {
            return *kv;
        }
    }
    auto k = std::make_unique<KaiserBestKernel>();
    k->ratio = ratio;
    k->win = kaiser_best_table();
    if (ratio < 1.0) {
        for (double& v : k->win) {
            v *= ratio;
        }
    }
    k->delta.resize(k->win.size());
    for (std::size_t j = 0; j + 1 < k->win.size(); ++j) {
        k->delta[j] = k->win[j + 1] - k->win[j];
    }
    k->delta.back() = 0.0;
    cache.push_back(std::move(k));
    return *cache.back();
}

} // namespace

std::vector<double> resample_kaiser_best(const std::vector<double>& in, int src_sr, int dst_sr) {
    if (src_sr == dst_sr || in.empty() || src_sr <= 0 || dst_sr <= 0) {
        return in;
    }
    const double ratio = static_cast<double>(dst_sr) / static_cast<double>(src_sr);
    // official: shape[axis] = int(shape[axis] * float(sr_new) / float(sr_orig)) (truncation)
    const int64_t n_out = static_cast<int64_t>(static_cast<double>(in.size()) *
                                               static_cast<double>(dst_sr) /
                                               static_cast<double>(src_sr));
    if (n_out < 1) {
        return {};
    }

    const KaiserBestKernel& kern = kaiser_best_kernel(ratio);
    const std::vector<double>& win = kern.win;
    const std::vector<double>& delta = kern.delta;
    const int64_t num_table = kKaiserNumTable;
    const int64_t nwin = static_cast<int64_t>(win.size());
    const int64_t n_orig = static_cast<int64_t>(in.size());

    const double scale = std::min(1.0, ratio);
    const double time_increment = 1.0 / ratio;
    const int64_t index_step = static_cast<int64_t>(scale * static_cast<double>(num_table));
    if (index_step <= 0) {
        // extreme downsampling ratio (src > num_table*dst); resampy is also unusable, explicitly return empty
        return {};
    }

    std::vector<double> out(static_cast<std::size_t>(n_out), 0.0);
    for (int64_t t = 0; t < n_out; ++t) {
        // resampy interpn._resample_loop (left wing -> right wing; order must match to keep floating-point accumulation identical)
        const double time_register =
            static_cast<double>(t) * time_increment;  // t_out = arange(n_out)*time_increment
        const int64_t n = static_cast<int64_t>(time_register);  // int() truncation
        double frac = scale * (time_register - static_cast<double>(n));

        double index_frac = frac * static_cast<double>(num_table);
        int64_t offset = static_cast<int64_t>(index_frac);
        double eta = index_frac - static_cast<double>(offset);

        double acc = 0.0;
        const int64_t i_max = std::min<int64_t>(n + 1, (nwin - offset) / index_step);
        for (int64_t i = 0; i < i_max; ++i) {
            const int64_t k = offset + i * index_step;
            const double weight = win[static_cast<std::size_t>(k)] +
                                  eta * delta[static_cast<std::size_t>(k)];
            acc += weight * in[static_cast<std::size_t>(n - i)];
        }

        frac = scale - frac;
        index_frac = frac * static_cast<double>(num_table);
        offset = static_cast<int64_t>(index_frac);
        eta = index_frac - static_cast<double>(offset);

        const int64_t k_max =
            std::min<int64_t>(n_orig - n - 1, (nwin - offset) / index_step);
        for (int64_t k = 0; k < k_max; ++k) {
            const int64_t idx = offset + k * index_step;
            const double weight = win[static_cast<std::size_t>(idx)] +
                                  eta * delta[static_cast<std::size_t>(idx)];
            acc += weight * in[static_cast<std::size_t>(n + k + 1)];
        }
        out[static_cast<std::size_t>(t)] = acc;
    }
    return out;
}

} // namespace rvc_train::prep
