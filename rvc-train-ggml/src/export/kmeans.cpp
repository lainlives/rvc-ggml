// kmeans.cpp —— M4-5b 实现（见 kmeans.h 头注释）

#include "export/kmeans.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <thread>
#include <vector>

namespace rvc_train::exporter {
namespace {

int effective_threads(int requested, int64_t work) {
    int t = requested;
    if (t <= 0) {
        const unsigned hw = std::thread::hardware_concurrency();
        t = hw == 0 ? 4 : static_cast<int>(hw);
    }
    if (t > 8) {
        t = 8;
    }
    if (work > 0 && static_cast<int64_t>(t) > work) {
        t = static_cast<int>(work);
    }
    if (t < 1) {
        t = 1;
    }
    return t;
}

// 简单并行 for：fn(begin, end, thread_index)；线程只在调用期间存在（工程规模足够）
template <typename Fn>
void parallel_for(int64_t n, int threads, Fn&& fn) {
    if (n <= 0) {
        return;
    }
    if (threads <= 1 || n == 1) {
        fn(int64_t{0}, n, 0);
        return;
    }
    const int64_t chunk = (n + threads - 1) / threads;
    std::vector<std::thread> pool;
    pool.reserve(static_cast<std::size_t>(threads));
    for (int t = 0; t < threads; ++t) {
        const int64_t begin = static_cast<int64_t>(t) * chunk;
        const int64_t end = std::min(n, begin + chunk);
        if (begin >= end) {
            break;
        }
        pool.emplace_back([&fn, begin, end, t]() { fn(begin, end, t); });
    }
    for (std::thread& th : pool) {
        th.join();
    }
}

// 点积：8 路展开 + 尾处理（不依赖 /fp:fast，保持与顺序累加相同的数值路径族）
inline float dot_f32(const float* a, const float* b, int64_t dim) {
    float s = 0.0f;
    int64_t i = 0;
    for (; i + 8 <= dim; i += 8) {
        s += a[i] * b[i] + a[i + 1] * b[i + 1] + a[i + 2] * b[i + 2] + a[i + 3] * b[i + 3];
        s += a[i + 4] * b[i + 4] + a[i + 5] * b[i + 5] + a[i + 6] * b[i + 6] + a[i + 7] * b[i + 7];
    }
    for (; i < dim; ++i) {
        s += a[i] * b[i];
    }
    return s;
}

void compute_centroid_norms(const float* c, int64_t k, int64_t dim, std::vector<float>* out,
                            int threads) {
    out->resize(static_cast<std::size_t>(k));
    parallel_for(k, threads, [&](int64_t b, int64_t e, int) {
        for (int64_t j = b; j < e; ++j) {
            (*out)[static_cast<std::size_t>(j)] = dot_f32(c + j * dim, c + j * dim, dim);
        }
    });
}

// 不重复随机采样 k 个行 id（Fisher-Yates 前 k 个；seed 可复现）
void sample_distinct_rows(int64_t n, int64_t k, std::mt19937* rng, std::vector<int64_t>* out) {
    out->resize(static_cast<std::size_t>(n));
    std::iota(out->begin(), out->end(), int64_t{0});
    for (int64_t i = 0; i < k; ++i) {
        std::uniform_int_distribution<int64_t> dist(i, n - 1);
        const int64_t j = dist(*rng);
        std::swap((*out)[static_cast<std::size_t>(i)], (*out)[static_cast<std::size_t>(j)]);
    }
}

} // namespace

double assign_to_centroids(const float* x, int64_t n, int64_t dim, const float* centroids, int64_t k,
                           std::vector<int32_t>* assign, int threads) {
    assign->assign(static_cast<std::size_t>(n), 0);
    if (x == nullptr || n <= 0 || dim <= 0 || centroids == nullptr || k <= 0) {
        return 0.0;
    }
    const int t = effective_threads(threads, n);
    std::vector<float> c_norm2;
    compute_centroid_norms(centroids, k, dim, &c_norm2, t);
    std::vector<double> partial(static_cast<std::size_t>(t), 0.0);
    parallel_for(n, t, [&](int64_t b, int64_t e, int ti) {
        double acc = 0.0;
        for (int64_t i = b; i < e; ++i) {
            const float* xi = x + i * dim;
            const float xn = dot_f32(xi, xi, dim);
            int64_t best = 0;
            float best_d = xn + c_norm2[0] - 2.0f * dot_f32(xi, centroids, dim);
            for (int64_t j = 1; j < k; ++j) {
                const float d =
                    xn + c_norm2[static_cast<std::size_t>(j)] - 2.0f * dot_f32(xi, centroids + j * dim, dim);
                if (d < best_d) {
                    best_d = d;
                    best = j;
                }
            }
            (*assign)[static_cast<std::size_t>(i)] = static_cast<int32_t>(best);
            acc += static_cast<double>(best_d);
        }
        partial[static_cast<std::size_t>(ti)] = acc;
    });
    double total = 0.0;
    for (double v : partial) {
        total += v;
    }
    return total;
}

bool kmeans_f32(const float* x, int64_t n, int64_t dim, int64_t k, const KMeansOptions& opt,
                std::vector<float>* centroids, std::string* err) {
    if (x == nullptr || n <= 0 || dim <= 0 || k <= 0 || k > n) {
        if (err) {
            *err = "kmeans 输入非法（n=" + std::to_string(n) + " dim=" + std::to_string(dim) +
                   " k=" + std::to_string(k) + "）";
        }
        return false;
    }
    const int64_t iters = opt.iters > 0 ? opt.iters : 25;
    const int threads = effective_threads(opt.threads, n);
    std::mt19937 rng(static_cast<uint32_t>(opt.seed));

    // 初始化：随机采样 k 个不同点
    std::vector<int64_t> sample;
    sample_distinct_rows(n, k, &rng, &sample);
    std::vector<float> c(static_cast<std::size_t>(k * dim), 0.0f);
    for (int64_t j = 0; j < k; ++j) {
        std::memcpy(c.data() + j * dim, x + sample[static_cast<std::size_t>(j)] * dim,
                    static_cast<std::size_t>(dim) * sizeof(float));
    }

    std::vector<float> x_norm2(static_cast<std::size_t>(n), 0.0f);
    parallel_for(n, threads, [&](int64_t b, int64_t e, int) {
        for (int64_t i = b; i < e; ++i) {
            x_norm2[static_cast<std::size_t>(i)] = dot_f32(x + i * dim, x + i * dim, dim);
        }
    });

    std::vector<float> c_norm2(static_cast<std::size_t>(k), 0.0f);
    std::vector<int32_t> assign(static_cast<std::size_t>(n), -1);
    std::vector<int32_t> prev(static_cast<std::size_t>(n), -1);
    std::vector<std::vector<float>> sum_buf(static_cast<std::size_t>(threads));
    std::vector<std::vector<int64_t>> cnt_buf(static_cast<std::size_t>(threads));
    for (int t = 0; t < threads; ++t) {
        sum_buf[static_cast<std::size_t>(t)].assign(static_cast<std::size_t>(k * dim), 0.0f);
        cnt_buf[static_cast<std::size_t>(t)].assign(static_cast<std::size_t>(k), 0);
    }

    std::vector<float> total_sum;
    std::vector<int64_t> total_cnt;
    for (int64_t iter = 0; iter < iters; ++iter) {
        compute_centroid_norms(c.data(), k, dim, &c_norm2, threads);
        std::atomic<bool> changed{false};
        std::vector<double> inertia(static_cast<std::size_t>(threads), 0.0);
        parallel_for(n, threads, [&](int64_t b, int64_t e, int t) {
            double acc = 0.0;
            for (int64_t i = b; i < e; ++i) {
                const float* xi = x + i * dim;
                const float xn = x_norm2[static_cast<std::size_t>(i)];
                int64_t best = 0;
                float best_d = xn + c_norm2[0] - 2.0f * dot_f32(xi, c.data(), dim);
                for (int64_t j = 1; j < k; ++j) {
                    const float d = xn + c_norm2[static_cast<std::size_t>(j)] -
                                    2.0f * dot_f32(xi, c.data() + j * dim, dim);
                    if (d < best_d) {
                        best_d = d;
                        best = j;
                    }
                }
                assign[static_cast<std::size_t>(i)] = static_cast<int32_t>(best);
                if (static_cast<int32_t>(best) != prev[static_cast<std::size_t>(i)]) {
                    changed.store(true, std::memory_order_relaxed);
                }
                acc += static_cast<double>(best_d);
            }
            inertia[static_cast<std::size_t>(t)] = acc;
        });

        for (int t = 0; t < threads; ++t) {
            std::fill(sum_buf[static_cast<std::size_t>(t)].begin(),
                      sum_buf[static_cast<std::size_t>(t)].end(), 0.0f);
            std::fill(cnt_buf[static_cast<std::size_t>(t)].begin(),
                      cnt_buf[static_cast<std::size_t>(t)].end(), int64_t{0});
        }
        parallel_for(n, threads, [&](int64_t b, int64_t e, int t) {
            float* sums = sum_buf[static_cast<std::size_t>(t)].data();
            int64_t* cnts = cnt_buf[static_cast<std::size_t>(t)].data();
            for (int64_t i = b; i < e; ++i) {
                const int32_t j = assign[static_cast<std::size_t>(i)];
                cnts[j] += 1;
                const float* xi = x + i * dim;
                float* acc = sums + static_cast<int64_t>(j) * dim;
                for (int64_t d = 0; d < dim; ++d) {
                    acc[d] += xi[d];
                }
            }
        });
        total_sum.assign(static_cast<std::size_t>(k * dim), 0.0f);
        total_cnt.assign(static_cast<std::size_t>(k), 0);
        for (int t = 0; t < threads; ++t) {
            const float* s = sum_buf[static_cast<std::size_t>(t)].data();
            const int64_t* ct = cnt_buf[static_cast<std::size_t>(t)].data();
            for (int64_t j = 0; j < k; ++j) {
                total_cnt[static_cast<std::size_t>(j)] += ct[j];
                for (int64_t d = 0; d < dim; ++d) {
                    total_sum[static_cast<std::size_t>(j * dim + d)] += s[j * dim + d];
                }
            }
        }
        for (int64_t j = 0; j < k; ++j) {
            const int64_t cnt = total_cnt[static_cast<std::size_t>(j)];
            if (cnt <= 0) {
                continue;
            }
            const float inv = 1.0f / static_cast<float>(cnt);
            float* dst = c.data() + j * dim;
            const float* src = total_sum.data() + j * dim;
            for (int64_t d = 0; d < dim; ++d) {
                dst[d] = src[d] * inv;
            }
        }
        // 空簇拆分：从最大簇搬一个点到空簇（确定性；与 faiss split_clusters 细节不同）
        int64_t cursor = 0;
        for (int64_t j = 0; j < k; ++j) {
            if (total_cnt[static_cast<std::size_t>(j)] > 0) {
                continue;
            }
            int64_t donor = -1;
            int64_t donor_cnt = 1;
            for (int64_t jj = 0; jj < k; ++jj) {
                if (total_cnt[static_cast<std::size_t>(jj)] > donor_cnt) {
                    donor_cnt = total_cnt[static_cast<std::size_t>(jj)];
                    donor = jj;
                }
            }
            if (donor < 0) {
                break;
            }
            while (cursor < n && assign[static_cast<std::size_t>(cursor)] != donor) {
                ++cursor;
            }
            if (cursor >= n) {
                break;
            }
            assign[static_cast<std::size_t>(cursor)] = static_cast<int32_t>(j);
            total_cnt[static_cast<std::size_t>(donor)] -= 1;
            total_cnt[static_cast<std::size_t>(j)] = 1;
            std::memcpy(c.data() + j * dim, x + cursor * dim,
                        static_cast<std::size_t>(dim) * sizeof(float));
            ++cursor;
        }

        if (opt.verbose) {
            double inertia_total = 0.0;
            for (double v : inertia) {
                inertia_total += v;
            }
            std::printf("[kmeans] iter %lld/%lld inertia=%.3f\n", static_cast<long long>(iter + 1),
                        static_cast<long long>(iters), inertia_total);
        }
        if (!changed.load(std::memory_order_relaxed)) {
            break;
        }
        prev = assign;
    }

    *centroids = std::move(c);
    if (err) {
        err->clear();
    }
    return true;
}

bool minibatch_kmeans_f32(const float* x, int64_t n, int64_t dim, int64_t k,
                          const MiniBatchKMeansOptions& opt, std::vector<float>* centroids,
                          std::string* err) {
    if (x == nullptr || n <= 0 || dim <= 0 || k <= 0) {
        if (err) {
            *err = "minibatch-kmeans 输入非法";
        }
        return false;
    }
    if (k > n) {
        k = n;
    }
    const int threads = effective_threads(opt.threads, n);
    const int64_t batch = std::min<int64_t>(n, 256LL * static_cast<int64_t>(threads));
    const int64_t max_epochs = opt.max_epochs > 0 ? opt.max_epochs : 1;
    std::mt19937 rng(static_cast<uint32_t>(opt.seed));

    std::vector<int64_t> sample;
    sample_distinct_rows(n, k, &rng, &sample);
    std::vector<float> c(static_cast<std::size_t>(k * dim), 0.0f);
    for (int64_t j = 0; j < k; ++j) {
        std::memcpy(c.data() + j * dim, x + sample[static_cast<std::size_t>(j)] * dim,
                    static_cast<std::size_t>(dim) * sizeof(float));
    }
    std::vector<int64_t> counts(static_cast<std::size_t>(k), 1);
    std::vector<int64_t> order(static_cast<std::size_t>(n));
    std::iota(order.begin(), order.end(), int64_t{0});
    std::vector<float> buf(static_cast<std::size_t>(batch * dim), 0.0f);
    std::vector<int32_t> assign(static_cast<std::size_t>(batch), 0);

    double prev_inertia = -1.0;
    int64_t no_improve = 0;
    for (int64_t epoch = 0; epoch < max_epochs; ++epoch) {
        std::shuffle(order.begin(), order.end(), rng);
        double inertia = 0.0;
        for (int64_t start = 0; start < n; start += batch) {
            const int64_t m = std::min(batch, n - start);
            for (int64_t p = 0; p < m; ++p) {
                std::memcpy(buf.data() + p * dim, x + order[static_cast<std::size_t>(start + p)] * dim,
                            static_cast<std::size_t>(dim) * sizeof(float));
            }
            inertia += assign_to_centroids(buf.data(), m, dim, c.data(), k, &assign, threads);
            // 增量更新（官方 sklearn MiniBatchKMeans 轻量版：eta=1/count）
            for (int64_t p = 0; p < m; ++p) {
                const int64_t i = order[static_cast<std::size_t>(start + p)];
                const int32_t j = assign[static_cast<std::size_t>(p)];
                counts[static_cast<std::size_t>(j)] += 1;
                const float eta = 1.0f / static_cast<float>(counts[static_cast<std::size_t>(j)]);
                const float* xi = x + i * dim;
                float* cj = c.data() + static_cast<int64_t>(j) * dim;
                for (int64_t d = 0; d < dim; ++d) {
                    cj[d] += eta * (xi[d] - cj[d]);
                }
            }
        }
        if (opt.verbose) {
            std::printf("[minibatch-kmeans] epoch %lld/%lld inertia=%.3f\n",
                        static_cast<long long>(epoch + 1), static_cast<long long>(max_epochs),
                        inertia);
        }
        if (prev_inertia > 0.0) {
            const double imp = (prev_inertia - inertia) / prev_inertia;
            if (imp < 1e-3) {
                ++no_improve;
            } else {
                no_improve = 0;
            }
            if (no_improve >= 2) {
                if (opt.verbose) {
                    std::printf("[minibatch-kmeans] 提前停止（连续 2 轮改善 <0.1%%）\n");
                }
                break;
            }
        }
        prev_inertia = inertia;
    }

    *centroids = std::move(c);
    if (err) {
        err->clear();
    }
    return true;
}

} // namespace rvc_train::exporter
