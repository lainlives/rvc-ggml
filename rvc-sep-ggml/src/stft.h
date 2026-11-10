#pragma once
/**
 * stft.h - STFT/ISTFT implementation (Optimized)
 * 
 * Implements:
 * - Table-based Hann window generation
 * - Table-based Radix-2 FFT (Twiddle factors & Bit-reversal)
 * - Thread-safe Memory Pooling (STFTBuffer)
 * - Center padding (reflect mode)
 * - Frame extraction
 */

#include <cmath>
#include <vector>
#include <complex>
#include <cstring>
#include <algorithm>
#include <memory>
#include <mutex>

#ifdef USE_OPENMP
#include <omp.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace stft {

using Complex = std::complex<float>;

//=============================================================================
// Memory Pooling
//=============================================================================

/**
 * Thread-local buffer storage to avoid frequent allocations in STFT/ISTFT loops.
 */
struct STFTBuffer {
    // FFT buffers
    std::vector<Complex> fft_in;
    std::vector<Complex> fft_out;
    std::vector<Complex> fft_scratch;
    
    // Frame buffers
    std::vector<float> frame_in;
    std::vector<float> frame_out;
    
    // Window buffers
    std::vector<float> window_padded;
    std::vector<float> padded_audio;
    
    void Resize(int n_fft, int padded_len = 0) {
        if (fft_in.size() != n_fft) fft_in.resize(n_fft);
        if (fft_out.size() != n_fft) fft_out.resize(n_fft);
        if (fft_scratch.size() != n_fft) fft_scratch.resize(n_fft);
        if (frame_in.size() != n_fft) frame_in.resize(n_fft);
        if (frame_out.size() != n_fft) frame_out.resize(n_fft);
        if (window_padded.size() != n_fft) window_padded.resize(n_fft);
        if (padded_len > 0 && padded_audio.size() < padded_len) padded_audio.resize(padded_len);
    }
};

//=============================================================================
// Window Functions
//=============================================================================

inline void hann_window(float* out, int size, bool periodic = true) {
    int divisor = periodic ? size : (size - 1);
    for (int i = 0; i < size; ++i) {
        out[i] = 0.5f * (1.0f - std::cos(2.0f * static_cast<float>(M_PI) * i / divisor));
    }
}

//=============================================================================
// FFT Implementation (Table-based Cooley-Tukey Radix-2)
//=============================================================================

class TableFFT {
public:
    static TableFFT& GetInstance(int n_fft) {
        static std::mutex mtx;
        static std::unique_ptr<TableFFT> instance;
        static int current_n_fft = -1;

        std::lock_guard<std::mutex> lock(mtx);
        if (!instance || current_n_fft != n_fft) {
            instance = std::make_unique<TableFFT>(n_fft);
            current_n_fft = n_fft;
        }
        return *instance;
    }

    TableFFT(int n) : n_(n) {
        Precomputetables();
    }

    void Forward(Complex* data) const {
        BitReverse(data);
        Compute(data, false);
    }
    
    void Inverse(Complex* data) const {
        BitReverse(data);
        Compute(data, true);
        
        // Normalize
        float inv_n = 1.0f / n_;
        for (int i = 0; i < n_; ++i) {
            data[i] *= inv_n;
        }
    }

private:
    int n_;
    std::vector<int> bit_reverse_indices_;
    std::vector<Complex> twiddles_fwd_;
    std::vector<Complex> twiddles_inv_;

    void Precomputetables() {
        // 1. Bit Reverse
        bit_reverse_indices_.resize(n_);
        int j = 0;
        for (int i = 0; i < n_ - 1; ++i) {
            bit_reverse_indices_[i] = (i < j) ? j : i; // Store swap target
            int m = n_ >> 1;
            while (j >= m && m > 0) {
                j -= m;
                m >>= 1;
            }
            j += m;
        }
        bit_reverse_indices_[n_ - 1] = n_ - 1;

        // 2. Twiddles
        // We only need twiddles for len = 2, 4, 8 ... n
        // Total count is roughly N.
        // Structure: [len=2: w], [len=4: w, w^2], ...
        // Simplification: Store W_N^k for k=0..N/2-1.
        // Then step=N/len.
        twiddles_fwd_.resize(n_ / 2);
        twiddles_inv_.resize(n_ / 2);
        
        for (int k = 0; k < n_ / 2; ++k) {
            float angle = -2.0f * static_cast<float>(M_PI) * k / n_;
            twiddles_fwd_[k] = Complex(std::cos(angle), std::sin(angle));
            twiddles_inv_[k] = std::conj(twiddles_fwd_[k]);
        }
    }

    void BitReverse(Complex* data) const {
        for (int i = 0; i < n_; ++i) {
            int j = bit_reverse_indices_[i];
            if (i < j) {
                std::swap(data[i], data[j]);
            }
        }
    }

    void Compute(Complex* data, bool inverse) const {
        const auto& twiddles = inverse ? twiddles_inv_ : twiddles_fwd_;
        
        for (int len = 2; len <= n_; len <<= 1) {
            int half_len = len >> 1;
            int step = n_ / len;
            
            for (int i = 0; i < n_; i += len) {
                for (int j = 0; j < half_len; ++j) {
                    Complex w = twiddles[j * step];
                    Complex u = data[i + j];
                    Complex t = w * data[i + j + half_len];
                    data[i + j] = u + t;
                    data[i + j + half_len] = u - t;
                }
            }
        }
    }
};


//=============================================================================
// STFT Wrapper (Optimized)
//=============================================================================

inline void rfft(const float* input, Complex* output, int n, STFTBuffer& buffer) {
    // 1. Copy to complex buffer
    for (int i = 0; i < n; ++i) {
        buffer.fft_scratch[i] = Complex(input[i], 0.0f);
    }
    
    // 2. FFT
    TableFFT::GetInstance(n).Forward(buffer.fft_scratch.data());
    
    // 3. Copy first N/2 + 1
    int n_out = n / 2 + 1;
    for (int i = 0; i < n_out; ++i) {
        output[i] = buffer.fft_scratch[i];
    }
}

inline void irfft(const Complex* input, float* output, int n_out, STFTBuffer& buffer) {
    int n_freq = n_out / 2 + 1;
    
    // 1. Reconstruct full spectrum
    for (int i = 0; i < n_freq; ++i) {
        buffer.fft_scratch[i] = input[i];
    }
    for (int i = n_freq; i < n_out; ++i) {
        buffer.fft_scratch[i] = std::conj(buffer.fft_scratch[n_out - i]);
    }
    
    // 2. IFFT
    TableFFT::GetInstance(n_out).Inverse(buffer.fft_scratch.data());
    
    // 3. Real part
    for (int i = 0; i < n_out; ++i) {
        output[i] = buffer.fft_scratch[i].real();
    }
}

inline void compute_stft(
    const float* audio,
    int n_samples,
    int n_fft,
    int hop_length,
    int win_length,
    const float* window,
    bool center,
    float* output,
    int* n_frames_out
) {
    // Center padding
    int pad_amount = center ? n_fft / 2 : 0;
    int padded_len = n_samples + 2 * pad_amount;
    
    // Calculate number of frames
    // PyTorch formula: (L - N) / H + 1
    int n_frames = 1 + (padded_len - n_fft) / hop_length;
    if (n_frames < 0) n_frames = 0;
    *n_frames_out = n_frames;
    
    // Prepare padding buffer (thread-local or single allocation if not parallel? 
    // Padding + Windowing is usually fast, but padding needs full copy.)
    // For safety and simplicity, let's allocate padded audio once here (It's one large buffer).
    // The previous implementation used thread_local for 'padded_audio' which is wrong because 
    // 'padded_audio' needs to hold the WHOLE signal? No, stft.h:52 says 'padded_audio'.
    // Analyzing original code: It copied the WHOLE signal to 'padded_audio' inside compute_stft.
    // That means 'tls_buffer' was huge! If we have multiple threads, each copying full audio? 
    // That's wasteful.
    // Better: Allocate 'padded' once on heap.
    
    std::vector<float> padded(padded_len);
    if (center) {
        // Reflect padding
        for (int i = 0; i < pad_amount; ++i) {
            int src_idx = pad_amount - i;
            if (src_idx >= n_samples) src_idx = n_samples - 1;
            padded[i] = audio[src_idx];
        }
        if (n_samples > 0) {
            std::memcpy(padded.data() + pad_amount, audio, n_samples * sizeof(float));
        }
        for (int i = 0; i < pad_amount; ++i) {
            int src_idx = n_samples - 2 - i;
            if (src_idx < 0) src_idx = 0;
            padded[pad_amount + n_samples + i] = audio[src_idx];
        }
    } else {
        std::memcpy(padded.data(), audio, n_samples * sizeof(float));
    }

    int n_freq = n_fft / 2 + 1;
    
    // Prepare window (Single copy)
    std::vector<float> window_padded(n_fft, 0.0f);
    if (win_length < n_fft) {
        int left = (n_fft - win_length) / 2;
        std::memcpy(window_padded.data() + left, window, win_length * sizeof(float));
    } else {
        std::memcpy(window_padded.data(), window, n_fft * sizeof(float));
    }
    
    // Prepare thread buffers
    int max_threads = 1;
    #ifdef USE_OPENMP
    max_threads = omp_get_max_threads();
    #endif
    std::vector<STFTBuffer> thread_buffers(max_threads);
    for(auto& buf : thread_buffers) buf.Resize(n_fft);

    // Process each frame
    #ifdef USE_OPENMP
    #pragma omp parallel for
    #endif
    for (int f = 0; f < n_frames; ++f) {
        int tid = 0;
        #ifdef USE_OPENMP
        tid = omp_get_thread_num();
        #endif
        STFTBuffer& buffer = thread_buffers[tid];

        std::vector<float>& frame = buffer.frame_in;
        int start = f * hop_length;
        
        for (int i = 0; i < n_fft; ++i) {
            frame[i] = padded[start + i] * window_padded[i];
        }
        
        // Compute FFT
        // Output pointer directly to destination
        // We need a place to store complex output before writing to planar output
        
        rfft(frame.data(), buffer.fft_out.data(), n_fft, buffer);
        
        // Write to output
        for (int k = 0; k < n_freq; ++k) {
            output[(k * n_frames + f) * 2 + 0] = buffer.fft_out[k].real();
            output[(k * n_frames + f) * 2 + 1] = buffer.fft_out[k].imag();
        }
    }
}

inline void compute_istft(
    const float* stft_data,
    int n_freq,
    int n_frames,
    int n_fft,
    int hop_length,
    int win_length,
    const float* window,
    bool center,
    int length,
    float* output
) {
    // Calculate expected output signal length
    int expected_len = n_fft + hop_length * (n_frames - 1);
    int pad_amount = center ? n_fft / 2 : 0;
    int output_len = (length > 0) ? length : (expected_len - 2 * pad_amount);
    
    // Prepare padded window
    std::vector<float> window_padded(n_fft, 0.0f);
    if (win_length < n_fft) {
        int left = (n_fft - win_length) / 2;
        std::memcpy(window_padded.data() + left, window, win_length * sizeof(float));
    } else {
        std::memcpy(window_padded.data(), window, n_fft * sizeof(float));
    }
    
    // Prepare thread buffers
    int max_threads = 1;
    #ifdef USE_OPENMP
    max_threads = omp_get_max_threads();
    #endif
    std::vector<STFTBuffer> thread_buffers(max_threads);
    for(auto& buf : thread_buffers) buf.Resize(n_fft);

    // Step 1: Compute all IFFTs in parallel
    std::vector<float> frames_time_domain(n_frames * n_fft);
    
    #ifdef USE_OPENMP
    #pragma omp parallel for
    #endif
    for (int f = 0; f < n_frames; ++f) {
        int tid = 0;
        #ifdef USE_OPENMP
        tid = omp_get_thread_num();
        #endif
        STFTBuffer& buffer = thread_buffers[tid];
        
        std::vector<Complex>& fft_in = buffer.fft_in;
        std::vector<float>& frame_out = buffer.frame_out;
        
        // Extract complex spectrum
        for (int k = 0; k < n_freq; ++k) {
            float re = stft_data[(k * n_frames + f) * 2 + 0];
            float im = stft_data[(k * n_frames + f) * 2 + 1];
            fft_in[k] = Complex(re, im);
        }
        
        // IFFT
        irfft(fft_in.data(), frame_out.data(), n_fft, buffer);
        
        // Store
        std::memcpy(&frames_time_domain[f * n_fft], frame_out.data(), n_fft * sizeof(float));
    }
    
    // Step 2: Overlap Add (Serial)
    std::vector<float> y(expected_len, 0.0f);
    std::vector<float> window_sum(expected_len, 0.0f);
    
    for (int f = 0; f < n_frames; ++f) {
        int start = f * hop_length;
        const float* frame_ptr = &frames_time_domain[f * n_fft];
        
        for (int i = 0; i < n_fft; ++i) {
            y[start + i] += frame_ptr[i] * window_padded[i];
            window_sum[start + i] += window_padded[i] * window_padded[i];
        }
    }
    
    // Normalize by window sum (avoid division by zero)
    for (int i = 0; i < expected_len; ++i) {
        if (window_sum[i] > 1e-8f) {
            y[i] /= window_sum[i];
        }
    }
    
    // Remove center padding and copy to output
    for (int i = 0; i < output_len; ++i) {
        if (pad_amount + i < expected_len) {
             output[i] = y[pad_amount + i];
        } else {
             output[i] = 0.0f;
        }
    }
}

} // namespace stft
