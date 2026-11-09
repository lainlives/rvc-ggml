#!/usr/bin/env python3
# verify_audio.py — P0 audio pipeline numerical comparison
# Recompute high-pass/RMS in Python (numpy/scipy/librosa) and compare sample by sample with the C++ self-test output.
# The resampler implementation differs from the Python reference (polyphase FIR vs ffmpeg/torchaudio),
# so it is only a quality check (correlation with scipy.resample_poly), not a sample-by-sample criterion.
#
# Usage:
#   <python> scripts/verify_audio.py <input_wav> <selftest_dir>

import os
import sys
import wave

import numpy as np
import scipy.signal as ss

try:
    import librosa

    def frame_rms_librosa(y, sr):
        return librosa.feature.rms(y=y, frame_length=sr // 2 * 2, hop_length=sr // 2)
except Exception:
    def frame_rms_librosa(y, sr):
        frame_length = sr // 2 * 2
        hop = sr // 2
        n = 1 + (len(y) - frame_length) // hop
        return np.array([
            np.sqrt(np.mean(np.abs(y[i * hop: i * hop + frame_length]) ** 2))
            for i in range(n)
        ])[None, :]

RVC_TGT_SR = 16000


def read_wav_mono_float(path):
    with wave.open(path, "rb") as w:
        n_ch = w.getnchannels()
        sr = w.getframerate()
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    data = data.astype(np.float32) / 32768.0
    if n_ch > 1:
        data = data.reshape(-1, n_ch).mean(axis=1)
    return data, sr


def compare(name, ref, test, atol):
    n = min(len(ref), len(test))
    if n == 0:
        print(f"  {name}: empty input, skipping")
        return False
    diff = np.abs(ref[:n] - test[:n])
    max_err = float(diff.max())
    ok = max_err <= atol
    flag = "OK  " if ok else "FAIL"
    print(f"  [{flag}] {name}: max_err={max_err:.3e} (atol={atol:.0e}, n={n})")
    return ok


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    input_wav, selftest_dir = sys.argv[1], sys.argv[2]

    print("== P0 audio pipeline verification ==")
    audio, sr = read_wav_mono_float(input_wav)
    print(f"input: {input_wav} sr={sr} n={len(audio)}")

    a_mono = np.fromfile(os.path.join(selftest_dir, "a_mono16k.raw"), dtype=np.float32)
    print(f"C++ mono16k: n={len(a_mono)}")

    # 1) Resample quality: correlation with scipy.resample_poly
    ref_poly = ss.resample_poly(audio, RVC_TGT_SR, sr).astype(np.float32)
    n = min(len(ref_poly), len(a_mono))
    corr = float(np.corrcoef(ref_poly[:n], a_mono[:n])[0, 1])
    print(f"  [INFO] resample quality corr(vs resample_poly)={corr:.6f}")
    ok_resample = corr > 0.99

    # 2) High-pass: recompute using the C++ mono16k output as input (isolates the filter logic)
    # Note: the boundary padding details of scipy filtfilt differ slightly from our odd-extension
    # implementation (affects only the last ~1000 samples; the interior is sample-by-sample identical). RVC semantics trim the boundaries.
    b, a = ss.butter(N=5, Wn=48, btype="high", fs=16000)
    hp_ref = ss.filtfilt(b, a, a_mono.astype(np.float64)).astype(np.float32)
    b_hp = np.fromfile(os.path.join(selftest_dir, "b_hp48.raw"), dtype=np.float32)
    ok_hp = compare("high-pass 48Hz (filtfilt)", hp_ref, b_hp, 5e-3)
    if ok_hp:
        interior = slice(100, -1200)
        err_i = float(np.abs(hp_ref[interior] - b_hp[interior]).max())
        ok_hp = err_i <= 1e-4
        print(f"  [{'OK  ' if ok_hp else 'FAIL'}] high-pass interior samples: max_err={err_i:.3e} (atol=1e-4)")

    # 3) RMS: likewise using the C++ output as input (float32 accumulation differences, tolerance 5e-5)
    rms_ref = frame_rms_librosa(a_mono.astype(np.float64), 16000)[0].astype(np.float32)
    d_rms = np.fromfile(os.path.join(selftest_dir, "d_rms16k.raw"), dtype=np.float32)
    ok_rms = compare("RMS frames", rms_ref, d_rms, 5e-5)

    all_ok = ok_resample and ok_hp and ok_rms
    print("== Result:", "all passed" if all_ok else "failures present", "==")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
