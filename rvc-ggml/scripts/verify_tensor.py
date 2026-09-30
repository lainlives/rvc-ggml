#!/usr/bin/env python3
# verify_tensor.py — 通用 raw float32 张量对比（C++ 输出 vs Python 参考）
#
# 用法:
#   <python> scripts/verify_tensor.py --ref output/ref/hubert_v1_finalproj.raw \
#       --test output/ref/cpp_hubert_v1_finalproj.raw --shape 249,256
#   <python> scripts/verify_tensor.py --ref a.raw --test b.raw --shape 249,768 --tolerance 1e-3

from __future__ import annotations

import argparse
import sys

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--shape", required=True, help="逗号分隔，如 249,256")
    parser.add_argument("--tolerance", type=float, default=1e-4, help="max abs err 容差")
    parser.add_argument("--corr-min", type=float, default=0.99, help="最小相关系数")
    args = parser.parse_args()

    shape = tuple(int(x) for x in args.shape.split(","))
    ref = np.fromfile(args.ref, dtype=np.float32)
    test = np.fromfile(args.test, dtype=np.float32)
    if ref.size != test.size:
        print(f"[FAIL] 元素数不一致: ref={ref.size} test={test.size}")
        return 1
    if ref.size != int(np.prod(shape)):
        print(f"[WARN] 元素数 {ref.size} 与 shape {shape} 不符")

    ref = ref.reshape(shape)
    test = test.reshape(shape)
    err = np.abs(ref - test)
    max_err = float(err.max())
    mean_err = float(err.mean())
    corr = float(np.corrcoef(ref.reshape(-1), test.reshape(-1))[0, 1]) if ref.size > 1 else 1.0

    ok = max_err <= args.tolerance and corr >= args.corr_min
    print(f"[{'OK  ' if ok else 'FAIL'}] {args.test}")
    print(f"  max_err={max_err:.3e} mean_err={mean_err:.3e} corr={corr:.8f} "
          f"(tolerance={args.tolerance}, corr_min={args.corr_min})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
