#!/usr/bin/env python3
# verify_gguf.py — verify consistency between RVC-GGML exported GGUF files and the original PyTorch weights
#
# Usage:
#   <python> scripts/verify_gguf.py hubert --gguf models/hubert.gguf --pth <hubert_base.pt>
#   <python> scripts/verify_gguf.py rmvpe  --gguf models/rmvpe.gguf  --pth <rmvpe.pt>
#   <python> scripts/verify_gguf.py model  --gguf models/soyo.gguf   --pth <soyo.pth>
#   <python> scripts/verify_gguf.py index  --gguf models/x.index.gguf --index <x.index>

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from gguf import GGUFReader
from gguf.quants import dequantize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_gguf import fold_weight_norm  # noqa: E402


def expected_layout(name: str, ref: torch.Tensor) -> np.ndarray:
    """GGUF (ggml ne order) matches the PyTorch shape; flattened layouts (rvc.conv_flat) merge the last dimension."""
    return np.ascontiguousarray(ref.float().numpy())


def read_gguf_tensor(t, ref_shape) -> np.ndarray:
    """Read a tensor: dequantize quantized types; restore flattened layouts to the PyTorch shape."""
    type_name = t.tensor_type.name
    if type_name in ("F32", "F16"):
        got = np.asarray(t.data).astype(np.float32)
    else:
        got = np.asarray(dequantize(t.data, t.tensor_type)).astype(np.float32)
    if tuple(got.shape) != tuple(ref_shape) and got.size == int(np.prod(ref_shape)):
        got = got.reshape(ref_shape)
    return got


def tolerance_for(type_name: str):
    """Returns (atol, corr_min): more aggressive quant levels use larger tolerances."""
    if type_name == "F32":
        return 1e-5, 0.999999
    if type_name == "F16":
        return 1e-2, 0.9999
    if type_name == "Q8_0":
        return 2e-2, 0.9999
    if type_name in ("Q5_0", "Q5_1"):
        return 5e-2, 0.999
    return 2e-1, 0.99   # Q4_0 / Q4_1


def check(name, ref: np.ndarray, got: np.ndarray, atol=1e-5, corr_min=0.0, enforce_atol=True):
    if tuple(ref.shape) != tuple(got.shape):
        print(f"  [FAIL] {name}: shape {ref.shape} vs {got.shape}")
        return False
    ref = ref.astype(np.float32)
    got = got.astype(np.float32)
    err = np.abs(ref - got).max()
    ref_f = ref.reshape(-1)
    got_f = got.reshape(-1)
    if ref_f.size > 1 and ref_f.std() > 0 and got_f.std() > 0:
        corr = float(np.corrcoef(ref_f, got_f)[0, 1])
    else:
        corr = 1.0
    # Quant levels are judged by correlation (per-tensor max_err is strongly affected by outlier weights); F32/F16 use atol
    ok = (err <= atol if enforce_atol else True) and corr >= corr_min
    flag = "OK  " if ok else "FAIL"
    print(f"  [{flag}] {name}: max_err={err:.3e} corr={corr:.8f} shape={tuple(ref.shape)}")
    return ok


def verify_state(gguf_path: str, state: dict, skip_suffix=(".num_batches_tracked",)) -> bool:
    reader = GGUFReader(gguf_path)
    reader_tensors = {t.name: t for t in reader.tensors}
    print(f"GGUF tensor count: {len(reader_tensors)}")
    ok = True
    checked = 0
    for key, value in state.items():
        if key.endswith(skip_suffix):
            continue
        if key not in reader_tensors:
            print(f"  [FAIL] missing tensor: {key}")
            ok = False
            continue
        t = reader_tensors[key]
        ref = expected_layout(key, value)
        got = read_gguf_tensor(t, ref.shape)
        atol, corr_min = tolerance_for(t.tensor_type.name)
        enforce_atol = t.tensor_type.name in ("F32", "F16")
        if not check(key, ref, got, atol=atol, corr_min=corr_min, enforce_atol=enforce_atol):
            ok = False
        checked += 1
    print(f"checked {checked} tensors")
    return ok


def verify_hubert(gguf_path: str, pth_path: str) -> bool:
    ckpt = torch.load(pth_path, map_location="cpu", weights_only=False)
    state = fold_weight_norm(ckpt["model"])
    return verify_state(gguf_path, state)


def verify_rmvpe(gguf_path: str, pth_path: str) -> bool:
    state = torch.load(pth_path, map_location="cpu", weights_only=False)
    ok = verify_state(gguf_path, state)
    reader = GGUFReader(gguf_path)
    names = {t.name for t in reader.tensors}
    if "buffer_mel_basis" in names:
        t = next(x for x in reader.tensors if x.name == "buffer_mel_basis")
        print(f"  [OK  ] buffer_mel_basis: shape={t.data.shape}")
    else:
        print("  [FAIL] missing buffer_mel_basis")
        ok = False
    return ok


def verify_model(gguf_path: str, pth_path: str) -> bool:
    ckpt = torch.load(pth_path, map_location="cpu", weights_only=False)
    state = fold_weight_norm(ckpt["weight"])
    return verify_state(gguf_path, state)


def verify_index(gguf_path: str, index_path: str) -> bool:
    import faiss
    index = faiss.read_index(index_path)
    reader = GGUFReader(gguf_path)
    rt = {t.name: t for t in reader.tensors}
    ok = True
    centroids_ref = np.ascontiguousarray(
        faiss.downcast_index(index.quantizer).reconstruct_n(0, index.nlist), dtype=np.float32)
    offsets_ref = np.zeros(index.nlist, dtype=np.float32)
    lengths_ref = np.zeros(index.nlist, dtype=np.float32)
    vectors_list = []
    cursor = 0
    for list_id in range(index.nlist):
        size = int(index.invlists.list_size(list_id))
        offsets_ref[list_id] = cursor
        lengths_ref[list_id] = size
        if size:
            codes = faiss.rev_swig_ptr(index.invlists.get_codes(list_id), size * index.code_size)
            vectors_list.append(np.frombuffer(codes, dtype=np.float32).reshape(size, index.d).copy())
        cursor += size
    vectors_ref = np.concatenate(vectors_list, axis=0) if vectors_list else np.zeros((0, index.d), np.float32)

    ok &= check("index.centroids", centroids_ref, rt["index.centroids"].data.astype(np.float32), atol=1e-6)
    ok &= check("index.vectors", vectors_ref, rt["index.vectors"].data.astype(np.float32), atol=1e-6)
    ok &= check("index.list_offsets", offsets_ref, rt["index.list_offsets"].data.astype(np.float32), atol=1e-6)
    ok &= check("index.list_lengths", lengths_ref, rt["index.list_lengths"].data.astype(np.float32), atol=1e-6)
    return ok


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("hubert", "rmvpe", "model"):
        p = sub.add_parser(name)
        p.add_argument("--gguf", required=True)
        p.add_argument("--pth", required=True)
    p = sub.add_parser("index")
    p.add_argument("--gguf", required=True)
    p.add_argument("--index", required=True)
    args = parser.parse_args()

    if args.cmd == "hubert":
        ok = verify_hubert(args.gguf, args.pth)
    elif args.cmd == "rmvpe":
        ok = verify_rmvpe(args.gguf, args.pth)
    elif args.cmd == "model":
        ok = verify_model(args.gguf, args.pth)
    else:
        ok = verify_index(args.gguf, args.index)
    print("== Result:", "PASS" if ok else "FAIL", "==")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
