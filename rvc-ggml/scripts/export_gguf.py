#!/usr/bin/env python3
# export_gguf.py — RVC-GGML model conversion tool
#
# Convert RVC ecosystem model artifacts to GGUF (for use by the RVC-GGML ggml inference core):
#   1. hubert:  assets/hubert/hubert_base.pt      -> hubert.gguf  (fairseq HuBERT/ContentVec)
#   2. rmvpe:   assets/rmvpe/rmvpe.pt             -> rmvpe.gguf   (E2E pitch model, includes the mel filterbank)
#   3. model:   assets/weights/*.pth              -> <name>.gguf  (SynthesizerTrnMs v1/v2, f0/non-f0)
#   4. index:   logs/*.index                      -> <name>.index.gguf (FAISS IVF/Flat retrieval index)
#
# Mixed quantization policy (--dtype):
#   fp32 / fp16 / q8_0 / q5_1 / q4_1 / q4_0
#   Numerically sensitive tensors are always F32: bias, norm/gamma/beta, embedding, GRU, BatchNorm statistics,
#   sine-source lookup (emb_pitch), speaker table (emb_g), flow coupling layers, transposed convolutions not natively supported by Vulkan, etc.;
#   remaining quantizable tensors are quantized per --dtype; when quantization is not feasible it falls back to F16 (not F32).
#
# Flattened convolution weight layout (rvc.conv_flat=true, layout version 2):
#   the row width (ne0) of a ggml quantized tensor must be a multiple of the block size (32), but Conv1d/Conv2d weight
#   kernel dimensions do not satisfy this; so at export time convolution weights are flattened to matrices with row width = input channels × kernel area:
#     Conv1d [O, I, K]      -> (O, I*K)         (im2col + quantized matrix multiply on the graph side)
#     Conv2d [O, I, KH, KW] -> (O, I*KH*KW)     (ConvTranspose2d is not flattened; stays F32)
#   the consumer side (C++) recognizes this via the rvc.conv_flat metadata; the old layout (3D/4D) remains supported.
#
# Usage:
#   python export_gguf.py hubert --input assets/hubert/hubert_base.pt --out models/hubert.gguf
#   python export_gguf.py rmvpe  --input assets/rmvpe/rmvpe.pt --out models/rmvpe.gguf
#   python export_gguf.py model  --input assets/weights/soyo.pth --out models/soyo.gguf --dtype fp16
#   python export_gguf.py index  --input logs/soyo.index --out models/soyo.index.gguf
#   python export_gguf.py all    --rvc-root <RVC upstream repo root> --out models --dtype fp16

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Some torch/BLAS builds (observed with a ROCm torch on Python 3.14) segfault when
# scipy's BLAS extensions are loaded after torch; preloading scipy.signal first
# avoids the dynamic-loader conflict. Missing scipy is fine (numpy mel fallback).
try:
    import scipy.signal  # noqa: F401
except ImportError:
    pass

import torch

try:
    from gguf import GGUFWriter
    from gguf.quants import GGMLQuantizationType, quantize
except ImportError:
    print(
        "Missing gguf package, please install first: pip install gguf -i https://pypi.tuna.tsinghua.edu.cn/simple"
    )
    raise


# ============================================================================
# Common utilities
# ============================================================================

DTYPE_MAP = {
    "f32": GGMLQuantizationType.F32,
    "fp32": GGMLQuantizationType.F32,
    "f16": GGMLQuantizationType.F16,
    "fp16": GGMLQuantizationType.F16,
    "q8_0": GGMLQuantizationType.Q8_0,
    "q5_0": GGMLQuantizationType.Q5_0,
    "q5_1": GGMLQuantizationType.Q5_1,
    "q4_0": GGMLQuantizationType.Q4_0,
    "q4_1": GGMLQuantizationType.Q4_1,
}

FILE_TYPE_ID = {
    GGMLQuantizationType.F32: 0,
    GGMLQuantizationType.F16: 1,
    GGMLQuantizationType.Q4_0: 2,
    GGMLQuantizationType.Q4_1: 3,
    GGMLQuantizationType.Q8_0: 7,
    GGMLQuantizationType.Q5_0: 8,
    GGMLQuantizationType.Q5_1: 9,
}


def parse_dtype(name: str) -> GGMLQuantizationType:
    key = name.lower()
    if key not in DTYPE_MAP:
        raise ValueError(
            f"Unsupported quantization type: {name} (available: {', '.join(sorted(DTYPE_MAP))})"
        )
    return DTYPE_MAP[key]


def load_torch(path: str) -> Any:
    return torch.load(path, map_location="cpu", weights_only=False)


def tensor_np(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().contiguous().numpy()


def fold_weight_norm(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Fold all weight_g/weight_v into weight.

    Two conventions:
      1. standard PyTorch weight_norm(Conv)  : weight_g [dim0, 1, 1], normalized along dim0 (RVC synthesizer)
      2. fairseq Conv1d weight_norm(dim=2): weight_g [1, 1, k], normalized along dim0/dim1 (HuBERT position convolution)
    """
    out: dict[str, torch.Tensor] = {}
    v_keys = [k for k in state if k.endswith(".weight_v")]
    for key, value in state.items():
        if key.endswith(".weight_g") or key.endswith(".weight_v"):
            continue
        out[key] = value
    for v_key in v_keys:
        prefix = v_key[: -len(".weight_v")]
        g_key = prefix + ".weight_g"
        if g_key not in state:
            raise RuntimeError(f"unpaired weight_norm tensors: {prefix}")
        v = state[v_key].float()
        g = state[g_key].float()

        if g.numel() == v.shape[0]:
            # Convention 1 (dim0 normalization)
            dim0 = v.shape[0]
            rest = int(np.prod(v.shape[1:]))
            v2 = v.reshape(dim0, rest)
            norm = v2.norm(dim=1, keepdim=True).clamp_min(1e-12)
            folded = (v2 / norm) * g.reshape(dim0, 1)
            out[prefix + ".weight"] = folded.reshape(v.shape).contiguous()
        else:
            # Convention 2 (fairseq dim=(0,1) normalization, norm broadcast to the last dimension)
            norm = v.norm(dim=(0, 1), keepdim=True).clamp_min(1e-12)
            folded = v / norm * g.reshape(*([1] * (v.dim() - 1)), -1)
            out[prefix + ".weight"] = folded.contiguous()
    return out


# Note (important):
#   the gguf Python library writes to disk in numpy shape (internally storing dims last-dim-first in C order),
#   and the ggml C API's ggml_tensor->ne reads exactly the same dims sequence.
#   Therefore PyTorch's weight storage shape == ggml ne order, with no transposition needed:
#     Linear  [N, K]  -> ne0=K, ne1=N   (ggml_mul_mat(W,K)... W viewed as [K,N])
#     Conv1d  [O, I, K] -> ne=[K, I, O] (used directly by ggml_conv_1d)
#     ConvT1d [I, O, K] -> ne=[K, O, I] (used directly by ggml_conv_transpose_1d)
#     Conv2d  [O, I, KH, KW] -> ne=[KW, KH, I, O]
#   Note that the gguf Python reader displays stored dims reversed (t.shape is the opposite of ne),
#   so during verification rely on the numpy shape of t.data (identical to the PyTorch shape).


def write_tensor_mixed(
    writer: GGUFWriter,
    name: str,
    data: np.ndarray,
    qtype: GGMLQuantizationType,
    policy: str,
    stats: dict,
) -> None:
    """Write a single tensor.

    policy:
      "f32"   — numerically sensitive, always F32
      "f16"   — force half precision
      "quant" — quantize per --dtype; falls back to F16 on failure or when qtype=F32
    """
    data = np.ascontiguousarray(data, dtype=np.float32)
    if policy == "f32":
        writer.add_tensor(name, data)
        stats["f32"] += 1
        return
    if policy == "f16":
        writer.add_tensor(name, data.astype(np.float16))
        stats["f16"] += 1
        return
    if qtype != GGMLQuantizationType.F32:
        try:
            quantized = quantize(data, qtype)
            writer.add_tensor(name, quantized, raw_dtype=qtype)
            stats["quantized"] += 1
            return
        except Exception as e:  # noqa: BLE001
            stats["warnings"].append(
                f"{name}: quantization failed ({e}), falling back to F16"
            )
    writer.add_tensor(name, data.astype(np.float16))
    stats["f16"] += 1


def flatten_conv1d(data: np.ndarray) -> np.ndarray:
    """Conv1d [O, I, K] -> (O, I*K); row-major memory order is preserved (i slow, k fast)."""
    o, i, k = data.shape
    return np.ascontiguousarray(data.reshape(o, i * k))


def flatten_conv2d(data: np.ndarray) -> np.ndarray:
    """Conv2d [O, I, KH, KW] -> (O, I*KH*KW); i slow, (kh,kw) fast, matching the im2col column order."""
    o, i, kh, kw = data.shape
    return np.ascontiguousarray(data.reshape(o, i * kh * kw))


def new_stats() -> dict:
    return {"quantized": 0, "f32": 0, "f16": 0, "warnings": []}


def finish_writer(writer: GGUFWriter, path: str) -> float:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return os.path.getsize(path) / (1024 * 1024)


def print_stats(label: str, stats: dict, size_mb: float) -> None:
    for w in stats["warnings"]:
        print(f"  [WARN] {w}")
    print(
        f"  {label}: quantized {stats['quantized']} / F16 {stats['f16']} / F32 {stats['f32']}, "
        f"file {size_mb:.1f} MB"
    )


# ============================================================================
# 1. HuBERT / ContentVec
# ============================================================================

HUBERT_F32_SUFFIXES = (".bias", ".gamma", ".beta")


def hubert_should_flatten(name: str, data: np.ndarray) -> bool:
    # Only the frontend convolutions (which take the vits::conv1d flattened path); pos_conv is a grouped convolution and keeps its original layout
    return (
        data.ndim == 3
        and name.endswith(".weight")
        and name.startswith("feature_extractor.conv_layers.")
    )


def hubert_policy(name: str, data: np.ndarray) -> str:
    # Numerically sensitive: LayerNorm / projection / position convolution stay F32; the remaining convolutions and attention linear layers can be quantized
    if data.ndim == 1:
        return "f32"
    if name.endswith(HUBERT_F32_SUFFIXES):
        return "f32"
    if "layer_norm" in name or "final_proj" in name:
        return "f32"
    if name.startswith("encoder.pos_conv"):
        return "f32"
    if name.endswith(".weight"):
        return "quant"
    return "f32"


def remap_hf_hubert_keys(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Remap HuggingFace Wav2Vec2/HuBERT/ContentVec key names to the fairseq-style names
    used by the RVC-GGML HuBERT module (RVC upstream ships fairseq checkpoints).
    Unmatched keys pass through unchanged, so fairseq checkpoints are unaffected."""
    import re

    out: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        k = key
        # conv feature extractor: HF "conv"/"layer_norm" -> fairseq Sequential slots (0=conv, 2=group norm)
        k = re.sub(
            r"^feature_extractor\.conv_layers\.(\d+)\.conv\.",
            r"feature_extractor.conv_layers.\1.0.",
            k,
        )
        k = re.sub(
            r"^feature_extractor\.conv_layers\.(\d+)\.layer_norm\.",
            r"feature_extractor.conv_layers.\1.2.",
            k,
        )
        # feature projection -> fairseq layer_norm (pre-projection) + post_extract_proj
        k = k.replace("feature_projection.layer_norm.", "layer_norm.")
        k = k.replace("feature_projection.projection.", "post_extract_proj.")
        # positional convolution
        k = k.replace("encoder.pos_conv_embed.conv.", "encoder.pos_conv.0.")
        # transformer layers
        k = re.sub(
            r"^encoder\.layers\.(\d+)\.attention\.", r"encoder.layers.\1.self_attn.", k
        )
        k = re.sub(
            r"^encoder\.layers\.(\d+)\.layer_norm\.",
            r"encoder.layers.\1.self_attn_layer_norm.",
            k,
        )
        k = re.sub(
            r"^encoder\.layers\.(\d+)\.feed_forward\.intermediate_dense\.",
            r"encoder.layers.\1.fc1.",
            k,
        )
        k = re.sub(
            r"^encoder\.layers\.(\d+)\.feed_forward\.output_dense\.",
            r"encoder.layers.\1.fc2.",
            k,
        )
        out[k] = value
    return out


def export_hubert(input_path: str, out_path: str, dtype: str) -> None:
    import json

    print(f"[hubert] loading {input_path}")
    ckpt = load_torch(input_path)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    cfg = ckpt.get("cfg", {}) if isinstance(ckpt, dict) else {}
    model_cfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}

    # HuggingFace-format checkpoints use different key names; remap to the fairseq
    # layout the RVC-GGML HuBERT module expects (fairseq checkpoints are unaffected).
    state = remap_hf_hubert_keys(state)

    # Architecture sizes: fairseq cfg > sibling config.json (HF) > defaults
    hf_cfg: dict = {}
    if not model_cfg:
        cfg_path = Path(input_path).with_name("config.json")
        if cfg_path.exists():
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    hf_cfg = json.load(f)
            except (OSError, ValueError):
                hf_cfg = {}

    layers = int(model_cfg.get("encoder_layers", hf_cfg.get("num_hidden_layers", 12)))
    hidden = int(model_cfg.get("encoder_embed_dim", hf_cfg.get("hidden_size", 768)))
    ffn = int(
        model_cfg.get("encoder_ffn_embed_dim", hf_cfg.get("intermediate_size", 3072))
    )
    heads = int(
        model_cfg.get("encoder_attention_heads", hf_cfg.get("num_attention_heads", 12))
    )
    conv_pos = int(model_cfg.get("conv_pos", 128))
    conv_pos_groups = int(model_cfg.get("conv_pos_groups", 16))
    final_dim = int(model_cfg.get("final_dim", 256))

    # Derive position-conv kernel/groups and the final-projection width from the tensors
    # (covers both naming schemes and HF exports missing cfg fields).
    for k, v in state.items():
        if not torch.is_tensor(v):
            continue
        if k.endswith(".weight_v") and "pos_conv" in k:
            conv_pos = int(v.shape[-1])
            if v.dim() == 3 and v.shape[1] > 0:
                conv_pos_groups = max(1, hidden // int(v.shape[1]))
        elif k.endswith("final_proj.weight"):
            final_dim = int(v.shape[0])

    print(
        f"  config: layers={layers} hidden={hidden} ffn={ffn} heads={heads} "
        f"conv_pos={conv_pos}/{conv_pos_groups} final_dim={final_dim}"
    )

    state = fold_weight_norm(state)

    qtype = parse_dtype(dtype)
    writer = GGUFWriter(out_path, "hubert")
    writer.add_name("RVC-GGML HuBERT/ContentVec")
    writer.add_description("fairseq HuBERT base for RVC content features")
    writer.add_file_type(FILE_TYPE_ID.get(qtype, 0))
    if qtype != GGMLQuantizationType.F32:
        writer.add_quantization_version(2)

    kp = "hubert."
    writer.add_uint32(kp + "n_layers", layers)
    writer.add_uint32(kp + "hidden_size", hidden)
    writer.add_uint32(kp + "intermediate_size", ffn)
    writer.add_uint32(kp + "n_heads", heads)
    writer.add_uint32(kp + "conv_pos", conv_pos)
    writer.add_uint32(kp + "conv_pos_groups", conv_pos_groups)
    writer.add_uint32(kp + "final_dim", final_dim)
    writer.add_bool("rvc.conv_flat", True)

    stats = new_stats()
    n = 0
    for key, value in state.items():
        if not torch.is_tensor(value):
            continue
        if key.endswith(".num_batches_tracked"):
            continue
        if value.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            continue
        data = tensor_np(value).astype(np.float32)
        policy = hubert_policy(key, data)
        if hubert_should_flatten(key, data):
            data = flatten_conv1d(data)
        write_tensor_mixed(writer, key, data, qtype, policy, stats)
        n += 1

    size_mb = finish_writer(writer, out_path)
    print(f"  done: {n} tensors")
    print_stats("[hubert]", stats, size_mb)


# ============================================================================
# 2. RMVPE
# ============================================================================

RMVPE_F32_MARKERS = (
    "running_mean",
    "running_var",
    "num_batches_tracked",
    "bn.",
    "shortcut",
    "gru.",
    "fc.",
    "conv1.0",
)


def rmvpe_should_flatten(name: str, data: np.ndarray) -> bool:
    # ConvTranspose2d (decoder upsampling conv1.0) is not flattened (native Vulkan operators only support F32)
    if data.ndim != 4 or not name.endswith(".weight"):
        return False
    if "conv1.0" in name:
        return False
    return True


def rmvpe_policy(name: str, data: np.ndarray) -> str:
    if data.ndim == 1:
        return "f32"
    if name.endswith(".bias"):
        return "f32"
    for marker in RMVPE_F32_MARKERS:
        if marker in name:
            return "f32"
    if data.ndim == 2:
        return "f32"  # GRU / head, etc.; stays F32
    if name.endswith(".weight"):
        return "quant"
    return "f32"


def make_rmvpe_mel_basis(
    sample_rate=16000, n_fft=1024, n_mels=128, fmin=30.0, fmax=8000.0
) -> np.ndarray:
    """Mel filterbank.

    Prefer librosa.filters.mel(htk=True) (fully consistent with RVC upstream MelSpectrogram);
    if librosa is unavailable, use an equivalent hand-written numpy implementation (Slaney area normalization + triangular interpolation).
    """
    try:
        import librosa

        return np.asarray(
            librosa.filters.mel(
                sr=sample_rate,
                n_fft=n_fft,
                n_mels=n_mels,
                fmin=fmin,
                fmax=fmax,
                htk=True,
            ),
            dtype=np.float32,
        )
    except ImportError:
        pass

    def hz_to_mel(hz):
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    n_bins = n_fft // 2 + 1
    mels = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    freqs = mel_to_hz(mels)
    fft_freqs = np.arange(n_bins, dtype=np.float64) * sample_rate / n_fft

    basis = np.zeros((n_mels, n_bins), dtype=np.float64)
    for m in range(n_mels):
        lower, center, upper = freqs[m], freqs[m + 1], freqs[m + 2]
        for k in range(n_bins):
            f = fft_freqs[k]
            if lower <= f <= center and center > lower:
                basis[m, k] = (f - lower) / (center - lower)
            elif center < f <= upper and upper > center:
                basis[m, k] = (upper - f) / (upper - center)
        # Slaney area normalization (matching the librosa htk=True default)
        basis[m] *= 2.0 / (freqs[m + 2] - freqs[m])
    return basis.astype(np.float32)


def export_rmvpe(input_path: str, out_path: str, dtype: str) -> None:
    print(f"[rmvpe] loading {input_path}")
    state = load_torch(input_path)
    if not isinstance(state, dict):
        state = {"model": state}
    if "model" in state and isinstance(state["model"], dict):
        state = state["model"]

    qtype = parse_dtype(dtype)
    writer = GGUFWriter(out_path, "rmvpe")
    writer.add_name("RVC-GGML RMVPE")
    writer.add_description("RMVPE E2E pitch extraction model with mel filterbank")
    writer.add_file_type(FILE_TYPE_ID.get(qtype, 0))
    if qtype != GGMLQuantizationType.F32:
        writer.add_quantization_version(2)

    # mel frontend parameters (matching MelSpectrogram in infer/lib/rmvpe.py)
    kp = "rmvpe."
    writer.add_uint32(kp + "sample_rate", 16000)
    writer.add_uint32(kp + "n_fft", 1024)
    writer.add_uint32(kp + "hop_length", 160)
    writer.add_uint32(kp + "win_length", 1024)
    writer.add_uint32(kp + "n_mels", 128)
    writer.add_uint32(kp + "n_classes", 360)
    writer.add_uint32(kp + "hidden_dim", 256)
    writer.add_uint32(kp + "feature_dim", 384)
    writer.add_float32(kp + "mel_fmin", 30.0)
    writer.add_float32(kp + "mel_fmax", 8000.0)
    writer.add_float32(kp + "mel_clamp", 1e-5)
    writer.add_bool("rvc.conv_flat", True)

    # mel filterbank (precomputed tensor)
    mel_basis = make_rmvpe_mel_basis()
    writer.add_tensor("buffer_mel_basis", mel_basis)
    print(f"  mel filterbank: {mel_basis.shape}")

    stats = new_stats()
    n = 0
    for key, value in state.items():
        if not torch.is_tensor(value):
            continue
        if key.endswith(".num_batches_tracked"):
            continue
        data = tensor_np(value).astype(np.float32)
        if data.ndim == 0:
            data = data.reshape(1)
        policy = rmvpe_policy(key, data)
        if rmvpe_should_flatten(key, data):
            data = flatten_conv2d(data)
        write_tensor_mixed(writer, key, data, qtype, policy, stats)
        n += 1

    size_mb = finish_writer(writer, out_path)
    print(f"  done: {n} tensors")
    print_stats("[rmvpe]", stats, size_mb)


# ============================================================================
# 3. RVC character model (SynthesizerTrnMs v1/v2, f0/non-f0)
# ============================================================================

# Numerically sensitive tensors (always F32)
RVC_FORCE_F32_EXACT = (
    "emb_g.weight",  # speaker table
    "enc_p.emb_pitch.weight",  # pitch lookup table
    "enc_p.emb_phone.weight",  # feature projection (256/768 -> 192)
    "dec.m_source.l_linear.weight",
)
RVC_FORCE_F32_MARKERS = (".bias",)
RVC_FORCE_F32_PREFIX = (
    "flow.flows.",  # flow coupling layers (small and numerically sensitive)
    "dec.ups.",  # Vulkan CONV_TRANSPOSE_1D only supports F32 weights
    "dec.noise_convs.",  # excitation noise convolutions (small; amplified stage by stage through the generator)
)


def rvc_should_flatten(name: str, data: np.ndarray) -> bool:
    # ConvTranspose1d (dec.ups) is not flattened (native Vulkan operators only support F32)
    return (
        data.ndim == 3 and name.endswith(".weight") and not name.startswith("dec.ups.")
    )


def rvc_policy(name: str, data: np.ndarray) -> str:
    if data.ndim == 1:
        return "f32"
    if name in RVC_FORCE_F32_EXACT:
        return "f32"
    for marker in RVC_FORCE_F32_MARKERS:
        if name.endswith(marker):
            return "f32"
    for prefix in RVC_FORCE_F32_PREFIX:
        if name.startswith(prefix):
            return "f32"
    if name.endswith(".weight"):
        return "quant"
    return "f32"


def export_rvc_model(input_path: str, out_path: str, dtype: str) -> None:
    print(f"[model] loading {input_path}")
    ckpt = load_torch(input_path)
    if not isinstance(ckpt, dict) or "weight" not in ckpt:
        raise RuntimeError("RVC model is missing the 'weight' field")

    config = list(ckpt["config"])
    weight = ckpt["weight"]
    n_spk = int(weight["emb_g.weight"].shape[0])
    if len(config) >= 3:
        config[-3] = n_spk  # override the speaker count

    version = ckpt.get("version")
    if version not in ("v1", "v2"):
        # Some checkpoints (e.g. plain training saves) carry no 'version' field;
        # infer it from the phone-embedding input width (v1: 256, v2: 768).
        emb = weight.get("emb_phone.weight", weight.get("enc_p.emb_phone.weight"))
        version = "v2" if (emb is not None and emb.shape[-1] >= 768) else "v1"
    if_f0 = int(ckpt.get("f0", 1))

    sr = config[-1]
    if isinstance(sr, str):
        sr = {"32k": 32000, "40k": 40000, "48k": 48000}[sr]

    (
        spec_channels,
        segment_size,
        inter_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        spk_embed_dim,
        gin_channels,
        sample_rate,
    ) = config[:18]

    print(
        f"  version={version} f0={if_f0} sr={sample_rate} spk={n_spk} "
        f"upsample={upsample_rates} kernels={upsample_kernel_sizes}"
    )

    state = fold_weight_norm(weight)

    qtype = parse_dtype(dtype)
    writer = GGUFWriter(out_path, "rvc")
    writer.add_name(f"RVC-GGML character model ({Path(input_path).stem})")
    writer.add_description(f"RVC {version} {'f0' if if_f0 else 'non-f0'} model")
    writer.add_file_type(FILE_TYPE_ID.get(qtype, 0))
    if qtype != GGMLQuantizationType.F32:
        writer.add_quantization_version(2)

    kp = "rvc."
    writer.add_string(kp + "version", version)
    writer.add_bool(kp + "has_f0", bool(if_f0))
    writer.add_uint32(kp + "sample_rate", int(sample_rate))
    writer.add_uint32(kp + "n_spk", n_spk)
    writer.add_uint32(kp + "spec_channels", int(spec_channels))
    writer.add_uint32(kp + "segment_size", int(segment_size))
    writer.add_uint32(kp + "inter_channels", int(inter_channels))
    writer.add_uint32(kp + "hidden_channels", int(hidden_channels))
    writer.add_uint32(kp + "filter_channels", int(filter_channels))
    writer.add_uint32(kp + "n_heads", int(n_heads))
    writer.add_uint32(kp + "n_layers", int(n_layers))
    writer.add_uint32(kp + "kernel_size", int(kernel_size))
    writer.add_string(kp + "resblock", str(resblock))
    writer.add_array(
        kp + "resblock_kernel_sizes", [int(x) for x in resblock_kernel_sizes]
    )
    dilation_flat = []
    for d in resblock_dilation_sizes:
        dilation_flat.extend(int(x) for x in d)
    writer.add_array(kp + "resblock_dilation_sizes", dilation_flat)
    writer.add_array(kp + "upsample_rates", [int(x) for x in upsample_rates])
    writer.add_uint32(kp + "upsample_initial_channel", int(upsample_initial_channel))
    writer.add_array(
        kp + "upsample_kernel_sizes", [int(x) for x in upsample_kernel_sizes]
    )
    writer.add_uint32(kp + "spk_embed_dim", int(spk_embed_dim))
    writer.add_uint32(kp + "gin_channels", int(gin_channels))
    writer.add_bool("rvc.conv_flat", True)

    # speaker metadata (optional)
    speaker_info = ckpt.get("speaker_info")
    if isinstance(speaker_info, list):
        names = [""] * n_spk
        for item in speaker_info:
            try:
                sid = int(item["id"])
                if 0 <= sid < n_spk:
                    names[sid] = str(item["name"])
            except (KeyError, TypeError, ValueError):
                continue
        if any(names):
            writer.add_array(kp + "speaker_names", names)

    stats = new_stats()
    n = 0
    for key, value in state.items():
        if not torch.is_tensor(value):
            continue
        data = tensor_np(value).astype(np.float32)
        policy = rvc_policy(key, data)
        if rvc_should_flatten(key, data):
            data = flatten_conv1d(data)
        write_tensor_mixed(writer, key, data, qtype, policy, stats)
        n += 1

    size_mb = finish_writer(writer, out_path)
    print(f"  done: {n} tensors")
    print_stats("[model]", stats, size_mb)


# ============================================================================
# 4. FAISS IVF retrieval index
# ============================================================================


def export_index(input_path: str, out_path: str) -> None:
    print(f"[index] loading {input_path}")
    import faiss

    index = faiss.read_index(input_path)
    if index.__class__.__name__ != "IndexIVFFlat":
        raise RuntimeError(
            f"Only IndexIVFFlat is supported (currently {index.__class__.__name__})"
        )

    dim = int(index.d)
    nlist = int(index.nlist)
    ntotal = int(index.ntotal)
    nprobe = int(getattr(index, "nprobe", 1))
    print(
        f"  dim={dim} nlist={nlist} ntotal={ntotal} metric={index.metric_type} nprobe={nprobe}"
    )

    quantizer = faiss.downcast_index(index.quantizer)
    if not quantizer.__class__.__name__.startswith("IndexFlat"):
        raise RuntimeError(
            f"Only Flat quantizers are supported (currently {quantizer.__class__.__name__})"
        )

    centroids = np.ascontiguousarray(
        quantizer.reconstruct_n(0, nlist), dtype=np.float32
    )

    vectors_list = []
    offsets = np.zeros(nlist, dtype=np.float32)
    lengths = np.zeros(nlist, dtype=np.float32)
    cursor = 0
    for list_id in range(nlist):
        size = int(index.invlists.list_size(list_id))
        offsets[list_id] = cursor
        lengths[list_id] = size
        if size:
            codes = faiss.rev_swig_ptr(
                index.invlists.get_codes(list_id), size * index.code_size
            )
            vectors_list.append(
                np.frombuffer(codes, dtype=np.float32).reshape(size, dim).copy()
            )
        cursor += size
    vectors = (
        np.concatenate(vectors_list, axis=0)
        if vectors_list
        else np.zeros((0, dim), dtype=np.float32)
    )

    writer = GGUFWriter(out_path, "rvc_index")
    writer.add_name(f"RVC-GGML retrieval index ({Path(input_path).stem})")
    writer.add_description("FAISS IndexIVFFlat sidecar for RVC retrieval blending")
    writer.add_uint32("index.dim", dim)
    writer.add_uint32("index.nlist", nlist)
    writer.add_uint32("index.ntotal", ntotal)
    writer.add_uint32("index.metric_type", int(index.metric_type))
    writer.add_uint32("index.nprobe", nprobe)
    writer.add_tensor("index.centroids", centroids)
    writer.add_tensor("index.vectors", vectors)
    writer.add_tensor("index.list_offsets", offsets)
    writer.add_tensor("index.list_lengths", lengths)

    size_mb = finish_writer(writer, out_path)
    print(f"  done: index {size_mb:.1f} MB")


# ============================================================================
# Batch conversion
# ============================================================================


def find_asset(root: Path, *candidates: str) -> Path:
    for rel in candidates:
        p = root / rel
        if p.is_file():
            return p
    raise FileNotFoundError(f"not found under {root}: {candidates}")


def export_all(rvc_root: str, out_dir: str, dtype: str) -> None:
    root = Path(rvc_root)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    hubert = find_asset(
        root,
        "assets/hubert/hubert_base.pt",
        "assets/hubert/hubert_base.pt".replace("/", "\\"),
    )
    export_hubert(str(hubert), str(out / "hubert.gguf"), "fp16")

    rmvpe = find_asset(root, "assets/rmvpe/rmvpe.pt")
    export_rmvpe(str(rmvpe), str(out / "rmvpe.gguf"), dtype)

    weights_dir = root / "assets" / "weights"
    if weights_dir.is_dir():
        for pth in sorted(weights_dir.glob("*.pth")):
            export_rvc_model(str(pth), str(out / f"{pth.stem}.gguf"), dtype)

    for idx_dir in [root / "logs", root / "assets" / "indices"]:
        if idx_dir.is_dir():
            for idx in sorted(idx_dir.glob("*.index")):
                export_index(str(idx), str(out / f"{idx.stem}.index.gguf"))


# ============================================================================
# CLI
# ============================================================================


def main() -> int:
    parser = argparse.ArgumentParser(description="RVC-GGML model conversion tool")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("hubert", help="Convert HuBERT/ContentVec")
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--dtype", default="fp16")

    p = sub.add_parser("rmvpe", help="Convert RMVPE")
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--dtype", default="fp16")

    p = sub.add_parser("model", help="Convert an RVC character model")
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--dtype", default="fp16")

    p = sub.add_parser("index", help="Convert a FAISS retrieval index")
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)

    p = sub.add_parser("all", help="Batch-convert from an RVC root directory")
    p.add_argument("--rvc-root", required=True)
    p.add_argument("--out", default="models")
    p.add_argument("--dtype", default="fp16")

    args = parser.parse_args()
    if args.cmd == "hubert":
        export_hubert(args.input, args.out, args.dtype)
    elif args.cmd == "rmvpe":
        export_rmvpe(args.input, args.out, args.dtype)
    elif args.cmd == "model":
        export_rvc_model(args.input, args.out, args.dtype)
    elif args.cmd == "index":
        export_index(args.input, args.out)
    elif args.cmd == "all":
        export_all(args.rvc_root, args.out, args.dtype)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
