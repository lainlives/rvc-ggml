#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pth_to_train_gguf.py -- official RVC .pth -> training-state GGUF (one-shot conversion tool)

Purpose: this script is a **tool**, not a training runtime dependency; training/resume/finetuning are all C++ (see rvc-train).

Supports two input forms:
  1) Training-state checkpoint: {"model": state_dict, "optimizer": ..., "iteration": ...}
     (official pretrained f0G*/f0D*, logs/<exp>/G_*.pth / D_*.pth)
  2) Small inference model: {"weight": state_dict, "config": [18 items], "sr": "40k", "f0": 1}
     (assets/weights/*.pth extracted by the WebUI; no enc_q)

Output: training-state GGUF (ggml v3 format, little-endian; aligned with train.cpp's GGUF reader)
  - Tensor names = torch state_dict keys (including weight_g/weight_v, **not folded**)
  - Shapes follow traincpp/ggml ne order: ne = reversed(torch.shape) (consistent with ggml)
  - Metadata prefix rvc.train.* (distinct from the inference-side rvc.* schema)

Usage examples:
  python tools\\pth_to_train_gguf.py \\
      --pth D:\\...\\pretrained\\f0G40k.pth --config configs\\v1-40k-f0.json \\
      --out work\\f0G40k.train.gguf
  python tools\\pth_to_train_gguf.py \\
      --pth D:\\...\\weights\\guanguanV1.pth --out work\\guanguanV1.train.gguf
"""

import argparse
import json
import os
import struct
import sys
from collections import Counter

import torch

# ---------------------------------------------------------------- GGUF constants

GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1

GGUF_U8, GGUF_I8, GGUF_U16, GGUF_I16, GGUF_U32, GGUF_I32, GGUF_F32 = range(7)
GGUF_BOOL, GGUF_STRING, GGUF_ARRAY, GGUF_U64, GGUF_I64, GGUF_F64 = range(7, 13)

ALIGNMENT = 32

# Official discriminator periods (consistent with infer/module/models.py)
D_PERIODS_V1 = [2, 3, 5, 7, 11, 17]
D_PERIODS_V2 = [2, 3, 5, 7, 11, 17, 23, 37]


# ---------------------------------------------------------------- GGUF writing

class GgufWriter:
    """Minimal GGUF v3 writer (standard library + numpy only) for the conversion tool."""

    def __init__(self):
        self.kv = []        # (key, type_id, value_bytes)
        self.tensors = []   # (name, ne, type_id, data_bytes)

    # ---- Metadata ----
    def _set(self, key, type_id, value):
        self.kv.append((key, type_id, value))

    def set_u32(self, key, v):
        self._set(key, GGUF_U32, struct.pack('<I', int(v)))

    def set_i32(self, key, v):
        self._set(key, GGUF_I32, struct.pack('<i', int(v)))

    def set_f32(self, key, v):
        self._set(key, GGUF_F32, struct.pack('<f', float(v)))

    def set_bool(self, key, v):
        self._set(key, GGUF_BOOL, struct.pack('<B', 1 if v else 0))

    def set_str(self, key, v):
        b = str(v).encode('utf-8')
        self._set(key, GGUF_STRING, struct.pack('<Q', len(b)) + b)

    def set_arr_i32(self, key, values):
        b = struct.pack('<i', GGUF_I32) + struct.pack('<Q', len(values))
        for v in values:
            b += struct.pack('<i', int(v))
        self._set(key, GGUF_ARRAY, b)

    # ---- Tensors ----
    def add_tensor(self, name, ne, type_id, data):
        assert len(ne) >= 1
        expect = 1
        for d in ne:
            expect *= int(d)
        if type_id == GGML_TYPE_F32:
            assert len(data) == expect * 4, (name, len(data), expect)
        elif type_id == GGML_TYPE_F16:
            assert len(data) == expect * 2, (name, len(data), expect)
        else:
            raise ValueError('unsupported tensor type %d' % type_id)
        self.tensors.append((name, [int(d) for d in ne], type_id, data))

    def write(self, path):
        out = bytearray()
        out += b'GGUF'
        out += struct.pack('<I', 3)                 # version
        out += struct.pack('<Q', len(self.tensors))  # tensor_count
        out += struct.pack('<Q', len(self.kv))       # metadata_kv_count

        # KV section (values packed tightly, no padding)
        for key, type_id, val in self.kv:
            kb = key.encode('utf-8')
            out += struct.pack('<Q', len(kb)) + kb
            out += struct.pack('<i', type_id) + val

        # Tensor info section + data offsets
        offset = 0
        data_blocks = []
        for (name, ne, type_id, data) in self.tensors:
            nb = name.encode('utf-8')
            out += struct.pack('<Q', len(nb)) + nb
            out += struct.pack('<I', len(ne))
            for d in ne:
                out += struct.pack('<q', d)
            out += struct.pack('<i', type_id)
            out += struct.pack('<Q', offset)
            aligned = (len(data) + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT
            data_blocks.append((data, aligned))
            offset += aligned

        # Align the header to 32, then the data section follows
        while len(out) % ALIGNMENT:
            out += b'\x00'
        for data, aligned in data_blocks:
            out += data
            out += b'\x00' * (aligned - len(data))

        with open(path, 'wb') as f:
            f.write(bytes(out))


# ---------------------------------------------------------------- pth parsing

def load_pth(path):
    """Return (top_level_dict, state_dict, form); form in {'train','small','raw'}"""
    d = torch.load(path, map_location='cpu')
    if not isinstance(d, dict):
        raise RuntimeError('top level is not a dict: %s' % type(d).__name__)
    if 'model' in d and isinstance(d['model'], dict):
        return d, d['model'], 'train'
    if 'weight' in d and isinstance(d['weight'], dict):
        return d, d['weight'], 'small'
    if d and all(torch.is_tensor(v) for v in d.values()):
        return d, d, 'raw'
    raise RuntimeError('unrecognized state_dict (neither a model nor a weight key)')


def small_config_to_cfg(config_list):
    """Convert process_ckpt.savee's 18-item config list into a config dict (missing entries use 0 for unknown)"""
    if len(config_list) != 18:
        raise RuntimeError('embedded config length is not 18: %d' % len(config_list))
    return {
        'data': {
            'spec_channels': int(config_list[0]),
            'filter_length': 0,  # derived from spec_channels
            'sampling_rate': int(config_list[17]),
            'hop_length': 0,
            'win_length': 0,
            'n_mel_channels': 0,
            'mel_fmin': 0.0,
            'mel_fmax': 0.0,
        },
        'model': {
            'inter_channels': int(config_list[2]),
            'hidden_channels': int(config_list[3]),
            'filter_channels': int(config_list[4]),
            'n_heads': int(config_list[5]),
            'n_layers': int(config_list[6]),
            'kernel_size': int(config_list[7]),
            'p_dropout': float(config_list[8]),
            'resblock': str(config_list[9]),
            'resblock_kernel_sizes': list(config_list[10]),
            'resblock_dilation_sizes': [list(x) for x in config_list[11]],
            'upsample_rates': list(config_list[12]),
            'upsample_initial_channel': int(config_list[13]),
            'upsample_kernel_sizes': list(config_list[14]),
            'spk_embed_dim': int(config_list[15]),
            'gin_channels': int(config_list[16]),
        },
        'train': {'segment_size': 0},
    }


def load_config_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


# ---------------------------------------------------------------- Conversion

def detect_kind(sd):
    if any(k.startswith('discriminators.') for k in sd):
        return 'D'
    if any(k.startswith('enc_p.') for k in sd):
        return 'G'
    raise RuntimeError('cannot determine model type: state_dict has neither discriminators.* nor enc_p.*')


def detect_version_g(sd):
    w = sd.get('enc_p.emb_phone.weight')
    if w is not None:
        ch = int(w.shape[-1])
        if ch == 256:
            return 'v1'
        if ch == 768:
            return 'v2'
    w2 = sd.get('enc_p.proj.weight')
    if w2 is not None:
        # proj: [384, hidden] (torch) -> hidden is in the last dimension
        ch = int(w2.shape[-1])
        if ch == 192:
            return 'v1' if sd.get('enc_p.emb_phone.weight') is not None and sd['enc_p.emb_phone.weight'].shape[-1] == 256 else 'v2'
    raise RuntimeError('cannot determine v1/v2: missing enc_p.emb_phone.weight')


def detect_version_d(sd):
    idx = sorted({int(k.split('.')[1]) for k in sd if k.startswith('discriminators.') and k.split('.')[1].isdigit()})
    n = len(idx)
    if n == 7:
        return 'v1'
    if n == 9:
        return 'v2'
    raise RuntimeError('cannot determine v1/v2: discriminator count %d (expected 7 or 9)' % n)


def to_ne(shape):
    if len(shape) == 0:
        return [1]
    return [int(x) for x in reversed(shape)]


def tensor_bytes(t, dtype):
    """torch tensor -> (ne, type_id, little-endian contiguous F32/F16 bytes)"""
    if t.dtype not in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
        raise RuntimeError('unsupported non-floating-point tensor dtype: %s' % t.dtype)
    a = t.detach().cpu().contiguous()
    if dtype == 'f16':
        a = a.half()
        return to_ne(t.shape), GGML_TYPE_F16, a.numpy().tobytes(order='C')
    a = a.float()
    return to_ne(t.shape), GGML_TYPE_F32, a.numpy().tobytes(order='C')


def main():
    ap = argparse.ArgumentParser(description='RVC .pth -> training-state GGUF (rvc.train.* metadata)')
    ap.add_argument('--pth', required=True, help='input .pth')
    ap.add_argument('--out', required=True, help='output .gguf')
    ap.add_argument('--config', default='', help='official training config JSON (recommended; if missing, try the pth embedded config)')
    ap.add_argument('--kind', default='auto', choices=['auto', 'g', 'd'], help='model type (default: auto)')
    ap.add_argument('--dtype', default='f32', choices=['f32', 'f16'], help='output tensor type (default: f32)')
    ap.add_argument('--sr', type=int, default=0, help='sample rate (overrides config)')
    ap.add_argument('--f0', type=int, default=-1, choices=[-1, 0, 1], help='whether this is an f0 model (overrides inference)')
    ap.add_argument('--version', default='', choices=['', 'v1', 'v2'], help='model version (overrides inference)')
    args = ap.parse_args()

    top, sd, form = load_pth(args.pth)
    kind = detect_kind(sd)
    if args.kind != 'auto':
        want = 'G' if args.kind == 'g' else 'D'
        if kind != want:
            raise RuntimeError('--kind=%s does not match the detected %s' % (args.kind, kind))

    # ---- Model configuration ----
    cfg = None
    if args.config:
        cfg = load_config_json(args.config)
    elif form == 'small' and 'config' in top:
        cfg = small_config_to_cfg(top['config'])
        print('[warn] using the pth embedded config (some data fields are missing; passing --config explicitly is recommended)')
    if cfg is None and kind == 'G':
        raise RuntimeError('G models require --config (official training JSON) or a small inference model with an embedded config')

    version = args.version or (detect_version_g(sd) if kind == 'G' else detect_version_d(sd))
    has_f0 = False
    if kind == 'G':
        has_f0 = ('enc_p.emb_pitch.weight' in sd) or any(k.startswith('dec.m_source') for k in sd)
    else:
        has_f0 = bool(args.f0 == 1)
    if args.f0 != -1:
        has_f0 = bool(args.f0 == 1)

    # ---- Assemble metadata ----
    w = GgufWriter()
    w.set_str('general.architecture', 'rvc-train')
    w.set_str('general.name', os.path.splitext(os.path.basename(args.out))[0])
    w.set_str('rvc.train.kind', kind)
    w.set_str('rvc.train.version', version)
    w.set_bool('rvc.train.has_f0', has_f0)
    w.set_str('rvc.train.form', form)
    w.set_str('rvc.train.source_pth', os.path.abspath(args.pth))
    w.set_str('rvc.train.out_dtype', args.dtype.upper())
    w.set_u32('rvc.train.tensor_count', len(sd))

    if cfg is not None:
        data = cfg.get('data', {})
        model = cfg.get('model', {})
        train = cfg.get('train', {})

        sr = args.sr or int(data.get('sampling_rate', 0))
        if sr == 0 and kind == 'G':
            raise RuntimeError('missing sample rate: pass --sr or set config.data.sampling_rate')

        spec_channels = int(data.get('spec_channels', 0))
        filter_length = int(data.get('filter_length', 0))
        if spec_channels == 0 and filter_length:
            spec_channels = filter_length // 2 + 1
        if filter_length == 0 and spec_channels:
            filter_length = (spec_channels - 1) * 2

        w.set_u32('rvc.train.sample_rate', sr)
        w.set_u32('rvc.train.spec_channels', spec_channels)
        w.set_u32('rvc.train.filter_length', filter_length)
        w.set_u32('rvc.train.hop_length', int(data.get('hop_length', 0)))
        w.set_u32('rvc.train.win_length', int(data.get('win_length', 0)))
        w.set_u32('rvc.train.n_mel_channels', int(data.get('n_mel_channels', 0)))
        w.set_f32('rvc.train.mel_fmin', float(data.get('mel_fmin', 0.0)))
        mf = data.get('mel_fmax', None)
        w.set_f32('rvc.train.mel_fmax', float(sr) / 2.0 if mf is None else float(mf))
        w.set_u32('rvc.train.segment_size', int(train.get('segment_size', 0)))

        w.set_u32('rvc.train.inter_channels', int(model.get('inter_channels', 0)))
        w.set_u32('rvc.train.hidden_channels', int(model.get('hidden_channels', 0)))
        w.set_u32('rvc.train.filter_channels', int(model.get('filter_channels', 0)))
        w.set_u32('rvc.train.n_heads', int(model.get('n_heads', 0)))
        w.set_u32('rvc.train.n_layers', int(model.get('n_layers', 0)))
        w.set_u32('rvc.train.kernel_size', int(model.get('kernel_size', 0)))
        w.set_f32('rvc.train.p_dropout', float(model.get('p_dropout', 0.0)))
        w.set_u32('rvc.train.gin_channels', int(model.get('gin_channels', 0)))
        w.set_u32('rvc.train.upsample_initial_channel', int(model.get('upsample_initial_channel', 0)))
        w.set_str('rvc.train.resblock', str(model.get('resblock', '1')))
        w.set_arr_i32('rvc.train.resblock_kernel_sizes', list(model.get('resblock_kernel_sizes', [])))
        flat_rds = []
        for pair in model.get('resblock_dilation_sizes', []):
            flat_rds.extend(int(x) for x in pair)
        w.set_arr_i32('rvc.train.resblock_dilation_sizes', flat_rds)
        w.set_arr_i32('rvc.train.upsample_rates', list(model.get('upsample_rates', [])))
        w.set_arr_i32('rvc.train.upsample_kernel_sizes', list(model.get('upsample_kernel_sizes', [])))

        spk_cfg = int(model.get('spk_embed_dim', 0))
        emb_g = sd.get('emb_g.weight')
        spk_actual = int(emb_g.shape[0]) if emb_g is not None else 0
        if spk_cfg and spk_actual and spk_cfg != spk_actual:
            print('[warn] config.spk_embed_dim=%d does not match emb_g.weight.shape[0]=%d; writing the actual value' % (spk_cfg, spk_actual))
        w.set_u32('rvc.train.spk_embed_dim', spk_actual or spk_cfg)
    else:
        emb_g = sd.get('emb_g.weight')
        if emb_g is not None:
            w.set_u32('rvc.train.spk_embed_dim', int(emb_g.shape[0]))

    if kind == 'D':
        w.set_arr_i32('rvc.train.d_periods', D_PERIODS_V1 if version == 'v1' else D_PERIODS_V2)

    # ---- Tensors ----
    dtype_count = Counter()
    for name, t in sd.items():
        ne, type_id, data = tensor_bytes(t, args.dtype)
        w.add_tensor(name, ne, type_id, data)
        dtype_count[str(t.dtype)] += 1

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    w.write(args.out)

    size = os.path.getsize(args.out)
    print('[ok] %s -> %s' % (os.path.abspath(args.pth), os.path.abspath(args.out)))
    print('     kind=%s version=%s has_f0=%s form=%s tensors=%d out=%s size=%.1f MiB' % (
        kind, version, has_f0, form, len(sd), args.dtype, size / (1024.0 * 1024.0)))
    print('     source dtypes: %s' % dict(dtype_count))


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print('[error] %s' % e, file=sys.stderr)
        sys.exit(1)
