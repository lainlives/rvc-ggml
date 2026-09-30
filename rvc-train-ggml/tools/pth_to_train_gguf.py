#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pth_to_train_gguf.py —— 官方 RVC .pth → 训练态 GGUF（一次性转换工具）

定位：本脚本是**工具**，不是训练运行期依赖；训练/续训/微调全 C++（见 rvc-train）。

支持两种输入形态：
  1) 训练态检查点：{"model": state_dict, "optimizer": ..., "iteration": ...}
     （官方预训练 f0G*/f0D*、logs/<exp>/G_*.pth / D_*.pth）
  2) 推理小模型：{"weight": state_dict, "config": [18 项], "sr": "40k", "f0": 1}
     （WebUI 提取的 assets/weights/*.pth；无 enc_q）

输出：训练态 GGUF（ggml v3 格式，小端；与 train.cpp 的 GGUF 读端对齐）
  - 张量名 = torch state_dict 键名（含 weight_g/weight_v，**不折叠**）
  - 形状按 traincpp/ggml 的 ne 序：ne = reversed(torch.shape)（与 ggml 一致）
  - 元数据前缀 rvc.train.*（与推理侧 rvc.* schema 区分）

用法示例：
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

# ---------------------------------------------------------------- GGUF 常量

GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1

GGUF_U8, GGUF_I8, GGUF_U16, GGUF_I16, GGUF_U32, GGUF_I32, GGUF_F32 = range(7)
GGUF_BOOL, GGUF_STRING, GGUF_ARRAY, GGUF_U64, GGUF_I64, GGUF_F64 = range(7, 13)

ALIGNMENT = 32

# 官方判别器周期（与 infer/module/models.py 一致）
D_PERIODS_V1 = [2, 3, 5, 7, 11, 17]
D_PERIODS_V2 = [2, 3, 5, 7, 11, 17, 23, 37]


# ---------------------------------------------------------------- GGUF 写入

class GgufWriter:
    """最小 GGUF v3 写入器（仅标准库 + numpy），供转换工具使用。"""

    def __init__(self):
        self.kv = []        # (key, type_id, value_bytes)
        self.tensors = []   # (name, ne, type_id, data_bytes)

    # ---- 元数据 ----
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

    # ---- 张量 ----
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

        # KV 区（值紧排，无填充）
        for key, type_id, val in self.kv:
            kb = key.encode('utf-8')
            out += struct.pack('<Q', len(kb)) + kb
            out += struct.pack('<i', type_id) + val

        # 张量信息区 + 数据偏移
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

        # 头区对齐到 32，随后是数据段
        while len(out) % ALIGNMENT:
            out += b'\x00'
        for data, aligned in data_blocks:
            out += data
            out += b'\x00' * (aligned - len(data))

        with open(path, 'wb') as f:
            f.write(bytes(out))


# ---------------------------------------------------------------- pth 解析

def load_pth(path):
    """返回 (top_level_dict, state_dict, form)；form ∈ {'train','small','raw'}"""
    d = torch.load(path, map_location='cpu')
    if not isinstance(d, dict):
        raise RuntimeError('顶层不是 dict：%s' % type(d).__name__)
    if 'model' in d and isinstance(d['model'], dict):
        return d, d['model'], 'train'
    if 'weight' in d and isinstance(d['weight'], dict):
        return d, d['weight'], 'small'
    if d and all(torch.is_tensor(v) for v in d.values()):
        return d, d, 'raw'
    raise RuntimeError('无法识别 state_dict（既无 model 也无 weight 键）')


def small_config_to_cfg(config_list):
    """把 process_ckpt.savee 的 18 项 config list 转成 config dict（缺失项用 0 表示未知）"""
    if len(config_list) != 18:
        raise RuntimeError('内嵌 config 长度不是 18：%d' % len(config_list))
    return {
        'data': {
            'spec_channels': int(config_list[0]),
            'filter_length': 0,  # 由 spec_channels 反推
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


# ---------------------------------------------------------------- 转换

def detect_kind(sd):
    if any(k.startswith('discriminators.') for k in sd):
        return 'D'
    if any(k.startswith('enc_p.') for k in sd):
        return 'G'
    raise RuntimeError('无法判断模型类型：state_dict 既无 discriminators.* 也无 enc_p.*')


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
        # proj: [384, hidden]（torch）→ hidden 在最后一维
        ch = int(w2.shape[-1])
        if ch == 192:
            return 'v1' if sd.get('enc_p.emb_phone.weight') is not None and sd['enc_p.emb_phone.weight'].shape[-1] == 256 else 'v2'
    raise RuntimeError('无法判断 v1/v2：缺少 enc_p.emb_phone.weight')


def detect_version_d(sd):
    idx = sorted({int(k.split('.')[1]) for k in sd if k.startswith('discriminators.') and k.split('.')[1].isdigit()})
    n = len(idx)
    if n == 7:
        return 'v1'
    if n == 9:
        return 'v2'
    raise RuntimeError('无法判断 v1/v2：判别器数量 %d（期望 7 或 9）' % n)


def to_ne(shape):
    if len(shape) == 0:
        return [1]
    return [int(x) for x in reversed(shape)]


def tensor_bytes(t, dtype):
    """torch 张量 → (ne, type_id, 小端 F32/F16 连续字节)"""
    if t.dtype not in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
        raise RuntimeError('不支持的非浮点张量 dtype：%s' % t.dtype)
    a = t.detach().cpu().contiguous()
    if dtype == 'f16':
        a = a.half()
        return to_ne(t.shape), GGML_TYPE_F16, a.numpy().tobytes(order='C')
    a = a.float()
    return to_ne(t.shape), GGML_TYPE_F32, a.numpy().tobytes(order='C')


def main():
    ap = argparse.ArgumentParser(description='RVC .pth -> 训练态 GGUF（rvc.train.* 元数据）')
    ap.add_argument('--pth', required=True, help='输入 .pth')
    ap.add_argument('--out', required=True, help='输出 .gguf')
    ap.add_argument('--config', default='', help='官方训练 config JSON（推荐；缺失时尝试读 pth 内嵌 config）')
    ap.add_argument('--kind', default='auto', choices=['auto', 'g', 'd'], help='模型类型（默认自动）')
    ap.add_argument('--dtype', default='f32', choices=['f32', 'f16'], help='输出张量类型（默认 f32）')
    ap.add_argument('--sr', type=int, default=0, help='采样率（覆盖 config）')
    ap.add_argument('--f0', type=int, default=-1, choices=[-1, 0, 1], help='是否 f0 模型（覆盖推断）')
    ap.add_argument('--version', default='', choices=['', 'v1', 'v2'], help='模型版本（覆盖推断）')
    args = ap.parse_args()

    top, sd, form = load_pth(args.pth)
    kind = detect_kind(sd)
    if args.kind != 'auto':
        want = 'G' if args.kind == 'g' else 'D'
        if kind != want:
            raise RuntimeError('--kind=%s 与检测到的 %s 不一致' % (args.kind, kind))

    # ---- 模型配置 ----
    cfg = None
    if args.config:
        cfg = load_config_json(args.config)
    elif form == 'small' and 'config' in top:
        cfg = small_config_to_cfg(top['config'])
        print('[warn] 使用 pth 内嵌 config（data 部分字段缺失，建议显式提供 --config）')
    if cfg is None and kind == 'G':
        raise RuntimeError('G 模型需要 --config（官方训练 JSON）或内嵌 config 的推理小模型')

    version = args.version or (detect_version_g(sd) if kind == 'G' else detect_version_d(sd))
    has_f0 = False
    if kind == 'G':
        has_f0 = ('enc_p.emb_pitch.weight' in sd) or any(k.startswith('dec.m_source') for k in sd)
    else:
        has_f0 = bool(args.f0 == 1)
    if args.f0 != -1:
        has_f0 = bool(args.f0 == 1)

    # ---- 汇总元数据 ----
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
            raise RuntimeError('缺少采样率：请提供 --sr 或 config.data.sampling_rate')

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
            print('[warn] config.spk_embed_dim=%d 与 emb_g.weight.shape[0]=%d 不一致，写实际值' % (spk_cfg, spk_actual))
        w.set_u32('rvc.train.spk_embed_dim', spk_actual or spk_cfg)
    else:
        emb_g = sd.get('emb_g.weight')
        if emb_g is not None:
            w.set_u32('rvc.train.spk_embed_dim', int(emb_g.shape[0]))

    if kind == 'D':
        w.set_arr_i32('rvc.train.d_periods', D_PERIODS_V1 if version == 'v1' else D_PERIODS_V2)

    # ---- 张量 ----
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
